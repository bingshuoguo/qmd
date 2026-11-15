"""Offline reward distribution for the Phase 2 draft reward_func.

Uses the SFT Q4 sampled expansions (T=0.7, seeds 42/43/44) and their path-A
bench results as GRPO groups of G=3 per qid. No retrieval is re-run.

  python3 -I offline_reward.py <exec-dir>/benchmarks
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

B = Path(sys.argv[1])
SETS = [("qmd-eval-v1-fiqa-test", "fiqa"), ("qmd-expansion-scifact-v1", "scifact"), ("qmd-eval-v1-cqadup-android", "android")]
SEEDS = (42, 43, 44)
FORMAT_PENALTY, LEN_START, LEN_END, LEN_MAX = -1.0, 320, 640, 0.2
EPS = 1e-9


def M(m, w_ndcg=0.7):
    return w_ndcg * m["ndcg_at_10"] + (1 - w_ndcg) * m["recall_at_30"]


def p_len(n):
    return LEN_MAX * min(max((n - LEN_START) / (LEN_END - LEN_START), 0.0), 1.0)


def load(path):
    return {r["qid"]: r for r in map(json.loads, open(path))}


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def advantages(rewards, mode, batch_std=None):
    mu = st.mean(rewards)
    if mode == "none":
        return [r - mu for r in rewards]
    sd = batch_std if mode == "batch" else st.pstdev(rewards)
    return [(r - mu) / (sd + 1e-4) for r in rewards]  # TRL uses std + 1e-4


groups = {}  # tag -> list of groups; group = list of item dicts
for d, tag in SETS:
    res_dir, exp_dir = B / d / "runs/results", B / d / "expansions"
    raw = load(res_dir / "raw.jsonl")
    per_seed = {s: (load(next(res_dir.glob(f"sft-seed{s}-{tag}-*.jsonl"))),
                    load(exp_dir / f"sft-seed{s}-{tag}.jsonl")) for s in SEEDS}
    out = []
    for qid, raw_row in raw.items():
        if raw_row.get("retrieval_status") != "ok" or not raw_row.get("metrics"):
            continue
        group = []
        for s in SEEDS:
            res, exp = per_seed[s][0][qid], per_seed[s][1][qid]
            gen = exp.get("generation") or {}
            item = {"tokens": gen.get("generated_tokens", 0),
                    "truncated": gen.get("stop_reason") != "eogToken",
                    "ok": res.get("expansion_status") == "ok" and not (gen.get("stop_reason") != "eogToken"),
                    "m": res["metrics"], "raw": raw_row["metrics"]}
            group.append(item)
        out.append(group)
    groups[tag] = out


def reward(item, w_ndcg=0.7, with_len=True):
    if not item["ok"]:
        return FORMAT_PENALTY
    return M(item["m"], w_ndcg) - M(item["raw"], w_ndcg) - (p_len(item["tokens"]) if with_len else 0.0)


print("== 1. Per-item reward components (all seeds pooled)")
for tag, gs in groups.items():
    items = [i for g in gs for i in g]
    ok = [i for i in items if i["ok"]]
    dm = [M(i["m"]) - M(i["raw"]) for i in ok]
    pl = [p_len(i["tokens"]) for i in ok]
    toks = [i["tokens"] for i in items]
    print(f"  {tag:8} n={len(items)} format_error={sum(not i['ok'] for i in items)} "
          f"| dM mean={st.mean(dm):+.4f} sd={st.pstdev(dm):.4f} p5/p50/p95={pct(dm,.05):+.3f}/{pct(dm,.5):+.3f}/{pct(dm,.95):+.3f} "
          f"min={min(dm):+.3f} | dM>0 {sum(x>EPS for x in dm)/len(dm):.0%} <0 {sum(x<-EPS for x in dm)/len(dm):.0%} "
          f"| P_len>0 {sum(x>0 for x in pl)/len(pl):.1%} mean={st.mean(pl):.4f} max={max(pl):.3f} "
          f"| tok p50/p95/max={pct(toks,.5)}/{pct(toks,.95)}/{max(toks)} >640={sum(t>640 for t in toks)}")

print("\n== 2. Group signal (G=3): fraction of groups with zero reward variance")
for w, name in ((1.0, "nDCG only"), (0.7, "0.7 nDCG + 0.3 R@30")):
    row = []
    for tag, gs in groups.items():
        zero = sum(st.pstdev([reward(i, w) for i in g]) < EPS for g in gs) / len(gs)
        row.append(f"{tag}={zero:.0%}")
    print(f"  {name:22} " + "  ".join(row))

# Estimate zero-variance rate for G=8 by assuming per-qid outcomes are drawn
# from the 3 observed samples (an upper bound: true diversity is higher).
print("\n== 3. Zero-variance rate vs G (resampling the 3 observed outcomes per qid; optimistic-for-zero bound)")
rng = random.Random(42)
for tag, gs in groups.items():
    row = []
    for G in (2, 3, 4, 8):
        z = 0
        for g in gs:
            rs = [reward(i) for i in g]
            if st.pstdev(rs) < EPS:
                z += 1
                continue
            # P(all G draws equal) when each draw is uniform over the 3 observed values
            vals = {}
            for r in rs:
                vals[round(r, 12)] = vals.get(round(r, 12), 0) + 1
            z += sum((c / 3) ** G for c in vals.values())
        row.append(f"G={G}:{z/len(gs):.0%}")
    print(f"  {tag:8} " + "  ".join(row))

print("\n== 4. Advantage scaling: group vs batch (G=3, all datasets pooled into one batch)")
all_groups = [g for gs in groups.values() for g in gs]
all_rewards = [reward(i) for g in all_groups for i in g]
batch_sd = st.pstdev(all_rewards)
print(f"  batch reward sd = {batch_sd:.4f}")
for mode in ("group", "batch"):
    pairs = []  # (|reward spread in group|, max |advantage|)
    for g in all_groups:
        rs = [reward(i) for i in g]
        spread = max(rs) - min(rs)
        if spread < EPS:
            continue
        adv = advantages(rs, mode, batch_sd)
        pairs.append((spread, max(abs(a) for a in adv)))
    tiny = [a for s, a in pairs if s < 0.02]
    large = [a for s, a in pairs if s >= 0.2]
    print(f"  {mode:6} nonzero groups={len(pairs)} | spread<0.02: n={len(tiny)} median max|A|={st.median(tiny):.2f} "
          f"| spread>=0.2: n={len(large)} median max|A|={st.median(large):.2f}")

print("\n== 5. Format-error groups: what the penalty does")
for g in all_groups:
    if any(not i["ok"] for i in g):
        rs = [reward(i) for i in g]
        adv = advantages(rs, "batch", batch_sd)
        print("  rewards=" + ", ".join(f"{r:+.3f}" for r in rs) + "  A(batch)=" + ", ".join(f"{a:+.2f}" for a in adv)
              + "  tokens=" + ",".join(str(i["tokens"]) for i in g))
ok_dm = [M(i["m"]) - M(i["raw"]) for g in all_groups for i in g if i["ok"]]
print(f"  worst legal dM = {min(ok_dm):+.3f}  -> margin to format penalty = {min(ok_dm) - FORMAT_PENALTY:.3f}")
print(f"  legal dM < -0.5: {sum(x < -0.5 for x in ok_dm)} / {len(ok_dm)}")

print("\n== 6. Ceiling queries (raw nDCG@10 == 1): reward sign")
for tag, gs in groups.items():
    ceil = [g for g in gs if g[0]["raw"]["ndcg_at_10"] >= 1 - EPS]
    if not ceil:
        continue
    rs = [reward(i) for g in ceil for i in g]
    zero = sum(st.pstdev([reward(i) for i in g]) < EPS for g in ceil) / len(ceil)
    print(f"  {tag:8} ceiling qids={len(ceil)}/{len(gs)} ({len(ceil)/len(gs):.0%}) "
          f"| reward<0 {sum(r < -EPS for r in rs)/len(rs):.0%} =0 {sum(abs(r) <= EPS for r in rs)/len(rs):.0%} "
          f"| zero-var groups {zero:.0%}")

print("\n== 7. Cross-check: does the length penalty ever change the in-group ordering?")
flips = 0
for g in all_groups:
    a = [reward(i) for i in g]
    b = [reward(i, with_len=False) for i in g]
    if sorted(range(3), key=lambda k: a[k]) != sorted(range(3), key=lambda k: b[k]):
        flips += 1
print(f"  groups whose ranking changes when P_len is added: {flips} / {len(all_groups)}")
