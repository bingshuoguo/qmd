"""Score bf16 G-sample groups through reward-server.ts and report reward distribution.

  python3 -I score_groups.py <samples.jsonl> [--url http://127.0.0.1:8787] [--q4-bench <exec>/benchmarks]

Writes <samples>.scored.jsonl (resumable). Reward = draft Phase 2 formula.
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_client import RewardClient  # noqa: E402

FORMAT_PENALTY, LEN_START, LEN_END, LEN_MAX = -1.0, 320, 640, 0.2
EPS = 1e-9

ap = argparse.ArgumentParser()
ap.add_argument("samples")
ap.add_argument("--url", default="http://127.0.0.1:8787")
ap.add_argument("--q4-bench", default=None)
args = ap.parse_args()

src = Path(args.samples)
scored_path = src.with_suffix(".scored.jsonl")
done = {}
if scored_path.exists():
    for r in map(json.loads, open(scored_path)):
        done[(r["source_id"], r["qid"])] = r
client = RewardClient(args.url)
with open(scored_path, "a") as f:
    for row in map(json.loads, open(src)):
        key = (row["source_id"], row["qid"])
        if key in done:
            continue
        res = client.score([{"source_id": row["source_id"], "qid": row["qid"], "completion": s["completion"],
                             "truncated": s["truncated"]} for s in row["samples"]])
        for s, r in zip(row["samples"], res):
            s.update(status=r["status"], error=r["error"], metrics=r["metrics"], raw=r["raw_metrics"], ms=r["ms"])
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        done[key] = row

groups = list(done.values())


def M(m, w=0.7):
    return w * m["ndcg_at_10"] + (1 - w) * m["recall_at_30"]


def reward(s, w=0.7):
    if s["status"] != "ok":
        return FORMAT_PENALTY
    over = min(max((s["tokens"] - LEN_START) / (LEN_END - LEN_START), 0.0), 1.0)
    return M(s["metrics"], w) - M(s["raw"], w) - LEN_MAX * over


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


DS = {"fiqa-test": "fiqa", "scifact-test": "scifact", "android": "android"}
by_ds = {}
for g in groups:
    by_ds.setdefault(DS.get(g["source_id"], g["source_id"]), []).append(g)

G = len(groups[0]["samples"])
print(f"groups={len(groups)} G={G}")

print("\n== 1. Per-sample components")
for t, gs in by_ds.items():
    ss = [s for g in gs for s in g["samples"]]
    ok = [s for s in ss if s["status"] == "ok"]
    dm = [M(s["metrics"]) - M(s["raw"]) for s in ok]
    toks = [s["tokens"] for s in ss]
    print(f"  {t:8} n={len(ss)} format_error={len(ss) - len(ok)} trunc={sum(s['truncated'] for s in ss)} "
          f"| dM mean={st.mean(dm):+.4f} sd={st.pstdev(dm):.4f} p5/p95={pct(dm,.05):+.3f}/{pct(dm,.95):+.3f} "
          f"| >0 {sum(x > EPS for x in dm)/len(dm):.0%} <0 {sum(x < -EPS for x in dm)/len(dm):.0%} "
          f"| tok p50/p95/max={pct(toks,.5)}/{pct(toks,.95)}/{max(toks)}")

print("\n== 2. Zero-variance groups (exact), by subsample size g")
rng = random.Random(0)
for t, gs in by_ds.items():
    row = []
    for g_size in (3, 4, 8):
        if g_size > G:
            continue
        z, trials = 0, 0
        for g in gs:
            rs = [reward(s) for s in g["samples"]]
            for _ in range(20 if g_size < G else 1):
                sub = rng.sample(rs, g_size) if g_size < G else rs
                z += st.pstdev(sub) < EPS
                trials += 1
        row.append(f"G={g_size}:{z / trials:.0%}")
    print(f"  {t:8} " + "  ".join(row))

print("\n== 3. Within-group spread (G=%d)" % G)
for t, gs in by_ds.items():
    spreads = [max(r) - min(r) for r in ([reward(s) for s in g["samples"]] for g in gs)]
    sds = [st.pstdev([reward(s) for s in g["samples"]]) for g in gs]
    best_gain = [max(reward(s) for s in g["samples"]) for g in gs]
    print(f"  {t:8} spread p50/p90={pct(spreads,.5):.3f}/{pct(spreads,.9):.3f}  in-group sd mean={st.mean(sds):.4f} "
          f"| best-of-G reward mean={st.mean(best_gain):+.4f} (mean of means={st.mean(st.mean(reward(s) for s in g['samples']) for g in gs):+.4f})")

print("\n== 4. Ceiling (raw nDCG@10 == 1)")
for t, gs in by_ds.items():
    ceil = [g for g in gs if g["samples"][0]["raw"]["ndcg_at_10"] >= 1 - EPS]
    if ceil:
        z = sum(st.pstdev([reward(s) for s in g["samples"]]) < EPS for g in ceil)
        print(f"  {t:8} ceiling {len(ceil)}/{len(gs)}  zero-var {z}/{len(ceil)}")

print("\n== 5. Signal variance decomposition: between-qid vs within-qid (all datasets)")
all_r = [[reward(s) for s in g["samples"]] for g in groups]
within = st.mean(st.pvariance(r) for r in all_r)
between = st.pvariance([st.mean(r) for r in all_r])
print(f"  within-qid var={within:.5f}  between-qid var={between:.5f}  within share={within / (within + between):.0%}")

if args.q4_bench:
    print("\n== 6. Same qids, Q4 T=0.7 (top-k20/top-p0.8/presence0.5, grammar), 3 seeds")
    B = Path(args.q4_bench)
    dirs = {"fiqa": ("qmd-eval-v1-fiqa-test", "fiqa"), "scifact": ("qmd-expansion-scifact-v1", "scifact"),
            "android": ("qmd-eval-v1-cqadup-android", "android")}
    for t, gs in by_ds.items():
        d, tag = dirs[t]
        res = [{r["qid"]: r for r in map(json.loads, open(next((B / d / "runs/results").glob(f"sft-seed{s}-{tag}-*.jsonl"))))}
               for s in (42, 43, 44)]
        raw = {r["qid"]: r for r in map(json.loads, open(B / d / "runs/results/raw.jsonl"))}
        q4_sd, q4_zero, bf_zero3 = [], 0, 0
        for g in gs:
            q = g["qid"]
            rs = [M(x[q]["metrics"]) - M(raw[q]["metrics"]) for x in res]
            q4_sd.append(st.pstdev(rs))
            q4_zero += st.pstdev(rs) < EPS
            bf_zero3 += st.pstdev([reward(s) for s in g["samples"][:3]]) < EPS
        bf_sd = [st.pstdev([reward(s) for s in g["samples"][:3]]) for g in gs]
        print(f"  {t:8} G=3 zero-var: Q4 {q4_zero / len(gs):.0%}  vs bf16 {bf_zero3 / len(gs):.0%}  "
              f"| in-group sd mean: Q4 {st.mean(q4_sd):.4f} vs bf16 {st.mean(bf_sd):.4f}")
