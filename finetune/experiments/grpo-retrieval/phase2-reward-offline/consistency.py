"""Reward-consistency check: does the qrels reward order rollouts the way a blind
LLM judge would?

For each G=8 group with reward spread >= --min-spread, take the highest- and
lowest-reward ok rollouts, fetch their top-10 rankings from reward-server
(with_ranking), and build a blinded pairwise item: which list better serves the
query? Side (A/B) is randomized; the key file maps back.

  # build items (reward-server must be running with the matching sources)
  python3 -I consistency.py build <scored.jsonl> [<scored.jsonl> ...] --prepared-map <json> --out-dir <dir>
  # after judging writes <dir>/pair-grades.jsonl
  python3 -I consistency.py analyze --out-dir <dir>
"""
import argparse
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["build", "analyze"])
ap.add_argument("scored", nargs="*")
ap.add_argument("--bench-map", help='JSON {"source_id": "<bench dir>"}; docs are read from its collection_root')
ap.add_argument("--out-dir", required=True)
ap.add_argument("--min-spread", type=float, default=0.1)
ap.add_argument("--per-source", type=int, default=40)
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--chars", type=int, default=300)
ap.add_argument("--url", default="http://127.0.0.1:8787")
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)


def M(m):
    return 0.7 * m["ndcg_at_10"] + 0.3 * m["recall_at_30"]


def reward(s):
    if s["status"] != "ok":
        return -1.0
    over = min(max((s["tokens"] - 320) / 320, 0.0), 1.0)
    return M(s["metrics"]) - M(s["raw"]) - 0.2 * over


if args.mode == "build":
    from reward_client import RewardClient
    client = RewardClient(args.url)
    benches = json.loads(Path(args.bench_map).read_text())
    rng = random.Random(args.seed)
    groups = [json.loads(l) for f in args.scored for l in open(f)]
    by_src = {}
    for g in groups:
        ok = [s for s in g["samples"] if s["status"] == "ok"]
        if len(ok) < 2:
            continue
        rs = [reward(s) for s in ok]
        if max(rs) - min(rs) >= args.min_spread:
            by_src.setdefault(g["source_id"], []).append(g)
    print({k: len(v) for k, v in by_src.items()}, "eligible groups (spread >= %.2f)" % args.min_spread)

    path_cache = {}

    def doc_text(sid, doc_id):
        bench = Path(benches[sid])
        if sid not in path_cache:
            prof = (bench / "retrieval-profile.yaml").read_text()
            corpus = Path(re.search(r"collection_root:\s*(\S+)", prof).group(1))
            if not corpus.is_absolute():
                corpus = bench / corpus
            paths = {d["doc_id"]: d["path"] for d in map(json.loads, open(bench / "documents.jsonl"))}
            path_cache[sid] = (corpus, paths)
        corpus, paths = path_cache[sid]
        text = (corpus / paths[doc_id]).read_text(errors="replace")
        lines = text.split("\n")
        head = lines[0].lstrip("# ").strip()
        body = " ".join(" ".join(lines[1:]).split())
        # FiQA titles are just the doc id; drop them
        return (body if head == doc_id else f"{head} — {body}")[: args.chars]

    items, key = [], {}
    for sid, gs in by_src.items():
        for g in rng.sample(gs, min(args.per_source, len(gs))):
            ok = [s for s in g["samples"] if s["status"] == "ok"]
            ok.sort(key=reward)
            lo, hi = ok[0], ok[-1]
            res = client.score([{"source_id": sid, "qid": g["qid"], "completion": s["completion"], "with_ranking": True}
                                for s in (hi, lo)])
            r_hi, r_lo = res[0]["ranking"][: args.top], res[1]["ranking"][: args.top]
            if r_hi == r_lo:
                continue
            hi_first = rng.random() < 0.5
            a, b = (r_hi, r_lo) if hi_first else (r_lo, r_hi)
            pid = f"q{len(items):03d}"
            items.append({"id": pid, "query": g["query"],
                          "list_A": [doc_text(sid, d) for d in a], "list_B": [doc_text(sid, d) for d in b]})
            raw = hi["raw"]
            bucket = "ceiling" if raw["ndcg_at_10"] >= 1 - 1e-9 else "r30zero" if raw["recall_at_30"] == 0 else "mid"
            key[pid] = {"source_id": sid, "qid": g["qid"], "bucket": bucket, "hi_is": "A" if hi_first else "B",
                        "reward_hi": reward(hi), "reward_lo": reward(lo),
                        "overlap_top10": len(set(r_hi) & set(r_lo))}
    rng.shuffle(items)
    with open(out / "pair-items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "pair-key.json").write_text(json.dumps(key, indent=0))
    print(f"wrote {len(items)} pairs to {out}/pair-items.jsonl")

else:
    key = json.loads((out / "pair-key.json").read_text())
    grades = {r["id"]: r for r in map(json.loads, open(out / "pair-grades.jsonl"))}
    assert set(grades) == set(key), "grade ids != key ids"
    rows = []
    for pid, k in key.items():
        pref = grades[pid]["preferred"]  # "A" | "B" | "tie"
        agree = None if pref == "tie" else (pref == k["hi_is"])
        rows.append({**k, "pref": pref, "agree": agree, "spread": k["reward_hi"] - k["reward_lo"]})

    def report(name, sel):
        dec = [r for r in sel if r["agree"] is not None]
        if not sel:
            return
        ag = sum(r["agree"] for r in dec)
        n = len(dec)
        # Wilson 95% interval for agreement among decided pairs
        if n:
            p = ag / n
            z = 1.96
            c = (p + z * z / (2 * n)) / (1 + z * z / n)
            h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
            ci = f"[{c - h:.0%},{c + h:.0%}]"
        else:
            p, ci = float("nan"), ""
        print(f"  {name:28} n={len(sel):3} tie={len(sel) - n:3} agree={ag}/{n} = {p:.0%} {ci}")

    print("== Judge vs qrels reward (decided pairs exclude ties)")
    report("ALL", rows)
    for b in ("mid", "r30zero", "ceiling"):
        report(f"bucket={b}", [r for r in rows if r["bucket"] == b])
    for sid in sorted({r["source_id"] for r in rows}):
        report(sid, [r for r in rows if r["source_id"] == sid])
    for lo, hi in ((0.1, 0.2), (0.2, 0.4), (0.4, 9)):
        report(f"spread [{lo},{hi})", [r for r in rows if lo <= r["spread"] < hi])
    print(f"  median top-10 overlap between hi and lo lists: {st.median(r['overlap_top10'] for r in rows)}")
