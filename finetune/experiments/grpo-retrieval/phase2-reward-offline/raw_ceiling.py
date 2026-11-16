"""Raw (no-expansion) path-A metrics for every scorable qid of the training sources,
via reward-server.ts: an empty completion is a format_error, so the server returns
raw_metrics. Reports the ceiling share (raw nDCG@10 == 1) and related buckets.

  python3 -I raw_ceiling.py --out <raw.jsonl> --source fiqa-train=<bench dir> [--source ...]

Resumable: qids already in --out are skipped.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_client import RewardClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--source", action="append", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8787")
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

client = RewardClient(args.url)
out = Path(args.out)
done = {}
if out.exists():
    for r in map(json.loads, open(out)):
        done[(r["source_id"], r["qid"])] = r

for spec in args.source:
    sid, bench = spec.split("=", 1)
    rel = set()
    for line in open(Path(bench) / "qrels.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].strip() == "1":
            rel.add(p[0])
    qids = [q["qid"] for q in map(json.loads, open(Path(bench) / "queries.jsonl")) if q["qid"] in rel]
    todo = [q for q in qids if (sid, q) not in done]
    started = time.time()
    with open(out, "a") as f:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            res = client.score([{"source_id": sid, "qid": q, "completion": ""} for q in chunk])
            for q, r in zip(chunk, res):
                row = {"source_id": sid, "qid": q, "raw": r["raw_metrics"]}
                f.write(json.dumps(row) + "\n")
                done[(sid, q)] = row
            f.flush()
            n = b + len(chunk)
            el = time.time() - started
            print(f"{sid} {n}/{len(todo)} {el / n * 1000:.0f}ms/q eta={el / n * (len(todo) - n) / 60:.1f}min", flush=True)

print()
print(f"{'source':14} {'n':>5} {'nDCG@10=1':>10} {'R@30=0':>8} {'MRR=1':>7} {'nDCG>=0.8':>10} {'mean nDCG':>10} {'mean R@30':>10}")
for sid in dict.fromkeys(s.split("=", 1)[0] for s in args.source):
    rows = [r["raw"] for (s, _), r in done.items() if s == sid]
    n = len(rows)
    ceil = sum(r["ndcg_at_10"] >= 1 - 1e-9 for r in rows)
    miss = sum(r["recall_at_30"] == 0 for r in rows)
    mrr1 = sum(r["mrr_at_10"] >= 1 - 1e-9 for r in rows)
    high = sum(r["ndcg_at_10"] >= 0.8 for r in rows)
    print(f"{sid:14} {n:5} {ceil / n:9.1%} {miss / n:7.1%} {mrr1 / n:6.1%} {high / n:9.1%} "
          f"{st.mean(r['ndcg_at_10'] for r in rows):10.4f} {st.mean(r['recall_at_30'] for r in rows):10.4f}")
