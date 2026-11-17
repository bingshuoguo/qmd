"""Verify and benchmark PooledRewardClient.

For n = 1, 2, 4, ... up to the number of servers, score a fresh, disjoint set of --rows queries
(greedy + 8 samples each, so raw caches are cold) with the first n servers and time it. Every pooled
result must equal a single-server rescore of the same items. Agreement with the stored scores in the
scored file is reported too; it must be 100% on the machine type that produced that file.

  python bench_pool.py <samples.scored.jsonl> --urls http://127.0.0.1:8787,http://127.0.0.1:8788,...
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_client import PooledRewardClient, RewardClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scored")
ap.add_argument("--urls", required=True)
ap.add_argument("--rows", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

urls = args.urls.split(",")
rows = [json.loads(line) for line in open(args.scored)]
random.Random(args.seed).shuffle(rows)
sizes = sorted({1, *[n for n in (2, 4, 8, 12, 16) if n <= len(urls)], len(urls)})
single = RewardClient(urls[0])


def same(a, b):
    return a["status"] == b["status"] and a["metrics"] == b["metrics"] and a["raw_metrics"] == b["raw_metrics"]


base = None
failed = False
for k, n in enumerate(sizes):
    batch = rows[k * args.rows:(k + 1) * args.rows]
    stored = [s for r in batch for s in [r["greedy"], *r["samples"]]]
    items = [{"source_id": r["source_id"], "qid": r["qid"], "completion": s["completion"], "truncated": s["truncated"]}
             for r in batch for s in [r["greedy"], *r["samples"]]]
    client = single if n == 1 else PooledRewardClient(urls[:n])
    started = time.time()
    pooled = client.score(items)
    secs = time.time() - started
    reference = single.score(items) if n > 1 else pooled
    mismatch = sum(not same(p, r) for p, r in zip(pooled, reference))
    vs_stored = sum(p["status"] == s["status"] and p["metrics"] == s["metrics"] and p["raw_metrics"] == s["raw"]
                    for p, s in zip(pooled, stored))
    base = base or len(items) / secs
    failed |= mismatch > 0
    print(f"n={n:2}  items={len(items)}  {secs:6.1f}s  {len(items) / secs:5.1f} items/s  "
          f"speedup={len(items) / secs / base:4.2f}x  pooled_vs_single_mismatch={mismatch}  "
          f"equal_to_stored={vs_stored}/{len(items)}", flush=True)
sys.exit(1 if failed else 0)
