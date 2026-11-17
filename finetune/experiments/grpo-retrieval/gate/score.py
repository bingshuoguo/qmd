"""Step 3.4/3.5: score greedy + samples through reward-server.ts; optional determinism recheck.

  python score.py <samples.jsonl>              # writes <samples>.scored.jsonl (resumable)
  python score.py <samples.jsonl> --recheck 50 # rescore 50 random completions, compare all metrics

--url takes a comma-separated list to score across several servers (see start-servers.sh).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward_client import make_client  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("samples")
ap.add_argument("--url", default="http://127.0.0.1:8787")
ap.add_argument("--recheck", type=int, default=0)
ap.add_argument("--rows-per-call", type=int, default=8, help="queries per request batch (9 completions each)")
args = ap.parse_args()

client = make_client(args.url)
scored_path = Path(args.samples).with_suffix(".scored.jsonl")


def item(row, s):
    return {"source_id": row["source_id"], "qid": row["qid"], "completion": s["completion"], "truncated": s["truncated"]}


if args.recheck:
    rows = [json.loads(line) for line in open(scored_path)]
    picks = random.Random(1).sample([(r, s) for r in rows for s in [r["greedy"], *r["samples"]]], args.recheck)
    again = client.score([item(r, s) for r, s in picks])
    diff = sum(a["status"] != s["status"] or a["metrics"] != s["metrics"] or a["raw_metrics"] != s["raw"]
               for (_, s), a in zip(picks, again))
    print(f"recheck {args.recheck} completions: {diff} differ")
    sys.exit(1 if diff else 0)

done = {(r["source_id"], r["qid"]) for r in map(json.loads, open(scored_path))} if scored_path.exists() else set()
rows = [r for r in map(json.loads, open(args.samples)) if (r["source_id"], r["qid"]) not in done]
print(f"todo={len(rows)} done={len(done)}", flush=True)
with open(scored_path, "a") as f:
    for b in range(0, len(rows), args.rows_per_call):
        batch = rows[b:b + args.rows_per_call]
        pairs = [(row, s) for row in batch for s in [row["greedy"], *row["samples"]]]
        for (_, s), r in zip(pairs, client.score([item(row, s) for row, s in pairs])):
            s.update(status=r["status"], error=r["error"], metrics=r["metrics"], raw=r["raw_metrics"], ms=r["ms"])
        for row in batch:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        print(f"scored {b + len(batch)}/{len(rows)}", flush=True)

rows = [json.loads(line) for line in open(scored_path)]
completions = [s for r in rows for s in [r["greedy"], *r["samples"]]]
ok = sum(s["status"] == "ok" for s in completions)
print(f"groups={len(rows)} completions={len(completions)} format_ok={ok / len(completions):.2%} "
      f"truncated={sum(s['truncated'] for s in completions)}")
