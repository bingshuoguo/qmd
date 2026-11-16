"""Step 3.1/3.2: pick training-source queries for the learnability gate and split them.

Pool per source = scorable qids (in train-raw.jsonl), minus
  - queries the policy saw in SFT (v2-VH sft-train.jsonl): by qid within the source and by normalized text,
  - any eval query (fiqa-test, scifact-test, android) by normalized text,
  - ceiling queries (raw nDCG@10 == 1): no expansion can improve them.
Then up to --per-source qids (seed 42), split 70/30 by qid within each source.

  python select_queries.py --train-raw <train-raw.jsonl> --sft <sft-train.jsonl> --out <dir>
"""
import argparse
import json
import random
import re
from pathlib import Path

PREPARED = Path("finetune/data/public-distill-v0/prepared")
TRAIN = {  # reward-server source_id -> (benchmark dir, source_id used in sft-train.jsonl)
    "fiqa-train": ("fiqa-train", "fiqa-train"),
    "scifact-train": ("qmd-distill-public-v0-scifact-train", "scifact-train"),
    "programmers": ("cqadup-programmers", "cqadup-programmers"),
    "unix": ("cqadup-unix", "cqadup-unix"),
}
G0 = Path("artifacts/sft-serving-baseline/g0-20261026/abc/bench")
EVAL = [G0 / "qmd-eval-v1-fiqa-test", G0 / "qmd-expansion-scifact-v1", G0 / "qmd-eval-v1-cqadup-android"]

ap = argparse.ArgumentParser()
ap.add_argument("--train-raw", required=True)
ap.add_argument("--sft", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-source", type=int, default=250)
ap.add_argument("--test-frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


raw = {(r["source_id"], r["qid"]): r["raw"] for r in load(args.train_raw)}
sft = load(args.sft)
sft_ids = {(r["source_id"], r["qid"]) for r in sft}
sft_text = {norm(r["query"]) for r in sft}
eval_text = {norm(q["query"]) for d in EVAL for q in load(d / "queries.jsonl")}

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
manifest = {"seed": args.seed, "per_source": args.per_source, "test_frac": args.test_frac, "sources": {}}
rows = []
for sid, (bench, sft_sid) in TRAIN.items():
    queries = load(PREPARED / bench / "queries.jsonl")
    counts = {"queries": len(queries), "not_scorable": 0, "sft_seen": 0, "eval_text": 0, "ceiling": 0}
    pool = []
    for q in queries:
        r = raw.get((sid, q["qid"]))
        if r is None:
            counts["not_scorable"] += 1
        elif (sft_sid, q["qid"]) in sft_ids or norm(q["query"]) in sft_text:
            counts["sft_seen"] += 1
        elif norm(q["query"]) in eval_text:
            counts["eval_text"] += 1
        elif r["ndcg_at_10"] >= 1 - 1e-9:
            counts["ceiling"] += 1
        else:
            pool.append(q)
    rng = random.Random(f"{args.seed}:{sid}")
    pick = rng.sample(pool, min(args.per_source, len(pool)))
    n_test = round(len(pick) * args.test_frac)
    for i, q in enumerate(pick):
        rows.append({"source_id": sid, "qid": q["qid"], "query": q["query"],
                     "split": "test" if i < n_test else "train", "raw": raw[(sid, q["qid"])]})
    counts.update(eligible=len(pool), picked=len(pick), test=n_test, train=len(pick) - n_test)
    manifest["sources"][sid] = counts
    print(sid, json.dumps(counts))

with open(out / "queries.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
(out / "query-manifest.json").write_text(json.dumps(manifest, indent=2))
print("total", len(rows))
