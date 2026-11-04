"""P0 check: the reward server must reproduce a sealed bench run.

Replays a frozen expansions file through the reward server and compares the
per-query metrics with the bench results file produced for the same run.

  python3 verify_p0.py --source fiqa \
    --expansions <bench>/expansions/g0d-fiqa.jsonl \
    --results <bench>/runs/results/g0d-fiqa-local-d-q8-nosys-greedy.jsonl \
    --raw-results <bench>/runs/results/raw.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_client import RewardClient  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--expansions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--raw-results", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--url", default="http://127.0.0.1:8787")
    args = parser.parse_args()

    client = RewardClient(args.url)
    print(json.dumps(client.health()), flush=True)

    expansions = load_jsonl(args.expansions)
    if args.limit:
        expansions = expansions[: args.limit]
    expected = {row["qid"]: row["metrics"] for row in load_jsonl(args.results)}
    expected_raw = {row["qid"]: row["metrics"] for row in load_jsonl(args.raw_results)}

    items = [{
        "source_id": args.source,
        "qid": record["qid"],
        "completion": record["raw_output"],
        "truncated": (record.get("generation") or {}).get("stop_reason") != "eogToken",
    } for record in expansions]

    results, started = [], time.time()
    for start in range(0, len(items), args.batch):
        results.extend(client.score(items[start:start + args.batch]))
        print(f"scored {len(results)}/{len(items)} elapsed={time.time() - started:.0f}s", flush=True)

    def compare(key: str, reference: dict) -> tuple[int, float, float, float]:
        mismatch = sum(
            1 for result in results
            if abs(result[key]["ndcg_at_10"] - reference[result["qid"]]["ndcg_at_10"]) > 1e-9
            or abs(result[key]["recall_at_30"] - reference[result["qid"]]["recall_at_30"]) > 1e-9
        )
        server = statistics.mean(result[key]["ndcg_at_10"] for result in results)
        frozen = statistics.mean(reference[result["qid"]]["ndcg_at_10"] for result in results)
        max_abs = max(abs(result[key]["ndcg_at_10"] - reference[result["qid"]]["ndcg_at_10"]) for result in results)
        return mismatch, server, frozen, max_abs

    status_mismatch = sum(
        1 for record, result in zip(expansions, results)
        if (record["status"] == "ok") != (result["status"] == "ok")
    )
    report = {"queries": len(results), "status_mismatch": status_mismatch}
    for label, key, reference in (("expansion", "metrics", expected), ("raw", "raw_metrics", expected_raw)):
        mismatch, server, frozen, max_abs = compare(key, reference)
        report[label] = {
            "per_query_mismatch": mismatch,
            "server_ndcg10": round(server, 4),
            "frozen_ndcg10": round(frozen, 4),
            "max_abs_diff": round(max_abs, 4),
        }
    timings = [result["ms"] for result in results]
    report["ms_per_item"] = {
        "p50": statistics.median(timings),
        "p95": sorted(timings)[int(0.95 * (len(timings) - 1))],
        "note": "first request per qid includes the raw retrieval",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
