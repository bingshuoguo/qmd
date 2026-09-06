#!/usr/bin/env python3
"""V2-A Pareto: Task 3 — Strict Pareto Selection (spec section 8).

Reads raw-retrieval.jsonl and candidate-retrieval.jsonl from Task 2,
applies the 5-metric strict Pareto guardrail per query, and selects
the lexicographic tie-breaker winner.

Output: selection-ledger.jsonl + selection-manifest.json
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# === Paths ==============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # finetune/
_EXPERIMENT_DIR = _SCRIPT_DIR / (
    "data/public-distill-v3-vh-pareto/experiments/"
    "public-main-v3-vh-pareto-prompt-v1"
)
_RAW_PATH = _EXPERIMENT_DIR / "2-retrieval-scoring" / "raw-retrieval.jsonl"
_CANDIDATE_PATH = _EXPERIMENT_DIR / "2-retrieval-scoring" / "candidate-retrieval.jsonl"
_OUTPUT_DIR = _EXPERIMENT_DIR / "3-pareto-selection"

# === Metrics ============================================================

PARETO_METRICS = [
    "recall_at_10",
    "recall_at_20",
    "recall_at_30",
    "mrr_at_10",
    "ndcg_at_10",
]


# === Helpers ============================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# === Pareto logic =======================================================


def is_pareto_eligible(
    raw_metrics: dict[str, float],
    cand_metrics: dict[str, Any],
) -> tuple[bool, int, list[str]]:
    """Check strict 5-metric Pareto guardrail.

    Returns (eligible, improved_count, degraded_metrics).
    """
    improved = 0
    degraded: list[str] = []
    for metric in PARETO_METRICS:
        raw_val = raw_metrics.get(metric, 0.0)
        cand_val = cand_metrics.get(metric, 0.0)
        if cand_val < raw_val:
            degraded.append(metric)
        elif cand_val > raw_val:
            improved += 1

    if degraded:
        return False, improved, degraded
    if improved == 0:
        return False, 0, ["no_improvement"]
    return True, improved, []


def tie_break_key(
    cand_metrics: dict[str, Any],
    raw_metrics: dict[str, float],
    improved_count: int,
    canonical_candidate_id: str,
):
    """Lexicographic tie-breaker key (larger = better for first 4, smaller for last)."""
    return (
        improved_count,  # more improved metrics = better
        cand_metrics.get("recall_at_30", 0.0) - raw_metrics.get("recall_at_30", 0.0),
        cand_metrics.get("mrr_at_10", 0.0) - raw_metrics.get("mrr_at_10", 0.0),
        cand_metrics.get("ndcg_at_10", 0.0) - raw_metrics.get("ndcg_at_10", 0.0),
        # Negative canonical_candidate_id for "smaller is better" (string comparison)
        -hash(canonical_candidate_id),
    )


# === Main ===============================================================


def run(
    raw_path: Path = _RAW_PATH,
    candidate_path: Path = _CANDIDATE_PATH,
    output_dir: Path = _OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw metrics
    print("Loading raw retrieval results...")
    raw_records = read_jsonl(raw_path)
    raw_by_input = {
        r["input_id"]: r for r in raw_records if r.get("status") == "scored"
    }
    print(f"  {len(raw_by_input)} raw queries with metrics")

    # Load candidate metrics
    print("Loading candidate retrieval results...")
    candidate_records = read_jsonl(candidate_path)
    candidate_by_input: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in candidate_records:
        if r.get("status") == "scored":
            candidate_by_input[r["input_id"]].append(r)
    print(f"  {len(candidate_records)} candidates across {len(candidate_by_input)} queries")

    # Selection
    ledger: list[dict[str, Any]] = []
    stats: dict[str, int] = Counter()
    source_stats: dict[str, Counter] = defaultdict(Counter)

    for input_id, raw in sorted(raw_by_input.items()):
        raw_metrics = raw.get("metrics", {})
        candidates = candidate_by_input.get(input_id, [])

        if not candidates:
            ledger.append({
                "input_id": input_id,
                "qid": raw["qid"],
                "source_id": raw["source_id"],
                "status": "no_pareto_winner",
                "reason": "no_scored_candidates",
                "candidate_count": 0,
                "eligible_count": 0,
                "raw_metrics": raw_metrics,
                "winner": None,
            })
            stats["no_pareto_winner"] += 1
            source_stats[raw["source_id"]]["no_pareto_winner"] += 1
            continue

        # Evaluate each candidate
        eligible: list[dict[str, Any]] = []
        for cand in candidates:
            cand_metrics = cand.get("metrics", {})
            ok, improved, degraded = is_pareto_eligible(raw_metrics, cand_metrics)

            if ok:
                eligible.append({
                    **cand,
                    "improved_count": improved,
                })

        if not eligible:
            # Determine why: check if any had candidates but none were eligible
            reason = "no_candidate_improves_raw"
            ledger.append({
                "input_id": input_id,
                "qid": raw["qid"],
                "source_id": raw["source_id"],
                "status": "no_pareto_winner",
                "reason": reason,
                "candidate_count": len(candidates),
                "eligible_count": 0,
                "raw_metrics": raw_metrics,
                "winner": None,
            })
            stats["no_pareto_winner"] += 1
            source_stats[raw["source_id"]]["no_pareto_winner"] += 1
            continue

        # Select winner via lexicographic tie-breaker
        best = max(
            eligible,
            key=lambda c: tie_break_key(
                c["metrics"],
                raw_metrics,
                c["improved_count"],
                c["canonical_candidate_id"],
            ),
        )

        deltas = {
            m: round(best["metrics"].get(m, 0.0) - raw_metrics.get(m, 0.0), 6)
            for m in PARETO_METRICS
        }

        ledger.append({
            "input_id": input_id,
            "qid": raw["qid"],
            "source_id": raw["source_id"],
            "status": "winner",
            "candidate_count": len(candidates),
            "eligible_count": len(eligible),
            "raw_metrics": raw_metrics,
            "winner": {
                "source_candidate_id": best["source_candidate_id"],
                "canonical_candidate_id": best["canonical_candidate_id"],
                "improved_count": best["improved_count"],
                "metrics": best["metrics"],
                "deltas": deltas,
            },
        })
        stats["winner"] += 1
        source_stats[raw["source_id"]]["winner"] += 1

    # Write outputs
    ledger_path = output_dir / "selection-ledger.jsonl"
    atomic_jsonl(ledger_path, ledger)
    print(f"\nLedger: {len(ledger)} records → {ledger_path}")

    # Manifest
    total_winners = stats["winner"]
    total_no_winner = stats["no_pareto_winner"]
    total = total_winners + total_no_winner

    manifest = {
        "schema_version": "qmd-v2a-selection-v1",
        "selection_rule": "strict_5_metric_pareto",
        "pareto_metrics": PARETO_METRICS,
        "tie_breaker": [
            "improved_metric_count",
            "recall_at_30_delta",
            "mrr_at_10_delta",
            "ndcg_at_10_delta",
            "smaller_canonical_candidate_id",
        ],
        "total_queries": total,
        "status_counts": {
            "winner": total_winners,
            "no_pareto_winner": total_no_winner,
        },
        "winner_rate": round(total_winners / total, 4) if total > 0 else 0,
        "by_source": {
            src: {
                "winner": counts.get("winner", 0),
                "no_pareto_winner": counts.get("no_pareto_winner", 0),
                "total": counts.get("winner", 0) + counts.get("no_pareto_winner", 0),
            }
            for src, counts in sorted(source_stats.items())
        },
        "inputs": {
            "raw_retrieval": str(raw_path),
            "candidate_retrieval": str(candidate_path),
        },
    }
    manifest_path = output_dir / "selection-manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    print(f"\nResults: {total_winners} winners, {total_no_winner} no_pareto_winner ({total} total queries)")

    return output_dir


if __name__ == "__main__":
    run()