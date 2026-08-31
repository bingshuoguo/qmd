from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .contracts import QualificationConfig, atomic_json, atomic_jsonl, load_config


ARMS = ("raw", "merged-v2", "teacher")
SOURCES = ("nfcorpus", "fiqa", "freshstack", "scifact")
DEFAULT_QMD = ["node", "--import", "tsx", "src/cli/qmd.ts"]


def validate_profile(path: Path, expected_embedding_model: str) -> dict[str, Any]:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if profile.get("rerank") is not False:
        raise ValueError(f"{path}: qualification requires rerank=false")
    if profile.get("embedding_model") != expected_embedding_model:
        raise ValueError(f"{path}: embedding model differs from the frozen configuration")
    if profile.get("collection_name") in {None, ""}:
        raise ValueError(f"{path}: collection_name is missing")
    return profile


def build_bench_command(
    qmd_command: list[str], benchmark: Path, arm: str, model: str | None
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"unknown retrieval arm {arm!r}")
    if arm == "raw" and model is not None:
        raise ValueError("raw retrieval cannot declare an expansion model")
    if arm != "raw" and not model:
        raise ValueError(f"{arm} retrieval requires a model identity")
    command = [*qmd_command, "bench", str(benchmark), "--run", arm]
    if model:
        command.extend(["--model", model])
    command.append("--json")
    return command


def _run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"command did not return JSON: {' '.join(command)}") from error
    return {"command": command, "stdout": payload, "stderr": completed.stderr}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def materialize_source_expansions(
    source_id: str, benchmark: Path, global_expansions: Path, arm: str
) -> Path:
    rows = _rows(global_expansions)
    selected = [
        {
            "qid": str(row["qid"]),
            "query": row["query"],
            "status": row["status"],
            "raw_output": row["raw_output"],
            "output": row["output"],
            "fallback_used": row["fallback_used"],
            "error": row["error"],
        }
        for row in rows
        if row["source_id"] == source_id
    ]
    output = benchmark / "expansions" / f"{arm}.jsonl"
    atomic_jsonl(output, selected)
    return output


def _normalize_results(source_id: str, benchmark: Path, run: dict[str, Any]) -> list[dict[str, Any]]:
    summary = run["stdout"]
    if summary.get("status") != "completed" or summary.get("metrics") is None:
        raise RuntimeError(f"{source_id}/{summary.get('variant')}: QMD run is incomplete")
    failures = summary.get("expansion_failures", {})
    if summary.get("variant") != "raw" and (
        failures.get("format_error_count") != 0
        or failures.get("generation_error_count") != 0
        or failures.get("fallback_count") != 0
    ):
        raise RuntimeError(f"{source_id}/{summary.get('variant')}: invalid expansion accounting")
    results_path = benchmark / str(summary["results"])
    rows = _rows(results_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if row.get("retrieval_status") != "ok" or row.get("metrics") is None:
            raise RuntimeError(f"{source_id}:{row.get('qid')}: retrieval did not complete")
        if row.get("fallback_used") not in {False, None}:
            raise RuntimeError(f"{source_id}:{row.get('qid')}: fallback was used")
        normalized.append({"sample_key": f"{source_id}:{row['qid']}", "source_id": source_id, **row})
    return normalized


def run_retrieval_arms(
    *,
    config: QualificationConfig,
    bundle_root: Path,
    expansions_root: Path,
    output_root: Path,
    qmd_command: list[str] | None = None,
) -> dict[str, Any]:
    qmd = qmd_command or DEFAULT_QMD
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    stages: list[dict[str, Any]] = []
    for source_id in SOURCES:
        benchmark = bundle_root / "benchmarks" / source_id
        profile = validate_profile(
            benchmark / "retrieval-profile.yaml", config.retrieval.embedding_model
        )
        benchmark_id = yaml.safe_load((benchmark / "benchmark.yaml").read_text(encoding="utf-8"))["benchmark_id"]
        if profile["collection_name"] != benchmark_id:
            raise ValueError(f"{source_id}: collection_name must equal benchmark_id")
        check = _run([*qmd, "bench", str(benchmark), "--check-index"])
        stages.append({"source_id": source_id, "stage": "check-index", **check})
        for arm in ("merged-v2", "teacher"):
            materialize_source_expansions(
                source_id, benchmark, expansions_root / f"{arm}.jsonl", arm
            )
        for arm in ARMS:
            model = None if arm == "raw" else arm
            run = _run(build_bench_command(qmd, benchmark, arm, model))
            stages.append({"source_id": source_id, "stage": arm, **run})
            by_arm[arm].extend(_normalize_results(source_id, benchmark, run))
    output_root.mkdir(parents=True, exist_ok=True)
    for arm, rows in by_arm.items():
        atomic_jsonl(output_root / f"{arm}.jsonl", rows)
    summary = {
        "schema_version": "qmd-teacher-qualification-retrieval-v1",
        "status": "completed",
        "counts": {arm: len(rows) for arm, rows in by_arm.items()},
        "stages": stages,
    }
    atomic_json(output_root / "retrieval-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical QMD teacher qualification retrieval")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expansions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config, mode="formal")
    result = run_retrieval_arms(
        config=config,
        bundle_root=args.bundle,
        expansions_root=args.expansions,
        output_root=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
