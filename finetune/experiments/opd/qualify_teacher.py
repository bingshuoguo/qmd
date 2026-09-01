from __future__ import annotations

import argparse
import json
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .contracts import SOURCES, atomic_json, atomic_jsonl, load_config, load_sealed_prompt
from .generation import generate_arm
from .run_retrieval import run_retrieval_arms
from .technical_preflight import run_technical_preflight
from .validate_generations import validate_generation_files, validate_vh


METRICS = ("recall_at_30", "mrr_at_10", "ndcg_at_10")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compute_metrics(retrieval_root: Path, *, resamples: int, seed: int) -> dict[str, Any]:
    by_arm = {arm: _rows(retrieval_root / f"{arm}.jsonl") for arm in ("raw", "merged-v2", "teacher")}
    keyed = {
        arm: {(row["source_id"], str(row["qid"])): row for row in rows}
        for arm, rows in by_arm.items()
    }
    if not keyed["merged-v2"] or set(keyed["teacher"]) != set(keyed["merged-v2"]):
        raise ValueError("teacher and merged-v2 retrieval rows do not align")
    if set(keyed["raw"]) != set(keyed["merged-v2"]):
        raise ValueError("raw retrieval rows do not align with generated arms")
    sources: dict[str, Any] = {}
    for source_id in SOURCES:
        keys = sorted((key for key in keyed["teacher"] if key[0] == source_id), key=lambda item: item[1].encode())
        if not keys:
            raise ValueError(f"retrieval is missing source {source_id}")
        arms: dict[str, dict[str, float]] = {}
        for arm in keyed:
            arms[arm] = {
                metric: mean(float(keyed[arm][key]["metrics"][metric]) for key in keys)
                for metric in METRICS
            }
        sources[source_id] = {
            "queries": len(keys),
            "arms": arms,
            "delta": {
                metric: arms["teacher"][metric] - arms["merged-v2"][metric]
                for metric in METRICS
            },
        }
    macro = {
        "arms": {
            arm: {metric: mean(sources[source]["arms"][arm][metric] for source in SOURCES) for metric in METRICS}
            for arm in keyed
        }
    }
    macro["delta"] = {
        metric: macro["arms"]["teacher"][metric] - macro["arms"]["merged-v2"][metric]
        for metric in METRICS
    }
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        source_deltas: list[float] = []
        for source_id in SOURCES:
            keys = sorted((key for key in keyed["teacher"] if key[0] == source_id), key=lambda item: item[1].encode())
            indexes = [rng.randrange(len(keys)) for _ in keys]
            deltas = [
                float(keyed["teacher"][keys[index]]["metrics"]["recall_at_30"])
                - float(keyed["merged-v2"][keys[index]]["metrics"]["recall_at_30"])
                for index in indexes
            ]
            source_deltas.append(mean(deltas))
        samples.append(mean(source_deltas))
    return {
        "schema_version": "qmd-teacher-qualification-metrics-v1",
        "sources": sources,
        "macro": macro,
        "bootstrap": {
            "metric": "recall_at_30",
            "resamples": resamples,
            "seed": seed,
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        },
    }


def decide(
    *,
    mode: str,
    validation_status: str,
    preflight_status: str,
    retrieval_status: str,
    metrics: dict[str, Any] | None,
    source_floor: float = -0.01,
) -> dict[str, Any]:
    blockers: list[str] = []
    if mode != "formal":
        blockers.append("dry-run is non-formal")
    if validation_status != "valid":
        blockers.append("generation validation is incomplete or invalid")
    if preflight_status != "passed":
        blockers.append("server technical preflight has not passed")
    if retrieval_status != "completed":
        blockers.append("retrieval is incomplete")
    if metrics is None:
        blockers.append("effect metrics are unavailable")
    if blockers:
        return {"status": "not_evaluated", "blockers": blockers, "gate_checks": None}
    macro_pass = float(metrics["macro"]["delta"]["recall_at_30"]) > 0
    source_checks = {
        f"{source_id}.{metric}": float(metrics["sources"][source_id]["delta"][metric]) >= source_floor
        for source_id in SOURCES
        for metric in METRICS
    }
    checks = {"macro_recall_at_30_strictly_positive": macro_pass, **source_checks}
    return {
        "status": "go" if all(checks.values()) else "no_go",
        "blockers": [],
        "gate_checks": checks,
        "bootstrap_is_diagnostic_only": True,
    }


def write_report(root: Path, mode: str, metrics: dict[str, Any] | None, decision: dict[str, Any]) -> None:
    lines = [
        "# Teacher qualification report",
        "",
        f"- Mode: `{mode}`",
        f"- Status: `{decision['status']}`",
        "- Prompt: sealed `qmd-student-expansion-v2-vh-v1`",
        "- Completion budget: 256 tokens",
        "- Retrieval: QMD canonical benchmark with `rerank=false`",
    ]
    if decision["blockers"]:
        lines.extend(["", "## Blockers", "", *[f"- {item}" for item in decision["blockers"]]])
    if metrics is not None:
        lines.extend(
            [
                "",
                "## Macro teacher minus merged-v2",
                "",
                *[f"- {metric}: `{metrics['macro']['delta'][metric]:.6f}`" for metric in METRICS],
                f"- R@30 paired bootstrap 95% CI: `[{metrics['bootstrap']['lower_95']:.6f}, {metrics['bootstrap']['upper_95']:.6f}]` (diagnostic only)",
            ]
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _synthetic_queries() -> list[dict[str, Any]]:
    labels = {
        "nfcorpus": ("nutrition and heart health", "benefits of daily exercise"),
        "fiqa": ("bonds and interest rates", "definition of dividend yield"),
        "freshstack": ("python sort dictionary", "javascript async errors"),
        "scifact": ("vitamin d bone density", "smoking cardiovascular risk"),
    }
    return [
        {"sample_key": f"{source}:{index}", "source_id": source, "qid": str(index), "query": query, "synthetic": True}
        for source, queries in labels.items()
        for index, query in enumerate(queries, 1)
    ]


def dry_run(output_root: Path, *, resamples: int = 2000) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    queries = _synthetic_queries()
    atomic_jsonl(output_root / "queries.jsonl", queries)
    generations: dict[str, list[dict[str, Any]]] = {"merged-v2": [], "teacher": []}
    for arm in generations:
        for row in queries:
            raw = f"hyde: Synthetic passage for {row['query']}.\nvec: {row['query']} retrieval"
            validate_vh(raw)
            generations[arm].append(
                {
                    **row,
                    "arm": arm,
                    "raw_output": raw,
                    "completion_token_ids": [1, 151645],
                    "finish_reason": "eos",
                    "truncated": False,
                    "generation_error": None,
                    "attempt": 1,
                    "selected_attempt": True,
                    "synthetic": True,
                }
            )
        atomic_jsonl(output_root / "generations" / f"{arm}.jsonl", generations[arm])
    validation = validate_generation_files(
        queries_path=output_root / "queries.jsonl",
        generation_paths={arm: output_root / "generations" / f"{arm}.jsonl" for arm in generations},
        output_root=output_root,
    )
    for arm in ("raw", "merged-v2", "teacher"):
        rows = []
        for index, query in enumerate(queries):
            base = 0.30 + (index % 2) * 0.10
            bonus = 0.01 if arm == "teacher" else 0.0
            rows.append(
                {
                    **query,
                    "variant": arm,
                    "retrieval_status": "ok",
                    "fallback_used": None if arm == "raw" else False,
                    "metrics": {
                        "recall_at_30": base + bonus,
                        "mrr_at_10": base / 2 + bonus,
                        "ndcg_at_10": base / 1.5 + bonus,
                    },
                }
            )
        atomic_jsonl(output_root / "retrieval" / f"{arm}.jsonl", rows)
    metrics = compute_metrics(output_root / "retrieval", resamples=resamples, seed=42)
    preflight = {"status": "pending", "reason": "dry-run does not load models or CUDA"}
    atomic_json(output_root / "preflight.json", preflight)
    atomic_json(output_root / "data-audit.json", {"status": "synthetic", "sources": list(SOURCES), "queries": len(queries)})
    decision = decide(
        mode="dry_run",
        validation_status=validation["status"],
        preflight_status=preflight["status"],
        retrieval_status="completed",
        metrics=metrics,
    )
    atomic_json(output_root / "metrics.json", metrics)
    atomic_json(output_root / "decision.json", decision)
    atomic_json(
        output_root / "evaluation-manifest.json",
        {"schema_version": "qmd-teacher-qualification-evaluation-v1", "mode": "dry_run", "synthetic": True},
    )
    write_report(output_root, "dry_run", metrics, decision)
    return decision


def formal_run(
    *,
    config_path: Path,
    bundle_root: Path,
    merged_model: Path,
    output_root: Path,
    merged_batch_size: int,
    teacher_batch_size: int,
) -> dict[str, Any]:
    config = load_config(config_path, mode="formal")
    if output_root.exists():
        raise ValueError(f"formal run output already exists: {output_root}")
    output_root.mkdir(parents=True)
    queries_path = bundle_root / "queries.jsonl"
    queries = _rows(queries_path)
    atomic_jsonl(output_root / "queries.jsonl", queries)
    preflight = run_technical_preflight(config, merged_model, output_root / "preflight.json")
    prompt = load_sealed_prompt(config.release_manifest)
    generation_paths = {
        "merged-v2": output_root / "generations/merged-v2.jsonl",
        "teacher": output_root / "generations/teacher.jsonl",
    }
    merged_config = replace(
        config.merged_v2,
        model_id=str(merged_model),
        revision=None,
        adapter=None,
        adapter_sha256=None,
    )
    generate_arm(model_config=merged_config, generation=config.generation, prompt=prompt, queries=queries, output_path=generation_paths["merged-v2"], arm="merged-v2", batch_size=merged_batch_size)
    generate_arm(model_config=config.teacher, generation=config.generation, prompt=prompt, queries=queries, output_path=generation_paths["teacher"], arm="teacher", batch_size=teacher_batch_size)
    validation = validate_generation_files(queries_path=queries_path, generation_paths=generation_paths, output_root=output_root)
    if validation["status"] != "valid":
        decision = decide(mode="formal", validation_status="invalid", preflight_status=preflight["status"], retrieval_status="blocked", metrics=None)
        atomic_json(output_root / "decision.json", decision)
        write_report(output_root, "formal", None, decision)
        return decision
    retrieval = run_retrieval_arms(config=config, bundle_root=bundle_root, expansions_root=output_root / "expansions", output_root=output_root / "retrieval")
    metrics = compute_metrics(output_root / "retrieval", resamples=config.statistics.bootstrap_resamples, seed=config.statistics.seed)
    decision = decide(mode="formal", validation_status=validation["status"], preflight_status=preflight["status"], retrieval_status=retrieval["status"], metrics=metrics, source_floor=config.statistics.source_floor)
    atomic_json(output_root / "metrics.json", metrics)
    atomic_json(output_root / "decision.json", decision)
    write_report(output_root, "formal", metrics, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B for OPD teacher qualification")
    parser.add_argument("--config", type=Path, default=Path("finetune/experiments/opd/config/teacher-qualification-v1.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--merged-model", type=Path)
    parser.add_argument("--merged-batch-size", type=int, default=8)
    parser.add_argument("--teacher-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.dry_run:
        if args.output:
            output = args.output
            temporary = None
        else:
            temporary = tempfile.TemporaryDirectory(prefix="qmd-teacher-dry-run-")
            output = Path(temporary.name) / "run"
        decision = dry_run(output)
        print(json.dumps({"output": str(output), **decision}, indent=2, sort_keys=True))
        if temporary is not None:
            temporary.cleanup()
        return 0
    if not args.output or not args.bundle or not args.merged_model:
        parser.error("formal run requires --output, --bundle, and --merged-model")
    decision = formal_run(config_path=args.config, bundle_root=args.bundle, merged_model=args.merged_model, output_root=args.output, merged_batch_size=args.merged_batch_size, teacher_batch_size=args.teacher_batch_size)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["status"] in {"go", "no_go", "not_evaluated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
