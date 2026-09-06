#!/usr/bin/env python3
"""V2-A Pareto: Task 7 — Evaluation (spec section 16-18).

Runs the full evaluation pipeline for each arm × dataset:
  1. Generate expansions (generate_expansions.py)
  2. Materialize expansions (materialize-expansions.ts)
  3. Run retrieval (qmd bench)
  4. Compute metrics

Usage:
  uv run python -m benchmarks.v2a-pareto.evaluate --arm all
  uv run python -m benchmarks.v2a-pareto.evaluate --arm v2a --dataset scifact
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# === Paths ==============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent  # benchmarks/v2a-pareto/
_FINETUNE = _SCRIPT_DIR.parent.parent  # finetune/
_REPO_ROOT = _FINETUNE.parent  # qmd/
_BENCHMARKS = _SCRIPT_DIR.parent  # finetune/benchmarks/

QMD_BIN = str(_REPO_ROOT / "bin" / "qmd")

# Base model (local cache)
BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Adapters
V2_VH_ADAPTER = str(
    _FINETUNE
    / "artifacts/sft-runs/02-public-main-v2-vh-prompt-v1/training/"
      "public-main-v2-vh-prompt-v1/final-adapter"
)
V2A_ADAPTER = str(
    _FINETUNE
    / "data/public-distill-v3-vh-pareto/experiments/"
      "public-main-v3-vh-pareto-prompt-v1/6-training/checkpoints/checkpoint-226"
)

# Evaluation datasets
DATASETS = {
    "scifact": str(_BENCHMARKS / "qmd-expansion-scifact-v1"),
    "fiqa-test": str(_FINETUNE / "data/public-eval-v1/benchmarks/qmd-eval-v1-fiqa-test"),
    "fiqa-dev": str(_FINETUNE / "data/public-eval-v1/benchmarks/qmd-eval-v1-fiqa-dev"),
    "android": str(_FINETUNE / "data/public-eval-v1/benchmarks/qmd-eval-v1-cqadup-android"),
    "webmasters": str(_FINETUNE / "data/public-eval-v1/benchmarks/qmd-eval-v1-cqadup-webmasters"),
    "nfcorpus-test": str(
        _FINETUNE
        / "data/nf-freshstack-v1/splits/nfcorpus/qmd-nf-freshstack-v1-nfcorpus-test"
    ),
    "freshstack-test": str(
        _FINETUNE
        / "data/nf-freshstack-v1/splits/freshstack/qmd-nf-freshstack-v1-freshstack-test"
    ),
}

PRIMARY_DATASETS = [
    "scifact", "fiqa-test", "android", "nfcorpus-test", "freshstack-test"
]
STRESS_DATASETS = ["webmasters", "fiqa-dev"]

# Evaluation arms: (arm_name, adapter_path, prompt_version)
ARMS = {
    "sft-vh-orig": (V2_VH_ADAPTER, "v2-vh"),
    "sft-vh-v2a": (V2_VH_ADAPTER, "v2a-pareto"),
    "v2a": (V2A_ADAPTER, "v2a-pareto"),
}

# Raw baseline is handled separately (no generation needed)


def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, **kwargs)


def generate_expansions(
    benchmark_dir: str,
    variant: str,
    adapter: str | None,
    prompt_version: str,
) -> Path:
    """Run generate_expansions.py for one arm on one dataset."""
    output_dir = Path(benchmark_dir) / "raw-generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{variant}.jsonl"

    if output_path.exists():
        print(f"  Generation exists: {output_path}", file=sys.stderr)
        return output_path

    cmd = [
        sys.executable, "-m", "benchmarks.public_eval.generate_expansions",
        "--benchmark", benchmark_dir,
        "--variant", variant,
        "--model", BASE_MODEL,
        "--revision", BASE_REVISION,
        "--prompt-version", prompt_version,
        "--batch-size", "4",
    ]
    if adapter:
        cmd.extend(["--adapter", adapter])

    run(cmd, cwd=str(_FINETUNE))
    return output_path


def materialize_expansions(
    benchmark_dir: str,
    variant: str,
) -> Path:
    """Run materialize-expansions.ts for one arm on one dataset."""
    output_path = Path(benchmark_dir) / "expansions" / f"{variant}.jsonl"
    if output_path.exists():
        print(f"  Materialized exists: {output_path}", file=sys.stderr)
        return output_path

    cmd = [
        "npx", "tsx",
        str(_BENCHMARKS / "public-eval" / "materialize-expansions.ts"),
        "--benchmark", benchmark_dir,
        "--variant", variant,
    ]
    run(cmd, cwd=str(_REPO_ROOT))
    return output_path


def run_retrieval(
    benchmark_dir: str,
    variant: str,
) -> Path:
    """Run qmd bench for one arm on one dataset."""
    runs_dir = Path(benchmark_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Check if already run
    existing = list(runs_dir.glob(f"{variant}-*.json"))
    if existing:
        print(f"  Retrieval exists: {existing[0]}", file=sys.stderr)
        return existing[0]

    run(
        [QMD_BIN, "bench", benchmark_dir],
        cwd=str(_REPO_ROOT),
    )
    # qmd bench produces a run file with a timestamped name; we need to find it
    # and rename to include the variant
    return runs_dir  # caller handles


def collect_metrics(benchmark_dir: str) -> dict[str, Any]:
    """Parse the latest run JSON for each variant."""
    runs_dir = Path(benchmark_dir) / "runs"
    metrics = {}
    for run_file in sorted(runs_dir.glob("*.json")):
        with open(run_file) as f:
            data = json.load(f)
        name = run_file.stem
        metrics[name] = {
            "recall_at_10": data.get("recall_at_10"),
            "recall_at_20": data.get("recall_at_20"),
            "recall_at_30": data.get("recall_at_30"),
            "mrr_at_10": data.get("mrr_at_10"),
            "ndcg_at_10": data.get("ndcg_at_10"),
        }
    return metrics


# === Main ===============================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V2-A Evaluation")
    parser.add_argument("--arm", default="all",
                        choices=["all", "raw", "sft-vh-orig", "sft-vh-v2a", "v2a"])
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = PRIMARY_DATASETS + STRESS_DATASETS
    else:
        datasets = [args.dataset]

    if args.arm == "all":
        arms = ARMS
    elif args.arm == "raw":
        arms = {}
    else:
        arms = {args.arm: ARMS[args.arm]}

    print(f"=== V2-A Evaluation ===", file=sys.stderr)
    print(f"Arms: {list(arms.keys()) if arms else ['raw']}", file=sys.stderr)
    print(f"Datasets: {datasets}", file=sys.stderr)
    print(f"Rerank: false", file=sys.stderr)

    # Step 1: Generate raw baseline retrieval (no expansion)
    if not args.skip_retrieval:
        print("\n--- Raw baseline retrieval ---", file=sys.stderr)
        for ds in datasets:
            ds_path = DATASETS[ds]
            print(f"\nDataset: {ds}", file=sys.stderr)
            run_retrieval(ds_path, "raw")

    # Step 2: Generate expansions for each arm
    if not args.skip_generation:
        for arm_name, (adapter, prompt_ver) in arms.items():
            print(f"\n=== Arm: {arm_name} (prompt={prompt_ver}) ===", file=sys.stderr)
            for ds in datasets:
                ds_path = DATASETS[ds]
                print(f"\nDataset: {ds}", file=sys.stderr)
                generate_expansions(ds_path, arm_name, adapter, prompt_ver)

    # Step 3: Materialize expansions
    if not args.skip_generation:
        for arm_name in arms:
            print(f"\n=== Materialize: {arm_name} ===", file=sys.stderr)
            for ds in datasets:
                ds_path = DATASETS[ds]
                print(f"\nDataset: {ds}", file=sys.stderr)
                materialize_expansions(ds_path, arm_name)

    # Step 4: Run retrieval for each arm
    if not args.skip_retrieval:
        for arm_name in arms:
            print(f"\n=== Retrieval: {arm_name} ===", file=sys.stderr)
            for ds in datasets:
                ds_path = DATASETS[ds]
                print(f"\nDataset: {ds}", file=sys.stderr)
                run_retrieval(ds_path, arm_name)

    # Step 5: Collect metrics
    print("\n=== Results ===", file=sys.stderr)
    all_metrics: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        ds_path = DATASETS[ds]
        all_metrics[ds] = collect_metrics(ds_path)

    # Summary table
    print("\nPrimary Macro (5 datasets):", file=sys.stderr)
    for metric in ["recall_at_10", "recall_at_20", "recall_at_30", "mrr_at_10", "ndcg_at_10"]:
        values = []
        for ds in PRIMARY_DATASETS:
            if ds in all_metrics:
                for run_name, run_metrics in all_metrics[ds].items():
                    if metric in run_metrics:
                        values.append(run_metrics[metric])
        if values:
            print(f"  {metric}: {sum(values)/len(values):.4f}", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()