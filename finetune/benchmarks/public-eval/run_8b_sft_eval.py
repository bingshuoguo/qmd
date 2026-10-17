#!/usr/bin/env python3
"""Preflight and run the 8B SFT retrieval evaluation without touching old arms.

The default ``--dry-run`` is CPU-only: it validates frozen benchmark inputs,
reports exact-query overlap with the SFT release, and prints the GPU
commands. ``--run`` executes only the new 8B SFT arm and refuses to overwrite
an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from generate_expansions import GENERATION, load_evaluation_release, runtime_environment


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen3-8B")
DEFAULT_ADAPTER = Path(
    "/root/autodl-tmp/runs/public-main-v2-vh-prompt-v1-fsdp-bf16-v11/final-adapter"
)
DEFAULT_RELEASE = ROOT / (
    "finetune/data/public-distill-v2-vh-prompt-v1/experiments/"
    "public-main-v2-vh-prompt-v1/release-manifest.json"
)
DEFAULT_VARIANT = "8b-sft-v2vh-sealed-v1"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SEALED_PROMPT_SHA256 = "fce6b582fff10e7e1a601097e0fa00273f5b106fcd821985422932a5ce797d93"
ENVIRONMENT_LOCK = ROOT / "finetune/requirements-eval.txt"
DEFAULT_INDEX = ROOT / "finetune/data/qwen3-8b-eval-index-v1/index.sqlite"
DEFAULT_CONFIG_DIR = ROOT / "finetune/data/qwen3-8b-eval-index-v1/config"


@dataclass(frozen=True)
class Benchmark:
    name: str
    path: Path


BENCHMARKS = (
    Benchmark("scifact", ROOT / "finetune/benchmarks/qmd-expansion-scifact-v1"),
    Benchmark("fiqa-test", ROOT / "finetune/data/public-eval-v1/benchmarks/qmd-eval-v1-fiqa-test"),
    Benchmark("fiqa-dev", ROOT / "finetune/data/public-eval-v1/benchmarks/qmd-eval-v1-fiqa-dev"),
    Benchmark("cqadup-android", ROOT / "finetune/data/public-eval-v1/benchmarks/qmd-eval-v1-cqadup-android"),
    Benchmark("cqadup-webmasters", ROOT / "finetune/data/public-eval-v1/benchmarks/qmd-eval-v1-cqadup-webmasters"),
    Benchmark("nfcorpus-test", ROOT / "finetune/data/nf-freshstack-v1/splits/nfcorpus/qmd-nf-freshstack-v1-nfcorpus-test"),
    Benchmark("freshstack-test", ROOT / "finetune/data/nf-freshstack-v1/splits/freshstack/qmd-nf-freshstack-v1-freshstack-test"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_query(value: str) -> str:
    return " ".join(value.casefold().split())


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")


def release_queries(release_path: Path) -> tuple[set[str], dict[str, Any]]:
    release = load_evaluation_release(release_path)
    if release.prompt_sha256 != SEALED_PROMPT_SHA256:
        raise ValueError("8B evaluation requires the full sealed V2-VH training prompt")
    records = release.train.records + release.validation.records
    return {normalize_query(record["query"]) for record in records}, release.provenance()


def verify_environment() -> dict[str, Any]:
    environment = runtime_environment()
    for line in ENVIRONMENT_LOCK.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, version = line.split("==")
        installed = environment["packages"].get(name)
        if installed != version:
            raise ValueError(f"evaluation environment mismatch: {name}=={version}, installed {installed}; "
                             "use the pinned evaluation interpreter, do not upgrade it in place")
    environment["requirements_sha256"] = sha256(ENVIRONMENT_LOCK)
    return environment


def preflight_benchmark(benchmark: Benchmark, train_queries: set[str], variant: str) -> dict[str, Any]:
    required = ("benchmark.yaml", "documents.jsonl", "queries.jsonl", "qrels.tsv", "retrieval-profile.yaml")
    for name in required:
        require_file(benchmark.path / name)
    manifest = yaml.safe_load((benchmark.path / "benchmark.yaml").read_text(encoding="utf-8"))
    profile = yaml.safe_load((benchmark.path / "retrieval-profile.yaml").read_text(encoding="utf-8"))
    queries = jsonl(benchmark.path / "queries.jsonl")
    if not queries:
        raise ValueError(f"{benchmark.name}: queries.jsonl is empty")
    if len({record["qid"] for record in queries}) != len(queries):
        raise ValueError(f"{benchmark.name}: duplicate qid")
    overlap = [record["qid"] for record in queries if normalize_query(record["query"]) in train_queries]
    outputs = {
        kind: benchmark.path / directory / f"{variant}{suffix}"
        for kind, directory, suffix in (
            ("raw", "raw-generations", ".jsonl"),
            ("raw_manifest", "raw-generations", "-manifest.json"),
            ("expansions", "expansions", ".jsonl"),
            ("run_all", "runs", "-qwen3-8b.json"),
            ("results", "runs/results", "-qwen3-8b.jsonl"),
            ("partial_results", "runs/results", "-qwen3-8b.jsonl.partial"),
        )
    }
    collisions = [kind for kind, path in outputs.items() if path.exists()]
    if collisions:
        raise ValueError(f"{benchmark.name}: target artifact already exists: {', '.join(collisions)}")
    index_manifest = benchmark.path / "index-manifest.json"
    blockers = []
    if manifest["benchmark_id"] != profile["collection_name"]:
        blockers.append(
            "benchmark_id and retrieval-profile collection_name differ; QMD bench rejects this configuration"
        )
    if profile.get("rerank") is not False:
        blockers.append("retrieval-profile rerank must be false for this evaluation")
    if not index_manifest.is_file():
        blockers.append(
            "missing index-manifest.json; run `bun src/cli/qmd.ts bench <benchmark> --check-index` after the benchmark collection is indexed"
        )
    return {
        "name": benchmark.name,
        "path": str(benchmark.path),
        "queries": len(queries),
        "training_query_overlap_qids": overlap,
        "benchmark_sha256": sha256(benchmark.path / "benchmark.yaml"),
        "queries_sha256": sha256(benchmark.path / "queries.jsonl"),
        "qrels_sha256": sha256(benchmark.path / "qrels.tsv"),
        "profile_sha256": sha256(benchmark.path / "retrieval-profile.yaml"),
        "index_manifest_sha256": sha256(index_manifest) if index_manifest.is_file() else None,
        "blockers": blockers,
        "outputs": {kind: str(path) for kind, path in outputs.items()},
    }


def commands(benchmark: Benchmark, model: Path, adapter: Path, variant: str, batch_size: int, release: Path, revision: str) -> list[list[str]]:
    generator = ROOT / "finetune/benchmarks/public-eval/generate_expansions.py"
    materialize = ROOT / "finetune/benchmarks/public-eval/materialize-expansions.ts"
    common = [sys.executable, "-B", str(generator), "--benchmark", str(benchmark.path),
              "--variant", variant, "--model", str(model), "--revision", revision,
              "--adapter", str(adapter), "--release-manifest", str(release), "--batch-size", str(batch_size)]
    return [
        common,
        ["node", "--import", "tsx", str(materialize), "--benchmark", str(benchmark.path), "--variant", variant],
        ["node", "--import", "tsx", str(ROOT / "src/cli/qmd.ts"), "bench", str(benchmark.path), "--run", variant, "--model", "Qwen3-8B", "--json"],
    ]


def verify_frozen_indexes(index_path: Path, benchmarks: tuple[Benchmark, ...]) -> dict[str, Any]:
    verifier = ROOT / "finetune/benchmarks/public-eval/verify_frozen_indexes.ts"
    command = ["node", "--import", "tsx", str(verifier), "--db", str(index_path)]
    for benchmark in benchmarks:
        command.extend(("--benchmark", str(benchmark.path)))
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute GPU generation and retrieval after preflight.")
    mode.add_argument("--dry-run", action="store_true", help="CPU-only preflight (the default).")
    parser.add_argument("--asset-root", type=Path, default=ROOT,
                        help="Existing repository containing frozen benchmark/release/index assets.")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.asset_root = args.asset_root.resolve()
    for field, default in (("release", DEFAULT_RELEASE), ("index_path", DEFAULT_INDEX),
                           ("config_dir", DEFAULT_CONFIG_DIR)):
        value = getattr(args, field)
        setattr(args, field, value.resolve() if value else args.asset_root / default.relative_to(ROOT))
    args.model = args.model.resolve()
    args.adapter = args.adapter.resolve()
    benchmarks = tuple(Benchmark(b.name, args.asset_root / b.path.relative_to(ROOT)) for b in BENCHMARKS)
    if not args.variant or Path(args.variant).name != args.variant or args.variant in (".", ".."):
        parser.error("--variant must be a filename component")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be an immutable 40-character model revision")
    environment = verify_environment()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    for path in (
        args.model / "config.json",
        args.adapter / "adapter_config.json",
        args.adapter / "adapter_model.safetensors",
        args.release,
        args.index_path,
        args.config_dir / "index.yml",
    ):
        require_file(path)
    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if Path(adapter_config["base_model_name_or_path"]).name != args.model.name:
        raise ValueError("adapter base model and --model do not match")

    train_queries, release = release_queries(args.release)
    benchmark_reports = [preflight_benchmark(item, train_queries, args.variant) for item in benchmarks]
    blockers = [
        f"{benchmark['name']}: {blocker}"
        for benchmark in benchmark_reports
        for blocker in benchmark["blockers"]
    ]
    if blockers:
        raise ValueError("cannot verify retrieval until preflight blockers are resolved:\n" + "\n".join(blockers))
    report_path = args.asset_root / "finetune/evaluation-runs" / f"{args.variant}.json"
    if report_path.exists():
        raise ValueError(f"evaluation preflight already exists, use a new variant: {report_path}")
    index_verification = verify_frozen_indexes(args.index_path, benchmarks)
    report = {
        "mode": "run" if args.run else "dry-run",
        "variant": args.variant,
        "retrieval_arms": ["all"],
        "revision": args.revision,
        "prompt": release,
        "environment": environment,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "finetune/benchmarks/public-eval/generate_expansions.py")},
        "model": str(args.model),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "release": str(args.release),
        "release_sha256": sha256(args.release),
        "index_path": str(args.index_path),
        "config_dir": str(args.config_dir),
        "index_verification": index_verification,
        "generation": {**GENERATION, "batch_size": args.batch_size},
        "benchmarks": benchmark_reports,
    }
    print(json.dumps(report, indent=2))
    if not args.run:
        print("\nGPU commands (not executed):")
        for benchmark in benchmarks:
            for command in commands(benchmark, args.model, args.adapter, args.variant, args.batch_size, args.release, args.revision):
                print(shlex.join(command))
        return 0

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    qmd_env = {
        **os.environ,
        "INDEX_PATH": str(args.index_path.resolve()),
        "QMD_CONFIG_DIR": str(args.config_dir.resolve()),
    }
    for benchmark in benchmarks:
        for command in commands(benchmark, args.model, args.adapter, args.variant, args.batch_size, args.release, args.revision):
            print("+", shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True, env=qmd_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
