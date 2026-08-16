#!/usr/bin/env python3
"""Freeze the independent evaluation manifest (spec section 15).

This is the artifact that must exist before the training launch gate can pass.
It pins, for every benchmark in scope:

- query / corpus / qrels content hashes and counts;
- the leakage (de-duplication vs training) result;
- collection isolation and the QMD index fingerprint;
- the retrieval profile, reranker and embedding model;
- the generation configuration shared by the raw/base/SFT arms;
- the metric hierarchy and the scoring commit.

Once sealed, any change to a benchmark input, index, retrieval profile or
scoring implementation requires a new evaluation version.

Usage:
    uv run python -m benchmarks.public_eval.freeze_manifest
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml

from dataset.public_distill import atomic_json, sha256_file
from dataset.public_distill_v1 import (
    PROMPT_VERSION,
    TOKENIZER_REVISION,
    prompt_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "finetune/data/public-eval-v1"
BENCHMARKS = EVAL_ROOT / "benchmarks"
SCIFACT_BENCHMARK = ROOT / "finetune/benchmarks/qmd-expansion-scifact-v1"
INDEX_DB = ROOT / "finetune/data/public-distill-v0/indexes/public-sft-v0/index.sqlite"

EVAL_VERSION = "public-eval-v1"
TRAINING_RELEASE = {
    "release_id": "public-distill-v1",
    "experiment_id": "public-main-v1",
}

# Spec section 16.  Frozen for all arms; the Python generator reads the same
# literal from dataset/public_distill_v1, this copy is the manifest's record.
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 768,
    "seed": 42,
    "use_cache": True,
    "generation_padding_side": "left",
    "grammar": None,
}

# Spec section 17.
METRICS = {
    "primary": "recall_at_10",
    "guardrail": ["ndcg_at_10", "mrr_at_10"],
    "diagnostic": ["recall_at_20", "recall_at_30"],
    "aggregation": "benchmark-equal-weight-macro",
    "invalid_output": (
        "Contract invalid / empty / parse failure / truncation stay in the "
        "denominator and fall back to the frozen raw retrieval path."
    ),
}

# (benchmark_id, role, directory, requires_local_index_manifest)
FORMAL = "formal"
DIAGNOSTIC = "diagnostic"
SCOPES = [
    ("qmd-eval-v1-fiqa-test", FORMAL, BENCHMARKS / "qmd-eval-v1-fiqa-test", True),
    ("qmd-eval-v1-fiqa-dev", DIAGNOSTIC, BENCHMARKS / "qmd-eval-v1-fiqa-dev", True),
    ("qmd-eval-v1-cqadup-android", FORMAL, BENCHMARKS / "qmd-eval-v1-cqadup-android", True),
    ("qmd-eval-v1-cqadup-webmasters", FORMAL, BENCHMARKS / "qmd-eval-v1-cqadup-webmasters", True),
    ("qmd-expansion-scifact-v1", FORMAL, SCIFACT_BENCHMARK, True),
]


def _rows(path: Path) -> int:
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _qrel_rows(path: Path) -> int:
    # qrels.tsv carries a header line.
    return max(0, _rows(path) - 1)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=path
    ).strip()


def _git_dirty(path: Path) -> list[str]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, cwd=path
    )
    return [line for line in status.splitlines() if line.strip()]


def load_benchmark(benchmark_id: str, directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValueError(f"benchmark directory is missing: {directory}")

    profile_path = directory / "retrieval-profile.yaml"
    benchmark_path = directory / "benchmark.yaml"
    index_manifest_path = directory / "index-manifest.json"
    for path in (profile_path, benchmark_path):
        if not path.is_file():
            raise ValueError(f"benchmark is missing {path.name}: {directory}")

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))

    if benchmark["benchmark_id"] != benchmark_id:
        raise ValueError(
            f"{directory}: benchmark_id {benchmark['benchmark_id']} != {benchmark_id}"
        )
    if profile["collection_name"] != benchmark_id:
        raise ValueError(
            f"{directory}: collection_name {profile['collection_name']} != benchmark_id"
        )

    queries = directory / "queries.jsonl"
    documents = directory / "documents.jsonl"
    qrels = directory / "qrels.tsv"
    leakage = directory / "leakage-report.json"

    leakage_report = json.loads(leakage.read_text(encoding="utf-8"))
    exclusions = leakage_report.get("exclusions", [])

    entry: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "directory": str(directory.relative_to(ROOT)),
        "queries_sha256": _sha(queries),
        "query_count": _rows(queries),
        "documents_sha256": _sha(documents),
        "document_count": _rows(documents),
        "qrels_sha256": _sha(qrels),
        "qrel_count": _qrel_rows(qrels),
        "leakage_report_sha256": _sha(leakage),
        "excluded_for_leakage": len(exclusions),
        "collection_name": profile["collection_name"],
        "collection_root": profile["collection_root"],
        "profile": {
            "profile_id": profile["profile_id"],
            "embedding_model": profile["embedding_model"],
            "reranker_model": profile["reranker_model"],
            "result_limit": profile["result_limit"],
            "per_list_limit": profile["per_list_limit"],
            "candidate_limit": profile["candidate_limit"],
            "rerank": profile["rerank"],
            "auto_expand": profile["auto_expand"],
            "strong_signal_bypass": profile["strong_signal_bypass"],
        },
        "profile_sha256": _sha(profile_path),
        "benchmark_sha256": _sha(benchmark_path),
    }

    if index_manifest_path.is_file():
        index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        if index_manifest["collection_name"] != benchmark_id:
            raise ValueError(
                f"{directory}: index-manifest collection mismatch"
            )
        if index_manifest["pending_embedding_count"] != 0:
            raise ValueError(
                f"{directory}: index has {index_manifest['pending_embedding_count']} "
                "pending embeddings; freeze requires a complete index"
            )
        if index_manifest["vector_document_count"] != index_manifest["document_count"]:
            raise ValueError(f"{directory}: vector/document count mismatch")
        entry["index"] = {
            "index_manifest_sha256": _sha(index_manifest_path),
            "index_fingerprint": index_manifest["index_fingerprint"],
            "embedding_fingerprint": index_manifest["embedding_fingerprint"],
            "document_count": index_manifest["document_count"],
            "vector_document_count": index_manifest["vector_document_count"],
            "vector_chunk_count": index_manifest["vector_chunk_count"],
            "pending_embedding_count": index_manifest["pending_embedding_count"],
        }
    else:
        entry["index"] = None

    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_ROOT / "evaluation-manifest.json",
        help="Where to write the frozen manifest",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"evaluation manifest already exists, refusing to overwrite: {output}")

    formal = []
    diagnostic = []
    for benchmark_id, role, directory, _needs_index in SCOPES:
        entry = load_benchmark(benchmark_id, directory)
        if entry["index"] is None:
            raise ValueError(
                f"{benchmark_id}: index-manifest.json is missing; "
                "run `qmd bench <dir> --check-index` before freezing"
            )
        (formal if role == FORMAL else diagnostic).append(entry)

    if len(formal) != 4:
        raise ValueError(f"expected 4 formal benchmarks, found {len(formal)}")

    profile_ids = {entry["profile"]["profile_id"] for entry in formal + diagnostic}
    embed_models = {entry["profile"]["embedding_model"] for entry in formal + diagnostic}
    rerank_models = {entry["profile"]["reranker_model"] for entry in formal + diagnostic}
    if len(embed_models) != 1 or len(rerank_models) != 1:
        raise ValueError("benchmarks do not share one embedding/reranker model")

    manifest = {
        "schema_version": "qmd-evaluation-manifest-v1",
        "evaluation_version": EVAL_VERSION,
        "status": "sealed",
        "training_release": TRAINING_RELEASE,
        "prompt": {
            "version": PROMPT_VERSION,
            "sha256": prompt_sha256(),
            "chat_template_tokenizer_revision": TOKENIZER_REVISION,
        },
        "generation": GENERATION,
        "metrics": METRICS,
        "models": {
            "embedding_model": next(iter(embed_models)),
            "reranker_model": next(iter(rerank_models)),
        },
        "scoring": {
            "implementation": "src/bench/bench.ts",
            "git_commit": _git_commit(ROOT),
            "git_dirty": bool(_git_dirty(ROOT)),
        },
        "benchmarks": {
            "formal": formal,
            "diagnostic": diagnostic,
        },
        "notes": [
            "raw/base/SFT share one query set, collection isolation, index "
            "fingerprint, retrieval profile, reranker and scoring implementation. "
            "The only permitted difference on the SFT arm is the loaded adapter.",
            "Distillation sources (fiqa-train, cqadup-programmers, cqadup-unix) are "
            "consumed training material and must never be used as independent test sets.",
            "Oracle selection metrics from public-distill-v0 validate label production "
            "only; they are not model gains and are not part of this manifest.",
        ],
    }

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(output, manifest)
    print(f"Wrote {output}")
    print(f"evaluation-manifest.json sha256: {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
