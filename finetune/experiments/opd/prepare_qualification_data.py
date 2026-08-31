from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import yaml

from .contracts import (
    PROMPT_SHA256,
    PROMPT_VERSION,
    QualificationConfig,
    atomic_json,
    atomic_jsonl,
    load_config,
    sha256_path,
)
from .data_audit import (
    audit_benchmark_rows,
    build_families,
    historical_sft_usage,
    normalize_query,
    select_families,
)


CANONICAL_FILES = (
    "benchmark.yaml",
    "documents.jsonl",
    "excluded-qids.json",
    "leakage-report.json",
    "qrels.tsv",
    "queries.jsonl",
    "source-manifest.json",
    "source-qrels.tsv",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _qrels(path: Path) -> list[tuple[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = 1 if lines and lines[0].lower().startswith("query-id") else 0
    rows: list[tuple[str, str]] = []
    for line in lines[start:]:
        fields = line.split("\t")
        if len(fields) < 3:
            raise ValueError(f"invalid qrel line in {path}: {line!r}")
        if float(fields[2]) >= 1:
            rows.append((fields[0], fields[1]))
    return rows


def _hash_bytes(value: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, value).hexdigest()


def _copy_existing_source(source_id: str, source: Path, output: Path, config: QualificationConfig) -> dict[str, Any]:
    benchmark_output = output / "benchmarks" / source_id
    corpus_output = output / "corpora" / source_id
    benchmark_output.mkdir(parents=True)
    for name in CANONICAL_FILES:
        path = source / name
        if not path.is_file():
            raise ValueError(f"{source_id}: missing canonical file {path}")
        shutil.copy2(path, benchmark_output / name)

    profile = yaml.safe_load((source / "retrieval-profile.yaml").read_text(encoding="utf-8"))
    collection_root = (source / str(profile["collection_root"])).resolve()
    if not collection_root.is_dir():
        raise ValueError(f"{source_id}: collection root is missing: {collection_root}")
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.symlink_to(
        os.path.relpath(collection_root, corpus_output.parent), target_is_directory=True
    )
    profile.update(
        {
            "profile_id": "qmd-opd-teacher-qualification-v1",
            "collection_root": f"../../corpora/{source_id}",
            "embedding_model": config.retrieval.embedding_model,
            "reranker_model": None,
            "result_limit": config.retrieval.result_limit,
            "per_list_limit": config.retrieval.per_list_limit,
            "candidate_limit": config.retrieval.candidate_limit,
            "rerank": False,
            "auto_expand": False,
            "strong_signal_bypass": False,
        }
    )
    (benchmark_output / "retrieval-profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    queries = _jsonl(benchmark_output / "queries.jsonl")
    documents = _jsonl(benchmark_output / "documents.jsonl")
    qrels = _qrels(benchmark_output / "qrels.tsv")
    counts = audit_benchmark_rows(queries, documents, qrels)
    return {
        "source_id": source_id,
        "benchmark_id": yaml.safe_load((benchmark_output / "benchmark.yaml").read_text(encoding="utf-8"))["benchmark_id"],
        "source_path": str(source),
        "source_sha256": _directory_manifest_hash(source, CANONICAL_FILES),
        "counts": counts,
        "queries": queries,
        "family_by_qid": {str(row["qid"]): f"{source_id}:{row['qid']}" for row in queries},
    }


def _directory_manifest_hash(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        digest.update(f"{name}\0{sha256_path(path)}\n".encode())
    return digest.hexdigest()


def _load_scifact(archive: Path) -> tuple[dict[str, str], dict[str, dict[str, str]], list[tuple[str, str]], bytes]:
    with zipfile.ZipFile(archive) as handle:
        queries = {
            str(row["_id"]): str(row["text"])
            for row in (json.loads(line) for line in handle.read("scifact/queries.jsonl").decode().splitlines() if line)
        }
        corpus = {
            str(row["_id"]): {"title": str(row.get("title", "")), "text": str(row.get("text", ""))}
            for row in (json.loads(line) for line in handle.read("scifact/corpus.jsonl").decode().splitlines() if line)
        }
        qrels_bytes = handle.read("scifact/qrels/train.tsv")
    qrels_lines = qrels_bytes.decode().splitlines()[1:]
    qrels = [(fields[0], fields[1]) for line in qrels_lines if line.strip() for fields in [line.split("\t")] if float(fields[2]) >= 1]
    return queries, corpus, qrels, qrels_bytes


def _write_scifact(config: QualificationConfig, output: Path, usage: dict[str, list[str]]) -> dict[str, Any]:
    archive = config.sources["scifact_archive"]
    queries, corpus, all_qrels, source_qrels = _load_scifact(archive)
    family_by_qid, family_audit = build_families(queries, all_qrels)
    seen_by_qid = {
        qid: usage[normalize_query(query)]
        for qid, query in queries.items()
        if normalize_query(query) in usage
    }
    selected = select_families(family_by_qid, seen_by_qid, target=200, seed=42)
    selected_set = set(selected)
    qrels = [(qid, doc_id) for qid, doc_id in all_qrels if qid in selected_set]
    relevant_docs = sorted({doc_id for _, doc_id in qrels}, key=lambda item: item.encode())
    benchmark = output / "benchmarks/scifact"
    corpus_root = output / "corpora/scifact"
    benchmark.mkdir(parents=True)
    corpus_root.mkdir(parents=True)
    query_rows = [{"qid": qid, "query": queries[qid]} for qid in selected]
    document_rows: list[dict[str, str]] = []
    for doc_id in relevant_docs:
        document = corpus[doc_id]
        filename = f"{doc_id}.md"
        body = f"# {document['title']}\n\n{document['text']}\n" if document["title"] else f"{document['text']}\n"
        (corpus_root / filename).write_text(body, encoding="utf-8")
        document_rows.append({"doc_id": doc_id, "path": filename})
    qrels_text = "query-id\tcorpus-id\tscore\n" + "".join(f"{qid}\t{doc_id}\t1\n" for qid, doc_id in qrels)
    atomic_jsonl(benchmark / "queries.jsonl", query_rows)
    atomic_jsonl(benchmark / "documents.jsonl", document_rows)
    (benchmark / "qrels.tsv").write_text(qrels_text, encoding="utf-8")
    (benchmark / "source-qrels.tsv").write_bytes(source_qrels)
    atomic_json(benchmark / "excluded-qids.json", sorted(set(queries) - selected_set, key=lambda item: item.encode()))
    leakage = {
        "normalization": family_audit["normalization"],
        "thresholds": family_audit["thresholds"],
        "family_count": family_audit["family_count"],
        "near_duplicate_pairs": family_audit["near_duplicate_pairs"],
        "selection_seed": 42,
        "target_queries": 200,
        "actual_queries": len(selected),
    }
    atomic_json(benchmark / "leakage-report.json", leakage)
    artifact_hashes = {
        "queries_sha256": sha256_path(benchmark / "queries.jsonl"),
        "qrels_sha256": sha256_path(benchmark / "qrels.tsv"),
        "documents_sha256": sha256_path(benchmark / "documents.jsonl"),
    }
    benchmark_manifest = {
        "benchmark_id": "qmd-opd-teacher-qualification-scifact-dev-v1",
        "source": {"url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip", "archive_md5": _hash_bytes(archive.read_bytes(), "md5"), "split": "train-derived-dev-seed-42"},
        "source_qrels_sha256": sha256_path(benchmark / "source-qrels.tsv"),
        "excluded_qids_sha256": sha256_path(benchmark / "excluded-qids.json"),
        "leakage_report_sha256": sha256_path(benchmark / "leakage-report.json"),
        "converted_data_sha256": _hash_bytes(json.dumps(artifact_hashes, sort_keys=True).encode()),
        "qrels": {"relevant_threshold": 1, "unjudged": "nonrelevant", "graded": False},
        "cutoffs": [1, 3, 5, 10, 20, 30],
        "metrics": ["recall_at_cutoffs", "mrr_at_10", "ndcg_at_10"],
    }
    (benchmark / "benchmark.yaml").write_text(yaml.safe_dump(benchmark_manifest, sort_keys=False), encoding="utf-8")
    profile = {
        "profile_id": "qmd-opd-teacher-qualification-v1",
        "collection_name": benchmark_manifest["benchmark_id"],
        "collection_root": "../../corpora/scifact",
        "embedding_model": config.retrieval.embedding_model,
        "reranker_model": None,
        "result_limit": config.retrieval.result_limit,
        "per_list_limit": config.retrieval.per_list_limit,
        "candidate_limit": config.retrieval.candidate_limit,
        "rerank": False,
        "auto_expand": False,
        "strong_signal_bypass": False,
    }
    (benchmark / "retrieval-profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    source_manifest = {
        "source_id": "scifact",
        "archive": str(archive),
        "archive_sha256": sha256_path(archive),
        "selection": leakage,
        "artifact_hashes": artifact_hashes,
    }
    atomic_json(benchmark / "source-manifest.json", source_manifest)
    counts = audit_benchmark_rows(query_rows, document_rows, qrels)
    return {
        "source_id": "scifact",
        "benchmark_id": benchmark_manifest["benchmark_id"],
        "source_path": str(archive),
        "source_sha256": sha256_path(archive),
        "counts": counts,
        "queries": query_rows,
        "family_by_qid": {qid: family_by_qid[qid] for qid in selected},
        "seen_by_qid": seen_by_qid,
    }


def prepare_bundle(config: QualificationConfig, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    usage = historical_sft_usage(config.release_manifest.parents[3])
    sources = [
        _copy_existing_source("nfcorpus", config.sources["nfcorpus"], output_root, config),
        _copy_existing_source("fiqa", config.sources["fiqa"], output_root, config),
        _copy_existing_source("freshstack", config.sources["freshstack"], output_root, config),
        _write_scifact(config, output_root, usage),
    ]
    combined_queries: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for source in sources:
        source_id = source["source_id"]
        for row in source.pop("queries"):
            qid = str(row["qid"])
            releases = usage.get(normalize_query(str(row["query"])), [])
            combined_queries.append(
                {
                    "sample_key": f"{source_id}:{qid}",
                    "source_id": source_id,
                    "qid": qid,
                    "query": row["query"],
                    "family_id": source["family_by_qid"][qid],
                    "sft_usage": "sft_seen" if releases else "fresh",
                    "sft_release_ids": releases,
                    "split_role": "teacher_qualification_and_retrieval_dev",
                }
            )
        source.pop("family_by_qid", None)
        source.pop("seen_by_qid", None)
        source_audit[source_id] = source
    atomic_jsonl(output_root / "queries.jsonl", combined_queries)
    data_audit = {
        "schema_version": "qmd-teacher-qualification-data-audit-v1",
        "sources": source_audit,
        "total_queries": len(combined_queries),
        "source_scoped_keys_unique": len({row["sample_key"] for row in combined_queries}) == len(combined_queries),
        "opd_train_overlap": 0,
    }
    atomic_json(output_root / "data-audit.json", data_audit)
    manifest = {
        "schema_version": "qmd-teacher-qualification-evaluation-v1",
        "mode": config.mode,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "config": config.manifest(),
        "queries_sha256": sha256_path(output_root / "queries.jsonl"),
        "data_audit_sha256": sha256_path(output_root / "data-audit.json"),
    }
    atomic_json(output_root / "evaluation-manifest.json", manifest)
    return data_audit


def check_inputs(config: QualificationConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_id in ("nfcorpus", "fiqa", "freshstack"):
        root = config.sources[source_id]
        for name in CANONICAL_FILES:
            if not (root / name).is_file():
                raise ValueError(f"{source_id}: missing canonical file {root / name}")
        profile = yaml.safe_load((root / "retrieval-profile.yaml").read_text(encoding="utf-8"))
        collection_root = (root / str(profile["collection_root"])).resolve()
        if not collection_root.is_dir():
            raise ValueError(f"{source_id}: missing collection root {collection_root}")
        counts = audit_benchmark_rows(
            _jsonl(root / "queries.jsonl"),
            _jsonl(root / "documents.jsonl"),
            _qrels(root / "qrels.tsv"),
        )
        result[source_id] = {"counts": counts, "collection_root": str(collection_root)}
    usage = historical_sft_usage(config.release_manifest.parents[3])
    queries, corpus, qrels, _ = _load_scifact(config.sources["scifact_archive"])
    family_by_qid, family_audit = build_families(queries, qrels)
    seen_by_qid = {
        qid: usage[normalize_query(query)]
        for qid, query in queries.items()
        if normalize_query(query) in usage
    }
    selected = select_families(family_by_qid, seen_by_qid, target=200, seed=42)
    relevant_docs = {doc_id for qid, doc_id in qrels if qid in set(selected)}
    if not relevant_docs <= set(corpus):
        raise ValueError("SciFact selected qrels reference missing documents")
    result["scifact"] = {
        "counts": {"queries": len(selected), "documents": len(relevant_docs), "qrels": sum(qid in set(selected) for qid, _ in qrels)},
        "family_count": family_audit["family_count"],
        "sft_seen_queries": sum(qid in seen_by_qid for qid in selected),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze teacher qualification benchmark inputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, mode="dry_run")
    if args.check:
        print(json.dumps(check_inputs(config), indent=2, sort_keys=True))
        return 0
    output = args.output or config.output_root / "input"
    audit = prepare_bundle(config, output)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
