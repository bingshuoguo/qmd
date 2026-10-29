#!/usr/bin/env python3
"""Prepare the isolated nf-freshstack-v1 retrieval/training release.

The network-facing inputs are pinned and content-addressed.  ``prepare --scan``
creates the FreshStack near-duplicate review file.  Formal ``prepare`` refuses
to publish splits until every candidate has an explicit boolean decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import unicodedata
import urllib.request
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as parquet


VERSION = "nf-freshstack-v1"
SEED = 42
NF_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
NF_MD5 = "a89dba18a62ef92f7d323ec890a0d38d"
HF_REVISION = "6faa3e4ff1c7d31824c32ee0a9dd580aba8bad11"
HF_BASE = f"https://huggingface.co/datasets/mteb/FreshStackRetrieval/resolve/{HF_REVISION}"
FRESH_FILES = {
    "corpus.parquet": (
        "corpus/test-00000-of-00001.parquet",
        "ae47de77103c99a82b554c8ac527b237eab304d046d447796bc1c1271464da6d",
    ),
    "queries.parquet": (
        "queries/test-00000-of-00001.parquet",
        "33c5f73319c67057f04c9b4776bb10cd3f9396c836239ee0f2ece5bfdfc513ee",
    ),
    "qrels.parquet": (
        "qrels/test-00000-of-00001.parquet",
        "5c4d418203a869959a2a4c5cf8a1625dd30d660709788f669aafb4c118a6b402",
    ),
}
EXPECTED = {
    "nfcorpus_docs": 3633,
    "nfcorpus_queries": {"train": 2590, "dev": 324, "test": 323},
    "freshstack_docs": 3804,
    "freshstack_queries": 672,
    "freshstack_qrels": 5445,
}
EMBEDDING_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf"
RERANKER_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def jsonl(values: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values)


def digest_bytes(data: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())


def byte_key(value: str) -> bytes:
    return value.encode("utf-8")


def safe_id(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).is_absolute() or any(c in value for c in "/\\\0"):
        raise ValueError(f"{label}: unsafe id {value!r}")
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def frozen_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"frozen artifact differs; refusing to overwrite: {path}")
        return
    atomic_text(path, content)


def download(url: str, path: Path, expected: str, algorithm: str = "sha256") -> None:
    if path.exists():
        actual = digest_file(path, algorithm)
        if actual != expected:
            raise ValueError(f"{path}: {algorithm} mismatch: expected {expected}, got {actual}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.download-{uuid.uuid4().hex}")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = digest_file(temporary, algorithm)
        if actual != expected:
            raise ValueError(f"{url}: {algorithm} mismatch: expected {expected}, got {actual}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_sources(root: Path, offline: bool) -> None:
    nf_path = root / "archives" / "nfcorpus.zip"
    fresh_root = root / "source" / "freshstack"
    if offline:
        required = [nf_path, *(fresh_root / name for name in FRESH_FILES)]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ValueError("offline mode is missing source files: " + ", ".join(missing))
    else:
        download(NF_URL, nf_path, NF_MD5, "md5")
        for local_name, (remote_name, expected) in FRESH_FILES.items():
            download(f"{HF_BASE}/{remote_name}", fresh_root / local_name, expected)
    if digest_file(nf_path, "md5") != NF_MD5:
        raise ValueError("NFCorpus archive MD5 mismatch")
    for local_name, (_, expected) in FRESH_FILES.items():
        if digest_file(fresh_root / local_name) != expected:
            raise ValueError(f"FreshStack source SHA-256 mismatch: {local_name}")


def read_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: expected object")
        result.append(value)
    return result


def zip_member(archive: zipfile.ZipFile, suffix: str) -> zipfile.ZipInfo:
    matches = [item for item in archive.infolist() if item.filename.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"NFCorpus: expected one member ending {suffix!r}, got {len(matches)}")
    return matches[0]


def load_nfcorpus(root: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    path = root / "archives" / "nfcorpus.zip"
    with zipfile.ZipFile(path) as archive:
        corpus_info = zip_member(archive, "/corpus.jsonl")
        query_info = zip_member(archive, "/queries.jsonl")
        corpus_raw = archive.read(corpus_info)
        query_raw = archive.read(query_info)
        corpus = []
        for value in read_jsonl_bytes(corpus_raw, corpus_info.filename):
            doc_id = safe_id(str(value.get("_id", "")), "NFCorpus document")
            title, text = value.get("title"), value.get("text")
            if not isinstance(title, str) or not isinstance(text, str):
                raise ValueError(f"NFCorpus document {doc_id}: title/text must be strings")
            corpus.append({"id": doc_id, "title": title, "text": text})
        query_map = {}
        for value in read_jsonl_bytes(query_raw, query_info.filename):
            qid, text = str(value.get("_id", "")), value.get("text")
            if not qid or not isinstance(text, str) or not text.strip() or qid in query_map:
                raise ValueError(f"NFCorpus query is invalid or duplicate: {qid!r}")
            query_map[qid] = text
        split_queries: dict[str, list[dict[str, str]]] = {}
        split_qrels: dict[str, list[dict[str, Any]]] = {}
        source_hashes = {"corpus": digest_bytes(corpus_raw), "queries": digest_bytes(query_raw)}
        doc_ids = {item["id"] for item in corpus}
        for split in ("train", "dev", "test"):
            info = zip_member(archive, f"/qrels/{split}.tsv")
            raw = archive.read(info)
            source_hashes[f"qrels_{split}"] = digest_bytes(raw)
            qrels = parse_tsv_qrels(raw.decode("utf-8"), query_map, doc_ids, f"NFCorpus {split}")
            qids = sorted({item["qid"] for item in qrels}, key=byte_key)
            split_queries[split] = [{"qid": qid, "query": query_map[qid]} for qid in qids]
            split_qrels[split] = qrels
    if len(corpus) != EXPECTED["nfcorpus_docs"]:
        raise ValueError(f"NFCorpus document count drift: {len(corpus)}")
    for split, expected in EXPECTED["nfcorpus_queries"].items():
        if len(split_queries[split]) != expected:
            raise ValueError(f"NFCorpus {split} query count drift: {len(split_queries[split])}")
    return sorted(corpus, key=lambda item: byte_key(item["id"])), split_queries, split_qrels, source_hashes


def parse_tsv_qrels(text: str, queries: dict[str, str], doc_ids: set[str], label: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    if not lines or lines[0] != "query-id\tcorpus-id\tscore":
        raise ValueError(f"{label}: invalid qrels header")
    result = set()
    for line_number, line in enumerate(lines[1:], 2):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{label}:{line_number}: expected 3 columns")
        qid, doc_id, score_text = fields
        if qid not in queries or doc_id not in doc_ids:
            raise ValueError(f"{label}:{line_number}: unknown qid/doc_id")
        if float(score_text) >= 1:
            result.add((qid, doc_id))
    return [{"qid": qid, "doc_id": doc_id, "relevance": 1} for qid, doc_id in sorted(result)]


def parquet_rows(path: Path) -> list[dict[str, Any]]:
    return parquet.read_table(path).to_pylist()


def load_freshstack(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]], dict[str, str]]:
    source = root / "source" / "freshstack"
    corpus = []
    for row in parquet_rows(source / "corpus.parquet"):
        doc_id = safe_id(str(row.get("id", "")), "FreshStack document")
        title, text = row.get("title"), row.get("text")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(f"FreshStack document {doc_id}: title/text must be strings")
        corpus.append({"id": doc_id, "title": title, "text": text})
    queries = []
    query_map = {}
    for row in parquet_rows(source / "queries.parquet"):
        qid, text = str(row.get("id", "")), row.get("text")
        if not qid or not isinstance(text, str) or not text.strip() or qid in query_map:
            raise ValueError(f"FreshStack query is invalid or duplicate: {qid!r}")
        query_map[qid] = text
        queries.append({"qid": qid, "query": text})
    doc_ids = {item["id"] for item in corpus}
    qrels_set = set()
    for row in parquet_rows(source / "qrels.parquet"):
        qid, doc_id, score = str(row.get("query-id", "")), str(row.get("corpus-id", "")), row.get("score")
        if qid not in query_map or doc_id not in doc_ids:
            raise ValueError(f"FreshStack qrel has unknown qid/doc_id: {qid}/{doc_id}")
        if not isinstance(score, int) or score < 1:
            continue
        qrels_set.add((qid, doc_id))
    qrels = [{"qid": qid, "doc_id": doc_id, "relevance": 1} for qid, doc_id in sorted(qrels_set)]
    if len(corpus) != EXPECTED["freshstack_docs"] or len(queries) != EXPECTED["freshstack_queries"] or len(qrels) != EXPECTED["freshstack_qrels"]:
        raise ValueError(f"FreshStack count drift: corpus={len(corpus)}, queries={len(queries)}, qrels={len(qrels)}")
    hashes = {name: digest_file(source / name) for name in FRESH_FILES}
    return sorted(corpus, key=lambda item: byte_key(item["id"])), sorted(queries, key=lambda item: byte_key(item["qid"])), qrels, hashes


def terms(value: str) -> set[str]:
    return set(value.split())


def trigrams(value: str) -> set[str]:
    chars = list(value)
    if len(chars) < 3:
        return {value} if value else set()
    return {"".join(chars[index:index + 3]) for index in range(len(chars) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def near_duplicate_candidates(queries: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for item in queries:
        value = normalize(item["query"])
        normalized.append((item["qid"], item["query"], value, terms(value), trigrams(value)))
    result = []
    for left_index, (left_id, left_text, left_norm, left_terms, left_grams) in enumerate(normalized):
        for right_id, right_text, right_norm, right_terms, right_grams in normalized[left_index + 1:]:
            if left_norm == right_norm:
                continue
            token_score = jaccard(left_terms, right_terms)
            trigram_score = jaccard(left_grams, right_grams)
            if token_score >= 0.8 or trigram_score >= 0.85:
                result.append({
                    "left_qid": left_id,
                    "right_qid": right_id,
                    "left_query": left_text,
                    "right_query": right_text,
                    "token_jaccard": round(token_score, 12),
                    "character_3gram_jaccard": round(trigram_score, 12),
                    "human_confirmation": None,
                })
    return result


def scan(root: Path, offline: bool) -> dict[str, Any]:
    ensure_sources(root, offline)
    _, queries, _, hashes = load_freshstack(root)
    candidates = near_duplicate_candidates(queries)
    path = root / "audits" / "freshstack-near-duplicate-candidates.jsonl"
    generated_keys = [(row["left_qid"], row["right_qid"]) for row in candidates]
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        existing_keys = [(row.get("left_qid"), row.get("right_qid")) for row in existing]
        if existing_keys != generated_keys:
            raise ValueError("FreshStack near-duplicate decisions are stale for the pinned source")
        for expected, current in zip(candidates, existing, strict=True):
            decision = current.get("human_confirmation")
            if decision not in {True, False, None}:
                raise ValueError("human_confirmation must be true, false, or null")
            for key in expected:
                if key != "human_confirmation" and current.get(key) != expected[key]:
                    raise ValueError(f"near-duplicate candidate metadata changed: {expected['left_qid']}/{expected['right_qid']}")
    else:
        frozen_text(path, jsonl(candidates))
    return {"candidate_count": len(candidates), "review_file": str(path), "source_hashes": hashes}


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root), key=byte_key)
            self.parent[high] = low


def fresh_families(queries: list[dict[str, str]], qrels: list[dict[str, Any]], decisions_path: Path, corpus: list[dict[str, str]] | None = None) -> dict[str, str]:
    uf = UnionFind(item["qid"] for item in queries)
    qids_by_doc: dict[str, list[str]] = defaultdict(list)
    for qrel in qrels:
        qids_by_doc[qrel["doc_id"]].append(qrel["qid"])
    for qids in qids_by_doc.values():
        for qid in qids[1:]:
            uf.union(qids[0], qid)
    if corpus is not None:
        docs_by_hash: dict[str, list[str]] = defaultdict(list)
        for document in corpus:
            docs_by_hash[digest_bytes(document["text"].encode())].append(document["id"])
        for doc_ids in docs_by_hash.values():
            qids = sorted({qid for doc_id in doc_ids for qid in qids_by_doc.get(doc_id, [])}, key=byte_key)
            for qid in qids[1:]:
                uf.union(qids[0], qid)
    owner_by_norm: dict[str, str] = {}
    for query in queries:
        normalized = normalize(query["query"])
        if normalized in owner_by_norm:
            uf.union(owner_by_norm[normalized], query["qid"])
        else:
            owner_by_norm[normalized] = query["qid"]
    decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line]
    pending = [item for item in decisions if item.get("human_confirmation") is None]
    if pending:
        raise ValueError(f"FreshStack near-duplicate review incomplete: {len(pending)} candidates pending")
    for item in decisions:
        if not isinstance(item.get("human_confirmation"), bool):
            raise ValueError("FreshStack decisions must be booleans")
        if item["human_confirmation"]:
            uf.union(item["left_qid"], item["right_qid"])
    groups: dict[str, list[str]] = defaultdict(list)
    for query in queries:
        groups[uf.find(query["qid"])].append(query["qid"])
    family_by_qid = {}
    for qids in groups.values():
        sorted_qids = sorted(qids, key=byte_key)
        family_id = hashlib.sha256((VERSION + "\0family\0" + "\0".join(sorted_qids)).encode()).hexdigest()
        for qid in sorted_qids:
            family_by_qid[qid] = family_id
    return family_by_qid


def split_families(family_by_qid: dict[str, str]) -> dict[str, str]:
    qids_by_family: dict[str, list[str]] = defaultdict(list)
    for qid, family in family_by_qid.items():
        qids_by_family[family].append(qid)
    families = sorted(qids_by_family.items(), key=lambda item: (
        -len(item[1]),
        hashlib.sha256(
            (f"{VERSION}\0seed={SEED}\0" + "\0".join(sorted(item[1], key=byte_key))).encode()
        ).hexdigest(),
    ))
    total = len(family_by_qid)
    targets = {"train": total * 0.70, "dev": total * 0.15, "test": total * 0.15}
    counts = {split: 0 for split in targets}
    result = {}
    order = {"train": 0, "dev": 1, "test": 2}
    for family, qids in families:
        size = len(qids)
        split = max(targets, key=lambda name: (targets[name] - counts[name], -order[name]))
        counts[split] += size
        for qid in qids:
            result[qid] = split
    return result


def fence_for(text: str) -> str:
    run = maximum = 0
    for char in text:
        if char == "`":
            run += 1
            maximum = max(maximum, run)
        else:
            run = 0
    return "`" * max(3, maximum + 1)


def render_corpus(root: Path, dataset: str, corpus: list[dict[str, str]]) -> list[dict[str, str]]:
    output = root / "corpus" / dataset
    output.mkdir(parents=True, exist_ok=True)
    documents = []
    for document in corpus:
        doc_id = safe_id(document["id"], f"{dataset} document")
        relative = f"{doc_id}.md"
        if dataset == "nfcorpus":
            title = f"# {document['title']}\n\n" if document["title"].strip() else ""
            content = title + document["text"].replace("\r\n", "\n").replace("\r", "\n") + "\n"
        else:
            text = document["text"].replace("\r\n", "\n").replace("\r", "\n")
            fence = fence_for(text)
            content = f"# {doc_id}\n\n{fence}\n{text}\n{fence}\n"
        frozen_text(output / relative, content)
        documents.append({"doc_id": doc_id, "path": relative})
    return documents


def resolve_query_collisions(splits: dict[str, list[dict[str, str]]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    priority = {"train": 0, "dev": 1, "test": 2}
    owners: dict[str, tuple[str, str]] = {}
    excluded: dict[str, list[dict[str, str]]] = {split: [] for split in splits}
    for split in ("test", "dev", "train"):
        for query in splits[split]:
            key = normalize(query["query"])
            if key in owners:
                owner_split, owner_qid = owners[key]
                if priority[owner_split] < priority[split]:
                    raise AssertionError("collision precedence invariant failed")
                excluded[split].append({"qid": query["qid"], "reason": f"normalized_query_collision:{owner_split}:{owner_qid}"})
            else:
                owners[key] = (split, query["qid"])
    excluded_ids = {split: {item["qid"] for item in rows} for split, rows in excluded.items()}
    return ({split: [query for query in rows if query["qid"] not in excluded_ids[split]] for split, rows in splits.items()}, excluded)


def existing_training_audit(root: Path, datasets: dict[str, dict[str, list[dict[str, str]]]]) -> tuple[dict[str, dict[str, set[str]]], dict[str, Any]]:
    training_files = sorted(root.parent.glob("public-distill-v*/experiments/*/sft.jsonl"), key=lambda path: byte_key(path.as_posix()))
    training = []
    file_records = []
    for path in training_files:
        count = 0
        display_path = path.relative_to(root.parent).as_posix()
        for row in read_jsonl(path):
            query = row.get("query") or row.get("seed_query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"existing SFT row has no query/seed_query: {path}")
            value = normalize(query)
            training.append((display_path, query, value, terms(value), trigrams(value)))
            count += 1
        file_records.append({"path": display_path, "sha256": digest_file(path), "query_count": count})
    exact_normalized = {item[2] for item in training}
    exact: list[dict[str, str]] = []
    near: list[dict[str, Any]] = []
    excluded = {dataset: {split: set() for split in splits} for dataset, splits in datasets.items()}
    for dataset, splits in datasets.items():
        for split, queries in splits.items():
            for query in queries:
                value = normalize(query["query"])
                if value in exact_normalized:
                    excluded[dataset][split].add(query["qid"])
                    exact.append({"dataset": dataset, "split": split, "qid": query["qid"], "query": query["query"]})
                    continue
                query_terms, query_grams = terms(value), trigrams(value)
                for path, training_query, _, training_terms, training_grams in training:
                    token_possible = min(len(query_terms), len(training_terms)) / max(len(query_terms), len(training_terms), 1) >= 0.8
                    trigram_possible = min(len(query_grams), len(training_grams)) / max(len(query_grams), len(training_grams), 1) >= 0.85
                    if not token_possible and not trigram_possible:
                        continue
                    token_score = jaccard(query_terms, training_terms) if token_possible else 0.0
                    trigram_score = jaccard(query_grams, training_grams) if trigram_possible else 0.0
                    if token_score >= 0.8 or trigram_score >= 0.85:
                        near.append({
                            "dataset": dataset,
                            "split": split,
                            "qid": query["qid"],
                            "query": query["query"],
                            "training_file": path,
                            "training_query": training_query,
                            "token_jaccard": round(token_score, 12),
                            "character_3gram_jaccard": round(trigram_score, 12),
                            "human_confirmation": None,
                        })
    report = {
        "version": VERSION,
        "training_files": file_records,
        "normalization": "NFKC-casefold-trim-collapse-whitespace",
        "thresholds": {"token_jaccard": 0.8, "character_3gram_jaccard": 0.85},
        "exact_matches": exact,
        "near_duplicate_candidates": near,
        "exact_exclusion_count": len(exact),
        "near_duplicate_candidate_count": len(near),
    }
    return excluded, report


def tree_hash(paths: Iterable[Path], base: Path) -> str:
    aggregate = hashlib.sha256()
    for path in sorted(paths, key=lambda item: byte_key(item.relative_to(base).as_posix())):
        relative = path.relative_to(base).as_posix()
        aggregate.update(f"{relative}\0{digest_file(path)}\n".encode())
    return aggregate.hexdigest()


def benchmark_id(dataset: str, split: str) -> str:
    return f"qmd-nf-freshstack-v1-{dataset}-{split}"


def benchmark_dir(root: Path, dataset: str, split: str) -> Path:
    return root / "splits" / dataset / benchmark_id(dataset, split)


def write_split(root: Path, dataset: str, split: str, queries: list[dict[str, str]], qrels: list[dict[str, Any]], documents: list[dict[str, str]], source: dict[str, Any], excluded: list[dict[str, str]], family_by_qid: dict[str, str] | None) -> None:
    output = benchmark_dir(root, dataset, split)
    output.mkdir(parents=True, exist_ok=True)
    qids = {query["qid"] for query in queries}
    final_qrels = [qrel for qrel in qrels if qrel["qid"] in qids]
    if {qrel["qid"] for qrel in final_qrels} != qids:
        raise ValueError(f"{dataset}/{split}: at least one query has no relevant qrel")
    query_bytes = jsonl(sorted(queries, key=lambda item: byte_key(item["qid"])))
    doc_bytes = jsonl(documents)
    excluded_ids = {row["qid"] for row in excluded}
    source_qids = qids | excluded_ids
    source_qrels = [qrel for qrel in qrels if qrel["qid"] in source_qids]
    source_qrel_bytes = "query-id\tcorpus-id\tscore\n" + "".join(
        f"{row['qid']}\t{row['doc_id']}\t1\n" for row in sorted(source_qrels, key=lambda row: (byte_key(row["qid"]), byte_key(row["doc_id"])))
    )
    qrel_bytes = "query-id\tcorpus-id\tscore\n" + "".join(
        f"{row['qid']}\t{row['doc_id']}\t1\n" for row in sorted(final_qrels, key=lambda row: (byte_key(row["qid"]), byte_key(row["doc_id"])))
    )
    excluded_text = "".join(f"{row['qid']}\n" for row in sorted(excluded, key=lambda row: byte_key(row["qid"])))
    excluded_json = stable_json(sorted(excluded_ids, key=byte_key))
    leakage = {
        "dataset": dataset,
        "split": split,
        "normalization": "NFKC-casefold-trim-collapse-whitespace",
        "precedence": "test > dev > train",
        "excluded": excluded,
        "family_count": len({family_by_qid[qid] for qid in qids}) if family_by_qid else None,
    }
    leakage_text = stable_json(leakage)
    converted = hashlib.sha256((digest_bytes(query_bytes.encode()) + digest_bytes(qrel_bytes.encode()) + digest_bytes(doc_bytes.encode())).encode()).hexdigest()
    current_benchmark_id = benchmark_id(dataset, split)
    collection = current_benchmark_id
    relative_corpus = f"../../../corpus/{dataset}"
    manifest = (
        f"benchmark_id: {current_benchmark_id}\n"
        "source:\n"
        f"  url: {source['url']}\n"
        f"  archive_md5: {source['archive_md5']}\n"
        f"  split: {split}\n"
        f"source_qrels_sha256: {digest_bytes(source_qrel_bytes.encode())}\n"
        f"excluded_qids_sha256: {digest_bytes(excluded_json.encode())}\n"
        f"leakage_report_sha256: {digest_bytes(leakage_text.encode())}\n"
        f"converted_data_sha256: {converted}\n"
        "qrels:\n  relevant_threshold: 1\n  unjudged: nonrelevant\n  graded: false\n"
        "cutoffs: [1, 3, 5, 10, 20, 30]\n"
        "metrics: [recall_at_cutoffs, mrr_at_10, ndcg_at_10]\n"
    )
    profile = (
        f"profile_id: {VERSION}\ncollection_name: {collection}\ncollection_root: {relative_corpus}\n"
        f"embedding_model: {EMBEDDING_MODEL}\nreranker_model: {RERANKER_MODEL}\n"
        "result_limit: 30\nper_list_limit: 30\ncandidate_limit: 40\nrerank: false\nauto_expand: false\nstrong_signal_bypass: false\n"
    )
    source_manifest = {**source, "dataset": dataset, "split": split, "query_count": len(queries), "qrel_count": len(final_qrels), "document_count": len(documents)}
    artifacts = {
        "queries.jsonl": query_bytes,
        "source-qrels.tsv": source_qrel_bytes,
        "qrels.tsv": qrel_bytes,
        "documents.jsonl": doc_bytes,
        "excluded-qids.json": excluded_json,
        "excluded-qids.txt": excluded_text,
        "leakage-report.json": leakage_text,
        "benchmark.yaml": manifest,
        "retrieval-profile.yaml": profile,
        "source-manifest.json": stable_json(source_manifest),
    }
    for name, content in artifacts.items():
        frozen_text(output / name, content)


def prepare(root: Path, offline: bool) -> dict[str, Any]:
    scan_result = scan(root, offline)
    decisions_path = root / "audits" / "freshstack-near-duplicate-candidates.jsonl"
    nf_corpus, nf_queries, nf_qrels, nf_hashes = load_nfcorpus(root)
    fresh_corpus, fresh_queries, fresh_qrels, fresh_hashes = load_freshstack(root)
    families = fresh_families(fresh_queries, fresh_qrels, decisions_path, fresh_corpus)
    fresh_assignment = split_families(families)
    fresh_split_queries = {split: [query for query in fresh_queries if fresh_assignment[query["qid"]] == split] for split in ("train", "dev", "test")}
    nf_queries, nf_excluded = resolve_query_collisions(nf_queries)
    fresh_split_queries, fresh_excluded = resolve_query_collisions(fresh_split_queries)
    existing_excluded, existing_report = existing_training_audit(root, {"nfcorpus": nf_queries, "freshstack": fresh_split_queries})
    for dataset, splits in (("nfcorpus", nf_queries), ("freshstack", fresh_split_queries)):
        target_excluded = nf_excluded if dataset == "nfcorpus" else fresh_excluded
        for split, rows in splits.items():
            for qid in sorted(existing_excluded[dataset][split], key=byte_key):
                target_excluded[split].append({"qid": qid, "reason": "exact_collision_with_existing_sft"})
            blocked = existing_excluded[dataset][split]
            splits[split] = [row for row in rows if row["qid"] not in blocked]
    frozen_text(root / "audits" / "existing-training-leakage.json", stable_json(existing_report))
    nf_docs = render_corpus(root, "nfcorpus", nf_corpus)
    fresh_docs = render_corpus(root, "freshstack", fresh_corpus)
    nf_source = {"url": NF_URL, "archive_md5": NF_MD5, **nf_hashes, "license": "CC-BY-SA-4.0"}
    fresh_source_md5 = digest_bytes("\n".join(fresh_hashes[name] for name in sorted(fresh_hashes)).encode(), "md5")
    fresh_source = {"url": f"https://huggingface.co/datasets/mteb/FreshStackRetrieval/tree/{HF_REVISION}", "archive_md5": fresh_source_md5, **fresh_hashes, "revision": HF_REVISION, "license": "MIT"}
    for split in ("train", "dev", "test"):
        write_split(root, "nfcorpus", split, nf_queries[split], nf_qrels[split], nf_docs, nf_source, nf_excluded[split], None)
        write_split(root, "freshstack", split, fresh_split_queries[split], fresh_qrels, fresh_docs, fresh_source, fresh_excluded[split], families)
    split_manifest = {
        "version": VERSION,
        "seed": SEED,
        "freshstack_target_ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "freshstack_actual_counts": {split: len(fresh_split_queries[split]) for split in ("train", "dev", "test")},
        "freshstack_family_count": len(set(families.values())),
        "nfcorpus_counts": {split: len(nf_queries[split]) for split in ("train", "dev", "test")},
        "document_counts": {"nfcorpus": len(nf_docs), "freshstack": len(fresh_docs), "total": len(nf_docs) + len(fresh_docs)},
        "review_file_sha256": digest_file(decisions_path),
    }
    frozen_text(root / "splits" / "split-manifest.json", stable_json(split_manifest))
    pool_result = build_pool(root)
    audit_result = audit(root)
    return {"scan": scan_result, "splits": split_manifest, "pool": pool_result, "audit": audit_result}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_pool(root: Path) -> dict[str, Any]:
    fresh = read_jsonl(benchmark_dir(root, "freshstack", "train") / "queries.jsonl")
    nf = read_jsonl(benchmark_dir(root, "nfcorpus", "train") / "queries.jsonl")
    target = len(fresh)
    if len(nf) < target:
        raise ValueError(f"NFCorpus train has {len(nf)} queries but balanced pool needs {target}")
    def record(dataset: str, row: dict[str, str]) -> dict[str, str]:
        source_id = f"{dataset}-train"
        sample_key = hashlib.sha256(f"{VERSION}\0seed={SEED}\0{source_id}\0{row['qid']}".encode()).hexdigest()
        return {"input_id": f"{source_id}:{row['qid']}", "source_id": source_id, "qid": row["qid"], "query": row["query"], "sample_key": sample_key}
    fresh_records = sorted((record("freshstack", row) for row in fresh), key=lambda row: (row["sample_key"], byte_key(row["qid"])))
    nf_records = sorted((record("nfcorpus", row) for row in nf), key=lambda row: (row["sample_key"], byte_key(row["qid"])))[:target]
    pool = sorted([*fresh_records, *nf_records], key=lambda row: byte_key(row["input_id"]))
    pool_text = jsonl(pool)
    manifest = {
        "version": VERSION,
        "seed": SEED,
        "selection": "all eligible FreshStack train plus equal-size NFCorpus train sample by sample_key/qid",
        "source_counts": {"freshstack-train": len(fresh_records), "nfcorpus-train": len(nf_records)},
        "total_count": len(pool),
        "pool_sha256": digest_bytes(pool_text.encode()),
        "teacher_candidate_count": 4,
        "winner_metric_order": ["recall_at_30", "mrr_at_10", "ndcg_at_10", "lower_candidate_index"],
        "sft_release": "public-distill-v2",
        "completion_only": True,
        "teacher_label_validation_ratio": 0.10,
        "retrieval_dev_is_separate": True,
        "sources": {
            "freshstack-train": str(benchmark_dir(root, "freshstack", "train").relative_to(root)),
            "nfcorpus-train": str(benchmark_dir(root, "nfcorpus", "train").relative_to(root)),
        },
    }
    frozen_text(root / "prepared" / "train-pool.jsonl", pool_text)
    frozen_text(root / "prepared" / "train-pool-manifest.json", stable_json(manifest))
    return manifest


def audit(root: Path, write_report: bool = True) -> dict[str, Any]:
    manifest = json.loads((root / "splits" / "split-manifest.json").read_text(encoding="utf-8"))
    if manifest["document_counts"]["total"] != 7437:
        raise ValueError("combined corpus must contain 7,437 documents")
    report: dict[str, Any] = {"version": VERSION, "datasets": {}, "status": "passed"}
    for dataset in ("nfcorpus", "freshstack"):
        seen_ids, seen_queries = set(), set()
        family_sets: list[set[str]] = []
        fresh_family_by_qid: dict[str, str] | None = None
        if dataset == "freshstack":
            source_corpus, source_queries, source_qrels, _ = load_freshstack(root)
            fresh_family_by_qid = fresh_families(
                source_queries,
                source_qrels,
                root / "audits" / "freshstack-near-duplicate-candidates.jsonl",
                source_corpus,
            )
        dataset_report = {}
        for split in ("train", "dev", "test"):
            base = benchmark_dir(root, dataset, split)
            queries = read_jsonl(base / "queries.jsonl")
            documents = read_jsonl(base / "documents.jsonl")
            qrels = []
            for line in (base / "qrels.tsv").read_text(encoding="utf-8").splitlines()[1:]:
                qid, doc_id, score = line.split("\t")
                if score != "1":
                    raise ValueError(f"{dataset}/{split}: non-binary qrel")
                qrels.append((qid, doc_id))
            qids, doc_ids = {row["qid"] for row in queries}, {row["doc_id"] for row in documents}
            normalized = {normalize(row["query"]) for row in queries}
            if len(qids) != len(queries) or len(normalized) != len(queries):
                raise ValueError(f"{dataset}/{split}: duplicate qid or normalized query")
            if seen_ids & qids or seen_queries & normalized:
                raise ValueError(f"{dataset}: cross-split query leakage")
            seen_ids |= qids
            seen_queries |= normalized
            if any(qid not in qids or doc_id not in doc_ids for qid, doc_id in qrels):
                raise ValueError(f"{dataset}/{split}: dangling qrel")
            dataset_report[split] = {"queries": len(queries), "qrels": len(qrels), "documents": len(documents)}
            leakage = json.loads((base / "leakage-report.json").read_text(encoding="utf-8"))
            if dataset == "freshstack":
                assert fresh_family_by_qid is not None
                families = {fresh_family_by_qid[qid] for qid in qids}
                if any(families & previous for previous in family_sets):
                    raise ValueError("freshstack: family crosses split boundaries")
                family_sets.append(families)
                dataset_report[split]["families"] = leakage["family_count"]
        report["datasets"][dataset] = dataset_report
    pool = read_jsonl(root / "prepared" / "train-pool.jsonl")
    counts: dict[str, int] = defaultdict(int)
    for row in pool:
        counts[row["source_id"]] += 1
    if not counts or 0 in counts.values() or len(set(counts.values())) != 1:
        raise ValueError("training pool is not source-balanced")
    report["training_pool_counts"] = dict(sorted(counts.items()))
    report_path = root / "audits" / "release-audit.json"
    report["artifact_tree_sha256"] = tree_hash((
        path for path in root.rglob("*")
        if path.is_file()
        and "archives" not in path.parts
        and "source" not in path.parts
        and path != report_path
    ), root)
    if write_report:
        frozen_text(report_path, stable_json(report))
    return report


def index_commands(root: Path, check: bool) -> dict[str, Any]:
    db_path = root / "indexes" / "qmd" / f"{VERSION}.sqlite"
    if check:
        command = [
            "node", "--import", "tsx",
            "finetune/benchmarks/public-eval/verify_frozen_indexes.ts",
            "--db", str(db_path),
        ]
        for dataset in ("nfcorpus", "freshstack"):
            for split in ("train", "dev", "test"):
                command.extend(("--benchmark", str(benchmark_dir(root, dataset, split))))
        subprocess.run(command, check=True)
        return {"checked": True, "command": command}
    config_dir = root / "indexes" / "config"
    config = ["collections:"]
    for dataset in ("nfcorpus", "freshstack"):
        corpus = (root / "corpus" / dataset).resolve()
        for split in ("train", "dev", "test"):
            config.extend((
                f"  {benchmark_id(dataset, split)}:",
                f"    path: {corpus}",
                '    pattern: "**/*.md"',
            ))
    config.extend((
        "models:",
        f"  embed: {EMBEDDING_MODEL}",
        f"  rerank: {RERANKER_MODEL}",
    ))
    frozen_text(config_dir / "index.yml", "\n".join(config) + "\n")
    env = {
        **os.environ,
        "INDEX_PATH": str(db_path),
        "QMD_CONFIG_DIR": str(config_dir),
    }
    commands = []
    update = ["npm", "run", "qmd", "--", "update"]
    subprocess.run(update, check=True, env=env)
    commands.append(update)
    embed = ["npm", "run", "qmd", "--", "embed", "--timeout", "0"]
    subprocess.run(embed, check=True, env=env)
    commands.append(embed)
    for dataset in ("nfcorpus", "freshstack"):
        for split in ("train", "dev", "test"):
            benchmark = benchmark_dir(root, dataset, split)
            command = ["npm", "run", "qmd", "--", "bench", str(benchmark), "--check-index"]
            subprocess.run(command, check=True, env=env)
            commands.append(command)
    return {"checked": False, "commands": commands}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "audit", "pool", "index"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, default=Path("finetune/data/nf-freshstack-v1"))
        command.add_argument("--check", action="store_true")
        if name == "prepare":
            command.add_argument("--scan", action="store_true")
            command.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        if args.check:
            result = audit(root, write_report=False)
        elif args.scan:
            result = scan(root, args.offline)
        else:
            result = prepare(root, args.offline)
    elif args.command == "audit":
        result = audit(root, write_report=not args.check)
    elif args.command == "pool":
        result = audit(root, write_report=False) if args.check else build_pool(root)
    else:
        result = index_commands(root, args.check)
    print(stable_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
