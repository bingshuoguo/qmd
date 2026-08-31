from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NORMALIZATION = "NFKC-casefold-trim-collapse-whitespace"
TOKEN_JACCARD = 0.8
TRIGRAM_JACCARD = 0.85


def normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())


def _terms(value: str) -> set[str]:
    return set(value.split())


def _trigrams(value: str) -> set[str]:
    if len(value) < 3:
        return {value} if value else set()
    return {value[index : index + 3] for index in range(len(value) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


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
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root), key=lambda item: item.encode())
        self.parent[high] = low


def build_families(
    queries: dict[str, str], qrels: list[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, Any]]:
    uf = UnionFind(queries)
    qids_by_doc: dict[str, list[str]] = defaultdict(list)
    for qid, doc_id in qrels:
        if qid not in queries:
            raise ValueError(f"qrel references missing query {qid}")
        qids_by_doc[doc_id].append(qid)
    for qids in qids_by_doc.values():
        for qid in qids[1:]:
            uf.union(qids[0], qid)

    prepared: list[tuple[str, str, set[str], set[str]]] = []
    exact_owner: dict[str, str] = {}
    for qid in sorted(queries, key=lambda item: item.encode()):
        normalized = normalize_query(queries[qid])
        owner = exact_owner.setdefault(normalized, qid)
        uf.union(owner, qid)
        prepared.append((qid, normalized, _terms(normalized), _trigrams(normalized)))

    near_pairs: list[dict[str, Any]] = []
    for index, (left_qid, left_norm, left_terms, left_grams) in enumerate(prepared):
        for right_qid, right_norm, right_terms, right_grams in prepared[index + 1 :]:
            if left_norm == right_norm:
                continue
            token_score = _jaccard(left_terms, right_terms)
            trigram_score = _jaccard(left_grams, right_grams)
            if token_score >= TOKEN_JACCARD or trigram_score >= TRIGRAM_JACCARD:
                uf.union(left_qid, right_qid)
                near_pairs.append(
                    {
                        "left_qid": left_qid,
                        "right_qid": right_qid,
                        "token_jaccard": round(token_score, 12),
                        "character_3gram_jaccard": round(trigram_score, 12),
                    }
                )

    groups: dict[str, list[str]] = defaultdict(list)
    for qid in queries:
        groups[uf.find(qid)].append(qid)
    family_by_qid: dict[str, str] = {}
    for qids in groups.values():
        ordered = sorted(qids, key=lambda item: item.encode())
        family_id = hashlib.sha256(
            ("scifact-family-v1\0" + "\0".join(ordered)).encode()
        ).hexdigest()
        for qid in ordered:
            family_by_qid[qid] = family_id
    return family_by_qid, {
        "normalization": NORMALIZATION,
        "thresholds": {
            "token_jaccard": TOKEN_JACCARD,
            "character_3gram_jaccard": TRIGRAM_JACCARD,
        },
        "near_duplicate_pairs": near_pairs,
        "family_count": len(groups),
    }


def select_families(
    family_by_qid: dict[str, str],
    sft_usage: dict[str, list[str]],
    *,
    target: int,
    seed: int,
) -> list[str]:
    qids_by_family: dict[str, list[str]] = defaultdict(list)
    for qid, family_id in family_by_qid.items():
        qids_by_family[family_id].append(qid)
    ordered = sorted(
        qids_by_family.items(),
        key=lambda item: (
            any(qid in sft_usage for qid in item[1]),
            hashlib.sha256(f"{seed}\0{item[0]}".encode()).digest(),
        ),
    )
    selected: list[str] = []
    for _, family_qids in ordered:
        if selected and abs(len(selected) - target) < abs(len(selected) + len(family_qids) - target):
            break
        selected.extend(sorted(family_qids, key=lambda item: item.encode()))
    return selected


def historical_sft_usage(release_root: Path) -> dict[str, list[str]]:
    usage: dict[str, set[str]] = defaultdict(set)
    for manifest_path in release_root.glob("**/release-manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        release_id = manifest.get("release_id")
        if not isinstance(release_id, str):
            continue
        root = manifest_path.parents[2]
        for entry in manifest.get("core_artifacts", {}).values():
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            path = root / entry["path"]
            if not path.is_file() or path.suffix != ".jsonl":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                query = record.get("query")
                if isinstance(query, str):
                    usage[normalize_query(query)].add(release_id)
    return {query: sorted(releases) for query, releases in usage.items()}


def audit_benchmark_rows(
    queries: list[dict[str, Any]], documents: list[dict[str, Any]], qrels: list[tuple[str, str]]
) -> dict[str, int]:
    qids = [str(row["qid"]) for row in queries]
    doc_ids = [str(row["doc_id"]) for row in documents]
    if len(qids) != len(set(qids)):
        raise ValueError("benchmark has duplicate qids")
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("benchmark has duplicate document ids")
    qid_set, doc_set = set(qids), set(doc_ids)
    for qid, doc_id in qrels:
        if qid not in qid_set:
            raise ValueError(f"orphan qrel query {qid}")
        if doc_id not in doc_set:
            raise ValueError(f"orphan qrel document {doc_id}")
    return {"queries": len(qids), "documents": len(doc_ids), "qrels": len(qrels)}
