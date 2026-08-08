#!/usr/bin/env python3
"""Build a deterministic, reviewable Contract v1 conflict ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from dataset.contract import CONTRACT_VERSION, validate_training_target
from dataset.validate_contract import (
    DEFAULT_TOKENIZER_MODEL,
    load_token_counter,
    resolve_paths,
    sha256_file,
    source_label,
)


LEDGER_VERSION = "conflict-ledger-v1"

# Rows that cannot carry a real input_key (bad JSON, non-dict record, empty
# query) get a synthetic per-line group key with this prefix instead.  The
# key is what is synthetic, not the row: an empty-query row still has a
# computed candidate_hash, so the prefix -- not candidate_hash -- marks these
# groups when entries are assembled.
_INVALID_KEY_PREFIX = "invalid:"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Row:
    """One parsed source line; the ledger's internal unit of work."""

    source_path: str
    line_number: int
    raw_output: Any
    canonical_output: Any
    candidate_hash: str | None
    query: str | None
    intent: str | None
    valid: bool
    quarantined: bool
    errors: tuple[str, ...]


def _invalid_row(source_path: str, line_number: int, code: str) -> _Row:
    return _Row(
        source_path=source_path,
        line_number=line_number,
        raw_output=None,
        canonical_output=None,
        candidate_hash=None,
        query=None,
        intent=None,
        valid=False,
        quarantined=False,
        errors=(code,),
    )


def _load_groups(
    paths: Iterable[Path],
    token_counter: Callable[[str], int],
    source_root: Path,
) -> tuple[dict[str, list[_Row]], list[dict[str, str]]]:
    """Read every source line once, validate it, and group rows by input_key."""
    groups: dict[str, list[_Row]] = defaultdict(list)
    source_files: list[dict[str, str]] = []

    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        label = source_label(path, source_root)
        source_files.append({"path": label, "sha256": sha256_file(path)})
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    key = f"{_INVALID_KEY_PREFIX}{label}:{line_number}"
                    groups[key].append(_invalid_row(label, line_number, "TRAIN_INVALID_JSON"))
                    continue
                if not isinstance(record, dict):
                    key = f"{_INVALID_KEY_PREFIX}{label}:{line_number}"
                    groups[key].append(_invalid_row(label, line_number, "TRAIN_INVALID_RECORD"))
                    continue

                result = validate_training_target(record, token_counter)
                key = result.input_key or f"{_INVALID_KEY_PREFIX}{label}:{line_number}"
                canonical_output = result.canonical_record["output"]
                groups[key].append(
                    _Row(
                        source_path=label,
                        line_number=line_number,
                        raw_output=record.get("output"),
                        canonical_output=canonical_output,
                        candidate_hash=_sha256_json(canonical_output),
                        query=result.canonical_record["query"],
                        intent=result.canonical_record.get("intent"),
                        valid=result.valid,
                        quarantined=result.quarantined,
                        errors=tuple(item.code for item in result.errors),
                    )
                )
    return groups, source_files


def _build_candidate(candidate_hash: str, rows: list[_Row]) -> dict[str, Any]:
    """Aggregate all rows that canonicalized to the same output."""
    raw_variants: dict[str, Any] = {}
    for row in rows:
        raw_variants.setdefault(_canonical_json(row.raw_output), row.raw_output)
    return {
        "candidate_hash": candidate_hash,
        "canonical_output": rows[0].canonical_output,
        "valid": all(row.valid for row in rows),
        "quarantined": any(row.quarantined for row in rows),
        "errors": sorted({code for row in rows for code in row.errors}),
        "raw_outputs": [raw_variants[key] for key in sorted(raw_variants)],
        "sources": sorted(
            ({"path": row.source_path, "line": row.line_number} for row in rows),
            key=lambda source: (source["path"], source["line"]),
        ),
    }


def _classify(
    rows: list[_Row],
    candidates: list[dict[str, Any]],
) -> tuple[str, str, str | None]:
    """Map one group's rows and candidates to (classification, decision, reason)."""
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if any(row.quarantined for row in rows):
        return "quarantined_only_mode", "exclude", "Contract v1 approved only-mode quarantine"
    if not valid_candidates:
        return "invalid", "exclude", "Contract v1 hard error"
    if len(valid_candidates) != len(candidates) or len(candidates) > 1:
        return "target_conflict", "unresolved", None
    # From here on there is exactly one candidate, and it is valid.
    if len(rows) == 1:
        return "unique", "select", "single valid target"
    if len(candidates[0]["raw_outputs"]) == 1:
        return "identical", "select", "identical target repeated across sources"
    return "order_or_format_only", "select", "deterministic canonicalization produced one target"


def _build_entry(input_key: str, rows: list[_Row]) -> dict[str, Any]:
    first = rows[0]
    rows_by_hash: dict[str, list[_Row]] = defaultdict(list)
    for row in rows:
        if row.candidate_hash is not None:
            rows_by_hash[row.candidate_hash].append(row)
    candidates = [
        _build_candidate(candidate_hash, rows_by_hash[candidate_hash])
        for candidate_hash in sorted(rows_by_hash)
    ]

    classification, decision, reason = _classify(rows, candidates)
    selected_hash = (
        candidates[0]["candidate_hash"]
        if decision == "select" and len(candidates) == 1
        else None
    )
    return {
        "input_key": input_key if not input_key.startswith(_INVALID_KEY_PREFIX) else None,
        "query": first.query,
        "intent": first.intent,
        "classification": classification,
        "errors": sorted({code for row in rows for code in row.errors}),
        "candidates": candidates,
        "decision": decision,
        "selected_candidate_hash": selected_hash,
        "replacement_output": None,
        "reason": reason,
        "decision_origin": "automatic" if decision != "unresolved" else None,
        "reviewer": None,
        "reviewed_at": None,
    }


def build_ledger(
    paths: Iterable[Path],
    token_counter: Callable[[str], int],
    source_root: Path,
) -> dict[str, Any]:
    """Build the ledger value without file I/O side effects beyond reading."""
    groups, source_files = _load_groups(paths, token_counter, source_root)
    entries = [_build_entry(input_key, groups[input_key]) for input_key in sorted(groups)]

    classification_counts = Counter(entry["classification"] for entry in entries)
    return {
        "ledger_version": LEDGER_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_files": source_files,
        "record_count": sum(len(rows) for rows in groups.values()),
        "entry_count": len(entries),
        "classification_counts": dict(sorted(classification_counts.items())),
        "unresolved_count": sum(entry["decision"] == "unresolved" for entry in entries),
        "entries": entries,
    }


def write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"ledger already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["finetune/data/*.jsonl"],
        help="JSONL files or glob patterns relative to the repository root",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    paths = resolve_paths(args.paths, repo_root)
    if not paths:
        print("No source files matched; no ledger was written.")
        return 2
    try:
        token_counter = load_token_counter(
            args.tokenizer_model,
            args.tokenizer_revision,
            args.local_files_only,
        )
        ledger = build_ledger(paths, token_counter, repo_root)
        ledger["tokenizer_model"] = args.tokenizer_model
        ledger["tokenizer_revision"] = args.tokenizer_revision
        ledger["validator_artifact_sha256"] = sha256_file(
            Path(__file__).with_name("contract.py")
        )
        write_ledger(args.output.resolve(), ledger)
    except Exception as error:
        print(f"Conflict ledger failed: {error}")
        return 2

    print(
        f"Conflict ledger wrote {ledger['entry_count']} entries to {args.output}; "
        f"{ledger['unresolved_count']} unresolved"
    )
    return 1 if ledger["unresolved_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
