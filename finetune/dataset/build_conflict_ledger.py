#!/usr/bin/env python3
"""Build a deterministic, reviewable Contract v1 conflict ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from dataset.contract import CONTRACT_VERSION, validate_training_target
from dataset.validate_contract import (
    DEFAULT_TOKENIZER_MODEL,
    load_token_counter,
    resolve_paths,
    sha256_file,
)


LEDGER_VERSION = "conflict-ledger-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_label(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_ledger(
    paths: Iterable[Path],
    token_counter: Callable[[str], int],
    source_root: Path,
) -> dict[str, Any]:
    """Build the ledger value without file I/O side effects beyond reading."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_files: list[dict[str, str]] = []

    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        label = _source_label(path, source_root)
        source_files.append({"path": label, "sha256": sha256_file(path)})
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    key = f"invalid:{label}:{line_number}"
                    groups[key].append(
                        {
                            "source_path": label,
                            "line_number": line_number,
                            "raw_output": None,
                            "canonical_output": None,
                            "candidate_hash": None,
                            "query": None,
                            "intent": None,
                            "valid": False,
                            "quarantined": False,
                            "errors": ["TRAIN_INVALID_JSON"],
                        }
                    )
                    continue
                if not isinstance(record, dict):
                    key = f"invalid:{label}:{line_number}"
                    groups[key].append(
                        {
                            "source_path": label,
                            "line_number": line_number,
                            "raw_output": None,
                            "canonical_output": None,
                            "candidate_hash": None,
                            "query": None,
                            "intent": None,
                            "valid": False,
                            "quarantined": False,
                            "errors": ["TRAIN_INVALID_RECORD"],
                        }
                    )
                    continue

                result = validate_training_target(record, token_counter)
                key = result.input_key or f"invalid:{label}:{line_number}"
                canonical_output = result.canonical_record["output"]
                groups[key].append(
                    {
                        "source_path": label,
                        "line_number": line_number,
                        "raw_output": record.get("output"),
                        "canonical_output": canonical_output,
                        "candidate_hash": _sha256_json(canonical_output),
                        "query": result.canonical_record["query"],
                        "intent": result.canonical_record.get("intent"),
                        "valid": result.valid,
                        "quarantined": result.quarantined,
                        "errors": [item.code for item in result.errors],
                    }
                )

    entries: list[dict[str, Any]] = []
    for input_key in sorted(groups):
        rows = groups[input_key]
        first = rows[0]
        candidates_by_hash: dict[str, dict[str, Any]] = {}
        for row in rows:
            candidate_hash = row["candidate_hash"]
            if candidate_hash is None:
                continue
            candidate = candidates_by_hash.setdefault(
                candidate_hash,
                {
                    "candidate_hash": candidate_hash,
                    "canonical_output": row["canonical_output"],
                    "raw_variants": {},
                    "sources": [],
                    "valid": True,
                    "quarantined": False,
                    "errors": set(),
                },
            )
            raw_key = _canonical_json(row["raw_output"])
            candidate["raw_variants"].setdefault(raw_key, row["raw_output"])
            candidate["valid"] = candidate["valid"] and row["valid"]
            candidate["quarantined"] = candidate["quarantined"] or row["quarantined"]
            candidate["errors"].update(row["errors"])
            candidate["sources"].append(
                {
                    "path": row["source_path"],
                    "line": row["line_number"],
                }
            )

        candidates = []
        for candidate_hash in sorted(candidates_by_hash):
            candidate = candidates_by_hash[candidate_hash]
            candidates.append(
                {
                    "candidate_hash": candidate_hash,
                    "canonical_output": candidate["canonical_output"],
                    "valid": candidate["valid"],
                    "quarantined": candidate["quarantined"],
                    "errors": sorted(candidate["errors"]),
                    "raw_outputs": [
                        candidate["raw_variants"][key]
                        for key in sorted(candidate["raw_variants"])
                    ],
                    "sources": sorted(
                        candidate["sources"],
                        key=lambda source: (source["path"], source["line"]),
                    ),
                }
            )

        valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
        if any(row["quarantined"] for row in rows):
            classification = "quarantined_only_mode"
            decision = "exclude"
            reason = "Contract v1 approved only-mode quarantine"
        elif not valid_candidates:
            classification = "invalid"
            decision = "exclude"
            reason = "Contract v1 hard error"
        elif len(valid_candidates) != len(candidates) or len(candidates) > 1:
            classification = "target_conflict"
            decision = "unresolved"
            reason = None
        elif len(rows) == 1:
            classification = "unique"
            decision = "select"
            reason = "single valid target"
        elif len(candidates[0]["raw_outputs"]) == 1:
            classification = "identical"
            decision = "select"
            reason = "identical target repeated across sources"
        else:
            classification = "order_or_format_only"
            decision = "select"
            reason = "deterministic canonicalization produced one target"

        selected_hash = (
            candidates[0]["candidate_hash"]
            if decision == "select" and len(candidates) == 1
            else None
        )
        errors = sorted({code for row in rows for code in row["errors"]})
        entries.append(
            {
                "input_key": input_key if not input_key.startswith("invalid:") else None,
                "query": first["query"],
                "intent": first["intent"],
                "classification": classification,
                "errors": errors,
                "candidates": candidates,
                "decision": decision,
                "selected_candidate_hash": selected_hash,
                "replacement_output": None,
                "reason": reason,
                "decision_origin": "automatic" if decision != "unresolved" else None,
                "reviewer": None,
                "reviewed_at": None,
            }
        )

    classification_counts: dict[str, int] = {}
    for entry in entries:
        classification = entry["classification"]
        classification_counts[classification] = classification_counts.get(classification, 0) + 1
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
