#!/usr/bin/env python3
"""Contract validation and smoke-only materialization for public distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dataset.contract import CONTRACT_VERSION, validate_training_target
from dataset.scifact_distill import load_tokenizer, render_sft_record

TOKENIZER_MODEL = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())


def load_manifest(run_dir: Path) -> dict[str, Any]:
    value = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("run-manifest.json must contain an object")
    return value


def validate(run_dir: Path, local_files_only: bool) -> None:
    candidates_path = run_dir / "candidates.jsonl"
    output_path = run_dir / "validated.jsonl"
    if output_path.exists():
        raise ValueError(f"validated artifact already exists: {output_path}")
    manifest = load_manifest(run_dir)
    if manifest.get("candidates_sha256") != sha256_file(candidates_path):
        raise ValueError("candidates.jsonl hash does not match run manifest")
    tokenizer = load_tokenizer(TOKENIZER_MODEL, TOKENIZER_REVISION, local_files_only)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    records = read_jsonl(candidates_path)
    seen: set[str] = set()
    for record in records:
        input_id = record.get("input_id")
        query = record.get("query")
        candidates = record.get("candidates")
        if not isinstance(input_id, str) or input_id in seen:
            raise ValueError(f"invalid or duplicate input_id: {input_id!r}")
        seen.add(input_id)
        if not isinstance(query, str) or not query:
            raise ValueError(f"{input_id}: query must be non-empty")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"{input_id}: exactly four candidates are required")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or candidate.get("candidate_index") != index:
                raise ValueError(f"{input_id}: candidate indexes must be contiguous")
            parsed = candidate.get("parsed_output")
            if not isinstance(parsed, list):
                raise ValueError(f"{input_id} candidate {index}: parsed_output must be a list")
            result = validate_training_target({"query": query, "output": parsed}, counter)
            canonical = result.canonical_record["output"]
            candidate["contract"] = {
                "version": CONTRACT_VERSION,
                "valid": result.valid,
                "errors": [asdict(item) for item in result.errors],
                "warnings": [asdict(item) for item in result.warnings],
                "canonical_output": canonical,
            }
            repeat_only = bool(canonical) and all(
                normalized(item[1]) == normalized(query) for item in canonical
            )
            candidate["repeat_check"] = {
                "version": "nfkc-casefold-whitespace-v1",
                "repeat_only": repeat_only,
                "valid": result.valid and not repeat_only,
            }
            candidate["retrieval"] = None
    atomic_jsonl(output_path, records)
    manifest["contract"] = {
        "version": CONTRACT_VERSION,
        "tokenizer_model": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    manifest["validated_sha256"] = sha256_file(output_path)
    manifest["contract_valid_candidates"] = sum(
        candidate["contract"]["valid"]
        for record in records
        for candidate in record["candidates"]
    )
    manifest["repeat_only_candidates"] = sum(
        candidate["repeat_check"]["repeat_only"]
        for record in records
        for candidate in record["candidates"]
    )
    atomic_json(run_dir / "run-manifest.json", manifest)


def materialize(run_dir: Path, local_files_only: bool) -> None:
    selected_path = run_dir / "selected.jsonl"
    output_path = run_dir / "sft-smoke.jsonl"
    if output_path.exists():
        raise ValueError(f"materialized artifact already exists: {output_path}")
    manifest = load_manifest(run_dir)
    if manifest.get("selected_sha256") != sha256_file(selected_path):
        raise ValueError("selected.jsonl hash does not match run manifest")
    tokenizer = load_tokenizer(TOKENIZER_MODEL, TOKENIZER_REVISION, local_files_only)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    records = read_jsonl(selected_path)
    materialized: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    seen_queries: set[str] = set()
    for record in records:
        status = record.get("selection_status")
        if status not in {"winner", "qualified_tie"}:
            continue
        input_id = record.get("input_id")
        query = record.get("query")
        selected_index = record.get("selected_candidate_index")
        if not isinstance(input_id, str) or not isinstance(query, str):
            raise ValueError("selected record is missing input_id/query")
        if input_id in seen_inputs or normalized(query) in seen_queries:
            raise ValueError(f"duplicate accepted smoke record: {input_id}")
        seen_inputs.add(input_id)
        seen_queries.add(normalized(query))
        if not isinstance(selected_index, int):
            raise ValueError(f"{input_id}: accepted record has no selected candidate")
        candidate = record["candidates"][selected_index]
        contract = candidate.get("contract")
        repeat_check = candidate.get("repeat_check")
        if not isinstance(contract, dict) or contract.get("valid") is not True:
            raise ValueError(f"{input_id}: selected candidate is not Contract-valid")
        if not isinstance(repeat_check, dict) or repeat_check.get("valid") is not True:
            raise ValueError(f"{input_id}: selected candidate failed repeat-only check")
        canonical = contract.get("canonical_output")
        validation = validate_training_target({"query": query, "output": canonical}, counter)
        if not validation.valid or validation.canonical_record["output"] != canonical:
            raise ValueError(f"{input_id}: selected candidate failed revalidation")
        rendered = render_sft_record(record["qid"], query, canonical, tokenizer)
        rendered.update(
            {
                "schema_version": "qmd-public-distill-v0",
                "input_id": input_id,
                "source_id": record["source_id"],
                "selection_label": status,
                "selected_candidate_index": selected_index,
                "smoke_only": True,
            }
        )
        materialized.append(rendered)
    atomic_jsonl(output_path, materialized)
    manifest["materialized_count"] = len(materialized)
    manifest["materialized_smoke_only"] = True
    manifest["sft_smoke_sha256"] = sha256_file(output_path)
    atomic_json(run_dir / "run-manifest.json", manifest)
    report_path = run_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["materialized_count"] = len(materialized)
        report["sft_smoke_sha256"] = manifest["sft_smoke_sha256"]
        atomic_json(report_path, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "materialize"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.command == "validate":
        validate(run_dir, args.local_files_only)
    else:
        materialize(run_dir, args.local_files_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
