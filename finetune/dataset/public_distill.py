#!/usr/bin/env python3
"""Contract validation and deterministic materialization for public distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dataset.contract import CONTRACT_VERSION, validate_training_target
from dataset.scifact_distill import load_tokenizer, render_sft_record

TOKENIZER_MODEL = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FORMAL_SOURCE_COUNTS = {
    "fiqa-train": 750,
    "cqadup-programmers": 866,
    "cqadup-unix": 884,
}


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


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if total < 0 or total > population:
        raise ValueError(f"invalid allocation total {total} for population {population}")
    allocation = {key: total * count // population for key, count in counts.items()}
    remaining = total - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(total * counts[key] % population), key))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def split_hash(input_id: str) -> str:
    return hashlib.sha256(
        f"qmd-public-v0-sft-split\0seed=42\0{input_id}".encode()
    ).hexdigest()


def git_provenance() -> tuple[str, list[str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    )
    dirty = [
        line for line in status.splitlines()
        if line and not line[3:].startswith("finetune/data/public-distill-v0/")
    ]
    return commit, dirty


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
    manifest = load_manifest(run_dir)
    smoke_only = manifest.get("smoke_only") is True
    output_paths = (
        [run_dir / "sft-smoke.jsonl"]
        if smoke_only
        else [run_dir / "sft.jsonl", run_dir / "sft-train.jsonl", run_dir / "sft-validation.jsonl"]
    )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise ValueError(f"materialized artifact already exists: {existing[0]}")
    if manifest.get("selected_sha256") != sha256_file(selected_path):
        raise ValueError("selected.jsonl hash does not match run manifest")
    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    materialization_commit, materialization_dirty = git_provenance()
    if not smoke_only:
        if materialization_dirty:
            raise ValueError(
                "formal materialization requires clean versioned source files: "
                + ", ".join(materialization_dirty)
            )
        if manifest.get("qmd_dirty") is not False or manifest.get("generation_errors") != 0:
            raise ValueError("formal materialization requires a clean run with zero generation errors")
        profile = manifest.get("retrieval_profile")
        if not isinstance(profile, dict) or profile.get("diagnostic_reduced_index") is not False:
            raise ValueError("formal materialization requires the complete formal index")
        if report.get("retrieval_errors") != 0 or report.get("input_queries") != 2500:
            raise ValueError("formal materialization requires 2500 inputs and zero retrieval errors")
    tokenizer = load_tokenizer(TOKENIZER_MODEL, TOKENIZER_REVISION, local_files_only)
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    records = read_jsonl(selected_path)
    if not smoke_only:
        source_counts = Counter(record.get("source_id") for record in records)
        if dict(source_counts) != FORMAL_SOURCE_COUNTS:
            raise ValueError(f"formal source counts do not match frozen quotas: {dict(source_counts)}")
    accepted = [
        record for record in records
        if record.get("selection_status") in {"winner", "qualified_tie"}
    ]
    accepted_strata = Counter(
        f'{record.get("source_id")}|{record.get("selection_status")}' for record in accepted
    )
    selected_records = accepted
    if not smoke_only and len(accepted) > 2000:
        cap_allocation = largest_remainder(dict(accepted_strata), 2000)
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in accepted:
            by_stratum[f'{record.get("source_id")}|{record.get("selection_status")}'].append(record)
        selected_records = []
        for stratum in sorted(by_stratum):
            ordered = sorted(
                by_stratum[stratum],
                key=lambda record: (record.get("sample_key", ""), record.get("qid", "")),
            )
            selected_records.extend(ordered[:cap_allocation[stratum]])
    if not smoke_only and len(selected_records) < 1000:
        raise ValueError("formal accepted set is below the minimum training scale")

    materialized: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    seen_queries: set[str] = set()
    for record in selected_records:
        status = record["selection_status"]
        input_id = record.get("input_id")
        query = record.get("query")
        selected_index = record.get("selected_candidate_index")
        if not isinstance(input_id, str) or not isinstance(query, str):
            raise ValueError("selected record is missing input_id/query")
        if input_id in seen_inputs or normalized(query) in seen_queries:
            raise ValueError(f"duplicate accepted record: {input_id}")
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
                "selection_status": status,
                "selected_candidate_index": selected_index,
                "experiment_id": manifest.get("experiment_id"),
                "smoke_only": smoke_only,
                "final_sft_eligible": not smoke_only,
            }
        )
        materialized.append(rendered)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    if smoke_only:
        atomic_jsonl(output_paths[0], materialized)
    else:
        capped_strata = Counter(
            f'{record["source_id"]}|{record["selection_status"]}' for record in materialized
        )
        validation_allocation = largest_remainder(dict(capped_strata), len(materialized) // 10)
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in materialized:
            by_stratum[f'{record["source_id"]}|{record["selection_status"]}'].append(record)
        for stratum in sorted(by_stratum):
            ordered = sorted(
                by_stratum[stratum],
                key=lambda record: (split_hash(record["input_id"]), record["input_id"]),
            )
            validation_count = validation_allocation[stratum]
            validation.extend({**record, "split": "validation"} for record in ordered[:validation_count])
            train.extend({**record, "split": "train"} for record in ordered[validation_count:])
        combined = train + validation
        atomic_jsonl(output_paths[0], combined)
        atomic_jsonl(output_paths[1], train)
        atomic_jsonl(output_paths[2], validation)

    manifest["materialized_count"] = len(materialized)
    manifest["materialized_smoke_only"] = smoke_only
    manifest["accepted_pre_cap_count"] = len(accepted)
    manifest["accepted_pre_cap_strata"] = dict(sorted(accepted_strata.items()))
    manifest["materialized_strata"] = dict(sorted(Counter(
        f'{record["source_id"]}|{record["selection_status"]}' for record in materialized
    ).items()))
    manifest["materialized_unique_input_count"] = len(seen_inputs)
    manifest["materialized_unique_query_count"] = len(seen_queries)
    manifest["materialization_qmd_commit"] = materialization_commit
    manifest["materialization_qmd_dirty"] = bool(materialization_dirty)
    manifest["final_sft_eligible"] = not smoke_only
    if smoke_only:
        manifest["sft_smoke_sha256"] = sha256_file(output_paths[0])
    else:
        manifest["sft_sha256"] = sha256_file(output_paths[0])
        manifest["sft_train_count"] = len(train)
        manifest["sft_train_sha256"] = sha256_file(output_paths[1])
        manifest["sft_validation_count"] = len(validation)
        manifest["sft_validation_sha256"] = sha256_file(output_paths[2])
    atomic_json(run_dir / "run-manifest.json", manifest)
    if report_path.exists():
        report["materialized_count"] = len(materialized)
        report["accepted_pre_cap_count"] = len(accepted)
        report["accepted_pre_cap_strata"] = manifest["accepted_pre_cap_strata"]
        report["materialized_strata"] = manifest["materialized_strata"]
        report["final_sft_eligible"] = not smoke_only
        if smoke_only:
            report["sft_smoke_sha256"] = manifest["sft_smoke_sha256"]
        else:
            report["sft_sha256"] = manifest["sft_sha256"]
            report["sft_train_count"] = len(train)
            report["sft_train_sha256"] = manifest["sft_train_sha256"]
            report["sft_validation_count"] = len(validation)
            report["sft_validation_sha256"] = manifest["sft_validation_sha256"]
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
