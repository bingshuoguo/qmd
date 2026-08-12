#!/usr/bin/env python3
"""Validate SciFact teacher candidates and materialize completion-only SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from dataset.completion import split_rendered_text
from dataset.contract import CONTRACT_VERSION, validate_training_target
from dataset.schema import output_items_to_text
from dataset.validate_contract import DEFAULT_TOKENIZER_MODEL

SCIFACT_SEMANTIC_GATE_VERSION = "scifact-observational-v3"


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_scifact_semantics(
    query: str, output: list[Any]
) -> dict[str, Any]:
    """Record auditable SciFact observations without blocking any candidate.

    Every check this gate used to enforce was a presence test over a fixed word
    list standing in for a semantic property, and each was measured to reject
    faithful expansions often enough that the benchmark scored the gate rather
    than the teacher. On ``distill-deepseek-v4-flash-v3-pilot-100``:
    ``unsupported_clinical_advice`` fired 17 times, all 17 on the model's own
    description of the retrieval task ("evidence should assess ...");
    ``established_fact_risk`` fired 3 times, all 3 on plain questions;
    ``unsupported_mechanism`` (77) and ``unsupported_comparison`` (37) fired
    roughly 60% of the time on claims that already carried the concept through
    wording the list cannot see, such as "by increasing X" or "as effective
    as"; ``negation_lost`` fired 171 times with 63 false. They are removed
    rather than downgraded: a list that blind produces noise, not weak signal.
    ``fixed_profile_count`` is gone too, since it bound scoring eligibility to
    one prompt's shape while Contract v1.1 already validates structure.

    ``negation_lost`` survives as the sole advisory because it marks the one
    failure mode a human audit independently rated critical (qid 1004). It is
    recorded for a future redesign and never affects ``valid``.

    ``valid`` is therefore always True: nothing here removes a candidate from
    scoring, selection, or materialization.
    """
    advisories: list[dict[str, str]] = []
    typed = [item for item in output if isinstance(item, list) and len(item) == 2]

    negation = (
        r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\black(?:s|ed|ing)?\b",
        r"\bneither\b", r"\bnor\b", r"\bfail(?:s|ed|ing)?\s+to\b",
        r"\bunable\s+to\b",
    )
    query_has_negation = _contains_any(query, negation)

    for index, item in enumerate(typed):
        _, text = item
        if not isinstance(text, str):
            continue
        if query_has_negation and not _contains_any(text, negation):
            advisories.append({
                "code": "negation_lost",
                "path": f"output[{index}][1]",
                "message": "the source claim contains negation but this expansion does not",
            })
    return {
        "version": SCIFACT_SEMANTIC_GATE_VERSION,
        "valid": True,
        "errors": [],
        "advisories": advisories,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(model: str, revision: str, local_files_only: bool) -> Any:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("tokenizer revision must be a 40-character commit hash")
    from transformers import AutoTokenizer

    source = model
    if local_files_only:
        from huggingface_hub import snapshot_download

        source = snapshot_download(model, revision=revision, local_files_only=True)
    return AutoTokenizer.from_pretrained(
        source,
        revision=None if local_files_only else revision,
        local_files_only=local_files_only,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            qid = record.get("qid")
            if not isinstance(qid, str) or not qid:
                raise ValueError(f"{path}:{line_number}: qid must be a non-empty string")
            if qid in seen:
                raise ValueError(f"{path}:{line_number}: duplicate qid {qid}")
            seen.add(qid)
            records.append(record)
    return records


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_candidates(
    records: list[dict[str, Any]], token_counter: Callable[[str], int],
    semantic_gate: str | None = None,
) -> list[dict[str, Any]]:
    for record in records:
        if record.get("selection_status") != "pending" or record.get("raw_metrics") is not None:
            raise ValueError(f"qid {record.get('qid')}: candidates have already been scored")
        query = record.get("query")
        candidates = record.get("candidates")
        if not isinstance(query, str) or not query:
            raise ValueError(f"qid {record.get('qid')}: query must be non-empty")
        if not isinstance(candidates, list) or len(candidates) != 4:
            raise ValueError(f"qid {record.get('qid')}: exactly four candidates are required")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or candidate.get("candidate_index") != candidate_index:
                raise ValueError(
                    f"qid {record.get('qid')}: candidate indexes must be contiguous from zero"
                )
            parsed_output = candidate.get("parsed_output")
            if not isinstance(parsed_output, list):
                raise ValueError(
                    f"qid {record.get('qid')} candidate {candidate_index}: parsed_output must be a list"
                )
            result = validate_training_target(
                {"query": query, "output": parsed_output}, token_counter
            )
            candidate["contract"] = {
                "version": CONTRACT_VERSION,
                "valid": result.valid,
                "errors": [asdict(item) for item in result.errors],
                "warnings": [asdict(item) for item in result.warnings],
                "canonical_output": result.canonical_record["output"],
            }
            candidate["semantic_gate"] = (
                validate_scifact_semantics(query, parsed_output)
                if semantic_gate == SCIFACT_SEMANTIC_GATE_VERSION
                else None
            )
            candidate["metrics"] = None
    return records


def render_sft_record(
    qid: str, query: str, output: list[list[str]], tokenizer: Any
) -> dict[str, Any]:
    completion_text = output_items_to_text(output)
    messages = [
        {"role": "user", "content": f"/no_think Expand this search query: {query}"},
        {"role": "assistant", "content": completion_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    ).replace("<think>\n\n</think>\n\n", "")
    prompt, completion = split_rendered_text(rendered)
    return {
        "qid": qid,
        "query": query,
        "output": output,
        "prompt": prompt,
        "completion": completion,
    }


def materialize_records(
    records: list[dict[str, Any]], tokenizer: Any
) -> dict[str, list[dict[str, Any]]]:
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for record in records:
        status = record.get("selection_status")
        if status == "pending":
            raise ValueError(f"qid {record.get('qid')}: candidates have not been scored")
        if status != "winner":
            continue
        split = record.get("split")
        if split not in output:
            raise ValueError(f"qid {record.get('qid')}: invalid split {split!r}")
        selected_index = record.get("selected_candidate_index")
        candidates = record.get("candidates")
        if not isinstance(selected_index, int) or not isinstance(candidates, list):
            raise ValueError(f"qid {record.get('qid')}: winner is missing selected candidate")
        try:
            selected = candidates[selected_index]
        except IndexError as error:
            raise ValueError(f"qid {record.get('qid')}: selected candidate is out of range") from error
        contract = selected.get("contract") if isinstance(selected, dict) else None
        canonical = contract.get("canonical_output") if isinstance(contract, dict) else None
        if contract is None or contract.get("valid") is not True or not isinstance(canonical, list):
            raise ValueError(f"qid {record.get('qid')}: selected candidate is not Contract-valid")
        semantic_gate = selected.get("semantic_gate")
        if isinstance(semantic_gate, dict) and semantic_gate.get("valid") is not True:
            raise ValueError(f"qid {record.get('qid')}: selected candidate failed semantic gate")
        validation = validate_training_target(
            {"query": record["query"], "output": canonical}, token_counter
        )
        if not validation.valid or validation.canonical_record["output"] != canonical:
            raise ValueError(f"qid {record.get('qid')}: selected candidate failed revalidation")
        output[split].append(
            render_sft_record(record["qid"], record["query"], canonical, tokenizer)
        )
    if not output["train"] or not output["val"]:
        raise ValueError("materialized train and val outputs must both be non-empty")
    return output


def update_manifest(run_dir: Path, updates: dict[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    manifest.update(updates)
    atomic_write_json(manifest_path, manifest)


def validate_command(args: argparse.Namespace, tokenizer: Any) -> None:
    run_dir = args.run_dir.resolve()
    candidates_path = run_dir / "candidates.jsonl"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_hash = sha256_file(candidates_path)
    if manifest.get("candidates_sha256") != current_hash:
        raise ValueError("candidates.jsonl does not match the generated artifact hash")
    counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    records = validate_candidates(read_jsonl(candidates_path), counter, args.semantic_gate)
    atomic_write_jsonl(candidates_path, records)
    update_manifest(
        run_dir,
        {
            "contract": {
                "version": CONTRACT_VERSION,
                "tokenizer_model": args.tokenizer_model,
                "tokenizer_revision": args.tokenizer_revision,
            },
            "validated_candidates_sha256": sha256_file(candidates_path),
            "semantic_gate": (
                {"version": args.semantic_gate} if args.semantic_gate is not None else None
            ),
        },
    )
    valid = sum(
        1
        for record in records
        for candidate in record["candidates"]
        if candidate["contract"]["valid"]
    )
    admitted = sum(
        1
        for record in records
        for candidate in record["candidates"]
        if candidate["contract"]["valid"]
        and (candidate.get("semantic_gate") or {}).get("valid", True)
    )
    print(json.dumps({
        "queries": len(records),
        "valid_candidates": valid,
        "semantic_admitted_candidates": admitted,
    }, indent=2))


def materialize_command(args: argparse.Namespace, tokenizer: Any) -> None:
    run_dir = args.run_dir.resolve()
    candidates_path = run_dir / "candidates.jsonl"
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("scored_candidates_sha256") != sha256_file(candidates_path):
        raise ValueError("candidates.jsonl does not match the scored artifact hash")
    prepared = materialize_records(read_jsonl(candidates_path), tokenizer)
    output_dir = run_dir / "sft"
    if output_dir.exists():
        raise FileExistsError(f"SFT output already exists: {output_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=".sft.", dir=run_dir))
    try:
        for split, records in prepared.items():
            atomic_write_jsonl(temporary / f"{split}.jsonl", records)
        os.replace(temporary, output_dir)
    except BaseException:
        for child in temporary.glob("*"):
            child.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    update_manifest(
        run_dir,
        {
            "sft": {
                "train_records": len(prepared["train"]),
                "val_records": len(prepared["val"]),
                "train_sha256": sha256_file(output_dir / "train.jsonl"),
                "val_sha256": sha256_file(output_dir / "val.jsonl"),
            }
        },
    )
    print(
        json.dumps(
            {"output": str(output_dir), **{k: len(v) for k, v in prepared.items()}},
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "materialize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-dir", required=True, type=Path)
        subparser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
        subparser.add_argument("--tokenizer-revision", required=True)
        subparser.add_argument("--local-files-only", action="store_true")
        if command == "validate":
            subparser.add_argument(
                "--semantic-gate", choices=[SCIFACT_SEMANTIC_GATE_VERSION]
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        tokenizer = load_tokenizer(
            args.tokenizer_model, args.tokenizer_revision, args.local_files_only
        )
        if args.command == "validate":
            validate_command(args, tokenizer)
        else:
            materialize_command(args, tokenizer)
    except Exception as error:
        print(f"SciFact distillation {args.command} failed: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
