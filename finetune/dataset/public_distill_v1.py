#!/usr/bin/env python3
"""Student-prompt rematerialization of the sealed public-distill-v0 SFT release.

Implements section 2.2 of the completion-only SFT training spec: v1 is derived
from v0 by re-rendering the student prompt only.  Every other field, the
completion bytes, the split membership and the record order are carried over
unchanged.  Teacher generation, retrieval, scoring and winner selection are
never re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from dataset.contract import CONTRACT_VERSION, validate_training_target
from dataset.public_distill import (
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
    atomic_json,
    atomic_jsonl,
    normalized,
    read_jsonl,
    sha256_file,
)
from dataset.schema import output_items_to_text
from dataset.scifact_distill import load_tokenizer

SOURCE_RELEASE_ID = "public-distill-v0"
SOURCE_EXPERIMENT_ID = "public-main-v0"
RELEASE_ID = "public-distill-v1"
EXPERIMENT_ID = "public-main-v1"
SCHEMA_VERSION = "qmd-public-distill-v1"

# Sealed v0 hashes (spec section 2.1).  These are pinned here so a
# rematerialization can never silently consume a different source release.
SOURCE_SHA256 = {
    "sft.jsonl": "470ed941be0c5d4821fe896f9de569cb96cbef2db0644ad503f96ca6e913d7df",
    "sft-train.jsonl": "615cc30d05363090d6d05d5be432dd8396d790b85998fb533ff74f1b95dab0b3",
    "sft-validation.jsonl": "53d4fc25ef3a11f841d0e0a5f199d69b2b0ba401e46a92f6663574ed3fbad916",
    "release-manifest.json": "7f84b7f2d7966a6e2b3e173135f0ae0bec390f170334ed34400171c6e45c5f0a",
    "final-audit.json": "9b841110425c13a0525f3229fd54691944bda6999bb1428bc6c5717ed5ab1703",
}

EXPECTED_COUNTS = {"sft.jsonl": 2000, "sft-train.jsonl": 1800, "sft-validation.jsonl": 200}

PROMPT_VERSION = "qmd-student-expansion-v1"

# Spec section 3.1.  The literal `{query}` placeholder is part of the versioned
# template; PROMPT_SHA256 covers the template, not any individual rendering.
USER_PROMPT_TEMPLATE = (
    "/no_think Expand this search query for hybrid retrieval.\n"
    "Output only newline-separated entries in this order:\n"
    "0-1 hyde: a concise hypothetical relevant passage\n"
    "1-3 lex: short keyword-focused queries for BM25\n"
    "1-3 vec: concise natural-language reformulations for vector search\n"
    "No explanations, bullets, Markdown, or other text.\n"
    "Query: {query}"
)

ASSISTANT_HEADER = "<|im_start|>assistant\n"
END_OF_TURN = "<|im_end|>\n"

# Frozen token identities for the pinned Qwen revision (spec section 5.2.11).
PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645

MAX_LENGTH = 1024


def prompt_sha256() -> str:
    return hashlib.sha256(USER_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def build_user_content(query: str) -> str:
    return USER_PROMPT_TEMPLATE.replace("{query}", query)


def expected_prompt(query: str) -> str:
    """The char-exact rendered prompt mandated by spec section 3.1."""
    return f"<|im_start|>user\n{build_user_content(query)}<|im_end|>\n{ASSISTANT_HEADER}"


def render_prompt(query: str, tokenizer: Any) -> str:
    """Render via the real chat template and assert it matches the frozen form.

    Going through `apply_chat_template` rather than string formatting is what
    makes the assertion meaningful: it fails loudly if the template ever starts
    injecting a default system message or an empty `<think>` block.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_user_content(query)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    target = expected_prompt(query)
    if rendered != target:
        raise ValueError(
            "rendered prompt does not match the frozen v1 template\n"
            f"  rendered: {rendered!r}\n"
            f"  expected: {target!r}"
        )
    return rendered


def convert_record(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Carry a v0 SFT record to v1, changing only the student prompt."""
    query = record["query"]
    completion = record["completion"]
    output = record["output"]

    if not isinstance(query, str) or not query:
        raise ValueError(f"{record.get('input_id')}: query must be a non-empty string")
    if not isinstance(completion, str) or not completion.endswith(END_OF_TURN):
        raise ValueError(
            f"{record.get('input_id')}: completion must end with {END_OF_TURN!r}"
        )
    if output_items_to_text(output) + END_OF_TURN != completion:
        raise ValueError(
            f"{record.get('input_id')}: structured output does not render to the completion"
        )
    if record.get("smoke_only") is not False:
        raise ValueError(f"{record.get('input_id')}: smoke records must not reach v1")

    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": RELEASE_ID,
        "experiment_id": EXPERIMENT_ID,
        "input_id": record["input_id"],
        "source_id": record["source_id"],
        "qid": record["qid"],
        "query": query,
        "prompt": render_prompt(query, tokenizer),
        "completion": completion,
        "output": output,
        "selected_candidate_index": record["selected_candidate_index"],
        "selection_status": record["selection_status"],
        "split": record["split"],
        "smoke_only": False,
        "final_sft_eligible": True,
    }


def token_preflight(
    record: dict[str, Any], tokenizer: Any, max_length: int = MAX_LENGTH
) -> tuple[int, int]:
    """Enforce the section 5.2 token invariants; return (prompt, total) lengths.

    Rejection, never truncation: an over-length record fails the whole run.
    """
    prompt, completion = record["prompt"], record["completion"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"{record['input_id']}: prompt tokenization is not a prefix of prompt + completion"
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"{record['input_id']}: completion has no token")
    if input_ids[-1] != EOS_TOKEN_ID and EOS_TOKEN_ID not in input_ids[len(prompt_ids) :]:
        raise ValueError(f"{record['input_id']}: completion carries no end-of-turn token")
    if len(input_ids) > max_length:
        raise ValueError(
            f"{record['input_id']}: {len(input_ids)} tokens exceed max_length={max_length}; "
            "refusing to truncate completion-only supervision"
        )
    return len(prompt_ids), len(input_ids)


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    index = min(len(sorted_values) - 1, int(len(sorted_values) * fraction))
    return sorted_values[index]


def _length_stats(prompt_lengths: list[int], total_lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(total_lengths)
    completion_lengths = [
        total - prompt for prompt, total in zip(prompt_lengths, total_lengths)
    ]
    return {
        "count": len(ordered),
        "prompt_tokens_min": min(prompt_lengths),
        "prompt_tokens_max": max(prompt_lengths),
        "total_tokens_min": ordered[0],
        "total_tokens_p50": _percentile(ordered, 0.50),
        "total_tokens_p95": _percentile(ordered, 0.95),
        "total_tokens_p99": _percentile(ordered, 0.99),
        "total_tokens_max": ordered[-1],
        "supervised_tokens_min": min(completion_lengths),
        "supervised_tokens_max": max(completion_lengths),
        "supervised_tokens_total": sum(completion_lengths),
        "input_tokens_total": sum(total_lengths),
    }


def _verify_source(source_dir: Path) -> dict[str, str]:
    """Check the five sealed v0 hashes before reading anything else."""
    observed: dict[str, str] = {}
    for name, expected in SOURCE_SHA256.items():
        path = source_dir / name
        if not path.is_file():
            raise ValueError(f"sealed source artifact is missing: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(
                f"{name}: sealed source hash mismatch\n"
                f"  expected {expected}\n  observed {digest}"
            )
        observed[name] = digest
    return observed


def _artifact_entry(path: Path, release_root: Path, rows: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(path.relative_to(release_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        entry["rows"] = rows
    return entry


def rematerialize(
    source_dir: Path,
    target_dir: Path,
    local_files_only: bool,
    tokenizer_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Produce the v1 release from the sealed v0 release."""
    source_hashes = _verify_source(source_dir)
    source_manifest = json.loads(
        (source_dir / "release-manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("status") != "sealed":
        raise ValueError("source release is not sealed")
    if source_manifest.get("final_sft_eligible") is not True:
        raise ValueError("source release is not final_sft_eligible")

    outputs = {name: target_dir / name for name in EXPECTED_COUNTS}
    outputs["final-audit.json"] = target_dir / "final-audit.json"
    outputs["release-manifest.json"] = target_dir / "release-manifest.json"
    existing = sorted(str(path) for path in outputs.values() if path.exists())
    if existing:
        raise ValueError(f"v1 artifact already exists, refusing to overwrite: {existing[0]}")

    splits = {
        name: read_jsonl(source_dir / name)
        for name in ("sft.jsonl", "sft-train.jsonl", "sft-validation.jsonl")
    }
    for name, expected in EXPECTED_COUNTS.items():
        if len(splits[name]) != expected:
            raise ValueError(f"{name}: expected {expected} rows, found {len(splits[name])}")

    train_ids = [record["input_id"] for record in splits["sft-train.jsonl"]]
    validation_ids = [record["input_id"] for record in splits["sft-validation.jsonl"]]
    combined_ids = [record["input_id"] for record in splits["sft.jsonl"]]
    if combined_ids != train_ids + validation_ids:
        raise ValueError("sft.jsonl is not the train split followed by the validation split")
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation splits overlap")
    if len(set(combined_ids)) != len(combined_ids):
        raise ValueError("duplicate input_id in the source release")

    tokenizer = (
        tokenizer_factory()
        if tokenizer_factory is not None
        else load_tokenizer(TOKENIZER_MODEL, TOKENIZER_REVISION, local_files_only)
    )

    converted: dict[str, list[dict[str, Any]]] = {}
    prompt_lengths: dict[str, list[int]] = {}
    total_lengths: dict[str, list[int]] = {}
    contract_invalid = 0
    normalized_queries: set[str] = set()

    for name in ("sft-train.jsonl", "sft-validation.jsonl"):
        records, prompts, totals = [], [], []
        for record in splits[name]:
            converted_record = convert_record(record, tokenizer)
            if converted_record["completion"] != record["completion"]:
                raise ValueError(f"{record['input_id']}: completion bytes changed")
            prompt_length, total_length = token_preflight(converted_record, tokenizer)
            result = validate_training_target(
                {"query": converted_record["query"], "output": converted_record["output"]},
                lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
            )
            if not result.valid:
                contract_invalid += 1
            normalized_queries.add(normalized(converted_record["query"]))
            records.append(converted_record)
            prompts.append(prompt_length)
            totals.append(total_length)
        converted[name] = records
        prompt_lengths[name] = prompts
        total_lengths[name] = totals

    if contract_invalid:
        raise ValueError(f"{contract_invalid} v1 records failed Contract revalidation")
    if len(normalized_queries) != sum(EXPECTED_COUNTS[name] for name in converted):
        raise ValueError("normalized queries are not unique across the v1 release")

    converted["sft.jsonl"] = converted["sft-train.jsonl"] + converted["sft-validation.jsonl"]
    prompt_lengths["sft.jsonl"] = (
        prompt_lengths["sft-train.jsonl"] + prompt_lengths["sft-validation.jsonl"]
    )
    total_lengths["sft.jsonl"] = (
        total_lengths["sft-train.jsonl"] + total_lengths["sft-validation.jsonl"]
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_COUNTS:
        atomic_jsonl(outputs[name], converted[name])

    release_root = target_dir.parents[1]
    core_artifacts = {
        name: _artifact_entry(outputs[name], release_root, rows=len(converted[name]))
        for name in EXPECTED_COUNTS
    }
    strata = Counter(
        f'{record["source_id"]}|{record["selection_status"]}'
        for record in converted["sft.jsonl"]
    )
    validation_strata = Counter(
        f'{record["source_id"]}|{record["selection_status"]}'
        for record in converted["sft-validation.jsonl"]
    )

    provenance = {
        "source_release_id": SOURCE_RELEASE_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_sha256": source_hashes,
        "transformation": "student-prompt-rematerialization",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "prompt_template": USER_PROMPT_TEMPLATE,
        "contract": {
            "version": CONTRACT_VERSION,
            "tokenizer_model": TOKENIZER_MODEL,
            "tokenizer_revision": TOKENIZER_REVISION,
        },
        "inherited": source_manifest.get("commits", {}),
    }

    audit = {
        "schema_version": "qmd-public-distill-final-audit-v1",
        "release_id": RELEASE_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "passed",
        "provenance": provenance,
        "checks": {
            "materialized": len(converted["sft.jsonl"]),
            "train": len(converted["sft-train.jsonl"]),
            "validation": len(converted["sft-validation.jsonl"]),
            "unique_input_ids": len(set(combined_ids)),
            "unique_normalized_queries": len(normalized_queries),
            "split_overlap": 0,
            "smoke_records": 0,
            "contract_invalid": contract_invalid,
            "completion_bytes_changed": 0,
            "prompt_template_mismatches": 0,
            "over_max_length": 0,
            "max_length": MAX_LENGTH,
            "pad_token_id": PAD_TOKEN_ID,
            "eos_token_id": EOS_TOKEN_ID,
        },
        "length_stats": {
            name: _length_stats(prompt_lengths[name], total_lengths[name])
            for name in ("sft.jsonl", "sft-train.jsonl", "sft-validation.jsonl")
        },
        "materialized_strata": dict(sorted(strata.items())),
        "validation_strata": dict(sorted(validation_strata.items())),
        "core_artifacts": core_artifacts,
        "oracle_metrics_note": (
            "Inherited v0 oracle selection metrics validate label production only; "
            "they are not model gains."
        ),
    }
    atomic_json(outputs["final-audit.json"], audit)

    manifest = {
        "schema_version": "qmd-public-distill-release-v1",
        "release_id": RELEASE_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed",
        "final_sft_eligible": True,
        "provenance": provenance,
        "dataset": {
            "materialized": len(converted["sft.jsonl"]),
            "train": len(converted["sft-train.jsonl"]),
            "validation": len(converted["sft-validation.jsonl"]),
            "materialized_strata": dict(sorted(strata.items())),
        },
        "core_artifacts": core_artifacts,
        "final_audit": _artifact_entry(outputs["final-audit.json"], release_root),
        "mutation_policy": (
            "Any change to a sealed core artifact requires a new release_id and experiment_id."
        ),
    }
    atomic_json(outputs["release-manifest.json"], manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/public-distill-v0/experiments/public-main-v0"),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/public-distill-v1/experiments/public-main-v1"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    manifest = rematerialize(
        args.source_dir.resolve(), args.target_dir.resolve(), args.local_files_only
    )
    print(json.dumps(manifest["core_artifacts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
