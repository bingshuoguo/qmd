#!/usr/bin/env python3
"""Manifest-gated loading of a sealed SFT release (spec section 2.4).

The trainer's only entry point is a release manifest.  Pointing at a bare JSONL
is not supported: that is exactly the path that silently re-splits a sealed
train file and produces a leaked validation set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset.public_distill import read_jsonl, sha256_file
from dataset.public_distill_v1 import (
    EOS_TOKEN_ID,
    EXPERIMENT_ID,
    MAX_LENGTH,
    PAD_TOKEN_ID,
    RELEASE_ID,
    expected_prompt,
)

TRAIN_ARTIFACT = "sft-train.jsonl"
VALIDATION_ARTIFACT = "sft-validation.jsonl"

# The trainer consumes exactly these two fields; everything else is provenance
# that must survive in the file but must never reach the tokenizer.
SUPERVISION_FIELDS = ("prompt", "completion")


@dataclass(frozen=True)
class ReleaseSplit:
    name: str
    path: Path
    sha256: str
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class Release:
    manifest_path: Path
    manifest_sha256: str
    release_id: str
    experiment_id: str
    prompt_version: str
    prompt_sha256: str
    train: ReleaseSplit
    validation: ReleaseSplit

    def provenance(self) -> dict[str, Any]:
        """The subset of release facts that must land in the run manifest."""
        return {
            "release_id": self.release_id,
            "experiment_id": self.experiment_id,
            "release_manifest_path": str(self.manifest_path),
            "release_manifest_sha256": self.manifest_sha256,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "train_path": str(self.train.path),
            "train_sha256": self.train.sha256,
            "train_rows": len(self.train.records),
            "validation_path": str(self.validation.path),
            "validation_sha256": self.validation.sha256,
            "validation_rows": len(self.validation.records),
        }


def _load_split(
    manifest: dict[str, Any], release_root: Path, artifact: str, expected_split: str
) -> ReleaseSplit:
    entry = manifest.get("core_artifacts", {}).get(artifact)
    if not isinstance(entry, dict):
        raise ValueError(f"release manifest does not declare {artifact}")

    path = release_root / entry["path"]
    if not path.is_file():
        raise ValueError(f"declared artifact is missing: {path}")

    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(
            f"{artifact}: sha256 mismatch\n"
            f"  manifest {entry['sha256']}\n  on disk  {digest}"
        )

    records = read_jsonl(path)
    if len(records) != entry["rows"]:
        raise ValueError(
            f"{artifact}: manifest declares {entry['rows']} rows, file has {len(records)}"
        )

    for record in records:
        input_id = record.get("input_id")
        if record.get("split") != expected_split:
            raise ValueError(f"{artifact}: {input_id} has split {record.get('split')!r}")
        if record.get("smoke_only") is not False:
            raise ValueError(f"{artifact}: {input_id} is a smoke record")
        if record.get("final_sft_eligible") is not True:
            raise ValueError(f"{artifact}: {input_id} is not final_sft_eligible")
        if record.get("release_id") != manifest["release_id"]:
            raise ValueError(f"{artifact}: {input_id} belongs to another release")
        if record.get("experiment_id") != manifest["experiment_id"]:
            raise ValueError(f"{artifact}: {input_id} belongs to another experiment")
        for field in SUPERVISION_FIELDS:
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"{artifact}: {input_id} has an empty {field}")

    return ReleaseSplit(name=expected_split, path=path, sha256=digest, records=records)


def load_release(
    manifest_path: Path,
    expected_release_id: str = RELEASE_ID,
    expected_experiment_id: str = EXPERIMENT_ID,
) -> Release:
    """Load and fully validate a sealed release; raise on any inconsistency."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("status") != "sealed":
        raise ValueError(f"release is not sealed: {manifest.get('status')!r}")
    if manifest.get("final_sft_eligible") is not True:
        raise ValueError("release is not final_sft_eligible")
    if manifest.get("release_id") != expected_release_id:
        raise ValueError(
            f"release_id {manifest.get('release_id')!r} != {expected_release_id!r}"
        )
    if manifest.get("experiment_id") != expected_experiment_id:
        raise ValueError(
            f"experiment_id {manifest.get('experiment_id')!r} != {expected_experiment_id!r}"
        )

    # core_artifacts paths are relative to the release root, two levels above
    # experiments/<experiment_id>/.
    release_root = manifest_path.parents[2]
    train = _load_split(manifest, release_root, TRAIN_ARTIFACT, "train")
    validation = _load_split(manifest, release_root, VALIDATION_ARTIFACT, "validation")

    train_ids = {record["input_id"] for record in train.records}
    validation_ids = {record["input_id"] for record in validation.records}
    if len(train_ids) != len(train.records) or len(validation_ids) != len(validation.records):
        raise ValueError("duplicate input_id within a split")
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"train/validation overlap on {len(overlap)} input_id(s)")

    declared = manifest.get("dataset", {})
    if declared.get("train") != len(train.records):
        raise ValueError("manifest dataset.train does not match the train file")
    if declared.get("validation") != len(validation.records):
        raise ValueError("manifest dataset.validation does not match the validation file")

    provenance = manifest.get("provenance", {})
    return Release(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        release_id=manifest["release_id"],
        experiment_id=manifest["experiment_id"],
        prompt_version=provenance.get("prompt_version", ""),
        prompt_sha256=provenance.get("prompt_sha256", ""),
        train=train,
        validation=validation,
    )


def assert_token_identities(tokenizer: Any) -> None:
    """Spec section 5.2 invariant 11: pad and eos must exist and differ."""
    if tokenizer.pad_token_id != PAD_TOKEN_ID:
        raise ValueError(
            f"pad_token_id {tokenizer.pad_token_id} != frozen {PAD_TOKEN_ID}"
        )
    if tokenizer.eos_token_id != EOS_TOKEN_ID:
        raise ValueError(
            f"eos_token_id {tokenizer.eos_token_id} != frozen {EOS_TOKEN_ID}"
        )
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError(
            "pad_token_id must differ from eos_token_id so that a masking bug "
            "cannot hide inside padding"
        )


def assert_prompt_template(records: list[dict[str, Any]]) -> None:
    """Spec section 3.1: every stored prompt must match the frozen template."""
    for record in records:
        target = expected_prompt(record["query"])
        if record["prompt"] != target:
            raise ValueError(
                f"{record['input_id']}: stored prompt does not match the frozen v1 template"
            )


def preflight_lengths(
    records: list[dict[str, Any]], tokenizer: Any, max_length: int = MAX_LENGTH
) -> dict[str, int]:
    """Tokenize every record before the model is loaded; reject over-length.

    Returns aggregate token counts for the run manifest.
    """
    input_tokens = 0
    supervised_tokens = 0
    longest = 0
    for record in records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(
            record["prompt"] + record["completion"], add_special_tokens=False
        )["input_ids"]
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"{record['input_id']}: prompt tokenization is not a prefix of the full sequence"
            )
        if len(input_ids) <= len(prompt_ids):
            raise ValueError(f"{record['input_id']}: completion has no token")
        if len(input_ids) > max_length:
            raise ValueError(
                f"{record['input_id']}: {len(input_ids)} tokens exceed max_length={max_length}; "
                "refusing to truncate completion-only supervision"
            )
        input_tokens += len(input_ids)
        supervised_tokens += len(input_ids) - len(prompt_ids)
        longest = max(longest, len(input_ids))
    return {
        "records": len(records),
        "input_tokens": input_tokens,
        "supervised_tokens": supervised_tokens,
        "longest_sequence": longest,
    }
