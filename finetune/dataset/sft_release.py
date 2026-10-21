"""Manifest-gated loading of a sealed SFT release.

A release is only ever addressed through its manifest. Pointing at a bare JSONL
is not supported: that is the path that silently re-splits a sealed train file
and produces a leaked validation set.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset.artifact_io import read_jsonl, sha256_file

TRAIN_ARTIFACT = "sft-train.jsonl"
VALIDATION_ARTIFACT = "sft-validation.jsonl"

# Supervision fields; everything else in a record is provenance.
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
    prompt_template: str
    train: ReleaseSplit
    validation: ReleaseSplit

    def provenance(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "experiment_id": self.experiment_id,
            "release_manifest_path": str(self.manifest_path),
            "release_manifest_sha256": self.manifest_sha256,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "prompt_template": self.prompt_template,
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

    path = Path(os.path.normpath(release_root / entry["path"]))
    if not path.is_relative_to(release_root):
        raise ValueError(f"{artifact}: path escapes the release root: {entry['path']}")
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


def load_release(manifest_path: Path) -> Release:
    """Load and fully validate a sealed release; raise on any inconsistency."""
    # abspath, not resolve(): an archived experiment directory may be a symlink,
    # and core_artifacts paths are relative to the release root as addressed.
    manifest_path = Path(os.path.abspath(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("status") != "sealed":
        raise ValueError(f"release is not sealed: {manifest.get('status')!r}")
    if manifest.get("final_sft_eligible") is not True:
        raise ValueError("release is not final_sft_eligible")
    for key in ("release_id", "experiment_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"release manifest {key} must be a non-empty string")

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
    prompt_template = provenance.get("prompt_template")
    prompt_sha256 = provenance.get("prompt_sha256")
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ValueError("release provenance.prompt_template must be a non-empty string")
    if not isinstance(prompt_sha256, str):
        raise ValueError("release provenance.prompt_sha256 must be a string")
    observed_prompt_sha256 = hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
    if observed_prompt_sha256 != prompt_sha256:
        raise ValueError(
            "release provenance.prompt_sha256 does not match prompt_template\n"
            f"  manifest {prompt_sha256}\n"
            f"  observed {observed_prompt_sha256}"
        )
    return Release(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        release_id=manifest["release_id"],
        experiment_id=manifest["experiment_id"],
        prompt_version=provenance.get("prompt_version", ""),
        prompt_sha256=prompt_sha256,
        prompt_template=prompt_template,
        train=train,
        validation=validation,
    )


def expected_prompt(prompt_template: str, query: str) -> str:
    """Render the sealed user template into the pinned Qwen chat envelope."""
    return (
        "<|im_start|>user\n"
        f"{prompt_template.replace('{query}', query)}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def assert_prompt_template(records: list[dict[str, Any]], prompt_template: str) -> None:
    """Every stored prompt must match the sealed release template."""
    for record in records:
        target = expected_prompt(prompt_template, record["query"])
        if record["prompt"] != target:
            raise ValueError(
                f"{record['input_id']}: stored prompt does not match the sealed release template"
            )
