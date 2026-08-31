from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from dataset import sft_release


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_VERSION = "qmd-student-expansion-v2-vh-v1"
PROMPT_SHA256 = "fce6b582fff10e7e1a601097e0fa00273f5b106fcd821985422932a5ce797d93"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ADAPTER_SHA256 = "d6eff80e87f43a33f9fe5ebb7dd6e0f9acb83d5d62b67254dc519ae154234942"
SOURCES = ("nfcorpus", "fiqa", "freshstack", "scifact")


@dataclass(frozen=True)
class PromptContract:
    version: str
    sha256: str
    template: str
    manifest_path: Path
    manifest_sha256: str

    def render(self, query: str) -> str:
        return sft_release.expected_prompt(self.template, query)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str | None
    adapter: Path | None = None
    adapter_sha256: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    max_prompt_tokens: int
    max_new_tokens: int
    max_model_len: int
    do_sample: bool
    num_beams: int
    responses_per_query: int
    dtype: str
    quantization: str | None
    seed: int


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str
    rerank: bool
    result_limit: int
    per_list_limit: int
    candidate_limit: int


@dataclass(frozen=True)
class StatisticsConfig:
    bootstrap_resamples: int
    seed: int
    source_floor: float


@dataclass(frozen=True)
class QualificationConfig:
    schema_version: str
    mode: Literal["formal", "dry_run"]
    teacher_revision_status: Literal["frozen", "unresolved"]
    release_manifest: Path
    output_root: Path
    merged_v2: ModelConfig
    teacher: ModelConfig
    generation: GenerationConfig
    retrieval: RetrievalConfig
    statistics: StatisticsConfig
    sources: dict[str, Path]

    def manifest(self) -> dict[str, Any]:
        value = asdict(self)
        return _paths_to_strings(value)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def evaluation_manifest(self) -> Path:
        return self.root / "evaluation-manifest.json"

    @property
    def decision(self) -> Path:
        return self.root / "decision.json"


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_paths_to_strings(item) for item in value]
    return value


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed_prompt(path: Path) -> PromptContract:
    release = sft_release.load_release(
        path,
        expected_release_id="public-distill-v2-vh-prompt-v1",
        expected_experiment_id="public-main-v2-vh-prompt-v1",
    )
    if release.prompt_version != PROMPT_VERSION:
        raise ValueError(f"prompt version {release.prompt_version!r} != {PROMPT_VERSION!r}")
    if release.prompt_sha256 != PROMPT_SHA256:
        raise ValueError(f"prompt sha256 {release.prompt_sha256!r} != {PROMPT_SHA256!r}")
    return PromptContract(
        version=release.prompt_version,
        sha256=release.prompt_sha256,
        template=release.prompt_template,
        manifest_path=release.manifest_path,
        manifest_sha256=release.manifest_sha256,
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_config(
    path: Path, mode: Literal["formal", "dry_run"] = "formal"
) -> QualificationConfig:
    if mode not in {"formal", "dry_run"}:
        raise ValueError("mode must be formal or dry_run")
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    models = _require_mapping(raw.get("models"), "models")
    merged_raw = _require_mapping(models.get("merged_v2"), "models.merged_v2")
    teacher_raw = _require_mapping(models.get("teacher"), "models.teacher")
    generation = GenerationConfig(**_require_mapping(raw.get("generation"), "generation"))
    retrieval = RetrievalConfig(**_require_mapping(raw.get("retrieval"), "retrieval"))
    statistics = StatisticsConfig(**_require_mapping(raw.get("statistics"), "statistics"))
    if (generation.max_prompt_tokens, generation.max_new_tokens, generation.max_model_len) != (512, 256, 1024):
        raise ValueError("generation token contract must be 512 prompt / 256 completion / 1024 model")
    if generation.do_sample or generation.num_beams != 1 or generation.responses_per_query != 1:
        raise ValueError("qualification generation must be one greedy response")
    if generation.dtype != "bfloat16" or generation.quantization is not None:
        raise ValueError("qualification generation must use unquantized bfloat16")
    if retrieval.rerank:
        raise ValueError("qualification retrieval must set rerank=false")

    adapter = resolve_repo_path(str(merged_raw["adapter"]))
    adapter_weights = adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise ValueError(f"adapter weights are missing: {adapter_weights}")
    if merged_raw.get("revision") != BASE_REVISION:
        raise ValueError("merged-v2 base revision does not match the sealed SFT run")
    if merged_raw.get("adapter_sha256") != ADAPTER_SHA256:
        raise ValueError("configured adapter sha256 does not match the sealed SFT run")
    if sha256_path(adapter_weights) != ADAPTER_SHA256:
        raise ValueError("adapter weight sha256 does not match the configured value")

    revision_env = str(teacher_raw.get("revision_env", ""))
    teacher_revision = os.environ.get(revision_env) if revision_env else None
    frozen = bool(teacher_revision and re.fullmatch(r"[0-9a-f]{40}", teacher_revision))
    if mode == "formal" and not frozen:
        raise ValueError("formal teacher revision must be a 40-character lowercase commit SHA")
    release_manifest = resolve_repo_path(str(raw["release_manifest"]))
    load_sealed_prompt(release_manifest)
    source_raw = _require_mapping(raw.get("sources"), "sources")
    required_source_keys = {"nfcorpus", "fiqa", "freshstack", "scifact_archive"}
    if set(source_raw) != required_source_keys:
        raise ValueError(f"sources must be exactly {sorted(required_source_keys)}")

    return QualificationConfig(
        schema_version=str(raw["schema_version"]),
        mode=mode,
        teacher_revision_status="frozen" if frozen else "unresolved",
        release_manifest=release_manifest,
        output_root=resolve_repo_path(str(raw["output_root"])),
        merged_v2=ModelConfig(
            model_id=str(merged_raw["model_id"]),
            revision=str(merged_raw["revision"]),
            adapter=adapter,
            adapter_sha256=str(merged_raw["adapter_sha256"]),
        ),
        teacher=ModelConfig(model_id=str(teacher_raw["model_id"]), revision=teacher_revision),
        generation=generation,
        retrieval=retrieval,
        statistics=statistics,
        sources={key: resolve_repo_path(str(value)) for key, value in source_raw.items()},
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(path)
