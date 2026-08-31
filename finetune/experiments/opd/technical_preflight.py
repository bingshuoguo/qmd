from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import QualificationConfig, atomic_json, load_sealed_prompt


def assert_immutable_revision(revision: str | None) -> None:
    if not revision or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model revision must be a 40-character lowercase commit SHA")


def merge_equivalence(
    before: Iterable[float],
    after: Iterable[float],
    *,
    max_tolerance: float = 5e-3,
    mean_tolerance: float = 5e-4,
) -> dict[str, Any]:
    left = np.asarray(list(before), dtype=np.float64)
    right = np.asarray(list(after), dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        return {"status": "failed", "error": "logit shapes differ or are empty"}
    difference = np.abs(left - right)
    maximum = float(difference.max())
    mean = float(difference.mean())
    return {
        "status": "passed" if maximum <= max_tolerance and mean <= mean_tolerance else "failed",
        "max_abs_diff": maximum,
        "mean_abs_diff": mean,
        "max_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
    }


def tokenizer_compatibility(
    student: Any, teacher: Any, samples: list[str]
) -> dict[str, Any]:
    errors: list[str] = []
    for field in ("vocab_size", "pad_token_id", "eos_token_id"):
        if getattr(student, field, None) != getattr(teacher, field, None):
            errors.append(
                f"{field} differs: student={getattr(student, field, None)!r}, teacher={getattr(teacher, field, None)!r}"
            )
    for sample in samples:
        student_ids = student(sample, add_special_tokens=False)["input_ids"]
        teacher_ids = teacher(sample, add_special_tokens=False)["input_ids"]
        if student_ids != teacher_ids:
            errors.append(f"token IDs differ for sample sha256={hashlib.sha256(sample.encode()).hexdigest()}")
    return {"status": "failed" if errors else "passed", "errors": errors, "sample_count": len(samples)}


def _assert_cuda_bf16(model: Any) -> None:
    import torch

    for name, parameter in model.named_parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError(f"parameter {name} is on {parameter.device}; CPU/offload is forbidden")
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"parameter {name} has dtype {parameter.dtype}; BF16 is required")


def _score_completion(model: Any, prompt_ids: list[int], completion_ids: list[int]) -> list[float]:
    import torch

    input_ids = torch.tensor([prompt_ids + completion_ids], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    start = len(prompt_ids) - 1
    positions = torch.arange(start, start + len(completion_ids), device="cuda")
    targets = torch.tensor(completion_ids, device="cuda", dtype=torch.long)
    values = log_probs[positions, targets]
    if not torch.isfinite(values).all():
        raise RuntimeError("teacher returned a non-finite completion token log-prob")
    return [float(value) for value in values.cpu()]


def run_technical_preflight(
    config: QualificationConfig, merged_model_path: Path, output_path: Path
) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("formal preflight requires CUDA")
    assert_immutable_revision(config.teacher.revision)
    prompt = load_sealed_prompt(config.release_manifest)
    torch.cuda.reset_peak_memory_stats()
    student_tokenizer = AutoTokenizer.from_pretrained(merged_model_path, trust_remote_code=False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        config.teacher.model_id, revision=config.teacher.revision, trust_remote_code=False
    )
    samples = [
        "hyde: a hypothetical relevant passage\nvec: a semantic query<|im_end|>\n",
        "hyde:",
        "vec:",
        "\n",
    ]
    compatibility = tokenizer_compatibility(student_tokenizer, teacher_tokenizer, samples)
    if compatibility["status"] != "passed":
        raise RuntimeError("teacher/student tokenizer compatibility failed")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher.model_id,
        revision=config.teacher.revision,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=False,
    ).to("cuda")
    _assert_cuda_bf16(teacher)
    teacher.eval()
    probe_query = "how do interest rates affect bond prices"
    completion = "hyde: Bond prices generally move inversely to market interest rates.\nvec: bond prices interest rate relationship<|im_end|>\n"
    prompt_ids = student_tokenizer(prompt.render(probe_query), add_special_tokens=False)["input_ids"]
    completion_ids = student_tokenizer(completion, add_special_tokens=False)["input_ids"]
    if teacher_tokenizer(completion, add_special_tokens=False)["input_ids"] != completion_ids:
        raise RuntimeError("teacher re-encoding changed the student completion token path")
    token_log_probs = _score_completion(teacher, prompt_ids, completion_ids)
    result = {
        "schema_version": "qmd-teacher-qualification-preflight-v1",
        "status": "passed",
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "teacher": {"model_id": config.teacher.model_id, "revision": config.teacher.revision},
        "merged_model_path": str(merged_model_path),
        "tokenizer_compatibility": compatibility,
        "probe": {
            "query_sha256": hashlib.sha256(probe_query.encode()).hexdigest(),
            "completion_token_ids": completion_ids,
            "token_log_probs": token_log_probs,
            "all_finite": all(math.isfinite(value) for value in token_log_probs),
        },
        "chat_templates": {
            "student_sha256": hashlib.sha256(str(student_tokenizer.chat_template).encode()).hexdigest(),
            "teacher_sha256": hashlib.sha256(str(teacher_tokenizer.chat_template).encode()).hexdigest(),
        },
    }
    atomic_json(output_path, result)
    return result
