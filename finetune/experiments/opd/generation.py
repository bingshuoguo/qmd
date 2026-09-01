from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .contracts import (
    GenerationConfig,
    ModelConfig,
    PromptContract,
    atomic_json,
    load_config,
    load_sealed_prompt,
    sha256_path,
)


def finish_from_tokens(
    token_ids: list[int], eos_ids: set[int], budget: int
) -> tuple[list[int], str, bool]:
    eos_index = next((index for index, token_id in enumerate(token_ids) if token_id in eos_ids), None)
    if eos_index is not None:
        return token_ids[: eos_index + 1], "eos", False
    if len(token_ids) >= budget:
        return token_ids[:budget], "length", True
    return token_ids, "unknown", False


def load_cuda_model(model: ModelConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("formal generation requires CUDA; CPU/MPS fallback is forbidden")
    local_model = Path(model.model_id).is_dir()
    if not model.revision and not local_model:
        raise ValueError("formal generation requires an immutable model revision")
    tokenizer = AutoTokenizer.from_pretrained(model.model_id, revision=model.revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    loaded = AutoModelForCausalLM.from_pretrained(
        model.model_id,
        revision=model.revision,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=False,
    )
    loaded = loaded.to("cuda")
    if model.adapter is not None:
        from peft import PeftModel

        loaded = PeftModel.from_pretrained(loaded, str(model.adapter))
    if any(parameter.device.type != "cuda" for parameter in loaded.parameters()):
        raise RuntimeError("model contains non-CUDA parameters; offload is forbidden")
    if any(parameter.is_floating_point() and parameter.dtype != torch.bfloat16 for parameter in loaded.parameters()):
        raise RuntimeError("model contains floating parameters outside BF16")
    loaded.eval()
    return loaded, tokenizer


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_arm(
    *,
    model_config: ModelConfig,
    generation: GenerationConfig,
    prompt: PromptContract,
    queries: list[dict[str, Any]],
    output_path: Path,
    arm: str,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    existing = _load_rows(output_path)
    expected_keys = [str(row["sample_key"]) for row in queries]
    existing_keys = [str(row.get("sample_key")) for row in existing]
    if existing_keys != expected_keys[: len(existing_keys)]:
        raise ValueError("generation output is not an exact sample-key prefix")
    model, tokenizer = load_cuda_model(model_config)
    eos_value = tokenizer.eos_token_id
    if eos_value is None:
        raise ValueError("tokenizer has no EOS token")
    eos_ids = {eos_value} if isinstance(eos_value, int) else set(eos_value)
    set_seed(generation.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = list(existing)
    with output_path.open("a", encoding="utf-8") as handle:
        for start in range(len(existing), len(queries), batch_size):
            batch = queries[start : start + batch_size]
            prompts = [prompt.render(str(row["query"])) for row in batch]
            prompt_lengths = [
                len(tokenizer(value, add_special_tokens=False)["input_ids"]) for value in prompts
            ]
            if any(length > generation.max_prompt_tokens for length in prompt_lengths):
                raise ValueError("rendered prompt exceeds the frozen 512-token input budget")
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded,
                        do_sample=generation.do_sample,
                        num_beams=generation.num_beams,
                        max_new_tokens=generation.max_new_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_value,
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                padded_prompt_length = encoded["input_ids"].shape[1]
                batch_records: list[dict[str, Any]] = []
                for index, row in enumerate(batch):
                    raw_ids = outputs[index][padded_prompt_length:].tolist()
                    token_ids, finish_reason, truncated = finish_from_tokens(
                        raw_ids, eos_ids, generation.max_new_tokens
                    )
                    record = {
                        "schema_version": "qmd-teacher-qualification-generation-v1",
                        "arm": arm,
                        "sample_key": row["sample_key"],
                        "source_id": row["source_id"],
                        "qid": str(row["qid"]),
                        "query": row["query"],
                        "attempt": 1,
                        "selected_attempt": True,
                        "raw_output": tokenizer.decode(token_ids, skip_special_tokens=True),
                        "completion_token_ids": token_ids,
                        "generated_tokens": len(token_ids),
                        "finish_reason": finish_reason,
                        "truncated": truncated,
                        "generation_error": (
                            "generation stopped without EOS before the token budget"
                            if finish_reason == "unknown"
                            else None
                        ),
                        "elapsed_ms": elapsed_ms / len(batch),
                        "synthetic": False,
                    }
                    batch_records.append(record)
            except Exception as error:
                elapsed_ms = (time.perf_counter() - started) * 1000
                batch_records = [
                    {
                        "schema_version": "qmd-teacher-qualification-generation-v1",
                        "arm": arm,
                        "sample_key": row["sample_key"],
                        "source_id": row["source_id"],
                        "qid": str(row["qid"]),
                        "query": row["query"],
                        "attempt": 1,
                        "selected_attempt": True,
                        "raw_output": "",
                        "completion_token_ids": [],
                        "generated_tokens": 0,
                        "finish_reason": "error",
                        "truncated": False,
                        "generation_error": f"{type(error).__name__}: {error}",
                        "elapsed_ms": elapsed_ms / len(batch),
                        "synthetic": False,
                    }
                    for row in batch
                ]
            for record in batch_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                completed.append(record)
    manifest = {
        "schema_version": "qmd-teacher-qualification-generation-manifest-v1",
        "arm": arm,
        "model_id": model_config.model_id,
        "revision": model_config.revision,
        "adapter_sha256": model_config.adapter_sha256,
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "generation": {
            "do_sample": generation.do_sample,
            "num_beams": generation.num_beams,
            "max_new_tokens": generation.max_new_tokens,
            "max_model_len": generation.max_model_len,
            "seed": generation.seed,
        },
        "expected": len(queries),
        "completed": len(completed),
        "output_sha256": sha256_path(output_path),
    }
    atomic_json(output_path.with_name(f"{output_path.stem}-manifest.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one teacher qualification arm")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("merged-v2", "teacher"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config, mode="formal")
    queries = _load_rows(args.queries)
    model = config.merged_v2 if args.arm == "merged-v2" else config.teacher
    generate_arm(
        model_config=model,
        generation=config.generation,
        prompt=load_sealed_prompt(config.release_manifest),
        queries=queries,
        output_path=args.output,
        arm=args.arm,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
