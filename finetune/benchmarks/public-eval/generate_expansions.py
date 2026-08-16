#!/usr/bin/env python3
"""Deterministic greedy expansion generation for the frozen evaluation (spec section 16).

Emits raw model output only.  Parsing into the canonical expansion artifact is
done by materialize-expansions.ts so that base and SFT go through exactly the
same parser the rest of the benchmark pipeline uses.

The base and SFT arms differ in one thing: whether an adapter is loaded.  Prompt,
chat template, tokenizer, generation parameters and seed are shared by
construction -- the prompt comes from the same module the training data was
materialized with.

Usage:
    uv run python -m benchmarks.public_eval.generate_expansions \
        --benchmark <benchmark-dir> --variant base --model <snapshot>
    ... --variant sft --model <snapshot> --adapter <run>/final-adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.public_distill import atomic_json, atomic_jsonl, sha256_file
from dataset.public_distill_v1 import (
    PROMPT_VERSION,
    TOKENIZER_REVISION,
    expected_prompt,
    prompt_sha256,
)

# Spec section 16.  Frozen for both arms.
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 768,
    "seed": 42,
    "use_cache": True,
    "generation_padding_side": "left",
    "grammar": None,
}


def load_queries(benchmark_dir: Path) -> list[dict[str, str]]:
    queries = []
    for line in (benchmark_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        queries.append({"qid": record["qid"], "query": record["query"]})
    if not queries:
        raise ValueError(f"{benchmark_dir}: queries.jsonl is empty")
    return queries


def load_model(model_path: str, revision: str | None, adapter: Path | None) -> tuple[Any, Any]:
    """Load with plain Transformers -- the evaluation stack, never Unsloth."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    # Section 16: batched generation must pad on the left or the decoded
    # continuations start at the wrong offset.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, dtype=torch.bfloat16, device_map="auto"
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def generate(
    model: Any, tokenizer: Any, queries: list[dict[str, str]], batch_size: int
) -> list[dict[str, Any]]:
    import torch

    records: list[dict[str, Any]] = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        prompts = [expected_prompt(item["query"]) for item in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=GENERATION["do_sample"],
                num_beams=GENERATION["num_beams"],
                max_new_tokens=GENERATION["max_new_tokens"],
                use_cache=GENERATION["use_cache"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        for index, item in enumerate(batch):
            generated = outputs[index][prompt_length:]
            truncated = len(generated) >= GENERATION["max_new_tokens"]
            records.append({
                "qid": item["qid"],
                "query": item["query"],
                "raw_output": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_tokens": int(len(generated)),
                "truncated": bool(truncated),
            })
        print(
            f"  {min(start + batch_size, len(queries))}/{len(queries)}",
            file=sys.stderr,
            flush=True,
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--variant", required=True, help="Arm name, e.g. base or sft")
    parser.add_argument("--model", required=True, help="Base model id or local snapshot")
    parser.add_argument("--revision", default=TOKENIZER_REVISION)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    benchmark_dir = args.benchmark.resolve()
    output_dir = benchmark_dir / "raw-generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.variant}.jsonl"
    manifest_path = output_dir / f"{args.variant}-manifest.json"
    if output_path.exists():
        raise ValueError(f"raw generation already exists, refusing to overwrite: {output_path}")

    from transformers import set_seed

    set_seed(GENERATION["seed"])

    queries = load_queries(benchmark_dir)
    print(f"{args.variant}: {len(queries)} queries from {benchmark_dir.name}", file=sys.stderr)

    model, tokenizer = load_model(args.model, args.revision, args.adapter)
    records = generate(model, tokenizer, queries, args.batch_size)

    atomic_jsonl(output_path, records)
    atomic_json(manifest_path, {
        "schema_version": "qmd-eval-raw-generation-v1",
        "benchmark_id": benchmark_dir.name,
        "variant": args.variant,
        "model": args.model,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": (
            sha256_file(args.adapter / "adapter_model.safetensors")
            if args.adapter and (args.adapter / "adapter_model.safetensors").is_file()
            else None
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "generation": GENERATION,
        "batch_size": args.batch_size,
        "queries": len(queries),
        "truncated": sum(record["truncated"] for record in records),
        "raw_sha256": sha256_file(output_path),
    })

    truncated = sum(record["truncated"] for record in records)
    print(
        f"Wrote {len(records)} generations to {output_path} ({truncated} hit the token budget)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
