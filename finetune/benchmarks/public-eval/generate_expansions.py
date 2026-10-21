#!/usr/bin/env python3
"""Deterministic greedy expansion generation from a sealed SFT release.

Emits raw model output only. materialize-expansions.ts validates the raw file
and parses it through the same parser every other benchmark run uses.

The prompt always comes from --release-manifest: the exact template the
adapter was trained on, re-checked against every stored training prompt.
Base and SFT arms therefore differ only in whether --adapter is loaded.

Usage (from the repository root, in an environment from requirements-eval.txt):
    python finetune/benchmarks/public-eval/generate_expansions.py \
        --release-manifest <release>/experiments/<id>/release-manifest.json \
        --benchmark <benchmark-dir> --variant base --model <snapshot>
    ... --variant sft --model <snapshot> --adapter <run>/final-adapter
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

FINETUNE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FINETUNE_ROOT))

from dataset.artifact_io import atomic_json, atomic_jsonl, sha256_file  # noqa: E402
from dataset.sft_release import (  # noqa: E402
    Release, assert_prompt_template, expected_prompt, load_release,
)

GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 768,
    "seed": 42,
    "use_cache": True,
    "generation_padding_side": "left",
    "grammar": None,
}


def load_evaluation_release(path: Path) -> Release:
    release = load_release(path)
    assert_prompt_template(release.train.records, release.prompt_template)
    assert_prompt_template(release.validation.records, release.prompt_template)
    return release


def source_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=FINETUNE_ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
    return {"qmd_commit": git("rev-parse", "HEAD"), "qmd_dirty": bool(git("status", "--porcelain"))}


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable, "packages": packages}


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
    # Batched generation must pad on the left or continuations start at the wrong offset.
    tokenizer.padding_side = "left"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, dtype=torch.bfloat16)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate(
    model: Any, tokenizer: Any, queries: list[dict[str, str]], batch_size: int, prompt_template: str,
) -> list[dict[str, Any]]:
    import torch

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or eos_token_id == []:
        raise ValueError("EOS-aware generation statistics require an EOS token ID")
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

    records: list[dict[str, Any]] = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        prompts = [expected_prompt(prompt_template, item["query"]) for item in batch]
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
                eos_token_id=eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        for index, item in enumerate(batch):
            generated = outputs[index][prompt_length:].tolist()
            eos_index = next(
                (i for i, token_id in enumerate(generated) if token_id in eos_ids),
                None,
            )
            if eos_index is not None:
                # Count the terminal EOS, never the batch padding after it.
                # EOS at max_new_tokens is a normal stop, not a truncation.
                generated = generated[:eos_index + 1]
                finish_reason = "eos"
            elif len(generated) >= GENERATION["max_new_tokens"]:
                finish_reason = "length"
            else:
                finish_reason = "unknown"
            records.append({
                "qid": item["qid"],
                "query": item["query"],
                "raw_output": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_tokens": int(len(generated)),
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "generation_error": (
                    "generation stopped without EOS before the token budget"
                    if finish_reason == "unknown" else None
                ),
            })
        print(f"  {min(start + batch_size, len(queries))}/{len(queries)}", file=sys.stderr, flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-manifest", type=Path, required=True,
                        help="Sealed SFT release whose exact training prompt is used")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--variant", required=True, help="Arm name, e.g. base or sft")
    parser.add_argument("--model", required=True, help="Base model id or local snapshot")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.variant or Path(args.variant).name != args.variant or args.variant in (".", ".."):
        parser.error("--variant must be a filename component")

    release = load_evaluation_release(args.release_manifest)
    benchmark_dir = args.benchmark.resolve()
    output_dir = benchmark_dir / "raw-generations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.variant}.jsonl"
    manifest_path = output_dir / f"{args.variant}-manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise ValueError(f"raw generation already exists, refusing to overwrite: {output_path}")

    from transformers import set_seed

    set_seed(GENERATION["seed"])
    queries = load_queries(benchmark_dir)
    print(f"{args.variant}: {len(queries)} queries from {benchmark_dir.name}", file=sys.stderr)

    model, tokenizer = load_model(args.model, args.revision, args.adapter)
    records = generate(model, tokenizer, queries, args.batch_size, release.prompt_template)
    atomic_jsonl(output_path, records)

    adapter_weights = args.adapter / "adapter_model.safetensors" if args.adapter else None
    atomic_json(manifest_path, {
        "schema_version": "qmd-eval-raw-generation-v2",
        "benchmark_id": benchmark_dir.name,
        "variant": args.variant,
        "model": args.model,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": sha256_file(adapter_weights) if adapter_weights and adapter_weights.is_file() else None,
        "release": release.provenance(),
        **source_state(),
        "environment": runtime_environment(),
        "generation": GENERATION,
        "generation_statistics_version": "eos-aware-v1",
        "generated_tokens_semantics": "includes terminal EOS; excludes post-EOS padding",
        "eos_token_id": tokenizer.eos_token_id,
        "batch_size": args.batch_size,
        "queries_sha256": sha256_file(benchmark_dir / "queries.jsonl"),
        "rows": len(records),
        "truncated": sum(record["truncated"] for record in records),
        "raw_sha256": sha256_file(output_path),
    })

    print(f"Wrote {len(records)} generations to {output_path} "
          f"({sum(r['truncated'] for r in records)} hit the token budget)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
