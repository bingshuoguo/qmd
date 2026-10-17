#!/usr/bin/env python3
"""Deterministic greedy expansion generation for the frozen evaluation (spec section 16).

Emits raw model output only.  Parsing into the canonical expansion artifact is
done by materialize-expansions.ts so that base and SFT go through exactly the
same parser the rest of the benchmark pipeline uses.

The base and SFT arms differ in one thing: whether an adapter is loaded.  Prompt,
chat template, tokenizer, generation parameters and seed are shared by
construction. For sealed SFT releases, --release-manifest loads the exact
training template and validates the stored training/validation prompts. Legacy
templates remain explicitly selectable for historical comparisons.

Usage:
    uv run python -m benchmarks.public_eval.generate_expansions \
        --benchmark <benchmark-dir> --variant base --model <snapshot>
    ... --variant sft --model <snapshot> --adapter <run>/final-adapter
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset.public_distill import atomic_json, atomic_jsonl, sha256_file
from dataset.public_distill_v1 import (
    PROMPT_VERSION as V1_PROMPT_VERSION,
    TOKENIZER_REVISION,
    USER_PROMPT_TEMPLATE as V1_USER_TEMPLATE,
    expected_prompt as v1_expected_prompt,
    prompt_sha256 as v1_prompt_sha256,
)

from dataset.sft_release import (
    Release, assert_prompt_template, expected_prompt as sealed_expected_prompt, load_release,
)
from dataset.v2a_pareto_release import V2A_PROMPT_TEMPLATE


def load_evaluation_release(path: Path) -> Release:
    """Use the trainer's release validation and exact stored prompt contract."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    release = load_release(path, manifest["release_id"], manifest["experiment_id"])
    assert_prompt_template(release.train.records, release.prompt_template)
    assert_prompt_template(release.validation.records, release.prompt_template)
    return release


def runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "prefix": sys.prefix, "packages": packages}


# === Prompt registry ====================================================

V2_VH_USER_TEMPLATE = (
    "/no_think You generate retrieval expansions for a search query.\n"
    "Return only newline-separated lines in exactly this format:\n"
    "hyde: <one hypothetical document passage>\n"
    "vec: <semantic search query>\n"
    "vec: <optional second complementary semantic search query>\n"
    "Query: {query}"
)


PROMPT_REGISTRY = {
    "v1": {
        "version": V1_PROMPT_VERSION,
        "sha256": v1_prompt_sha256(),
        "template": V1_USER_TEMPLATE,
    },
    "v2-vh-legacy": {
        "version": "qmd-student-expansion-v2-vh-legacy-eval-v1",
        "template": V2_VH_USER_TEMPLATE,
        "sha256": hashlib.sha256(V2_VH_USER_TEMPLATE.encode("utf-8")).hexdigest(),
    },
    "v2a-pareto": {
        "version": "qmd-student-expansion-v3-vh-pareto-v1",
        "template": V2A_PROMPT_TEMPLATE,
        "sha256": hashlib.sha256(V2A_PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
    },
}


def expected_prompt(query: str, prompt_version: str = "v1", *, release: Release | None = None) -> str:
    """Render the Qwen chat envelope for a given prompt version."""
    if release is not None:
        return sealed_expected_prompt(release.prompt_template, query)
    if prompt_version == "v1":
        return v1_expected_prompt(query)
    template = PROMPT_REGISTRY[prompt_version]["template"]
    user_content = template.replace("{query}", query)
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def prompt_sha256(prompt_version: str = "v1") -> str:
    return PROMPT_REGISTRY[prompt_version]["sha256"]


def prompt_version_string(prompt_version: str = "v1") -> str:
    return PROMPT_REGISTRY[prompt_version]["version"]

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

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, dtype=torch.bfloat16
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate(
    model: Any, tokenizer: Any, queries: list[dict[str, str]], batch_size: int,
    prompt_version: str = "v1", *, release: Release | None = None,
) -> list[dict[str, Any]]:
    import torch

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or eos_token_id == []:
        raise ValueError("EOS-aware generation statistics require an EOS token ID")
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

    records: list[dict[str, Any]] = []
    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        prompts = [expected_prompt(item["query"], prompt_version, release=release) for item in batch]
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
                # Count the terminal EOS, but never the batch padding after it.
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
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--release-manifest", type=Path,
                        help="Use and verify the exact sealed training prompt (preferred for SFT).")
    prompt.add_argument("--prompt-version", choices=list(PROMPT_REGISTRY),
                        help="Legacy evaluation template; v2-vh-legacy is NOT the sealed training prompt.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.variant or Path(args.variant).name != args.variant or args.variant in (".", ".."):
        parser.error("--variant must be a filename component")
    release = load_evaluation_release(args.release_manifest) if args.release_manifest else None
    prompt_version = args.prompt_version or "v1"

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
    records = generate(model, tokenizer, queries, args.batch_size, prompt_version, release=release)

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
        "prompt_version": release.prompt_version if release else prompt_version_string(prompt_version),
        "prompt_sha256": release.prompt_sha256 if release else prompt_sha256(prompt_version),
        "prompt_template": release.prompt_template if release else PROMPT_REGISTRY[prompt_version]["template"],
        "release": release.provenance() if release else None,
        "generator_sha256": sha256_file(Path(__file__)),
        "environment": runtime_environment(),
        "generation": GENERATION,
        "generation_statistics_version": "eos-aware-v1",
        "generated_tokens_semantics": "includes terminal EOS; excludes post-EOS padding",
        "eos_token_id": tokenizer.eos_token_id,
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
