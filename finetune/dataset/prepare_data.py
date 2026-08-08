#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.45.0",
#     "pydantic>=2.0",
#     "jinja2",
# ]
# ///
"""Prepare QMD query expansion data for training.

Loads all data/*.jsonl via the strict Pydantic schema, applies the Qwen3
chat template, deduplicates by query, and writes train/val splits.

The prepared train files are ephemeral build artifacts — the canonical
data lives in data/*.jsonl and is always loaded through the schema.
"""

import argparse
import glob
import json
import random
import os
import sys
from pathlib import Path

from dataset.schema import (
    TrainingExample,
    load_examples,
    output_items_to_text,
)
from dataset.completion import split_rendered_text

from transformers import AutoTokenizer

_tokenizer = None
_tokenizer_model = None


def get_tokenizer():
    global _tokenizer, _tokenizer_model
    model_name = os.environ.get("QMD_BASE_MODEL", "Qwen/Qwen3-1.7B")
    if _tokenizer is None or _tokenizer_model != model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_model = model_name
    return _tokenizer


def get_token_counter():
    """Return a Contract v1 token counter backed by the real Qwen tokenizer."""
    tokenizer = get_tokenizer()
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def filter_only_mode(examples: list[TrainingExample], include_only_mode: bool) -> list[TrainingExample]:
    """Include or exclude Contract v1 only-mode (quarantined) records.

    Only-mode records (query carries a "/only:lex|vec|hyde" directive) are
    flagged by ``load_examples`` via ``example.quarantined``. They are valid but
    scoped out of the default Contract v1 training target.
    """
    if include_only_mode:
        return list(examples)
    return [ex for ex in examples if not ex.quarantined]


def format_for_training(ex: TrainingExample) -> dict:
    """Format a validated TrainingExample for SFT training."""
    tokenizer = get_tokenizer()
    output_text = output_items_to_text(ex.output)

    user_prompt = f"/no_think Expand this search query: {ex.query}"
    if ex.intent:
        user_prompt = (
            f"/no_think Expand this search query: {ex.query}\n"
            f"Query intent: {ex.intent.strip()}"
        )

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
        {"role": "assistant", "content": output_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Strip empty <think> tags — /no_think should suppress them
    text = text.replace("<think>\n\n</think>\n\n", "")
    prompt, completion = split_rendered_text(text)

    return {
        "query": ex.query,
        "output": ex.output_as_lists(),
        "prompt": prompt,
        "completion": completion,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--input",
        type=str,
        default="data/*.jsonl",
        help="Input JSONL file(s) - supports glob patterns",
    )
    parser.add_argument(
        "--output", type=str, default="data/train", help="Output directory"
    )
    parser.add_argument(
        "--split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Shuffle seed",
    )
    parser.add_argument(
        "--only-mode",
        choices=["include", "exclude"],
        default="exclude",
        help=(
            "Whether to keep Contract v1 only-mode records (queries with a "
            "'/only:lex|vec|hyde' directive). Default 'exclude' matches the "
            "Contract v1 training target; 'include' restores the legacy behavior "
            "of training on them."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve input files (relative to the current directory, unlike the
    # repo-root-relative audit tools: this script's default is finetune-local)
    if "*" in args.input:
        input_files = sorted(glob.glob(args.input))
        if not input_files:
            print(f"Error: No files found matching: {args.input}")
            sys.exit(1)
        print(f"Found {len(input_files)} input files")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        input_files = [str(input_path)]

    # Load all examples through Contract v1 (via the schema loader), using the
    # real tokenizer for token-length checks.
    token_counter = get_token_counter()
    all_examples: list[TrainingExample] = []
    for input_file in input_files:
        examples = load_examples(input_file, token_counter=token_counter)
        print(f"  {Path(input_file).name}: {len(examples)} examples")
        all_examples.extend(examples)

    print(f"Loaded {len(all_examples)} examples total")
    if not all_examples:
        print("No examples loaded; nothing to prepare.")
        sys.exit(1)

    # Include or exclude Contract v1 only-mode (quarantined) records.
    kept = filter_only_mode(all_examples, include_only_mode=(args.only_mode == "include"))
    dropped = len(all_examples) - len(kept)
    if dropped:
        print(f"only-mode {args.only_mode}: dropped {dropped} quarantined records")
    all_examples = kept

    # Deduplicate by query.  This intentionally uses a looser key than
    # Contract v1 identity (compute_input_key): queries dedup
    # case-insensitively so near-identical queries cannot leak across the
    # train/val split.  Intent stays in the key because it changes the
    # rendered prompt.
    seen: set[tuple[str, str]] = set()
    deduped: list[TrainingExample] = []
    for ex in all_examples:
        key = (ex.query.lower().strip(), (ex.intent or "").strip())
        if key not in seen:
            seen.add(key)
            deduped.append(ex)
    if len(deduped) < len(all_examples):
        print(f"Deduplicated: {len(all_examples)} -> {len(deduped)}")
    all_examples = deduped

    # Shuffle
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Format each example using the Pydantic model
    formatted = [format_for_training(ex) for ex in all_examples]

    # Split
    split_idx = int(len(formatted) * (1 - args.split))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    # Write (these are ephemeral build artifacts)
    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        with open(output_dir / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    # Stats
    short_final = sum(1 for ex in all_examples if len(ex.query.split()) <= 2)
    print(f"\n=== Summary ===")
    print(f"Total examples: {len(all_examples)}")
    print(f"Short queries: {short_final} ({100 * short_final / len(all_examples):.1f}%)")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")

    dataset_info = {
        "dataset_name": "qmd-query-expansion",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "short_query_pct": round(100 * short_final / len(all_examples), 1),
        "columns": ["prompt", "completion"],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)


if __name__ == "__main__":
    main()
