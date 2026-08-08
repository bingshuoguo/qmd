"""Completion-only dataset helpers for QMD supervised fine-tuning."""

from __future__ import annotations

from typing import Any


ASSISTANT_MARKER = "<|im_start|>assistant\n"


def split_rendered_text(text: str) -> tuple[str, str]:
    """Split rendered Qwen chat text immediately after the assistant header."""
    occurrences = text.count(ASSISTANT_MARKER)
    if occurrences != 1:
        raise ValueError(
            "Expected exactly one Qwen assistant header, "
            f"found {occurrences}."
        )

    boundary = text.index(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
    prompt, completion = text[:boundary], text[boundary:]
    if not prompt or not completion:
        raise ValueError("Prompt and completion must both be non-empty.")
    return prompt, completion


def completion_fields(example: dict[str, Any]) -> tuple[str, str]:
    """Read the current prompt/completion contract or migrate a legacy text record."""
    prompt = example.get("prompt")
    completion = example.get("completion")
    if isinstance(prompt, str) and prompt and isinstance(completion, str) and completion:
        return prompt, completion

    text = example.get("text")
    if isinstance(text, str) and text:
        return split_rendered_text(text)

    raise ValueError(
        "Example must contain non-empty prompt/completion fields "
        "or a legacy rendered text field."
    )


def tokenize_completion_example(
    example: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """Preserve the rendered sequence and mark only completion tokens for loss."""
    prompt, completion = completion_fields(example)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokenization is not a prefix of prompt + completion tokenization."
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("Completion has no token after tokenization.")
    if len(input_ids) > max_length:
        raise ValueError(
            f"Example has {len(input_ids)} tokens, exceeding max_length={max_length}; "
            "refusing to truncate completion-only supervision."
        )

    return {
        "input_ids": input_ids,
        "completion_mask": [0] * len(prompt_ids)
        + [1] * (len(input_ids) - len(prompt_ids)),
    }


def prepare_completion_dataset(
    dataset: Any, tokenizer: Any, max_length: int, name: str
) -> Any:
    """Tokenize a dataset into the exact sequence and its completion mask."""
    return dataset.map(
        tokenize_completion_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {name} dataset",
    )
