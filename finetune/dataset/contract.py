"""Pure Contract v1 validation for QMD query-expansion training targets.

This module owns training-target-v1 semantics.  It performs no file I/O and
does not load a tokenizer; callers must inject the pinned tokenizer's counter.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Mapping


CONTRACT_VERSION = "training-target-v1.1"
OUTPUT_TYPES = ("hyde", "lex", "vec")
_ONLY_MODE = re.compile(r"(?:/only:|\bonly:\s*)(?:lex|vec|hyde)\s*$")
_CHAT_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    quarantined: bool
    input_key: str | None
    canonical_record: dict[str, Any]
    errors: tuple[Diagnostic, ...]
    warnings: tuple[Diagnostic, ...]


TokenCounter = Callable[[str], int]


def _canonical_input(query: str, intent: str | None) -> bytes:
    value = {
        "query": query.strip(),
        "intent": intent.strip() if intent is not None else None,
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def compute_input_key(query: str, intent: str | None = None) -> str:
    """Return the exact identity hash defined by Contract v1."""
    return hashlib.sha256(_canonical_input(query, intent)).hexdigest()


def _comparison_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _has_forbidden_control(text: str) -> bool:
    return any(
        character not in "\r\n" and unicodedata.category(character) == "Cc"
        for character in text
    )


def _has_positive_lex_term(text: str) -> bool:
    # Quote balance is checked separately.  This tokenizer only needs to
    # distinguish negative terms from at least one positive word or phrase.
    tokens = re.findall(r'-?"[^"]*"|\S+', text)
    for token in tokens:
        if token.startswith("-"):
            continue
        candidate = token[1:-1] if token.startswith('"') else token
        if any(character.isalnum() or character == "_" for character in candidate):
            return True
    return False


def _diagnostic(code: str, path: str, message: str) -> Diagnostic:
    return Diagnostic(code=code, path=path, message=message)


def validate_training_target(
    record: Mapping[str, Any],
    token_counter: TokenCounter,
) -> ValidationResult:
    """Validate and deterministically canonicalize one raw training record."""
    errors: list[Diagnostic] = []
    warnings: list[Diagnostic] = []

    raw_query = record.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) else ""
    if not query:
        errors.append(_diagnostic("TRAIN_EMPTY_QUERY", "query", "query must not be empty"))
    elif "\r" in raw_query or "\n" in raw_query:
        errors.append(
            _diagnostic("TRAIN_MULTILINE_QUERY", "query", "query must be a single line")
        )

    raw_intent = record.get("intent")
    intent: str | None
    if raw_intent is None:
        intent = None
    elif not isinstance(raw_intent, str) or not raw_intent.strip():
        intent = "" if not isinstance(raw_intent, str) else raw_intent.strip()
        errors.append(
            _diagnostic("TRAIN_EMPTY_INTENT", "intent", "intent must be non-empty when present")
        )
    else:
        intent = raw_intent.strip()
        if "\r" in raw_intent or "\n" in raw_intent:
            errors.append(
                _diagnostic("TRAIN_MULTILINE_INTENT", "intent", "intent must be a single line")
            )

    quarantined = bool(query and _ONLY_MODE.search(query))
    if quarantined:
        errors.append(
            _diagnostic(
                "TRAIN_ONLY_MODE",
                "query",
                "only-mode records are quarantined from training-target-v1",
            )
        )

    raw_output = record.get("output")
    if not isinstance(raw_output, list) or not raw_output:
        errors.append(_diagnostic("TRAIN_EMPTY_OUTPUT", "output", "output must not be empty"))
        output_items: list[Any] = []
    else:
        output_items = raw_output

    canonical_items: list[list[str]] = []
    items_by_type: dict[str, list[tuple[int, str]]] = {kind: [] for kind in OUTPUT_TYPES}
    for index, item in enumerate(output_items):
        path = f"output[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            errors.append(
                _diagnostic("TRAIN_INVALID_OUTPUT_PAIR", path, "output item must be [type, text]")
            )
            continue
        kind, raw_text = item
        if kind not in OUTPUT_TYPES:
            errors.append(
                _diagnostic("TRAIN_INVALID_OUTPUT_TYPE", f"{path}[0]", "unknown output type")
            )
            continue
        if not isinstance(raw_text, str) or not raw_text.strip():
            errors.append(
                _diagnostic("TRAIN_EMPTY_OUTPUT_TEXT", f"{path}[1]", "output text must not be empty")
            )
            continue

        text = raw_text.strip()
        canonical_items.append([kind, text])
        items_by_type[kind].append((index, text))

        if "\r" in raw_text or "\n" in raw_text:
            errors.append(
                _diagnostic("TRAIN_MULTILINE_OUTPUT", f"{path}[1]", "output text must be a single line")
            )
        if _has_forbidden_control(raw_text):
            errors.append(
                _diagnostic(
                    "TRAIN_CONTROL_CHARACTER",
                    f"{path}[1]",
                    "output text contains a forbidden control character",
                )
            )
        if any(token in raw_text for token in _CHAT_TOKENS):
            errors.append(
                _diagnostic(
                    "TRAIN_CHAT_TEMPLATE_LEAK",
                    f"{path}[1]",
                    "output text contains a chat-template control token",
                )
            )

    if not quarantined:
        lex_count = len(items_by_type["lex"])
        vec_count = len(items_by_type["vec"])
        hyde_count = len(items_by_type["hyde"])
        if lex_count == 0:
            errors.append(_diagnostic("TRAIN_MISSING_LEX", "output", "one lex output is required"))
        elif lex_count > 3:
            errors.append(_diagnostic("TRAIN_TOO_MANY_LEX", "output", "at most three lex outputs are allowed"))
        if vec_count == 0:
            errors.append(_diagnostic("TRAIN_MISSING_VEC", "output", "one vec output is required"))
        elif vec_count > 3:
            errors.append(_diagnostic("TRAIN_TOO_MANY_VEC", "output", "at most three vec outputs are allowed"))
        if hyde_count > 1:
            errors.append(_diagnostic("TRAIN_TOO_MANY_HYDE", "output", "at most one hyde output is allowed"))

    for kind in OUTPUT_TYPES:
        seen: set[str] = set()
        for index, text in items_by_type[kind]:
            comparison = _comparison_text(text)
            if comparison in seen:
                errors.append(
                    _diagnostic(
                        "TRAIN_DUPLICATE_OUTPUT",
                        f"output[{index}]",
                        f"duplicate {kind} output after comparison normalization",
                    )
                )
            seen.add(comparison)

    for index, text in items_by_type["lex"]:
        path = f"output[{index}][1]"
        if text.count('"') % 2:
            errors.append(
                _diagnostic("TRAIN_LEX_UNMATCHED_QUOTE", path, "lex output has unmatched quotes")
            )
        elif not _has_positive_lex_term(text):
            errors.append(
                _diagnostic("TRAIN_LEX_NO_POSITIVE_TERM", path, "lex output needs a positive term")
            )
        token_count = token_counter(text)
        if token_count > 16:
            warnings.append(_diagnostic("WARN_LEX_TOO_LONG", path, "lex output exceeds 16 tokens"))

    for kind in ("vec", "hyde"):
        for index, text in items_by_type[kind]:
            path = f"output[{index}][1]"
            token_count = token_counter(text)
            if kind == "vec":
                if token_count < 4:
                    warnings.append(_diagnostic("WARN_VEC_TOO_SHORT", path, "vec output is under 4 tokens"))
                elif token_count > 48:
                    warnings.append(_diagnostic("WARN_VEC_TOO_LONG", path, "vec output exceeds 48 tokens"))
            else:
                if token_count > 128:
                    errors.append(
                        _diagnostic("TRAIN_HYDE_TOO_LONG", path, "HyDE output exceeds 128 tokens")
                    )
                elif token_count < 32:
                    warnings.append(
                        _diagnostic("WARN_HYDE_TOO_SHORT", path, "HyDE output is under 32 tokens")
                    )
    for kind in OUTPUT_TYPES:
        for index, text in items_by_type[kind]:
            normalized = _comparison_text(text)
            if "how to how to" in normalized:
                warnings.append(
                    _diagnostic(
                        "WARN_TEMPLATE_FINGERPRINT",
                        f"output[{index}][1]",
                        "output contains a known repeated template phrase",
                    )
                )
            if re.search(r"\b(?:latest|current version|ranking|score)\b|\b20\d{2}\b", normalized):
                warnings.append(
                    _diagnostic(
                        "WARN_TIME_SENSITIVE",
                        f"output[{index}][1]",
                        "output may contain time-sensitive claims",
                    )
                )

    canonical_items.sort(key=lambda pair: OUTPUT_TYPES.index(pair[0]))
    canonical_record: dict[str, Any] = {"query": query, "output": canonical_items}
    if raw_intent is not None:
        canonical_record["intent"] = intent

    input_key = compute_input_key(query, intent) if query else None
    return ValidationResult(
        valid=not errors,
        quarantined=quarantined,
        input_key=input_key,
        canonical_record=canonical_record,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
