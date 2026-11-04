"""Pure Contract v1 validation for QMD query-expansion training targets.

This module owns training-target-v1 semantics.  It performs no file I/O and
does not load a tokenizer; callers must inject the pinned tokenizer's counter.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple


CONTRACT_VERSION = "training-target-v1.1"
OUTPUT_TYPES = ("hyde", "lex", "vec")
_TYPE_ORDER = {kind: index for index, kind in enumerate(OUTPUT_TYPES)}
_ONLY_MODE = re.compile(r"(?:/only:|\bonly:\s*)(?:lex|vec|hyde)\s*$")
_CHAT_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<think>",
    "</think>",
)
_TEMPLATE_FINGERPRINT = "how to how to"
_TIME_SENSITIVE = re.compile(r"\b(?:latest|current version|ranking|score)\b|\b20\d{2}\b")


class _TokenLimit(NamedTuple):
    """Contract v1 token-length bounds for one output type."""

    label: str  # type name as written in diagnostic messages
    minimum: int  # warn below this many tokens
    maximum: int  # above this many tokens: warning, or error if over_is_error
    over_is_error: bool


# These numbers are the contract; tune them here, never inside the checks.
_TOKEN_LIMITS = {
    "vec": _TokenLimit("vec", 4, 48, False),
    "hyde": _TokenLimit("HyDE", 32, 128, False),
}


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


@dataclass(frozen=True)
class _OutputItem:
    """One well-formed [type, text] output with its check inputs pre-computed."""

    kind: str
    index: int
    text: str  # stripped
    token_count: int
    comparison: str  # normalized for the duplicate and fingerprint checks


def _validate_query(record: Mapping[str, Any], errors: list[Diagnostic]) -> str:
    raw_query = record.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) else ""
    if not query:
        errors.append(Diagnostic("TRAIN_EMPTY_QUERY", "query", "query must not be empty"))
    elif "\r" in raw_query or "\n" in raw_query:
        errors.append(
            Diagnostic("TRAIN_MULTILINE_QUERY", "query", "query must be a single line")
        )
    return query


def _validate_intent(record: Mapping[str, Any], errors: list[Diagnostic]) -> str | None:
    raw_intent = record.get("intent")
    if raw_intent is None:
        return None
    if not isinstance(raw_intent, str) or not raw_intent.strip():
        errors.append(
            Diagnostic("TRAIN_EMPTY_INTENT", "intent", "intent must be non-empty when present")
        )
        return ""
    intent = raw_intent.strip()
    if "\r" in raw_intent or "\n" in raw_intent:
        errors.append(
            Diagnostic("TRAIN_MULTILINE_INTENT", "intent", "intent must be a single line")
        )
    return intent


def _parse_outputs(
    record: Mapping[str, Any],
    token_counter: TokenCounter,
    errors: list[Diagnostic],
) -> list[_OutputItem]:
    raw_output = record.get("output")
    if not isinstance(raw_output, list) or not raw_output:
        errors.append(Diagnostic("TRAIN_EMPTY_OUTPUT", "output", "output must not be empty"))
        return []

    items: list[_OutputItem] = []
    for index, raw_item in enumerate(raw_output):
        path = f"output[{index}]"
        if not isinstance(raw_item, (list, tuple)) or len(raw_item) != 2:
            errors.append(
                Diagnostic("TRAIN_INVALID_OUTPUT_PAIR", path, "output item must be [type, text]")
            )
            continue
        kind, raw_text = raw_item
        if kind not in OUTPUT_TYPES:
            errors.append(
                Diagnostic("TRAIN_INVALID_OUTPUT_TYPE", f"{path}[0]", "unknown output type")
            )
            continue
        if not isinstance(raw_text, str) or not raw_text.strip():
            errors.append(
                Diagnostic("TRAIN_EMPTY_OUTPUT_TEXT", f"{path}[1]", "output text must not be empty")
            )
            continue

        text = raw_text.strip()
        items.append(
            _OutputItem(
                kind=kind,
                index=index,
                text=text,
                token_count=token_counter(text),
                comparison=_comparison_text(text),
            )
        )

        if "\r" in raw_text or "\n" in raw_text:
            errors.append(
                Diagnostic("TRAIN_MULTILINE_OUTPUT", f"{path}[1]", "output text must be a single line")
            )
        if _has_forbidden_control(raw_text):
            errors.append(
                Diagnostic(
                    "TRAIN_CONTROL_CHARACTER",
                    f"{path}[1]",
                    "output text contains a forbidden control character",
                )
            )
        if any(token in raw_text for token in _CHAT_TOKENS):
            errors.append(
                Diagnostic(
                    "TRAIN_CHAT_TEMPLATE_LEAK",
                    f"{path}[1]",
                    "output text contains a chat-template control token",
                )
            )
    return items


def _check_counts(items: list[_OutputItem], errors: list[Diagnostic]) -> None:
    counts = Counter(item.kind for item in items)
    if counts["lex"] == 0:
        errors.append(Diagnostic("TRAIN_MISSING_LEX", "output", "one lex output is required"))
    elif counts["lex"] > 3:
        errors.append(Diagnostic("TRAIN_TOO_MANY_LEX", "output", "at most three lex outputs are allowed"))
    if counts["vec"] == 0:
        errors.append(Diagnostic("TRAIN_MISSING_VEC", "output", "one vec output is required"))
    elif counts["vec"] > 3:
        errors.append(Diagnostic("TRAIN_TOO_MANY_VEC", "output", "at most three vec outputs are allowed"))
    if counts["hyde"] > 1:
        errors.append(Diagnostic("TRAIN_TOO_MANY_HYDE", "output", "at most one hyde output is allowed"))


def _check_duplicates(items: list[_OutputItem], errors: list[Diagnostic]) -> None:
    for kind in OUTPUT_TYPES:
        seen: set[str] = set()
        for item in items:
            if item.kind != kind:
                continue
            if item.comparison in seen:
                errors.append(
                    Diagnostic(
                        "TRAIN_DUPLICATE_OUTPUT",
                        f"output[{item.index}]",
                        f"duplicate {kind} output after comparison normalization",
                    )
                )
            seen.add(item.comparison)


def _check_lex(
    items: list[_OutputItem],
    errors: list[Diagnostic],
    warnings: list[Diagnostic],
) -> None:
    for item in items:
        if item.kind != "lex":
            continue
        path = f"output[{item.index}][1]"
        if item.text.count('"') % 2:
            errors.append(
                Diagnostic("TRAIN_LEX_UNMATCHED_QUOTE", path, "lex output has unmatched quotes")
            )
        elif not _has_positive_lex_term(item.text):
            errors.append(
                Diagnostic("TRAIN_LEX_NO_POSITIVE_TERM", path, "lex output needs a positive term")
            )
        if item.token_count > 16:
            warnings.append(Diagnostic("WARN_LEX_TOO_LONG", path, "lex output exceeds 16 tokens"))


def _check_lengths(
    items: list[_OutputItem],
    errors: list[Diagnostic],
    warnings: list[Diagnostic],
) -> None:
    for kind in ("vec", "hyde"):
        limit = _TOKEN_LIMITS[kind]
        for item in items:
            if item.kind != kind:
                continue
            path = f"output[{item.index}][1]"
            if item.token_count < limit.minimum:
                warnings.append(
                    Diagnostic(
                        f"WARN_{kind.upper()}_TOO_SHORT",
                        path,
                        f"{limit.label} output is under {limit.minimum} tokens",
                    )
                )
            elif item.token_count > limit.maximum:
                severity = "TRAIN" if limit.over_is_error else "WARN"
                target = errors if limit.over_is_error else warnings
                target.append(
                    Diagnostic(
                        f"{severity}_{kind.upper()}_TOO_LONG",
                        path,
                        f"{limit.label} output exceeds {limit.maximum} tokens",
                    )
                )


def _check_fingerprints(items: list[_OutputItem], warnings: list[Diagnostic]) -> None:
    for kind in OUTPUT_TYPES:
        for item in items:
            if item.kind != kind:
                continue
            path = f"output[{item.index}][1]"
            if _TEMPLATE_FINGERPRINT in item.comparison:
                warnings.append(
                    Diagnostic(
                        "WARN_TEMPLATE_FINGERPRINT",
                        path,
                        "output contains a known repeated template phrase",
                    )
                )
            if _TIME_SENSITIVE.search(item.comparison):
                warnings.append(
                    Diagnostic(
                        "WARN_TIME_SENSITIVE",
                        path,
                        "output may contain time-sensitive claims",
                    )
                )


def validate_training_target(
    record: Mapping[str, Any],
    token_counter: TokenCounter,
) -> ValidationResult:
    """Validate and deterministically canonicalize one raw training record."""
    errors: list[Diagnostic] = []
    warnings: list[Diagnostic] = []

    query = _validate_query(record, errors)
    intent = _validate_intent(record, errors)

    quarantined = bool(query and _ONLY_MODE.search(query))
    if quarantined:
        errors.append(
            Diagnostic(
                "TRAIN_ONLY_MODE",
                "query",
                "only-mode records are quarantined from training-target-v1",
            )
        )

    items = _parse_outputs(record, token_counter, errors)
    if not quarantined:
        _check_counts(items, errors)
    _check_duplicates(items, errors)
    _check_lex(items, errors, warnings)
    _check_lengths(items, errors, warnings)
    _check_fingerprints(items, warnings)

    canonical_items = [[item.kind, item.text] for item in items]
    canonical_items.sort(key=lambda pair: _TYPE_ORDER[pair[0]])
    canonical_record: dict[str, Any] = {"query": query, "output": canonical_items}
    if intent is not None:
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
