#!/usr/bin/env python3
"""
Typed data model, loader, and renderers for QMD training data.

Every JSONL file in data/ MUST conform to this format:

    {"query": "auth config", "output": [["hyde", "..."], ["lex", "..."], ["vec", "..."]]}

- query: non-empty string
- output: list of [type, text] pairs where type is "lex", "vec", or "hyde"
- Extra fields (category, intent, is_short, etc.) are allowed but ignored

Responsibility split
--------------------
This module is the *typed view + I/O + rendering* layer:

- ``TrainingExample`` / ``OutputPair`` give callers an ergonomic Pydantic model.
- ``load_examples()`` is the fail-fast loader used by training and analysis tools.
- The ``output_items_to_text`` / ``parse_output_text`` / ``normalize_output_items``
  helpers render and normalize expansion lines.

The **validation rules themselves live in ``dataset.contract``** (Contract v1),
which is the single source of truth for what a valid training target is.
``load_examples()`` delegates every record to ``contract.validate_training_target``
rather than re-implementing rules here, so the two can never drift apart.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Callable, Iterable

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    field_validator,
)

from dataset.contract import OUTPUT_TYPES, validate_training_target


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class OutputType(str, Enum):
    lex = "lex"
    vec = "vec"
    hyde = "hyde"


# The vocabulary and its canonical order are owned by Contract v1
# (dataset.contract.OUTPUT_TYPES); the enum exists so Pydantic can type it.
VALID_OUTPUT_TYPES = set(OUTPUT_TYPES)


class OutputPair(BaseModel):
    """A single expansion line: [type, text]."""

    type: OutputType
    text: str

    model_config = ConfigDict(frozen=True)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v

    def to_list(self) -> list[str]:
        return [self.type.value, self.text]


def _coerce_output_pairs(v: list) -> list[OutputPair]:
    """Accept [["lex", "..."], ...] from JSON and coerce to OutputPair list."""
    pairs = []
    for i, item in enumerate(v):
        if isinstance(item, OutputPair):
            pairs.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append(OutputPair(type=item[0], text=item[1]))
        else:
            raise ValueError(
                f"output[{i}] must be [type, text], got {item!r}"
            )
    return pairs


# ---------------------------------------------------------------------------
# Pydantic model — single source of truth for the JSONL schema
# ---------------------------------------------------------------------------

class TrainingExample(BaseModel):
    """One training example in the canonical JSONL format."""

    query: str
    output: Annotated[list[OutputPair], BeforeValidator(_coerce_output_pairs)]

    # Optional metadata — present in some files, ignored during training.
    category: str | None = None
    intent: str | None = None
    is_short: bool | None = None

    # Contract-derived metadata, populated by load_examples (not read from data).
    # quarantined: record is an only-mode target quarantined by Contract v1.
    # input_key: Contract v1 identity hash of (query, intent).
    quarantined: bool = False
    input_key: str | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v

    @field_validator("output")
    @classmethod
    def output_not_empty(cls, v: list[OutputPair]) -> list[OutputPair]:
        if not v:
            raise ValueError("output must not be empty")
        return v

    def output_as_lists(self) -> list[list[str]]:
        """Return output as list-of-lists for JSON serialization."""
        return [p.to_list() for p in self.output]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _naive_token_counter(text: str) -> int:
    """Approximate token count (whitespace split) for callers without a tokenizer.

    Only Contract v1's token-length checks are affected. The authoritative
    token-length measurement is the offline audit (dataset.validate_contract),
    which runs the pinned Qwen tokenizer. The training path passes the real
    tokenizer through ``load_examples(..., token_counter=...)``.
    """
    return len(text.split())


def load_examples(
    path: str | Path,
    token_counter: Callable[[str], int] | None = None,
) -> list[TrainingExample]:
    """Load and validate a JSONL file. Fails loudly on any invalid line.

    Validation is delegated to Contract v1 (``dataset.contract``), the single
    source of truth for training-target rules. A record quarantined by Contract
    v1 *only* for only-mode (``TRAIN_ONLY_MODE``) is not an error here — it is
    loaded with ``quarantined=True`` so callers (e.g. prepare_data) can decide
    whether to include it. Any other contract error is fatal.

    ``token_counter`` is forwarded to Contract v1 for its token-length checks;
    pass the real tokenizer's counter on the training path. When omitted, an
    approximate whitespace counter is used (see ``_naive_token_counter``).
    """
    path = Path(path)
    count = token_counter or _naive_token_counter
    examples: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_num}: record must be a JSON object")

            result = validate_training_target(obj, count)
            fatal = [d for d in result.errors if d.code != "TRAIN_ONLY_MODE"]
            if fatal:
                detail = "; ".join(f"{d.code} ({d.path}): {d.message}" for d in fatal)
                raise ValueError(f"{path}:{line_num}: {detail}")

            try:
                example = TrainingExample.model_validate(obj)
            except Exception as e:
                raise ValueError(f"{path}:{line_num}: {e}") from e
            example.quarantined = result.quarantined
            example.input_key = result.input_key
            examples.append(example)
    return examples


# ---------------------------------------------------------------------------
# Helpers (used by prepare_data.py, reward.py, and other tools)
# ---------------------------------------------------------------------------

def parse_output_text(text: str) -> list[list[str]]:
    """Parse prefixed output text into list pairs.

    >>> parse_output_text("lex: foo\\nvec: bar")
    [["lex", "foo"], ["vec", "bar"]]
    """
    items: list[list[str]] = []
    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("lex:"):
            items.append(["lex", line[4:].strip()])
        elif line.startswith("vec:"):
            items.append(["vec", line[4:].strip()])
        elif line.startswith("hyde:"):
            items.append(["hyde", line[5:].strip()])
    return items


def reorder_hyde_first(items: list[list[str]]) -> list[list[str]]:
    """Reorder items into Contract v1 canonical order (hyde, lex, vec)."""
    return [
        item
        for kind in OUTPUT_TYPES
        for item in items
        if item and item[0] == kind
    ]


def normalize_output_items(
    items: Iterable, hyde_first: bool = True
) -> list[list[str]]:
    """Normalize output pairs (filter invalid, trim whitespace, reorder).

    Accepts list[OutputPair] or list[list[str]].
    """
    normalized: list[list[str]] = []
    for item in items:
        if isinstance(item, OutputPair):
            normalized.append([item.type.value, item.text.strip()])
            continue
        if not item:
            continue
        try:
            kind, text = item[0], item[1]
        except Exception:
            continue
        if kind not in VALID_OUTPUT_TYPES:
            continue
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        normalized.append([kind, text])

    if hyde_first:
        normalized = reorder_hyde_first(normalized)

    return normalized


def output_items_to_text(
    items: Iterable, hyde_first: bool = True
) -> str:
    """Render output pairs to prefixed text lines.

    Accepts list[OutputPair] or list[list[str]].
    """
    normalized = normalize_output_items(items, hyde_first)
    return "\n".join(f"{kind}: {text}" for kind, text in normalized)


def has_type(items: Iterable, kind: str) -> bool:
    for item in items:
        if isinstance(item, OutputPair):
            if item.type.value == kind:
                return True
        elif item and item[0] == kind:
            return True
    return False
