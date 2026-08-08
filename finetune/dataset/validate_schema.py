#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Validate JSONL files against Contract v1 (the QMD training-target schema).

This is the lightweight, tokenizer-free checker. The authoritative rules live in
``dataset.contract``; this CLI simply applies them record-by-record and reports
every violation. Only-mode records (``TRAIN_ONLY_MODE``) are reported as
quarantined rather than as hard errors, since they are valid-but-scoped-out of
the default Contract v1 training target.

Token-length checks use an approximate whitespace counter here; run
``dataset.validate_contract`` for the authoritative pinned-tokenizer audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.contract import validate_training_target
from dataset.validate_contract import resolve_paths


def _naive_token_counter(text: str) -> int:
    return len(text.split())


def validate_file(path: Path) -> tuple[int, int, int]:
    """Return (total_lines, hard_error_count, quarantined_count)."""
    total = 0
    errors = 0
    quarantined = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{path}:{line_num}: invalid JSON ({e})")
                errors += 1
                continue
            if not isinstance(obj, dict):
                print(f"{path}:{line_num}: record must be a JSON object")
                errors += 1
                continue

            result = validate_training_target(obj, _naive_token_counter)
            if result.quarantined:
                quarantined += 1
            for diagnostic in result.errors:
                if diagnostic.code == "TRAIN_ONLY_MODE":
                    continue  # quarantine is reported separately, not a hard error
                print(
                    f"{path}:{line_num}: {diagnostic.code} "
                    f"({diagnostic.path}): {diagnostic.message}"
                )
                errors += 1

    return total, errors, quarantined


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QMD JSONL against Contract v1")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["finetune/data/*.jsonl"],
        help="JSONL files or glob patterns (default: finetune/data/*.jsonl)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    files = resolve_paths(args.paths, repo_root)
    if not files:
        print("No files found to validate.")
        return 1

    total_lines = 0
    total_errors = 0
    total_quarantined = 0
    for path in sorted(files):
        lines, errors, quarantined = validate_file(path)
        total_lines += lines
        total_errors += errors
        total_quarantined += quarantined
        status = "OK" if errors == 0 else f"{errors} error(s)"
        note = f", {quarantined} quarantined" if quarantined else ""
        print(f"{path}: {lines} lines, {status}{note}")

    if total_errors:
        print(
            f"\nValidation failed: {total_errors} error(s) across {total_lines} lines"
        )
        return 1

    print(
        f"\nValidation passed: {total_lines} lines checked, "
        f"{total_quarantined} quarantined (only-mode)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
