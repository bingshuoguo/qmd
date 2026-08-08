#!/usr/bin/env python3
"""Report Contract v1 violations without modifying canonical source data."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from dataset.contract import CONTRACT_VERSION, validate_training_target


DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-1.7B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_label(path: Path, source_root: Path) -> str:
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _diagnostic(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _invalid_audit_row(
    label: str, line_number: int, code: str, message: str
) -> dict[str, Any]:
    return {
        "source_path": label,
        "line_number": line_number,
        "input_key": None,
        "status": "invalid",
        "errors": [_diagnostic(code, "$", message)],
        "warnings": [],
        "canonical_output": None,
    }


def audit_paths(
    paths: Iterable[Path],
    token_counter: Callable[[str], int],
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic row and summary reports for source paths."""
    rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        label = source_label(path, source_root)
        source_files.append(
            {"path": label, "sha256": sha256_file(path)}
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    row = _invalid_audit_row(
                        label,
                        line_number,
                        "TRAIN_INVALID_JSON",
                        f"invalid JSON: {error.msg}",
                    )
                    error_counts.update(item["code"] for item in row["errors"])
                    rows.append(row)
                    continue

                if not isinstance(value, dict):
                    row = _invalid_audit_row(
                        label,
                        line_number,
                        "TRAIN_INVALID_RECORD",
                        "record must be a JSON object",
                    )
                    error_counts.update(item["code"] for item in row["errors"])
                    rows.append(row)
                    continue

                result = validate_training_target(value, token_counter)
                errors = [asdict(item) for item in result.errors]
                warnings = [asdict(item) for item in result.warnings]
                error_counts.update(item["code"] for item in errors)
                warning_counts.update(item["code"] for item in warnings)
                status = (
                    "quarantined"
                    if result.quarantined
                    else "valid" if result.valid else "invalid"
                )
                rows.append(
                    {
                        "source_path": label,
                        "line_number": line_number,
                        "input_key": result.input_key,
                        "status": status,
                        "errors": errors,
                        "warnings": warnings,
                        "canonical_output": result.canonical_record["output"],
                    }
                )

    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "source_files": source_files,
        "records": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }
    return rows, summary


def write_report_directory(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Publish a complete report directory with one atomic rename."""
    if output_dir.exists():
        raise FileExistsError(f"report directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        rows_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
        (temporary / "violations.jsonl").write_text(rows_text, encoding="utf-8")
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def resolve_paths(patterns: list[str], repo_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        absolute_pattern = str(candidate if candidate.is_absolute() else repo_root / candidate)
        for match in glob.glob(absolute_pattern):
            path = Path(match)
            if path.is_file():
                paths.add(path.resolve())
    return sorted(paths, key=lambda item: item.as_posix())


def load_token_counter(
    model: str,
    revision: str,
    local_files_only: bool,
) -> Callable[[str], int]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("tokenizer revision must be a 40-character commit hash")
    from transformers import AutoTokenizer

    tokenizer_source = model
    if local_files_only:
        # Resolve the exact cached snapshot first.  Recent transformers may
        # still query Hub metadata even with local_files_only=True when given
        # a model id; a resolved snapshot path has no network ambiguity.
        from huggingface_hub import snapshot_download

        tokenizer_source = snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if local_files_only else revision,
        local_files_only=local_files_only,
    )

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["finetune/data/*.jsonl"],
        help="JSONL files or glob patterns relative to the repository root",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    paths = resolve_paths(args.paths, repo_root)
    if not paths:
        print("No source files matched; no report was written.")
        return 2
    try:
        token_counter = load_token_counter(
            args.tokenizer_model,
            args.tokenizer_revision,
            args.local_files_only,
        )
        rows, summary = audit_paths(paths, token_counter, repo_root)
        summary["tokenizer_model"] = args.tokenizer_model
        summary["tokenizer_revision"] = args.tokenizer_revision
        write_report_directory(args.output_dir.resolve(), rows, summary)
    except Exception as error:
        print(f"Contract audit failed: {error}")
        return 2

    invalid = summary["status_counts"].get("invalid", 0)
    quarantined = summary["status_counts"].get("quarantined", 0)
    print(
        f"Contract audit wrote {len(rows)} records to {args.output_dir}: "
        f"{invalid} invalid, {quarantined} quarantined"
    )
    return 1 if invalid or quarantined else 0


if __name__ == "__main__":
    raise SystemExit(main())
