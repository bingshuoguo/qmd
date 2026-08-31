from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .contracts import atomic_json, atomic_jsonl


LINE = re.compile(r"(hyde|vec):[ \t]+(\S(?:.*\S)?)$")
ARMS = ("merged-v2", "teacher")


def validate_vh(raw: str) -> list[tuple[str, str]]:
    if not raw:
        raise ValueError("empty completion")
    lines = raw.splitlines()
    if not lines:
        raise ValueError("empty completion")
    parsed: list[tuple[str, str]] = []
    for line_number, line in enumerate(lines, 1):
        match = LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"line {line_number} is not a strict hyde/vec line")
        parsed.append((match.group(1), match.group(2)))
    if parsed[0][0] != "hyde" or sum(kind == "hyde" for kind, _ in parsed) != 1:
        raise ValueError("output must contain exactly one leading hyde line")
    vec_count = sum(kind == "vec" for kind, _ in parsed)
    if vec_count not in {1, 2} or any(kind != "vec" for kind, _ in parsed[1:]):
        raise ValueError("output must contain one or two vec lines after hyde")
    return parsed


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_generation_files(
    *,
    queries_path: Path,
    generation_paths: dict[str, Path],
    output_root: Path,
) -> dict[str, Any]:
    queries = _rows(queries_path)
    expected = [str(row["sample_key"]) for row in queries]
    query_by_key = {str(row["sample_key"]): row for row in queries}
    failures: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    expansions_root = output_root / "expansions"
    for arm in ARMS:
        path = generation_paths[arm]
        records = _rows(path)
        keys = [str(row.get("sample_key")) for row in records]
        if keys != expected:
            failures.append(
                {
                    "arm": arm,
                    "error_type": "coverage_mismatch",
                    "error": f"expected {len(expected)} ordered keys, observed {len(keys)}",
                }
            )
        canonical: list[dict[str, Any]] = []
        valid = 0
        for record in records:
            key = str(record.get("sample_key"))
            error: str | None = None
            parsed: list[tuple[str, str]] = []
            if record.get("generation_error"):
                error = str(record["generation_error"])
            elif record.get("truncated") is True:
                error = "generation reached the token budget without EOS"
            elif record.get("finish_reason") != "eos":
                error = f"invalid finish reason {record.get('finish_reason')!r}"
            else:
                try:
                    parsed = validate_vh(str(record.get("raw_output", "")))
                except ValueError as exception:
                    error = str(exception)
            if key not in query_by_key:
                error = "generation sample key is not in the frozen query set"
            if error is not None:
                failures.append(
                    {
                        "arm": arm,
                        "sample_key": key,
                        "source_id": record.get("source_id"),
                        "qid": record.get("qid"),
                        "error_type": "invalid_generation",
                        "error": error,
                        "raw_output": record.get("raw_output", ""),
                    }
                )
                continue
            valid += 1
            query = query_by_key[key]
            canonical.append(
                {
                    "sample_key": key,
                    "source_id": query["source_id"],
                    "qid": str(query["qid"]),
                    "query": query["query"],
                    "status": "ok",
                    "raw_output": record["raw_output"],
                    "output": [list(item) for item in parsed],
                    "fallback_used": False,
                    "error": None,
                }
            )
        atomic_jsonl(expansions_root / f"{arm}.jsonl", canonical)
        arms[arm] = {
            "expected": len(expected),
            "completed": len(records),
            "valid": valid,
            "invalid": len(records) - valid + (0 if keys == expected else 1),
        }
    result = {
        "schema_version": "qmd-teacher-qualification-validation-v1",
        "status": "valid" if not failures and all(value["valid"] == len(expected) for value in arms.values()) else "invalid",
        "arms": arms,
        "failure_count": len(failures),
    }
    atomic_json(output_root / "validation.json", result)
    atomic_jsonl(output_root / "failures.jsonl", failures)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly validate teacher qualification generations")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--merged-v2", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.fixture:
        queries = args.fixture / "queries.jsonl"
        paths = {arm: args.fixture / "generations" / f"{arm}.jsonl" for arm in ARMS}
        output = args.fixture / "validation-output"
    else:
        if not all((args.queries, args.merged_v2, args.teacher, args.output)):
            parser.error("provide --fixture or all explicit input/output paths")
        queries = args.queries
        paths = {"merged-v2": args.merged_v2, "teacher": args.teacher}
        output = args.output
    result = validate_generation_files(queries_path=queries, generation_paths=paths, output_root=output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
