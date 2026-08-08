"""Contract v1 fixture and invariant tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.contract import compute_input_key, validate_training_target
from dataset.build_conflict_ledger import build_ledger, write_ledger
from dataset.validate_contract import (
    audit_paths,
    load_token_counter,
    write_report_directory,
)


FIXTURE_PATH = (
    Path(__file__).parent.parent
    / "fixtures"
    / "query-expansion-contract-v1.json"
)


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FixtureTokenCounter:
    def __init__(self, counts: dict[str, int]):
        self.counts = counts

    def __call__(self, text: str) -> int:
        return self.counts.get(text, len(text.split()))


class ContractFixtureTests(unittest.TestCase):
    def test_training_fixture_expectations(self):
        for case in load_fixture()["cases"]:
            with self.subTest(case=case["name"]):
                record = {
                    "query": case.get("query"),
                    "output": case["training_record"]["output"],
                }
                if "intent" in case:
                    record["intent"] = case["intent"]

                result = validate_training_target(
                    record,
                    FixtureTokenCounter(case.get("token_counts", {})),
                )
                expected = case["training"]
                self.assertEqual(result.valid, expected["valid"])
                self.assertEqual(result.quarantined, expected["quarantined"])
                self.assertEqual(
                    [diagnostic.code for diagnostic in result.errors],
                    expected["errors"],
                )
                if "warnings" in expected:
                    self.assertEqual(
                        [diagnostic.code for diagnostic in result.warnings],
                        expected["warnings"],
                    )
                if "canonical_output" in expected:
                    self.assertEqual(
                        result.canonical_record["output"],
                        expected["canonical_output"],
                    )
                if "input_key" in case:
                    self.assertEqual(result.input_key, case["input_key"])

    def test_input_key_preserves_identity_significant_text(self):
        base = compute_input_key("C++ 内存管理", "比较 Bob 的方案")

        self.assertEqual(
            base,
            compute_input_key("  C++ 内存管理  ", "  比较 Bob 的方案  "),
        )
        self.assertNotEqual(base, compute_input_key("c++ 内存管理", "比较 Bob 的方案"))
        self.assertNotEqual(base, compute_input_key("C++  内存管理", "比较 Bob 的方案"))
        self.assertNotEqual(base, compute_input_key("C++ 内存管理", None))

    def test_diagnostics_are_structured(self):
        result = validate_training_target(
            {"query": "auth", "output": [["lex", "auth"]]},
            FixtureTokenCounter({}),
        )

        self.assertEqual(result.errors[0].code, "TRAIN_MISSING_VEC")
        self.assertEqual(result.errors[0].path, "output")


class ContractAuditTests(unittest.TestCase):
    def test_tokenizer_revision_must_be_an_immutable_commit(self):
        with self.assertRaisesRegex(ValueError, "40-character commit hash"):
            load_token_counter("Qwen/Qwen3-1.7B", "main", True)

    def test_audit_is_read_only_and_reports_every_nonempty_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "query": "auth",
                                "output": [
                                    ["lex", "auth"],
                                    ["vec", "how to configure authentication"],
                                ],
                            }
                        ),
                        "{not json}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            before = source.read_bytes()

            rows, summary = audit_paths(
                [source], FixtureTokenCounter({}), root
            )

            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["status"], "valid")
            self.assertEqual(rows[1]["errors"][0]["code"], "TRAIN_INVALID_JSON")
            self.assertEqual(summary["records"], 2)

    def test_report_directory_is_complete_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "report"
            rows = [{"source_path": "source.jsonl", "line_number": 1}]
            summary = {"records": 1}

            write_report_directory(output_dir, rows, summary)

            self.assertTrue((output_dir / "violations.jsonl").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            with self.assertRaises(FileExistsError):
                write_report_directory(output_dir, rows, summary)


class ConflictLedgerTests(unittest.TestCase):
    def test_ledger_classifies_equivalent_and_real_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            records = [
                {
                    "query": "identical",
                    "output": [["lex", "identical"], ["vec", "how to find identical records"]],
                },
                {
                    "query": "identical",
                    "output": [["lex", "identical"], ["vec", "how to find identical records"]],
                },
                {
                    "query": "format",
                    "output": [["vec", "how to normalize format records"], ["lex", "format"]],
                },
                {
                    "query": "format",
                    "output": [["lex", " format "], ["vec", "how to normalize format records"]],
                },
                {
                    "query": "conflict",
                    "output": [["lex", "conflict"], ["vec", "how to find conflict records"]],
                },
                {
                    "query": "conflict",
                    "output": [["lex", "conflict target"], ["vec", "how to resolve conflict records"]],
                },
                {
                    "query": "mixed",
                    "output": [["lex", "mixed"], ["vec", "how to find mixed records"]],
                },
                {
                    "query": "mixed",
                    "output": [["lex", "\"mixed"], ["vec", "how to find mixed records"]],
                },
            ]
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            ledger = build_ledger([source], FixtureTokenCounter({}), root)
            by_query = {entry["query"]: entry for entry in ledger["entries"]}

            self.assertEqual(by_query["identical"]["classification"], "identical")
            self.assertEqual(by_query["format"]["classification"], "order_or_format_only")
            self.assertEqual(by_query["conflict"]["classification"], "target_conflict")
            self.assertEqual(by_query["conflict"]["decision"], "unresolved")
            self.assertEqual(by_query["mixed"]["classification"], "target_conflict")
            self.assertEqual(
                sorted(candidate["valid"] for candidate in by_query["mixed"]["candidates"]),
                [False, True],
            )
            self.assertEqual(ledger["unresolved_count"], 2)

    def test_ledger_writer_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ledger.json"
            write_ledger(path, {"ledger_version": "test"})

            with self.assertRaises(FileExistsError):
                write_ledger(path, {"ledger_version": "replacement"})


if __name__ == "__main__":
    unittest.main()
