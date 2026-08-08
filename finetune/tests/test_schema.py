"""Tests for the schema <-> contract delegation and the only-mode switch.

schema.py is the typed model + I/O + rendering layer; dataset.contract is the
single source of truth for validation rules. These tests pin the delegation:
load_examples enforces Contract v1 hard rules, flags (but does not reject)
only-mode records, and prepare_data.filter_only_mode applies the switch.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset.prepare_data import filter_only_mode
from dataset.schema import load_examples


def _write_jsonl(records: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.close()
    return Path(handle.name)


FULL_RECORD = {
    "query": "auth config",
    "output": [
        ["hyde", "Authentication is configured via the AUTH_SECRET environment variable."],
        ["lex", "authentication configuration"],
        ["vec", "how to configure authentication settings"],
    ],
}

ONLY_MODE_RECORD = {
    "query": "knife skills basics /only:vec",
    "output": [["vec", "how to knife skills basics"]],
}

# No lex line -> TRAIN_MISSING_LEX is a hard Contract v1 error.
MISSING_LEX_RECORD = {
    "query": "auth setup",
    "output": [["vec", "how to set up authentication"]],
}


class LoadExamplesDelegationTests(unittest.TestCase):
    def test_loads_valid_record_with_contract_metadata(self):
        path = _write_jsonl([FULL_RECORD])
        (example,) = load_examples(path)
        self.assertEqual(example.query, "auth config")
        self.assertFalse(example.quarantined)
        self.assertIsNotNone(example.input_key)

    def test_only_mode_is_loaded_and_flagged_not_rejected(self):
        path = _write_jsonl([ONLY_MODE_RECORD])
        (example,) = load_examples(path)
        self.assertTrue(example.quarantined)

    def test_hard_contract_violation_raises(self):
        path = _write_jsonl([MISSING_LEX_RECORD])
        with self.assertRaisesRegex(ValueError, "TRAIN_MISSING_LEX"):
            load_examples(path)

    def test_delegation_matches_contract_on_real_data(self):
        # load_examples must agree with a direct Contract v1 run on every real
        # record: no hard (non-only-mode) errors anywhere, and the quarantine
        # flags match one-for-one. Data-driven, so it survives data growth.
        from dataset.contract import validate_training_target

        data_dir = Path(__file__).parent.parent / "data"
        naive = lambda text: len(text.split())
        checked = 0
        for path in sorted(data_dir.glob("*.jsonl")):
            examples = load_examples(path)  # raises if any hard contract error
            expected_quarantined = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    result = validate_training_target(json.loads(line), naive)
                    hard = [d for d in result.errors if d.code != "TRAIN_ONLY_MODE"]
                    self.assertEqual(hard, [], f"{path}: unexpected hard error")
                    expected_quarantined += 1 if result.quarantined else 0
            actual_quarantined = sum(1 for ex in examples if ex.quarantined)
            self.assertEqual(actual_quarantined, expected_quarantined, f"{path}")
            checked += len(examples)
        self.assertGreater(checked, 0)


class OnlyModeFilterTests(unittest.TestCase):
    def test_exclude_drops_quarantined(self):
        path = _write_jsonl([FULL_RECORD, ONLY_MODE_RECORD])
        examples = load_examples(path)
        kept = filter_only_mode(examples, include_only_mode=False)
        self.assertEqual([ex.query for ex in kept], ["auth config"])

    def test_include_keeps_everything(self):
        path = _write_jsonl([FULL_RECORD, ONLY_MODE_RECORD])
        examples = load_examples(path)
        kept = filter_only_mode(examples, include_only_mode=True)
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
