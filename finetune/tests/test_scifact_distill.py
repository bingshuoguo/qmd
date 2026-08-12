"""Tests for the minimal SciFact distillation Contract/SFT bridge."""

from __future__ import annotations

import copy
import unittest

from dataset.scifact_distill import (
    SCIFACT_SEMANTIC_GATE_VERSION,
    materialize_records,
    validate_candidates,
    validate_scifact_semantics,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))

    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        self.assert_false(tokenize, add_generation_prompt)
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{messages[1]['content']}<|im_end|>\n"
        )

    @staticmethod
    def assert_false(tokenize: bool, add_generation_prompt: bool) -> None:
        if tokenize or add_generation_prompt:
            raise AssertionError("unexpected chat-template flags")


def raw_record(qid: str, split: str) -> dict:
    candidates = []
    for index in range(4):
        candidates.append(
            {
                "candidate_index": index,
                "generation_status": "ok",
                "raw_output": "lex: alpha evidence\nvec: how to find alpha evidence",
                "parsed_output": [
                    ["lex", "alpha evidence"],
                    ["vec", "how to find alpha evidence"],
                ],
                "generation_error": None,
                "contract": None,
                "metrics": None,
            }
        )
    return {
        "qid": qid,
        "split": split,
        "query": f"query {qid}",
        "raw_metrics": None,
        "candidates": candidates,
        "selected_candidate_index": None,
        "selection_status": "pending",
    }


class SciFactDistillContractTests(unittest.TestCase):
    def test_validation_uses_contract_and_preserves_raw_output(self):
        record = raw_record("q1", "train")
        before = [candidate["raw_output"] for candidate in record["candidates"]]

        validated = validate_candidates([record], lambda text: len(text.split()))

        self.assertEqual(
            [candidate["raw_output"] for candidate in validated[0]["candidates"]],
            before,
        )
        for candidate in validated[0]["candidates"]:
            self.assertTrue(candidate["contract"]["valid"])
            self.assertEqual(candidate["contract"]["version"], "training-target-v1.1")
            self.assertEqual(
                candidate["contract"]["canonical_output"],
                [
                    ["lex", "alpha evidence"],
                    ["vec", "how to find alpha evidence"],
                ],
            )

    def test_validation_refuses_to_destroy_scored_results(self):
        record = raw_record("q1", "train")
        record["selection_status"] = "winner"
        with self.assertRaisesRegex(ValueError, "already been scored"):
            validate_candidates([record], lambda text: len(text.split()))

    def test_scifact_gate_accepts_neutral_fixed_profile(self):
        output = [
            ["lex", "protein X does not increase outcome Y"],
            ["lex", "protein X lacks increased outcome Y"],
            ["lex", "no increase outcome Y protein X"],
            ["vec", "Evidence assessing whether protein X does not increase outcome Y"],
            ["vec", "Studies evaluating the claim that protein X lacks an increase in outcome Y"],
            ["vec", "Research examining whether protein X has no increase in outcome Y"],
            ["hyde", "Evidence evaluating whether protein X does not increase outcome Y remains focused on the reported relationship. The assessment tests the claim without reporting that the relationship is established."],
        ]
        result = validate_scifact_semantics("Protein X does not increase outcome Y.", output)
        self.assertTrue(result["valid"], result["errors"])

    def test_scifact_gate_reports_negation_loss_without_blocking(self):
        output = [
            ["lex", "protein X outcome Y increase"],
            ["lex", "protein X does not increase outcome Y"],
            ["lex", "no increase outcome Y protein X"],
            ["vec", "Evidence assessing whether protein X increases outcome Y"],
            ["vec", "Studies evaluating the claim that protein X lacks an increase in outcome Y"],
            ["vec", "Research examining whether protein X has no increase in outcome Y"],
            ["hyde", "Evidence evaluating whether protein X does not increase outcome Y remains focused on the reported relationship. The assessment tests the claim without reporting that the relationship is established."],
        ]
        result = validate_scifact_semantics("Protein X does not increase outcome Y.", output)
        advisory_paths = {
            item["path"] for item in result["advisories"]
            if item["code"] == "negation_lost"
        }
        self.assertEqual(advisory_paths, {"output[0][1]", "output[3][1]"})
        self.assertNotIn(
            "negation_lost", {error["code"] for error in result["errors"]}
        )
        self.assertTrue(result["valid"], result["errors"])

    def test_scifact_gate_never_blocks_on_assertion_or_unsupported_content(self):
        """The removed word-list checks must not come back as blocking errors.

        Every expansion below tripped one of them: mechanism ("via"),
        comparison ("compared to"), clinical advice ("should", "clinicians"),
        established-fact framing (no hedge word), and a non-3/3/1 shape.
        """
        output = [
            ["lex", "protein X outcome Y"],
            ["lex", "protein X outcome Y evidence"],
            ["vec", "Protein X increases outcome Y via mechanism Z"],
            ["vec", "Protein X increases outcome Y compared to controls"],
            ["vec", "Clinicians should monitor outcome Y"],
            ["hyde", "Protein X increases outcome Y and therefore guides clinical decisions for patients."],
        ]
        result = validate_scifact_semantics("Protein X increases outcome Y.", output)
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["valid"])
        self.assertEqual(result["advisories"], [])

    def test_validation_records_optional_scifact_gate(self):
        record = raw_record("q1", "train")
        validated = validate_candidates(
            [record], lambda text: len(text.split()), SCIFACT_SEMANTIC_GATE_VERSION
        )
        for candidate in validated[0]["candidates"]:
            self.assertEqual(
                candidate["semantic_gate"]["version"], SCIFACT_SEMANTIC_GATE_VERSION
            )
            self.assertTrue(candidate["semantic_gate"]["valid"])
            self.assertEqual(candidate["semantic_gate"]["errors"], [])


class SciFactDistillMaterializationTests(unittest.TestCase):
    def test_materializes_only_winners_into_fixed_splits(self):
        tokenizer = FakeTokenizer()
        train = validate_candidates(
            [raw_record("train-1", "train")], lambda text: len(text.split())
        )[0]
        val = validate_candidates(
            [raw_record("val-1", "val")], lambda text: len(text.split())
        )[0]
        skipped = validate_candidates(
            [raw_record("train-2", "train")], lambda text: len(text.split())
        )[0]
        for record in (train, val):
            record["selection_status"] = "winner"
            record["selected_candidate_index"] = 0
        skipped["selection_status"] = "no_winner"

        materialized = materialize_records(
            [copy.deepcopy(train), copy.deepcopy(skipped), copy.deepcopy(val)], tokenizer
        )

        self.assertEqual([item["qid"] for item in materialized["train"]], ["train-1"])
        self.assertEqual([item["qid"] for item in materialized["val"]], ["val-1"])
        self.assertTrue(materialized["train"][0]["prompt"].endswith("assistant\n"))
        self.assertIn("lex: alpha evidence", materialized["train"][0]["completion"])


if __name__ == "__main__":
    unittest.main()
