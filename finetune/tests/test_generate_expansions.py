"""Regression tests for per-sample EOS accounting in batched generation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/public-eval/generate_expansions.py"
SPEC = importlib.util.spec_from_file_location("generate_expansions", SCRIPT)
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)

EOS = 151645
PAD = 151643
BUDGET = generation.GENERATION["max_new_tokens"]


class GenerateTests(unittest.TestCase):
    def run_generation(
        self,
        rows: list[list[int]],
        eos_token_id: int | list[int] | None = EOS,
        pad_token_id: int = PAD,
    ) -> tuple[list[dict], Mock, Mock]:
        """Run the actual generation loop with deterministic CPU tensor outputs."""
        # An EOS inside the left-padded prompt must not end the continuation.
        prompts = torch.tensor([[pad_token_id, EOS, 42] for _ in rows])
        inputs = Mock()
        inputs.to.return_value = {"input_ids": prompts}
        tokenizer = Mock(eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        tokenizer.return_value = inputs
        tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(map(str, ids))
        model = Mock(device="cpu")
        model.generate.return_value = torch.cat((prompts, torch.tensor(rows)), dim=1)
        queries = [{"qid": str(i), "query": f"query {i}"} for i in range(len(rows))]
        records = generation.generate(model, tokenizer, queries, batch_size=len(rows))
        return records, tokenizer, model

    def test_mixed_batch_only_marks_the_unfinished_sample_truncated(self):
        rows = [[7, EOS] + [PAD] * (BUDGET - 2), [8] * BUDGET]
        records, tokenizer, model = self.run_generation(rows)

        self.assertEqual([r["truncated"] for r in records], [False, True])
        self.assertEqual([r["generated_tokens"] for r in records], [2, BUDGET])
        self.assertEqual([r["finish_reason"] for r in records], ["eos", "length"])
        self.assertEqual(tokenizer.decode.call_args_list[0].args[0], [7, EOS])
        self.assertEqual(model.generate.call_args.kwargs["eos_token_id"], EOS)
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], BUDGET)

    def test_eos_at_exact_budget_is_not_truncated(self):
        records, _, _ = self.run_generation([[7] * (BUDGET - 1) + [EOS]])
        self.assertFalse(records[0]["truncated"])
        self.assertEqual(records[0]["finish_reason"], "eos")
        self.assertEqual(records[0]["generated_tokens"], BUDGET)

    def test_first_token_eos_counts_as_one_generated_token(self):
        records, tokenizer, _ = self.run_generation([[EOS, PAD, PAD]])
        self.assertEqual(records[0]["generated_tokens"], 1)
        self.assertFalse(records[0]["truncated"])
        tokenizer.decode.assert_called_once_with([EOS], skip_special_tokens=True)

    def test_shared_pad_and_eos_id_keeps_the_first_eos(self):
        records, tokenizer, _ = self.run_generation(
            [[7, EOS, EOS, EOS]], pad_token_id=EOS
        )
        self.assertEqual(records[0]["generated_tokens"], 2)
        tokenizer.decode.assert_called_once_with([7, EOS], skip_special_tokens=True)

    def test_multiple_eos_ids_use_the_first_matching_token(self):
        records, tokenizer, model = self.run_generation(
            [[7, 99, EOS, PAD]], eos_token_id=[EOS, 99]
        )
        self.assertEqual(records[0]["finish_reason"], "eos")
        self.assertEqual(records[0]["generated_tokens"], 2)
        tokenizer.decode.assert_called_once_with([7, 99], skip_special_tokens=True)
        self.assertEqual(model.generate.call_args.kwargs["eos_token_id"], [EOS, 99])

    def test_short_output_without_eos_is_a_generation_error(self):
        records, _, _ = self.run_generation([[7, 8]])
        self.assertFalse(records[0]["truncated"])
        self.assertEqual(records[0]["finish_reason"], "unknown")
        self.assertIn("without EOS", records[0]["generation_error"])

    def test_success_does_not_set_generation_error(self):
        records, _, _ = self.run_generation([[7, EOS]])
        self.assertIsNone(records[0].get("generation_error"))

    def test_missing_eos_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.run_generation([[7, 8]], eos_token_id=None)


if __name__ == "__main__":
    unittest.main()
