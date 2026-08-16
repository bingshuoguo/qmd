"""Tests for the v0 -> v1 student-prompt rematerialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset import public_distill_v1 as v1


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "data/public-distill-v0/experiments/public-main-v0"


def _v0_record(**overrides):
    record = {
        "schema_version": "qmd-public-distill-v0",
        "input_id": "fiqa-train:1",
        "source_id": "fiqa-train",
        "qid": "1",
        "query": "how to file taxes",
        "prompt": "<|im_start|>user\n/no_think Expand this search query: how to file taxes"
        "<|im_end|>\n<|im_start|>assistant\n",
        "completion": "hyde: A passage about filing taxes.\nlex: file taxes\n"
        "vec: how do I file my taxes<|im_end|>\n",
        "output": [
            ["hyde", "A passage about filing taxes."],
            ["lex", "file taxes"],
            ["vec", "how do I file my taxes"],
        ],
        "selected_candidate_index": 0,
        "selection_status": "winner",
        "split": "train",
        "smoke_only": False,
        "final_sft_eligible": True,
        "experiment_id": "public-main-v0",
    }
    record.update(overrides)
    return record


class PromptTemplateTests(unittest.TestCase):
    def test_expected_prompt_is_char_exact(self):
        prompt = v1.expected_prompt("auth config")

        self.assertTrue(prompt.startswith("<|im_start|>user\n/no_think Expand this search query"))
        self.assertTrue(prompt.endswith("<|im_end|>\n<|im_start|>assistant\n"))
        self.assertIn("Query: auth config<|im_end|>", prompt)
        self.assertNotIn("<|im_start|>system", prompt)
        self.assertNotIn("<think>", prompt)

    def test_prompt_sha256_is_stable(self):
        self.assertEqual(v1.prompt_sha256(), v1.prompt_sha256())
        self.assertEqual(len(v1.prompt_sha256()), 64)

    def test_render_prompt_rejects_template_drift(self):
        class DriftingTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return (
                    "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                    f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

        with self.assertRaisesRegex(ValueError, "does not match the frozen v1 template"):
            v1.render_prompt("auth config", DriftingTokenizer())


class ConvertRecordTests(unittest.TestCase):
    def setUp(self):
        class FrozenTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return (
                    f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

        self.tokenizer = FrozenTokenizer()

    def test_completion_and_provenance_are_carried_over_unchanged(self):
        source = _v0_record()

        converted = v1.convert_record(source, self.tokenizer)

        self.assertEqual(converted["completion"], source["completion"])
        self.assertEqual(converted["output"], source["output"])
        for field in ("input_id", "source_id", "qid", "query", "split",
                      "selected_candidate_index", "selection_status"):
            self.assertEqual(converted[field], source[field], field)

    def test_version_metadata_is_updated(self):
        converted = v1.convert_record(_v0_record(), self.tokenizer)

        self.assertEqual(converted["schema_version"], v1.SCHEMA_VERSION)
        self.assertEqual(converted["release_id"], v1.RELEASE_ID)
        self.assertEqual(converted["experiment_id"], v1.EXPERIMENT_ID)

    def test_prompt_is_replaced_by_the_v1_template(self):
        source = _v0_record()

        converted = v1.convert_record(source, self.tokenizer)

        self.assertNotEqual(converted["prompt"], source["prompt"])
        self.assertEqual(converted["prompt"], v1.expected_prompt(source["query"]))

    def test_rejects_output_that_does_not_render_to_the_completion(self):
        source = _v0_record(output=[["lex", "something else"]])

        with self.assertRaisesRegex(ValueError, "does not render to the completion"):
            v1.convert_record(source, self.tokenizer)

    def test_rejects_completion_without_end_of_turn(self):
        source = _v0_record(completion="lex: file taxes\n")

        with self.assertRaisesRegex(ValueError, "must end with"):
            v1.convert_record(source, self.tokenizer)

    def test_rejects_smoke_records(self):
        source = _v0_record(smoke_only=True)

        with self.assertRaisesRegex(ValueError, "smoke records must not reach v1"):
            v1.convert_record(source, self.tokenizer)


class TokenPreflightTests(unittest.TestCase):
    class CharTokenizer:
        """Deterministic per-character tokenizer with a distinct EOS id."""

        def __call__(self, text, add_special_tokens=False):
            ids = []
            index = 0
            while index < len(text):
                if text.startswith("<|im_end|>", index):
                    ids.append(v1.EOS_TOKEN_ID)
                    index += len("<|im_end|>")
                else:
                    ids.append(ord(text[index]))
                    index += 1
            return {"input_ids": ids}

    def _record(self, prompt="<|im_start|>assistant\n", completion="lex: a<|im_end|>\n"):
        return {"input_id": "x:1", "prompt": prompt, "completion": completion}

    def test_returns_prompt_and_total_lengths(self):
        tokenizer = self.CharTokenizer()
        record = self._record()

        prompt_length, total_length = v1.token_preflight(record, tokenizer, max_length=1024)

        self.assertEqual(prompt_length, len(tokenizer(record["prompt"])["input_ids"]))
        self.assertGreater(total_length, prompt_length)

    def test_rejects_over_length_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            v1.token_preflight(self._record(), self.CharTokenizer(), max_length=5)

    def test_rejects_completion_without_end_of_turn_token(self):
        record = self._record(completion="lex: a\n")

        with self.assertRaisesRegex(ValueError, "no end-of-turn token"):
            v1.token_preflight(record, self.CharTokenizer(), max_length=1024)

    def test_rejects_non_prefix_prompt_tokenization(self):
        class MergingTokenizer:
            def __call__(self, text, add_special_tokens=False):
                # Emits a different first token for the concatenated text.
                return {"input_ids": [len(text)] + [ord(c) for c in text]}

        with self.assertRaisesRegex(ValueError, "not a prefix"):
            v1.token_preflight(self._record(), MergingTokenizer(), max_length=1024)


class SourceVerificationTests(unittest.TestCase):
    def test_rejects_a_tampered_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in v1.SOURCE_SHA256:
                (source / name).write_text("tampered\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sealed source hash mismatch"):
                v1._verify_source(source)

    def test_rejects_a_missing_source_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sealed source artifact is missing"):
                v1._verify_source(Path(directory))

    @unittest.skipUnless(SOURCE_DIR.is_dir(), "sealed v0 release is not present locally")
    def test_accepts_the_real_sealed_release(self):
        observed = v1._verify_source(SOURCE_DIR)

        self.assertEqual(observed, v1.SOURCE_SHA256)


@unittest.skipUnless(SOURCE_DIR.is_dir(), "sealed v0 release is not present locally")
class RematerializeIntegrationTests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "experiments" / "public-main-v1"
            target.mkdir(parents=True)
            (target / "sft.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                v1.rematerialize(SOURCE_DIR, target, local_files_only=True)


if __name__ == "__main__":
    unittest.main()
