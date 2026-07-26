"""Regression tests for QMD completion-only supervision."""

from __future__ import annotations

import tempfile
import unittest

from dataset.completion import (
    ASSISTANT_MARKER,
    split_rendered_text,
    tokenize_completion_example,
)


class CharacterTokenizer:
    """Small deterministic tokenizer for boundary and length unit tests."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class CompletionUnitTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.text = (
            "<|im_start|>user\n/no_think Expand this search query: auth"
            "<|im_end|>\n"
            f"{ASSISTANT_MARKER}"
            "lex: authentication\nvec: how to configure authentication"
            "<|im_end|>\n"
        )

    def test_split_reconstructs_rendered_text(self):
        prompt, completion = split_rendered_text(self.text)

        self.assertEqual(prompt + completion, self.text)
        self.assertTrue(prompt.endswith(ASSISTANT_MARKER))
        self.assertTrue(completion.startswith("lex:"))

    def test_split_requires_one_assistant_header(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            split_rendered_text("no assistant header")
        with self.assertRaisesRegex(ValueError, "found 2"):
            split_rendered_text(self.text + ASSISTANT_MARKER)

    def test_legacy_text_and_prompt_completion_have_identical_masks(self):
        prompt, completion = split_rendered_text(self.text)

        legacy = tokenize_completion_example(
            {"text": self.text}, self.tokenizer, max_length=1000
        )
        current = tokenize_completion_example(
            {"prompt": prompt, "completion": completion},
            self.tokenizer,
            max_length=1000,
        )

        self.assertEqual(legacy, current)
        prompt_length = len(prompt)
        self.assertEqual(legacy["completion_mask"][:prompt_length], [0] * prompt_length)
        self.assertTrue(all(legacy["completion_mask"][prompt_length:]))

    def test_refuses_to_truncate_completion(self):
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            tokenize_completion_example(
                {"text": self.text}, self.tokenizer, max_length=10
            )


class TrlCollatorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def test_real_collator_masks_all_prompt_tokens(self):
        from datasets import Dataset
        from transformers import GPT2Config, GPT2LMHeadModel
        from trl import SFTConfig, SFTTrainer

        fixtures = [
            (
                "/no_think Expand this search query: authentication setup",
                "lex: authentication configuration\n"
                "vec: how to configure authentication",
            ),
            (
                "/no_think Expand this search query: jaguar\n"
                "Query intent: the animal rather than the car",
                "lex: jaguar animal\nvec: information about jaguars in the wild",
            ),
            (
                "/no_think Expand this search query: authentication /only:lex",
                "lex: authentication configuration",
            ),
            (
                "/no_think Expand this search query: authentication /only:vec",
                "vec: how to configure authentication",
            ),
            (
                "/no_think Expand this search query: authentication /only:hyde",
                "hyde: Authentication is configured through application settings.",
            ),
        ]

        tokenized = []
        prompt_lengths = []
        rendered_ids = []
        for user_prompt, output in fixtures:
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": output},
                ],
                tokenize=False,
                add_generation_prompt=False,
            ).replace("<think>\n\n</think>\n\n", "")
            prompt, completion = split_rendered_text(text)
            record = tokenize_completion_example(
                {"prompt": prompt, "completion": completion},
                self.tokenizer,
                max_length=512,
            )
            tokenized.append(record)
            prompt_lengths.append(
                len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            )
            rendered_ids.append(
                self.tokenizer(text, add_special_tokens=False)["input_ids"]
            )

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=max(self.tokenizer.get_vocab().values()) + 1,
                n_positions=512,
                n_ctx=512,
                n_embd=32,
                n_layer=1,
                n_head=1,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        )

        with tempfile.TemporaryDirectory(prefix="qmd-completion-test-") as output_dir:
            trainer = SFTTrainer(
                model=model,
                train_dataset=Dataset.from_list(tokenized),
                args=SFTConfig(
                    output_dir=output_dir,
                    max_length=512,
                    per_device_train_batch_size=1,
                    report_to="none",
                    completion_only_loss=True,
                ),
                processing_class=self.tokenizer,
            )

            for index, record in enumerate(trainer.train_dataset):
                batch = trainer.data_collator([record])
                input_ids = batch["input_ids"][0].tolist()
                labels = batch["labels"][0].tolist()
                prompt_length = prompt_lengths[index]

                self.assertEqual(input_ids, rendered_ids[index])
                self.assertTrue(
                    all(label == -100 for label in labels[:prompt_length])
                )
                self.assertEqual(
                    labels[prompt_length:],
                    input_ids[prompt_length:],
                )


if __name__ == "__main__":
    unittest.main()
