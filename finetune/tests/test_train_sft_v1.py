"""Tests for the completion-only SFT training path (spec sections 7 and 10)."""

from __future__ import annotations

import unittest
from pathlib import Path

import train_sft_v1 as trainer_module
from dataset.completion import tokenize_completion_example
from dataset.public_distill_v1 import EOS_TOKEN_ID, PAD_TOKEN_ID, expected_prompt


FINETUNE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = FINETUNE_ROOT / "configs/sft-v1.yaml"
V2_VH_CONFIG_PATH = FINETUNE_ROOT / "configs/sft-v2-vh-prompt-v1.yaml"


class FakeParameter:
    def __init__(self, numel: int, requires_grad: bool):
        self._numel = numel
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._numel


class FakeModel:
    def __init__(self, parameters, is_gradient_checkpointing=False):
        self._parameters = parameters
        self.is_gradient_checkpointing = is_gradient_checkpointing

    def named_parameters(self):
        return iter(self._parameters)


class FakeArgs:
    def __init__(self, gradient_checkpointing: bool):
        self.gradient_checkpointing = gradient_checkpointing


class ConfigTests(unittest.TestCase):
    def test_frozen_config_loads_and_matches_the_spec(self):
        config = trainer_module.load_config(CONFIG_PATH)

        self.assertEqual(config["release"]["release_id"], "public-distill-v1")
        self.assertEqual(config["model"]["model_id"], "Qwen/Qwen3-1.7B")
        self.assertEqual(
            config["model"]["revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        )
        self.assertEqual(config["model"]["max_seq_length"], 1024)
        self.assertEqual(config["lora"]["r"], 16)
        self.assertEqual(config["lora"]["lora_alpha"], 32)
        self.assertEqual(config["lora"]["lora_dropout"], 0.05)
        self.assertEqual(config["training"]["num_train_epochs"], 3)
        self.assertEqual(config["training"]["learning_rate"], 2.0e-4)
        self.assertEqual(config["training"]["warmup_ratio"], 0.05)
        self.assertEqual(config["training"]["weight_decay"], 0.01)
        self.assertFalse(config["precision"]["gradient_checkpointing"])
        self.assertEqual(config["precision"]["seed"], 42)
        self.assertEqual(trainer_module.effective_batch_size(config["training"]), 16)

    def test_v2_vh_config_targets_the_sealed_v2_release(self):
        self.assertTrue(V2_VH_CONFIG_PATH.is_file())
        config = trainer_module.load_config(V2_VH_CONFIG_PATH)

        self.assertEqual(
            config["release"]["manifest"],
            "data/public-distill-v2-vh-prompt-v1/experiments/"
            "public-main-v2-vh-prompt-v1/release-manifest.json",
        )
        self.assertEqual(config["release"]["release_id"], "public-distill-v2-vh-prompt-v1")
        self.assertEqual(
            config["release"]["experiment_id"], "public-main-v2-vh-prompt-v1"
        )
        self.assertEqual(config["model"]["max_seq_length"], 1024)

    def test_rejects_four_bit_loading(self):
        config = trainer_module.load_config(CONFIG_PATH)
        config["model"]["load_in_4bit"] = True

        with self.assertRaisesRegex(ValueError, "forbids 4-bit"):
            self._reload(config)

    def test_rejects_a_missing_section(self):
        config = trainer_module.load_config(CONFIG_PATH)
        del config["lora"]

        with self.assertRaisesRegex(ValueError, "missing 'lora' section"):
            self._reload(config)

    def test_rejects_fp16(self):
        config = trainer_module.load_config(CONFIG_PATH)
        config["precision"]["fp16"] = True

        with self.assertRaisesRegex(ValueError, "requires bf16"):
            self._reload(config)

    def _reload(self, config):
        """Re-run the config guards against an in-memory mutation."""
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(config, handle)
            path = Path(handle.name)
        try:
            return trainer_module.load_config(path)
        finally:
            path.unlink(missing_ok=True)


class TrainableParameterTests(unittest.TestCase):
    TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def test_accepts_lora_only(self):
        model = FakeModel([
            ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
             FakeParameter(32_768, True)),
            ("base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
             FakeParameter(32_768, True)),
            ("base_model.model.layers.0.self_attn.q_proj.base_layer.weight",
             FakeParameter(4_194_304, False)),
        ])

        trainable = trainer_module.assert_trainable_parameters(model, self.TARGETS)

        self.assertEqual(len(trainable), 2)
        self.assertEqual(sum(trainable.values()), 65_536)

    def test_rejects_trainable_embeddings(self):
        model = FakeModel([
            ("base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
             FakeParameter(32_768, True)),
            ("base_model.model.model.embed_tokens.weight", FakeParameter(311_164_928, True)),
        ])

        with self.assertRaisesRegex(ValueError, "non-LoRA parameter is trainable"):
            trainer_module.assert_trainable_parameters(model, self.TARGETS)

    def test_rejects_lora_outside_approved_modules(self):
        model = FakeModel([
            ("base_model.model.layers.0.mlp.router.lora_A.default.weight",
             FakeParameter(1024, True)),
        ])

        with self.assertRaisesRegex(ValueError, "outside the approved modules"):
            trainer_module.assert_trainable_parameters(model, self.TARGETS)

    def test_rejects_a_model_with_no_trainable_parameters(self):
        model = FakeModel([("weight", FakeParameter(10, False))])

        with self.assertRaisesRegex(ValueError, "LoRA was not attached"):
            trainer_module.assert_trainable_parameters(model, self.TARGETS)


class GradientCheckpointingTests(unittest.TestCase):
    def test_accepts_disabled(self):
        trainer_module.assert_gradient_checkpointing_disabled(
            FakeModel([], is_gradient_checkpointing=False), FakeArgs(False)
        )

    def test_rejects_enabled_in_args(self):
        with self.assertRaisesRegex(ValueError, "SFTConfig.gradient_checkpointing"):
            trainer_module.assert_gradient_checkpointing_disabled(
                FakeModel([], is_gradient_checkpointing=False), FakeArgs(True)
            )

    def test_rejects_enabled_on_model(self):
        with self.assertRaisesRegex(ValueError, "model reports gradient checkpointing"):
            trainer_module.assert_gradient_checkpointing_disabled(
                FakeModel([], is_gradient_checkpointing=True), FakeArgs(False)
            )


class RealCollatorSupervisionTests(unittest.TestCase):
    """Drive the real TRL collator, mirroring what the trainer will build."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        # Section 6: max_length=None so the collator cannot silently truncate.
        cls.collator = DataCollatorForLanguageModeling(
            pad_token_id=PAD_TOKEN_ID,
            max_length=None,
            completion_only_loss=True,
        )

    def _records(self, queries):
        records = []
        for index, query in enumerate(queries):
            prompt = expected_prompt(query)
            completion = f"lex: term {index}\nvec: about {index}<|im_end|>\n"
            records.append(
                tokenize_completion_example(
                    {"prompt": prompt, "completion": completion},
                    self.tokenizer,
                    max_length=1024,
                )
            )
        return records

    def test_prompt_and_padding_are_masked_across_uneven_lengths(self):
        records = self._records(["auth", "how do I configure hybrid retrieval end to end"])
        batch = self.collator(records)

        supervised = trainer_module.assert_batch_supervision(batch, records)

        self.assertEqual(supervised, sum(sum(r["completion_mask"]) for r in records))
        self.assertGreater(batch["input_ids"].shape[1], len(records[0]["input_ids"]))

    def test_end_of_turn_token_is_supervised(self):
        records = self._records(["auth"])
        batch = self.collator(records)

        labels = batch["labels"][0].tolist()
        self.assertIn(EOS_TOKEN_ID, labels)

    def test_detects_a_mask_that_leaks_prompt_supervision(self):
        records = self._records(["auth"])
        # Collate with an all-ones mask so every label is supervised, then check
        # against the true prompt boundary: the assertion must spot the leak.
        leaked = dict(records[0], completion_mask=[1] * len(records[0]["input_ids"]))
        batch = self.collator([leaked])

        with self.assertRaisesRegex(ValueError, "prompt token is supervised"):
            trainer_module.assert_batch_supervision(batch, records)

    def test_detects_a_batch_row_count_mismatch(self):
        records = self._records(["auth", "taxes"])
        batch = self.collator(records)

        with self.assertRaisesRegex(ValueError, "rows for"):
            trainer_module.assert_batch_supervision(batch, records[:1])


if __name__ == "__main__":
    unittest.main()
