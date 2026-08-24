"""Tests for manifest-gated release loading (spec section 2.4)."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from dataset import sft_release
from dataset.public_distill import atomic_json, atomic_jsonl, sha256_file
from dataset.public_distill_v1 import (
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    USER_PROMPT_TEMPLATE,
    expected_prompt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPO_ROOT / "data/public-distill-v1"
MANIFEST = RELEASE_ROOT / "experiments/public-main-v1/release-manifest.json"
V2_USER_PROMPT_TEMPLATE = (
    "/no_think You generate retrieval expansions for a search query.\n\n"
    "Return only newline-separated lines in exactly this format:\n"
    "hyde: <one hypothetical document passage>\n"
    "vec: <semantic search query>\n"
    "vec: <optional second complementary semantic search query>\n\n"
    "Search query: {query}"
)


def _record(
    index: int,
    split: str,
    *,
    release_id: str = "public-distill-v1",
    experiment_id: str = "public-main-v1",
    prompt_template: str = USER_PROMPT_TEMPLATE,
) -> dict:
    query = f"query {index}"
    return {
        "schema_version": "qmd-public-distill-v1",
        "release_id": release_id,
        "experiment_id": experiment_id,
        "input_id": f"fiqa-train:{index}",
        "source_id": "fiqa-train",
        "qid": str(index),
        "query": query,
        "prompt": (
            f"<|im_start|>user\n{prompt_template.replace('{query}', query)}"
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "completion": f"lex: term {index}\nvec: about {index}<|im_end|>\n",
        "output": [["lex", f"term {index}"], ["vec", f"about {index}"]],
        "selected_candidate_index": 0,
        "selection_status": "winner",
        "split": split,
        "smoke_only": False,
        "final_sft_eligible": True,
    }


def _build_release(
    directory: Path,
    train_count: int = 3,
    validation_count: int = 2,
    *,
    release_id: str = "public-distill-v1",
    experiment_id: str = "public-main-v1",
    prompt_template: str = USER_PROMPT_TEMPLATE,
) -> Path:
    """Write a minimal but structurally faithful release into `directory`."""
    experiment_dir = directory / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True)

    train = [
        _record(i, "train", release_id=release_id, experiment_id=experiment_id,
                prompt_template=prompt_template)
        for i in range(train_count)
    ]
    validation = [
        _record(train_count + i, "validation", release_id=release_id,
                experiment_id=experiment_id, prompt_template=prompt_template)
        for i in range(validation_count)
    ]
    atomic_jsonl(experiment_dir / "sft-train.jsonl", train)
    atomic_jsonl(experiment_dir / "sft-validation.jsonl", validation)

    def entry(name: str, rows: int) -> dict:
        path = experiment_dir / name
        return {
            "path": f"experiments/{experiment_id}/{name}",
            "bytes": path.stat().st_size,
            "rows": rows,
            "sha256": sha256_file(path),
        }

    manifest_path = experiment_dir / "release-manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": "qmd-public-distill-release-v1",
            "release_id": release_id,
            "experiment_id": experiment_id,
            "status": "sealed",
            "final_sft_eligible": True,
            "provenance": {
                "prompt_version": "qmd-student-expansion-v1",
                "prompt_sha256": hashlib.sha256(
                    prompt_template.encode("utf-8")
                ).hexdigest(),
                "prompt_template": prompt_template,
            },
            "dataset": {"train": train_count, "validation": validation_count},
            "core_artifacts": {
                "sft-train.jsonl": entry("sft-train.jsonl", train_count),
                "sft-validation.jsonl": entry("sft-validation.jsonl", validation_count),
            },
        },
    )
    return manifest_path


class SyntheticReleaseTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.manifest = _build_release(self.root)

    def tearDown(self):
        self._temp.cleanup()

    def _mutate_manifest(self, **changes):
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        value.update(changes)
        atomic_json(self.manifest, value)

    def test_loads_a_valid_release(self):
        release = sft_release.load_release(self.manifest)

        self.assertEqual(release.release_id, "public-distill-v1")
        self.assertEqual(len(release.train.records), 3)
        self.assertEqual(len(release.validation.records), 2)
        self.assertEqual(release.provenance()["train_rows"], 3)
        self.assertIn("prompt_template", release.provenance())

    def test_accepts_a_sealed_v2_prompt_template(self):
        manifest = _build_release(
            self.root / "v2",
            release_id="public-distill-v2-vh-prompt-v1",
            experiment_id="public-main-v2-vh-prompt-v1",
            prompt_template=V2_USER_PROMPT_TEMPLATE,
        )
        release = sft_release.load_release(
            manifest,
            expected_release_id="public-distill-v2-vh-prompt-v1",
            expected_experiment_id="public-main-v2-vh-prompt-v1",
        )

        sft_release.assert_prompt_template(release.train.records, release.prompt_template)

    def test_rejects_unsealed_release(self):
        self._mutate_manifest(status="draft")

        with self.assertRaisesRegex(ValueError, "not sealed"):
            sft_release.load_release(self.manifest)

    def test_rejects_non_eligible_release(self):
        self._mutate_manifest(final_sft_eligible=False)

        with self.assertRaisesRegex(ValueError, "not final_sft_eligible"):
            sft_release.load_release(self.manifest)

    def test_rejects_prompt_template_whose_declared_hash_drifted(self):
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        value["provenance"]["prompt_sha256"] = "0" * 64
        atomic_json(self.manifest, value)

        with self.assertRaisesRegex(ValueError, "prompt_sha256"):
            sft_release.load_release(self.manifest)

    def test_rejects_unexpected_release_id(self):
        self._mutate_manifest(release_id="public-distill-v0")

        with self.assertRaisesRegex(ValueError, "release_id"):
            sft_release.load_release(self.manifest)

    def test_rejects_modified_split_file(self):
        train_path = self.manifest.parent / "sft-train.jsonl"
        with train_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record(99, "train")) + "\n")

        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            sft_release.load_release(self.manifest)

    def test_rejects_row_count_disagreement(self):
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        value["core_artifacts"]["sft-train.jsonl"]["rows"] = 5
        atomic_json(self.manifest, value)

        with self.assertRaisesRegex(ValueError, "manifest declares 5 rows"):
            sft_release.load_release(self.manifest)

    def test_rejects_dataset_count_disagreement(self):
        self._mutate_manifest(dataset={"train": 7, "validation": 2})

        with self.assertRaisesRegex(ValueError, "dataset.train does not match"):
            sft_release.load_release(self.manifest)

    def test_rejects_smoke_record(self):
        train_path = self.manifest.parent / "sft-train.jsonl"
        records = [json.loads(line) for line in train_path.read_text().splitlines()]
        records[0]["smoke_only"] = True
        atomic_jsonl(train_path, records)
        self._rehash("sft-train.jsonl")

        with self.assertRaisesRegex(ValueError, "is a smoke record"):
            sft_release.load_release(self.manifest)

    def test_rejects_split_overlap(self):
        validation_path = self.manifest.parent / "sft-validation.jsonl"
        records = [json.loads(line) for line in validation_path.read_text().splitlines()]
        records[0]["input_id"] = "fiqa-train:0"
        atomic_jsonl(validation_path, records)
        self._rehash("sft-validation.jsonl")

        with self.assertRaisesRegex(ValueError, "train/validation overlap"):
            sft_release.load_release(self.manifest)

    def test_rejects_wrong_split_label(self):
        validation_path = self.manifest.parent / "sft-validation.jsonl"
        records = [json.loads(line) for line in validation_path.read_text().splitlines()]
        records[0]["split"] = "train"
        atomic_jsonl(validation_path, records)
        self._rehash("sft-validation.jsonl")

        with self.assertRaisesRegex(ValueError, "has split 'train'"):
            sft_release.load_release(self.manifest)

    def _rehash(self, name: str) -> None:
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        path = self.manifest.parent / name
        value["core_artifacts"][name]["sha256"] = sha256_file(path)
        value["core_artifacts"][name]["bytes"] = path.stat().st_size
        atomic_json(self.manifest, value)


class TokenIdentityTests(unittest.TestCase):
    class Tokenizer:
        def __init__(self, pad, eos):
            self.pad_token_id, self.eos_token_id = pad, eos

    def test_accepts_frozen_identities(self):
        sft_release.assert_token_identities(self.Tokenizer(PAD_TOKEN_ID, EOS_TOKEN_ID))

    def test_rejects_pad_equal_to_eos(self):
        with self.assertRaisesRegex(ValueError, "pad_token_id"):
            sft_release.assert_token_identities(self.Tokenizer(EOS_TOKEN_ID, EOS_TOKEN_ID))

    def test_rejects_drifted_eos(self):
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            sft_release.assert_token_identities(self.Tokenizer(PAD_TOKEN_ID, 999))


class PromptTemplateGuardTests(unittest.TestCase):
    def test_rejects_a_stored_prompt_that_drifted(self):
        records = [_record(0, "train")]
        records[0]["prompt"] = "<|im_start|>user\nold prompt<|im_end|>\n<|im_start|>assistant\n"

        with self.assertRaisesRegex(ValueError, "does not match the sealed release template"):
            sft_release.assert_prompt_template(records, USER_PROMPT_TEMPLATE)

    def test_accepts_frozen_prompts(self):
        sft_release.assert_prompt_template(
            [_record(i, "train") for i in range(3)], USER_PROMPT_TEMPLATE
        )


@unittest.skipUnless(MANIFEST.is_file(), "v1 release is not present locally")
class RealReleaseTests(unittest.TestCase):
    def test_loads_the_materialized_v1_release(self):
        release = sft_release.load_release(MANIFEST)

        self.assertEqual(len(release.train.records), 1800)
        self.assertEqual(len(release.validation.records), 200)
        self.assertEqual(release.prompt_version, "qmd-student-expansion-v1")

    def test_every_stored_prompt_matches_the_frozen_template(self):
        release = sft_release.load_release(MANIFEST)

        sft_release.assert_prompt_template(release.train.records, release.prompt_template)
        sft_release.assert_prompt_template(release.validation.records, release.prompt_template)


if __name__ == "__main__":
    unittest.main()
