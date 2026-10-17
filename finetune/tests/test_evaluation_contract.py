"""CPU regressions for sealed prompts and non-destructive evaluation entrypoints."""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/public-eval"))
import generate_expansions as generation
import run_8b_sft_eval as runner
from test_sft_release import _build_release


class EvaluationContractTests(unittest.TestCase):
    def test_sealed_prompt_is_rendered_exactly_as_training(self):
        with tempfile.TemporaryDirectory() as directory:
            template = "/no_think\nRules: one hyde then complementary vec.\nSearch query: {query}"
            release = generation.load_evaluation_release(
                _build_release(Path(directory), prompt_template=template)
            )
            for record in release.train.records + release.validation.records:
                self.assertEqual(generation.expected_prompt(record["query"], release=release), record["prompt"])
            self.assertEqual(release.prompt_sha256, hashlib.sha256(template.encode()).hexdigest())

    def test_changed_template_with_stale_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _build_release(Path(directory))
            data = json.loads(manifest.read_text())
            data["provenance"]["prompt_template"] += " changed"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "prompt_sha256"):
                generation.load_evaluation_release(manifest)

    def test_self_consistent_manifest_cannot_hide_training_prompt_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _build_release(Path(directory))
            data = json.loads(manifest.read_text())
            template = data["provenance"]["prompt_template"] + " changed"
            data["provenance"].update(prompt_template=template, prompt_sha256=hashlib.sha256(template.encode()).hexdigest())
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "stored prompt"):
                generation.load_evaluation_release(manifest)

    def test_8b_runner_rejects_abbreviated_release_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = _build_release(Path(directory), prompt_template=generation.V2_VH_USER_TEMPLATE)
            with self.assertRaisesRegex(ValueError, "full sealed V2-VH"):
                runner.release_queries(manifest)

    def test_ambiguous_v2_vh_selector_fails_before_model_loading(self):
        argv = ["generate", "--benchmark", "/unused", "--model", "/unused", "--variant", "new", "--prompt-version", "v2-vh"]
        with patch.object(sys, "argv", argv), patch.object(generation, "load_model") as model:
            with self.assertRaises(SystemExit):
                generation.main()
            model.assert_not_called()
        self.assertNotEqual(generation.prompt_version_string("v2-vh-legacy"), "qmd-student-expansion-v2-vh-v1")

    def test_orphan_manifest_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw-generations"
            raw.mkdir()
            manifest = raw / "new-manifest.json"
            manifest.write_text('{"original":true}')
            argv = ["generate", "--benchmark", directory, "--model", "/unused", "--variant", "new"]
            with patch.object(sys, "argv", argv), patch.object(generation, "load_model") as model:
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    generation.main()
                model.assert_not_called()
            self.assertEqual(manifest.read_text(), '{"original":true}')

    def test_runner_builds_only_all_with_sealed_release_and_same_interpreter(self):
        commands = runner.commands(runner.Benchmark("sample", Path("/assets/sample")), Path("/model"),
                                   Path("/adapter"), "new", 4, Path("/release.json"), runner.DEFAULT_REVISION)
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0][0], sys.executable)
        self.assertIn("--release-manifest", commands[0])
        self.assertIn("/release.json", commands[0])
        self.assertIn(runner.DEFAULT_REVISION, commands[0])
        self.assertFalse(any("--only" in command or "--prompt-version" in command for command in commands))
        self.assertEqual(commands[-1][-1], "--json")

    def test_existing_v2a_entrypoint_uses_registered_explicit_prompts(self):
        source = Path(__file__).resolve().parents[1] / "benchmarks/v2a-pareto/evaluate.py"
        arms = runpy.run_path(str(source))["ARMS"]
        for _, prompt in arms.values():
            self.assertIn(prompt, generation.PROMPT_REGISTRY)
        self.assertEqual(arms["sft-vh-orig"][1], "v2-vh-legacy")

    def test_wrong_inference_environment_is_blocked(self):
        with patch.object(runner, "runtime_environment", return_value={"packages": {"torch": "wrong"}}):
            with self.assertRaisesRegex(ValueError, "environment mismatch"):
                runner.verify_environment()

    def test_orphan_results_block_runner_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmark.yaml").write_text("benchmark_id: sample\n")
            (root / "retrieval-profile.yaml").write_text("collection_name: sample\nrerank: false\n")
            (root / "documents.jsonl").write_text('{}\n')
            (root / "queries.jsonl").write_text('{"qid":"1","query":"sample"}\n')
            (root / "qrels.tsv").write_text("")
            results = root / "runs/results/new-qwen3-8b.jsonl.partial"
            results.parent.mkdir(parents=True)
            results.write_text("original partial result")
            with self.assertRaisesRegex(ValueError, "partial_results"):
                runner.preflight_benchmark(runner.Benchmark("sample", root), set(), "new")
            self.assertEqual(results.read_text(), "original partial result")


if __name__ == "__main__":
    unittest.main()
