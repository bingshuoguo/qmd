from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from .contracts import (
    ADAPTER_SHA256,
    atomic_json,
    load_config,
    load_sealed_prompt,
    sha256_path,
)
from .technical_preflight import merge_equivalence


PROBE_QUERIES = (
    "how do interest rates affect bond prices",
    "vitamin d and bone density",
    "python sort dictionary by value",
)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def merge_and_verify(config_path: Path, output_dir: Path) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path, mode="formal")
    if not torch.cuda.is_available():
        raise RuntimeError("formal merge requires CUDA")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"merge output is not empty: {output_dir}")
    prompt = load_sealed_prompt(config.release_manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        config.merged_v2.model_id,
        revision=config.merged_v2.revision,
        trust_remote_code=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.merged_v2.model_id,
        revision=config.merged_v2.revision,
        dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=False,
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(config.merged_v2.adapter)).eval()
    encoded = tokenizer(
        [prompt.render(query) for query in PROBE_QUERIES],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to("cuda")
    with torch.inference_mode():
        before = model(**encoded).logits[:, -1, :].float().cpu().flatten()
    merged = model.merge_and_unload(safe_merge=True).eval()
    with torch.inference_mode():
        after = merged(**encoded).logits[:, -1, :].float().cpu().flatten()
    equivalence = merge_equivalence(before.tolist(), after.tolist())
    if equivalence["status"] != "passed":
        raise RuntimeError(f"merge logits equivalence failed: {equivalence}")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    file_hashes = _tree_hashes(output_dir)
    manifest = {
        "schema_version": "qmd-merged-v2-manifest-v1",
        "status": "passed",
        "base": {"model_id": config.merged_v2.model_id, "revision": config.merged_v2.revision},
        "adapter": {"path": str(config.merged_v2.adapter), "sha256": ADAPTER_SHA256},
        "prompt_sha256": prompt.sha256,
        "merge_equivalence": equivalence,
        "files": file_hashes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    atomic_json(output_dir / "merged-base-manifest.json", manifest)
    atomic_json(output_dir / "merge-verification.json", equivalence)
    return manifest


def check(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path, mode="dry_run")
    prompt = load_sealed_prompt(config.release_manifest)
    adapter_weights = config.merged_v2.adapter / "adapter_model.safetensors"
    return {
        "status": "configuration_valid",
        "model_id": config.merged_v2.model_id,
        "base_revision": config.merged_v2.revision,
        "adapter_sha256": sha256_path(adapter_weights),
        "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "gpu_merge_pending": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge the sealed SFT-VH adapter on AutoDL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/qmd-qwen3-1.7b-v2-merged"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check(args.config) if args.check else merge_and_verify(args.config, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
