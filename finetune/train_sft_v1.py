#!/usr/bin/env python3
"""Completion-only LoRA SFT on the sealed public-distill-v1 release.

Implements docs/superpowers/specs/2026-08-15-qmd-completion-only-sft-training-spec.md.

The trainer only ever sees `input_ids` and `completion_mask`; TRL's collator turns
the mask into `-100` labels so the prompt and the padding never reach the loss.

Usage:
    uv run train_sft_v1.py --config configs/sft-v1.yaml --run-root <data-disk>/runs
    uv run train_sft_v1.py --config configs/sft-v1.yaml --preflight-only
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from dataset.completion import tokenize_completion_example
from dataset.public_distill import atomic_json, sha256_file
from dataset.sft_release import (
    Release,
    assert_prompt_template,
    assert_token_identities,
    load_release,
    preflight_lengths,
)

FINETUNE_ROOT = Path(__file__).resolve().parent

# The only two columns the trainer is allowed to carry into the collator.
TRAINER_COLUMNS = ("input_ids", "completion_mask")

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path, *, allow_4bit: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config must be a mapping")
    for section in ("release", "model", "lora", "training", "precision",
                    "checkpointing", "logging"):
        if section not in config:
            raise ValueError(f"{path}: missing '{section}' section")
    if (config["model"]["load_in_4bit"] or config["model"]["full_finetuning"]) and not allow_4bit:
        raise ValueError("spec section 4 forbids 4-bit loading and full finetuning")
    if config["precision"]["fp16"] or not config["precision"]["bf16"]:
        raise ValueError("spec section 9 requires bf16 and forbids fp16")
    return config


def effective_batch_size(training: dict[str, Any]) -> int:
    return (
        training["per_device_train_batch_size"]
        * training["gradient_accumulation_steps"]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> Any:
    """Tokenize a release split down to exactly input_ids + completion_mask.

    Dropping every other column is what guarantees TRL takes the already-processed
    path and never re-applies a chat template.
    """
    from datasets import Dataset

    dataset = Dataset.from_list(records)
    prepared = dataset.map(
        tokenize_completion_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    if tuple(prepared.column_names) != TRAINER_COLUMNS:
        raise ValueError(
            f"prepared dataset columns {prepared.column_names} != {list(TRAINER_COLUMNS)}"
        )
    return prepared


# ---------------------------------------------------------------------------
# Assertions (spec sections 7 and 10)
# ---------------------------------------------------------------------------


def assert_trainable_parameters(model: Any, target_modules: list[str]) -> dict[str, int]:
    """Section 7: only LoRA A/B on the approved target modules may train."""
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable:
        raise ValueError("no trainable parameters; LoRA was not attached")
    for name in sorted(trainable):
        if "lora_A" not in name and "lora_B" not in name:
            raise ValueError(f"non-LoRA parameter is trainable: {name}")
        if not any(module in name for module in target_modules):
            raise ValueError(f"LoRA parameter outside the approved modules: {name}")
    return trainable


def assert_gradient_checkpointing_disabled(model: Any, args: Any, is_teacher: bool = False) -> None:
    if is_teacher:
        return  # 8B teacher needs gradient checkpointing to fit 24 GB
    """Section 10 item 6: the final state must be off, not the framework default."""
    if getattr(args, "gradient_checkpointing", False):
        raise ValueError("SFTConfig.gradient_checkpointing is enabled")
    enabled = getattr(model, "is_gradient_checkpointing", False)
    if enabled:
        raise ValueError("model reports gradient checkpointing enabled")


def assert_batch_supervision(
    batch: dict[str, Any], records: list[dict[str, list[int]]]
) -> int:
    """Section 10 items 1-3: labels must equal the completion mask, exactly.

    `records` are the pre-collation rows that produced `batch`, in order.
    Returns the number of supervised tokens in the batch.
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    if not (input_ids.shape == labels.shape == attention_mask.shape):
        raise ValueError(
            f"shape mismatch: input_ids {tuple(input_ids.shape)}, "
            f"labels {tuple(labels.shape)}, attention_mask {tuple(attention_mask.shape)}"
        )
    if input_ids.shape[0] != len(records):
        raise ValueError(f"batch holds {input_ids.shape[0]} rows for {len(records)} records")

    supervised = 0
    for row, record in enumerate(records):
        mask = record["completion_mask"]
        length = len(record["input_ids"])
        if len(mask) != length:
            raise ValueError(f"row {row}: completion_mask length != input_ids length")

        row_labels = labels[row].tolist()
        row_inputs = input_ids[row].tolist()
        row_attention = attention_mask[row].tolist()

        for position in range(length):
            if mask[position] == 0:
                if row_labels[position] != IGNORE_INDEX:
                    raise ValueError(
                        f"row {row} position {position}: prompt token is supervised"
                    )
            else:
                if row_labels[position] != row_inputs[position]:
                    raise ValueError(
                        f"row {row} position {position}: completion label != input token"
                    )
                supervised += 1
            if row_attention[position] != 1:
                raise ValueError(f"row {row} position {position}: real token is masked out")

        for position in range(length, len(row_labels)):
            if row_labels[position] != IGNORE_INDEX:
                raise ValueError(f"row {row} position {position}: padding is supervised")
            if row_attention[position] != 0:
                raise ValueError(f"row {row} position {position}: padding is attended")

    if supervised <= 0:
        raise ValueError("batch has no supervised token")
    return supervised


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def git_provenance() -> dict[str, Any]:
    """Section 14: code cleanliness only; data provenance lives in manifests."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=FINETUNE_ROOT
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, cwd=FINETUNE_ROOT
    )
    dirty = [line for line in status.splitlines() if line.strip()]
    return {"git_commit": commit, "git_dirty": bool(dirty), "git_dirty_paths": dirty}


def environment_provenance() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name, key in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("trl", "trl"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
    ):
        try:
            versions[key] = __import__(module_name).__version__
        except Exception:
            versions[key] = None

    # Importing Unsloth after Transformers/TRL mutates CUDA state and can stall
    # an FSDP worker during initialization. Package metadata records its version
    # without executing any of Unsloth's runtime patches.
    try:
        versions["unsloth"] = version("unsloth")
    except PackageNotFoundError:
        versions["unsloth"] = None

    lock = FINETUNE_ROOT / "uv.lock"
    if lock.is_file():
        versions["uv_lock_sha256"] = sha256_file(lock)

    try:
        import torch

        if torch.cuda.is_available():
            versions["cuda"] = torch.version.cuda
            versions["gpu"] = torch.cuda.get_device_name(0)
            versions["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return versions


def build_run_manifest(
    run_id: str,
    config: dict[str, Any],
    release: Release,
    token_stats: dict[str, dict[str, int]],
    trainable: dict[str, int],
    started_at: str,
    finished_at: str | None,
    status: str,
    resume_from: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "qmd-sft-run-v1",
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "resume_from": resume_from,
        "artifact_class": "research_candidate",
        "production_approved": False,
        "code": git_provenance(),
        "environment": environment_provenance(),
        "release": release.provenance(),
        "model": config["model"],
        "lora": config["lora"],
        "training": {
            **config["training"],
            "effective_batch_size": effective_batch_size(config["training"]),
        },
        "precision": config["precision"],
        "checkpointing": config["checkpointing"],
        "logging": config["logging"],
        "trainable_parameters": {
            "count": len(trainable),
            "total": sum(trainable.values()),
        },
        "token_stats": token_stats,
        "paths": {
            "checkpoints": "checkpoints",
            "final_adapter": "final-adapter",
            "tokenizer": "tokenizer",
            "logs": "logs",
            "metrics": "metrics",
        },
        "note": (
            "Training and validation loss are diagnostics only. They must never be "
            "used to claim retrieval gains."
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def resolve_release(config: dict[str, Any]) -> Release:
    manifest = Path(config["release"]["manifest"])
    if not manifest.is_absolute():
        manifest = FINETUNE_ROOT / manifest
    return load_release(
        manifest,
        expected_release_id=config["release"]["release_id"],
        expected_experiment_id=config["release"]["experiment_id"],
    )


def run_preflight(
    config: dict[str, Any], tokenizer: Any
) -> tuple[Release, dict[str, dict[str, int]]]:
    """Everything that must pass before the model is loaded (sections 6 and 10)."""
    release = resolve_release(config)
    assert_token_identities(tokenizer)
    max_length = config["model"]["max_seq_length"]

    stats: dict[str, dict[str, int]] = {}
    for split in (release.train, release.validation):
        assert_prompt_template(split.records, release.prompt_template)
        stats[split.name] = preflight_lengths(split.records, tokenizer, max_length)
    return release, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, help="Directory that holds run folders")
    parser.add_argument("--run-id", type=str, help="Defaults to the release experiment id")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate release, prompts and token lengths, then exit without a GPU",
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="Teacher SFT mode: allows gradient checkpointing and 4-bit for 8B on 24 GB",
    )
    args = parser.parse_args()

    # Teacher mode permits a 4-bit configuration when explicitly requested;
    # the BF16 FSDP teacher configuration keeps this disabled.
    config = load_config(args.config.resolve(), allow_4bit=args.teacher)
    if args.teacher:
        config["_allow_4bit"] = True
        config["_teacher"] = True

    if args.preflight_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["model_id"], revision=config["model"]["revision"]
        )
        release, stats = run_preflight(config, tokenizer)
        print(json.dumps({"release": release.provenance(), "token_stats": stats}, indent=2))
        return 0

    if args.run_root is None:
        parser.error("--run-root is required unless --preflight-only is set")

    return train(config, args)


def train(config: dict[str, Any], args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    import os
    import torch
    from accelerate import PartialState
    from accelerate.utils import DistributedType
    from transformers import set_seed
    from trl import SFTConfig, SFTTrainer

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    precision_cfg = config["precision"]

    distributed_state = PartialState()
    # ``PartialState`` is initialized before SFTTrainer creates Accelerate's
    # FSDP plugin, so at this point it still reports MULTI_GPU. The launcher
    # exposes the requested mode explicitly through this environment variable.
    is_fsdp = os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    if is_fsdp and distributed_state.distributed_type != DistributedType.MULTI_GPU:
        raise RuntimeError(
            "FSDP launch did not initialize a multi-GPU process group; "
            "use configs/accelerate_fsdp2.yaml with accelerate launch"
        )
    set_seed(precision_cfg["seed"])

    run_id = args.run_id or config["release"]["experiment_id"]
    run_dir = args.run_root.resolve() / run_id
    if distributed_state.is_main_process:
        if run_dir.exists() and args.resume_from_checkpoint is None:
            raise ValueError(f"run directory already exists, refusing to overwrite: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    # This is also safe when the script is launched directly (one process),
    # where torch.distributed has not been initialized.
    distributed_state.wait_for_everyone()
    started_at = datetime.now(timezone.utc).isoformat()

    if distributed_state.is_main_process:
        print(f"Loading {model_cfg['model_id']}@{model_cfg['revision'][:12]}...")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # Unsloth defaults to ``device_map="sequential"``. In a DDP launch that
    # default starts every worker at cuda:0, so rank 0 and rank 1 each load a
    # complete model onto GPU 0. Pin DDP replicas to their assigned devices.
    # FSDP is different: it must receive an unplaced model, then shards its
    # parameters over both ranks during Trainer preparation.
    # Passing ``None`` lets Unsloth choose CUDA despite FSDP. An explicit CPU
    # map prevents a full BF16 replica from reaching one GPU before FSDP has
    # allocated its parameter shards.
    device_map = {"": "cpu"} if is_fsdp else {"": local_rank}
    if distributed_state.is_main_process:
        parallel_mode = "FSDP full parameter sharding" if is_fsdp else "DDP replicas"
        print(
            f"Distributed workers: {distributed_state.num_processes}; {parallel_mode}; "
            + ("loading model on CPU before sharding" if is_fsdp
               else f"loading this model replica on cuda:{local_rank}")
        )
    if is_fsdp:
        # Unsloth's trainer moves the complete model onto CUDA before FSDP can
        # shard it. Native Transformers/PEFT keeps the full state on CPU until
        # Accelerate materializes only the FSDP shards.
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["model_id"], revision=model_cfg["revision"],
            trust_remote_code=model_cfg["trust_remote_code"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"], revision=model_cfg["revision"],
            dtype=getattr(torch, model_cfg["dtype"]), device_map={"": "cpu"},
            trust_remote_code=model_cfg["trust_remote_code"],
        )
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"], bias=lora_cfg["bias"],
            target_modules=lora_cfg["target_modules"], task_type="CAUSAL_LM",
        ))
    else:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["model_id"], revision=model_cfg["revision"],
            max_seq_length=model_cfg["max_seq_length"], dtype=getattr(torch, model_cfg["dtype"]),
            load_in_4bit=model_cfg["load_in_4bit"], load_in_8bit=model_cfg.get("load_in_8bit", False),
            full_finetuning=model_cfg["full_finetuning"], trust_remote_code=model_cfg["trust_remote_code"],
            device_map=device_map,
        )

    release, token_stats = run_preflight(config, tokenizer)
    if distributed_state.is_main_process:
        print(
            f"Release {release.release_id}/{release.experiment_id}: "
            f"train={len(release.train.records)} validation={len(release.validation.records)}"
        )
    if distributed_state.is_main_process:
        for name, stats in token_stats.items():
            print(
                f"  {name}: {stats['input_tokens']:,} input / "
                f"{stats['supervised_tokens']:,} supervised tokens, "
                f"longest {stats['longest_sequence']}"
            )

    if not is_fsdp:
        model = FastLanguageModel.get_peft_model(
            model, r=lora_cfg["r"], lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"], bias=lora_cfg["bias"],
            target_modules=lora_cfg["target_modules"],
            use_gradient_checkpointing=precision_cfg["gradient_checkpointing"],
            random_state=precision_cfg["seed"],
        )
    trainable = assert_trainable_parameters(model, lora_cfg["target_modules"])
    if distributed_state.is_main_process:
        print(f"Trainable: {len(trainable)} tensors / {sum(trainable.values()):,} parameters")

    train_dataset = build_dataset(
        release.train.records, tokenizer, model_cfg["max_seq_length"]
    )
    eval_dataset = build_dataset(
        release.validation.records, tokenizer, model_cfg["max_seq_length"]
    )

    sft_config = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        # Section 6: None keeps the collator's silent truncation path unused;
        # length is enforced by the preflight and the model's max_seq_length.
        max_length=None,
        packing=False,
        padding_free=False,
        completion_only_loss=True,
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_ratio=training_cfg["warmup_ratio"],
        num_train_epochs=training_cfg["num_train_epochs"],
        max_steps=training_cfg["max_steps"],
        optim=training_cfg["optim"],
        adam_beta1=training_cfg["adam_beta1"],
        adam_beta2=training_cfg["adam_beta2"],
        adam_epsilon=training_cfg["adam_epsilon"],
        weight_decay=training_cfg["weight_decay"],
        max_grad_norm=training_cfg["max_grad_norm"],
        bf16=precision_cfg["bf16"],
        fp16=precision_cfg["fp16"],
        gradient_checkpointing=precision_cfg["gradient_checkpointing"],
        full_determinism=precision_cfg["full_determinism"],
        seed=precision_cfg["seed"],
        data_seed=precision_cfg["data_seed"],
        eval_strategy=config["checkpointing"]["eval_strategy"],
        save_strategy=config["checkpointing"]["save_strategy"],
        save_total_limit=config["checkpointing"]["save_total_limit"],
        save_only_model=config["checkpointing"]["save_only_model"],
        load_best_model_at_end=config["checkpointing"]["load_best_model_at_end"],
        logging_strategy=config["logging"]["logging_strategy"],
        logging_steps=config["logging"]["logging_steps"],
        logging_dir=str(run_dir / "logs" / "tensorboard"),
        report_to=config["logging"]["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Section 10: first-batch assertions before any optimizer step.
    assert_gradient_checkpointing_disabled(model, sft_config, is_teacher=config.get("_teacher", False))
    probe_records = [train_dataset[index] for index in range(
        min(sft_config.per_device_train_batch_size, len(train_dataset))
    )]
    probe_batch = trainer.data_collator(probe_records)
    supervised = assert_batch_supervision(probe_batch, probe_records)
    if distributed_state.is_main_process:
        print(f"First-batch supervision verified: {supervised} supervised tokens")

    if distributed_state.is_main_process:
        device = next(model.parameters()).device
        if is_fsdp:
            # The trainer performs FSDP wrapping immediately before its first
            # batch. A manual forward here would run before sharding and undo
            # the memory benefit we need to verify.
            print("Skipping unsharded probe; Trainer will run the first batch after FSDP wrapping")
        else:
            model.train()
            outputs = model(**{key: value.to(device) for key, value in probe_batch.items()})
            outputs.loss.backward()
            if not torch.isfinite(outputs.loss):
                raise ValueError(f"probe loss is not finite: {outputs.loss}")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float("inf")
            )
            if not torch.isfinite(grad_norm):
                raise ValueError(f"probe gradient norm is not finite: {grad_norm}")
            print(f"Probe loss {outputs.loss.item():.4f}, gradient norm {grad_norm.item():.4f}")
            model.zero_grad(set_to_none=True)

    if distributed_state.is_main_process:
        atomic_json(
            run_dir / "training-config.json",
            {"config_path": str(args.config), "config": config},
        )
        atomic_json(
            run_dir / "run-manifest.json",
            build_run_manifest(
                run_id, config, release, token_stats, trainable,
                started_at, None, "running", args.resume_from_checkpoint,
            ),
        )

    if distributed_state.is_main_process:
        print("Starting training...")
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = run_dir / "final-adapter"
    if is_fsdp:
        # FSDP state-dict collection is a collective operation: every rank
        # must enter ``save_model``. Restricting this call to rank 0 leaves it
        # waiting forever once the other worker exits.
        trainer.save_model(str(adapter_dir))
        distributed_state.wait_for_everyone()
    elif distributed_state.is_main_process:
        model.save_pretrained(str(adapter_dir))

    if distributed_state.is_main_process:
        tokenizer.save_pretrained(str(run_dir / "tokenizer"))

    if distributed_state.is_main_process:
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        atomic_json(metrics_dir / "train-metrics.json", dict(result.metrics))
        atomic_json(metrics_dir / "validation-metrics.json", {
            "history": [
                entry for entry in trainer.state.log_history if "eval_loss" in entry
            ]
        })
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        atomic_json(logs_dir / "trainer-state.json", {"log_history": trainer.state.log_history})

        manifest = build_run_manifest(
            run_id, config, release, token_stats, trainable, started_at,
            datetime.now(timezone.utc).isoformat(), "completed", args.resume_from_checkpoint,
        )
        adapter_weights = adapter_dir / "adapter_model.safetensors"
        if adapter_weights.is_file():
            manifest["final_adapter_sha256"] = sha256_file(adapter_weights)
        atomic_json(run_dir / "run-manifest.json", manifest)

        print(f"Done. Run artifacts: {run_dir}")
        print(f"run-manifest.json sha256: {sha256_file(run_dir / 'run-manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
