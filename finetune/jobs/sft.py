# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "datasets",
#     "bitsandbytes",
#     "torch",
# ]
# ///
"""
SFT training for QMD query expansion (Qwen3-1.7B).

Self-contained script for HuggingFace Jobs:
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/sft.py
"""

import os
import sys
from huggingface_hub import login

# --- Config (inlined from configs/sft.yaml) ---
BASE_MODEL = "Qwen/Qwen3-1.7B"
OUTPUT_MODEL = "tobil/qmd-query-expansion-1.7B-sft"
DATASET = "tobil/qmd-query-expansion-train"
MAX_LENGTH = 512
ASSISTANT_MARKER = "<|im_start|>assistant\n"


def split_rendered_text(text):
    """Split a rendered Qwen conversation after its sole assistant header."""
    occurrences = text.count(ASSISTANT_MARKER)
    if occurrences != 1:
        raise ValueError(
            f"Expected exactly one Qwen assistant header, found {occurrences}."
        )
    boundary = text.index(ASSISTANT_MARKER) + len(ASSISTANT_MARKER)
    prompt, completion = text[:boundary], text[boundary:]
    if not prompt or not completion:
        raise ValueError("Prompt and completion must both be non-empty.")
    return prompt, completion


def tokenize_completion_example(example, tokenizer):
    """Preserve the rendered token sequence and supervise only its completion."""
    prompt = example.get("prompt")
    completion = example.get("completion")
    if not (
        isinstance(prompt, str)
        and prompt
        and isinstance(completion, str)
        and completion
    ):
        text = example.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(
                "Example must contain prompt/completion or a legacy text field."
            )
        prompt, completion = split_rendered_text(text)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Prompt tokenization is not a prefix of prompt + completion tokenization."
        )
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("Completion has no token after tokenization.")
    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Example has {len(input_ids)} tokens, exceeding max_length={MAX_LENGTH}."
        )
    return {
        "input_ids": input_ids,
        "completion_mask": [0] * len(prompt_ids)
        + [1] * (len(input_ids) - len(prompt_ids)),
    }

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and split dataset
print(f"Loading dataset: {DATASET}...")
dataset = load_dataset(DATASET)
train_dataset = dataset["train"]
eval_dataset = None
for validation_name in ("validation", "val", "test"):
    if validation_name in dataset:
        eval_dataset = dataset[validation_name]
        break
if eval_dataset is None:
    split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")
train_dataset = train_dataset.map(
    tokenize_completion_example,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train dataset",
)
eval_dataset = eval_dataset.map(
    tokenize_completion_example,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing eval dataset",
)
print(f"Tokenized: {len(train_dataset)} train, {len(eval_dataset)} eval")

# SFT config
config = SFTConfig(
    output_dir="qmd-query-expansion-1.7B-sft",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_strategy="every_save",

    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=MAX_LENGTH,

    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,

    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    completion_only_loss=True,

    report_to="none",
)

# LoRA: rank 16, all projection layers
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

print("Initializing SFT trainer...")
trainer = SFTTrainer(
    model=BASE_MODEL,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    peft_config=peft_config,
    processing_class=tokenizer,
)

print("Starting SFT training...")
trainer.train()

print("Pushing to Hub...")
trainer.push_to_hub()
print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")

# --- Automatic evaluation ---
_eval_common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_common.py")
if not os.path.exists(_eval_common_path):
    import urllib.request
    _url = "https://huggingface.co/datasets/tobil/hf-cli-jobs-uv-run-scripts/resolve/main/eval_common.py"
    _opener = urllib.request.build_opener()
    _token = os.environ.get("HF_TOKEN", "")
    if _token:
        _opener.addheaders = [("Authorization", f"Bearer {_token}")]
    with open(_eval_common_path, "wb") as _f:
        _f.write(_opener.open(_url).read())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import run_eval

print("\nStarting automatic evaluation...")
trainer.model.eval()
run_eval(trainer.model, tokenizer, "sft")
