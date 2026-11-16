"""GRPO on the retrieval reward (Phase 2 draft reward), TRL 1.0 + LoRA.

Reward (phase2-reward-offline/README.md §1):
  format_error / truncated -> -1
  ok                       -> ΔM - P_len,  M = 0.7·nDCG@10 + 0.3·R@30,
                              P_len = 0.2·clip((tokens - 320) / 320, 0, 1)

The reward server (reward-server.ts) must be running.

  python grpo_train.py --model <merged sft> --queries <jsonl: source_id, qid, query> \
    --template phase2-reward-offline/template.txt --train-sample <sft-train.jsonl> \
    --out <dir> --max-steps 2
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_client import RewardClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--queries", required=True, help="jsonl with source_id, qid, query")
ap.add_argument("--template", required=True)
ap.add_argument("--train-sample", required=True, help="SFT train jsonl, used to assert prompt rendering")
ap.add_argument("--out", required=True)
ap.add_argument("--max-steps", type=int, default=200)
ap.add_argument("--queries-per-step", type=int, default=8)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--url", default="http://127.0.0.1:8787")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
template = Path(args.template).read_text()


def render(query: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": template.replace("{query}", query)}],
        tokenize=False, add_generation_prompt=True,
    )


train_row = json.loads(next(open(args.train_sample)))
assert render(train_row["query"]) == train_row["prompt"], "prompt rendering differs from SFT training"

rows = [json.loads(line) for line in open(args.queries) if line.strip()]
dataset = Dataset.from_list([
    {"prompt": render(r["query"]), "source_id": r["source_id"], "qid": r["qid"]} for r in rows
])

client = RewardClient(args.url)
eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
last: dict = {}  # results of the latest scoring call, read by the zero-weight logging funcs


def M(m: dict) -> float:
    return 0.7 * m["ndcg_at_10"] + 0.3 * m["recall_at_30"]


def retrieval_reward(prompts, completions, completion_ids, source_id, qid, **_):
    items, lengths = [], []
    for text, ids, sid, q in zip(completions, completion_ids, source_id, qid):
        cut = next((k for k, t in enumerate(ids) if t in eos_ids), None)
        lengths.append(len(ids) if cut is None else cut)
        items.append({"source_id": sid, "qid": q, "completion": text, "truncated": cut is None})
    results = client.score(items)
    last["ok"] = [r["status"] == "ok" for r in results]
    last["delta_m"] = [M(r["metrics"]) - M(r["raw_metrics"]) for r in results]
    return [
        d - 0.2 * min(max((n - 320) / 320, 0.0), 1.0) if ok else -1.0
        for ok, d, n in zip(last["ok"], last["delta_m"], lengths)
    ]


# Logged only (weight 0): TRL reports rewards/<name>/mean per step.
def format_ok(prompts, completions, **_):
    return [float(x) for x in last["ok"]]


def delta_m(prompts, completions, **_):
    return last["delta_m"]


config = GRPOConfig(
    output_dir=args.out,
    max_steps=args.max_steps,
    learning_rate=args.lr,
    per_device_train_batch_size=args.queries_per_step * args.G,
    gradient_accumulation_steps=1,
    num_generations=args.G,
    max_completion_length=640,
    temperature=0.7,
    top_p=1.0,
    top_k=0,  # the model's generation_config sets top_k=20; disable it explicitly
    beta=args.beta,
    loss_type="dapo",
    scale_rewards="batch",
    mask_truncated_completions=False,
    reward_weights=[1.0, 0.0, 0.0],
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    save_steps=50,
    report_to="none",
    seed=args.seed,
)

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    reward_funcs=[retrieval_reward, format_ok, delta_m],
    args=config,
    train_dataset=dataset,
    peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
trainer.save_model(str(Path(args.out) / "final-adapter"))
