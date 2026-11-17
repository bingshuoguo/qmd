"""Step 3.3: per query, 1 greedy + G samples (pure temperature, top_k=0, top_p=1) on CUDA.

  python sample.py --model <merged sft> --queries <queries.jsonl> --template <template.txt> \
    --train-sample <sft-train.jsonl> --out <samples.jsonl>

Output rows: {source_id, qid, query, split, greedy: {...}, samples: [{completion, tokens, truncated}]}.
Resumable: qids already in --out are skipped.
Evaluation of a trained policy: --adapter <LoRA dir> --G 0 (greedy only).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None, help="LoRA adapter merged onto --model before generating")
ap.add_argument("--queries", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--train-sample", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max-new-tokens", type=int, default=640)
ap.add_argument("--batch-qids", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
template = Path(args.template).read_text()


def render(query: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": template.replace("{query}", query)}],
        tokenize=False, add_generation_prompt=True,
    )


train_row = json.loads(next(open(args.train_sample)))
assert render(train_row["query"]) == train_row["prompt"], "prompt rendering differs from SFT training"

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model = model.to("cuda").eval()
eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

out = Path(args.out)
done = {(r["source_id"], r["qid"]) for r in map(json.loads, open(out))} if out.exists() else set()
todo = [q for q in map(json.loads, open(args.queries)) if (q["source_id"], q["qid"]) not in done]
if args.limit:
    todo = todo[: args.limit]
print(f"todo={len(todo)} done={len(done)} G={args.G} T={args.temperature}", flush=True)


def decode(ids):
    cut = next((k for k, t in enumerate(ids) if t in eos_ids), None)
    body = ids if cut is None else ids[:cut]
    return {"completion": tok.decode(body, skip_special_tokens=True), "tokens": len(body), "truncated": cut is None}


started = time.time()
with open(out, "a") as f:
    for b in range(0, len(todo), args.batch_qids):
        batch = todo[b:b + args.batch_qids]
        enc = tok([render(q["query"]) for q in batch], return_tensors="pt", padding=True).to("cuda")
        width = enc["input_ids"].shape[1]
        common = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
        with torch.no_grad():
            greedy = model.generate(**enc, do_sample=False, **common)
            sampled = []
            if args.G:
                torch.manual_seed(args.seed + b)
                sampled = model.generate(**enc, do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0,
                                         num_return_sequences=args.G, **common)[:, width:].tolist()
        greedy = greedy[:, width:].tolist()
        for i, q in enumerate(batch):
            row = {k: q.get(k) for k in ("source_id", "qid", "query", "split")}
            row["greedy"] = decode(greedy[i])
            row["samples"] = [decode(ids) for ids in sampled[i * args.G:(i + 1) * args.G]]
            row["sample_seed"] = args.seed + b
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        n = b + len(batch)
        el = time.time() - started
        print(f"{n}/{len(todo)} elapsed={el:.0f}s per_qid={el / n:.1f}s eta={el / n * (len(todo) - n) / 60:.0f}min", flush=True)
