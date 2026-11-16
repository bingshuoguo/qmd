"""Sample G completions per qid from the bf16 merged SFT model, GRPO-style.

GRPO rollout conditions: plain temperature sampling (no top-k/top-p/presence
penalty, no grammar), prompt rendered exactly as in SFT training.

  python sample_bf16.py --model <merged dir> --out <file.jsonl> --G 8 --temperature 0.7 \
      --source fiqa-test=<bench dir>:<n> [--source ...] [--seed 0] [--batch-qids 2]

Resumable: qids already in --out are skipped.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--template", required=True, help="file with the V2-VH prompt template ({query})")
ap.add_argument("--train-sample", required=True, help="an SFT train jsonl, used to assert prompt rendering")
ap.add_argument("--source", action="append", required=True, help="id=<bench dir>:<n qids>")
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max-new-tokens", type=int, default=640)
ap.add_argument("--batch-qids", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit-batches", type=int, default=0)
ap.add_argument("--qids", default=None, help="TSV of source_id<TAB>qid; restricts sampling to these")
args = ap.parse_args()

template = Path(args.template).read_text()
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"


def render(query: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": template.replace("{query}", query)}],
        tokenize=False, add_generation_prompt=True,
    )


# Prompt rendering must match SFT training byte for byte.
train_row = json.loads(next(open(args.train_sample)))
assert render(train_row["query"]) == train_row["prompt"], "prompt rendering differs from SFT training"

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
eos_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}

done = set()
out_path = Path(args.out)
if out_path.exists():
    done = {(r["source_id"], r["qid"]) for r in map(json.loads, open(out_path))}

rng = random.Random(args.seed)
allow = None
if args.qids:
    allow = {tuple(l.rstrip("\n").split("\t")) for l in open(args.qids) if l.strip()}
jobs = []
for spec in args.source:
    sid, rest = spec.split("=", 1)
    bench, n = rest.rsplit(":", 1)
    queries = [json.loads(l) for l in open(Path(bench) / "queries.jsonl")]
    rels = {}
    for i, line in enumerate(open(Path(bench) / "qrels.tsv")):
        p = line.rstrip("\n").split("\t")
        if i == 0 and not p[-1].lstrip("-").isdigit():
            continue
        if len(p) >= 3 and int(p[2]) == 1:
            rels.setdefault(p[0], set()).add(p[1])
    scorable = [q for q in queries if q["qid"] in rels]
    if allow is not None:
        scorable = [q for q in scorable if (sid, q["qid"]) in allow]
    rng.shuffle(scorable)
    jobs += [(sid, q["qid"], q["query"]) for q in scorable[: int(n)]]

todo = [j for j in jobs if (j[0], j[1]) not in done]
print(f"device={device} total={len(jobs)} todo={len(todo)} G={args.G} T={args.temperature}", flush=True)

torch.manual_seed(args.seed)
started, n_done = time.time(), 0
with open(out_path, "a") as f:
    for b in range(0, len(todo), args.batch_qids):
        if args.limit_batches and b // args.batch_qids >= args.limit_batches:
            break
        batch = todo[b:b + args.batch_qids]
        enc = tok([render(q) for _, _, q in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0,
                num_return_sequences=args.G, max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        new = gen[:, enc["input_ids"].shape[1]:].tolist()
        for i, (sid, qid, query) in enumerate(batch):
            samples = []
            for ids in new[i * args.G:(i + 1) * args.G]:
                # trim at first eos; truncated == no eos within budget
                cut = next((k for k, t in enumerate(ids) if t in eos_ids), None)
                body = ids if cut is None else ids[:cut]
                samples.append({"completion": tok.decode(body, skip_special_tokens=True),
                                "tokens": len(body), "truncated": cut is None})
            f.write(json.dumps({"source_id": sid, "qid": qid, "query": query, "samples": samples}, ensure_ascii=False) + "\n")
            f.flush()
        n_done += len(batch)
        el = time.time() - started
        print(f"{n_done}/{len(todo)} elapsed={el:.0f}s per_qid={el / n_done:.1f}s eta={el / n_done * (len(todo) - n_done) / 60:.0f}min", flush=True)
