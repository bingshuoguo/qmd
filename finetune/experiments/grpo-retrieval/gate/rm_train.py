"""Steps 4.4-4.5 of the learnability gate: train a reward model on preference pairs, evaluate on held-out queries.

Spec: docs/specs/2026-10-21-dpo-preference-learnability-gate.md §8.3-8.4.
  backbone   Qwen/Qwen3-1.7B (no SFT adapter: the RM must not inherit the student's preferences)
  structure  LoRA r=16, alpha=32 + scalar head (Qwen3ForSequenceClassification, num_labels=1)
  input      v2-VH prompt (with query) + candidate completion
  loss       Bradley-Terry: -log sigmoid(r_chosen - r_rejected)
  training   ≤3 epochs; checkpoint chosen by dev pair accuracy (dev = 15% of train-split queries)
  test       read once, after checkpoint selection

  python rm_train.py --dir <gate-v1> --template <template.txt> --out <run dir>
      [--frac 0.5]       # learning curve: keep this share of train queries
      [--min-gap 0.2]    # sensitivity: train only on pairs with gap >= this

Writes <out>/result.json (dev curve, test pair accuracy overall/per source/by gap, RM-BoN-8) and <out>/best/.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=8, help="pairs per optimizer step")
ap.add_argument("--micro", type=int, default=4, help="pairs per forward pass")
ap.add_argument("--evals-per-epoch", type=int, default=2)
ap.add_argument("--frac", type=float, default=1.0)
ap.add_argument("--min-gap", type=float, default=0.0)
ap.add_argument("--dev-frac", type=float, default=0.15)
ap.add_argument("--max-train-pairs", type=int, default=0, help="smoke test only")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

root, out = Path(args.dir), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
log = open(out / "train.log", "a")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def M(m):
    return 0.7 * m["ndcg_at_10"] + 0.3 * m["recall_at_30"]


# ---------------- data ----------------
rows = {(r["source_id"], r["qid"]): r for r in map(json.loads, open(root / "samples.scored.jsonl"))}
pairs = [p for p in map(json.loads, open(root / "pairs.jsonl")) if p["kind"] == "main"]

train_q = sorted({k for k, r in rows.items() if r["split"] == "train"})
rng = random.Random(f"{args.seed}:dev")
dev_q = set()
for sid in sorted({s for s, _ in train_q}):
    qs = [k for k in train_q if k[0] == sid]
    dev_q |= set(rng.sample(qs, round(len(qs) * args.dev_frac)))
fit_q = [k for k in train_q if k not in dev_q]
if args.frac < 1:
    rng = random.Random(f"{args.seed}:frac")
    fit_q = rng.sample(fit_q, round(len(fit_q) * args.frac))
fit_q = set(fit_q)

key = lambda p: (p["source_id"], p["qid"])
train_pairs = [p for p in pairs if key(p) in fit_q and p["gap"] >= args.min_gap - 1e-12]
dev_pairs = [p for p in pairs if key(p) in dev_q]
test_pairs = [p for p in pairs if p["split"] == "test"]
if args.max_train_pairs:
    train_pairs = train_pairs[: args.max_train_pairs]
    dev_pairs = dev_pairs[: args.max_train_pairs // 2]
say(f"queries fit={len(fit_q)} dev={len(dev_q)} test={sum(r['split'] == 'test' for r in rows.values())}  "
    f"pairs train={len(train_pairs)} dev={len(dev_pairs)} test={len(test_pairs)}  args={vars(args)}")

tok = AutoTokenizer.from_pretrained(args.base)
template = Path(args.template).read_text()
prompt_cache = {}


def text(k, idx):
    """idx = sample index, or -1 for greedy."""
    r = rows[k]
    if k not in prompt_cache:
        prompt_cache[k] = tok.apply_chat_template(
            [{"role": "user", "content": template.replace("{query}", r["query"])}],
            tokenize=False, add_generation_prompt=True)
    s = r["greedy"] if idx < 0 else r["samples"][idx]
    return prompt_cache[k] + s["completion"] + "<|im_end|>"


# ---------------- model ----------------
model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1, dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model = get_peft_model(model, LoraConfig(
    task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()  # fp32 master weights for LoRA + head; frozen base stays bf16
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.cuda()
model.print_trainable_parameters()


def forward(texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**enc).logits.squeeze(-1).float()


@torch.no_grad()
def score_texts(items, bs=16):
    model.eval()
    outs = []
    for i in range(0, len(items), bs):
        outs.append(forward([text(k, j) for k, j in items[i:i + bs]]).cpu())
    model.train()
    return torch.cat(outs).tolist() if outs else []


def pair_eval(ps):
    """Accuracy per pair (ties = 0.5), macro over queries; also per source."""
    items = sorted({(key(p), p["chosen"]) for p in ps} | {(key(p), p["rejected"]) for p in ps})
    sc = dict(zip(items, score_texts(items)))
    per_q = {}
    hits = []
    for p in ps:
        a, b = sc[(key(p), p["chosen"])], sc[(key(p), p["rejected"])]
        h = 1.0 if a > b else 0.5 if a == b else 0.0
        hits.append(h)
        per_q.setdefault(key(p), []).append(h)
    qacc = {k: float(np.mean(v)) for k, v in per_q.items()}
    return {"pairs": len(ps), "queries": len(qacc), "acc_pairs": float(np.mean(hits)) if hits else float("nan"),
            "acc_query_macro": float(np.mean(list(qacc.values()))) if qacc else float("nan")}, qacc, sc


def boot_mean(vals, n=10000, seed=42):
    vals = np.asarray(vals)
    g = np.random.default_rng(seed)
    b = vals[g.integers(0, len(vals), (n, len(vals)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


# ---------------- train ----------------
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train_pairs) / args.batch)
total = steps_per_epoch * args.epochs
warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

best, curve, step, t0 = -1.0, [], 0, time.time()
model.train()
for ep in range(args.epochs):
    order = train_pairs[:]
    random.Random(f"{args.seed}:ep{ep}").shuffle(order)
    for b in range(0, len(order), args.batch):
        batch = order[b:b + args.batch]
        opt.zero_grad(set_to_none=True)
        losses, accs = [], []
        for m in range(0, len(batch), args.micro):
            mb = batch[m:m + args.micro]
            r = forward([text(key(p), p["chosen"]) for p in mb] + [text(key(p), p["rejected"]) for p in mb])
            margin = r[:len(mb)] - r[len(mb):]
            loss = -F.logsigmoid(margin).mean() * len(mb) / len(batch)
            loss.backward()
            losses.append(loss.item())
            accs += (margin > 0).float().tolist()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 10 == 0:
            say(f"step {step}/{total} ep {ep} loss {sum(losses):.4f} acc {np.mean(accs):.3f} "
                f"gnorm {gn:.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        if step % eval_every == 0 or step == total:
            dev, _, _ = pair_eval(dev_pairs)
            curve.append({"step": step, "epoch": round(step / steps_per_epoch, 2), **dev})
            say(f"DEV step {step}: acc(query macro) {dev['acc_query_macro']:.4f} acc(pairs) {dev['acc_pairs']:.4f}")
            if dev["acc_query_macro"] > best:
                best = dev["acc_query_macro"]
                model.save_pretrained(out / "best")
                say(f"  new best -> {out / 'best'}")

# ---------------- test (read once, best checkpoint) ----------------
from peft import PeftModel  # noqa: E402

del model, opt
torch.cuda.empty_cache()
base = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1, dtype=torch.bfloat16)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, out / "best").cuda().eval()
reloaded, _, _ = pair_eval(dev_pairs)  # the scalar head must round-trip through save/load
say(f"reloaded best: dev acc(query macro) {reloaded['acc_query_macro']:.4f} (saved {best:.4f})")
assert abs(reloaded["acc_query_macro"] - best) < 0.02, "best checkpoint did not reload correctly"
say("TEST READ 1 (best dev checkpoint)")

test, qacc, _ = pair_eval(test_pairs)
test["ci95_query_macro"] = boot_mean(list(qacc.values()))
by_source = {}
for sid in sorted({k[0] for k in qacc}):
    v = [a for k, a in qacc.items() if k[0] == sid]
    by_source[sid] = {"queries": len(v), "acc_query_macro": float(np.mean(v)), "ci95": boot_mean(v)}
by_gap = {}
for name, ps in (("gap>=0.2", [p for p in test_pairs if p["gap"] >= 0.2]),
                 ("gap<0.2", [p for p in test_pairs if p["gap"] < 0.2])):
    res, qa, _ = pair_eval(ps)
    res["ci95_query_macro"] = boot_mean(list(qa.values()))
    by_gap[name] = res
mid_q = {k for k in qacc if rows[k]["greedy"]["raw"]["recall_at_30"] > 0}
mid = [a for k, a in qacc.items() if k in mid_q]
test_mid = {"queries": len(mid), "acc_query_macro": float(np.mean(mid)), "ci95": boot_mean(mid)}

# RM-BoN-8 over every test query: pick the RM's top format-ok sample (fallback greedy)
test_keys = sorted(k for k, r in rows.items() if r["split"] == "test")
items = [(k, i) for k in test_keys for i, s in enumerate(rows[k]["samples"]) if s["status"] == "ok"]
sc = dict(zip(items, score_texts(items)))
per = {}
for k in test_keys:
    r = rows[k]
    ok = [(i, s) for i, s in enumerate(r["samples"]) if s["status"] == "ok"]
    g = M(r["greedy"]["metrics"])
    pick = M(max(ok, key=lambda t: sc[(k, t[0])])[1]["metrics"]) if ok else g
    rand = float(np.mean([M(s["metrics"]) for s in r["samples"]]))  # format errors score as raw
    oracle = max(M(s["metrics"]) for s in r["samples"])
    per[k] = {"bon_minus_greedy": pick - g, "random_minus_greedy": rand - g,
              "bon_minus_random": pick - rand, "oracle_minus_greedy": oracle - g}


def macro_ci(field, keys):
    """Mean of per-source means; CI by bootstrap stratified by source."""
    srcs = sorted({k[0] for k in keys})
    vals = {s: np.array([per[k][field] for k in keys if k[0] == s]) for s in srcs}
    point = float(np.mean([v.mean() for v in vals.values()]))
    g = np.random.default_rng(42)
    boots = np.mean([v[g.integers(0, len(v), (10000, len(v)))].mean(1) for v in vals.values()], axis=0)
    return {"mean": point, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "by_source": {s: float(v.mean()) for s, v in vals.items()}}


bon = {f: macro_ci(f, test_keys) for f in
       ("bon_minus_greedy", "random_minus_greedy", "bon_minus_random", "oracle_minus_greedy")}
bon_mid = {f: macro_ci(f, [k for k in test_keys if rows[k]["greedy"]["raw"]["recall_at_30"] > 0])
           for f in ("bon_minus_greedy", "bon_minus_random")}

result = {"args": vars(args), "best_dev_acc": best, "dev_curve": curve, "test": test,
          "test_by_source": by_source, "test_by_gap": by_gap, "test_mid_class": test_mid,
          "rm_bon8": bon, "rm_bon8_mid_class": bon_mid, "train_pairs": len(train_pairs),
          "minutes": round((time.time() - t0) / 60, 1)}
(out / "result.json").write_text(json.dumps(result, indent=2))
say(json.dumps({"test": test, "by_source": {s: round(v["acc_query_macro"], 4) for s, v in by_source.items()},
                "bon_minus_greedy": bon["bon_minus_greedy"]["mean"], "ci": bon["bon_minus_greedy"]["ci95"]}))
