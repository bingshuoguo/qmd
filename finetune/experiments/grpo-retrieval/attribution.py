"""Attribution analyses behind RESULTS.md §5 (why GRPO did not help). Reads archived artifacts only.

  finetune/.venv/bin/python finetune/experiments/grpo-retrieval/attribution.py [artifacts/grpo-retrieval]

A  probe-BoN-8: pick from the 8 samples with the +embedding shallow probe (the only signal that generalised)
B  reward roughness: |ΔM| between two completions vs their word Jaccard (gate-v1 within group; eval SFT vs GRPO)
C  variance split of sample rewards: between-query vs within-query (GRPO only sees the latter)
D  GRPO vs SFT per query on the eval sets: win / loss / tie
E  training diagnostics from trainer_state.json (first vs last 50 steps) and on-policy reward slope
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/grpo-retrieval")
STOP = set("the a an of to in for and or on with is are be what how why does do can i my it this that as by from at if "
           "when which who whose about into than then there their your you we our".split())


def M(m):
    return 0.7 * m["ndcg_at_10"] + 0.3 * m["recall_at_30"]


def words(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def content(t):
    return {w for w in words(t) if w not in STOP and len(w) > 2}


def jaccard(a, b):
    a, b = set(words(a)), set(words(b))
    return len(a & b) / max(len(a | b), 1)


def boot_macro(groups, n=10000, seed=42):
    """Mean of per-group means; 95% CI by bootstrap stratified by group."""
    g = np.random.default_rng(seed)
    vals = [np.asarray(v) for v in groups]
    b = np.mean([v[g.integers(0, len(v), (n, len(v)))].mean(1) for v in vals], axis=0)
    return float(np.mean([v.mean() for v in vals])), np.percentile(b, [2.5, 97.5])


gate = [json.loads(line) for line in open(root / "gate-v1/samples.scored.jsonl")]

# ---- A. probe-BoN-8 (same features and fit as gate/analyze.py probes, +embedding set) ----
texts = json.loads((root / "gate-v1/embed-texts.json").read_text())
vecs = np.fromfile(root / "gate-v1/embed-vectors.f32", dtype=np.float32).reshape(len(texts), -1)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
emb = {t: vecs[i] for i, t in enumerate(texts)}


def features(q, s):
    lines = [l.split(": ", 1) for l in s["completion"].splitlines() if l.startswith(("hyde: ", "vec: "))]
    hyde = " ".join(t for k, t in lines if k == "hyde")
    vec = [t for k, t in lines if k == "vec"]
    Q, vs = content(q), [content(v) for v in vec]
    exp = content(hyde + " " + " ".join(vec))
    jac = len(vs[0] & vs[1]) / max(len(vs[0] | vs[1]), 1) if len(vs) >= 2 else 0.0
    qe = emb[q]
    cos_h = float(qe @ emb[hyde]) if hyde in emb else 0.0
    cos_v = float(np.mean([qe @ emb[v] for v in vec])) if vec and all(v in emb for v in vec) else 0.0
    cos_vv = float(emb[vec[0]] @ emb[vec[1]]) if len(vec) >= 2 and vec[0] in emb and vec[1] in emb else 0.0
    return np.array([len(words(hyde)), s["tokens"], len(vec), len(Q & content(hyde)) / max(len(Q), 1),
                     len(Q & set().union(*vs)) / max(len(Q), 1) if vs else 0.0, len(exp - Q) / max(len(exp), 1),
                     jac, cos_h, cos_v, cos_vv])


by_q = {(r["source_id"], r["qid"]): r for r in gate}
rng = random.Random(42)
X, y = [], []
for p in map(json.loads, open(root / "gate-v1/pairs.jsonl")):
    if p["kind"] != "main" or p["split"] != "train":
        continue
    r = by_q[(p["source_id"], p["qid"])]
    d = features(r["query"], r["samples"][p["chosen"]]) - features(r["query"], r["samples"][p["rejected"]])
    flip = rng.random() < 0.5
    X.append(-d if flip else d)
    y.append(0.0 if flip else 1.0)
X, y = np.array(X), np.array(y)
sd = X.std(0) + 1e-9
Xs, w = X / sd, np.zeros(X.shape[1])
for _ in range(50):  # L2 logistic regression, no intercept, Newton steps
    p_ = 1 / (1 + np.exp(-Xs @ w))
    H = (Xs.T * (p_ * (1 - p_))) @ Xs / len(y) + 1e-3 * np.eye(len(w))
    w -= np.linalg.solve(H, Xs.T @ (p_ - y) / len(y) + 1e-3 * w)
w = w / sd

per = {}
for r in gate:
    ok = [s for s in r["samples"] if s["status"] == "ok"]
    if r["split"] != "test" or not ok:
        continue
    g = M(r["greedy"]["metrics"])
    pick = max(ok, key=lambda s: w @ features(r["query"], s))
    per.setdefault(r["source_id"], []).append(
        (M(pick["metrics"]) - g, np.mean([M(s["metrics"]) for s in r["samples"]]) - g,
         max(M(s["metrics"]) for s in r["samples"]) - g))
print("A. selection from the 8 samples, vs greedy (gate-v1 test, source macro)")
for i, name in enumerate(["probe-BoN-8", "random sample", "oracle best-of-8"]):
    mean, ci = boot_macro([[t[i] for t in v] for v in per.values()])
    print(f"   {name:17} {mean * 100:+.2f}pp [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]")
mean, ci = boot_macro([[t[0] - t[1] for t in v] for v in per.values()])
print(f"   probe-BoN − random {mean * 100:+.2f}pp [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]")

# ---- B. reward roughness ----
print("B. |ΔM| between two completions by word Jaccard")
bins = {}
for r in gate:
    ok = [s for s in r["samples"] if s["status"] == "ok"]
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            if ok[i]["completion"] != ok[j]["completion"]:
                b = min(int(jaccard(ok[i]["completion"], ok[j]["completion"]) * 10), 9)
                bins.setdefault(b, []).append(abs(M(ok[i]["metrics"]) - M(ok[j]["metrics"])))
for b in sorted(bins):
    v = np.array(bins[b])
    if len(v) >= 100:
        print(f"   gate-v1 within group  J {b / 10:.1f}-{(b + 1) / 10:.1f}  n={len(v):5}  mean|ΔM| {v.mean() * 100:.2f}pp  "
              f"P(|ΔM|>=5pp) {np.mean(v >= 0.05):.2f}")


def load_eval(name):
    return {(r["source_id"], r["qid"]): r["greedy"] for r in map(json.loads, open(root / f"grpo-runs/eval/{name}"))}


sft, grpo = load_eval("sft.scored.jsonl"), load_eval("grpo-s42-300.scored.jsonl")
keys = sorted(sft)
eb = {}
for k in keys:
    a, b = sft[k]["completion"], grpo[k]["completion"]
    J = jaccard(a, b)
    lab = "identical" if a == b else ">=0.9" if J >= .9 else "0.7-0.9" if J >= .7 else "0.5-0.7" if J >= .5 else "<0.5"
    eb.setdefault(lab, []).append(M(grpo[k]["metrics"]) - M(sft[k]["metrics"]))
for lab in ["identical", ">=0.9", "0.7-0.9", "0.5-0.7", "<0.5"]:
    v = np.array(eb.get(lab, []))
    if len(v):
        print(f"   eval SFT vs GRPO  {lab:9} n={len(v):4}  mean ΔM {v.mean() * 100:+.2f}pp  "
              f"mean|ΔM| {abs(v).mean() * 100:.2f}pp  P(|ΔM|>=5pp) {np.mean(abs(v) >= 0.05):.2f}")

# ---- C. variance split ----
groups = [[M(s["metrics"]) for s in r["samples"] if s["status"] == "ok"] for r in gate]
groups = [g for g in groups if len(g) > 1]
between, within = np.var([np.mean(g) for g in groups]), np.mean([np.var(g) for g in groups])
print(f"C. sample-reward variance: between-query {between:.4f}  within-query {within:.4f}  "
      f"(within share {within / (between + within) * 100:.1f}%)")

# ---- D. per-query GRPO vs SFT ----
d = np.array([M(grpo[k]["metrics"]) - M(sft[k]["metrics"]) for k in keys])
print(f"D. eval per query: GRPO better {np.mean(d > 1e-9) * 100:.1f}%  worse {np.mean(d < -1e-9) * 100:.1f}%  "
      f"tie {np.mean(abs(d) <= 1e-9) * 100:.1f}%  |d|>=0.1 better {(d >= 0.1).sum()} worse {(d <= -0.1).sum()}  "
      f"mean|d| {abs(d).mean() * 100:.2f}pp  completions changed {np.mean([sft[k]['completion'] != grpo[k]['completion'] for k in keys]) * 100:.1f}%")

# ---- E. training diagnostics ----
hist = [h for h in json.loads((root / "grpo-runs/s42-300/trainer_state.json").read_text())["log_history"] if "loss" in h]
print(f"E. training ({len(hist)} steps), first 50 → last 50")
for k in ["rewards/delta_m/mean", "reward_std", "kl", "entropy", "grad_norm", "clip_ratio/region_mean",
          "completions/mean_length", "rewards/format_ok/mean"]:
    v = np.array([float(h[k]) for h in hist])
    print(f"   {k:26} {v[:50].mean():.4f} → {v[-50:].mean():.4f}")
r = np.array([float(h["rewards/delta_m/mean"]) for h in hist])
x = np.arange(len(r))
g = np.random.default_rng(42)
PP_PER_100_STEPS = 100 * 100  # slope is ΔM per step; ×100 steps, ×100 for percentage points
slopes = [np.polyfit(x[i], r[i], 1)[0] for i in g.integers(0, len(x), (2000, len(x)))]
lo, hi = np.percentile(slopes, [2.5, 97.5]) * PP_PER_100_STEPS
tokens = np.mean([float(h["completions/mean_length"]) for h in hist])
kl_tok = np.mean([float(h["kl"]) for h in hist[-50:]])
print(f"   on-policy ΔM slope per 100 steps {np.polyfit(x, r, 1)[0] * PP_PER_100_STEPS:+.2f}pp [{lo:+.2f}, {hi:+.2f}]; "
      f"KL per sequence ≈ {kl_tok:.4f} × {tokens:.0f} tokens = {kl_tok * tokens:.2f} nats")
