#!/usr/bin/env python3
"""Official tobil model vs our SFT: per-arm metrics and paired, dataset-stratified bootstrap.

  finetune/.venv/bin/python finetune/experiments/official-model-eval/analyze.py <bench-root>

<bench-root> is artifacts/official-model-eval/bench-rr (rerank path) or bench-norr (path A).
Per-query metrics are recomputed from runs/results/*.jsonl rankings + qrels.tsv, same
definitions as merge-eval bootstrap-analysis.py (graded nDCG@10, binary recall).
Arms missing on disk are skipped; a comparison runs only on datasets where both arms exist.
"""
import glob, json, math, sys
import numpy as np

ROOT = sys.argv[1]
GROUPS = {
    "main": [("fiqa", "qmd-eval-v1-fiqa-test"), ("scifact", "qmd-expansion-scifact-v1"),
             ("android", "qmd-eval-v1-cqadup-android")],
    "ood": [("webmasters", "qmd-eval-v1-cqadup-webmasters"),
            ("freshstack", "qmd-nf-freshstack-v1-freshstack-test")],
}
# arm -> run-name pattern; our SFT Q8 arm is the frozen G0 profile-D file on the main sets
ARMS = {
    "raw": lambda t: "raw",
    "sft": lambda t: f"g0d-{t}-*" if t in ("fiqa", "scifact", "android") else f"sftq8-{t}-*",
    "offq8": lambda t: f"offq8-{t}-*",
    "offvh": lambda t: f"offvh-{t}-*",
    "offprod": lambda t: f"offprod-{t}-*",
    "offfix": lambda t: f"offfix-{t}-*",
}
COMPARISONS = [
    ("offq8", "raw"), ("sft", "raw"), ("sft", "offq8"),
    ("offvh", "offq8"), ("sft", "offvh"),
    ("offprod", "raw"), ("offq8", "offprod"),
    ("offfix", "offprod"),
]
METRICS = ("ndcg10", "r10", "r30")
N_BOOT, SEED = 20000, 42


def qrels(bench):
    rels = {}
    for i, line in enumerate(open(f"{bench}/qrels.tsv")):
        p = line.split()
        if i == 0 and not p[-1].isdigit():
            continue
        if len(p) == 3:
            rels.setdefault(p[0], {})[p[1]] = int(p[2])
    return rels


def per_query(row, rel):
    if row["retrieval_status"] != "ok":
        raise SystemExit(f"retrieval error for qid {row['qid']}")
    ranked = [x["doc_id"] for x in row["ranking"]]
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
    total = sum(1 for v in rel.values() if v > 0)
    rec = lambda k: sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / total
    return (dcg / idcg if idcg else 0.0, rec(10), rec(30))


def load(bench, pattern):
    files = glob.glob(f"{bench}/runs/results/{pattern}.jsonl")
    if not files:
        return None
    assert len(files) == 1, files
    return {r["qid"]: r for r in map(json.loads, open(files[0]))}


def boot(per_ds):
    """Mean over datasets of per-dataset mean delta; stratified resampling within each dataset."""
    rng = np.random.default_rng(SEED)
    obs = np.mean([d.mean() for d in per_ds])
    draws = np.mean([d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1) for d in per_ds], axis=0)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return obs, lo, hi


for group, sets in GROUPS.items():
    data = {}  # tag -> arm -> np.array[q, 3]
    for tag, d in sets:
        bench = f"{ROOT}/{d}"
        rels = qrels(bench)
        loaded = {}
        for arm, pat in ARMS.items():
            rows = load(bench, pat(tag))
            if rows is None:
                continue
            missing = [q for q in rels if q not in rows]
            if missing:
                print(f"[skip] {tag}/{arm}: {len(missing)} queries missing (incomplete run)")
                continue
            loaded[arm] = rows
        # A query whose retrieval failed in any arm is dropped from every arm (keeps pairing).
        bad = {q for rows in loaded.values() for q in rels if rows[q]["retrieval_status"] != "ok"}
        if bad:
            print(f"[note] {tag}: excluded {sorted(bad)} (retrieval error in some arm)")
        qids = sorted(set(rels) - bad)
        for arm, rows in loaded.items():
            data.setdefault(tag, {})[arm] = np.array([per_query(rows[q], rels[q]) for q in qids])
    if not data:
        continue
    print(f"\n===== {group} ({ROOT.rstrip('/').split('/')[-1]}) =====")
    print(f"{'arm':8s} " + " ".join(f"{t:>26s}" for t in data))
    for arm in ARMS:
        cells = []
        for tag in data:
            a = data[tag].get(arm)
            cells.append("—".rjust(26) if a is None else
                         ("nDCG %.4f R30 %.4f" % (a[:, 0].mean(), a[:, 2].mean())).rjust(26))
        print(f"{arm:8s} " + " ".join(cells))
    print(f"\n{'comparison':16s} {'metric':7s} {'Δ macro':>9s} {'95% CI':>20s}  verdict   per-dataset Δ")
    for a, b in COMPARISONS:
        tags = [t for t in data if a in data[t] and b in data[t]]
        if not tags:
            continue
        for m, metric in enumerate(METRICS):
            per_ds = [data[t][a][:, m] - data[t][b][:, m] for t in tags]
            obs, lo, hi = boot(per_ds)
            sig = "SIG" if lo > 0 or hi < 0 else "n.s."
            detail = " ".join(f"{t}:{d.mean():+.4f}" for t, d in zip(tags, per_ds))
            print(f"{a + '-' + b:16s} {metric:7s} {obs:+9.4f}  [{lo:+.4f},{hi:+.4f}]  {sig:5s}   {detail}")
        print()
