"""Steps 4.1-4.3 of the learnability gate: preference pairs, supply statistics, shallow probes.

  python analyze.py pairs  <dir>   # reads <dir>/samples.scored.jsonl; writes pairs, supply, embed-texts.json
  (bun/tsx embed.ts <dir>)         # embeds embed-texts.json with the index's embedding model
  python analyze.py probes <dir>   # logistic-regression probes on pair feature differences

Pre-registered (plan §4): M = 0.7·nDCG@10 + 0.3·R@30; a pair needs a gap ≥ 0.05; candidates are the 8
format-ok, de-duplicated samples (greedy is a baseline only); ≤ 4 pairs per query (best vs worst, then by gap).
Contrast definition: same with R@30 alone.
"""
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

DELTA, MAX_PAIRS, SEED = 0.05, 4, 42
mode, root = sys.argv[1], Path(sys.argv[2])


def M(m):
    return 0.7 * m["ndcg_at_10"] + 0.3 * m["recall_at_30"]


def R30(m):
    return m["recall_at_30"]


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))] if xs else float("nan")


def candidates(row):
    seen, out = set(), []
    for i, s in enumerate(row["samples"]):
        if s["status"] == "ok" and s["completion"] not in seen:
            seen.add(s["completion"])
            out.append((i, s))
    return out


def make_pairs(cands, score):
    """Up to MAX_PAIRS (chosen_idx, rejected_idx, gap): best vs worst first, then by gap."""
    allp = sorted(((score(a["metrics"]) - score(b["metrics"]), ia, ib)
                   for ia, a in cands for ib, b in cands if ia != ib), reverse=True)
    allp = [(ia, ib, gap) for gap, ia, ib in allp if gap >= DELTA - 1e-12]
    return allp[:MAX_PAIRS]  # the largest gap is exactly best vs worst


rows = [json.loads(line) for line in open(root / "samples.scored.jsonl")]
sources = list(dict.fromkeys(r["source_id"] for r in rows))

if mode == "pairs":
    pairs, supply = [], {}
    for sid in sources + ["macro"]:
        supply[sid] = {}
    for split in ("all", "test"):
        for sid in sources:
            rs = [r for r in rows if r["source_id"] == sid and (split == "all" or r["split"] == "test")]
            allc = [s for r in rs for s in [r["greedy"], *r["samples"]]]
            samp = [s for r in rs for s in r["samples"]]
            with_main = with_r30 = zero_var = 0
            ranges, d_greedy, d_rand, d_oracle, r_greedy, r_rand, r_oracle = [], [], [], [], [], [], []
            for r in rs:
                c = candidates(r)
                pm, pr = make_pairs(c, M), make_pairs(c, R30)
                with_main += bool(pm)
                with_r30 += bool(pr)
                ms = [M(s["metrics"]) for s in r["samples"] if s["status"] == "ok"]
                zero_var += len(set(ms)) <= 1
                if ms:
                    ranges.append(max(ms) - min(ms))
                raw = r["greedy"]["raw"]
                d_greedy.append(M(r["greedy"]["metrics"]) - M(raw))
                d_rand.append(st.mean(M(s["metrics"]) for s in r["samples"]) - M(raw))
                d_oracle.append(max(M(s["metrics"]) for s in r["samples"]) - M(raw))
                r_greedy.append(R30(r["greedy"]["metrics"]) - R30(raw))
                r_rand.append(st.mean(R30(s["metrics"]) for s in r["samples"]) - R30(raw))
                r_oracle.append(max(R30(s["metrics"]) for s in r["samples"]) - R30(raw))
                if split == "all":
                    for kind, ps in (("main", pm), ("r30", pr)):
                        for a, b, gap in ps:
                            pairs.append({"source_id": sid, "qid": r["qid"], "split": r["split"], "kind": kind,
                                          "chosen": a, "rejected": b, "gap": round(gap, 6)})
            n = len(rs)
            supply[sid][split] = {
                "queries": n,
                "format_ok_all9": round(sum(s["status"] == "ok" for s in allc) / len(allc), 4),
                "format_ok_samples": round(sum(s["status"] == "ok" for s in samp) / len(samp), 4),
                "truncated": sum(s["truncated"] for s in allc),
                "q_with_pair_main": round(with_main / n, 4),
                "q_with_pair_r30": round(with_r30 / n, 4),
                "zero_var_groups": round(zero_var / n, 4),
                "range_M_p50": round(pct(ranges, .5), 4), "range_M_p90": round(pct(ranges, .9), 4),
                "dM_greedy": round(st.mean(d_greedy), 4), "dM_random": round(st.mean(d_rand), 4),
                "dM_oracle8": round(st.mean(d_oracle), 4),
                "dR30_greedy": round(st.mean(r_greedy), 4), "dR30_random": round(st.mean(r_rand), 4),
                "dR30_oracle8": round(st.mean(r_oracle), 4),
            }
        keys = [k for k, v in supply[sources[0]][split].items() if isinstance(v, float)]
        supply["macro"][split] = {k: round(st.mean(supply[s][split][k] for s in sources), 4) for k in keys}

    with open(root / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    (root / "supply.json").write_text(json.dumps(supply, indent=2))

    # texts for the embedding probe: each query + every hyde/vec line of every candidate in a pair
    texts = {r["query"] for r in rows}
    for r in rows:
        for s in r["samples"]:
            if s["status"] == "ok":
                for line in s["completion"].splitlines():
                    if line.startswith(("hyde: ", "vec: ")):
                        texts.add(line.split(": ", 1)[1])
    (root / "embed-texts.json").write_text(json.dumps(sorted(texts), ensure_ascii=False))

    cols = ["queries", "format_ok_all9", "q_with_pair_main", "q_with_pair_r30", "zero_var_groups",
            "range_M_p50", "range_M_p90", "dM_greedy", "dM_random", "dM_oracle8", "dR30_greedy", "dR30_oracle8"]
    for split in ("all", "test"):
        print(f"\n== supply ({split})")
        print(f"{'source':14}" + "".join(f"{c[:15]:>16}" for c in cols))
        for sid in sources + ["macro"]:
            v = supply[sid][split]
            print(f"{sid:14}" + "".join(f"{v.get(c, ''):>16}" for c in cols))
    by = {}
    for p in pairs:
        by.setdefault((p["kind"], p["split"]), 0)
        by[(p["kind"], p["split"])] += 1
    print("\npairs:", {f"{k}/{s}": n for (k, s), n in sorted(by.items())}, f"embed texts: {len(texts)}")

elif mode == "probes":
    texts = json.loads((root / "embed-texts.json").read_text())
    vecs = np.fromfile(root / "embed-vectors.f32", dtype=np.float32).reshape(len(texts), -1)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    emb = {t: vecs[i] for i, t in enumerate(texts)}
    STOP = set("the a an of to in for and or on with is are be what how why does do can i my it this that as by "
               "from at if when which who whose about into than then there their your you we our".split())

    def words(t):
        return [w for w in re.findall(r"[a-z0-9]+", t.lower())]

    def content(t):
        return {w for w in words(t) if w not in STOP and len(w) > 2}

    def features(query, s):
        lines = [l.split(": ", 1) for l in s["completion"].splitlines() if l.startswith(("hyde: ", "vec: "))]
        hyde = " ".join(t for k, t in lines if k == "hyde")
        vec = [t for k, t in lines if k == "vec"]
        q = content(query)
        exp = content(hyde + " " + " ".join(vec))
        vsets = [content(v) for v in vec]
        jac = (len(vsets[0] & vsets[1]) / max(len(vsets[0] | vsets[1]), 1)) if len(vsets) >= 2 else 0.0
        qe = emb[query]
        cos_h = float(qe @ emb[hyde]) if hyde in emb else 0.0
        cos_v = float(np.mean([qe @ emb[v] for v in vec])) if vec else 0.0
        cos_vv = float(emb[vec[0]] @ emb[vec[1]]) if len(vec) >= 2 else 0.0
        return {
            "length": [len(words(hyde)), s["tokens"], len(vec)],
            "lexical": [len(q & content(hyde)) / max(len(q), 1),
                        len(q & set().union(*vsets)) / max(len(q), 1) if vsets else 0.0,
                        len(exp - q) / max(len(exp), 1), jac],
            "embedding": [cos_h, cos_v, cos_vv],
        }

    by_q = {(r["source_id"], r["qid"]): r for r in rows}
    pairs = [json.loads(line) for line in open(root / "pairs.jsonl")]
    rng = random.Random(SEED)
    sets = {"length": ["length"], "+lexical": ["length", "lexical"], "+embedding": ["length", "lexical", "embedding"]}
    report = {}
    for kind in ("main", "r30"):
        data = {"train": [], "test": []}
        for p in pairs:
            if p["kind"] != kind:
                continue
            r = by_q[(p["source_id"], p["qid"])]
            fa, fb = features(r["query"], r["samples"][p["chosen"]]), features(r["query"], r["samples"][p["rejected"]])
            flip = rng.random() < 0.5  # random orientation so the label is balanced
            first, second, y = (fb, fa, 0) if flip else (fa, fb, 1)
            data[p["split"]].append((first, second, y, (p["source_id"], p["qid"])))
        report[kind] = {}
        for name, groups in sets.items():
            def X(part):
                return np.array([sum((a[g] for g in groups), []) for a, _, _, _ in part]) - \
                       np.array([sum((b[g] for g in groups), []) for _, b, _, _ in part])
            Xtr, Xte = X(data["train"]), X(data["test"])
            ytr = np.array([y for *_, y, _ in data["train"]], float)
            yte = np.array([y for *_, y, _ in data["test"]], float)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            # L2 logistic regression without intercept (antisymmetric: f(a,b) = -f(b,a)), Newton steps
            w = np.zeros(Xtr.shape[1])
            for _ in range(50):
                p_ = 1 / (1 + np.exp(-Xtr @ w))
                grad = Xtr.T @ (p_ - ytr) / len(ytr) + 1e-3 * w
                H = (Xtr.T * (p_ * (1 - p_))) @ Xtr / len(ytr) + 1e-3 * np.eye(len(w))
                w -= np.linalg.solve(H, grad)
            hit = ((Xte @ w > 0) == (yte == 1)).astype(float)
            qkeys = sorted({k for *_, k in data["test"]})
            per_q = {k: [] for k in qkeys}
            for h, (*_, k) in zip(hit, data["test"]):
                per_q[k].append(h)
            qacc = np.array([np.mean(per_q[k]) for k in qkeys])
            brng = np.random.default_rng(SEED)
            boots = [qacc[brng.integers(0, len(qacc), len(qacc))].mean() for _ in range(10000)]
            report[kind][name] = {
                "train_pairs": len(ytr), "test_pairs": len(yte), "test_queries": len(qkeys),
                "train_acc": round(float(((Xtr @ w > 0) == (ytr == 1)).mean()), 4),
                "test_acc_pairs": round(float(hit.mean()), 4),
                "test_acc_query_macro": round(float(qacc.mean()), 4),
                "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                "weights": dict(zip([f"{g}{i}" for g in groups for i in range(len(fa[g]))], np.round(w, 3).tolist())),
            }
            v = report[kind][name]
            print(f"{kind:5} {name:11} train_pairs={v['train_pairs']:4} test_pairs={v['test_pairs']:4} "
                  f"train_acc={v['train_acc']:.3f} test_acc(pairs)={v['test_acc_pairs']:.3f} "
                  f"test_acc(query macro)={v['test_acc_query_macro']:.3f} CI95={v['ci95']}")
    (root / "probes.json").write_text(json.dumps(report, indent=2))
