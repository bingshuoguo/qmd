"""Why does raw path-A retrieval find zero relevant docs in the top 30 for some qids?

Structural comparison of R@30=0 vs R@30>0 qids per source, plus a sample of
R@30=0 (query, relevant doc title) pairs for manual inspection.

  python3 -I r30zero_analysis.py <train-raw.jsonl> <prepared dir> [--show 8]
"""
import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("prepared")
ap.add_argument("--show", type=int, default=8)
args = ap.parse_args()

DIRS = {"fiqa-train": "fiqa-train", "programmers": "cqadup-programmers", "unix": "cqadup-unix",
        "scifact-train": "qmd-distill-public-v0-scifact-train"}
P = Path(args.prepared)
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on for and or is are was be how what why do does can i my it with this that from at by as if not".split())

raw = {}
for r in map(json.loads, open(args.raw)):
    raw[(r["source_id"], r["qid"])] = r["raw"]


def toks(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def title_body(path):
    text = path.read_text(errors="replace")
    lines = text.split("\n")
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else ""
    return title, text


rng = random.Random(0)
for sid, d in DIRS.items():
    root = P / d
    queries = {q["qid"]: q["query"] for q in map(json.loads, open(root / "queries.jsonl"))}
    path_of = {x["doc_id"]: x["path"] for x in map(json.loads, open(root / "documents.jsonl"))}
    rels = {}
    for line in open(root / "qrels.tsv"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2].strip() == "1":
            rels.setdefault(p[0], []).append(p[1])

    stats = {"zero": [], "hit": []}
    examples = []
    for qid, query in queries.items():
        m = raw.get((sid, qid))
        if m is None or qid not in rels:
            continue
        qt = set(toks(query))
        overlaps, dlens, missing = [], [], 0
        titles = []
        for doc in rels[qid]:
            path = path_of.get(doc)
            if path is None or not (root / "corpus" / path).exists():
                missing += 1
                continue
            title, body = title_body(root / "corpus" / path)
            bt = set(toks(body))
            overlaps.append(len(qt & bt) / len(qt) if qt else 0.0)
            dlens.append(len(body))
            titles.append(title)
        bucket = "zero" if m["recall_at_30"] == 0 else "hit"
        stats[bucket].append({
            "n_rel": len(rels[qid]), "qlen": len(qt), "missing": missing,
            "overlap": max(overlaps) if overlaps else 0.0,
            "dlen": st.median(dlens) if dlens else 0,
        })
        if bucket == "zero":
            examples.append((qid, query, titles, max(overlaps) if overlaps else 0.0))

    def summarize(rows):
        if not rows:
            return "n=0"
        return (f"n={len(rows):5} n_rel mean={st.mean(r['n_rel'] for r in rows):.2f} "
                f"(=1: {sum(r['n_rel'] == 1 for r in rows) / len(rows):.0%}) "
                f"q_terms med={st.median(r['qlen'] for r in rows):.0f} "
                f"best-rel term overlap med={st.median(r['overlap'] for r in rows):.2f} "
                f"(<=0.25: {sum(r['overlap'] <= 0.25 for r in rows) / len(rows):.0%}) "
                f"rel doc chars med={st.median(r['dlen'] for r in rows):.0f} "
                f"missing_rel={sum(r['missing'] for r in rows)}")

    print(f"== {sid}")
    print(f"  R@30=0 : {summarize(stats['zero'])}")
    print(f"  R@30>0 : {summarize(stats['hit'])}")
    for qid, query, titles, ov in rng.sample(examples, min(args.show, len(examples))):
        t = " || ".join(x[:90] for x in titles[:2])
        print(f"    [{qid}] ov={ov:.2f} Q: {query[:110]}")
        print(f"           REL: {t}")
    print()
