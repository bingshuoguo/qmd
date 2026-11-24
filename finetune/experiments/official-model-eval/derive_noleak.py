#!/usr/bin/env python3
"""Derive the leak-free production arm: strip the "</think>" that qmd's default grammar
makes the official model append to its hyde line. Everything else is left unchanged.

  python3 finetune/experiments/official-model-eval/derive_noleak.py <bench-dir> <tag>

Reads expansions/offprod-<tag>.jsonl, writes expansions/offfix-<tag>.jsonl.
"""
import json, sys

bench, tag = sys.argv[1], sys.argv[2]
src = f"{bench}/expansions/offprod-{tag}.jsonl"
dst = f"{bench}/expansions/offfix-{tag}.jsonl"
n = fixed = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        r = json.loads(line)
        if r["status"] == "ok" and "</think>" in r["raw_output"]:
            r["output"] = [[t, x.replace("</think>", "").rstrip()] for t, x in r["output"]]
            r["raw_output"] = "".join(f"{t}: {x}\n" for t, x in r["output"])
            fixed += 1
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print(json.dumps({"records": n, "leaks_removed": fixed, "out": dst}))
