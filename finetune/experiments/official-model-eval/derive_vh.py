#!/usr/bin/env python3
"""Derive the official-model VH-only arm: drop lex lines from an offq8 expansion file.

  python3 finetune/experiments/official-model-eval/derive_vh.py <bench-dir> <tag>

Reads expansions/offq8-<tag>.jsonl, writes expansions/offvh-<tag>.jsonl with the same
records minus lex entries (raw_output rebuilt from the kept lines, order preserved).
"""
import json, sys

bench, tag = sys.argv[1], sys.argv[2]
src = f"{bench}/expansions/offq8-{tag}.jsonl"
dst = f"{bench}/expansions/offvh-{tag}.jsonl"
n = dropped = emptied = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        r = json.loads(line)
        if r["status"] == "ok":
            kept = [p for p in r["output"] if p[0] != "lex"]
            dropped += len(r["output"]) - len(kept)
            if not kept:
                emptied += 1
                r.update(status="format_error", output=[], raw_output="", error="no vec/hyde lines after lex removal")
            else:
                r.update(output=kept, raw_output="\n".join(f"{t}: {x}" for t, x in kept))
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        n += 1
print(json.dumps({"records": n, "lex_lines_dropped": dropped, "emptied": emptied, "out": dst}))
