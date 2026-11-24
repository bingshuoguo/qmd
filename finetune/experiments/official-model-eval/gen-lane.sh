#!/usr/bin/env bash
# Generate all expansion files into bench-rr (sequential, GPU).
set -u; source "$(dirname "$0")/common.sh"
Q8=$E/models/qmd-query-expansion-1.7B-q8_0.gguf
Q4=hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf
for t in $ALL; do B=$E/bench-rr/$(dir_of $t)
  [ -f $B/expansions/offq8-$t.jsonl ] || { log START offq8/$t; $TSX finetune/experiments/official-model-eval/expand-official.ts --benchmark $B --run offq8-$t --model $Q8; log END offq8/$t exit=$?; }
  [ -f $B/expansions/offvh-$t.jsonl ] || python3 finetune/experiments/official-model-eval/derive_vh.py $B $t
done
for t in webmasters freshstack; do B=$E/bench-rr/$(dir_of $t)
  [ -f $B/expansions/sftq8-$t.jsonl ] || { log START sftq8/$t; $TSX finetune/benchmarks/expand.ts --benchmark $B --run sftq8-$t --profile $PWD/artifacts/sft-serving-baseline/g0-20261026/profiles/D-q8-nosys-greedy.json --seed 42; log END sftq8/$t exit=$?; }
done
for t in $ALL; do B=$E/bench-rr/$(dir_of $t)
  [ -f $B/expansions/offprod-$t.jsonl ] || { log START offprod/$t; $TSX finetune/benchmarks/expand.ts --benchmark $B --run offprod-$t --model $Q4; log END offprod/$t exit=$?; }
done
log GEN-DONE
