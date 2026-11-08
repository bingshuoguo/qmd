#!/bin/zsh
# R1/R2/R4 probes and G3/G4 gates, then R3. Each probe output lands in
# artifacts/fts-scope-isolation/probe/<variant>/<dataset>/<mode>.jsonl.
cd "$(dirname "$0")/../../.."
X=$PWD/artifacts/fts-scope-isolation
E=finetune/experiments/fts-scope-isolation
log() { echo "[$(date '+%F %T')] $*"; }
probe() {  # $1 mode, $2 variant, $3 dataset dir
  local out=$X/probe/$2/$3/$1.jsonl
  if [[ -s $out ]]; then log "SKIP $1 $2 $3"; return; fi
  log "START $1 $2 $3"
  node_modules/.bin/tsx $E/probe.ts $1 $2 $X/bench/S/$3 $out.tmp >> $X/logs/probe-$1.log 2>&1 && mv $out.tmp $out
  log "END $1 $2 $3 exit=$?"
}

SETS=(qmd-eval-v1-fiqa-test:fiqa qmd-expansion-scifact-v1:scifact qmd-eval-v1-cqadup-android:android)

for pair in $SETS; do d=${pair%%:*}; t=${pair##*:}
  probe gates I-$t $d
  probe gates S $d
  probe bm25 S $d
  probe bm25 I-$t $d
done
probe gates D-fiqa qmd-eval-v1-fiqa-test
probe gates D-scifact qmd-expansion-scifact-v1
probe bm25 D-fiqa qmd-eval-v1-fiqa-test
probe bm25 D-scifact qmd-expansion-scifact-v1

for pair in $SETS; do d=${pair%%:*}; t=${pair##*:}
  probe vec S $d
  probe vec I-$t $d
done
for pair in $SETS; do d=${pair%%:*}; t=${pair##*:}
  probe strong S $d
  probe strong I-$t $d
done
log PROBES-DONE

$E/run-r3.sh
