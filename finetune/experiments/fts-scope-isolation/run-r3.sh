#!/bin/zsh
# R3: hybrid + rerank bench runs, interleaved S/I per dataset so drift over the
# night hits both arms alike. Resumable: bench keeps a .partial per run and a
# finished run's JSON is skipped.
cd "$(dirname "$0")/../../.."
X=$PWD/artifacts/fts-scope-isolation
log() { echo "[$(date '+%F %T')] $*"; }
bench() {  # $1 index variant, $2 bench group (S|I), $3 dataset dir, rest: bench args
  INDEX_PATH=$X/index/$1/index.sqlite QMD_CONFIG_DIR=$X/config/$1 \
    node_modules/.bin/tsx src/cli/qmd.ts bench $X/bench/$2/$3 "${@:4}"
}

for pair in qmd-expansion-scifact-v1:scifact qmd-eval-v1-cqadup-android:android qmd-eval-v1-fiqa-test:fiqa; do
  d=${pair%%:*}; t=${pair##*:}
  for arm in raw q8; do
    for g in S I; do
      v=$([[ $g == S ]] && echo S || echo I-$t)
      if [[ $arm == raw ]]; then out=$X/bench/$g/$d/runs/raw.json; args=(--run raw)
      else out=$X/bench/$g/$d/runs/g0d-$t-local-d-q8-nosys-greedy.json; args=(--run g0d-$t --model local-d-q8-nosys-greedy); fi
      if [[ -f $out ]]; then log "SKIP $g/$arm/$t (done)"; continue; fi
      log "START $g/$arm/$t ($v)"
      bench $v $g $d $args > $X/logs/r3-$g-$arm-$t.log 2>&1
      log "END $g/$arm/$t exit=$?"
    done
  done
done
log R3-DONE
