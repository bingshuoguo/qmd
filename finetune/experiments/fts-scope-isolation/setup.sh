#!/bin/zsh
# Build the isolated / dedup index copies from the S copies already in place:
# remove collections with `qmd collection remove` (never via config sync),
# then run the same `qmd cleanup` on every copy (gate G5).
set -e
cd "$(dirname "$0")/../../.."
X=$PWD/artifacts/fts-scope-isolation
qmd() { INDEX_PATH=$X/index/$1/index.sqlite QMD_CONFIG_DIR=$X/config/$1 node_modules/.bin/tsx src/cli/qmd.ts "${@:2}"; }
log() { echo "[$(date '+%F %T')] $*"; }

ALL=(
  qmd-distill-public-v0-cqadup-programmers
  qmd-distill-public-v0-cqadup-unix
  qmd-distill-public-v0-nfcorpus-train-dev
  qmd-distill-public-v0-scifact-train
  qmd-eval-v1-cqadup-android
  qmd-eval-v1-cqadup-webmasters
  qmd-eval-v1-fiqa-dev
  qmd-eval-v1-fiqa-test
  qmd-expansion-scifact-v1
  qmd-nf-freshstack-v1-freshstack-test
  qmd-nf-freshstack-v1-nfcorpus-test
)

keep_only() {  # $1 = variant, $2 = collection to keep
  for c in $ALL; do
    [[ $c == $2 ]] && continue
    log "$1: remove $c"; qmd $1 collection remove $c
  done
}

# D-fiqa: removed manually first (qmd-eval-v1-fiqa-dev)
log "D-scifact"; qmd D-scifact collection remove qmd-distill-public-v0-scifact-train
keep_only I-fiqa qmd-eval-v1-fiqa-test
keep_only I-scifact qmd-expansion-scifact-v1
keep_only I-android qmd-eval-v1-cqadup-android

for v in S D-fiqa D-scifact I-fiqa I-scifact I-android; do
  log "cleanup $v"; qmd $v cleanup
done
log SETUP-DONE
