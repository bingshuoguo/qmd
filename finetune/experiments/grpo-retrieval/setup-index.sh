#!/bin/zsh
# Build the GRPO reward-environment index files from the G0 sealed index,
# following spec §3: one index FILE per corpus, one collection per file.
#
#   fiqa/index.sqlite        -> qmd-eval-v1-fiqa-test
#   scifact/index.sqlite     -> qmd-distill-public-v0-scifact-train
#   programmers/index.sqlite -> qmd-distill-public-v0-cqadup-programmers
#   unix/index.sqlite        -> qmd-distill-public-v0-cqadup-unix
#   android/index.sqlite     -> qmd-eval-v1-cqadup-android
#
# No re-embedding: every collection is a byte-identical copy of the G0
# collection (same documents_sha256 / index_fingerprint). Stage 1 trims G0
# (11 collections) to a 5-collection source; stage 2 splits that into 5
# single-collection files.
#
# Each index gets its OWN config dir (QMD_CONFIG_DIR=$X/config/<name>), seeded
# from the full collection list before its removes. This is required: `qmd
# collection remove` deletes the collection from the config it reads, and
# `syncConfigToDb` deletes any collection missing from that config. A single
# shared config loses the later targets as earlier removes trim it (the bug
# this script previously hit).
set -e
cd "$(dirname "$0")/../../.."

X=$PWD/artifacts/grpo-retrieval
SRC=$PWD/artifacts/sft-serving-baseline/g0-20261026/abc/index/index.sqlite
# Full 11-collection config with correct paths (verified in fts-scope-isolation).
CFG=$PWD/artifacts/fts-scope-isolation/config/S/index.yml
QMD=(node_modules/.bin/tsx src/cli/qmd.ts)

[ -f "$SRC" ] || { echo "missing G0 index: $SRC"; exit 1; }
[ -f "$CFG" ] || { echo "missing reference config: $CFG"; exit 1; }

UNUSED=(
  qmd-distill-public-v0-nfcorpus-train-dev
  qmd-eval-v1-cqadup-webmasters
  qmd-eval-v1-fiqa-dev
  qmd-nf-freshstack-v1-freshstack-test
  qmd-nf-freshstack-v1-nfcorpus-test
  qmd-expansion-scifact-v1
)

typeset -A KEEP
KEEP=(
  fiqa        qmd-eval-v1-fiqa-test
  scifact     qmd-distill-public-v0-scifact-train
  programmers qmd-distill-public-v0-cqadup-programmers
  unix        qmd-distill-public-v0-cqadup-unix
  android     qmd-eval-v1-cqadup-android
)
ALL_TARGETS=(${(v)KEEP})

mkdir -p "$X/index" "$X/config"

# Stage 1: combined 5-collection source index.
COMBINED="$X/index/index.sqlite"
mkdir -p "$X/config/combined"
cp "$CFG" "$X/config/combined/index.yml"
if [ ! -f "$COMBINED" ]; then
  echo "[setup] stage 1: copy G0 -> $COMBINED"
  cp "$SRC" "$COMBINED"
fi
for c in $UNUSED; do
  echo "[setup] stage 1: remove $c"
  INDEX_PATH="$COMBINED" QMD_CONFIG_DIR="$X/config/combined" $QMD collection remove "$c"
done
echo "[setup] stage 1: cleanup combined"
INDEX_PATH="$COMBINED" QMD_CONFIG_DIR="$X/config/combined" $QMD cleanup

# Stage 2: 5 single-collection indexes, one config dir each. The config is
# seeded with the 5-target combined config so `collection remove` can find each
# of the 4 other targets, and so syncConfigToDb keeps the one survivor.
for name in ${(k)KEEP}; do
  keep=$KEEP[$name]
  out="$X/index/$name/index.sqlite"
  cfgdir="$X/config/$name"
  mkdir -p "$X/index/$name" "$cfgdir"
  cp "$X/config/combined/index.yml" "$cfgdir/index.yml"
  echo "[setup] stage 2: $name <- copy combined"
  cp "$COMBINED" "$out"
  for c in $ALL_TARGETS; do
    [[ "$c" == "$keep" ]] && continue
    echo "[setup] stage 2: $name remove $c"
    INDEX_PATH="$out" QMD_CONFIG_DIR="$cfgdir" $QMD collection remove "$c"
  done
  echo "[setup] stage 2: $name cleanup"
  INDEX_PATH="$out" QMD_CONFIG_DIR="$cfgdir" $QMD cleanup
  echo "[setup] stage 2: $name done"
  INDEX_PATH="$out" QMD_CONFIG_DIR="$cfgdir" $QMD collection list
done

echo "[setup] done — single-collection indexes under $X/index/"

# Combined is only an intermediate source for stage 2; drop it so the output is
# exactly the 5 single-collection files (spec §3: one collection per file).
rm -f "$COMBINED"
rm -rf "$X/config/combined"
