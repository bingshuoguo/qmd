# Shared settings for the official-model eval drivers (sourced).
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
E=$PWD/artifacts/official-model-eval
export INDEX_PATH=$E/index/index.sqlite QMD_CONFIG_DIR=$E/config
TSX=node_modules/.bin/tsx
log(){ echo "[$(date '+%F %T')] $*"; }
dir_of(){ case $1 in
  fiqa) echo qmd-eval-v1-fiqa-test;; scifact) echo qmd-expansion-scifact-v1;;
  android) echo qmd-eval-v1-cqadup-android;; webmasters) echo qmd-eval-v1-cqadup-webmasters;;
  freshstack) echo qmd-nf-freshstack-v1-freshstack-test;; esac; }
# arm -> expansion run name and bench --model label
run_of(){ case $2 in
  raw) echo raw;; sft) case $1 in fiqa|scifact|android) echo g0d-$1;; *) echo sftq8-$1;; esac;;
  *) echo $2-$1;; esac; }
label_of(){ case $1 in
  sft) echo local:D-q8-nosys-greedy;; offq8) echo official-q8-nosys-greedy;;
  offvh) echo official-q8-nosys-greedy-nolex;; offprod) echo official-q4-prod-default;; esac; }
ALL="fiqa scifact android webmasters freshstack"
