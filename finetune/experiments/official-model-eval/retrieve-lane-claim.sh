#!/usr/bin/env bash
# retrieve-lane-claim.sh <bench-rr|bench-norr> <tag:arm> ...
# Like retrieve-lane.sh, but several lanes can share one list: each (dataset, arm) is
# claimed with an atomic mkdir under runs/.claims/ and skipped if another lane holds it.
set -u; source "$(dirname "$0")/common.sh"
ROOT=$1; shift
for item in "$@"; do t=${item%%:*}; arm=${item##*:}; d=$(dir_of $t); B=$E/$ROOT/$d; run=$(run_of $t $arm)
  mkdir -p $B/runs/.claims
  mkdir $B/runs/.claims/$run 2>/dev/null || continue
  if [ $arm = raw ]; then
    [ -f $B/runs/results/raw.jsonl ] && continue
    log START $ROOT/$t/raw; $TSX src/cli/qmd.ts bench $B --run raw; log END $ROOT/$t/raw exit=$?
    continue
  fi
  ls $B/runs/results/$run-*.jsonl >/dev/null 2>&1 && continue
  until [ -f $E/bench-rr/$d/expansions/$run.jsonl ]; do sleep 60; done
  mkdir -p $B/expansions; [ $ROOT = bench-rr ] || cp -n $E/bench-rr/$d/expansions/$run.jsonl $B/expansions/ 2>/dev/null || true
  log START $ROOT/$t/$arm; $TSX src/cli/qmd.ts bench $B --run $run --model $(label_of $arm); log END $ROOT/$t/$arm exit=$?
done
log LANE-DONE $ROOT $$
