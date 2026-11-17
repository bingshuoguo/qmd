#!/usr/bin/env bash
# Start N reward-server processes on consecutive ports, for PooledRewardClient.
#
#   finetune/experiments/grpo-retrieval/start-servers.sh [N]   # default N=4, BASE_PORT=8787
#   finetune/experiments/grpo-retrieval/start-servers.sh stop
#
# The first server verifies every index against its frozen manifest; the others reuse the same
# files and start with --skip-index-verify. All servers embed on the same device (never set
# QMD_FORCE_CPU for only some of them): mixed devices would make scores depend on the server.
# PIDs go to $LOG/reward-servers.pid so stopping never needs `pkill -f`.
set -euo pipefail
cd "$(dirname "$0")/../../.."

N=${1:-4}
BASE=${BASE_PORT:-8787}
LOG=artifacts/grpo-retrieval/logs
PIDS=$LOG/reward-servers.pid
mkdir -p "$LOG"

if [ "$N" = stop ]; then
  [ -f "$PIDS" ] || { echo "no $PIDS"; exit 0; }
  while read -r pid; do kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true; done < "$PIDS"
  rm -f "$PIDS"
  echo stopped
  exit 0
fi
[ -f "$PIDS" ] && { echo "servers already running (see $PIDS); run '$0 stop' first" >&2; exit 1; }

I=artifacts/grpo-retrieval/index
B=artifacts/sft-serving-baseline/g0-20261026/abc/bench
T=finetune/data/public-distill-v0/prepared
ARGS=(
  --db qmd-eval-v1-fiqa-test=$I/fiqa/index.sqlite
  --db qmd-distill-public-v0-scifact-train=$I/scifact/index.sqlite
  --db qmd-distill-public-v0-cqadup-programmers=$I/programmers/index.sqlite
  --db qmd-distill-public-v0-cqadup-unix=$I/unix/index.sqlite
  --db qmd-eval-v1-cqadup-android=$I/android/index.sqlite
  --source fiqa-train=$T/fiqa-train --source fiqa-test=$B/qmd-eval-v1-fiqa-test
  --source scifact-train=$T/qmd-distill-public-v0-scifact-train --source scifact-test=$B/qmd-expansion-scifact-v1
  --source programmers=$T/cqadup-programmers --source unix=$T/cqadup-unix
  --source android=$B/qmd-eval-v1-cqadup-android
)
# setsid (Linux) gives each server its own process group, so stop can kill tsx and its node child together.
SETSID=$(command -v setsid || true)

launch() {
  local port=$1; shift
  $SETSID nohup node_modules/.bin/tsx finetune/experiments/grpo-retrieval/reward-server.ts \
    "${ARGS[@]}" --port "$port" "$@" > "$LOG/server-$port.log" 2>&1 < /dev/null &
  echo $! >> "$PIDS"
}

wait_ready() {
  for _ in $(seq 1 120); do
    curl -sf -m 2 "http://127.0.0.1:$1/health" > /dev/null && return 0
    sleep 5
  done
  echo "server on port $1 not ready; see $LOG/server-$1.log" >&2
  return 1
}

for k in $(seq 0 $((N - 1))); do
  if (echo > "/dev/tcp/127.0.0.1/$((BASE + k))") 2> /dev/null; then
    echo "port $((BASE + k)) is already in use; pick another BASE_PORT" >&2
    exit 1
  fi
done

launch "$BASE"
wait_ready "$BASE"
for k in $(seq 1 $((N - 1))); do launch $((BASE + k)) --skip-index-verify; done
for k in $(seq 1 $((N - 1))); do wait_ready $((BASE + k)); done

URLS=$(for k in $(seq 0 $((N - 1))); do printf "http://127.0.0.1:%d," $((BASE + k)); done)
echo "ready: $N servers"
echo "URLS=${URLS%,}"
