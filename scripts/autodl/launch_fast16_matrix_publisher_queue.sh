#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime/worktrees/fast16_matrix_publisher}
PYTHON_BIN=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
QUEUE_MANIFEST=${QUEUE_MANIFEST:?Set QUEUE_MANIFEST}
HEARTBEAT_PATH=${HEARTBEAT_PATH:?Set HEARTBEAT_PATH}
LOG_PATH=${LOG_PATH:?Set LOG_PATH}
PID_PATH=${PID_PATH:?Set PID_PATH}

mkdir -p "$(dirname "$HEARTBEAT_PATH")" "$(dirname "$LOG_PATH")" "$(dirname "$PID_PATH")"
if [[ -f "$PID_PATH" ]]; then
  prior_pid=$(<"$PID_PATH")
  if [[ "$prior_pid" =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "publisher queue already running pid=$prior_pid" >&2
    exit 1
  fi
fi
cd "$PROJECT_ROOT"
export PYTHONPATH=$PWD
nohup setsid "$PYTHON_BIN" scripts/autodl/run_fast16_matrix_publisher_queue.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --queue-manifest "$QUEUE_MANIFEST" \
  --heartbeat-path "$HEARTBEAT_PATH" \
  >>"$LOG_PATH" 2>&1 </dev/null &
queue_pid=$!
printf '%s\n' "$queue_pid" >"$PID_PATH"
echo "publisher_queue_pid=$queue_pid"
echo "heartbeat=$HEARTBEAT_PATH"
echo "log=$LOG_PATH"
