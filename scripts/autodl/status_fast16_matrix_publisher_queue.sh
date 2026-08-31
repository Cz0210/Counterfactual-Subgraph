#!/usr/bin/env bash
set -euo pipefail

PID_PATH=${PID_PATH:?Set PID_PATH}
HEARTBEAT_PATH=${HEARTBEAT_PATH:?Set HEARTBEAT_PATH}
STATE_PATH=${STATE_PATH:-/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority/state.json}
AUTODL_PYTHON=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}

if [[ -f "$PID_PATH" ]]; then
  queue_pid=$(<"$PID_PATH")
  if [[ "$queue_pid" =~ ^[0-9]+$ ]] && kill -0 "$queue_pid" 2>/dev/null; then
    echo "publisher_queue_process=RUNNING pid=$queue_pid"
  else
    echo "publisher_queue_process=NOT_RUNNING recorded_pid=$queue_pid"
  fi
else
  echo "publisher_queue_process=NO_PID_FILE"
fi
for artifact in "$HEARTBEAT_PATH" "$STATE_PATH"; do
  echo "artifact=$artifact"
  if [[ -f "$artifact" ]]; then
    "$AUTODL_PYTHON" -m json.tool "$artifact"
  else
    echo "ABSENT"
  fi
done
