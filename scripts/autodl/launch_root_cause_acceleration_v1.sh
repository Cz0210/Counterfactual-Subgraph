#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_CONTROL_ROOT:?AUTODL_CONTROL_ROOT is required}"
: "${AUTODL_ROOT_CAUSE_SPEC:?AUTODL_ROOT_CAUSE_SPEC is required}"
PY="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
CONTROLLER_ID="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["controller_id"])' "$AUTODL_ROOT_CAUSE_SPEC")"
ROOT="$AUTODL_CONTROL_ROOT/root_cause_acceleration/$CONTROLLER_ID"
mkdir -p "$ROOT"
if [[ -s "$ROOT/controller.pid" ]]; then
  old_pid="$(cat "$ROOT/controller.pid")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[ROOT_CAUSE_CONTROLLER_ALREADY_RUNNING] pid=$old_pid root=$ROOT"
    exit 0
  fi
fi
cd "$PROJECT_ROOT"
nohup env PYTHONPATH="$PROJECT_ROOT" "$PY" \
  scripts/autodl/run_root_cause_acceleration_controller.py \
  --config configs/hpc.yaml \
  --spec "$AUTODL_ROOT_CAUSE_SPEC" \
  --control-root "$AUTODL_CONTROL_ROOT/root_cause_acceleration" \
  --poll-seconds "${SCHEDULER_POLL_SECONDS:-60}" \
  >>"$ROOT/controller.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$ROOT/controller.pid"
echo "[ROOT_CAUSE_ACCELERATION_CONTROLLER_RUNNING] pid=$pid root=$ROOT"
