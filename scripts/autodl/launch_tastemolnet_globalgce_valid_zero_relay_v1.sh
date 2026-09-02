#!/usr/bin/env bash
set -euo pipefail

: "${AUTODL_RUNTIME_ROOT:=/autodl-fs/data/counterfactual-subgraph-runtime}"
: "${AUTODL_CONTROL_ROOT:=$AUTODL_RUNTIME_ROOT/control}"
: "${AUTODL_PYTHON:=/root/miniconda3/envs/smiles_pip118/bin/python}"
: "${TASTE_GLOBALGCE_SOURCE_ROOT:?set TASTE_GLOBALGCE_SOURCE_ROOT}"
: "${TASTE_GLOBALGCE_ATTEMPT_RECEIPT:?set TASTE_GLOBALGCE_ATTEMPT_RECEIPT}"
: "${TASTE_GLOBALGCE_TEST_CSV:?set TASTE_GLOBALGCE_TEST_CSV}"
: "${TASTE_GLOBALGCE_THRESHOLD_CONTRACT:?set TASTE_GLOBALGCE_THRESHOLD_CONTRACT}"
: "${TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT:?set TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT}"
: "${TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT:?set TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT}"
: "${TASTE_GLOBALGCE_SCIENCE_PID:?set TASTE_GLOBALGCE_SCIENCE_PID}"
: "${TASTE_GLOBALGCE_SCIENCE_START_TICKS:?set TASTE_GLOBALGCE_SCIENCE_START_TICKS}"

CONTROL_ROOT="${TASTE_GLOBALGCE_ZERO_RELAY_CONTROL_ROOT:-$AUTODL_CONTROL_ROOT/tastemolnet-globalgce-valid-zero-relay-v1}"
AUTHORIZATION="${TASTE_GLOBALGCE_ZERO_AUTHORIZATION:-$CONTROL_ROOT/authorization.json}"
LEASE="${TASTE_GLOBALGCE_ZERO_RELAY_LEASE:-$AUTODL_CONTROL_ROOT/tastemolnet-globalgce-valid-zero-relay-v1.lock}"
POLL_SECONDS="${TASTE_GLOBALGCE_ZERO_RELAY_POLL_SECONDS:-30}"
LOG="$AUTODL_RUNTIME_ROOT/logs/tastemolnet-globalgce-valid-zero-relay-v1.log"
SESSION="cf-taste-globalgce-valid-zero-relay-v1"
mkdir -p "$(dirname "$LOG")"

COMMAND=(
  nice -n 10
  "$AUTODL_PYTHON" -I -B
  scripts/autodl/run_tastemolnet_globalgce_valid_zero_relay_v1.py
  --source-root "$TASTE_GLOBALGCE_SOURCE_ROOT"
  --attempt-receipt "$TASTE_GLOBALGCE_ATTEMPT_RECEIPT"
  --authorization-receipt "$AUTHORIZATION"
  --test-csv "$TASTE_GLOBALGCE_TEST_CSV"
  --threshold-contract "$TASTE_GLOBALGCE_THRESHOLD_CONTRACT"
  --valid-zero-output-root "$TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT"
  --control-root "$CONTROL_ROOT"
  --lease-path "$LEASE"
  --science-pid "$TASTE_GLOBALGCE_SCIENCE_PID"
  --science-start-ticks "$TASTE_GLOBALGCE_SCIENCE_START_TICKS"
  --poll-seconds "$POLL_SECONDS"
  --execution-commit "$TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT"
)

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "existing_session=$SESSION"
    exit 0
  fi
  printf -v TMUX_COMMAND '%q ' "${COMMAND[@]}"
  tmux new-session -d -s "$SESSION" "$TMUX_COMMAND >>$(printf '%q' "$LOG") 2>&1"
  echo "launcher=tmux session=$SESSION"
else
  nohup "${COMMAND[@]}" >>"$LOG" 2>&1 </dev/null &
  echo "launcher=nohup pid=$!"
fi
echo "control_root=$CONTROL_ROOT"
echo "heartbeat=$CONTROL_ROOT/heartbeat.json"
echo "terminal=$CONTROL_ROOT/terminal.json"
echo "log=$LOG"
echo "sqlite_opened=false"
echo "signal_sent=false"
echo "training_started=false"
