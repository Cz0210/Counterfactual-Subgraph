#!/usr/bin/env bash
# Launch the CPU-only, typed AIDS disconnected-exact recovery controller.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/controller_manifest.json [fresh|resume]" >&2
  exit 64
fi

MANIFEST=$1
MODE=${2:-fresh}
if [[ "$MANIFEST" != /* ]]; then
  echo "manifest must be absolute" >&2
  exit 64
fi
if [[ "$MODE" != "fresh" && "$MODE" != "resume" ]]; then
  echo "mode must be fresh or resume" >&2
  exit 64
fi

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
PROJECT_ROOT=${AUTODL_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

export PYTHONPATH=$PROJECT_ROOT
export CUDA_VISIBLE_DEVICES=""
export DEVICE=cpu
export GPU_REQUIRED=0

LAUNCH_TMP=$(mktemp -d)
trap 'rm -f "$LAUNCH_TMP/context"; rmdir "$LAUNCH_TMP"' EXIT
"$PY" "$PROJECT_ROOT/scripts/autodl/run_aids_comrecgc_exact_recovery_controller.py" \
  --manifest "$MANIFEST" \
  --prepare-only \
  --launch-mode "$MODE" \
  --context-lines > "$LAUNCH_TMP/context"
readarray -t CONTEXT < "$LAUNCH_TMP/context"
if [[ ${#CONTEXT[@]} -ne 8 ]]; then
  echo "invalid prelaunch context" >&2
  exit 70
fi
CONTROLLER_ROOT=${CONTEXT[0]}
CID=${CONTEXT[1]}
LAUNCH_ID=${CONTEXT[2]}
LOG=${CONTEXT[3]}
PID_PATH=${CONTEXT[4]}
SESSION=${CONTEXT[5]}
PRELAUNCH_RECEIPT=${CONTEXT[6]}
THREAD_COUNT=${CONTEXT[7]}
[[ "$CONTROLLER_ROOT" == /* && "$LOG" == "$CONTROLLER_ROOT"/* ]] || exit 70
[[ "$PID_PATH" == "$CONTROLLER_ROOT"/* && "$PRELAUNCH_RECEIPT" == "$CONTROLLER_ROOT"/* ]] || exit 70
[[ "$THREAD_COUNT" == "12" ]] || exit 70

export OMP_NUM_THREADS=$THREAD_COUNT
export MKL_NUM_THREADS=$THREAD_COUNT
export OPENBLAS_NUM_THREADS=$THREAD_COUNT
export NUMEXPR_NUM_THREADS=$THREAD_COUNT

( set -o noclobber; : > "$LOG" )
CMD=(
  "$PY"
  "$PROJECT_ROOT/scripts/autodl/run_aids_comrecgc_exact_recovery_controller.py"
  --manifest "$MANIFEST"
  --resume
)

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 73
  fi
  printf -v QUOTED '%q ' "${CMD[@]}"
  tmux new-session -d -s "$SESSION" "cd $(printf '%q' "$PROJECT_ROOT") && exec $QUOTED >>$(printf '%q' "$LOG") 2>&1"
  PANE_PID=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
  ( set -o noclobber; printf '%s\n' "$PANE_PID" > "$PID_PATH" )
  echo "launched cid=$CID launch_id=$LAUNCH_ID tmux_session=$SESSION pid=$PANE_PID log=$LOG receipt=$PRELAUNCH_RECEIPT"
else
  cd "$PROJECT_ROOT"
  nohup "${CMD[@]}" >>"$LOG" 2>&1 </dev/null &
  CONTROLLER_PID=$!
  ( set -o noclobber; printf '%s\n' "$CONTROLLER_PID" > "$PID_PATH" )
  echo "launched cid=$CID launch_id=$LAUNCH_ID pid=$CONTROLLER_PID log=$LOG receipt=$PRELAUNCH_RECEIPT"
fi
