#!/usr/bin/env bash
# Launch the persistent four-GPU control plane. Scientific commands remain in
# the frozen manifest and are delegated to exp_run.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
export PYTHONDONTWRITEBYTECODE=1

export RUN_TASTEMOLNET=0
MANIFEST="${1:-${FOUR_GPU_RECOVERY_MANIFEST:-$AUTODL_CONTROL_ROOT/four_gpu_recovery_manifest.json}}"
if [[ "$MANIFEST" != /* || ! -s "$MANIFEST" ]]; then
  echo "FOUR_GPU_RECOVERY_MANIFEST must be an absolute nonempty file: $MANIFEST" >&2
  exit 64
fi

CONTROLLER_ID="$("$AUTODL_PYTHON" - "$MANIFEST" <<'PY'
import pathlib
import sys

from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest

print(load_controller_manifest(pathlib.Path(sys.argv[1])).controller_id)
PY
)"
if ! [[ "$CONTROLLER_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Unsafe controller_id in manifest: $CONTROLLER_ID" >&2
  exit 64
fi

CONTROL_DIR="$AUTODL_CONTROL_ROOT/four_gpu_recovery/$CONTROLLER_ID"
mkdir -p "$CONTROL_DIR"
LAUNCH_LOG="$CONTROL_DIR/controller.log"
SESSION="cf-four-gpu-${CONTROLLER_ID:0:48}"
COMMAND=(
  "$AUTODL_PYTHON"
  "$SCRIPT_DIR/run_four_gpu_recovery_controller.py"
  --project-root "$PROJECT_ROOT"
  --data-root "$AUTODL_DATA_ROOT"
  --control-root "$AUTODL_CONTROL_ROOT"
  --python "$AUTODL_PYTHON"
  run
  --manifest "$MANIFEST"
)

"$AUTODL_PYTHON" "$SCRIPT_DIR/run_four_gpu_recovery_controller.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  --control-root "$AUTODL_CONTROL_ROOT" \
  --python "$AUTODL_PYTHON" \
  validate --manifest "$MANIFEST" >/dev/null

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Four-GPU controller tmux session already exists: $SESSION" >&2
    exit 3
  fi
  printf -v TMUX_COMMAND '%q ' "${COMMAND[@]}"
  tmux new-session -d -s "$SESSION" "$TMUX_COMMAND"
  echo "launcher=tmux session=$SESSION"
else
  nohup "${COMMAND[@]}" >>"$LAUNCH_LOG" 2>&1 </dev/null &
  LAUNCHER_PID=$!
  echo "launcher=nohup pid=$LAUNCHER_PID log=$LAUNCH_LOG"
fi

echo "controller_id=$CONTROLLER_ID"
echo "manifest=$MANIFEST"
echo "status: $AUTODL_PYTHON $SCRIPT_DIR/status_four_gpu_recovery.py --project-root $PROJECT_ROOT --data-root $AUTODL_DATA_ROOT --control-root $AUTODL_CONTROL_ROOT --controller-id $CONTROLLER_ID --watch 60"
