#!/usr/bin/env bash
# Launch exactly one persistent Mut historical-50k successor.  This launcher
# never stops or signals the superseded 440-GiB waiter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${MUT_LEGACY_PROJECT_ROOT:?absolute immutable 7f7ed51 worktree required}"
: "${MUT_INSTRUMENTATION_PROJECT_ROOT:?absolute immutable 66487c0 worktree required}"
export MUT_SEMANTIC_FINALIZER_PROJECT_ROOT="${MUT_SEMANTIC_FINALIZER_PROJECT_ROOT:-/root/autodl-tmp/worktrees/final-five-closeout-582bc4b-20260902T040000Z}"
[[ "$MUT_LEGACY_PROJECT_ROOT" == /* && -d "$MUT_LEGACY_PROJECT_ROOT" ]] || exit 64
[[ "$MUT_INSTRUMENTATION_PROJECT_ROOT" == /* && -d "$MUT_INSTRUMENTATION_PROJECT_ROOT" ]] || exit 64
[[ "$MUT_SEMANTIC_FINALIZER_PROJECT_ROOT" == /* \
  && -d "$MUT_SEMANTIC_FINALIZER_PROJECT_ROOT" \
  && ! -L "$MUT_SEMANTIC_FINALIZER_PROJECT_ROOT" ]] || exit 64

TIMESTAMP="${MUT_FAST_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$TIMESTAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "unsafe MUT_FAST_TIMESTAMP=$TIMESTAMP" >&2
  exit 64
}
SPEC_DIR="$AUTODL_CONTROL_ROOT/mut_fast_accurate_v2/specs"
SPEC="$SPEC_DIR/mut_fast_accurate_v2_${TIMESTAMP}.json"
mkdir -p "$SPEC_DIR"
"$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_fast_accurate_v2.py" \
  --config configs/hpc.yaml materialize \
  --template "$PROJECT_ROOT/configs/autodl/mut_fast_accurate_v2.template.json" \
  --output "$SPEC" --project-root "$PROJECT_ROOT" \
  --legacy-project-root "$MUT_LEGACY_PROJECT_ROOT" \
  --instrumentation-project-root "$MUT_INSTRUMENTATION_PROJECT_ROOT" \
  --timestamp "$TIMESTAMP"
"$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_fast_accurate_v2.py" \
  --config configs/hpc.yaml validate --spec "$SPEC" >/dev/null

CONTROLLER_ID="mut_fast_accurate_v2_${TIMESTAMP}"
LOG_DIR="$AUTODL_RUNTIME_ROOT/logs/mut_fast_accurate_v2"
LOG="$LOG_DIR/${CONTROLLER_ID}.log"
mkdir -p "$LOG_DIR"
SESSION="cf-mut-fast-${TIMESTAMP}"
COMMAND=(
  "$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_fast_accurate_v2.py"
  --config configs/hpc.yaml run --spec "$SPEC"
)
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "successor tmux session already exists: $SESSION" >&2
    exit 3
  }
  printf -v TMUX_COMMAND '%q ' "${COMMAND[@]}"
  tmux new-session -d -s "$SESSION" "$TMUX_COMMAND >>$(printf '%q' "$LOG") 2>&1"
  echo "launcher=tmux session=$SESSION"
else
  nohup "${COMMAND[@]}" >>"$LOG" 2>&1 </dev/null &
  echo "launcher=nohup pid=$!"
fi

echo "controller_id=$CONTROLLER_ID"
echo "spec=$SPEC"
echo "log=$LOG"
echo "state=WAITING_FOR_INITIAL_HEARTBEAT"
echo "old_440_waiter_signaled_by_launcher=false"
echo "old_440_waiter_handover_owner=main_agent_exact_pid_start_ticks_command_verifier"
echo "status_command=$AUTODL_PYTHON $SCRIPT_DIR/status_mut_fast_accurate_v2.py --spec $SPEC"
