#!/usr/bin/env bash
# Launch the predeployed one-shot Mut post-A/B successor owner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${MUT_NEXT_STAGE_TASK_SPEC:?absolute immutable task spec required}"
[[ "$MUT_NEXT_STAGE_TASK_SPEC" == /* && -f "$MUT_NEXT_STAGE_TASK_SPEC" && ! -L "$MUT_NEXT_STAGE_TASK_SPEC" ]] || {
  echo "physical Mut successor task spec required" >&2
  exit 64
}

TIMESTAMP="${MUT_NEXT_STAGE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="$AUTODL_RUNTIME_ROOT/logs/mut_next_stage_executor"
LOG="$LOG_DIR/$TIMESTAMP.log"
SESSION="cf-mut-next-$TIMESTAMP"
mkdir -p "$LOG_DIR"
COMMAND=(
  "$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_next_stage_executor_v1.py"
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --task-spec "$MUT_NEXT_STAGE_TASK_SPEC"
  --poll-seconds "${SCHEDULER_POLL_SECONDS:-60}"
)
export RUN_LLM_ABLATION=0
export RUN_GNN_ABLATION=0
if command -v tmux >/dev/null 2>&1; then
  printf -v TMUX_COMMAND '%q ' "${COMMAND[@]}"
  tmux new-session -d -s "$SESSION" "$TMUX_COMMAND >>$(printf '%q' "$LOG") 2>&1"
  echo "launcher=tmux session=$SESSION"
else
  nohup "${COMMAND[@]}" >>"$LOG" 2>&1 </dev/null &
  echo "launcher=nohup pid=$!"
fi
echo "task_spec=$MUT_NEXT_STAGE_TASK_SPEC"
echo "log=$LOG"
