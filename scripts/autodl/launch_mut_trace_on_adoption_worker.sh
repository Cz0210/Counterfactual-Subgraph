#!/usr/bin/env bash
# Attach one fail-closed Mut trace-on adoption worker to the live successor.
# The worker, not this launcher, owns the persistent lease and all GPU locks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${MUT_FAST_SPEC:?absolute live mut_fast_accurate_v2 spec required}"
: "${MUT_TRACE_PROTECTED_MANIFEST:?absolute protected-task manifest required}"
: "${MUT_TRACE_HISTORICAL_PROJECT_ROOT:?absolute immutable 7f7ed51 worktree required}"
: "${MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT:?absolute immutable 66487c0 worktree required}"
: "${MUT_TRACE_CONTROLLER_PID:?exact live successor PID required}"
: "${MUT_TRACE_CONTROLLER_START_TICKS:?exact live successor start ticks required}"

for mut_trace_path in \
  "$MUT_FAST_SPEC" \
  "$MUT_TRACE_PROTECTED_MANIFEST" \
  "$MUT_TRACE_HISTORICAL_PROJECT_ROOT" \
  "$MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT"; do
  [[ "$mut_trace_path" == /* && -e "$mut_trace_path" && ! -L "$mut_trace_path" ]] || {
    echo "physical absolute Mut trace path required: $mut_trace_path" >&2
    exit 64
  }
done
unset mut_trace_path

[[ "$MUT_TRACE_CONTROLLER_PID" =~ ^[1-9][0-9]*$ ]] || {
  echo "unsafe MUT_TRACE_CONTROLLER_PID" >&2
  exit 64
}
[[ "$MUT_TRACE_CONTROLLER_START_TICKS" =~ ^[1-9][0-9]*$ ]] || {
  echo "unsafe MUT_TRACE_CONTROLLER_START_TICKS" >&2
  exit 64
}
if [[ -n "${MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE:-}" ]]; then
  [[ "$MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE" == /* \
    && -f "$MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE" \
    && ! -L "$MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE" ]] || {
    echo "physical terminal-controller evidence required" >&2
    exit 64
  }
fi

TIMESTAMP="${MUT_TRACE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$TIMESTAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || {
  echo "unsafe MUT_TRACE_TIMESTAMP=$TIMESTAMP" >&2
  exit 64
}

CONTROLLER_ID="$($AUTODL_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["controller_id"])' "$MUT_FAST_SPEC")"
[[ "$CONTROLLER_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "unsafe controller_id=$CONTROLLER_ID" >&2
  exit 64
}
CONTROL_DIR="$AUTODL_CONTROL_ROOT/mut_fast_accurate_v2/$CONTROLLER_ID"
[[ -d "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || {
  echo "live successor control directory missing: $CONTROL_DIR" >&2
  exit 66
}

AUTHORIZATION="${MUT_TRACE_AUTHORIZATION_RECEIPT:-$CONTROL_DIR/trace_on_adoption_authorization_${TIMESTAMP}.json}"
OUTPUT_ROOT="${MUT_TRACE_OUTPUT_ROOT:-$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_trace_on_adoption_${TIMESTAMP}}"
[[ "$AUTHORIZATION" == /* && ! -e "$AUTHORIZATION" && ! -L "$AUTHORIZATION" ]] || {
  echo "fresh physical authorization path required: $AUTHORIZATION" >&2
  exit 64
}
[[ "$OUTPUT_ROOT" == /* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "fresh physical worker output required: $OUTPUT_ROOT" >&2
  exit 64
}

"$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py" \
  --config configs/hpc.yaml authorize \
  --spec "$MUT_FAST_SPEC" \
  --output "$AUTHORIZATION"

LOG_DIR="$AUTODL_RUNTIME_ROOT/logs/mut_trace_on_adoption"
LOG="$LOG_DIR/${CONTROLLER_ID}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
SESSION="cf-mut-trace-adopt-${TIMESTAMP}"
COMMAND=(
  "$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py"
  --config configs/hpc.yaml run
  --spec "$MUT_FAST_SPEC"
  --authorization-receipt "$AUTHORIZATION"
  --protected-manifest "$MUT_TRACE_PROTECTED_MANIFEST"
  --historical-project-root "$MUT_TRACE_HISTORICAL_PROJECT_ROOT"
  --instrumentation-project-root "$MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --controller-pid "$MUT_TRACE_CONTROLLER_PID"
  --controller-start-ticks "$MUT_TRACE_CONTROLLER_START_TICKS"
  --successor-guard-script run_mut_checkpoint_instrumentation_equivalence.py
  --successor-guard-action run-pair
)
if [[ -n "${MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE:-}" ]]; then
  COMMAND+=(
    --terminal-controller-evidence
    "$MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE"
  )
fi

export RUN_GNN_ABLATION=0
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>/dev/null && {
    echo "Mut trace adoption tmux session already exists: $SESSION" >&2
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
echo "authorization_receipt=$AUTHORIZATION"
echo "output_root=$OUTPUT_ROOT"
echo "worker_heartbeat=$CONTROL_DIR/trace_on_adoption_worker_heartbeat.json"
echo "terminal_controller_evidence=${MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE:-none}"
echo "log=$LOG"
echo "gnn_ablation_started=false"
echo "fresh_50k_launched=false"
