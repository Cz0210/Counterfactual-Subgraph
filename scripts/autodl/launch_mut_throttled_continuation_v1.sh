#!/usr/bin/env bash
# Launch exactly one low-priority Mut adoption continuation on GPU0.
# This script never starts a fresh 50k generation and never touches T8/T12/T14.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${MUT_FAST_SPEC:?absolute existing Mut spec required}"
: "${MUT_TRACE_AUTHORIZATION_RECEIPT:?existing authorization receipt required}"
: "${MUT_TRACE_PROTECTED_MANIFEST:?absolute protected T14 manifest required}"
: "${MUT_TRACE_HISTORICAL_PROJECT_ROOT:?immutable historical worktree required}"
: "${MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT:?immutable instrumentation worktree required}"
: "${MUT_TRACE_SEMANTIC_FINALIZER_PROJECT_ROOT:?immutable finalizer worktree required}"
: "${MUT_TRACE_CONTROLLER_PID:?terminal/live controller PID identity required}"
: "${MUT_TRACE_CONTROLLER_START_TICKS:?controller start ticks required}"
: "${MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE:?terminal attachment receipt required}"
: "${MUT_COMPLETED_A_ARM_ROOT:?completed read-only Mut A/legacy arm required}"

for mut_path in \
  "$MUT_FAST_SPEC" \
  "$MUT_TRACE_AUTHORIZATION_RECEIPT" \
  "$MUT_TRACE_PROTECTED_MANIFEST" \
  "$MUT_TRACE_HISTORICAL_PROJECT_ROOT" \
  "$MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT" \
  "$MUT_TRACE_SEMANTIC_FINALIZER_PROJECT_ROOT" \
  "$MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE" \
  "$MUT_COMPLETED_A_ARM_ROOT"; do
  [[ "$mut_path" == /* && -e "$mut_path" && ! -L "$mut_path" ]] || {
    echo "physical absolute Mut continuation input required: $mut_path" >&2
    exit 64
  }
done
unset mut_path

[[ "$MUT_TRACE_CONTROLLER_PID" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "$MUT_TRACE_CONTROLLER_START_TICKS" =~ ^[1-9][0-9]*$ ]] || exit 64

# Read-only preflight.  The worker repeats this check while holding its
# canonical lease, so this early check is diagnostic rather than authoritative.
"$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py" \
  --config configs/hpc.yaml owner-preflight \
  --proc-root /proc \
  --controller-pid "$MUT_TRACE_CONTROLLER_PID"

MUT_CPUSET="$($AUTODL_PYTHON "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py" \
  --config configs/hpc.yaml select-cpus --proc-root /proc --sample-seconds 1)"
[[ "$MUT_CPUSET" =~ ^[0-9]+,[0-9]+$ ]] || {
  echo "two verified non-SMT Mut CPUs were not selected: $MUT_CPUSET" >&2
  exit 65
}

TIMESTAMP="${MUT_TRACE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$TIMESTAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || exit 64
CONTROLLER_ID="$($AUTODL_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["controller_id"])' "$MUT_FAST_SPEC")"
[[ "$CONTROLLER_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
CONTROL_DIR="$AUTODL_CONTROL_ROOT/mut_fast_accurate_v2/$CONTROLLER_ID"
[[ -d "$CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || exit 66

OUTPUT_ROOT="${MUT_TRACE_OUTPUT_ROOT:-$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/mut_throttled_continuation_${TIMESTAMP}}"
[[ "$OUTPUT_ROOT" == /* && ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "fresh Mut continuation output root required: $OUTPUT_ROOT" >&2
  exit 64
}

export RUN_GNN_ABLATION=0
export MUT_PROTECTED_BASELINE_MAX_WAIT_SECONDS=1800
export MUT_EXACT_WORKERS=2
export MUT_CPU_WORKERS=2
export MUT_PREFETCH=1
export MUT_PREFETCH_FACTOR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

LOG_DIR="$AUTODL_RUNTIME_ROOT/logs/mut_throttled_continuation"
LOG="$LOG_DIR/${CONTROLLER_ID}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
SESSION="cf-mut-throttled-${TIMESTAMP}"
COMMAND=(
  nice -n 10
  ionice -c 2 -n 7
  taskset -c "$MUT_CPUSET"
  "$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py"
  --config configs/hpc.yaml run
  --spec "$MUT_FAST_SPEC"
  --authorization-receipt "$MUT_TRACE_AUTHORIZATION_RECEIPT"
  --protected-manifest "$MUT_TRACE_PROTECTED_MANIFEST"
  --historical-project-root "$MUT_TRACE_HISTORICAL_PROJECT_ROOT"
  --instrumentation-project-root "$MUT_TRACE_INSTRUMENTATION_PROJECT_ROOT"
  --semantic-finalizer-project-root "$MUT_TRACE_SEMANTIC_FINALIZER_PROJECT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --adopt-complete-legacy-root "$MUT_COMPLETED_A_ARM_ROOT"
  --controller-pid "$MUT_TRACE_CONTROLLER_PID"
  --controller-start-ticks "$MUT_TRACE_CONTROLLER_START_TICKS"
  --terminal-controller-evidence "$MUT_TRACE_TERMINAL_CONTROLLER_EVIDENCE"
  --successor-guard-script run_mut_checkpoint_instrumentation_equivalence.py
  --successor-guard-action run-pair
  --throttle-profile robust-v2
)

if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>/dev/null && exit 3
  printf -v TMUX_COMMAND '%q ' "${COMMAND[@]}"
  tmux new-session -d -s "$SESSION" "$TMUX_COMMAND >>$(printf '%q' "$LOG") 2>&1"
  echo "launcher=tmux session=$SESSION"
else
  nohup "${COMMAND[@]}" >>"$LOG" 2>&1 </dev/null &
  echo "launcher=nohup pid=$!"
fi

echo "controller_id=$CONTROLLER_ID"
echo "cpu_affinity=$MUT_CPUSET"
echo "output_root=$OUTPUT_ROOT"
echo "heartbeat=$CONTROL_DIR/trace_on_adoption_worker_heartbeat.json"
echo "log=$LOG"
echo "fresh_50k_launched=false"
echo "pair_store_recomputed=false"
echo "dbscan_recomputed=false"
