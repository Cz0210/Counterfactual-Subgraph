#!/usr/bin/env bash
# Launch one fresh-vs-fresh, low-priority Mut adoption successor.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${MUT_PREDECESSOR_SPEC:?absolute predecessor task spec required}"
: "${MUT_PREDECESSOR_TERMINAL:?absolute predecessor terminal required}"

[[ "$MUT_PREDECESSOR_SPEC" == /* && -f "$MUT_PREDECESSOR_SPEC" && ! -L "$MUT_PREDECESSOR_SPEC" ]] || exit 64
[[ "$MUT_PREDECESSOR_TERMINAL" == /* && -f "$MUT_PREDECESSOR_TERMINAL" && ! -L "$MUT_PREDECESSOR_TERMINAL" ]] || exit 64

LOCK_ROOT="$AUTODL_CONTROL_ROOT/final-main16-closeout-v2"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/mut-successor-submit.lock"
flock -n 9 || {
  echo "mut successor submit lock is already held" >&2
  exit 75
}

# Refuse a duplicate before allocating any output path.  The owner repeats
# this check while binding its fresh task spec.
"$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" owner-preflight \
  --proc-root /proc

ATTEMPT_UUID="$(tr 'A-F' 'a-f' </proc/sys/kernel/random/uuid)"
[[ "$ATTEMPT_UUID" =~ ^[0-9a-f-]{36}$ ]] || exit 65
SHORT_ID="${ATTEMPT_UUID%%-*}"
TASK_ID="mut-clean-successor-$SHORT_ID"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SPEC_ROOT="$AUTODL_CONTROL_ROOT/main-ready-task-specs/${STAMP}-${TASK_ID}"
OWNER_RUNTIME_ROOT="$AUTODL_CONTROL_ROOT/final-main16-closeout-v2/owners/$TASK_ID/runtime"
SCIENCE_OUTPUT_ROOT="$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs/${TASK_ID}-${ATTEMPT_UUID}"
LEASE_PATH="$AUTODL_CONTROL_ROOT/main-ready-dispatch-leases/mut-gpu0-${SHORT_ID}.lock"
PREFLIGHT="$AUTODL_CONTROL_ROOT/final-main16-closeout-v2/${TASK_ID}-preflight.json"
GPU_UUID="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, '$1 ~ /^[[:space:]]*0[[:space:]]*$/ {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')"
[[ "$GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]] || {
  echo "cannot resolve physical GPU0 UUID" >&2
  exit 69
}

"$AUTODL_PYTHON" -m src.utils.mut_clean_successor_v2 \
  --prior-spec "$MUT_PREDECESSOR_SPEC" \
  --prior-terminal "$MUT_PREDECESSOR_TERMINAL" \
  --repo-root "$PROJECT_ROOT" \
  --spec-root "$SPEC_ROOT" \
  --owner-runtime-root "$OWNER_RUNTIME_ROOT" \
  --science-output-root "$SCIENCE_OUTPUT_ROOT" \
  --lease-path "$LEASE_PATH" \
  --task-id "$TASK_ID" \
  --attempt-uuid "$ATTEMPT_UUID" \
  --gpu-index 0 \
  --gpu-uuid "$GPU_UUID" \
  --preflight-output "$PREFLIGHT"

SPEC="$SPEC_ROOT/$TASK_ID.json"
MUT_CPUSET="$("$AUTODL_PYTHON" "$SCRIPT_DIR/run_mut_trace_on_adoption_worker.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" select-cpus \
  --proc-root /proc --sample-seconds 1)"
[[ "$MUT_CPUSET" =~ ^[0-9]+,[0-9]+$ ]] || exit 65

LOG_ROOT="$AUTODL_RUNTIME_ROOT/logs/final-main16-closeout-v2"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/${TASK_ID}.log"

export RUN_GNN_ABLATION=0
export RUN_LLM_ABLATION=0
export MUT_PROTECTED_BASELINE_MAX_WAIT_SECONDS=1800
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

nohup nice -n 10 ionice -c 2 -n 7 taskset -c "$MUT_CPUSET" \
  "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_mut_clean_trace_equivalence_v1.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --task-spec "$SPEC" \
  >>"$LOG" 2>&1 </dev/null &
OWNER_PID=$!

echo "task_id=$TASK_ID"
echo "attempt_uuid=$ATTEMPT_UUID"
echo "owner_pid=$OWNER_PID"
echo "spec=$SPEC"
echo "science_output_root=$SCIENCE_OUTPUT_ROOT"
echo "owner_heartbeat=$OWNER_RUNTIME_ROOT/heartbeat.json"
echo "cpu_affinity=$MUT_CPUSET"
echo "log=$LOG"
echo "fresh_vs_fresh=true"
echo "pair_store_recomputed=false"
echo "dbscan_recomputed=false"
