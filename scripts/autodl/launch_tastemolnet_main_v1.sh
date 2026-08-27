#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-launch}"
case "$ACTION" in
  launch|run|restart|status) ;;
  *) echo "usage: $0 [launch|run|restart|status]" >&2; exit 64 ;;
esac

[[ "$RUN_TASTEMOLNET" == "1" ]] || { echo "RUN_TASTEMOLNET must be 1" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-}" == "1" ]] || { echo "TASTE_RESEARCH_COMPUTE_ALLOWED must be 1" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-}" == "1" ]] || { echo "TASTE_PAPER_RESULTS_ALLOWED must be 1" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-}" == "0" ]] || { echo "TASTE_DATA_REDISTRIBUTION_ALLOWED must be 0" >&2; exit 64; }
[[ "${TASTE_UPSTREAM_LICENSE_STATUS:-}" == "NOT_EXPLICITLY_STATED" ]] || { echo "Taste upstream status must remain NOT_EXPLICITLY_STATED" >&2; exit 64; }
[[ "${PRIMARY_TASTE_SOURCE_LABEL:-}" == "1" ]] || { echo "Taste source label must be 1" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-}" == "0" ]] || { echo "GNN backbone ablation must remain disabled" >&2; exit 64; }
[[ "${MAX_CONCURRENT_TASTE_FULL:-}" == "2" ]] || { echo "Taste full concurrency must be 2" >&2; exit 64; }
[[ "${MIN_FREE_AFTER_RESERVATIONS_GB:-}" == "100" ]] || { echo "post-reservation floor must be 100 GiB" >&2; exit 64; }
[[ "${AUTODL_MAX_GPUS:-}" == "4" ]] || { echo "AUTODL_MAX_GPUS must be 4" >&2; exit 64; }

export TASTEMOLNET_GPU_INDEX=1
export TASTEMOLNET_STORAGE_RESERVATION_GB=20
export MIN_PERSISTENT_FREE_GB=100
export MIN_FREE_AFTER_RESERVATIONS_GB=100
export PRIMARY_GNN_BACKBONE=gine
export PRIMARY_SEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export NVIDIA_TF32_OVERRIDE=0
export CUDNN_DETERMINISTIC=1
export TASTEMOLNET_GPU_WAIT_DEADLINE_SECONDS="${TASTEMOLNET_GPU_WAIT_DEADLINE_SECONDS:-604800}"
export TASTEMOLNET_GPU_WAIT_POLL_SECONDS="${TASTEMOLNET_GPU_WAIT_POLL_SECONDS:-30}"

export TASTEMOLNET_POLICY_FILE="${TASTEMOLNET_POLICY_FILE:-$PROJECT_ROOT/configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml}"
autodl_require_file "$TASTEMOLNET_POLICY_FILE"
OBSERVED_POLICY_SHA="$($AUTODL_PYTHON -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$TASTEMOLNET_POLICY_FILE")"
if [[ -n "${TASTEMOLNET_POLICY_SHA256:-}" && "$TASTEMOLNET_POLICY_SHA256" != "$OBSERVED_POLICY_SHA" ]]; then
  echo "Taste policy SHA conflicts with immutable source" >&2
  exit 64
fi
export TASTEMOLNET_POLICY_SHA256="$OBSERVED_POLICY_SHA"
: "${TASTEMOLNET_POLICY_RECEIPT:?TASTEMOLNET_POLICY_RECEIPT is required}"
autodl_require_file "$TASTEMOLNET_POLICY_RECEIPT"
autodl_require_dir "$TASTEMOLNET_PREPARED_ROOT"
autodl_require_dir "$TASTEMOLNET_GRAPH_CACHE_ROOT"

OLD_SOURCE_MANIFEST="${TASTEMOLNET_OLD_SOURCE_MANIFEST:-$AUTODL_CONTROL_ROOT/four_methods_four_datasets_continuation/manifests/four_methods_four_datasets_continuation_v1.json}"
OLD_TASK_ROOT="${TASTEMOLNET_OLD_TASK_ROOT:-$AUTODL_CONTROL_ROOT/four_methods_four_datasets_continuation/four_methods_four_datasets_continuation_v1/tasks/tastemolnet_foundation}"
autodl_require_file "$OLD_SOURCE_MANIFEST"
autodl_require_dir "$OLD_TASK_ROOT"

if [[ "$ACTION" == "launch" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  SHORT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --short=8 HEAD)"
  export TASTEMOLNET_MAIN_CONTROLLER_ID="${TASTEMOLNET_MAIN_CONTROLLER_ID:-tastemolnet-main-v1-${STAMP}-${SHORT_COMMIT}}"
  export TASTEMOLNET_MAIN_CONTROLLER_ROOT="${TASTEMOLNET_MAIN_CONTROLLER_ROOT:-$AUTODL_CONTROL_ROOT/tastemolnet-main-v1/$TASTEMOLNET_MAIN_CONTROLLER_ID}"
  export TASTEMOLNET_GINE_CONTROLLER_CID="${TASTEMOLNET_GINE_CONTROLLER_CID:-tastemolnet_gine_v2_${STAMP}_${SHORT_COMMIT}}"
  export TASTEMOLNET_GINE_CONTROLLER_ROOT="${TASTEMOLNET_GINE_CONTROLLER_ROOT:-$AUTODL_CONTROL_ROOT/tastemolnet-gine-v2/$TASTEMOLNET_GINE_CONTROLLER_CID}"
  export TASTEMOLNET_GNN_FULL_OUTPUT="${TASTEMOLNET_GNN_FULL_OUTPUT:-$AUTODL_RUNTIME_ROOT/outputs/gnn_oracles/tastemolnet/gine/seed7/full-${STAMP}}"
  export TASTEMOLNET_GNN_TRAINING_STATE_ROOT="${TASTEMOLNET_GNN_TRAINING_STATE_ROOT:-$AUTODL_CONTROL_ROOT/tastemolnet-gine-training-v2/$TASTEMOLNET_GINE_CONTROLLER_CID}"
else
  : "${TASTEMOLNET_MAIN_CONTROLLER_ID:?TASTEMOLNET_MAIN_CONTROLLER_ID is required}"
  : "${TASTEMOLNET_MAIN_CONTROLLER_ROOT:?TASTEMOLNET_MAIN_CONTROLLER_ROOT is required}"
  : "${TASTEMOLNET_GINE_CONTROLLER_CID:?TASTEMOLNET_GINE_CONTROLLER_CID is required}"
  : "${TASTEMOLNET_GINE_CONTROLLER_ROOT:?TASTEMOLNET_GINE_CONTROLLER_ROOT is required}"
  : "${TASTEMOLNET_GNN_FULL_OUTPUT:?TASTEMOLNET_GNN_FULL_OUTPUT is required}"
  : "${TASTEMOLNET_GNN_TRAINING_STATE_ROOT:?TASTEMOLNET_GNN_TRAINING_STATE_ROOT is required}"
fi
export TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT="$TASTEMOLNET_GINE_CONTROLLER_ROOT/published_output_resume_adoption.json"

export TASTE_MATRIX_STATUS_PATH="${TASTE_MATRIX_STATUS_PATH:-$AUTODL_RUNTIME_ROOT/outputs/autodl/paper_matrix/four_methods_four_datasets_v1/matrix_status.json}"
MAIN_ARGS=(
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --controller-id "$TASTEMOLNET_MAIN_CONTROLLER_ID"
  --control-root "$AUTODL_CONTROL_ROOT"
  --runtime-root "$AUTODL_RUNTIME_ROOT"
  --controller-root "$TASTEMOLNET_MAIN_CONTROLLER_ROOT"
  --old-source-manifest "$OLD_SOURCE_MANIFEST"
  --old-task-root "$OLD_TASK_ROOT"
  --policy "$TASTEMOLNET_POLICY_FILE"
  --policy-receipt "$TASTEMOLNET_POLICY_RECEIPT"
  --prepared-root "$TASTEMOLNET_PREPARED_ROOT"
  --graph-cache-root "$TASTEMOLNET_GRAPH_CACHE_ROOT"
  --project-root "$PROJECT_ROOT"
  --gine-controller-root "$TASTEMOLNET_GINE_CONTROLLER_ROOT"
  --gine-output-root "$TASTEMOLNET_GNN_FULL_OUTPUT"
  --gine-training-state-root "$TASTEMOLNET_GNN_TRAINING_STATE_ROOT"
  --reservation-gb 20
  --minimum-free-after-reservations-gb 100
)

if [[ "$ACTION" == "status" ]]; then
  exec "$AUTODL_PYTHON" "$SCRIPT_DIR/status_tastemolnet_main_v1.py" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --controller-root "$TASTEMOLNET_MAIN_CONTROLLER_ROOT"
fi
if [[ "$ACTION" == "run" ]]; then
  exec "$AUTODL_PYTHON" "$SCRIPT_DIR/run_tastemolnet_main_v1.py" run "${MAIN_ARGS[@]}"
fi
if [[ "$ACTION" == "restart" ]]; then
  exec "$AUTODL_PYTHON" "$SCRIPT_DIR/run_tastemolnet_main_v1.py" run "${MAIN_ARGS[@]}" --resume
fi

[[ ! -e "$AUTODL_CONTROL_ROOT/tastemolnet-main-v1" ]] || { echo "Taste main namespace already exists; use explicit restart authority" >&2; exit 73; }
[[ ! -e "$TASTEMOLNET_GINE_CONTROLLER_ROOT" ]] || { echo "Taste GINE controller root must be fresh" >&2; exit 73; }
[[ ! -e "$TASTEMOLNET_GNN_FULL_OUTPUT" ]] || { echo "Taste GINE output root must be fresh" >&2; exit 73; }
[[ ! -e "$TASTEMOLNET_GNN_TRAINING_STATE_ROOT" ]] || { echo "Taste GINE training-state root must be fresh" >&2; exit 73; }
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || { echo "Taste launch requires a clean immutable worktree" >&2; exit 73; }

LOG_ROOT="$AUTODL_RUNTIME_ROOT/logs/tastemolnet-main-v1/$TASTEMOLNET_MAIN_CONTROLLER_ID"
mkdir -p "$LOG_ROOT"
LOG_PATH="$LOG_ROOT/controller.log"
PID_PATH="$LOG_ROOT/controller.pid"
nohup "$AUTODL_PYTHON" "$SCRIPT_DIR/run_tastemolnet_main_v1.py" run "${MAIN_ARGS[@]}" \
  >"$LOG_PATH" 2>&1 < /dev/null &
CONTROLLER_PID=$!
printf '%s\n' "$CONTROLLER_PID" >"$PID_PATH"

for _ in $(seq 1 60); do
  if [[ -s "$TASTEMOLNET_MAIN_CONTROLLER_ROOT/heartbeat.json" ]]; then
    printf 'controller_id=%s\ncontroller_pid=%s\ncontroller_root=%s\nlog=%s\n' \
      "$TASTEMOLNET_MAIN_CONTROLLER_ID" "$CONTROLLER_PID" \
      "$TASTEMOLNET_MAIN_CONTROLLER_ROOT" "$LOG_PATH"
    exit 0
  fi
  if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    echo "Taste main controller exited before heartbeat; inspect $LOG_PATH" >&2
    exit 70
  fi
  sleep 1
done
echo "Taste main controller did not publish a heartbeat within 60 seconds" >&2
exit 75
