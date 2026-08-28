#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AUTODL_MAX_GPUS="${AUTODL_MAX_GPUS:-4}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-launch}"
case "$ACTION" in
  launch|run|status|request-t4-lease|request-neurosed-lease) ;;
  *) echo "usage: $0 [launch|run|status|request-t4-lease|request-neurosed-lease]" >&2; exit 64 ;;
esac

[[ "${AUTO_TERMINATE_UNCONTROLLED_CHILDREN:-0}" == "0" ]] || {
  echo "AUTO_TERMINATE_UNCONTROLLED_CHILDREN must remain exactly 0" >&2
  exit 64
}
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0

if [[ "$ACTION" == "status" ]]; then
  : "${TASTE_MAIN_V2_CONTROLLER_ROOT:?TASTE_MAIN_V2_CONTROLLER_ROOT is required}"
  exec "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/status_taste_main_v2.py" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --controller-root "$TASTE_MAIN_V2_CONTROLLER_ROOT"
fi

[[ "${RUN_TASTEMOLNET:-}" == "1" ]] || { echo "RUN_TASTEMOLNET must be 1" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-}" == "1" ]] || { echo "TASTE_RESEARCH_COMPUTE_ALLOWED must be 1" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-}" == "1" ]] || { echo "TASTE_PAPER_RESULTS_ALLOWED must be 1" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-}" == "0" ]] || { echo "TASTE_DATA_REDISTRIBUTION_ALLOWED must be 0" >&2; exit 64; }
[[ "${PRIMARY_TASTE_SOURCE_LABEL:-}" == "1" ]] || { echo "PRIMARY_TASTE_SOURCE_LABEL must be 1" >&2; exit 64; }
[[ "${MIN_FREE_AFTER_RESERVATIONS_GB:-}" == "100" ]] || { echo "MIN_FREE_AFTER_RESERVATIONS_GB must be 100" >&2; exit 64; }
[[ "${SCHEDULER_POLL_SECONDS:-}" == "60" ]] || { echo "SCHEDULER_POLL_SECONDS must be 60" >&2; exit 64; }
[[ "${AUTODL_MAX_GPUS:-4}" == "4" ]] || { echo "AUTODL_MAX_GPUS must be 4" >&2; exit 64; }
[[ "${MAX_CONCURRENT_TASTE_FULL:-}" == "2" ]] || { echo "MAX_CONCURRENT_TASTE_FULL must be 2" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-}" == "0" ]] || { echo "RUN_GNN_ABLATION must be 0" >&2; exit 64; }
[[ "$AUTODL_RUNTIME_ROOT" == "$AUTODL_DATA_ROOT/counterfactual-subgraph-runtime" ]] || {
  echo "AUTODL_RUNTIME_ROOT must be the canonical data-root runtime" >&2
  exit 64
}
[[ "$AUTODL_CONTROL_ROOT" == "$AUTODL_RUNTIME_ROOT/control" ]] || {
  echo "AUTODL_CONTROL_ROOT must be the canonical runtime control root" >&2
  exit 64
}
"$AUTODL_PYTHON" -I -B -c 'import shutil,sys; free=shutil.disk_usage(sys.argv[1]).free; minimum=100*1024**3; assert free >= minimum, f"persistent free bytes {free} < {minimum}"' "$AUTODL_RUNTIME_ROOT"

GIT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)"
GIT_TREE="$(git -C "$PROJECT_ROOT" rev-parse --verify 'HEAD^{tree}')"
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
  echo "Taste main-v2 requires a clean immutable checkout" >&2
  exit 73
}

if [[ "$ACTION" == "launch" || "$ACTION" == "run" ]]; then
  CONTROLLER_UUID="${TASTE_MAIN_V2_CONTROLLER_UUID:-$($AUTODL_PYTHON -I -B -c 'import uuid; print(uuid.uuid4())')}"
  CONTROLLER_ID="${TASTE_MAIN_V2_CONTROLLER_ID:-taste-main-v2-$CONTROLLER_UUID}"
  CONTROLLER_ROOT="${TASTE_MAIN_V2_CONTROLLER_ROOT:-$AUTODL_CONTROL_ROOT/taste-main-v2/controllers/$CONTROLLER_UUID}"
  LAUNCHER_ROOT="${TASTE_MAIN_V2_LAUNCHER_ROOT:-$AUTODL_CONTROL_ROOT/taste-main-v2/launches/$CONTROLLER_UUID}"
else
  : "${TASTE_MAIN_V2_CONTROLLER_UUID:?TASTE_MAIN_V2_CONTROLLER_UUID is required}"
  : "${TASTE_MAIN_V2_CONTROLLER_ID:?TASTE_MAIN_V2_CONTROLLER_ID is required}"
  : "${TASTE_MAIN_V2_CONTROLLER_ROOT:?TASTE_MAIN_V2_CONTROLLER_ROOT is required}"
  CONTROLLER_UUID="$TASTE_MAIN_V2_CONTROLLER_UUID"
  CONTROLLER_ID="$TASTE_MAIN_V2_CONTROLLER_ID"
  CONTROLLER_ROOT="$TASTE_MAIN_V2_CONTROLLER_ROOT"
fi

RECEIPT_PATH="$CONTROLLER_ROOT/controller_receipt.json"

if [[ "$ACTION" == "request-t4-lease" || "$ACTION" == "request-neurosed-lease" ]]; then
  if [[ "$ACTION" == "request-t4-lease" ]]; then
    : "${TASTE_T4_PHYSICAL_GPU_UUID:?TASTE_T4_PHYSICAL_GPU_UUID is required}"
    LEASE_TASK="T4_ORACLE_SMOKE"
    LEASE_GPU_INDEX="1"
    LEASE_GPU_UUID="$TASTE_T4_PHYSICAL_GPU_UUID"
  else
    : "${TASTE_NEUROSED_PHYSICAL_GPU_UUID:?TASTE_NEUROSED_PHYSICAL_GPU_UUID is required}"
    LEASE_TASK="TASTE_GCF_NEUROSED"
    LEASE_GPU_INDEX="2"
    LEASE_GPU_UUID="$TASTE_NEUROSED_PHYSICAL_GPU_UUID"
  fi
  exec "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_taste_main_v2.py" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" request-lease \
    --controller-receipt "$RECEIPT_PATH" \
    --task-id "$LEASE_TASK" \
    --physical-gpu-index "$LEASE_GPU_INDEX" \
    --physical-gpu-uuid "$LEASE_GPU_UUID"
fi

LAUNCH_ARGS=(
  --config "$PROJECT_ROOT/configs/hpc.yaml" launch
  --control-root "$AUTODL_CONTROL_ROOT"
  --controller-root "$CONTROLLER_ROOT"
  --launcher-root "$LAUNCHER_ROOT"
  --controller-id "$CONTROLLER_ID"
  --controller-uuid "$CONTROLLER_UUID"
  --project-root "$PROJECT_ROOT"
  --persistent-storage-root "$AUTODL_RUNTIME_ROOT"
  --expected-git-commit "$GIT_COMMIT"
  --expected-git-tree "$GIT_TREE"
)
if [[ "$ACTION" == "run" ]]; then
  LOG_ROOT="$AUTODL_RUNTIME_ROOT/logs/taste-main-v2/$CONTROLLER_UUID"
  mkdir -p "$LOG_ROOT"
  exec "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_taste_main_v2.py" \
    "${LAUNCH_ARGS[@]}" --controller-log "$LOG_ROOT/controller.log"
fi

[[ ! -e "$CONTROLLER_ROOT" && ! -e "$LAUNCHER_ROOT" ]] || {
  echo "controller/launcher UUID namespace is already burned" >&2
  exit 73
}
LOG_ROOT="$AUTODL_RUNTIME_ROOT/logs/taste-main-v2/$CONTROLLER_UUID"
mkdir -p "$LOG_ROOT"
CONTROLLER_LOG_PATH="$LOG_ROOT/controller.log"
SUPERVISOR_LOG_PATH="$LOG_ROOT/launcher-supervisor.log"
nohup "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_taste_main_v2.py" \
  "${LAUNCH_ARGS[@]}" --controller-log "$CONTROLLER_LOG_PATH" \
  >"$SUPERVISOR_LOG_PATH" 2>&1 < /dev/null &
SUPERVISOR_PID="$!"

READY_PATH="$LAUNCHER_ROOT/launcher_ready.json"
for _ in $(seq 1 240); do
  if [[ -s "$READY_PATH" ]]; then
    "$AUTODL_PYTHON" -I -B -c \
      'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert p["state"] == "RUNNING" and p["science_released"] is False; print(json.dumps(p, sort_keys=True))' \
      "$READY_PATH"
    printf 'controller_id=%s\ncontroller_uuid=%s\ncontroller_root=%s\nlauncher_supervisor_pid=%s\nlauncher_ready=%s\ncontroller_log=%s\nsupervisor_log=%s\n' \
      "$CONTROLLER_ID" "$CONTROLLER_UUID" "$CONTROLLER_ROOT" "$SUPERVISOR_PID" \
      "$READY_PATH" "$CONTROLLER_LOG_PATH" "$SUPERVISOR_LOG_PATH"
    exit 0
  fi
  sleep 0.25
done
echo "Taste main-v2 launcher did not attest readiness within 60 seconds; inspect $SUPERVISOR_LOG_PATH" >&2
exit 75
