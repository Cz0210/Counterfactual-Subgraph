#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

RETRY_REQUESTED="${ALLOW_T14_ROUTE_C_FRESH_RETRY_AFTER_RESOURCE_WATCHDOG:-0}"
[[ "${ALLOW_T14_ROUTE_C:-0}" == "1" || "$RETRY_REQUESTED" == "1" ]] \
  || { echo "ALLOW_T14_ROUTE_C=1 or the exact fresh-retry authorization is required" >&2; exit 64; }
if [[ "$RETRY_REQUESTED" == "1" ]]; then
  [[ "${T14_ROUTE_C_FRESH_RETRY_MAX_ATTEMPTS:-}" == "1" \
    && "${PRESERVE_FAILED_ROUTE_C_ATTEMPT:-}" == "1" \
    && "${REUSE_PARTIAL_STEP161:-}" == "0" ]] \
    || { echo "T14 Route C fresh-retry contract is incomplete" >&2; exit 64; }
fi
[[ "${RUN_GNN_ABLATION:-0}" == "0" && "${RUN_LLM_ABLATION:-0}" == "0" ]] \
  || { echo "Ablation science must remain disabled" >&2; exit 64; }
export RUN_GNN_ABLATION=0
export RUN_LLM_ABLATION=0

RUNTIME="${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}"
CONTROL="${AUTODL_CONTROL_ROOT:-$RUNTIME/control}"
PY="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
LEGACY_ROOT="${T14_FORBIDDEN_LEGACY_ROOT:?required}"
MATRIX_AUTHORITY_ROOT="$CONTROL/fast16_matrix_authority"
MATRIX_STATE="$MATRIX_AUTHORITY_ROOT/state.json"
MATRIX_LOCK="$MATRIX_AUTHORITY_ROOT/publish.lock"
CURRENT="$CONTROL/t14_route_c/current"
mkdir -p "$CONTROL/t14_route_c"
exec 9>"$CONTROL/t14_route_c/launch.lock"
flock -n 9 || { echo "another T14 Route C launch is in progress" >&2; exit 73; }
if [[ ! -e "$CURRENT" && ! -L "$CURRENT" ]]; then
  mkdir -m 700 "$CURRENT"
fi
[[ -d "$CURRENT" && ! -L "$CURRENT" ]] \
  || { echo "T14 Route C current pointer root is invalid" >&2; exit 74; }
FRESH_RETRY_RECEIPT=""

matrix_launch_gate() {
  local receipt="$1"
  "$PY" -I -B -c 'import pathlib,sys; sys.path.insert(0,sys.argv[4]); from src.baselines.tastemolnet_t14_route_c_fresh import audit_route_c_matrix_cell_absent; audit_route_c_matrix_cell_absent(state_path=pathlib.Path(sys.argv[1]),lock_path=pathlib.Path(sys.argv[2]),receipt_path=pathlib.Path(sys.argv[3]))' \
    "$MATRIX_STATE" "$MATRIX_LOCK" "$receipt" "$PROJECT_ROOT"
}

matrix_launch_gate "$CONTROL/t14_route_c/matrix_launch_gate_latest.json"

launch_owner() {
  local spec="$1"
  local owner_root="$2"
  local continuation_spec="$3"
  nohup "$PY" -I -B "$PROJECT_ROOT/scripts/autodl/run_t14_route_c_owner.py" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --task-spec "$spec" \
    --continuation-spec "$continuation_spec" \
    >"$owner_root/owner.out" 2>"$owner_root/owner.err" </dev/null &
  local owner_pid=$!
  local pid_tmp="$CURRENT/.owner.pid.$owner_pid.tmp"
  local ticks_tmp="$CURRENT/.owner.start_ticks.$owner_pid.tmp"
  local spec_tmp="$CURRENT/.task_spec.$owner_pid.tmp"
  printf '%s\n' "$owner_pid" > "$pid_tmp"
  awk '{print $22}' "/proc/$owner_pid/stat" > "$ticks_tmp"
  printf '%s\n' "$spec" > "$spec_tmp"
  mv "$pid_tmp" "$CURRENT/owner.pid"
  mv "$ticks_tmp" "$CURRENT/owner.start_ticks"
  mv "$spec_tmp" "$CURRENT/task_spec.path"
  echo "$owner_pid"
}

if [[ -e "$CURRENT/owner.pid" || -e "$CURRENT/owner.start_ticks" || -e "$CURRENT/task_spec.path" ]]; then
  [[ -f "$CURRENT/owner.pid" && ! -L "$CURRENT/owner.pid" \
    && -f "$CURRENT/owner.start_ticks" && ! -L "$CURRENT/owner.start_ticks" \
    && -f "$CURRENT/task_spec.path" && ! -L "$CURRENT/task_spec.path" ]] \
    || { echo "sealed current Route C owner evidence is incomplete" >&2; exit 74; }
  OLD_PID="$(tr -d '[:space:]' < "$CURRENT/owner.pid")"
  OLD_TICKS="$(tr -d '[:space:]' < "$CURRENT/owner.start_ticks")"
  SEALED_SPEC="$(tr -d '\r\n' < "$CURRENT/task_spec.path")"
  SEALED_CONTINUATION="$(dirname "$SEALED_SPEC")/T14_ROUTE_C_CONTINUATION_SPEC.json"
  [[ "$OLD_PID" =~ ^[1-9][0-9]*$ && "$OLD_TICKS" =~ ^[1-9][0-9]*$ \
    && "$SEALED_SPEC" == /* && -f "$SEALED_SPEC" && ! -L "$SEALED_SPEC" \
    && -f "$SEALED_CONTINUATION" && ! -L "$SEALED_CONTINUATION" ]] \
    || { echo "sealed current Route C owner identity is malformed" >&2; exit 74; }
  if [[ -r "/proc/$OLD_PID/stat" ]]; then
    LIVE_TICKS="$(awk '{print $22}' "/proc/$OLD_PID/stat")"
    if [[ "$LIVE_TICKS" == "$OLD_TICKS" ]]; then
      CMDLINE="$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline")"
      [[ "$CMDLINE" == *"run_t14_route_c_owner.py"* \
        && "$CMDLINE" == *"--task-spec $SEALED_SPEC"* \
        && "$CMDLINE" == *"--continuation-spec $SEALED_CONTINUATION"* ]] \
        || { echo "live PID does not match sealed Route C owner command" >&2; exit 74; }
      echo "WAITING_EXISTING_T14_ROUTE_C_OWNER pid=$OLD_PID" >&2
      exit 75
    fi
  fi
  SEALED_OWNER_ROOT="$($PY -I -B -c '
import json, pathlib, sys
spec_path = pathlib.Path(sys.argv[1])
old_pid, old_ticks = int(sys.argv[2]), int(sys.argv[3])
spec = json.loads(spec_path.read_text(encoding="utf-8"))
owner_root = pathlib.Path(spec["owner_root"])
owner = json.loads((owner_root / "owner.json").read_text(encoding="utf-8"))
if owner.get("task_spec") != str(spec_path): raise SystemExit(74)
if owner.get("task_spec_sha256") != spec.get("spec_sha256"): raise SystemExit(74)
if owner.get("owner_pid") != old_pid or owner.get("owner_start_ticks") != old_ticks: raise SystemExit(74)
print(owner_root)
' "$SEALED_SPEC" "$OLD_PID" "$OLD_TICKS")" \
    || { echo "sealed Route C task/owner evidence failed validation" >&2; exit 74; }
  if [[ "$RETRY_REQUESTED" == "1" ]]; then
    FRESH_RETRY_RECEIPT="$($PY -I -B -c '
import pathlib,sys
sys.path.insert(0,sys.argv[4])
from src.baselines.tastemolnet_t14_route_c_fresh import retire_failed_route_c_current
receipt=retire_failed_route_c_current(
    current_root=pathlib.Path(sys.argv[1]),
    retired_root=pathlib.Path(sys.argv[2]),
)
print(pathlib.Path(receipt["retired_pointer_root"])/"retirement_receipt.json")
' "$CURRENT" "$CONTROL/t14_route_c/retired" "$CONTROL" "$PROJECT_ROOT")" \
      || { echo "failed Route C pointer was not eligible for its one fresh retry" >&2; exit 74; }
    mkdir -m 700 "$CURRENT"
  else
    OWNER_PID="$(launch_owner "$SEALED_SPEC" "$SEALED_OWNER_ROOT" "$SEALED_CONTINUATION")"
    echo "[T14_ROUTE_C_OWNER_SAME_ROOT_RESUMED]"
    echo "t14_route_c_owner_pid=$OWNER_PID"
    echo "t14_route_c_task_spec=$SEALED_SPEC"
    exit 0
  fi
fi

if [[ "$RETRY_REQUESTED" == "1" && -z "$FRESH_RETRY_RECEIPT" ]]; then
  echo "T14 fresh retry requires one eligible failed current pointer" >&2
  exit 74
fi

ATTEMPT="$($PY -c 'import uuid; print(uuid.uuid4())')"
EXECUTION_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
OWNER_ROOT="$CONTROL/t14_route_c/owners/route-c-$ATTEMPT"
OUTPUT_ROOT="$RUNTIME/outputs/autodl/tastemolnet/comrecgc_route_c/route-c-$ATTEMPT"
if [[ -n "$FRESH_RETRY_RECEIPT" ]]; then
  SPEC="$OWNER_ROOT/T14_ROUTE_C_FRESH_RETRY_TASK_SPEC.json"
else
  SPEC="$OWNER_ROOT/T14_ROUTE_C_TASK_SPEC.json"
fi
CONTINUATION_SPEC="$OWNER_ROOT/T14_ROUTE_C_CONTINUATION_SPEC.json"
mkdir -p "$OWNER_ROOT" "$(dirname "$OUTPUT_ROOT")"

SPEC_ARGS=(
  "$PY" -I -B "$PROJECT_ROOT/scripts/autodl/build_t14_route_c_task_spec.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --attempt-uuid "$ATTEMPT" \
  --execution-commit "$EXECUTION_COMMIT" \
  --python "$PY" \
  --science-wrapper "$PROJECT_ROOT/scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh" \
  --owner-entrypoint "$PROJECT_ROOT/scripts/autodl/run_t14_route_c_owner.py" \
  --output-root "$OUTPUT_ROOT" \
  --owner-root "$OWNER_ROOT" \
  --forbidden-legacy-root "$LEGACY_ROOT" \
  --cgroup-limit-file /sys/fs/cgroup/memory/memory.limit_in_bytes \
  --cgroup-current-file /sys/fs/cgroup/memory/memory.usage_in_bytes \
  --cgroup-failcnt-file /sys/fs/cgroup/memory/memory.failcnt \
  --no-live-owner-receipt "$OWNER_ROOT/no_live_t14_owner_receipt.json" \
  --matrix-authority-state "$MATRIX_STATE" \
  --matrix-authority-lock "$MATRIX_LOCK" \
  --matrix-cell-absent-receipt "$OWNER_ROOT/matrix_cell_absent_receipt.json" \
  --launch-headroom-bytes "$((384 * 1024 * 1024 * 1024))" \
  --runtime-headroom-bytes "$((96 * 1024 * 1024 * 1024))" \
  --sample-seconds 30 \
  --launch-samples-required 3 \
  --runtime-low-headroom-samples 3 \
  --spec-out "$SPEC"
)
if [[ -n "$FRESH_RETRY_RECEIPT" ]]; then
  SPEC_ARGS+=(--fresh-retry-receipt "$FRESH_RETRY_RECEIPT")
fi
"${SPEC_ARGS[@]}"

POSTPROCESS_SCIENCE_ROOT="${T14_ROUTE_C_POSTPROCESS_SCIENCE_ROOT:-$RUNTIME/outputs/autodl/tastemolnet/comrecgc_route_c_postprocess/science-$ATTEMPT}"
POSTPROCESS_FINAL_ROOT="${T14_ROUTE_C_POSTPROCESS_FINAL_ROOT:-$RUNTIME/outputs/autodl/tastemolnet/comrecgc_route_c_postprocess/final-$ATTEMPT}"
WNODE_CACHE_DB="${WNODE_CACHE_DB:-$RUNTIME/cache/tastemolnet/t14-route-c/wnode.sqlite}"
NODE_EMBEDDING_CACHE_DIR="${NODE_EMBEDDING_CACHE_DIR:-$RUNTIME/cache/tastemolnet/t14-route-c/molclr_nodes}"

"$PY" -I -B "$PROJECT_ROOT/scripts/autodl/build_t14_route_c_continuation_spec.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  --route-c-spec "$SPEC" \
  --science-root "$POSTPROCESS_SCIENCE_ROOT" \
  --final-root "$POSTPROCESS_FINAL_ROOT" \
  --locator-path "${T14_ROUTE_C_LOCATOR_PATH:?required}" \
  --calibration-csv "${TASTEMOLNET_CALIBRATION_CSV:?required}" \
  --test-csv "${TASTEMOLNET_TEST_CSV:?required}" \
  --t3-output-root "${TASTEMOLNET_T3_OUTPUT_ROOT:?required}" \
  --molclr-root "${MOLCLR_ROOT:?required}" \
  --molclr-checkpoint "${MOLCLR_CHECKPOINT:?required}" \
  --threshold-contract "${TASTEMOLNET_WNODE_THRESHOLD_JSON:?required}" \
  --wnode-cache-db "$WNODE_CACHE_DB" \
  --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR" \
  --autodl-data-root "${AUTODL_DATA_ROOT:-/autodl-fs/data}" \
  --autodl-runtime-root "$RUNTIME" \
  --autodl-control-root "$CONTROL" \
  --publisher-queue-manifest "${T14_ROUTE_C_PUBLISHER_QUEUE_MANIFEST:?required}" \
  --publisher-heartbeat "${T14_ROUTE_C_PUBLISHER_HEARTBEAT:?required}" \
  --publisher-pid-file "${T14_ROUTE_C_PUBLISHER_PID_FILE:?required}" \
  --spec-out "$CONTINUATION_SPEC"

OWNER_PID="$(launch_owner "$SPEC" "$OWNER_ROOT" "$CONTINUATION_SPEC")"

echo "[T14_ROUTE_C_IMPLEMENTATION_PASS]"
if [[ -n "$FRESH_RETRY_RECEIPT" ]]; then
  echo "[T14_FAILED_ATTEMPT_PRESERVED]"
  echo "[T14_STALE_POINTER_RETIRED]"
  echo "[T14_FRESH_RETRY_OWNER_PASS]"
  echo "[T14_ROUTE_C_FRESH_RETRY_LAUNCHED]"
  echo "t14_route_c_retry_retirement_receipt=$FRESH_RETRY_RECEIPT"
fi
echo "t14_route_c_owner_pid=$OWNER_PID"
echo "t14_route_c_task_spec=$SPEC"
echo "t14_route_c_output_root=$OUTPUT_ROOT"
