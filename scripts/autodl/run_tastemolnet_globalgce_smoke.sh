#!/usr/bin/env bash
set -euo pipefail

# Stage-frozen implementation only. A reviewed one-parent release successor
# may change this literal together with the typed release config. Environment
# variables cannot bypass the tracked refusal.
TASTE_T8_GLOBALGCE_WRAPPER_RELEASED=0
[[ "$TASTE_T8_GLOBALGCE_WRAPPER_RELEASED" == "1" ]] \
  || { echo "TASTE_T8_GLOBALGCE_WRAPPER_NOT_RELEASED" >&2; exit 78; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "${RUN_TASTEMOLNET:-0}" == "1" ]] \
  || { echo "RUN_TASTEMOLNET=1 is required" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-0}" == "1" ]] \
  || { echo "Taste research compute is not authorized" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-0}" == "1" ]] \
  || { echo "Taste aggregate reporting is not authorized" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-1}" == "0" ]] \
  || { echo "Taste data redistribution must remain forbidden" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] \
  || { echo "GNN ablation is outside the Taste main route" >&2; exit 64; }
[[ "${TASTEMOLNET_T8_GPU_INDEX:-2}" == "2" ]] \
  || { echo "T8 GlobalGCE is frozen to physical GPU2" >&2; exit 64; }

: "${TASTEMOLNET_T8_GPU_UUID:?managed physical GPU2 UUID is required}"
: "${TASTEMOLNET_T8_T2_ADOPTION:?T2 adoption root is required}"
: "${TASTEMOLNET_T8_T3_OUTPUT:?T3 output root is required}"
: "${TASTEMOLNET_T8_T4_OUTPUT:?T4 output root is required}"
: "${TASTEMOLNET_T8_GNN_CHECKPOINT:?frozen GINE checkpoint is required}"
: "${TASTEMOLNET_T8_TRAIN_CSV:?frozen train CSV is required}"
: "${TASTEMOLNET_T8_OFFICIAL_ROOT:?pinned official GlobalGCE root is required}"
: "${TASTEMOLNET_T8_STATE:?fresh private T8 state root is required}"
: "${TASTEMOLNET_T8_OUTPUT:?fresh aggregate T8 output root is required}"

[[ ! -e "$TASTEMOLNET_T8_STATE" && ! -L "$TASTEMOLNET_T8_STATE" ]] \
  || { echo "T8 state root must be fresh" >&2; exit 64; }
[[ ! -e "$TASTEMOLNET_T8_OUTPUT" && ! -L "$TASTEMOLNET_T8_OUTPUT" ]] \
  || { echo "T8 output root must be fresh" >&2; exit 64; }

MIN_FREE_AFTER_RESERVATIONS_GB="${MIN_FREE_AFTER_RESERVATIONS_GB:-100}"
TASTEMOLNET_T8_STORAGE_RESERVATION_GB="${TASTEMOLNET_T8_STORAGE_RESERVATION_GB:-20}"
[[ "$MIN_FREE_AFTER_RESERVATIONS_GB" =~ ^[0-9]+$ ]] \
  && (( MIN_FREE_AFTER_RESERVATIONS_GB >= 100 )) \
  || { echo "T8 requires MIN_FREE_AFTER_RESERVATIONS_GB>=100" >&2; exit 64; }
[[ "$TASTEMOLNET_T8_STORAGE_RESERVATION_GB" == "20" ]] \
  || { echo "T8 freezes one 20 GiB managed reservation" >&2; exit 64; }
AVAILABLE_KB="$(df -Pk "$AUTODL_RUNTIME_ROOT" | awk 'NR == 2 {print $4}')"
[[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] \
  && (( AVAILABLE_KB >= (MIN_FREE_AFTER_RESERVATIONS_GB + TASTEMOLNET_T8_STORAGE_RESERVATION_GB) * 1024 * 1024 )) \
  || { echo "T8 storage gate is not ready; no science was started" >&2; exit 75; }

RUNNER="$PROJECT_ROOT/scripts/run_tastemolnet_globalgce_smoke.py"
autodl_require_file "$RUNNER"
autodl_require_file "$PROJECT_ROOT/configs/hpc.yaml"
export PYTHONNOUSERSITE=1

GPU_JSON="$(
  "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$AUTODL_DATA_ROOT" \
    --max-gpus 4 \
    --gpu-hard-limit 4 \
    --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
    --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
    --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" \
    --format json
)"
GPU_LINE="$(
  printf '%s' "$GPU_JSON" | "$AUTODL_PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
matches = [row for row in payload["gpus"] if row["index"] == 2 and row["stable_idle"] and row["selected"]]
if len(matches) != 1:
    raise SystemExit(75)
print(str(matches[0]["index"]) + "\t" + str(matches[0]["uuid"]))
'
)" || {
  rc=$?
  [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU2" >&2
  exit "$rc"
}
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
[[ "$GPU_INDEX" == "2" && "$GPU_UUID" == "$TASTEMOLNET_T8_GPU_UUID" ]] \
  || { echo "T8 managed physical GPU2 UUID binding failed" >&2; exit 64; }

echo "T8_MANAGED_V2_GPU_ACTIVE_AUTHORITY_ADAPTER_NOT_FROZEN" >&2
exit 78

# Unreachable CLI parity only. A later reviewed controller adapter must inject
# the held GPU/ACTIVE authority into the Python worker and independently call
# verify_and_publish_t8_sealed; this wrapper never falls back to managed v1.
export AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0
exec "$AUTODL_PYTHON" -I -B "$RUNNER" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --stage T8_GLOBALGCE_SMOKE \
  --t2-adoption "$TASTEMOLNET_T8_T2_ADOPTION" \
  --t3-output "$TASTEMOLNET_T8_T3_OUTPUT" \
  --t4-output "$TASTEMOLNET_T8_T4_OUTPUT" \
  --gnn-checkpoint "$TASTEMOLNET_T8_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_T8_TRAIN_CSV" \
  --official-root "$TASTEMOLNET_T8_OFFICIAL_ROOT" \
  --downstream-policy "$PROJECT_ROOT/configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json" \
  --base-policy "$PROJECT_ROOT/configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml" \
  --state-dir "$TASTEMOLNET_T8_STATE" \
  --output-dir "$TASTEMOLNET_T8_OUTPUT" \
  --set inference.fallback_to_heuristic=false
