#!/usr/bin/env bash
set -euo pipefail

# This implementation commit is stage-frozen, not released.  A reviewed
# one-parent successor may change this literal only together with the typed,
# SHA-pinned release config.  Environment variables cannot bypass it.
TASTE_T7_GCF_WRAPPER_RELEASED=0
[[ "$TASTE_T7_GCF_WRAPPER_RELEASED" == "1" ]] \
  || { echo "TASTE_T7_GCF_WRAPPER_NOT_RELEASED" >&2; exit 78; }

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
[[ "${TASTEMOLNET_T7_GPU_INDEX:-0}" == "0" ]] \
  || { echo "T7 GCF smoke is frozen to physical GPU0" >&2; exit 64; }

: "${TASTEMOLNET_T7_OUTPUT:?TASTEMOLNET_T7_OUTPUT is required and must be fresh}"
[[ ! -e "$TASTEMOLNET_T7_OUTPUT" && ! -L "$TASTEMOLNET_T7_OUTPUT" ]] \
  || { echo "T7 output must be one fresh absent path" >&2; exit 64; }

MIN_FREE_AFTER_RESERVATIONS_GB="${MIN_FREE_AFTER_RESERVATIONS_GB:-100}"
TASTEMOLNET_T7_STORAGE_RESERVATION_GB="${TASTEMOLNET_T7_STORAGE_RESERVATION_GB:-10}"
[[ "$MIN_FREE_AFTER_RESERVATIONS_GB" =~ ^[0-9]+$ ]] \
  && (( MIN_FREE_AFTER_RESERVATIONS_GB >= 100 )) \
  || { echo "T7 requires MIN_FREE_AFTER_RESERVATIONS_GB>=100" >&2; exit 64; }
[[ "$TASTEMOLNET_T7_STORAGE_RESERVATION_GB" == "10" ]] \
  || { echo "T7 freezes a 10 GiB planning reservation" >&2; exit 64; }
AVAILABLE_KB="$(df -Pk "$AUTODL_RUNTIME_ROOT" | awk 'NR == 2 {print $4}')"
[[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] \
  && (( AVAILABLE_KB >= (MIN_FREE_AFTER_RESERVATIONS_GB + TASTEMOLNET_T7_STORAGE_RESERVATION_GB) * 1024 * 1024 )) \
  || { echo "T7 storage gate is not ready; no science was started" >&2; exit 75; }

RUNNER="$PROJECT_ROOT/scripts/run_tastemolnet_gcf_smoke.py"
autodl_require_file "$RUNNER"
autodl_require_file "$PROJECT_ROOT/configs/hpc.yaml"

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
matches = [row for row in payload["gpus"] if row["index"] == 0 and row["stable_idle"] and row["selected"]]
if len(matches) != 1:
    raise SystemExit(75)
print(str(matches[0]["index"]) + "\t" + str(matches[0]["uuid"]))
'
)" || {
  rc=$?
  [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU0" >&2
  exit "$rc"
}
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
[[ "$GPU_INDEX" == "0" && "$GPU_UUID" == GPU-* ]] \
  || { echo "T7 physical GPU0 UUID binding failed" >&2; exit 64; }

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset tastemolnet \
  --stage T7_GCF_SMOKE \
  --heavy \
  --gpu-index 0 \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --gpu-lock-mode exclusive \
  --max-gpus 4 \
  --gpu-hard-limit 4 \
  --config-file "$PROJECT_ROOT/configs/hpc.yaml" \
  --expected-output "$TASTEMOLNET_T7_OUTPUT" \
  --required-output-file input_hashes.json \
  --required-output-file state.json \
  --required-output-file manifest.json \
  --required-output-file gcf_smoke.json \
  --required-output-file candidate_trace.jsonl \
  --required-output-file gate.json \
  --required-output-file output_hashes.json \
  --required-output-file PASS \
  -- \
  "$AUTODL_PYTHON" -B "$RUNNER" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --stage T7_GCF_SMOKE \
    --output-dir "$TASTEMOLNET_T7_OUTPUT" \
    --set inference.fallback_to_heuristic=false
