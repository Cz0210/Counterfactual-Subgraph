#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "${RUN_TASTEMOLNET:-0}" == "1" ]] || { echo "RUN_TASTEMOLNET=1 is required" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-0}" == "1" ]] || { echo "Taste research compute is not authorized" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-0}" == "1" ]] || { echo "Taste paper reporting is not authorized" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-1}" == "0" ]] || { echo "Taste redistribution must remain forbidden" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "GNN ablation remains disabled before 16/16" >&2; exit 64; }

for variable in \
  TASTEMOLNET_T14_GENERATION_ROOT TASTEMOLNET_T14_POSTPROCESS_ROOT \
  TASTEMOLNET_T14_FINAL_ROOT TASTEMOLNET_CALIBRATION_CSV TASTEMOLNET_TEST_CSV \
  TASTEMOLNET_T3_OUTPUT_ROOT MOLCLR_ROOT MOLCLR_CHECKPOINT \
  TASTEMOLNET_WNODE_THRESHOLD_JSON WNODE_CACHE_DB NODE_EMBEDDING_CACHE_DIR \
  TASTEMOLNET_T14_POSTPROCESS_RUN_ID; do
  [[ -n "${!variable:-}" ]] || { echo "$variable is required" >&2; exit 64; }
done

TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX="${TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX:-2}"
TASTEMOLNET_T14_POSTPROCESS_RESUME="${TASTEMOLNET_T14_POSTPROCESS_RESUME:-0}"
[[ "$TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX" =~ ^[0-3]$ ]] \
  || { echo "TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX must be 0..3" >&2; exit 64; }
[[ "$TASTEMOLNET_T14_POSTPROCESS_RESUME" =~ ^[01]$ ]] \
  || { echo "TASTEMOLNET_T14_POSTPROCESS_RESUME must be 0 or 1" >&2; exit 64; }
[[ -f "$TASTEMOLNET_T14_GENERATION_ROOT/GENERATION_PASS" ]] \
  || { echo "T14 generation PASS is required" >&2; exit 75; }
[[ ! -e "$TASTEMOLNET_T14_FINAL_ROOT" && ! -L "$TASTEMOLNET_T14_FINAL_ROOT" ]] \
  || { echo "T14 final root must be fresh" >&2; exit 64; }

GPU_JSON="$(
  "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
    --project-root "$PROJECT_ROOT" --data-root "$AUTODL_DATA_ROOT" \
    --max-gpus 4 --gpu-hard-limit 4 \
    --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
    --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
    --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" --format json
)"
GPU_UUID="$(printf '%s' "$GPU_JSON" | "$AUTODL_PYTHON" -c '
import json, sys
index = int(sys.argv[1])
rows = [row for row in json.load(sys.stdin)["gpus"] if row["index"] == index and row["stable_idle"]]
if len(rows) != 1: raise SystemExit(75)
print(rows[0]["uuid"])
' "$TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX")" || {
  rc=$?
  [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU_FOR_T14_POSTPROCESS" >&2
  exit "$rc"
}

COMMON_ARGS=(
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --generation-root "$TASTEMOLNET_T14_GENERATION_ROOT"
  --science-root "$TASTEMOLNET_T14_POSTPROCESS_ROOT"
  --calibration-csv "$TASTEMOLNET_CALIBRATION_CSV"
  --test-csv "$TASTEMOLNET_TEST_CSV"
  --gnn-checkpoint "$TASTEMOLNET_T3_OUTPUT_ROOT/artifacts/checkpoint"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --threshold-contract "$TASTEMOLNET_WNODE_THRESHOLD_JSON"
  --set inference.fallback_to_heuristic=false
)
POSTPROCESS_ARGS=(
  "$AUTODL_PYTHON" -I -B "$PROJECT_ROOT/scripts/run_tastemolnet_comrecgc_postprocess.py"
  --mode postprocess "${COMMON_ARGS[@]}"
  --wnode-cache-db "$WNODE_CACHE_DB"
  --node-embedding-cache-dir "$NODE_EMBEDDING_CACHE_DIR"
  --device cuda:0
)
if [[ "$TASTEMOLNET_T14_POSTPROCESS_RESUME" == "1" ]]; then
  POSTPROCESS_ARGS+=(--resume)
fi

"$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_lock.py" \
  --project-root "$PROJECT_ROOT" --data-root "$AUTODL_DATA_ROOT" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" run \
  --gpu-index "$TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX" \
  --gpu-uuid "$GPU_UUID" --run-id "$TASTEMOLNET_T14_POSTPROCESS_RUN_ID" -- \
  "${POSTPROCESS_ARGS[@]}"

exec "$AUTODL_PYTHON" -I -B \
  "$PROJECT_ROOT/scripts/run_tastemolnet_comrecgc_postprocess.py" \
  --mode verify "${COMMON_ARGS[@]}" --final-root "$TASTEMOLNET_T14_FINAL_ROOT"
