#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ "${RUN_TASTEMOLNET:-0}" != "1" ]]; then
  echo "RUN_TASTEMOLNET=1 is required by the active Taste downstream policy" >&2
  exit 64
fi

STAGE_SCRIPT="$PROJECT_ROOT/scripts/autodl/tastemolnet_gnn_stage.py"
BASE_POLICY="$PROJECT_ROOT/configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
DOWNSTREAM_POLICY="$PROJECT_ROOT/configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json"
SOURCE_CHECKPOINT="${TASTEMOLNET_T2_BUNDLE:?Set TASTEMOLNET_T2_BUNDLE to the verified immutable T2 bundle}"
T3_OUTPUT="${TASTEMOLNET_T3_OUTPUT:?Set TASTEMOLNET_T3_OUTPUT to the passed T3 evidence root}"
for required in \
  "$STAGE_SCRIPT" \
  "$BASE_POLICY" \
  "$DOWNSTREAM_POLICY" \
  "$SOURCE_CHECKPOINT/sha256sums.txt" \
  "$T3_OUTPUT/gate.json" \
  "$T3_OUTPUT/sha256sums.txt" \
  "$TASTEMOLNET_GRAPH_CACHE_ROOT/manifest.json" \
  "$TASTEMOLNET_GRAPH_CACHE_ROOT/calibration.pt"; do
  autodl_require_file "$required"
done

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
matches = [row for row in payload["gpus"] if row["index"] == 1 and row["stable_idle"] and row["selected"]]
if len(matches) != 1:
    raise SystemExit(75)
print(str(matches[0]["index"]) + "\t" + str(matches[0]["uuid"]))
'
)" || {
  rc=$?
  if [[ $rc -eq 75 ]]; then
    echo "WAITING_FOR_IDLE_GPU1" >&2
  fi
  exit "$rc"
}
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
if [[ "$GPU_INDEX" != "1" || "$GPU_UUID" != GPU-* ]]; then
  echo "GPU1 UUID binding failed" >&2
  exit 64
fi

OUTPUT_DIR="${TASTEMOLNET_T4_OUTPUT:-$(autodl_new_output_dir tastemolnet gine t4-oracle-smoke)}"

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset tastemolnet \
  --stage T4_ORACLE_SMOKE \
  --heavy \
  --gpu-index 1 \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --gpu-lock-mode exclusive \
  --max-gpus 4 \
  --gpu-hard-limit 4 \
  --config-file "$PROJECT_ROOT/configs/hpc.yaml" \
  --config-file "$PROJECT_ROOT/configs/gnn/gine.yaml" \
  --config-file "$PROJECT_ROOT/configs/autodl/tastemolnet_gine_research_v1.yaml" \
  --config-file "$DOWNSTREAM_POLICY" \
  --input-manifest "$T3_OUTPUT/sha256sums.txt" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file oracle_smoke.json \
  --required-output-file oracle_provenance.json \
  --required-output-file data_access_manifest.json \
  --required-output-file policy_binding.json \
  --required-output-file gate.json \
  --required-output-file sha256sums.txt \
  --required-output-file TASTE_MULTICLASS_ORACLE_PASS \
  --required-log-marker "[TASTE_MULTICLASS_ORACLE_PASS]" \
  -- \
  "$AUTODL_PYTHON" -B "$STAGE_SCRIPT" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    t4-oracle-smoke \
    --checkpoint-dir "$SOURCE_CHECKPOINT" \
    --t3-gate "$T3_OUTPUT/gate.json" \
    --graph-cache-root "$TASTEMOLNET_GRAPH_CACHE_ROOT" \
    --artifact-root "$AUTODL_ARTIFACT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --downstream-policy "$DOWNSTREAM_POLICY" \
    --base-policy "$BASE_POLICY" \
    --physical-gpu-index 1 \
    --gpu-uuid "$GPU_UUID" \
    --device cuda:0 \
    --batch-size 32 \
    --source-count 16 \
    --max-deletions-per-parent 4
