#!/usr/bin/env bash
set -euo pipefail

# Release requires a one-parent reviewed successor that also fills the static
# config and continuation-controller manifests. Environment cannot bypass it.
TASTE_T9_COMRECGC_WRAPPER_RELEASED=0
[[ "$TASTE_T9_COMRECGC_WRAPPER_RELEASED" == "1" ]] \
  || { echo "TASTE_T9_COMRECGC_WRAPPER_NOT_RELEASED" >&2; exit 78; }

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

for variable in \
  TASTEMOLNET_T9_OUTPUT \
  TASTEMOLNET_T2_ADOPTION_ROOT \
  TASTEMOLNET_T2_ADOPTION_GATE_SHA256 \
  TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256 \
  TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256 \
  TASTEMOLNET_T3_OUTPUT_ROOT \
  TASTEMOLNET_T4_OUTPUT_ROOT \
  TASTEMOLNET_T2_BUNDLE \
  TASTEMOLNET_TRAIN_CSV \
  COMRECGC_OFFICIAL_ROOT \
  TASTEMOLNET_MANAGED_CONTROLLER_MANIFEST \
  TASTEMOLNET_MANAGED_TASK_MANIFEST \
  TASTEMOLNET_MANAGED_RUN_ID; do
  [[ -n "${!variable:-}" ]] \
    || { echo "$variable is required" >&2; exit 64; }
done
[[ ! -e "$TASTEMOLNET_T9_OUTPUT" && ! -L "$TASTEMOLNET_T9_OUTPUT" ]] \
  || { echo "T9 output must be one fresh absent path" >&2; exit 64; }

RUNNER="$PROJECT_ROOT/scripts/run_tastemolnet_comrecgc_smoke.py"
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
[[ "$GPU_INDEX" == "2" && "$GPU_UUID" == GPU-* ]] \
  || { echo "T9 physical GPU2 UUID binding failed" >&2; exit 64; }

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  launch \
  --dataset tastemolnet \
  --stage T9_COMRECGC_SMOKE \
  --run-id "$TASTEMOLNET_MANAGED_RUN_ID" \
  --heavy \
  --foreground \
  --gpu-index 2 \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --gpu-lock-mode exclusive \
  --max-gpus 4 \
  --gpu-hard-limit 4 \
  --managed-controller-manifest "$TASTEMOLNET_MANAGED_CONTROLLER_MANIFEST" \
  --managed-task-manifest "$TASTEMOLNET_MANAGED_TASK_MANIFEST" \
  --execution-receipt-kind taste_t9_gpu2_v1 \
  --strict-result-validator taste_t9_v1 \
  --config-file "$PROJECT_ROOT/configs/hpc.yaml" \
  --expected-output "$TASTEMOLNET_T9_OUTPUT" \
  --required-output-file input_hashes.json \
  --required-output-file state.json \
  --required-output-file manifest.json \
  --required-output-file comrecgc_smoke.json \
  --required-output-file gate.json \
  --required-output-file output_hashes.json \
  --required-output-file PASS \
  -- \
  "$AUTODL_PYTHON" -B "$RUNNER" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --stage T9_COMRECGC_SMOKE \
    --output-dir "$TASTEMOLNET_T9_OUTPUT" \
    --t2-adoption-root "$TASTEMOLNET_T2_ADOPTION_ROOT" \
    --t2-adoption-gate-sha256 "$TASTEMOLNET_T2_ADOPTION_GATE_SHA256" \
    --t2-adoption-receipt-sha256 "$TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256" \
    --t2-source-evidence-sha256 "$TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256" \
    --t3-output-root "$TASTEMOLNET_T3_OUTPUT_ROOT" \
    --t4-output-root "$TASTEMOLNET_T4_OUTPUT_ROOT" \
    --checkpoint-dir "$TASTEMOLNET_T2_BUNDLE" \
    --train-csv "$TASTEMOLNET_TRAIN_CSV" \
    --official-root "$COMRECGC_OFFICIAL_ROOT" \
    --set inference.fallback_to_heuristic=false
