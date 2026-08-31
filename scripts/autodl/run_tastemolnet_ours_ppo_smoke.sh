#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

# Successor authority e871236 retains the reviewed GPU0 policy and exact
# model/predecessor pins, while authorizing the set-semantic PEFT reload fix;
# its immutable JSON was independently reloaded before this release-only
# commit. No environment variable can bypass this boundary.
TASTE_T6_WRAPPER_RELEASED=1
[[ "$TASTE_T6_WRAPPER_RELEASED" == "1" ]] \
  || { echo "TASTE_T6_WRAPPER_NOT_RELEASED" >&2; exit 78; }

[[ "${RUN_TASTEMOLNET:-0}" == "1" ]] || { echo "RUN_TASTEMOLNET=1 is required" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-0}" == "1" ]] || { echo "Taste research compute is not authorized" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-0}" == "1" ]] || { echo "Taste paper reporting is not authorized" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-1}" == "0" ]] || { echo "Taste data redistribution must remain forbidden" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "GNN ablation is outside the Taste main route" >&2; exit 64; }
[[ "${TASTEMOLNET_T6_GPU_INDEX:-0}" == "0" ]] || { echo "T6 Ours smoke is frozen to physical GPU0" >&2; exit 64; }

MIN_PERSISTENT_FREE_GB="${MIN_PERSISTENT_FREE_GB:-100}"
MIN_FREE_AFTER_RESERVATIONS_GB="${MIN_FREE_AFTER_RESERVATIONS_GB:-100}"
TASTEMOLNET_T6_STORAGE_RESERVATION_GB="${TASTEMOLNET_T6_STORAGE_RESERVATION_GB:-20}"
[[ "$MIN_PERSISTENT_FREE_GB" =~ ^[0-9]+$ ]] && (( MIN_PERSISTENT_FREE_GB >= 100 )) \
  || { echo "T6 requires MIN_PERSISTENT_FREE_GB>=100" >&2; exit 64; }
[[ "$MIN_FREE_AFTER_RESERVATIONS_GB" =~ ^[0-9]+$ ]] && (( MIN_FREE_AFTER_RESERVATIONS_GB >= 100 )) \
  || { echo "T6 requires MIN_FREE_AFTER_RESERVATIONS_GB>=100" >&2; exit 64; }
[[ "$TASTEMOLNET_T6_STORAGE_RESERVATION_GB" == "20" ]] \
  || { echo "T6 freezes a 20 GiB planning reservation" >&2; exit 64; }
AVAILABLE_KB="$(df -Pk "$AUTODL_RUNTIME_ROOT" | awk 'NR == 2 {print $4}')"
[[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] \
  && (( AVAILABLE_KB >= (MIN_FREE_AFTER_RESERVATIONS_GB + TASTEMOLNET_T6_STORAGE_RESERVATION_GB) * 1024 * 1024 )) \
  || { echo "T6 storage gate is not ready; no science was started" >&2; exit 75; }

: "${TASTEMOLNET_T2_BUNDLE:?TASTEMOLNET_T2_BUNDLE is required}"
: "${TASTEMOLNET_T3_CHECKPOINT:?TASTEMOLNET_T3_CHECKPOINT is required}"
: "${TASTEMOLNET_T5_OUTPUT:?TASTEMOLNET_T5_OUTPUT is required}"
: "${TASTEMOLNET_T6_OUTPUT:?TASTEMOLNET_T6_OUTPUT is required and must be fresh}"
: "${TASTEMOLNET_TRAIN_CSV:?TASTEMOLNET_TRAIN_CSV is required}"
: "${CHEMLLM_MODEL_PATH:?CHEMLLM_MODEL_PATH is required}"

BASE_POLICY="$PROJECT_ROOT/configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
DOWNSTREAM_POLICY="$PROJECT_ROOT/configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json"

TRAIN_SCRIPT="$PROJECT_ROOT/scripts/train_tastemolnet_gnn_ppo.py"
for required in \
  "$TRAIN_SCRIPT" \
  "$PROJECT_ROOT/configs/hpc.yaml" \
  "$BASE_POLICY" \
  "$DOWNSTREAM_POLICY" \
  "$TASTEMOLNET_T2_BUNDLE/sha256sums.txt" \
  "$TASTEMOLNET_T3_CHECKPOINT/model.pt" \
  "$TASTEMOLNET_T3_CHECKPOINT/sha256sums.txt" \
  "$TASTEMOLNET_T5_OUTPUT/gate.json" \
  "$TASTEMOLNET_T5_OUTPUT/verification.json" \
  "$TASTEMOLNET_T5_OUTPUT/PASS" \
  "$TASTEMOLNET_TRAIN_CSV"; do
  autodl_require_file "$required"
done
autodl_require_dir "$CHEMLLM_MODEL_PATH"
[[ ! -e "$TASTEMOLNET_T6_OUTPUT" && ! -L "$TASTEMOLNET_T6_OUTPUT" ]] \
  || { echo "T6 output must be a fresh absent path" >&2; exit 64; }

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
  || { echo "T6 physical GPU0 UUID binding failed" >&2; exit 64; }

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset tastemolnet \
  --stage T6_OURS_SMOKE \
  --heavy \
  --gpu-index 0 \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --gpu-lock-mode exclusive \
  --max-gpus 4 \
  --gpu-hard-limit 4 \
  --config-file "$PROJECT_ROOT/configs/hpc.yaml" \
  --config-file "$DOWNSTREAM_POLICY" \
  --input-manifest "$TASTEMOLNET_T5_OUTPUT/verification.json" \
  --expected-output "$TASTEMOLNET_T6_OUTPUT" \
  --required-output-file manifest.json \
  --required-output-file state.json \
  --required-output-file gate.json \
  --required-output-file input_hashes.json \
  --required-output-file output_hashes.json \
  --required-output-file ppo_smoke_manifest.json \
  --required-output-file policy_provenance.json \
  --required-output-file downstream_policy_binding.json \
  --required-output-file parent_selection.json \
  --required-output-file candidate_pool.jsonl \
  --required-output-file adapter_config.json \
  --required-output-file adapter_model.safetensors \
  --required-output-file PASS \
  --required-log-marker '[TASTE_T6_OURS_PPO_SMOKE_PASS]' \
  -- \
  "$AUTODL_PYTHON" -B "$TRAIN_SCRIPT" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --stage T6_OURS_SMOKE \
    --model-path "$CHEMLLM_MODEL_PATH" \
    --dataset-path "$TASTEMOLNET_TRAIN_CSV" \
    --output-dir "$TASTEMOLNET_T6_OUTPUT" \
    --gnn-checkpoint "$TASTEMOLNET_T3_CHECKPOINT" \
    --t5-output "$TASTEMOLNET_T5_OUTPUT" \
    --downstream-policy "$DOWNSTREAM_POLICY" \
    --base-policy "$BASE_POLICY" \
    --gnn-device cuda \
    --oracle-batch-size "${TASTEMOLNET_T6_ORACLE_BATCH_SIZE:-256}" \
    --updates "${TASTEMOLNET_T6_UPDATES:-5}" \
    --parent-count "${TASTEMOLNET_T6_PARENT_COUNT:-16}" \
    --batch-size "${TASTEMOLNET_T6_BATCH_SIZE:-2}" \
    --seed 7
