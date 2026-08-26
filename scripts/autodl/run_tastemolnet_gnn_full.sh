#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "$RUN_TASTEMOLNET" == "1" ]] || { echo "RUN_TASTEMOLNET must be 1" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-}" == "1" ]] || { echo "TASTE_RESEARCH_COMPUTE_ALLOWED must be 1" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-}" == "1" ]] || { echo "TASTE_PAPER_RESULTS_ALLOWED must be 1" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-}" == "0" ]] || { echo "TASTE_DATA_REDISTRIBUTION_ALLOWED must be 0" >&2; exit 64; }
[[ "${TASTE_UPSTREAM_LICENSE_STATUS:-}" == "NOT_EXPLICITLY_STATED" ]] || { echo "Taste upstream status must remain NOT_EXPLICITLY_STATED" >&2; exit 64; }
[[ "$PRIMARY_GNN_BACKBONE" == "gine" ]] || { echo "Taste full route freezes GINE" >&2; exit 64; }
[[ "$PRIMARY_SEED" == "7" ]] || { echo "Taste full route freezes seed 7" >&2; exit 64; }
[[ "${CUBLAS_WORKSPACE_CONFIG:-}" == ":4096:8" ]] || { echo "Taste full route requires CUBLAS_WORKSPACE_CONFIG=:4096:8" >&2; exit 64; }
[[ "${PYTHONHASHSEED:-}" == "7" ]] || { echo "Taste full route requires PYTHONHASHSEED=7" >&2; exit 64; }
[[ "${NVIDIA_TF32_OVERRIDE:-}" == "0" ]] || { echo "Taste full route requires NVIDIA_TF32_OVERRIDE=0" >&2; exit 64; }
[[ "${CUDNN_DETERMINISTIC:-}" == "1" ]] || { echo "Taste full route requires CUDNN_DETERMINISTIC=1" >&2; exit 64; }
: "${TASTEMOLNET_POLICY_FILE:?TASTEMOLNET_POLICY_FILE is required}"
: "${TASTEMOLNET_POLICY_SHA256:?TASTEMOLNET_POLICY_SHA256 is required}"
: "${TASTEMOLNET_POLICY_RECEIPT:?TASTEMOLNET_POLICY_RECEIPT is required}"

MIN_PERSISTENT_FREE_GB="${MIN_PERSISTENT_FREE_GB:-20}"
[[ "$MIN_PERSISTENT_FREE_GB" =~ ^[0-9]+$ ]] && (( MIN_PERSISTENT_FREE_GB >= 20 )) \
  || { echo "Taste full route requires MIN_PERSISTENT_FREE_GB>=20" >&2; exit 64; }
RESOURCE_WAIT_STARTED="$(date +%s)"
RESOURCE_WAIT_DEADLINE_SECONDS="${TASTEMOLNET_GPU_WAIT_DEADLINE_SECONDS:-604800}"
RESOURCE_WAIT_POLL_SECONDS="${TASTEMOLNET_GPU_WAIT_POLL_SECONDS:-30}"
RESOURCE_WAIT_DEADLINE_EPOCH="${TASTEMOLNET_RESOURCE_WAIT_DEADLINE_EPOCH:-$((RESOURCE_WAIT_STARTED + RESOURCE_WAIT_DEADLINE_SECONDS))}"
[[ "$RESOURCE_WAIT_DEADLINE_EPOCH" =~ ^[0-9]+$ ]] || { echo "Taste resource deadline must be an epoch integer" >&2; exit 64; }
RESOURCE_WAIT_POLLS=0

TRAIN_SCRIPT="${MOLECULAR_GNN_TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/train_molecular_gnn.py}"
HPC_CONFIG="$PROJECT_ROOT/configs/hpc.yaml"
GNN_CONFIG="$PROJECT_ROOT/configs/gnn/${PRIMARY_GNN_BACKBONE}.yaml"
AUTODL_CONFIG="$PROJECT_ROOT/configs/autodl/tastemolnet_gine_research_v1.yaml"
autodl_require_file "$TRAIN_SCRIPT"
autodl_require_file "$HPC_CONFIG"
autodl_require_file "$GNN_CONFIG"
autodl_require_file "$AUTODL_CONFIG"
autodl_require_file "$TASTEMOLNET_POLICY_FILE"
autodl_require_file "$TASTEMOLNET_POLICY_RECEIPT"
autodl_require_dir "$TASTEMOLNET_PREPARED_ROOT"
autodl_require_dir "$TASTEMOLNET_SPLIT_ROOT"
autodl_require_dir "$TASTEMOLNET_GRAPH_CACHE_ROOT"

while true; do
  AVAILABLE_KB="$(df -Pk "$AUTODL_RUNTIME_ROOT" | awk 'NR == 2 {print $4}')"
  DISK_READY=0
  if [[ "$AVAILABLE_KB" =~ ^[0-9]+$ ]] \
    && (( AVAILABLE_KB >= MIN_PERSISTENT_FREE_GB * 1024 * 1024 )); then
    DISK_READY=1
  fi
  GPU_LINE=""
  GPU_READY=0
  if (( DISK_READY == 1 )); then
    set +e
    GPU_INVENTORY="$(
      "$AUTODL_PYTHON" "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
        --project-root "$PROJECT_ROOT" \
        --data-root "$AUTODL_DATA_ROOT" \
        --max-gpus 4 \
        --gpu-hard-limit 4 \
        --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
        --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
        --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" \
        --format lines \
        --require-idle
    )"
    GPU_RC=$?
    set -e
    if [[ $GPU_RC -ne 0 && $GPU_RC -ne 3 ]]; then
      exit "$GPU_RC"
    fi
    GPU_LINE="$(printf '%s\n' "$GPU_INVENTORY" | awk -F '\t' '$1 == "2" {print; exit}')"
    if [[ -n "$GPU_LINE" ]]; then
      GPU_READY=1
    fi
  fi
  if (( DISK_READY == 1 && GPU_READY == 1 )); then
    break
  fi
  RESOURCE_WAIT_NOW="$(date +%s)"
  if (( RESOURCE_WAIT_NOW >= RESOURCE_WAIT_DEADLINE_EPOCH )); then
    echo "TASTEMOLNET_RESOURCE_WAIT_DEADLINE_EXCEEDED disk_ready=$DISK_READY gpu2_ready=$GPU_READY" >&2
    exit 75
  fi
  RESOURCE_WAIT_POLLS=$((RESOURCE_WAIT_POLLS + 1))
  if (( RESOURCE_WAIT_POLLS <= 20 || RESOURCE_WAIT_POLLS % 120 == 0 )); then
    echo "WAITING_FOR_PHYSICAL_GPU2_AND_DISK elapsed_seconds=$((RESOURCE_WAIT_NOW - RESOURCE_WAIT_STARTED)) deadline_epoch=$RESOURCE_WAIT_DEADLINE_EPOCH disk_ready=$DISK_READY gpu2_ready=$GPU_READY available_kb=$AVAILABLE_KB minimum_gb=$MIN_PERSISTENT_FREE_GB" >&2
  fi
  sleep "$RESOURCE_WAIT_POLL_SECONDS"
done
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
[[ "$GPU_INDEX" == "2" ]] || { echo "Taste route requires physical GPU2" >&2; exit 75; }

OUTPUT_DIR="${TASTEMOLNET_GNN_FULL_OUTPUT:-$(autodl_new_output_dir tastemolnet "$PRIMARY_GNN_BACKBONE" full)}"
TRAINING_STATE_ROOT="${TASTEMOLNET_GNN_TRAINING_STATE_ROOT:-${OUTPUT_DIR}.training_state}"
canonical_path() {
  "$AUTODL_PYTHON" -c 'import os,sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' "$1"
}
paths_overlap() {
  local left right
  left="$(canonical_path "$1")"
  right="$(canonical_path "$2")"
  [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]
}
if paths_overlap "$OUTPUT_DIR" "$TASTEMOLNET_PREPARED_ROOT" \
  || paths_overlap "$OUTPUT_DIR" "$TASTEMOLNET_GRAPH_CACHE_ROOT"; then
  echo "Taste output must be disjoint from prepared/cache roots" >&2
  exit 64
fi
if paths_overlap "$TRAINING_STATE_ROOT" "$TASTEMOLNET_PREPARED_ROOT" \
  || paths_overlap "$TRAINING_STATE_ROOT" "$TASTEMOLNET_GRAPH_CACHE_ROOT" \
  || paths_overlap "$TRAINING_STATE_ROOT" "$OUTPUT_DIR"; then
  echo "Taste training state must be disjoint from output/prepared/cache roots" >&2
  exit 64
fi
if [[ -L "$TRAINING_STATE_ROOT" ]]; then
  echo "Taste GINE training-state root may not be a symlink" >&2
  exit 64
fi
RESUME_ARGS=()
if [[ -e "$TRAINING_STATE_ROOT" ]]; then
  [[ -d "$TRAINING_STATE_ROOT" ]] || { echo "Taste GINE training-state root is not a directory" >&2; exit 64; }
  RESUME_ARGS=(--resume-training)
fi
PUBLISHED_ADOPTION_EXP_ARGS=()
PUBLISHED_ADOPTION_TRAIN_ARGS=()
if [[ -e "${TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT:-}" ]]; then
  [[ -f "$TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT" && ! -L "$TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT" ]] \
    || { echo "Taste published-output adoption receipt is not one physical file" >&2; exit 64; }
  PUBLISHED_ADOPTION_EXP_ARGS=(--resume-published-output-receipt "$TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT")
  PUBLISHED_ADOPTION_TRAIN_ARGS=(--resume-published-output-receipt "$TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT")
fi
INPUT_ARGS=()
INPUT_MANIFEST="$(autodl_find_split_manifest "$TASTEMOLNET_SPLIT_ROOT" || true)"
if [[ -n "$INPUT_MANIFEST" ]]; then
  INPUT_ARGS=(--input-manifest "$INPUT_MANIFEST")
fi

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset tastemolnet \
  --stage TASTEMOLNET_GINE_FULL_RESEARCH_V1 \
  --gpu-index "$GPU_INDEX" \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --heavy \
  --max-gpus 4 \
  --gpu-hard-limit 4 \
  --foreground \
  --config-file "$HPC_CONFIG" \
  --config-file "$GNN_CONFIG" \
  --config-file "$AUTODL_CONFIG" \
  "${INPUT_ARGS[@]}" \
  "${PUBLISHED_ADOPTION_EXP_ARGS[@]}" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file model.pt \
  --required-output-file model_card.json \
  --required-output-file feature_schema.json \
  --required-output-file training_metrics.json \
  --required-output-file test_evaluation_status.json \
  --required-output-file temperature_scaling.json \
  --required-output-file data_use_policy_binding.json \
  --required-output-file graph_cache_usage.json \
  --required-output-file oracle_manifest.json \
  --required-output-file sha256sums.txt \
  --required-log-marker "[TASTE_GINE_THREE_CLASS_PASS]" \
  -- \
  "$AUTODL_PYTHON" "$TRAIN_SCRIPT" \
    --config "$HPC_CONFIG" \
    --config "$GNN_CONFIG" \
    --config "$AUTODL_CONFIG" \
    --dataset tastemolnet \
    --data-dir "$TASTEMOLNET_SPLIT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --profile full \
    --device cuda:0 \
    --backbone "$PRIMARY_GNN_BACKBONE" \
    --seed "$PRIMARY_SEED" \
    --graph-cache-root "$TASTEMOLNET_GRAPH_CACHE_ROOT" \
    --taste-policy-file "$TASTEMOLNET_POLICY_FILE" \
    --taste-policy-sha256 "$TASTEMOLNET_POLICY_SHA256" \
    --taste-policy-receipt "$TASTEMOLNET_POLICY_RECEIPT" \
    --taste-prepared-root "$TASTEMOLNET_PREPARED_ROOT" \
    --training-state-dir "$TRAINING_STATE_ROOT" \
    "${RESUME_ARGS[@]}" \
    "${PUBLISHED_ADOPTION_TRAIN_ARGS[@]}"
