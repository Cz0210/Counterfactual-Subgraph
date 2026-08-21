#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ "$RUN_TASTEMOLNET" != "1" ]]; then
  echo "TASTEMOLNET_HEAVY_RUN_DISABLED: RUN_TASTEMOLNET=$RUN_TASTEMOLNET" >&2
  echo "[TASTEMOLNET_FOUNDATION_READY_NOT_RUN]" >&2
  exit 64
fi

TRAIN_SCRIPT="${MOLECULAR_GNN_TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/train_molecular_gnn.py}"
HPC_CONFIG="$PROJECT_ROOT/configs/hpc.yaml"
GNN_CONFIG="$PROJECT_ROOT/configs/gnn/${PRIMARY_GNN_BACKBONE}.yaml"
AUTODL_CONFIG="$PROJECT_ROOT/configs/autodl/tastemolnet_gine.yaml"
autodl_require_file "$TRAIN_SCRIPT"
autodl_require_file "$HPC_CONFIG"
autodl_require_file "$GNN_CONFIG"
autodl_require_file "$AUTODL_CONFIG"
autodl_require_dir "$TASTEMOLNET_SPLIT_ROOT"

GPU_LINE="$(autodl_select_one_gpu)" || exit $?
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
OUTPUT_DIR="${TASTEMOLNET_GNN_FULL_OUTPUT:-$(autodl_new_output_dir tastemolnet "$PRIMARY_GNN_BACKBONE" full)}"
INPUT_ARGS=()
INPUT_MANIFEST="$(autodl_find_split_manifest "$TASTEMOLNET_SPLIT_ROOT" || true)"
if [[ -n "$INPUT_MANIFEST" ]]; then
  INPUT_ARGS=(--input-manifest "$INPUT_MANIFEST")
fi

exec python "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset tastemolnet \
  --stage GNN_FULL \
  --gpu-index "$GPU_INDEX" \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --heavy \
  --config-file "$HPC_CONFIG" \
  --config-file "$GNN_CONFIG" \
  --config-file "$AUTODL_CONFIG" \
  "${INPUT_ARGS[@]}" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file model.pt \
  --required-output-file model_card.json \
  --required-output-file feature_schema.json \
  --required-output-file training_metrics.json \
  --required-output-file temperature_scaling.json \
  --required-output-file sha256sums.txt \
  --required-log-marker "[MOLECULAR_GNN_TRAIN_OK]" \
  -- \
  python "$TRAIN_SCRIPT" \
    --config "$HPC_CONFIG" \
    --config "$GNN_CONFIG" \
    --config "$AUTODL_CONFIG" \
    --dataset tastemolnet \
    --data-dir "$TASTEMOLNET_SPLIT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --profile full \
    --device cuda:0 \
    --backbone "$PRIMARY_GNN_BACKBONE" \
    --seed "$PRIMARY_SEED"
