#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ "${RUN_GNN_BACKBONE_ABLATION:-0}" != "1" ]]; then
  echo "GNN_BACKBONE_ABLATION_DISABLED: set RUN_GNN_BACKBONE_ABLATION=1 explicitly" >&2
  exit 64
fi

DATASET="${GNN_ABLATION_DATASET:-bace}"
BACKBONE="${GNN_BACKBONE:-gcn}"
PROFILE="${GNN_ABLATION_PROFILE:-full}"
case "$BACKBONE" in gine|gin|gcn|gatv2|gps) ;; *) echo "unsupported backbone: $BACKBONE" >&2; exit 2 ;; esac
case "$PROFILE" in smoke|full) ;; *) echo "unsupported profile: $PROFILE" >&2; exit 2 ;; esac
case "$DATASET" in
  bace) DATA_ROOT="$BACE_SPLIT_ROOT" ;;
  tastemolnet)
    if [[ "$RUN_TASTEMOLNET" != "1" ]]; then
      echo "TASTEMOLNET_HEAVY_RUN_DISABLED" >&2
      exit 64
    fi
    DATA_ROOT="$TASTEMOLNET_SPLIT_ROOT"
    ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

TRAIN_SCRIPT="${MOLECULAR_GNN_TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/train_molecular_gnn.py}"
HPC_CONFIG="$PROJECT_ROOT/configs/hpc.yaml"
GNN_CONFIG="$PROJECT_ROOT/configs/gnn/${BACKBONE}.yaml"
autodl_require_file "$TRAIN_SCRIPT"
autodl_require_file "$GNN_CONFIG"
autodl_require_dir "$DATA_ROOT"
GPU_LINE="$(autodl_select_one_gpu)" || exit $?
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
OUTPUT_DIR="${GNN_ABLATION_OUTPUT:-$(autodl_new_output_dir "$DATASET" "$BACKBONE" "ablation-$PROFILE")}"

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset "${DATASET}_ablation" \
  --stage "GNN_${PROFILE}_${BACKBONE}" \
  --gpu-index "$GPU_INDEX" \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --heavy \
  --config-file "$HPC_CONFIG" \
  --config-file "$GNN_CONFIG" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file model.pt \
  --required-output-file model_card.json \
  --required-output-file training_metrics.json \
  --required-output-file test_evaluation_status.json \
  --required-output-file sha256sums.txt \
  --required-log-marker "[MOLECULAR_GNN_TRAIN_OK]" \
  -- \
  "$AUTODL_PYTHON" "$TRAIN_SCRIPT" \
    --config "$HPC_CONFIG" \
    --config "$GNN_CONFIG" \
    --dataset "$DATASET" \
    --data-dir "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --profile "$PROFILE" \
    --device cuda:0 \
    --backbone "$BACKBONE" \
    --seed "$PRIMARY_SEED"
