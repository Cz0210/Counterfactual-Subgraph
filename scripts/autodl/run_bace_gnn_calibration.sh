#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

STAGE_SCRIPT="$PROJECT_ROOT/scripts/autodl/bace_gnn_stage.py"
HPC_CONFIG="$PROJECT_ROOT/configs/hpc.yaml"
GNN_CONFIG="$PROJECT_ROOT/configs/gnn/${PRIMARY_GNN_BACKBONE}.yaml"
DATASET_CONFIG="$PROJECT_ROOT/configs/datasets/bace_gnn.yaml"
AUTODL_CONFIG="$PROJECT_ROOT/configs/autodl/bace_gine.yaml"
for required in "$STAGE_SCRIPT" "$HPC_CONFIG" "$GNN_CONFIG" "$DATASET_CONFIG" "$AUTODL_CONFIG"; do
  autodl_require_file "$required"
done
autodl_require_file "$BACE_SPLIT_ROOT/val.csv"

SOURCE_CHECKPOINT="$(
  autodl_passed_stage_output B3_GNN_FULL \
    --required-output-file model.pt \
    --required-output-file split_manifest.json \
    --required-output-file validation_predictions.csv \
    --required-output-file temperature_scaling.json \
    --required-output-file sha256sums.txt
)"
OUTPUT_DIR="${BACE_GNN_CALIBRATED_OUTPUT:-$(autodl_new_output_dir bace "$PRIMARY_GNN_BACKBONE" calibrated)}"

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset bace \
  --stage B4_GNN_CALIBRATED \
  --config-file "$HPC_CONFIG" \
  --config-file "$GNN_CONFIG" \
  --config-file "$DATASET_CONFIG" \
  --config-file "$AUTODL_CONFIG" \
  --input-manifest "$SOURCE_CHECKPOINT/sha256sums.txt" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file model.pt \
  --required-output-file model_card.json \
  --required-output-file split_manifest.json \
  --required-output-file validation_predictions.csv \
  --required-output-file temperature_scaling.json \
  --required-output-file sha256sums.txt \
  --required-output-file b4_calibration.json \
  --required-log-marker "[BACE_GNN_CALIBRATION_PASS]" \
  -- \
  "$AUTODL_PYTHON" "$STAGE_SCRIPT" \
    --config "$HPC_CONFIG" \
    --config "$GNN_CONFIG" \
    --config "$DATASET_CONFIG" \
    --config "$AUTODL_CONFIG" \
    calibrate \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --output-checkpoint "$OUTPUT_DIR" \
    --validation-csv "$BACE_SPLIT_ROOT/val.csv" \
    --max-iter 100
