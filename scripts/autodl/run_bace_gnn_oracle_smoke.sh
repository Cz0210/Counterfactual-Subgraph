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
autodl_require_file "$BACE_SPLIT_ROOT/calibration.csv"

SOURCE_CHECKPOINT="$(
  autodl_passed_stage_output B4_GNN_CALIBRATED \
    --required-output-file model.pt \
    --required-output-file temperature_scaling.json \
    --required-output-file b4_calibration.json \
    --required-output-file sha256sums.txt
)"
GPU_LINE="$(autodl_select_one_gpu)" || exit $?
IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
if [[ -z "${GPU_INDEX:-}" || -z "${GPU_UUID:-}" ]]; then
  echo "WAITING_FOR_IDLE_GPU" >&2
  exit 75
fi
OUTPUT_DIR="${BACE_GNN_ORACLE_SMOKE_OUTPUT:-$(autodl_new_output_dir bace "$PRIMARY_GNN_BACKBONE" oracle-smoke)}"

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset bace \
  --stage B5_ORACLE_SMOKE \
  --gpu-index "$GPU_INDEX" \
  --gpu-uuid "$GPU_UUID" \
  --gpu-required \
  --config-file "$HPC_CONFIG" \
  --config-file "$GNN_CONFIG" \
  --config-file "$DATASET_CONFIG" \
  --config-file "$AUTODL_CONFIG" \
  --input-manifest "$SOURCE_CHECKPOINT/sha256sums.txt" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file predictions.csv \
  --required-output-file metrics.json \
  --required-output-file oracle_smoke.json \
  --required-output-file deletion_records.jsonl \
  --required-log-marker "[BACE_GNN_ORACLE_SMOKE_PASS]" \
  -- \
  "$AUTODL_PYTHON" "$STAGE_SCRIPT" \
    --config "$HPC_CONFIG" \
    --config "$GNN_CONFIG" \
    --config "$DATASET_CONFIG" \
    --config "$AUTODL_CONFIG" \
    oracle-smoke \
    --checkpoint-dir "$SOURCE_CHECKPOINT" \
    --calibration-csv "$BACE_SPLIT_ROOT/calibration.csv" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --batch-size 32 \
    --source-count 16 \
    --max-deletions-per-parent 4
