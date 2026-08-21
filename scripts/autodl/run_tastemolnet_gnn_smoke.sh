#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

TRAIN_SCRIPT="${MOLECULAR_GNN_TRAIN_SCRIPT:-$PROJECT_ROOT/scripts/train_molecular_gnn.py}"
HPC_CONFIG="$PROJECT_ROOT/configs/hpc.yaml"
GNN_CONFIG="$PROJECT_ROOT/configs/gnn/${PRIMARY_GNN_BACKBONE}.yaml"
AUTODL_CONFIG="$PROJECT_ROOT/configs/autodl/tastemolnet_gine.yaml"
autodl_require_file "$TRAIN_SCRIPT"
autodl_require_file "$HPC_CONFIG"
autodl_require_file "$GNN_CONFIG"
autodl_require_file "$AUTODL_CONFIG"
autodl_require_dir "$TASTEMOLNET_SPLIT_ROOT"

# RUN_TASTEMOLNET=0 still permits this bounded CPU forward/training smoke.  It
# allocates no GPU and cannot start the full TasteMolNet route.
OUTPUT_DIR="${TASTEMOLNET_GNN_SMOKE_OUTPUT:-$(autodl_new_output_dir tastemolnet "$PRIMARY_GNN_BACKBONE" cpu-smoke)}"
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
  --stage GNN_CPU_SMOKE \
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
    --profile smoke \
    --device cpu \
    --backbone "$PRIMARY_GNN_BACKBONE" \
    --seed "$PRIMARY_SEED" \
    --max-epochs 1 \
    --train-limit 32 \
    --validation-limit 16 \
    --test-limit 16
