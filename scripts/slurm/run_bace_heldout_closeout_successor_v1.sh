#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=bace-heldout-closeout

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export RUN_GNN_ABLATION=0

echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("cuda_devices=", torch.cuda.device_count())'

: "${BACE_HELDOUT_CONTROLLER_ID:?set BACE_HELDOUT_CONTROLLER_ID}"
: "${BACE_HELDOUT_CONTROL_DIR:?set BACE_HELDOUT_CONTROL_DIR}"
: "${BACE_HELDOUT_OUTPUT_ROOT:?set BACE_HELDOUT_OUTPUT_ROOT}"
: "${BACE_HELDOUT_SOURCE_ROOT:?set BACE_HELDOUT_SOURCE_ROOT}"
: "${BACE_HELDOUT_SELECTION_RECEIPT:?set BACE_HELDOUT_SELECTION_RECEIPT}"
: "${BACE_HELDOUT_SELECTION_RECEIPT_SHA256:?set BACE_HELDOUT_SELECTION_RECEIPT_SHA256}"
: "${BACE_GNN_CHECKPOINT:?set BACE_GNN_CHECKPOINT}"
: "${BACE_TEST_SPLIT:?set BACE_TEST_SPLIT}"
: "${MOLCLR_ROOT:?set MOLCLR_ROOT}"
: "${MOLCLR_CHECKPOINT:?set MOLCLR_CHECKPOINT}"
: "${MATRIX_AUTHORITY_STATE:?set MATRIX_AUTHORITY_STATE}"
: "${MATRIX_AUTHORITY_LOCK:?set MATRIX_AUTHORITY_LOCK}"
: "${AUTODL_RUNTIME_ROOT:?set AUTODL_RUNTIME_ROOT}"

python scripts/autodl/run_bace_heldout_closeout_successor_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  run \
  --project-root "$PWD" \
  --python "$(which python)" \
  --runtime-root "$AUTODL_RUNTIME_ROOT" \
  --controller-id "$BACE_HELDOUT_CONTROLLER_ID" \
  --control-dir "$BACE_HELDOUT_CONTROL_DIR" \
  --output-root "$BACE_HELDOUT_OUTPUT_ROOT" \
  --source-root "$BACE_HELDOUT_SOURCE_ROOT" \
  --selection-adoption-receipt "$BACE_HELDOUT_SELECTION_RECEIPT" \
  --expected-selection-receipt-sha256 "$BACE_HELDOUT_SELECTION_RECEIPT_SHA256" \
  --gnn-checkpoint "$BACE_GNN_CHECKPOINT" \
  --test-split "$BACE_TEST_SPLIT" \
  --molclr-root "$MOLCLR_ROOT" \
  --molclr-checkpoint "$MOLCLR_CHECKPOINT" \
  --matrix-authority-state "$MATRIX_AUTHORITY_STATE" \
  --matrix-authority-lock "$MATRIX_AUTHORITY_LOCK" \
  --gpu-index "${BACE_HELDOUT_GPU_INDEX:-0}" \
  --poll-seconds "${SCHEDULER_POLL_SECONDS:-30}"
