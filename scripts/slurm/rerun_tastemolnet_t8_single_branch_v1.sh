#!/usr/bin/env bash
#SBATCH --job-name=taste_t8_one_branch
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("cuda_devices=", torch.cuda.device_count())'

: "${T8_BRANCH_ATTEMPT_ID:?set a fresh UUIDv4}"
: "${T8_SOURCE_ATTEMPT_ID:?set the rejected source UUIDv4}"
: "${T8_RERUN_TARGET:?set exactly 0 or 2}"
: "${TASTEMOLNET_T3_OUTPUT:?set T3 root}"
: "${TASTEMOLNET_T4_OUTPUT:?set T4 root}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set train CSV}"
: "${TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT:?set official source root}"
: "${T8_BRANCH_STATE_ROOT:?set fresh one-branch state root}"
: "${T8_BRANCH_GSPAN_SCRATCH_ROOT:?set fresh scratch root}"

python scripts/autodl/rerun_tastemolnet_t8_single_branch_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --attempt-id "$T8_BRANCH_ATTEMPT_ID" \
  --source-attempt-id "$T8_SOURCE_ATTEMPT_ID" \
  --target "$T8_RERUN_TARGET" \
  --t3-output "$TASTEMOLNET_T3_OUTPUT" \
  --t4-output "$TASTEMOLNET_T4_OUTPUT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
  --state-root "$T8_BRANCH_STATE_ROOT" \
  --gspan-scratch-root "$T8_BRANCH_GSPAN_SCRATCH_ROOT" \
  --device cuda:0
