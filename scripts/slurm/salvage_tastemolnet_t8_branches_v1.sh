#!/usr/bin/env bash
#SBATCH --job-name=taste_t8_salvage
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

: "${T8_SALVAGE_ATTEMPT_ID:?set a fresh UUIDv4}"
: "${T8_SOURCE_ATTEMPT_ID:?set the completed source UUIDv4}"
: "${T8_TARGET_0_ROOT:?set target-0 root}"
: "${T8_TARGET_2_ROOT:?set target-2 root}"
: "${TASTEMOLNET_T3_OUTPUT:?set T3 root}"
: "${TASTEMOLNET_T4_OUTPUT:?set T4 root}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set train CSV}"
: "${TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT:?set official source root}"
: "${T8_SALVAGE_STATE_ROOT:?set fresh state root}"
: "${T8_SALVAGE_OUTPUT_ROOT:?set fresh output root}"
: "${T8_SALVAGE_RERUN_REQUEST:?set persistent rerun receipt path}"

python scripts/autodl/salvage_tastemolnet_t8_branches_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --attempt-id "$T8_SALVAGE_ATTEMPT_ID" \
  --source-attempt-id "$T8_SOURCE_ATTEMPT_ID" \
  --target-0-root "$T8_TARGET_0_ROOT" \
  --target-2-root "$T8_TARGET_2_ROOT" \
  --t3-output "$TASTEMOLNET_T3_OUTPUT" \
  --t4-output "$TASTEMOLNET_T4_OUTPUT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
  --state-root "$T8_SALVAGE_STATE_ROOT" \
  --output-root "$T8_SALVAGE_OUTPUT_ROOT" \
  --rerun-request "$T8_SALVAGE_RERUN_REQUEST" \
  --device cuda:0
