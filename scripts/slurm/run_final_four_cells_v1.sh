#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

: "${FINAL_FOUR_STATE_ROOT:?Set FINAL_FOUR_STATE_ROOT}"
: "${MATRIX_AUTHORITY_ROOT:?Set MATRIX_AUTHORITY_ROOT}"
python scripts/autodl/run_final_four_cells_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --state-root "$FINAL_FOUR_STATE_ROOT" \
  --matrix-authority "$MATRIX_AUTHORITY_ROOT" \
  --poll-seconds "${SCHEDULER_POLL_SECONDS:-60}"
