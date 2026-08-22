#!/bin/bash
# Static CLI-parity wrapper. The AutoDL continuation does not submit this file.
#SBATCH --job-name=bace_cell_standardize
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export TOKENIZERS_PARALLELISM=false

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/standardize_bace_frozen_cell.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --method "${METHOD:?Set METHOD to Ours, GCFExplainer, or ComRecGC}" \
  --source-final-root "${SOURCE_FINAL_ROOT:?Set SOURCE_FINAL_ROOT}" \
  --gnn-checkpoint "${GNN_CHECKPOINT:?Set GNN_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:?Set OUTPUT_DIR}"
