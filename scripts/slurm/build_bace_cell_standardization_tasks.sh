#!/bin/bash
# Static CLI-parity wrapper for all four frozen BACE cells, including native
# GlobalGCE. The AutoDL continuation does not submit this file.
#SBATCH --job-name=bace_cell_tasks
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/build_bace_cell_standardization_tasks.py \
  --config configs/hpc.yaml \
  --controller-id "${CONTROLLER_ID:?Set CONTROLLER_ID}" \
  --output-root "${OUTPUT_ROOT:?Set OUTPUT_ROOT}" \
  --gnn-checkpoint "${GNN_CHECKPOINT:?Set GNN_CHECKPOINT}" \
  --fragment-output "${FRAGMENT_OUTPUT:?Set FRAGMENT_OUTPUT}"
