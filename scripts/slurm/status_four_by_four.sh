#!/bin/bash
#SBATCH --job-name=four_by_four_status
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static CLI parity only; the read-only status command is run directly on
# AutoDL and this wrapper is not submitted by the current campaign.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${CONTROLLER_ID:?CONTROLLER_ID is required}"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/status_four_by_four.py \
  --config configs/hpc.yaml \
  --controller-id "$CONTROLLER_ID" --format json
