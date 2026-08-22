#!/bin/bash
#SBATCH --job-name=four_by_four_ctl
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper only; the active campaign launches persistently on
# AutoDL and does not submit this file to HPC.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${FOUR_BY_FOUR_MANIFEST:?FOUR_BY_FOUR_MANIFEST is required}"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/run_four_by_four_controller.py \
  --config configs/hpc.yaml \
  run --manifest "$FOUR_BY_FOUR_MANIFEST"
