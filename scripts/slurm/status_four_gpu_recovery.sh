#!/bin/bash
# Read-only status pairing for the four-GPU AutoDL controller state mirror.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1

DATA_ROOT="${AUTODL_DATA_ROOT:?Set AUTODL_DATA_ROOT to the mounted persistent state root}"
CONTROLLER_ID="${FOUR_GPU_RECOVERY_CONTROLLER_ID:-autodl-four-gpu-recovery-v1}"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/status_four_gpu_recovery.py \
  --config configs/hpc.yaml \
  --project-root "$PWD" \
  --data-root "$DATA_ROOT" \
  --controller-id "$CONTROLLER_ID" \
  --format table
