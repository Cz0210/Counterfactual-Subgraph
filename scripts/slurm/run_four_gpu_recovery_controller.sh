#!/bin/bash
# Slurm pairing contract only: validate the AutoDL manifest, never launch the
# AutoDL controller from an HPC compute node.
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

MANIFEST="${FOUR_GPU_RECOVERY_MANIFEST:-configs/autodl/four_gpu_recovery.template.json}"
echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "AutoDL controller launch is disabled on Slurm; validating manifest only."
python scripts/autodl/run_four_gpu_recovery_controller.py \
  --config configs/hpc.yaml \
  validate \
  --manifest "$MANIFEST"
