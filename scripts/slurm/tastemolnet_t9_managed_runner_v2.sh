#!/usr/bin/env bash
#SBATCH --job-name=taste-t9-managed-runner
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
echo "TASTE_T9_AUTODL_ONLY: managed runner is static Slurm CLI parity" >&2
exit 64

# Documentation-only parity:
# python scripts/autodl/tastemolnet_t9_managed_runner_v2.py \
#   --config configs/hpc.yaml --set inference.fallback_to_heuristic=false
