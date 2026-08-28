#!/usr/bin/env bash
#SBATCH --job-name=taste-t4-worker-v2
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
echo "TasteMolNet T4 managed-v2 science is AutoDL-only; this Slurm wrapper is static CLI parity." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python scripts/autodl/tastemolnet_t4_oracle_smoke_worker_v2.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t3-root /absolute/seed7/calibrated-reviewed \
  --graph-cache-root /absolute/private/graph-cache \
  --gpu-uuid GPU-REVIEWED
