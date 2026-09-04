#!/bin/bash
#SBATCH --job-name=mut-route-b-owner
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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

: "${MUT_ROUTE_B_SPEC:?MUT_ROUTE_B_SPEC is required}"
python scripts/autodl/run_mut_route_b_owner_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --task-spec "$MUT_ROUTE_B_SPEC"
