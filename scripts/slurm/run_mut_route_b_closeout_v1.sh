#!/usr/bin/env bash
# This job writes a typed blocker only; it never launches Route-B generation.
#SBATCH --job-name=mut_route_b_closeout
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
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
exec python scripts/autodl/run_mut_route_b_closeout_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
