#!/usr/bin/env bash
#SBATCH --job-name=taste-t7-pins
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
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

echo "TASTE_T7_TYPED_RELEASE_AUTODL_ONLY: use the retained AutoDL managed roots" >&2
exit 64

# Documentation-only CLI parity (unreachable by design):
# python scripts/autodl/tastemolnet_t7_typed_release_v1.py \
#   --config configs/hpc.yaml validate \
#   --release-root "$TASTEMOLNET_T7_RELEASE_ROOT"
