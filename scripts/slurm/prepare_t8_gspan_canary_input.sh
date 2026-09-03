#!/usr/bin/env bash
#SBATCH --job-name=t8-gspan-input
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()}")'

python scripts/autodl/prepare_t8_gspan_canary_input.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  "$@"
