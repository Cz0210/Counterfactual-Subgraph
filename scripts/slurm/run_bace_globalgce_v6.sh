#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${GLOBALGCE_V6_ARGS:?set GLOBALGCE_V6_ARGS to reviewed build-manifest arguments}"
python scripts/autodl/run_bace_globalgce_v6.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  build-manifest ${GLOBALGCE_V6_ARGS}
