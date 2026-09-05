#!/usr/bin/env bash
# Thin paired entrypoint. Mut causal capture is opt-in via the Python CLI;
# this wrapper does not authorize production Route B or waive 500-step parity.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
exec python scripts/baselines/comrecgc/run_generation.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
