#!/usr/bin/env bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Task-specific override: receipt-only adoption must not reserve a GPU.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python)"
python --version
echo "CPU-only metadata adoption; CUDA hidden"
python scripts/autodl/adopt_mut_historical_independent.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
