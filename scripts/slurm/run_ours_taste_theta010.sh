#!/bin/bash
# Explicit user override: HPC CPU-only. Never request a GPU.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd "${OURS_TASTE_CODE:-/share/home/u20526/czx/counterfactual-subgraph}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "CPU theta selection python=$(command -v python)"
python -V
# Raw-matrix operations do not use inference fallback.
python -I -B scripts/run_ours_taste_theta010.py --config configs/hpc.yaml "$@"
