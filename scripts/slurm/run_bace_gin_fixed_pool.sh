#!/bin/bash
# BACE-specific CPU evaluation exception to the repository GPU training default.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Set immutable worktree with sbatch --chdir; no login-node inference, no GPU.
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
test -f scripts/experiments/run_bace_gin_fixed_pool.py
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/run_bace_gin_fixed_pool.py --config configs/hpc.yaml "$@"
