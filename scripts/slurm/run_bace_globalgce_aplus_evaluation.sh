#!/bin/bash
# User-authorized BACE exact CPU exception: native cal/freeze/test, no GPU.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
# --chdir must be the immutable execution worktree under /share/home/u20526/czx.
test -f scripts/experiments/run_bace_globalgce_aplus_evaluation.py
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/run_bace_globalgce_aplus_evaluation.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
