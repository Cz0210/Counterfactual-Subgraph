#!/bin/bash
# User-authorized saved-record-only CPU exception; no GPU/model/OT.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
# Submit with --chdir set to the immutable worktree under /share/home/u20526/czx.
test -f scripts/experiments/export_bace_gin_aplus_funnel.py
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/export_bace_gin_aplus_funnel.py --config configs/hpc.yaml "$@"
