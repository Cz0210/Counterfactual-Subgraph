#!/usr/bin/env bash
#SBATCH --job-name=freeze_recourse
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Dataset-specific CPU publication override; no inference or GPU allocation.
set -eo pipefail
source /share/home/u20526/anaconda3/etc/profile.d/conda.sh
conda activate smiles_pip118
set -u
: "${AIDS_WORKTREE:?immutable worktree required}"
cd "$AIDS_WORKTREE"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "python=$(command -v python)"
python --version
exec nice -n 10 python -B scripts/baselines/comrecgc/freeze_recovery_result.py --config configs/hpc.yaml "$@"
