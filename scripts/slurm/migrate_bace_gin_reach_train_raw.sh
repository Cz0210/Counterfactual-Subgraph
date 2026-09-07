#!/bin/bash
# Read-only science-source migration; CPU exception to GPU training default.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
test -f scripts/experiments/migrate_bace_gin_reach_train_raw.py
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/migrate_bace_gin_reach_train_raw.py --config configs/hpc.yaml "$@"
