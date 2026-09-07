#!/bin/bash
# Existing-record audit only: authorized CPU exception, no CUDA/model/OT work.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
test -f scripts/experiments/audit_bace_gin_fixed_pool.py
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
echo "Python: $(command -v python)"
python --version
echo "CUDA disabled; completed-record audit only"
exec python -I -B scripts/experiments/audit_bace_gin_fixed_pool.py --config configs/hpc.yaml "$@"
