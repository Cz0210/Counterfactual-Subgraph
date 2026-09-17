#!/bin/bash
# V7: explicit CPU-only independent audit.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${SELECTOR_EXECUTION_ROOT:?}"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "python=$(command -v python)"
python --version
python -I -B scripts/audit_selector_selected_test_v7.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --spec "${SELECTOR_SPEC:?}" --adapter "${SELECTOR_ADAPTER:?}"
