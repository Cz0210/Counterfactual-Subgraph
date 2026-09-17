#!/bin/bash
# V7 explicit CPU-only, saved-data diagnostics.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
python -I -B scripts/run_selector_v7_diagnostics.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false --spec /share/home/u20526/czx/p0-v7-bace-spec-r3.json /share/home/u20526/czx/p0-v7-mut-spec-r2.json
