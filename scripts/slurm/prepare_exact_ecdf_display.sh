#!/bin/bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# CPU-only frozen-artifact postprocessing: no GPU is needed or authorized here.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "Python: $(command -v python)"
python --version
echo 'CPU-only ECDF export; CUDA is not initialized.'
# No inference or heuristic fallback exists in this read-only numeric export.
python -B scripts/paper/prepare_exact_ecdf_display.py --config configs/hpc.yaml "$@"
