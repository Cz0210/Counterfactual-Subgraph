#!/usr/bin/env bash
# CPU-only presentation: task-specific exception to the generic GPU template.
# No inference, training, matrix append, or main-task GPU reservation occurs.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "python=$(command -v python)"
python --version
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES (CPU-only presentation)"
: "${MATRIX_AUTHORITY_STATE:?read-only mirrored unique authority state required}"
: "${OUTPUT_ROOT:?fresh PARTIAL staging output required}"
exec nice -n 10 python scripts/autodl/export_partial_main_results.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --matrix-authority-state "$MATRIX_AUTHORITY_STATE" --output-root "$OUTPUT_ROOT"
