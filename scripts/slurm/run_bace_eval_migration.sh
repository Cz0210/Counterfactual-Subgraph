#!/bin/bash
# Explicit user-authorized CPU-only evaluation exception to GPU training defaults.
# --action resume reuses committed parent/timing units; plan verifies source audit schema.
# Optional --raw-reconciliation-root uses fresh LLM-only conflict receipts.
# --action accept-package --acceptance-root audits saved records without science.
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Do not enable nounset before the site's shell bootstrap.
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
test -f scripts/experiments/run_bace_eval_migration.py
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
exec python -I -B scripts/experiments/run_bace_eval_migration.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
