#!/bin/bash
# Explicit CPU-only rendering exception; no molecular inference or GPU needed.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -eo pipefail
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
echo "Python: $(command -v python)"
python --version
# This display-only CLI consumes explicit source CSVs; project config is not applicable.
exec python -I -B scripts/experiments/replot_bace_gin.py "$@"
