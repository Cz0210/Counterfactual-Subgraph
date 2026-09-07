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
# This display-only CLI consumes --source-csv/--output/--version-label.
# One parent reducer supplies Figure3 coverage+cost, Figure4 K10/K20 and Table2 K10.
# Config and inference.fallback_to_heuristic are not applicable (no inference).
exec python -I -B scripts/experiments/replot_bace_gin.py "$@"
