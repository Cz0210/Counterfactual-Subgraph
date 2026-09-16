#!/bin/bash
# Explicit CPU-only plotting, task-specific override of repository GPU default.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
echo "CPU-only renderer python=$(command -v python)"
python -V
# Renderer consumes frozen CSVs, not an inference config or heuristic fallback.
python -I -B scripts/replot_ours_taste_theta010.py "$@"
