#!/bin/bash
# User's explicit Taste-only CPU analysis exception overrides default GPU template.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
source ~/.bashrc
conda activate smiles_pip118
set -euo pipefail
: "${OURS_TASTE_CODE:?immutable execution worktree}"
: "${OURS_TASTE_INPUT:?compact saved matrix root}"
: "${OURS_TASTE_OUTPUT:?fresh analysis root}"
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
cd "$OURS_TASTE_CODE"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "job=$SLURM_JOB_ID CPU-only host=$(hostname) python=$(command -v python)"
python -V
# No inference occurs here; heuristic fallback cannot exist in this saved-record path.
python -I -B scripts/run_ours_taste_focus.py --config configs/hpc.yaml --action bounds-select --source "$OURS_TASTE_INPUT" --output "$OURS_TASTE_OUTPUT"
