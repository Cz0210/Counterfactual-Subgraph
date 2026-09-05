#!/usr/bin/env bash
# Read-only CPU metadata status; task-specific exception to the GPU template.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable worktree required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "Python: $(command -v python)"
python --version
# Metadata-only existing CLI has no --config option and never runs inference.
exec python scripts/hpc/gnn/status_bace_gnn_seed7.py "$@"
