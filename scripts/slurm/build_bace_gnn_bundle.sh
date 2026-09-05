#!/usr/bin/env bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# CPU-only exception explicitly authorized 2026-09-05; no GPU request.
set -euo pipefail
# Site bashrc/Conda scripts read optional unset variables during initialization.
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${GNN_EXECUTION_WORKTREE:?set immutable execution worktree}"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
echo "python=$(command -v python) host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python --version
python scripts/hpc/gnn/build_bace_gnn_bundle.py --config configs/hpc.yaml "$@"
