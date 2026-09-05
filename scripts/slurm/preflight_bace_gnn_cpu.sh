#!/usr/bin/env bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicitly authorized CPU-only preflight.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd "${GNN_EXECUTION_WORKTREE:?}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "python=$(command -v python) host=$(hostname)"
python --version
python scripts/hpc/gnn/preflight_bace_gnn_cpu.py --config configs/hpc.yaml "$@"
