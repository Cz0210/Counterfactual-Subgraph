#!/usr/bin/env bash
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicitly authorized CPU-only preflight. New effective ablation configs must
# explicitly request validation temperature fitting; this job never fits it.
set -euo pipefail
# Site bashrc/Conda scripts read optional unset variables during initialization.
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${GNN_EXECUTION_WORKTREE:?}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "python=$(command -v python) host=$(hostname)"
python --version
python -I -B scripts/hpc/gnn/preflight_bace_gnn_cpu.py --config configs/hpc.yaml "$@"
