#!/bin/bash
# Explicitly authorized CPU-only continuation of an existing committed epoch.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
cd "${GNN_EXECUTION_WORKTREE:?new immutable resume-driver worktree required}"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
command -v python
python --version
python scripts/hpc/gnn/resume_bace_gnn_cpu.py "$@" &
gnn_child_pid=$!
gnn_signal_forwarded=0
trap 'gnn_signal_forwarded=1; kill -USR1 "$gnn_child_pid" 2>/dev/null || true' USR1 TERM
set +e
wait "$gnn_child_pid"
gnn_exit_code=$?
if [[ "$gnn_signal_forwarded" == 1 && "$gnn_exit_code" -gt 128 ]]; then
  wait "$gnn_child_pid"
  gnn_exit_code=$?
fi
exit "$gnn_exit_code"
