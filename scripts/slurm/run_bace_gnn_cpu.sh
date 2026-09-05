#!/bin/bash
# User-authorized CPU-only exception to the generic A800 submission template.
#SBATCH --job-name=bace-gnn-cpu
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd "${GNN_EXECUTION_WORKTREE:-/share/home/u20526/czx/counterfactual-subgraph}"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
command -v python
python --version
python -c 'import torch; print({"torch": torch.__version__, "cuda_visible": torch.cuda.is_available(), "CPU_only": True})'
# --config is the frozen bundle architecture YAML, not generic GPU hpc.yaml.
python scripts/hpc/gnn/run_bace_gnn_cpu.py "$@" &
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
