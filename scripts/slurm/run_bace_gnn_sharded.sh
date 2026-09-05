#!/usr/bin/env bash
# Explicit BACE CPU exact-evaluation authorization overrides the generic A800 template.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable evaluation worktree required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
echo "CPU-only exact parent partitions; no model training or temperature fit"
exec nice -n 10 python scripts/hpc/gnn/run_bace_gnn_sharded.py --config configs/hpc.yaml "$@"
