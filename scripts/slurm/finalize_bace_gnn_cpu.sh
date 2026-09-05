#!/usr/bin/env bash
# Scoped publication-only repair, explicitly authorized on intel CPUs, no GPU.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable publication repair worktree required}"
: "${GNN_INPUT_BUNDLE:?original frozen BACE bundle required}"
: "${GNN_MODEL_ROOTS_JSON:?original five-classifier root map required}"
: "${GNN_FINALIZATION_ROOT:?fresh dedicated GNN publication receipt root required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
echo "Python: $(command -v python)"
python --version
echo "Publication only: CPU reload, no trainer, no temperature fitting, no GPU"
exec nice -n 10 python scripts/hpc/gnn/finalize_bace_gnn_cpu.py \
  --config configs/hpc.yaml --bundle-root "$GNN_INPUT_BUNDLE" \
  --model-roots-json "$GNN_MODEL_ROOTS_JSON" --output-root "$GNN_FINALIZATION_ROOT"
