#!/usr/bin/env bash
# CPU-only afterok packaging; never trains or writes the main matrix.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
# Site bashrc/Conda scripts read optional unset variables during initialization.
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?exact immutable execution worktree required}"
: "${GNN_EXECUTION_COMMIT:?exact Git commit required}"
: "${GNN_EVALUATION_ROOT:?sealed evaluation root required}"
: "${GNN_PACKAGE_ROOT:?fresh package output required}"
: "${GNN_ENVIRONMENT_MANIFEST:?environment manifest required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "Python: $(command -v python)"
python --version
echo "CPU-only sealed package"
if [[ ! -f "$GNN_EVALUATION_ROOT/GNN_CORE_SEED7_PASS" ]]; then
  echo "WAITING_EVALUATION_CORE_PASS: inspect CPU admission or missing classifier output"
  exit 0
fi
exec nice -n 10 python scripts/hpc/gnn/package_bace_gnn_seed7.py \
  --config configs/hpc.yaml --evaluation-root "$GNN_EVALUATION_ROOT" \
  --output-root "$GNN_PACKAGE_ROOT" --environment-manifest "$GNN_ENVIRONMENT_MANIFEST" \
  --execution-commit "$GNN_EXECUTION_COMMIT"
