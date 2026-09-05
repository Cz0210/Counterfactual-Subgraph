#!/usr/bin/env bash
# CPU-only portable import; AutoDL may call the Python entrypoint directly.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable importer worktree required}"
: "${GNN_VERIFIED_ARCHIVE:?verified archive required}"
: "${GNN_VERIFIED_ARCHIVE_SHA:?transport SHA required}"
: "${GNN_IMPORT_ROOT:?fresh isolated import root required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "Python: $(command -v python)"
python --version
echo "Portable byte/binding verification only; no classifier or OT execution"
exec nice -n 10 python scripts/hpc/gnn/import_bace_gnn_verified.py \
  --config configs/hpc.yaml --archive-path "$GNN_VERIFIED_ARCHIVE" \
  --expected-sha256 "$GNN_VERIFIED_ARCHIVE_SHA" --output-root "$GNN_IMPORT_ROOT"
