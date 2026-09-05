#!/usr/bin/env bash
# Submission-only CPU metadata action; no inference on the login node.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118
set -u
: "${GNN_EXECUTION_WORKTREE:?immutable code required}"
cd "$GNN_EXECUTION_WORKTREE"
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=""
echo "Python: $(command -v python)"
python --version
exec python scripts/hpc/gnn/submit_bace_gnn_sharded.py --config configs/hpc.yaml "$@"
