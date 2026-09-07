#!/usr/bin/env bash
#SBATCH --job-name=bace-reach-raw-test
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicit task exception: index completed raw costs; no GPU or model inference.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
[[ "${REACH_GNN_EXECUTION_WORKTREE:?}" == /share/home/u20526/czx/worktrees/* ]]
cd "$REACH_GNN_EXECUTION_WORKTREE"
[[ "$(git rev-parse HEAD)" == "${REACH_GNN_EXECUTION_COMMIT:?}" ]]
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "Python: $(command -v python)"
python --version
echo "Metadata-only, no heuristic/model inference, no OT computation"
python -I -B scripts/hpc/gnn/prepare_bace_reach_raw_test_index.py --config configs/hpc.yaml "$@"
