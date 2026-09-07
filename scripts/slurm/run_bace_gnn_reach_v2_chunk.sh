#!/usr/bin/env bash
#SBATCH --job-name=bace-gnn-reach-pair
#SBATCH --partition=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Submit only through the existing <=2-heavy-job campaign; no GPU request.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
if [[ -n "${REACH_GNN_EXECUTION_WORKTREE:-}" ]]; then
  [[ "$REACH_GNN_EXECUTION_WORKTREE" == /share/home/u20526/czx/worktrees/* ]]
  cd "$REACH_GNN_EXECUTION_WORKTREE"
  [[ "$(git rev-parse HEAD)" == "${REACH_GNN_EXECUTION_COMMIT:?Pinned execution commit required}" ]]
fi
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
echo "Python: $(command -v python)"
python --version
echo "CPU-only new-pool evaluation; no training or temperature fitting"
# No heuristic inference fallback exists in the frozen scientific evaluator.
# --merge-calibration-only reuses this CPU entry after all calibration chunks;
# it freezes ten global orders and does not perform test inference or core audit.
# --prepare-raw-reuse-only migrates only accepted split-scoped match costs;
# this is a bounded metadata/chemistry step, not model inference or OT solving.
# --merge-test-only verifies all own-backbone pair/minimum records and produces
# the actual v2 tables; --verify-only and --package-only are distinct successors.
# Test raw-reuse creates the post-freeze dependency receipt without mutating
# the original execution spec. No GPU or generic campaign controller is added.
python -I -B scripts/hpc/gnn/run_bace_gnn_reach_v2_chunk.py --config configs/hpc.yaml "$@"
