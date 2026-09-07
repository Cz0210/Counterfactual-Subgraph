#!/usr/bin/env bash
#SBATCH --job-name=bace-gin-original66-time
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Explicit task-specific CPU authorization overrides the generic GPU template.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u
cd /share/home/u20526/czx/counterfactual-subgraph
if [[ -n "${GIN_FIXED_EXECUTION_WORKTREE:-}" ]]; then
  [[ "$GIN_FIXED_EXECUTION_WORKTREE" == /share/home/u20526/czx/worktrees/* ]]
  cd "$GIN_FIXED_EXECUTION_WORKTREE"
  [[ "$(git rev-parse HEAD)" == "${GIN_FIXED_EXECUTION_COMMIT:?Pinned commit required}" ]]
fi
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
echo "Python: $(command -v python)"
python --version
echo "CPU-only two train parents, original66; no training/fit/selector/test/GPU"
# Frozen classifier evaluation has no heuristic inference fallback.
python -I -B scripts/experiments/time_bace_gin_ours.py --config configs/hpc.yaml "$@"
