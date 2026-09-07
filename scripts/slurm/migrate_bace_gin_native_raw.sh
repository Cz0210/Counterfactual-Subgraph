#!/usr/bin/env bash
#SBATCH --job-name=bace-gin-native-raw
#SBATCH --partition=intel
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Task-specific CPU authorization overrides the generic GPU template.
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
echo "No GPU/model/OT/heuristic fallback; test needs actual hash-bound new GIN freeze"
python -I -B scripts/experiments/migrate_bace_gin_native_raw.py --config configs/hpc.yaml "$@"
