#!/usr/bin/env bash
# Paired thin wrapper for the CPU-only afterok storage-safe merge/package job.
#SBATCH --job-name=t8-storage-safe-chain
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
set +u
source ~/.bashrc
set -u
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH="$PWD"
: "${T8_EXECUTION_WORKTREE:?T8_EXECUTION_WORKTREE is required}"
cd "$T8_EXECUTION_WORKTREE"
export PYTHONPATH="$PWD"
exec bash scripts/hpc/t8/slurm_storage_safe_merge_package.sh "$@"
