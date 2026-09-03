#!/usr/bin/env bash
# Paired thin wrapper for CPU-only deterministic merge/parity/bundling. No
# GRES is requested because this stage contains no neural inference.
#SBATCH --job-name=t8-gspan-merge
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH="$PWD"
: "${T8_EXECUTION_WORKTREE:?T8_EXECUTION_WORKTREE is required}"
cd "$T8_EXECUTION_WORKTREE"
export PYTHONPATH="$PWD"
exec bash scripts/hpc/t8/slurm_merge.sh "$@"
