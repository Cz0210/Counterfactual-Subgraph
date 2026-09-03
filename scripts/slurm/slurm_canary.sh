#!/usr/bin/env bash
# Paired thin wrapper. CPU-only/no-GRES intentionally overrides the repository
# GPU baseline because exact gSpan mining performs no neural inference.
#SBATCH --job-name=t8-gspan-canary
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
exec bash scripts/hpc/t8/slurm_canary.sh "$@"
