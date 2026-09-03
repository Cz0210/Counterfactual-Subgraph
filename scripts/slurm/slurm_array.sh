#!/usr/bin/env bash
# Paired thin wrapper for the CPU-only exact-mining Slurm array. No GRES is
# requested because the stage is deliberately isolated from neural inference.
#SBATCH --job-name=t8-gspan-full
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-0%1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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
exec bash scripts/hpc/t8/slurm_array.sh "$@"
