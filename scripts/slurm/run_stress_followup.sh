#!/usr/bin/env bash
# Paired thin wrapper for the CPU-only T8 stress continuation. No GRES is
# requested because follow-up decisions, exact merge, and packaging use no GPU.
#SBATCH --job-name=t8-stress-followup
#SBATCH --partition=intel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
set +u
source ~/.bashrc
set -u
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
: "${T8_CONTROLLER_WORKTREE:?T8_CONTROLLER_WORKTREE is required}"
cd "$T8_CONTROLLER_WORKTREE"
export PYTHONPATH="$PWD"
exec bash scripts/hpc/t8/slurm_stress_followup.sh "$@"
