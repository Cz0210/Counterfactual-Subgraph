#!/usr/bin/env bash
#SBATCH --job-name=wnode_ccrcov
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

# Naming-compatible entrypoint. Runtime configuration and diagnostics live in
# the shared tiny/smoke/calibration/final wrapper requested for this distance.
source ~/.bashrc
conda activate "${CLEAR_CONDA_ENV:-smiles_pip118}"
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH="$PWD"
exec bash scripts/slurm/molclr_node_wasserstein_eval_ccrcov.sh
