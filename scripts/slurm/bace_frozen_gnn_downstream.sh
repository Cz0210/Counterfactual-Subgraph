#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(which python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

ACTION="${ACTION:?Set ACTION to a bace_frozen_gnn_downstream.py subcommand}"
# The AutoDL route is primary.  This paired Slurm wrapper exists only to keep
# the repository entrypoint contract synchronized and is never auto-submitted.
python scripts/autodl/bace_frozen_gnn_downstream.py \
  --config configs/hpc.yaml \
  "$ACTION" \
  "$@"
