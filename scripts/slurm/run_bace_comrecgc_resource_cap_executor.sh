#!/usr/bin/env bash
#SBATCH --job-name=bace-comrec-cap
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${BACE_COMRECGC_EXECUTOR_ARGS:?set BACE_COMRECGC_EXECUTOR_ARGS to a reviewed argument file}"
mapfile -t EXECUTOR_ARGS < "$BACE_COMRECGC_EXECUTOR_ARGS"
python scripts/autodl/run_bace_comrecgc_resource_cap_executor.py \
  --config configs/hpc.yaml \
  run "${EXECUTOR_ARGS[@]}"
