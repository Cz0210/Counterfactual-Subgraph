#!/bin/bash
#SBATCH --job-name=four_by_four_manifest
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static CLI parity only; production manifest composition runs on AutoDL CPU.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${CONTROLLER_ID:?CONTROLLER_ID is required}"
: "${TASK_FRAGMENT:?TASK_FRAGMENT is required}"
: "${OUTPUT_MANIFEST:?OUTPUT_MANIFEST is required}"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/build_four_by_four_manifest.py \
  --config configs/hpc.yaml \
  --controller-id "$CONTROLLER_ID" \
  --task-fragment "$TASK_FRAGMENT" \
  --output "$OUTPUT_MANIFEST"
