#!/bin/bash
#SBATCH --job-name=taste_license
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static CLI parity only; the active campaign runs this CPU audit on AutoDL.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${TASTEMOLNET_PREPARED_ROOT:?TASTEMOLNET_PREPARED_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/audit_tastemolnet_license.py \
  --config configs/hpc.yaml \
  --prepared-root "$TASTEMOLNET_PREPARED_ROOT" \
  --output-dir "$OUTPUT_ROOT"
