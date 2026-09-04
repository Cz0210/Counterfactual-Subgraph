#!/usr/bin/env bash
# CPU-only registry reconciliation.  Override INPUT/OUTPUT for AutoDL control.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${INPUT:?absolute owner registry input required}"
: "${OUTPUT:?absolute owner registry output required}"
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
python scripts/autodl/reconcile_final16_owner_registry_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --input "$INPUT" \
  --output "$OUTPUT"
