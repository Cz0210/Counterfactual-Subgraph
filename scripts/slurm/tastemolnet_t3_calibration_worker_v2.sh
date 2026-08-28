#!/usr/bin/env bash
#SBATCH --job-name=taste-t3-worker-v2
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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

: "${TASTE_T2_ADOPTION_ROOT:?required}"
: "${TASTE_T2_ARTIFACT_ROOT:?required}"

python scripts/autodl/tastemolnet_t3_calibration_worker_v2.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t2-receipt-root "$TASTE_T2_ADOPTION_ROOT" \
  --source-bundle "$TASTE_T2_ARTIFACT_ROOT"
