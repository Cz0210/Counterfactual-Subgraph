#!/usr/bin/env bash
#SBATCH --job-name=taste-t3-verifier-v2
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

: "${TASTE_T3_SEALED_ROOT:?required}"
: "${TASTE_T3_FINAL_ROOT:?required}"
: "${TASTE_T2_ADOPTION_ROOT:?required}"
: "${TASTE_T2_ARTIFACT_ROOT:?required}"
: "${TASTE_T3_ATTEMPT_ID:?required}"
: "${TASTE_T3_GENERATION_TOKEN:?required}"
: "${TASTE_CONTROLLER_ID:?required}"
: "${TASTE_EXECUTION_COMMIT:?required}"

python scripts/autodl/tastemolnet_t3_calibration_verifier_v2.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --sealed "$TASTE_T3_SEALED_ROOT" \
  --final-path "$TASTE_T3_FINAL_ROOT" \
  --t2-receipt-root "$TASTE_T2_ADOPTION_ROOT" \
  --source-bundle "$TASTE_T2_ARTIFACT_ROOT" \
  --expected-attempt-id "$TASTE_T3_ATTEMPT_ID" \
  --expected-generation-token "$TASTE_T3_GENERATION_TOKEN" \
  --expected-controller-id "$TASTE_CONTROLLER_ID" \
  --expected-git-commit "$TASTE_EXECUTION_COMMIT"
