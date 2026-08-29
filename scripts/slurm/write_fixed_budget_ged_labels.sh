#!/usr/bin/env bash
#SBATCH --job-name=taste-ged-labels
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

: "${GEDLIB_BUILD_MANIFEST:?set GEDLIB_BUILD_MANIFEST}"
: "${NON_MIP_SELECTION_MANIFEST:?set NON_MIP_SELECTION_MANIFEST}"
: "${NON_MIP_VERIFIER_RECEIPT:?set NON_MIP_VERIFIER_RECEIPT}"
: "${NEUROSED_TRAIN_PAIR_ROOT:?set NEUROSED_TRAIN_PAIR_ROOT}"
: "${NEUROSED_VALIDATION_PAIR_ROOT:?set NEUROSED_VALIDATION_PAIR_ROOT}"
: "${GED_LABEL_OUTPUT_ROOT:?set GED_LABEL_OUTPUT_ROOT to a fresh directory}"

CACHE_ARGS=()
if [[ -n "${GED_LABEL_CACHE_ROOT:-}" ]]; then
  CACHE_ARGS+=(--cache-root "$GED_LABEL_CACHE_ROOT")
fi

python -B scripts/tastemolnet/write_fixed_budget_ged_labels.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --build-manifest "$GEDLIB_BUILD_MANIFEST" \
  --non-mip-selection-manifest "$NON_MIP_SELECTION_MANIFEST" \
  --non-mip-verifier-receipt "$NON_MIP_VERIFIER_RECEIPT" \
  --train-pair-root "$NEUROSED_TRAIN_PAIR_ROOT" \
  --validation-pair-root "$NEUROSED_VALIDATION_PAIR_ROOT" \
  --output-root "$GED_LABEL_OUTPUT_ROOT" \
  --workers "${GED_LABEL_WORKERS:-1}" \
  --pair-timeout-seconds "${GED_LABEL_PAIR_TIMEOUT_SECONDS:-300}" \
  "${CACHE_ARGS[@]}"
