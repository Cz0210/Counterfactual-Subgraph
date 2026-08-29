#!/usr/bin/env bash
#SBATCH --job-name=taste-neurosed-fixed
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

: "${GED_LABEL_ROOT:?set GED_LABEL_ROOT}"
: "${NEUROSED_TRAIN_PAIR_ROOT:?set NEUROSED_TRAIN_PAIR_ROOT}"
: "${NEUROSED_VALIDATION_PAIR_ROOT:?set NEUROSED_VALIDATION_PAIR_ROOT}"
: "${NEUROSED_FEATURE_SCHEMA_JSON:?set NEUROSED_FEATURE_SCHEMA_JSON}"
: "${NON_MIP_SELECTION_MANIFEST:?set NON_MIP_SELECTION_MANIFEST}"
: "${NON_MIP_VERIFIER_RECEIPT:?set NON_MIP_VERIFIER_RECEIPT}"
: "${NEUROSED_OUTPUT_ROOT:?set NEUROSED_OUTPUT_ROOT to a fresh directory}"
: "${EXECUTION_GIT_COMMIT:?set EXECUTION_GIT_COMMIT}"
: "${EXECUTION_GIT_TREE:?set EXECUTION_GIT_TREE}"

python -B scripts/tastemolnet/train_fixed_budget_neurosed.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --ged-label-root "$GED_LABEL_ROOT" \
  --train-pair-root "$NEUROSED_TRAIN_PAIR_ROOT" \
  --validation-pair-root "$NEUROSED_VALIDATION_PAIR_ROOT" \
  --feature-schema-json "$NEUROSED_FEATURE_SCHEMA_JSON" \
  --non-mip-selection-manifest "$NON_MIP_SELECTION_MANIFEST" \
  --non-mip-verifier-receipt "$NON_MIP_VERIFIER_RECEIPT" \
  --vendored-gcf-root "$PWD/baselines/gcfexplainer_official" \
  --output-root "$NEUROSED_OUTPUT_ROOT" \
  --execution-git-commit "$EXECUTION_GIT_COMMIT" \
  --execution-git-tree "$EXECUTION_GIT_TREE" \
  --device "${NEUROSED_DEVICE:-cuda:0}" \
  --train-and-verify
