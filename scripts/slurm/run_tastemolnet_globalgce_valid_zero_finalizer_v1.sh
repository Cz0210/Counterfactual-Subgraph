#!/bin/bash
#SBATCH --job-name=taste-gce-zero
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
: "${TASTE_GLOBALGCE_SOURCE_ROOT:?set TASTE_GLOBALGCE_SOURCE_ROOT}"
: "${TASTE_GLOBALGCE_ATTEMPT_RECEIPT:?set TASTE_GLOBALGCE_ATTEMPT_RECEIPT}"
: "${TASTE_GLOBALGCE_ZERO_AUTHORIZATION:?set TASTE_GLOBALGCE_ZERO_AUTHORIZATION}"
: "${TASTE_GLOBALGCE_ZERO_OBSERVATION:?set TASTE_GLOBALGCE_ZERO_OBSERVATION}"
: "${TASTE_GLOBALGCE_TEST_CSV:?set TASTE_GLOBALGCE_TEST_CSV}"
: "${TASTE_GLOBALGCE_THRESHOLD_CONTRACT:?set TASTE_GLOBALGCE_THRESHOLD_CONTRACT}"
: "${TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT:?set TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT}"
: "${TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT:?set TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT}"
python scripts/autodl/run_tastemolnet_globalgce_valid_zero_finalizer_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  finalize \
  --source-root "$TASTE_GLOBALGCE_SOURCE_ROOT" \
  --attempt-receipt "$TASTE_GLOBALGCE_ATTEMPT_RECEIPT" \
  --authorization-receipt "$TASTE_GLOBALGCE_ZERO_AUTHORIZATION" \
  --recovery-observation "$TASTE_GLOBALGCE_ZERO_OBSERVATION" \
  --test-csv "$TASTE_GLOBALGCE_TEST_CSV" \
  --threshold-contract "$TASTE_GLOBALGCE_THRESHOLD_CONTRACT" \
  --output-root "$TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT" \
  --execution-commit "$TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT"
