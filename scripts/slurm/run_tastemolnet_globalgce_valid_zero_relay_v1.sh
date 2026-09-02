#!/bin/bash
#SBATCH --job-name=taste-gce-zero-relay
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00
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
python -I -B scripts/autodl/run_tastemolnet_globalgce_valid_zero_relay_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --source-root "$TASTE_GLOBALGCE_SOURCE_ROOT" \
  --attempt-receipt "$TASTE_GLOBALGCE_ATTEMPT_RECEIPT" \
  --authorization-receipt "$TASTE_GLOBALGCE_ZERO_AUTHORIZATION" \
  --test-csv "$TASTE_GLOBALGCE_TEST_CSV" \
  --threshold-contract "$TASTE_GLOBALGCE_THRESHOLD_CONTRACT" \
  --valid-zero-output-root "$TASTE_GLOBALGCE_ZERO_OUTPUT_ROOT" \
  --control-root "$TASTE_GLOBALGCE_ZERO_RELAY_CONTROL_ROOT" \
  --lease-path "$TASTE_GLOBALGCE_ZERO_RELAY_LEASE" \
  --science-pid "$TASTE_GLOBALGCE_SCIENCE_PID" \
  --science-start-ticks "$TASTE_GLOBALGCE_SCIENCE_START_TICKS" \
  --poll-seconds "${TASTE_GLOBALGCE_ZERO_RELAY_POLL_SECONDS:-30}" \
  --execution-commit "$TASTE_GLOBALGCE_ZERO_EXECUTION_COMMIT"
