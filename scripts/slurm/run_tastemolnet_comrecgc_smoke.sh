#!/usr/bin/env bash
#SBATCH --job-name=taste-t9-comrecgc
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

echo "TASTE_T9_AUTODL_ONLY: run the trusted-single-operator managed-v2 GPU1 route; Slurm execution is disabled" >&2
exit 64

# Documentation-only CLI parity (unreachable by design):
# python scripts/run_tastemolnet_comrecgc_smoke.py \
#   --config configs/hpc.yaml --stage T9_COMRECGC_SMOKE \
#   --stage-root "$TASTEMOLNET_T9_STAGE_ROOT" \
#   --output-dir "$TASTEMOLNET_T9_OUTPUT" \
#   --run-id "$TASTEMOLNET_T9_RUN_ID" \
#   --gpu-uuid "$TASTEMOLNET_GPU1_UUID" \
#   --t2-adoption-root "$TASTEMOLNET_T2_ADOPTION_ROOT" \
#   --t2-adoption-gate-sha256 "$TASTEMOLNET_T2_ADOPTION_GATE_SHA256" \
#   --t2-adoption-receipt-sha256 "$TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256" \
#   --t2-source-evidence-sha256 "$TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256" \
#   --t3-output-root "$TASTEMOLNET_T3_OUTPUT_ROOT" \
#   --t4-output-root "$TASTEMOLNET_T4_OUTPUT_ROOT" \
#   --checkpoint-dir "$TASTEMOLNET_T2_BUNDLE" \
#   --train-csv "$TASTEMOLNET_TRAIN_CSV" \
#   --official-root "$COMRECGC_OFFICIAL_ROOT" \
#   --set inference.fallback_to_heuristic=false
