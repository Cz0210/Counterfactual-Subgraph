#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${MATRIX_STATUS:?Set MATRIX_STATUS to the final 16/16 matrix_status.json}"
: "${FINAL_EXPORT_ROOT:?Set FINAL_EXPORT_ROOT to the frozen final export root}"
: "${COMPARISON_AUDIT_ROOT:?Set COMPARISON_AUDIT_ROOT to a fresh output root}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'

python scripts/autodl/comparison_protocol_audit.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --matrix-status "$MATRIX_STATUS" \
  --final-export-root "$FINAL_EXPORT_ROOT" \
  --frozen-contract "$PWD/configs/autodl/final_paper_evaluation_v1.json" \
  --output-root "$COMPARISON_AUDIT_ROOT"
