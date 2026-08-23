#!/usr/bin/env bash
# Staging-only renderer.  It performs no model inference and never writes paper/.
# The GPU request is retained solely for repository-wide Slurm wrapper parity.
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

: "${MATRIX_STATUS:?set MATRIX_STATUS to the canonical 12/16 matrix_status.json}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh three_datasets_complete_v1 root}"
: "${PAPER_STAGING_ROOT:?set PAPER_STAGING_ROOT to a fresh runtime paper-staging root}"

exec python scripts/autodl/export_three_dataset_main_results.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --matrix-status "$MATRIX_STATUS" \
  --output-root "$OUTPUT_ROOT" \
  --paper-staging-root "$PAPER_STAGING_ROOT" \
  --project-root "$PWD"

