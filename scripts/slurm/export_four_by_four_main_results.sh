#!/usr/bin/env bash
# Static CLI-parity wrapper only. The active campaign runs on AutoDL and this
# presentation-only exporter performs no model inference, so the inference
# fallback override is not applicable.
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

: "${MATRIX_STATUS:?set MATRIX_STATUS to the final matrix_status.json}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to one fresh persistent output root}"

exec python scripts/autodl/export_four_by_four_main_results.py export \
  --config configs/hpc.yaml \
  --matrix-status "$MATRIX_STATUS" \
  --output-root "$OUTPUT_ROOT" \
  --project-root "$PWD" \
  --require-complete
