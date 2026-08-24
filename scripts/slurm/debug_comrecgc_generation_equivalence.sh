#!/usr/bin/env bash
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

: "${LEGACY_ROOT:?Set LEGACY_ROOT}"
: "${OPTIMIZED_ROOT:?Set OPTIMIZED_ROOT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

python scripts/debug_comrecgc_generation_equivalence.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --legacy-root "$LEGACY_ROOT" \
  --optimized-root "$OPTIMIZED_ROOT" \
  --output-dir "$OUTPUT_DIR"
