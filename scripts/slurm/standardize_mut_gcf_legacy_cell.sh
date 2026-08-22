#!/usr/bin/env bash
# Static parity wrapper. The current four-by-four campaign runs this on AutoDL.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${HELDOUT_ROOT:?Set HELDOUT_ROOT to the passing Mut GCF held-out container}"
: "${FROZEN_ROOT:?Set FROZEN_ROOT to the frozen candidate/threshold package}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a fresh destination}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/standardize_mut_gcf_legacy_cell.py \
  --config configs/hpc.yaml \
  --heldout-root "$HELDOUT_ROOT" \
  --frozen-root "$FROZEN_ROOT" \
  --output-dir "$OUTPUT_DIR"
