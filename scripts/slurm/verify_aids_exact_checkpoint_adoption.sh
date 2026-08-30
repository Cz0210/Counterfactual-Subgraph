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

: "${AIDS_EXACT_CONTROLLER_MANIFEST:?absolute controller manifest is required}"
: "${AIDS_EXACT_EXPECTED_PROGRESS_ROWS:?frozen checkpoint progress is required}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/verify_aids_exact_checkpoint_adoption.py \
  --config configs/hpc.yaml \
  --manifest "$AIDS_EXACT_CONTROLLER_MANIFEST" \
  --expected-progress-rows "$AIDS_EXACT_EXPECTED_PROGRESS_ROWS"
