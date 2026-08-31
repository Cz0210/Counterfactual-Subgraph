#!/usr/bin/env bash
#SBATCH --job-name=bace-cap-post
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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

: "${BACE_COMRECGC_CAP_FRAGMENT:?set the completed postprocess.tasks.json}"
: "${BACE_COMRECGC_GENERIC_FRAGMENT:?set one fresh generic fragment path}"
: "${BACE_COMRECGC_POSTPROCESS_MANIFEST:?set one fresh controller manifest path}"
: "${BACE_COMRECGC_POSTPROCESS_CONTROLLER_ID:?set one fresh controller ID}"

python scripts/autodl/prepare_bace_comrecgc_resource_cap_postprocess.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --source-fragment "$BACE_COMRECGC_CAP_FRAGMENT" \
  --generic-fragment-output "$BACE_COMRECGC_GENERIC_FRAGMENT" \
  --manifest-output "$BACE_COMRECGC_POSTPROCESS_MANIFEST" \
  --controller-id "$BACE_COMRECGC_POSTPROCESS_CONTROLLER_ID"
