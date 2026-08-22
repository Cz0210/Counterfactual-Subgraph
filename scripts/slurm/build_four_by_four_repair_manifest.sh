#!/bin/bash
# Static CLI-parity wrapper required by repository policy.  The active repair
# campaign is AutoDL-only: do not submit this file for the current experiment.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=build-4x4-repair

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

: "${FOUR_BY_FOUR_REPAIR_SPEC:?Set an absolute configured repair spec}"
: "${FOUR_BY_FOUR_REPAIR_MANIFEST:?Set a fresh absolute manifest path}"

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

python scripts/autodl/build_four_by_four_repair_manifest.py \
  --config configs/hpc.yaml \
  build \
  --spec "$FOUR_BY_FOUR_REPAIR_SPEC" \
  --output "$FOUR_BY_FOUR_REPAIR_MANIFEST"
