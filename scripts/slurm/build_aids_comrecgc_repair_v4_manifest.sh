#!/bin/bash
# Static CLI-parity wrapper.  Repair-v4 is AutoDL-only; do not submit.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=build-aids-comrecgc-repair-v4

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

: "${AIDS_COMRECGC_REPAIR_V4_SPEC:?Set an absolute configured repair-v4 spec}"
: "${AIDS_COMRECGC_REPAIR_V4_MANIFEST:?Set the exact fresh manifest path}"

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

python scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py \
  --config configs/hpc.yaml \
  build \
  --spec "$AIDS_COMRECGC_REPAIR_V4_SPEC" \
  --output "$AIDS_COMRECGC_REPAIR_V4_MANIFEST"
