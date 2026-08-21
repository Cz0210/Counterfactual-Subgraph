#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a frozen GNN bundle}"
VALIDATION_CSV="${VALIDATION_CSV:?Set VALIDATION_CSV to validation.csv or val.csv}"
DEVICE="${DEVICE:-cuda:0}"

echo "python=$(which python)"
python --version
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY

python scripts/calibrate_gnn_classifier.py \
  --config configs/hpc.yaml \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --validation-csv "${VALIDATION_CSV}" \
  --split validation \
  --device "${DEVICE}"
