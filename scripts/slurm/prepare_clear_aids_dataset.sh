#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=clear_aids_prep

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

AIDS_CSV="${AIDS_CSV:-data/raw/AIDS/HIV.csv}"
SMILES_COLUMN="${SMILES_COLUMN:-smiles}"
LABEL_COLUMN="${LABEL_COLUMN:-HIV_active}"
DATASET="${DATASET:-aids}"
OUT_DIR="${OUT_DIR:-baselines/clear_official/dataset}"
SEED="${SEED:-0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
NUM_SPLITS="${NUM_SPLITS:-10}"
SUMMARY_PATH="${SUMMARY_PATH:-outputs/hpc/baselines/clear/${DATASET}/dataset/clear_${DATASET}_dataset_summary.json}"
MAX_NUM_NODES="${MAX_NUM_NODES:-100}"
X_DIM="${X_DIM:-11}"

mkdir -p logs outputs/hpc/baselines/clear/"${DATASET}"/dataset "${OUT_DIR}"

echo "===== CLEAR AIDS DATASET PREP ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-unknown}"
echo "which python: $(which python)"
echo "python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "AIDS_CSV=${AIDS_CSV}"
echo "SMILES_COLUMN=${SMILES_COLUMN}"
echo "LABEL_COLUMN=${LABEL_COLUMN}"
echo "DATASET=${DATASET}"
echo "OUT_DIR=${OUT_DIR}"
echo "SEED=${SEED}"
echo "TRAIN_RATIO=${TRAIN_RATIO}"
echo "VAL_RATIO=${VAL_RATIO}"
echo "TEST_RATIO=${TEST_RATIO}"
echo "NUM_SPLITS=${NUM_SPLITS}"
echo "MAX_NUM_NODES=${MAX_NUM_NODES}"
echo "X_DIM=${X_DIM}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
python - <<'PY'
import importlib.util
try:
    import rdkit
    print("rdkit importable: True")
except Exception as exc:
    print("rdkit import failed:", repr(exc))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
except Exception as exc:
    print("torch diagnostics failed:", repr(exc))
print("torch_geometric importable:", importlib.util.find_spec("torch_geometric") is not None)
PY
echo "==================================="

python scripts/baselines/clear/prepare_clear_aids_dataset.py \
  --config configs/hpc.yaml \
  --aids-csv "${AIDS_CSV}" \
  --smiles-column "${SMILES_COLUMN}" \
  --label-column "${LABEL_COLUMN}" \
  --out-dir "${OUT_DIR}" \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --num-splits "${NUM_SPLITS}" \
  --max-num-nodes "${MAX_NUM_NODES}" \
  --x-dim "${X_DIM}" \
  --summary-path "${SUMMARY_PATH}"
