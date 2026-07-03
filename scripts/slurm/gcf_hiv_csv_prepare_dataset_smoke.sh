#!/bin/bash
#SBATCH -J gcf_hiv_prep_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

CSV_PATH=${CSV_PATH:-/share/home/u20526/czx/counterfactual-subgraph/data/raw/AIDS/HIV.csv}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-HIV_active}
OUT_DIR=${OUT_DIR:-outputs/hpc/gcfexplainer_hiv_csv/dataset}
MAX_ROWS=${MAX_ROWS:-1000}
CF_MODE=${CF_MODE:-strict_flip}

echo "===== GCF HIV CSV PREPARE DATASET SMOKE ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CF_MODE=${CF_MODE}"
python --version

python scripts/gcf_hiv_csv_prepare_dataset.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --csv-path "${CSV_PATH}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --out-dir "${OUT_DIR}" \
  --max-rows "${MAX_ROWS}" \
  --seed "${SEED:-0}"

echo "===== GCF HIV CSV PREPARE DATASET SMOKE DONE ====="
cat "${OUT_DIR}/dataset_summary.json"

