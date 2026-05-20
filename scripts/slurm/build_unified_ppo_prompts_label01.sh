#!/bin/bash
#SBATCH -J build_unified_p01
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118
set -u

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

mkdir -p logs

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
DATASET_DIR=${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset
LABEL1_CSV=${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label1.csv
LABEL0_CSV=${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0.csv
LABEL0_JSON=${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0.summary.json
UNIFIED_CSV=${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_unified_label01.csv
UNIFIED_JSON=${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_unified_label01.summary.json
BALANCE_JSON=${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_unified_label01.balance.json

SOURCE_INPUT_CSV=${SOURCE_INPUT_CSV:-}
FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS:-false}
MAX_PER_LABEL=${MAX_PER_LABEL:-0}
LABEL_COL=${LABEL_COL:-HIV_active}
SMILES_COL=${SMILES_COL:-smiles}

if [ -z "${SOURCE_INPUT_CSV}" ]; then
  for candidate in \
    "${DATASET_DIR}/sft_v3_hiv_train.csv" \
    "${DATASET_DIR}/sft_v3_hiv_train_label01.csv" \
    "${DATASET_DIR}/train.csv"
  do
    if [ -f "${candidate}" ]; then
      SOURCE_INPUT_CSV="${candidate}"
      break
    fi
  done
fi

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "python path: $(which python)"
python --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
echo "DATASET_DIR=${DATASET_DIR}"
echo "SOURCE_INPUT_CSV=${SOURCE_INPUT_CSV}"
echo "LABEL1_CSV=${LABEL1_CSV}"
echo "LABEL0_CSV=${LABEL0_CSV}"
echo "UNIFIED_CSV=${UNIFIED_CSV}"
echo "MAX_PER_LABEL=${MAX_PER_LABEL}"
echo "FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS}"
echo "LABEL_COL=${LABEL_COL}"
echo "SMILES_COL=${SMILES_COL}"
echo "====================="

mkdir -p "${DATASET_DIR}"

if [ ! -f "${LABEL1_CSV}" ]; then
  echo "[ERROR] existing label1 prompt CSV not found: ${LABEL1_CSV}"
  exit 1
fi

if [ -z "${SOURCE_INPUT_CSV}" ] || [ ! -f "${SOURCE_INPUT_CSV}" ]; then
  echo "[ERROR] could not resolve SOURCE_INPUT_CSV. Set SOURCE_INPUT_CSV=/path/to/train.csv"
  exit 1
fi

if [ "${FORCE_REBUILD_PROMPTS}" = "true" ] || [ ! -s "${LABEL0_CSV}" ]; then
  python scripts/make_label_specific_ppo_prompts.py \
    --config configs/hpc.yaml \
    --input-csv "${SOURCE_INPUT_CSV}" \
    --label 0 \
    --out-csv "${LABEL0_CSV}" \
    --out-json "${LABEL0_JSON}" \
    --label-col "${LABEL_COL}" \
    --smiles-col "${SMILES_COL}"
else
  echo "skip label0 build: existing non-empty ${LABEL0_CSV}"
fi

python scripts/build_unified_ppo_prompts.py \
  --config configs/hpc.yaml \
  --label0-csv "${LABEL0_CSV}" \
  --label1-csv "${LABEL1_CSV}" \
  --out-csv "${UNIFIED_CSV}" \
  --out-json "${UNIFIED_JSON}" \
  --balance-labels \
  --seed 13 \
  --max-per-label "${MAX_PER_LABEL}"

python scripts/check_unified_prompt_balance.py \
  --config configs/hpc.yaml \
  --dataset-path "${UNIFIED_CSV}" \
  --out-json "${BALANCE_JSON}" \
  --block-size 50

echo "===== OUTPUT CHECK ====="
ls -lh "${LABEL0_CSV}" "${LABEL0_JSON}" "${UNIFIED_CSV}" "${UNIFIED_JSON}" "${BALANCE_JSON}"
echo "===== BALANCE SUMMARY ====="
cat "${BALANCE_JSON}"
