#!/bin/bash
# Build label=0 PPO prompt CSV with the same minimal smiles,label schema as label=1.

#SBATCH -J build_l0_prompts
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

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_DIR}"
export PYTHONPATH=$PWD

mkdir -p logs

DATASET_DIR=${DATASET_DIR:-${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset}
SOURCE_PATH=${SOURCE_PATH:-}
TARGET_LABEL=${TARGET_LABEL:-0}
LABEL_COL=${LABEL_COL:-}
SMILES_COL=${SMILES_COL:-}
FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS:-false}
SHUFFLE_SEED=${SHUFFLE_SEED:-13}

OUT_CSV=${OUT_CSV:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0.csv}
OUT_JSON=${OUT_JSON:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0.summary.json}
SHUFFLE_CSV=${SHUFFLE_CSV:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0_shuffle_seed13.csv}
SHUFFLE_JSON=${SHUFFLE_JSON:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label0_shuffle_seed13.summary.json}
LABEL1_CSV=${LABEL1_CSV:-${DATASET_DIR}/sft_v3_hiv_ppo_prompts_train_label1.csv}

case "${FORCE_REBUILD_PROMPTS}" in
  true|TRUE|1|yes|YES) FORCE_REBUILD_PROMPTS=true ;;
  *) FORCE_REBUILD_PROMPTS=false ;;
esac

if [ -z "${SOURCE_PATH}" ]; then
  for candidate in \
    "${DATASET_DIR}/sft_v3_hiv_train.jsonl" \
    "${DATASET_DIR}/sft_v3_hiv_train.csv" \
    "${DATASET_DIR}/train.jsonl" \
    "${DATASET_DIR}/train.csv" \
    "${PROJECT_DIR}/data/raw/AIDS/HIV.csv"
  do
    if [ -f "${candidate}" ]; then
      SOURCE_PATH="${candidate}"
      break
    fi
  done
fi

if [ -z "${LABEL_COL}" ]; then
  case "$(basename "${SOURCE_PATH:-}")" in
    HIV.csv) LABEL_COL=HIV_active ;;
    *) LABEL_COL=label ;;
  esac
fi
if [ -z "${SMILES_COL}" ]; then
  SMILES_COL=smiles
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
echo "SOURCE_PATH=${SOURCE_PATH}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "LABEL_COL=${LABEL_COL}"
echo "SMILES_COL=${SMILES_COL}"
echo "OUT_CSV=${OUT_CSV}"
echo "OUT_JSON=${OUT_JSON}"
echo "SHUFFLE_CSV=${SHUFFLE_CSV}"
echo "SHUFFLE_JSON=${SHUFFLE_JSON}"
echo "LABEL1_CSV=${LABEL1_CSV}"
echo "FORCE_REBUILD_PROMPTS=${FORCE_REBUILD_PROMPTS}"
echo "SHUFFLE_SEED=${SHUFFLE_SEED}"
echo "====================="

echo "[BUILD_LABEL_PROMPTS_CONFIG] target_label=${TARGET_LABEL} source_path=${SOURCE_PATH} label_col=${LABEL_COL} smiles_col=${SMILES_COL} out_csv=${OUT_CSV} shuffle_seed=${SHUFFLE_SEED}"

if [ -z "${SOURCE_PATH}" ] || [ ! -f "${SOURCE_PATH}" ]; then
  echo "[ERROR] source dataset not found. Set SOURCE_PATH to the same source used for label1."
  echo "Tried: ${DATASET_DIR}/sft_v3_hiv_train.jsonl, ${DATASET_DIR}/sft_v3_hiv_train.csv, ${DATASET_DIR}/train.csv, ${PROJECT_DIR}/data/raw/AIDS/HIV.csv"
  exit 1
fi

mkdir -p "${DATASET_DIR}"

if [ "${FORCE_REBUILD_PROMPTS}" = "true" ] || [ ! -s "${OUT_CSV}" ]; then
  python scripts/build_label_ppo_prompt_csv.py \
    --config configs/hpc.yaml \
    --source-path "${SOURCE_PATH}" \
    --out-csv "${OUT_CSV}" \
    --out-json "${OUT_JSON}" \
    --target-label "${TARGET_LABEL}" \
    --label-col "${LABEL_COL}" \
    --smiles-col "${SMILES_COL}"
else
  echo "skip base label prompt build: existing non-empty ${OUT_CSV}"
fi

if [ ! -s "${OUT_CSV}" ]; then
  echo "[ERROR] label prompt CSV is missing or empty after build: ${OUT_CSV}"
  exit 1
fi

if [ "${FORCE_REBUILD_PROMPTS}" = "true" ] || [ ! -s "${SHUFFLE_CSV}" ]; then
  python scripts/make_stratified_ppo_prompts.py \
    --config configs/hpc.yaml \
    --dataset-path "${OUT_CSV}" \
    --out-csv "${SHUFFLE_CSV}" \
    --out-json "${SHUFFLE_JSON}" \
    --seed "${SHUFFLE_SEED}" \
    --smiles-col smiles \
    --label-col label
else
  echo "skip shuffle build: existing non-empty ${SHUFFLE_CSV}"
fi

echo "===== OUTPUT CHECK ====="
for output_path in "${OUT_CSV}" "${OUT_JSON}" "${SHUFFLE_CSV}" "${SHUFFLE_JSON}"; do
  if [ -e "${output_path}" ]; then
    ls -lh "${output_path}"
  else
    echo "[WARN] output missing: ${output_path}"
  fi
done
echo "[BUILD_LABEL_PROMPTS_OUTPUT] out_csv=${OUT_CSV} out_json=${OUT_JSON} shuffle_csv=${SHUFFLE_CSV} shuffle_json=${SHUFFLE_JSON}"
echo "===== LABEL COUNTS ====="
OUT_CSV="${OUT_CSV}" python - <<'PY'
import csv
import os
from collections import Counter

path = os.environ.get("OUT_CSV")
with open(path, "r", encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
counts = Counter(str(row.get("label", "")).strip() for row in rows)
print(f"[BUILD_LABEL_PROMPTS_COUNTS] path={path} num_rows={len(rows)} label_counts={dict(sorted(counts.items()))}")
PY
