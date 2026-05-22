#!/bin/bash
#SBATCH -J abl_l1_sft_gen
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
ROOT=${PROJECT_DIR}/outputs/hpc/ablations/label1_sft_ppo

DATASET_PATH=${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
BASE_MODEL_PATH=${PROJECT_DIR}/pretrained_models/ChemLLM-7B-Chat
SFT_LORA_PATH=${PROJECT_DIR}/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500
TEACHER_PATH=${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl

OUT_DIR=${ROOT}/sft_only_n4
POOL=${OUT_DIR}/candidate_pool.jsonl
OUT_SUMMARY=${OUT_DIR}/generation_summary.json
AUDIT_DIR=${ROOT}/audits/sft_only_n4

NUM_RETURN_SEQUENCES=4
GEN_TEMPERATURE=0.5
GEN_TOP_P=0.8
GEN_DO_SAMPLE=true
MAX_NEW_TOKENS=96
FORCE_REGEN=${FORCE_REGEN:-false}

mkdir -p "${OUT_DIR}" "${AUDIT_DIR}"

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
echo "DATASET_PATH=${DATASET_PATH}"
echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "OUT_DIR=${OUT_DIR}"
echo "POOL=${POOL}"
echo "OUT_SUMMARY=${OUT_SUMMARY}"
echo "AUDIT_DIR=${AUDIT_DIR}"
echo "NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE}"
echo "GEN_TOP_P=${GEN_TOP_P}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "FORCE_REGEN=${FORCE_REGEN}"
echo "====================="

for p in "${DATASET_PATH}" "${BASE_MODEL_PATH}" "${SFT_LORA_PATH}" "${TEACHER_PATH}"; do
  if [ ! -e "${p}" ]; then
    echo "[ERROR] missing path: ${p}"
    exit 1
  fi
done

for p in scripts/generate_full_candidate_pool.py scripts/audit_candidate_pool.py scripts/export_candidate_pool_audit_artifacts.py; do
  if [ ! -f "${p}" ]; then
    echo "[ERROR] missing script: ${p}"
    exit 1
  fi
done

POOL_NONEMPTY=false
if [ -s "${POOL}" ]; then
  POOL_LINES=$(wc -l < "${POOL}")
  if [ "${POOL_LINES}" -gt 0 ]; then
    POOL_NONEMPTY=true
  fi
fi

if [ "${FORCE_REGEN}" = "true" ] || [ "${POOL_NONEMPTY}" != "true" ]; then
  echo "===== GENERATING ABLATION POOL: B SFT-only ====="
  python scripts/generate_full_candidate_pool.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --dataset-path "${DATASET_PATH}" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --sft-lora-path "${SFT_LORA_PATH}" \
    --teacher-path "${TEACHER_PATH}" \
    --out-jsonl "${POOL}" \
    --out-summary-json "${OUT_SUMMARY}" \
    --label-col label \
    --smiles-col smiles \
    --target-label 1 \
    --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
    --generation-temperature "${GEN_TEMPERATURE}" \
    --generation-top-p "${GEN_TOP_P}" \
    --generation-do-sample "${GEN_DO_SAMPLE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size 1 \
    --seed 13 \
    --enable-parent-projection \
    --enable-projected-cf-reward \
    --enable-substructure-distance-reward \
    --substructure-distance-reward-weight 0.3 \
    --projection-penalty 1.0 \
    --enable-minimal-syntax-repair \
    --enable-component-salvage
else
  echo "===== SKIP GENERATION: existing non-empty pool found ====="
fi

echo "===== POOL CHECK ====="
ls -lh "${POOL}"
wc -l "${POOL}"
head -n 2 "${POOL}" || true

echo "===== RUNNING AUDIT ====="
python scripts/audit_candidate_pool.py \
  --config configs/hpc.yaml \
  --pool_jsonl "${POOL}" \
  --out_json "${AUDIT_DIR}/audit_summary.json" \
  --out_txt "${AUDIT_DIR}/audit_report.txt" \
  --group_by_label \
  --sim_sample_size 10000 \
  --topk_show 30

python scripts/export_candidate_pool_audit_artifacts.py \
  --config configs/hpc.yaml \
  --audit-json "${AUDIT_DIR}/audit_summary.json" \
  --out-dir "${AUDIT_DIR}" \
  --topk 30

echo "===== AUDIT DONE ====="
ls -lh "${AUDIT_DIR}"
echo "===== AUDIT REPORT ====="
cat "${AUDIT_DIR}/audit_report.txt"
