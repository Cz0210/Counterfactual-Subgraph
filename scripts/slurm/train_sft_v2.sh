#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=train_sft_v2

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
BASE_MODEL=${BASE_MODEL:-${PROJECT_DIR}/pretrained_models/ChemLLM-7B-Chat}
TRAIN_FILE=${TRAIN_FILE:-${PROJECT_DIR}/data/sft_train_v2.jsonl}
VAL_FILE=${VAL_FILE:-${PROJECT_DIR}/data/sft_val_v2.jsonl}
RUN_NAME=${RUN_NAME:-sft_v2_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hpc/sft_checkpoints/${RUN_NAME}}
MAX_STEPS=${MAX_STEPS:-500}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-100}
EVAL_STEPS=${EVAL_STEPS:-100}
LEARNING_RATE=${LEARNING_RATE:-2e-4}
SEED=${SEED:-7}
REPORT_TO=${REPORT_TO:-none}

export HF_HOME=/share/home/u20526/.cache/huggingface
export HF_MODULES_CACHE=/share/home/u20526/.cache/huggingface/modules
export TRANSFORMERS_CACHE=/share/home/u20526/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/share/home/u20526/.cache/huggingface/hub

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "python path: $(which python)"
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "HF_HOME=${HF_HOME}"
echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "REPORT_TO=${REPORT_TO}"
echo "====================="

cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONPATH=$PWD

echo "===== TRAIN SFT V2 ====="
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "VAL_FILE=${VAL_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "SAVE_STEPS=${SAVE_STEPS}"
echo "EVAL_STEPS=${EVAL_STEPS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "SEED=${SEED}"
echo "REPORT_TO=${REPORT_TO}"
echo "========================"

if [ ! -f "${TRAIN_FILE}" ]; then
  echo "[ERROR] TRAIN_FILE not found: ${TRAIN_FILE}"
  exit 1
fi
if [ ! -f "${VAL_FILE}" ]; then
  echo "[ERROR] VAL_FILE not found: ${VAL_FILE}"
  exit 1
fi

python scripts/train_sft.py \
  --config configs/hpc.yaml \
  --model-path "${BASE_MODEL}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --logging-steps "${LOGGING_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --eval-steps "${EVAL_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --report-to "${REPORT_TO}" \
  --seed "${SEED}" \
  "$@"
