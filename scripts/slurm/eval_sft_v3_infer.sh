#!/bin/bash
# Evaluate a trained SFT v3 checkpoint on fragment-quality metrics.

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=eval_sft_v3_infer

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=${PROJECT_DIR:-/share/home/u20526/czx/counterfactual-subgraph}
BASE_MODEL=${BASE_MODEL:-${PROJECT_DIR}/pretrained_models/ChemLLM-7B-Chat}
LORA_ROOT=${LORA_ROOT:-${PROJECT_DIR}/outputs/hpc/sft_checkpoints}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
EVAL_FILE=${EVAL_FILE:-${PROJECT_DIR}/data/sft_v3_hiv_val.jsonl}
RUN_NAME=${RUN_NAME:-eval_sft_v3_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hpc/eval/${RUN_NAME}}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}

export HF_HOME=/share/home/u20526/.cache/huggingface
export HF_MODULES_CACHE=/share/home/u20526/.cache/huggingface/modules
export TRANSFORMERS_CACHE=/share/home/u20526/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/share/home/u20526/.cache/huggingface/hub

cd "${PROJECT_DIR}"
mkdir -p logs

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
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
echo "BASE_MODEL=${BASE_MODEL}"
echo "LORA_ROOT=${LORA_ROOT}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "EVAL_FILE=${EVAL_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "====================="

export PYTHONPATH=$PWD

if [ ! -f "${EVAL_FILE}" ]; then
  echo "[ERROR] EVAL_FILE not found: ${EVAL_FILE}"
  exit 1
fi

CMD=(
  python scripts/eval_sft_fragment_quality.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --base-model "${BASE_MODEL}"
  --lora-root "${LORA_ROOT}"
  --eval-file "${EVAL_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --max-examples "${MAX_EXAMPLES}"
)

if [ -n "${CHECKPOINT_PATH}" ]; then
  CMD+=(--checkpoint-path "${CHECKPOINT_PATH}")
fi

"${CMD[@]}" "$@"
