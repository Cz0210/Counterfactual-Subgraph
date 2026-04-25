#!/bin/bash
# 50-step decoded-chem PPO diagnostic run with parent projection enabled.
#
# Usage on HPC:
#   cd /share/home/u20526/czx/counterfactual-subgraph
#   sbatch scripts/slurm/train_decoded_chem_ppo_projection_diag50.sh

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ppo_proj_diag50

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
RUN_NAME=decoded_chem_diag50_projection_v1
MAX_STEPS=50
LOGGING_STEPS=1
SFT_LORA_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_checkpoints/checkpoint-300
GEN_MAX_NEW_TOKENS=48
GEN_TEMPERATURE=0.6
GEN_TOP_P=0.85
GEN_DO_SAMPLE=true
FULL_PARENT_PENALTY=-6.0
EMPTY_RESIDUAL_PENALTY=-4.0
REWARD_MAX_FRAGMENT_CHARS=80
ENABLE_PARENT_PROJECTION=true
PROJECTION_MIN_SCORE=0.35
PROJECTION_MAX_CANDIDATES=128
PROJECTION_MIN_ATOMS=3
PROJECTION_MAX_ATOM_RATIO=0.70
PROJECTION_PENALTY=0.5
PROJECTION_ENABLE_KHOP3=false
PROJECTION_MCS_TIMEOUT=1
TEACHER_PATH=${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl
OUTPUT_DIR=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}

export HF_HOME=/share/home/u20526/.cache/huggingface
export HF_MODULES_CACHE=/share/home/u20526/.cache/huggingface/modules
export TRANSFORMERS_CACHE=/share/home/u20526/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/share/home/u20526/.cache/huggingface/hub

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
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
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ENABLE_PARENT_PROJECTION=${ENABLE_PARENT_PROJECTION}"
echo "PROJECTION_MIN_SCORE=${PROJECTION_MIN_SCORE}"
echo "PROJECTION_MAX_CANDIDATES=${PROJECTION_MAX_CANDIDATES}"
echo "PROJECTION_MIN_ATOMS=${PROJECTION_MIN_ATOMS}"
echo "PROJECTION_MAX_ATOM_RATIO=${PROJECTION_MAX_ATOM_RATIO}"
echo "PROJECTION_PENALTY=${PROJECTION_PENALTY}"
echo "PROJECTION_ENABLE_KHOP3=${PROJECTION_ENABLE_KHOP3}"
echo "PROJECTION_MCS_TIMEOUT=${PROJECTION_MCS_TIMEOUT}"
echo "====================="

cd /share/home/u20526/czx/counterfactual-subgraph
mkdir -p logs

export PYTHONPATH=$PWD

echo "===== REPO CHECK ====="
echo "pwd after cd: $(pwd)"
echo "git root: $(git rev-parse --show-toplevel)"
echo "git commit: $(git rev-parse HEAD)"
echo "PYTHONPATH(after cd)=${PYTHONPATH}"
echo "======================"

echo "===== TEACHER CHECK ====="
echo "TEACHER_PATH=${TEACHER_PATH}"
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] Teacher file not found: ${TEACHER_PATH}"
  exit 1
fi
echo "[OK] Teacher file found."
echo "========================="

echo "===== RUNNING DECODED CHEM PPO PROJECTION DIAG50 ====="

python scripts/train_rl.py \
  --config configs/hpc.yaml \
  --max-steps "${MAX_STEPS}" \
  --logging-steps "${LOGGING_STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  --sft-lora-path "${SFT_LORA_PATH}" \
  --diagnose-reward-flow \
  --ppo-loop decoded_chem \
  --require-chemistry-reward-path \
  --decoded-chem-smoke-test \
  --teacher-path "${TEACHER_PATH}" \
  --require-teacher-sem \
  --teacher-sem-scale 1.0 \
  --teacher-sem-missing-penalty -5.0 \
  --teacher-cf-flip-bonus 1.0 \
  --gen-max-new-tokens "${GEN_MAX_NEW_TOKENS}" \
  --gen-temperature "${GEN_TEMPERATURE}" \
  --gen-top-p "${GEN_TOP_P}" \
  --gen-do-sample \
  --full-parent-penalty "${FULL_PARENT_PENALTY}" \
  --empty-residual-penalty "${EMPTY_RESIDUAL_PENALTY}" \
  --reward-max-fragment-chars "${REWARD_MAX_FRAGMENT_CHARS}" \
  --enable-parent-projection \
  --projection-min-score "${PROJECTION_MIN_SCORE}" \
  --projection-max-candidates "${PROJECTION_MAX_CANDIDATES}" \
  --projection-min-atoms "${PROJECTION_MIN_ATOMS}" \
  --projection-max-atom-ratio "${PROJECTION_MAX_ATOM_RATIO}" \
  --projection-penalty "${PROJECTION_PENALTY}" \
  --no-projection-enable-khop3 \
  --projection-mcs-timeout "${PROJECTION_MCS_TIMEOUT}"

echo "===== DONE ====="
