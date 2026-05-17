#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl}
ORACLE_PATH=${ORACLE_PATH:-${TEACHER_PATH}}
DATASET_PATH=${DATASET_PATH:-}
RUN_NAME=${RUN_NAME:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
if [ -n "${RUN_NAME}" ] && [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}
fi
SFT_LORA_PATH=${SFT_LORA_PATH:-}
MAX_STEPS=${MAX_STEPS:-}
SAVE_STEPS=${SAVE_STEPS:-}
LOGGING_STEPS=${LOGGING_STEPS:-}
PPO_LOOP=${PPO_LOOP:-decoded_chem}

echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
python - <<'PY'
import torch
print("python version:", __import__("sys").version.replace("\n", " "))
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY
echo "DATASET_PATH=${DATASET_PATH:-<unset>}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH:-<unset>}"
echo "TEACHER_PATH=${TEACHER_PATH:-<unset>}"
echo "RUN_NAME=${RUN_NAME:-<unset>}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-<unset>}"
echo "PPO_LEARNING_RATE=${PPO_LEARNING_RATE:-<unset>}"
echo "PPO_CLIP_RANGE=${PPO_CLIP_RANGE:-<unset>}"
echo "PPO_EPOCHS=${PPO_EPOCHS:-<unset>}"
echo "MAX_GRAD_NORM=${MAX_GRAD_NORM:-<unset>}"
echo "TARGET_KL=${TARGET_KL:-<unset>}"
echo "HARD_KL=${HARD_KL:-<unset>}"
echo "ENABLE_ADAPTIVE_KL=${ENABLE_ADAPTIVE_KL:-<unset>}"
echo "KL_PENALTY_INIT=${KL_PENALTY_INIT:-<unset>}"
echo "KL_PENALTY_MULTIPLIER=${KL_PENALTY_MULTIPLIER:-<unset>}"
echo "REWARD_CLIP_MIN=${REWARD_CLIP_MIN:-<unset>}"
echo "REWARD_CLIP_MAX=${REWARD_CLIP_MAX:-<unset>}"
echo "NORMALIZE_REWARD=${NORMALIZE_REWARD:-<unset>}"
echo "NORMALIZE_ADVANTAGE=${NORMALIZE_ADVANTAGE:-<unset>}"
echo "ENABLE_TEACHER_CONFIDENCE_GATE=${ENABLE_TEACHER_CONFIDENCE_GATE:-<unset>}"
echo "MIN_TEACHER_P_BEFORE=${MIN_TEACHER_P_BEFORE:-<unset>}"
echo "LOW_CONF_CF_WEIGHT=${LOW_CONF_CF_WEIGHT:-<unset>}"
echo "ENABLE_STABLE_EARLY_STOP=${ENABLE_STABLE_EARLY_STOP:-<unset>}"
echo "SAVE_BEST_CHECKPOINT=${SAVE_BEST_CHECKPOINT:-<unset>}"
echo "EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-<unset>}"
echo "VAL_DATASET_PATH=${VAL_DATASET_PATH:-<unset>}"
echo "EVAL_NUM_SAMPLES=${EVAL_NUM_SAMPLES:-<unset>}"

cmd=(
  python
  scripts/train_ppo_stable.py
  --config
  configs/hpc.yaml
  --teacher-path
  "${TEACHER_PATH}"
  --oracle-path
  "${ORACLE_PATH}"
  --ppo-loop
  "${PPO_LOOP}"
)

if [ -n "${OUTPUT_DIR}" ]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi
if [ -n "${DATASET_PATH}" ]; then
  cmd+=(--dataset-path "${DATASET_PATH}")
fi
if [ -n "${SFT_LORA_PATH}" ]; then
  cmd+=(--sft-lora-path "${SFT_LORA_PATH}")
fi
if [ -n "${MAX_STEPS}" ]; then
  cmd+=(--max-steps "${MAX_STEPS}")
fi
if [ -n "${SAVE_STEPS}" ]; then
  cmd+=(--save-steps "${SAVE_STEPS}")
fi
if [ -n "${LOGGING_STEPS}" ]; then
  cmd+=(--logging-steps "${LOGGING_STEPS}")
fi

"${cmd[@]}"
