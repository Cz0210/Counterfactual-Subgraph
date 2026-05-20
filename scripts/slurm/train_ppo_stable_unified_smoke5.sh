#!/bin/bash
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
DATASET_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_unified_label01.csv
SFT_LORA_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500
TEACHER_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/oracle/aids_rf_model.pkl
RUN_NAME=decoded_chem_ppo_stable_unified_smoke5_sftv3_projcf_dist03_projpen1_label01_ckpt500
OUTPUT_DIR=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}

MAX_STEPS=5
SAVE_STEPS=5
LOGGING_STEPS=1
PPO_LOOP=decoded_chem
DIAGNOSE_REWARD_FLOW=true
REQUIRE_CHEMISTRY_REWARD_PATH=true

GEN_TEMPERATURE=0.5
GEN_TOP_P=0.8
GEN_DO_SAMPLE=true

ENABLE_PARENT_PROJECTION=true
ENABLE_PROJECTED_CF_REWARD=true
ENABLE_SUBSTRUCTURE_DISTANCE_REWARD=true
SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT=0.3
PROJECTION_PENALTY=1.0
ENABLE_MINIMAL_SYNTAX_REPAIR=true
ENABLE_COMPONENT_SALVAGE=true

PPO_LEARNING_RATE=1e-6
PPO_CLIP_RANGE=0.05
PPO_EPOCHS=1
MAX_GRAD_NORM=0.5
TARGET_KL=0.30
HARD_KL=0.80
ENABLE_ADAPTIVE_KL=true
KL_PENALTY_INIT=0.05
KL_PENALTY_MULTIPLIER=1.5
REWARD_CLIP_MIN=-5.0
REWARD_CLIP_MAX=5.0
NORMALIZE_REWARD=true
NORMALIZE_ADVANTAGE=true

ENABLE_TEACHER_CONFIDENCE_GATE=true
MIN_TEACHER_P_BEFORE=0.5
LOW_CONF_CF_WEIGHT=0.3
LOG_UNIFIED_PPO_SAMPLES=true

ENABLE_STABLE_EARLY_STOP=false
SAVE_BEST_CHECKPOINT=false
EVAL_EVERY_STEPS=0
VAL_DATASET_PATH=
EVAL_NUM_SAMPLES=0

export PPO_LEARNING_RATE PPO_CLIP_RANGE PPO_EPOCHS MAX_GRAD_NORM TARGET_KL HARD_KL
export ENABLE_ADAPTIVE_KL KL_PENALTY_INIT KL_PENALTY_MULTIPLIER
export REWARD_CLIP_MIN REWARD_CLIP_MAX NORMALIZE_REWARD NORMALIZE_ADVANTAGE
export ENABLE_TEACHER_CONFIDENCE_GATE MIN_TEACHER_P_BEFORE LOW_CONF_CF_WEIGHT
export ENABLE_STABLE_EARLY_STOP SAVE_BEST_CHECKPOINT EVAL_EVERY_STEPS VAL_DATASET_PATH EVAL_NUM_SAMPLES
export LOG_UNIFIED_PPO_SAMPLES

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
echo "DATASET_PATH=${DATASET_PATH}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_UNIFIED_PPO_SAMPLES=${LOG_UNIFIED_PPO_SAMPLES}"

python scripts/train_ppo_stable.py \
  --config configs/hpc.yaml \
  --dataset-path "${DATASET_PATH}" \
  --sft-lora-path "${SFT_LORA_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --oracle-path "${TEACHER_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" \
  --save-steps "${SAVE_STEPS}" \
  --logging-steps "${LOGGING_STEPS}" \
  --ppo-loop "${PPO_LOOP}" \
  --require-chemistry-reward-path \
  --require-teacher-sem \
  --diagnose-reward-flow \
  --gen-temperature "${GEN_TEMPERATURE}" \
  --gen-top-p "${GEN_TOP_P}" \
  --gen-do-sample \
  --enable-parent-projection \
  --enable-projected-cf-reward \
  --enable-substructure-distance-reward \
  --substructure-distance-reward-weight "${SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT}" \
  --projection-penalty "${PROJECTION_PENALTY}" \
  --enable-minimal-syntax-repair \
  --enable-component-salvage \
  --no-only-positive \
  --log-unified-ppo-samples
