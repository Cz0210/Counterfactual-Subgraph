#!/bin/bash
# HPC full training for the explicit decoded-SMILES chemistry PPO path.
#
# Usage on HPC:
#   cd /share/home/u20526/czx/counterfactual-subgraph
#   mkdir -p logs
#   sbatch scripts/slurm/train_decoded_chem_ppo_full.sh

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=train_decoded_chem_ppo_full

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl}
RUN_NAME=${RUN_NAME:-decoded_chem_ppo_full_$(date +%Y%m%d_%H%M%S)}
MAX_STEPS=${MAX_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}}
SFT_LORA_PATH=${SFT_LORA_PATH:-}
FULL_PARENT_PENALTY=${FULL_PARENT_PENALTY:-}
EMPTY_RESIDUAL_PENALTY=${EMPTY_RESIDUAL_PENALTY:-}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-}
GEN_TEMPERATURE=${GEN_TEMPERATURE:-}
GEN_TOP_P=${GEN_TOP_P:-}
GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-}
ENABLE_PARENT_AWARE_REPAIR=${ENABLE_PARENT_AWARE_REPAIR:-}
REPAIR_MIN_SIMILARITY=${REPAIR_MIN_SIMILARITY:-}
REPAIR_MAX_CANDIDATES=${REPAIR_MAX_CANDIDATES:-}

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
echo "HF_HOME=${HF_HOME}"
echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE}"
echo "PYTHONPATH(before cd)=${PYTHONPATH:-}"
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH:-<unset>}"
echo "FULL_PARENT_PENALTY=${FULL_PARENT_PENALTY:-<unset>}"
echo "EMPTY_RESIDUAL_PENALTY=${EMPTY_RESIDUAL_PENALTY:-<unset>}"
echo "GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-<unset>}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE:-<unset>}"
echo "GEN_TOP_P=${GEN_TOP_P:-<unset>}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE:-<unset>}"
echo "ENABLE_PARENT_AWARE_REPAIR=${ENABLE_PARENT_AWARE_REPAIR:-<unset>}"
echo "REPAIR_MIN_SIMILARITY=${REPAIR_MIN_SIMILARITY:-<unset>}"
echo "REPAIR_MAX_CANDIDATES=${REPAIR_MAX_CANDIDATES:-<unset>}"
echo "====================="

cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONPATH=$PWD

echo "===== REPO CHECK ====="
echo "pwd after cd: $(pwd)"
echo "git root:"
git rev-parse --show-toplevel
echo "git commit:"
git rev-parse HEAD
echo "PYTHONPATH(after cd)=${PYTHONPATH}"
echo "======================"

echo "===== TEACHER CHECK ====="
echo "TEACHER_PATH=${TEACHER_PATH}"
if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] Teacher file not found: ${TEACHER_PATH}"
  echo "[ERROR] Please set TEACHER_PATH=/path/to/teacher classifier bundle"
  exit 1
fi
echo "[OK] Teacher file found."
echo "========================="

echo "===== CHEMLLM SOURCE MODEL PATCH CHECK ====="
CHEMLLM_SOURCE_FILE="${PROJECT_DIR}/pretrained_models/ChemLLM-7B-Chat/modeling_internlm2.py"
SOURCE_CHECK_LOG="logs/${SLURM_JOB_ID:-manual}_decoded_chem_source_check.log"

if [ ! -f "${CHEMLLM_SOURCE_FILE}" ]; then
  echo "[ERROR] Source ChemLLM modeling file not found: ${CHEMLLM_SOURCE_FILE}"
  exit 1
fi

python tools/check_or_patch_chemllm_cache.py \
  --path "${CHEMLLM_SOURCE_FILE}" | tee "${SOURCE_CHECK_LOG}"

if ! grep -q "unguarded_count=0" "${SOURCE_CHECK_LOG}"; then
  echo "[ERROR] Source ChemLLM modeling_internlm2.py still has unguarded past_key_values access."
  exit 1
fi

echo "[OK] Source ChemLLM patch check passed: unguarded_count=0"
echo "==========================================="

echo "===== CHEMLLM RUNTIME CACHE PRE-CHECK ====="
CHEMLLM_CACHE_FILE="${HF_MODULES_CACHE}/transformers_modules/ChemLLM_hyphen_7B_hyphen_Chat/modeling_internlm2.py"
CACHE_PRE_CHECK_LOG="logs/${SLURM_JOB_ID:-manual}_decoded_chem_cache_pre_check.log"

if [ -f "${CHEMLLM_CACHE_FILE}" ]; then
  python tools/check_or_patch_chemllm_cache.py \
    --path "${CHEMLLM_CACHE_FILE}" | tee "${CACHE_PRE_CHECK_LOG}"

  if ! grep -q "unguarded_count=0" "${CACHE_PRE_CHECK_LOG}"; then
    echo "[ERROR] Existing ChemLLM runtime cache still has unguarded past_key_values access."
    exit 1
  fi

  echo "[OK] Existing ChemLLM runtime cache patch check passed: unguarded_count=0"
else
  echo "[WARN] ChemLLM runtime cache file does not exist yet."
  echo "[WARN] The model loader should regenerate it from the patched source model file."
fi

echo "=========================================="

echo "===== RUNTIME PYTHON CHECK ====="
python - <<'PY'
import os
import sys

print("sys.executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HF_MODULES_CACHE:", os.environ.get("HF_MODULES_CACHE"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
PY
echo "==============================="

echo "===== RUNNING DECODED CHEM PPO FULL TRAINING ====="

cmd=(
  python
  scripts/train_rl.py
  --config
  configs/hpc.yaml
  --max-steps
  "${MAX_STEPS}"
  --logging-steps
  "${LOGGING_STEPS}"
  --output-dir
  "${OUTPUT_DIR}"
  --diagnose-reward-flow
  --ppo-loop
  decoded_chem
  --require-chemistry-reward-path
  --teacher-path
  "${TEACHER_PATH}"
  --require-teacher-sem
  --teacher-sem-scale
  1.0
  --teacher-sem-missing-penalty
  -5.0
  --teacher-cf-flip-bonus
  1.0
)

if [ -n "${SFT_LORA_PATH}" ]; then
  cmd+=(--sft-lora-path "${SFT_LORA_PATH}")
fi
if [ -n "${FULL_PARENT_PENALTY}" ]; then
  cmd+=(--full-parent-penalty "${FULL_PARENT_PENALTY}")
fi
if [ -n "${EMPTY_RESIDUAL_PENALTY}" ]; then
  cmd+=(--empty-residual-penalty "${EMPTY_RESIDUAL_PENALTY}")
fi
if [ -n "${GEN_MAX_NEW_TOKENS}" ]; then
  cmd+=(--gen-max-new-tokens "${GEN_MAX_NEW_TOKENS}")
fi
if [ -n "${GEN_TEMPERATURE}" ]; then
  cmd+=(--gen-temperature "${GEN_TEMPERATURE}")
fi
if [ -n "${GEN_TOP_P}" ]; then
  cmd+=(--gen-top-p "${GEN_TOP_P}")
fi
if [ -n "${GEN_DO_SAMPLE}" ]; then
  case "${GEN_DO_SAMPLE,,}" in
    1|true|yes|on)
      cmd+=(--gen-do-sample)
      ;;
    0|false|no|off)
      cmd+=(--no-gen-do-sample)
      ;;
    *)
      echo "[ERROR] GEN_DO_SAMPLE must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
if [ -n "${ENABLE_PARENT_AWARE_REPAIR}" ]; then
  case "${ENABLE_PARENT_AWARE_REPAIR,,}" in
    1|true|yes|on)
      cmd+=(--enable-parent-aware-repair)
      ;;
    0|false|no|off)
      cmd+=(--no-enable-parent-aware-repair)
      ;;
    *)
      echo "[ERROR] ENABLE_PARENT_AWARE_REPAIR must be one of: true/false/1/0/yes/no/on/off"
      exit 1
      ;;
  esac
fi
if [ -n "${REPAIR_MIN_SIMILARITY}" ]; then
  cmd+=(--repair-min-similarity "${REPAIR_MIN_SIMILARITY}")
fi
if [ -n "${REPAIR_MAX_CANDIDATES}" ]; then
  cmd+=(--repair-max-candidates "${REPAIR_MAX_CANDIDATES}")
fi
cmd+=("$@")

set +e
"${cmd[@]}"
TRAIN_EXIT_CODE=$?
set -e

echo "===== DECODED CHEM PPO FULL TRAINING EXIT CODE: ${TRAIN_EXIT_CODE} ====="

echo "===== CHEMLLM RUNTIME CACHE POST-CHECK ====="
CACHE_POST_CHECK_LOG="logs/${SLURM_JOB_ID:-manual}_decoded_chem_cache_post_check.log"

if [ -f "${CHEMLLM_CACHE_FILE}" ]; then
  python tools/check_or_patch_chemllm_cache.py \
    --path "${CHEMLLM_CACHE_FILE}" | tee "${CACHE_POST_CHECK_LOG}"

  if ! grep -q "unguarded_count=0" "${CACHE_POST_CHECK_LOG}"; then
    echo "[ERROR] Regenerated ChemLLM runtime cache still has unguarded past_key_values access."
    exit 1
  fi

  echo "[OK] Runtime cache post-check passed: unguarded_count=0"
else
  echo "[WARN] Runtime cache file still does not exist after full training."
fi

echo "=========================================="

if [ "${TRAIN_EXIT_CODE}" -ne 0 ]; then
  echo "[ERROR] Decoded chemistry PPO full training failed with exit code ${TRAIN_EXIT_CODE}."
  echo "[ERROR] Please inspect this job log and logs/${SLURM_JOB_ID:-manual}_decoded_chem_cache_post_check.log if it exists."
  exit "${TRAIN_EXIT_CODE}"
fi

echo "===== DONE ====="
