#!/bin/bash
# HPC smoke test for the explicit decoded-SMILES chemistry PPO path.
#
# Usage on HPC:
#   cd /share/home/u20526/czx/counterfactual-subgraph
#   mkdir -p logs
#   sbatch scripts/slurm/debug_decoded_chem_ppo_smoketest.sh

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=/share/home/u20526/czx/counterfactual-subgraph/logs/%j.out
#SBATCH --error=/share/home/u20526/czx/counterfactual-subgraph/logs/%j.err
#SBATCH --job-name=debug_decoded_chem_ppo

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph

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
echo "====================="

cd "${PROJECT_DIR}"
mkdir -p logs

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "===== REPO CHECK ====="
echo "pwd after cd: $(pwd)"
echo "git root:"
git rev-parse --show-toplevel
echo "git commit:"
git rev-parse HEAD
echo "PYTHONPATH(after cd)=${PYTHONPATH}"
echo "======================"

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

echo "===== RUNNING DECODED CHEM PPO SMOKE TEST ====="

set +e
python scripts/train_rl.py \
  --config configs/hpc.yaml \
  --max-steps 2 \
  --logging-steps 1 \
  --max-prompt-examples 8 \
  --output-dir outputs/hpc/rl_checkpoints/debug_decoded_chem_ppo \
  --diagnose-reward-flow \
  --ppo-loop decoded_chem \
  --require-chemistry-reward-path \
  --decoded-chem-smoke-test \
  "$@"
TRAIN_EXIT_CODE=$?
set -e

echo "===== DECODED CHEM PPO SMOKE TEST EXIT CODE: ${TRAIN_EXIT_CODE} ====="

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
  echo "[WARN] Runtime cache file still does not exist after smoke test."
fi

echo "=========================================="

if [ "${TRAIN_EXIT_CODE}" -ne 0 ]; then
  echo "[ERROR] Decoded chemistry PPO smoke test failed with exit code ${TRAIN_EXIT_CODE}."
  echo "[ERROR] Please inspect this job log and logs/${SLURM_JOB_ID:-manual}_decoded_chem_cache_post_check.log if it exists."
  exit "${TRAIN_EXIT_CODE}"
fi

echo "===== DONE ====="
