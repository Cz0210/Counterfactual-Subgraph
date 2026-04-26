#!/bin/bash
# 50-step decoded-chem PPO diagnostic run with minimal parse repair,
# connected-component salvage, and parent-constrained projection enabled.
#
# Usage on HPC:
#   sbatch scripts/slurm/train_decoded_chem_ppo_parsefix_connectfix_diag50.sh

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=ppo_parsefix_diag50

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

export RUN_NAME=decoded_chem_diag50_parsefix_connectfix_v2
export OUTPUT_DIR=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}
export MAX_STEPS=50
export LOGGING_STEPS=1
export SFT_LORA_PATH=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_checkpoints/checkpoint-300
export PPO_LOOP=decoded_chem
export DIAGNOSE_REWARD_FLOW=true
export REQUIRE_CHEMISTRY_REWARD_PATH=true
export DECODED_CHEM_SMOKE_TEST=true
export TEACHER_PATH=${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl

export GEN_MAX_NEW_TOKENS=40
export GEN_TEMPERATURE=0.5
export GEN_TOP_P=0.8
export GEN_DO_SAMPLE=true
export FULL_PARENT_PENALTY=-6.0
export EMPTY_RESIDUAL_PENALTY=-6.0
export REWARD_MAX_FRAGMENT_CHARS=72

export ENABLE_PARENT_PROJECTION=true
export PROJECTION_MIN_SCORE=0.35
export PROJECTION_MAX_CANDIDATES=128
export PROJECTION_MIN_ATOMS=3
export PROJECTION_MAX_ATOM_RATIO=0.70
export PROJECTION_PENALTY=0.5
export PROJECTION_ENABLE_KHOP3=false
export PROJECTION_MCS_TIMEOUT=1

export ENABLE_MINIMAL_SYNTAX_REPAIR=true
export REPAIR_MAX_EDITS=4
export REPAIR_MIN_ATOMS=3
export REPAIR_ALLOW_PARENTHESES_FIX=true
export REPAIR_ALLOW_RING_FIX=true
export REPAIR_ALLOW_TAIL_TRIM=true
export REPAIR_ALLOW_BALANCED_PREFIX_SALVAGE=true
export REPAIR_PREFER_PREFIX_SALVAGE=true
export REPAIR_MAX_SUFFIX_TRIM=8
export REPAIR_MAX_ADDED_CLOSURES=2

export ENABLE_COMPONENT_SALVAGE=true
export COMPONENT_SALVAGE_METHOD=largest_then_best_parent_match
export COMPONENT_SALVAGE_MIN_ATOMS=3
export MULTI_DUMMY_HARD_FAIL_THRESHOLD=3
export ENABLE_LIGHT_DUMMY_SALVAGE=false
export NEAR_PARENT_HARD_RATIO=0.85
export MIN_RESIDUAL_ATOMS=3
export MIN_RESIDUAL_RATIO=0.10

export HF_HOME=/share/home/u20526/.cache/huggingface
export HF_MODULES_CACHE=/share/home/u20526/.cache/huggingface/modules
export TRANSFORMERS_CACHE=/share/home/u20526/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/share/home/u20526/.cache/huggingface/hub

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd after cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
which python
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "PYTHONPATH(after export): ${PYTHONPATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "SFT_LORA_PATH=${SFT_LORA_PATH}"
echo "GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE}"
echo "GEN_TOP_P=${GEN_TOP_P}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE}"
echo "ENABLE_MINIMAL_SYNTAX_REPAIR=${ENABLE_MINIMAL_SYNTAX_REPAIR}"
echo "ENABLE_COMPONENT_SALVAGE=${ENABLE_COMPONENT_SALVAGE}"
echo "ENABLE_PARENT_PROJECTION=${ENABLE_PARENT_PROJECTION}"
echo "====================="

if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] Teacher file not found: ${TEACHER_PATH}"
  exit 1
fi

echo "===== RUNNING DECODED CHEM PPO PARSEFIX CONNECTFIX DIAG50 ====="
bash scripts/slurm/train_ppo.sh
echo "===== DONE ====="
