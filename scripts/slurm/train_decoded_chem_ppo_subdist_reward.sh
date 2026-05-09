#!/bin/bash
# Decoded-chem PPO training with nearest-parent-subgraph distance reward enabled.
#
# Usage on HPC:
#   sbatch scripts/slurm/train_decoded_chem_ppo_subdist_reward.sh

#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=train_decoded_chem_ppo_subdist

set -eo pipefail

source ~/.bashrc
conda activate smiles_pip118

cd /share/home/u20526/czx/counterfactual-subgraph

mkdir -p logs
export PYTHONPATH=$PWD

export ENABLE_SUBSTRUCTURE_DISTANCE_REWARD=1
export SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT=0.5
export SUBSTRUCTURE_DISTANCE_MIN_ATOM_RATIO=0.10
export SUBSTRUCTURE_DISTANCE_MAX_ATOM_RATIO=0.65
export SUBSTRUCTURE_DISTANCE_TOPK=20
export SUBSTRUCTURE_DISTANCE_MCS_TIMEOUT=1
export SUBSTRUCTURE_DISTANCE_SIM_THRESHOLD=0.0
export DISABLE_PROJECTED_CF_REWARD=1

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
TEACHER_PATH=${TEACHER_PATH:-${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl}
RUN_NAME=${RUN_NAME:-decoded_chem_ppo_subdist_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/outputs/hpc/rl_checkpoints/${RUN_NAME}}
MAX_STEPS=${MAX_STEPS:-1000}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SFT_LORA_PATH=${SFT_LORA_PATH:-}

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "date: $(date)"
echo "pwd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
echo "python path: $(which python)"
python -V
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "PYTHONPATH=${PYTHONPATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LOGGING_STEPS=${LOGGING_STEPS}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "ENABLE_SUBSTRUCTURE_DISTANCE_REWARD=${ENABLE_SUBSTRUCTURE_DISTANCE_REWARD}"
echo "SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT=${SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT}"
echo "SUBSTRUCTURE_DISTANCE_MIN_ATOM_RATIO=${SUBSTRUCTURE_DISTANCE_MIN_ATOM_RATIO}"
echo "SUBSTRUCTURE_DISTANCE_MAX_ATOM_RATIO=${SUBSTRUCTURE_DISTANCE_MAX_ATOM_RATIO}"
echo "SUBSTRUCTURE_DISTANCE_TOPK=${SUBSTRUCTURE_DISTANCE_TOPK}"
echo "SUBSTRUCTURE_DISTANCE_MCS_TIMEOUT=${SUBSTRUCTURE_DISTANCE_MCS_TIMEOUT}"
echo "SUBSTRUCTURE_DISTANCE_SIM_THRESHOLD=${SUBSTRUCTURE_DISTANCE_SIM_THRESHOLD}"
echo "DISABLE_PROJECTED_CF_REWARD=${DISABLE_PROJECTED_CF_REWARD}"
echo "====================="

cmd=(
  python
  scripts/train_rl.py
  --config
  configs/hpc.yaml
  --ppo-loop
  decoded_chem
  --require-chemistry-reward-path
  --diagnose-reward-flow
  --output-dir
  "${OUTPUT_DIR}"
  --max-steps
  "${MAX_STEPS}"
  --logging-steps
  "${LOGGING_STEPS}"
  --teacher-path
  "${TEACHER_PATH}"
  --require-teacher-sem
  --teacher-sem-scale
  1.0
  --teacher-sem-missing-penalty
  -5.0
  --teacher-cf-flip-bonus
  1.0
  --enable-substructure-distance-reward
  --substructure-distance-reward-weight
  "${SUBSTRUCTURE_DISTANCE_REWARD_WEIGHT}"
  --substructure-distance-min-atom-ratio
  "${SUBSTRUCTURE_DISTANCE_MIN_ATOM_RATIO}"
  --substructure-distance-max-atom-ratio
  "${SUBSTRUCTURE_DISTANCE_MAX_ATOM_RATIO}"
  --substructure-distance-topk
  "${SUBSTRUCTURE_DISTANCE_TOPK}"
  --substructure-distance-mcs-timeout
  "${SUBSTRUCTURE_DISTANCE_MCS_TIMEOUT}"
  --substructure-distance-sim-threshold
  "${SUBSTRUCTURE_DISTANCE_SIM_THRESHOLD}"
  --disable-projected-cf-reward
)

if [ -n "${SFT_LORA_PATH}" ]; then
  cmd+=(--sft-lora-path "${SFT_LORA_PATH}")
fi

"${cmd[@]}"
