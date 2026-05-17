#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=audit_reward_teacher

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
DATASET_PATH=${PROJECT_DIR}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
CANDIDATE_POOL=${PROJECT_DIR}/outputs/hpc/rl_checkpoints/decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_failfix_ckpt500/candidate_pool.jsonl
TEACHER_PATH=${PROJECT_DIR}/outputs/hpc/oracle/aids_rf_model.pkl
OUT_DIR=${PROJECT_DIR}/outputs/hpc/audits/reward_teacher_audit_ppo100_label1
LABEL_COL=label
SMILES_COL=parent_smiles
SIM_SAMPLE_SIZE=5000

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
which python
python -V
python - <<'PY'
import os
import torch
print("python executable:", os.sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
echo "====================="

cd /share/home/u20526/czx/counterfactual-subgraph

export PYTHONPATH=$PWD

echo "repo pwd: $(pwd)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "git_commit=$(git rev-parse HEAD)"
echo "dataset_path=${DATASET_PATH}"
echo "candidate_pool=${CANDIDATE_POOL}"
echo "teacher_path=${TEACHER_PATH}"
echo "out_dir=${OUT_DIR}"

if [ ! -f "${DATASET_PATH}" ]; then
  echo "[ERROR] dataset not found: ${DATASET_PATH}"
  exit 1
fi

if [ ! -f "${CANDIDATE_POOL}" ]; then
  echo "[ERROR] candidate pool not found: ${CANDIDATE_POOL}"
  exit 1
fi

if [ ! -f "${TEACHER_PATH}" ]; then
  echo "[ERROR] teacher path not found: ${TEACHER_PATH}"
  exit 1
fi

python scripts/audit_reward_teacher.py \
  --config configs/hpc.yaml \
  --dataset-path "${DATASET_PATH}" \
  --candidate-pool "${CANDIDATE_POOL}" \
  --teacher-path "${TEACHER_PATH}" \
  --out-dir "${OUT_DIR}" \
  --label-col "${LABEL_COL}" \
  --smiles-col "${SMILES_COL}" \
  --sim-sample-size "${SIM_SAMPLE_SIZE}"
