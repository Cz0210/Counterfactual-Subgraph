#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=filter_ppo_t05_l1

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_DIR=/share/home/u20526/czx/counterfactual-subgraph
DATASET=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
TEACHER=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/oracle/aids_rf_model.pkl
OUT_CSV=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1_teacher_p05.csv
OUT_JSON=/share/home/u20526/czx/counterfactual-subgraph/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1_teacher_p05.summary.json

echo "===== ENV CHECK ====="
echo "host: $(hostname)"
echo "pwd before cd: $(pwd)"
echo "conda env: ${CONDA_DEFAULT_ENV:-}"
echo "python path: $(which python)"
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
echo "dataset=${DATASET}"
echo "teacher=${TEACHER}"
echo "out_csv=${OUT_CSV}"
echo "out_json=${OUT_JSON}"

if [ ! -f "${DATASET}" ]; then
  echo "[ERROR] dataset not found: ${DATASET}"
  exit 1
fi

if [ ! -f "${TEACHER}" ]; then
  echo "[ERROR] teacher not found: ${TEACHER}"
  exit 1
fi

python scripts/filter_ppo_prompts_by_teacher_confidence.py \
  --config configs/hpc.yaml \
  --dataset-path "${DATASET}" \
  --teacher-path "${TEACHER}" \
  --label-col label \
  --smiles-col parent_smiles \
  --target-label 1 \
  --min-p-label 0.5 \
  --require-teacher-correct \
  --out-csv "${OUT_CSV}" \
  --out-json "${OUT_JSON}"
