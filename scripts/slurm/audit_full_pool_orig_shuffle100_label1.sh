#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=audit_orig100

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=/share/home/u20526/czx/counterfactual-subgraph
POOL=${PROJECT_ROOT}/outputs/hpc/full_candidate_pools/label1/orig_shuffle100_n4_temp07_top09/candidate_pool_label1_orig_shuffle100_n4.jsonl
DATASET=${PROJECT_ROOT}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
TEACHER=${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl
OUT_DIR=${PROJECT_ROOT}/outputs/hpc/full_candidate_pools/label1/orig_shuffle100_n4_temp07_top09/audit

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
echo "git_commit=$(git rev-parse HEAD)"
echo "POOL=${POOL}"
echo "DATASET=${DATASET}"
echo "TEACHER=${TEACHER}"
echo "OUT_DIR=${OUT_DIR}"

python scripts/audit_full_candidate_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pool-jsonl "${POOL}" \
  --dataset-path "${DATASET}" \
  --teacher-path "${TEACHER}" \
  --out-dir "${OUT_DIR}" \
  --label-col label \
  --smiles-col parent_smiles \
  --target-label 1 \
  --sim-sample-size 5000 \
  --topk-show 10
