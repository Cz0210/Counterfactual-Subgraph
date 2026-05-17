#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=fullpool_run100

set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
set -u

PROJECT_ROOT=/share/home/u20526/czx/counterfactual-subgraph
DATASET=${PROJECT_ROOT}/outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv
BASE_MODEL=${PROJECT_ROOT}/pretrained_models/ChemLLM-7B-Chat
SFT_LORA=${PROJECT_ROOT}/outputs/hpc/sft_checkpoints/sft_v3_hiv_20260508_resplit_lr2e4_seed7_fix_columns/checkpoint-500
PPO_CKPT=${PROJECT_ROOT}/outputs/hpc/rl_checkpoints/decoded_chem_ppo_sanity100_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500
TEACHER=${PROJECT_ROOT}/outputs/hpc/oracle/aids_rf_model.pkl
OUT_DIR=${PROJECT_ROOT}/outputs/hpc/full_candidate_pools/label1/orig_shuffle100_n4_temp07_top09
OUT_JSONL=${OUT_DIR}/candidate_pool_label1_orig_shuffle100_n4.jsonl
OUT_SUMMARY=${OUT_DIR}/generation_summary.json
AUDIT_DIR=${OUT_DIR}/audit

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
echo "DATASET=${DATASET}"
echo "PPO_CKPT=${PPO_CKPT}"
echo "SFT_LORA=${SFT_LORA}"
echo "TEACHER=${TEACHER}"
echo "OUT_JSONL=${OUT_JSONL}"
echo "OUT_SUMMARY=${OUT_SUMMARY}"
echo "AUDIT_DIR=${AUDIT_DIR}"

python scripts/generate_full_candidate_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-path "${DATASET}" \
  --base-model-path "${BASE_MODEL}" \
  --sft-lora-path "${SFT_LORA}" \
  --ppo-checkpoint-path "${PPO_CKPT}" \
  --teacher-path "${TEACHER}" \
  --out-jsonl "${OUT_JSONL}" \
  --out-summary-json "${OUT_SUMMARY}" \
  --label-col label \
  --smiles-col parent_smiles \
  --target-label 1 \
  --num-return-sequences 4 \
  --generation-temperature 0.7 \
  --generation-top-p 0.9 \
  --generation-do-sample true \
  --max-new-tokens 96 \
  --batch-size 1 \
  --seed 13 \
  --enable-parent-projection \
  --enable-projected-cf-reward \
  --enable-substructure-distance-reward \
  --substructure-distance-reward-weight 0.3 \
  --projection-penalty 1.0 \
  --enable-minimal-syntax-repair \
  --enable-component-salvage

python scripts/audit_full_candidate_pool.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --pool-jsonl "${OUT_JSONL}" \
  --dataset-path "${DATASET}" \
  --teacher-path "${TEACHER}" \
  --out-dir "${AUDIT_DIR}" \
  --label-col label \
  --smiles-col parent_smiles \
  --target-label 1 \
  --sim-sample-size 5000 \
  --topk-show 10
