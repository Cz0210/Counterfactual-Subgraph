#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=fullpool_orig100

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

# Future template only; do not execute until shuffle200 finishes and is verified.
# PPO200_CKPT=${PROJECT_ROOT}/outputs/hpc/rl_checkpoints/decoded_chem_ppo_sanity200_sftv3_projcf_dist03_projpen1_orig_shuffle13_ckpt500

NUM_RETURN_SEQUENCES=4
GEN_TEMPERATURE=0.7
GEN_TOP_P=0.9
GEN_DO_SAMPLE=true
MAX_NEW_TOKENS=96
BATCH_SIZE=1
SEED=13

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
echo "BASE_MODEL=${BASE_MODEL}"
echo "PPO_CKPT=${PPO_CKPT}"
echo "SFT_LORA=${SFT_LORA}"
echo "TEACHER=${TEACHER}"
echo "OUT_JSONL=${OUT_JSONL}"
echo "OUT_SUMMARY=${OUT_SUMMARY}"
echo "NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES}"
echo "GEN_TEMPERATURE=${GEN_TEMPERATURE}"
echo "GEN_TOP_P=${GEN_TOP_P}"
echo "GEN_DO_SAMPLE=${GEN_DO_SAMPLE}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "SEED=${SEED}"

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
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --generation-temperature "${GEN_TEMPERATURE}" \
  --generation-top-p "${GEN_TOP_P}" \
  --generation-do-sample "${GEN_DO_SAMPLE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --enable-parent-projection \
  --enable-projected-cf-reward \
  --enable-substructure-distance-reward \
  --substructure-distance-reward-weight 0.3 \
  --projection-penalty 1.0 \
  --enable-minimal-syntax-repair \
  --enable-component-salvage
