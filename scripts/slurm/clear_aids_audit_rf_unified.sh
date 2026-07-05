#!/bin/bash
#SBATCH -J clear_rf_audit
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -eo pipefail
set +u
source ~/.bashrc
conda activate smiles_pip118

PROJECT_ROOT=${PROJECT_ROOT:-/share/home/u20526/czx/counterfactual-subgraph}
cd "${PROJECT_ROOT}"
export PYTHONPATH=$PWD
mkdir -p logs

DATASET=${DATASET:-aids}
CLEAR_POOL=${CLEAR_POOL:-outputs/hpc/baselines/clear/aids/candidate_pool/clear_aids_candidate_pool.with_graphs.jsonl}
OUT_JSON=${OUT_JSON:-outputs/hpc/baselines/clear/aids/rf_unified/audit_clear_rf_feasibility.json}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}
DATASET_CSV=${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
TARGET_LABEL=${TARGET_LABEL:-1}
CF_MODE=${CF_MODE:-strict_flip}
MIN_VALID_CANDIDATES=${MIN_VALID_CANDIDATES:-20}
MIN_VALID_RATE=${MIN_VALID_RATE:-0.001}
ADJ_THRESHOLD=${ADJ_THRESHOLD:-0.5}

echo "===== CLEAR RF UNIFIED AUDIT ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATASET=${DATASET}"
echo "CLEAR_POOL=${CLEAR_POOL}"
echo "OUT_DIR=$(dirname "${OUT_JSON}")"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "DATASET_CSV=${DATASET_CSV}"
echo "SMILES_COL=${SMILES_COL}"
echo "LABEL_COL=${LABEL_COL}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "CF_MODE=${CF_MODE}"
echo "MIN_VALID_CANDIDATES=${MIN_VALID_CANDIDATES}"
echo "MIN_VALID_RATE=${MIN_VALID_RATE}"
echo "ADJ_THRESHOLD=${ADJ_THRESHOLD}"

args=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --clear-pool "${CLEAR_POOL}"
  --dataset "${DATASET}"
  --out-json "${OUT_JSON}"
  --min-valid-candidates "${MIN_VALID_CANDIDATES}"
  --min-valid-rate "${MIN_VALID_RATE}"
  --adj-threshold "${ADJ_THRESHOLD}"
)
if [ -n "${MAX_RECORDS:-}" ]; then
  args+=(--max-records "${MAX_RECORDS}")
fi

python scripts/baselines/clear/audit_clear_pool_for_rf_eval.py "${args[@]}"

echo "===== CLEAR RF UNIFIED AUDIT DONE ====="
cat "${OUT_JSON}"
