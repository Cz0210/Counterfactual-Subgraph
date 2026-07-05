#!/bin/bash
#SBATCH -J clear_rf_greed
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
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
CLEAR_CANDIDATES_PATH=${CLEAR_CANDIDATES_PATH:-outputs/hpc/baselines/clear/aids/rf_unified/clear_aids_rf_fullgraph_candidates.csv}
DATASET_CSV=${DATASET_CSV:-outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset/sft_v3_hiv_ppo_prompts_train_label1.csv}
SMILES_COL=${SMILES_COL:-smiles}
LABEL_COL=${LABEL_COL:-label}
TARGET_LABEL=${TARGET_LABEL:-1}
TEACHER_PATH=${TEACHER_PATH:-outputs/hpc/oracle/aids_rf_model.pkl}
GREED_CHECKPOINT=${GREED_CHECKPOINT:-outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/eval/ccrcov_greed_hiv_full/clear_rf_fullgraph_greed_ged}
THRESHOLDS=${THRESHOLDS:-0.05,0.10,0.20}
CF_MODE=${CF_MODE:-strict_flip}
MIN_CF_DROP=${MIN_CF_DROP:-0.0}

echo "===== CLEAR RF GREED FULL ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATASET=${DATASET}"
echo "CLEAR_POOL=${CLEAR_CANDIDATES_PATH}"
echo "OUT_DIR=${OUTPUT_ROOT}"
echo "TEACHER_PATH=${TEACHER_PATH}"
echo "DATASET_CSV=${DATASET_CSV}"
echo "SMILES_COL=${SMILES_COL}"
echo "LABEL_COL=${LABEL_COL}"
echo "TARGET_LABEL=${TARGET_LABEL}"
echo "CF_MODE=${CF_MODE}"
echo "GREED_CHECKPOINT=${GREED_CHECKPOINT}"
echo "THRESHOLDS=${THRESHOLDS}"

python scripts/evaluate_ccrcov_with_greed.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --clear-fullgraph-candidates-path "${CLEAR_CANDIDATES_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --greed-checkpoint "${GREED_CHECKPOINT}" \
  --label "${TARGET_LABEL}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --thresholds "${THRESHOLDS}" \
  --cf-mode "${CF_MODE}" \
  --min-cf-drop "${MIN_CF_DROP}" \
  --output-root "${OUTPUT_ROOT}" \
  --partial-every "${PARTIAL_EVERY:-5000}"

echo "===== CLEAR RF GREED FULL DONE ====="
cat "${OUTPUT_ROOT}/combined/combined_threshold_summary.csv"
