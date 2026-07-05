#!/bin/bash
#SBATCH -J clear_rf_molclr
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
EMBEDDING_DIR=${EMBEDDING_DIR:-outputs/hpc/molclr_ccrcov_embeddings}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/hpc/eval/ccrcov_molclr_hiv_full/clear_rf_fullgraph_molclr_embedding}
EMBEDDING_THRESHOLDS=${EMBEDDING_THRESHOLDS:-0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02}
CF_MODE=${CF_MODE:-strict_flip}
MIN_CF_DROP=${MIN_CF_DROP:-0.0}

echo "===== CLEAR RF MOLCLR FULL ====="
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
echo "EMBEDDING_DIR=${EMBEDDING_DIR}"
echo "EMBEDDING_THRESHOLDS=${EMBEDDING_THRESHOLDS}"
echo "[NOTE] EMBEDDING_DIR must include CLEAR candidate SMILES embeddings, e.g. from precompute_molclr_embeddings_for_ccrcov.py --clear-fullgraph-candidates-path."

python scripts/evaluate_ccrcov_with_molclr.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-csv "${DATASET_CSV}" \
  --clear-fullgraph-candidates-path "${CLEAR_CANDIDATES_PATH}" \
  --teacher-path "${TEACHER_PATH}" \
  --embedding-dir "${EMBEDDING_DIR}" \
  --label "${TARGET_LABEL}" \
  --smiles-col "${SMILES_COL}" \
  --label-col "${LABEL_COL}" \
  --embedding-thresholds "${EMBEDDING_THRESHOLDS}" \
  --cf-mode "${CF_MODE}" \
  --min-cf-drop "${MIN_CF_DROP}" \
  --output-root "${OUTPUT_ROOT}" \
  --partial-every "${PARTIAL_EVERY:-5000}"

echo "===== CLEAR RF MOLCLR FULL DONE ====="
cat "${OUTPUT_ROOT}/combined/combined_threshold_summary.csv"
