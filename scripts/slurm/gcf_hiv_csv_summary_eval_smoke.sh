#!/bin/bash
#SBATCH -J gcf_hiv_eval_smoke
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
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

DATASET_DIR=${DATASET_DIR:-outputs/hpc/gcfexplainer_hiv_csv/dataset}
GNN_DIR=${GNN_DIR:-outputs/hpc/gcfexplainer_hiv_csv/gnn_smoke}
GCF_ALPHA=${GCF_ALPHA:-0.5}
GCF_TRAIN_THETA=${GCF_TRAIN_THETA:-0.05}
GCF_EVAL_THETA=${GCF_EVAL_THETA:-0.1}
GCF_MAX_STEPS=${GCF_MAX_STEPS:-200}
GCF_TOP_K_LIST=${GCF_TOP_K_LIST:-1,5,10}
TARGET_LABEL=${TARGET_LABEL:-1}
CF_MODE=${CF_MODE:-strict_flip}
RUN_DIR=${RUN_DIR:-outputs/hpc/gcfexplainer_hiv_csv/smoke/alpha_${GCF_ALPHA}_theta_${GCF_TRAIN_THETA}_steps_${GCF_MAX_STEPS}}
OUT_DIR=${OUT_DIR:-${RUN_DIR}/native_eval}
SELECTED_GRAPHS=${SELECTED_GRAPHS:-${RUN_DIR}/summary_export/selected_counterfactual_graphs.pt}

echo "===== GCF HIV CSV SUMMARY + EVAL SMOKE ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CF_MODE=${CF_MODE}"
python --version
echo "GCF_TOP_K_LIST=${GCF_TOP_K_LIST}"

python scripts/gcf_hiv_csv_export_summary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --run-dir "${RUN_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --gnn-dir "${GNN_DIR}" \
  --top-k-list "${GCF_TOP_K_LIST}" \
  --eval-theta "${GCF_EVAL_THETA}" \
  --target-label "${TARGET_LABEL}"

python scripts/evaluate_gcf_hiv_csv_native.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "${DATASET_DIR}" \
  --gnn-dir "${GNN_DIR}" \
  --selected-graphs "${SELECTED_GRAPHS}" \
  --theta-list "${THETA_LIST:-0.05,0.10,0.20}" \
  --target-label "${TARGET_LABEL}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE:-cuda}"

echo "===== GCF HIV CSV SUMMARY + EVAL SMOKE DONE ====="
cat "${OUT_DIR}/native_ccrcov_summary.csv"

