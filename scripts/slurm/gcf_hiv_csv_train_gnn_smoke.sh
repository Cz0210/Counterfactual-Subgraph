#!/bin/bash
#SBATCH -J gcf_hiv_gnn_smoke
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
OUT_DIR=${OUT_DIR:-outputs/hpc/gcfexplainer_hiv_csv/gnn_smoke}
GNN_EPOCHS=${GNN_EPOCHS:-5}

echo "===== GCF HIV CSV TRAIN GNN SMOKE ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python --version
echo "DATASET_DIR=${DATASET_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "GNN_EPOCHS=${GNN_EPOCHS}"
echo "CF_MODE=strict_flip"

python scripts/gcf_hiv_csv_train_gnn.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --epochs "${GNN_EPOCHS}" \
  --seed "${SEED:-0}" \
  --device "${DEVICE:-cuda}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --checkpoint-metric "${CHECKPOINT_METRIC:-macro_f1}"

echo "===== GCF HIV CSV TRAIN GNN SMOKE DONE ====="
cat "${OUT_DIR}/gnn_train_summary.json"

