#!/bin/bash
#SBATCH -J gcf_hiv_vrrw_smoke
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
GCF_MAX_STEPS=${GCF_MAX_STEPS:-200}
TARGET_LABEL=${TARGET_LABEL:-1}
CF_MODE=${CF_MODE:-strict_flip}
RUN_DIR=${RUN_DIR:-outputs/hpc/gcfexplainer_hiv_csv/smoke/alpha_${GCF_ALPHA}_theta_${GCF_TRAIN_THETA}_steps_${GCF_MAX_STEPS}}

echo "===== GCF HIV CSV VRRW SMOKE ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "git commit: $(git rev-parse HEAD || true)"
echo "python path: $(which python)"
echo "conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CF_MODE=${CF_MODE}"
python --version

python scripts/gcf_hiv_csv_run_vrrw.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "${DATASET_DIR}" \
  --gnn-dir "${GNN_DIR}" \
  --run-dir "${RUN_DIR}" \
  --alpha "${GCF_ALPHA}" \
  --theta "${GCF_TRAIN_THETA}" \
  --max-steps "${GCF_MAX_STEPS}" \
  --teleport "${GCF_TELEPORT:-0.1}" \
  --sample-size "${GCF_SAMPLE_SIZE:-128}" \
  --target-label "${TARGET_LABEL}" \
  --device "${DEVICE:-cuda}" \
  --seed "${SEED:-0}"

echo "===== GCF HIV CSV VRRW SMOKE DONE ====="
cat "${RUN_DIR}/run_config.json"

