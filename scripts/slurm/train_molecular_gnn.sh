#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

DATASET="${DATASET:-bace}"
DATA_DIR="${DATA_DIR:-data/processed/BACE}"
PROFILE="${PROFILE:-smoke}"
GNN_BACKBONE="${GNN_BACKBONE:-gine}"
RUN_NAME="${RUN_NAME:-${DATASET}_${GNN_BACKBONE}_${PROFILE}_${SLURM_JOB_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hpc/oracle/${DATASET}_gnn/${RUN_NAME}}"
DEVICE="${DEVICE:-cuda:0}"

echo "python=$(which python)"
python --version
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name=", torch.cuda.get_device_name(0))
PY
echo "dataset=${DATASET} profile=${PROFILE} backbone=${GNN_BACKBONE}"
echo "data_dir=${DATA_DIR} output_dir=${OUTPUT_DIR}"

python scripts/train_molecular_gnn.py \
  --config configs/hpc.yaml \
  --config "configs/gnn/${GNN_BACKBONE}.yaml" \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --profile "${PROFILE}" \
  --device "${DEVICE}"
