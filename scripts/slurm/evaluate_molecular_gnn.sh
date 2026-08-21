#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a frozen GNN bundle}"
DATASET_CSV="${DATASET_CSV:?Set DATASET_CSV to one frozen split CSV}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR}/evaluation_${SPLIT}_${SLURM_JOB_ID}}"
DEVICE="${DEVICE:-cuda:0}"

echo "python=$(which python)"
python --version
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY

# This classifier evaluator has no generation or heuristic-fallback code path.
python scripts/evaluate_molecular_gnn.py \
  --config configs/hpc.yaml \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --dataset-csv "${DATASET_CSV}" \
  --split "${SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"
