#!/usr/bin/env bash
#SBATCH --job-name=bace-globalgce-recover
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/autodl/recover_bace_globalgce_terminal.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  recover \
  --failed-controller-root "${FAILED_CONTROLLER_ROOT:?Set FAILED_CONTROLLER_ROOT}" \
  --source-round-root "${SOURCE_ROUND_ROOT:?Set SOURCE_ROUND_ROOT}" \
  --source-manifest "${SOURCE_MANIFEST:?Set SOURCE_MANIFEST}" \
  --native-train-csv "${NATIVE_TRAIN_CSV:?Set NATIVE_TRAIN_CSV}" \
  --official-root "${OFFICIAL_ROOT:?Set OFFICIAL_ROOT}" \
  --gnn-checkpoint "${GNN_CHECKPOINT:?Set GNN_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR:?Set OUTPUT_DIR}" \
  --proc-root /proc
