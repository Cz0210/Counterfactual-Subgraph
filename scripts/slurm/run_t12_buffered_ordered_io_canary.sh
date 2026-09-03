#!/usr/bin/env bash
#SBATCH --job-name=t12-buffered-ordered-io-canary
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${T12_ORDERED_IO_INPUT_JSONL:?set a frozen 510-row capture including graph identity and exact embedding bytes}"
: "${T12_ORDERED_IO_CANARY_ROOT:?set a fresh persistent output root}"

python -B scripts/autodl/run_t12_buffered_ordered_io_canary.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --input-jsonl "$T12_ORDERED_IO_INPUT_JSONL" \
  --output-root "$T12_ORDERED_IO_CANARY_ROOT" \
  --checkpoint-at "${T12_ORDERED_IO_CHECKPOINT_AT:-500}" \
  --post-reload-records "${T12_ORDERED_IO_POST_RELOAD_RECORDS:-10}" \
  --buffered-batch-records "${T12_ORDERED_IO_BATCH_RECORDS:-256}" \
  --workers "${T12_ORDERED_IO_WORKERS:-4}" \
  --executor "${T12_ORDERED_IO_EXECUTOR:-process}"
