#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=bace-gine-matrix
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("device_count=", torch.cuda.device_count())'

: "${BACE_GCF_DATASET_DIR:?set BACE_GCF_DATASET_DIR}"
: "${BACE_GINE_CHECKPOINT:?set BACE_GINE_CHECKPOINT}"
: "${BACE_GINE_INFERENCE_BENCHMARK_OUTPUT:?set a fresh output root}"

python scripts/autodl/benchmark_bace_gnn_inference_matrix.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --dataset-dir "$BACE_GCF_DATASET_DIR" \
  --checkpoint-dir "$BACE_GINE_CHECKPOINT" \
  --output-dir "$BACE_GINE_INFERENCE_BENCHMARK_OUTPUT" \
  --batch-sizes "${BACE_GINE_INFERENCE_BATCH_SIZES:-1,8,32,128,512}" \
  --warmups "${BACE_GINE_INFERENCE_WARMUPS:-2}" \
  --repeats "${BACE_GINE_INFERENCE_REPEATS:-5}"
