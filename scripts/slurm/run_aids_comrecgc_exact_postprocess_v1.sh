#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=128G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=aids_exact_post

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${AIDS_EXACT_CONTROLLER_MANIFEST:?set AIDS_EXACT_CONTROLLER_MANIFEST}"
: "${AIDS_EXACT_RECEIPT:?set AIDS_EXACT_RECEIPT}"
: "${AIDS_POSTPROCESS_OUTPUT_ROOT:?set AIDS_POSTPROCESS_OUTPUT_ROOT}"
: "${AIDS_POSTPROCESS_HEARTBEAT:?set AIDS_POSTPROCESS_HEARTBEAT}"
: "${AIDS_POSTPROCESS_MAX_WORKERS:=8}"

export CUDA_VISIBLE_DEVICES=""
export DEVICE=cpu
export GPU_REQUIRED=0
export OMP_NUM_THREADS="$AIDS_POSTPROCESS_MAX_WORKERS"
export MKL_NUM_THREADS="$AIDS_POSTPROCESS_MAX_WORKERS"
export OPENBLAS_NUM_THREADS="$AIDS_POSTPROCESS_MAX_WORKERS"
export NUMEXPR_NUM_THREADS="$AIDS_POSTPROCESS_MAX_WORKERS"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "route=cpu_only_exact_dbscan_adoption"

python scripts/autodl/run_aids_comrecgc_exact_postprocess_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --controller-manifest "$AIDS_EXACT_CONTROLLER_MANIFEST" \
  --exact-receipt "$AIDS_EXACT_RECEIPT" \
  --output-root "$AIDS_POSTPROCESS_OUTPUT_ROOT" \
  --heartbeat-path "$AIDS_POSTPROCESS_HEARTBEAT" \
  --max-workers "$AIDS_POSTPROCESS_MAX_WORKERS"
