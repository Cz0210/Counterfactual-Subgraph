#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=bace-gine-defer
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available()); print("device_count=", torch.cuda.device_count())'

: "${BACE_GINE_DEFERRED_ARGS:?set the reviewed deferred-controller arguments}"

# Static paired entrypoint only.  The reviewed AutoDL deployment uses a
# persistent nohup controller and does not submit this script to HPC.
python scripts/autodl/run_deferred_bace_gnn_inference_benchmark.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  ${BACE_GINE_DEFERRED_ARGS}
