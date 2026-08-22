#!/usr/bin/env bash
#SBATCH --job-name=gpu_colocation_gate
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
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

: "${SINGLE_PROFILE_0:?set SINGLE_PROFILE_0}"
: "${SINGLE_PROFILE_1:?set SINGLE_PROFILE_1}"
: "${COLOCATED_PROFILE:?set COLOCATED_PROFILE}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

python scripts/autodl/gate_gpu_colocation_benchmark.py \
  --config configs/hpc.yaml \
  --single-profile "$SINGLE_PROFILE_0" \
  --single-profile "$SINGLE_PROFILE_1" \
  --colocated-profile "$COLOCATED_PROFILE" \
  --output-dir "$OUTPUT_DIR"
