#!/bin/bash
#SBATCH --job-name=comrecgc-full-gate
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${M500_ROOT:?set M500_ROOT}"
: "${M1000_ROOT:?set M1000_ROOT}"
: "${OUTPUT_DIR:?set a fresh OUTPUT_DIR}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

python scripts/baselines/comrecgc/build_full_acceleration_gate.py \
  --config configs/hpc.yaml \
  --m500-root "$M500_ROOT" \
  --m1000-root "$M1000_ROOT" \
  --output-dir "$OUTPUT_DIR"
