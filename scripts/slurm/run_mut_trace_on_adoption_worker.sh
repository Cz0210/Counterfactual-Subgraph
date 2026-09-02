#!/bin/bash
#SBATCH --job-name=mut-trace-adopt
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
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
echo "AutoDL-only attached one-shot worker; do not submit this Slurm wrapper."
echo "The AutoDL one-shot must bind --semantic-finalizer-project-root to exact commit 582bc4b."
python scripts/autodl/run_mut_trace_on_adoption_worker.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  run --help
exit 2
