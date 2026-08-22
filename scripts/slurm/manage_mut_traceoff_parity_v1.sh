#!/bin/bash
#SBATCH --job-name=mut-traceoff-controller
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
echo "AutoDL-only continuation; do not submit this Slurm wrapper."
exit 2

python scripts/autodl/manage_mut_traceoff_parity_v1.py \
  --config configs/hpc.yaml validate \
  --spec configs/autodl/mut_traceoff_parity_v1.template.json
