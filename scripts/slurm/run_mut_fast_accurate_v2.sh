#!/bin/bash
#SBATCH --job-name=mut-fast-v2
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
export MUT_SEMANTIC_FINALIZER_PROJECT_ROOT=/root/autodl-tmp/worktrees/final-five-closeout-582bc4b-20260902T040000Z
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "AutoDL-only persistent successor; do not submit this Slurm guard."
python scripts/autodl/run_mut_fast_accurate_v2.py --config configs/hpc.yaml --help
exit 2
