#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
which python
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
# A future immutable task may bind the existing inherited-owner activation;
# this wrapper never modifies an already running reference process.
# recovery-segment is only a child interface: the canonical owner must pass
# T12_OWNER_HELD_GPU_FD via pass_fds and a full CUDA_VISIBLE_DEVICES UUID.
# Submitting a standalone recovery Slurm job without that owner fails closed.
python -I -B scripts/autodl/run_t12_reference_500_v1.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
