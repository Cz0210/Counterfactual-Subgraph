#!/usr/bin/env bash
# Repository GPU wrapper parity only. The current rematerialize action is CPU
# and is launched by the existing AutoDL task authority, not on HPC login.
# A future training action must first be explicitly bound to the GPU0 lease.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -e
source ~/.bashrc
conda activate smiles_pip118
set -uo pipefail
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
exec python -I -B scripts/run_bace_globalgce_chemaligned.py --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false "$@"
