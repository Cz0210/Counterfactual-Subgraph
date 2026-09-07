#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# The actual A+ campaign uses AutoDL's inherited existing owner lease.
# This paired wrapper only declares the same CLI; it cannot bypass that lease.
set -eo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
set -u
echo "Python: $(command -v python)"
python --version
python -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
python -I -B scripts/run_bace_globalgce_aplus.py --config configs/hpc.yaml --set inference.fallback_to_heuristic=false "$@"
