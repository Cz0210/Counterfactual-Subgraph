#!/usr/bin/env bash
# Static CLI-parity wrapper. The current TasteMolNet campaign is AutoDL-only;
# do not submit this command to HPC.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
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
echo "TasteMolNet public-artifact auditing is AutoDL-only; HPC is forbidden." >&2
exit 64
