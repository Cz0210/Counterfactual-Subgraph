#!/usr/bin/env bash
#SBATCH --job-name=taste-t5-base-worker-v2
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

echo "REFUSING_HPC_EXECUTION: TasteMolNet clean-base adoption is AutoDL-only." >&2
echo "CLI parity: python scripts/autodl/tastemolnet_t5_clean_base_worker_v2.py build --config configs/hpc.yaml --source-model SOURCE --expected-source-inventory-sha256 SHA256" >&2
exit 75
