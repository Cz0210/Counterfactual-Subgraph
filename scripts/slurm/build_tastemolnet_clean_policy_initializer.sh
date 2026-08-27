#!/usr/bin/env bash
# Static CLI-parity wrapper. TasteMolNet policy v2 authorizes AutoDL only.
#SBATCH --job-name=taste-t5-static-refusal
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:05:00

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'
echo "REFUSING_HPC_EXECUTION: TasteMolNet T5 is AutoDL-only." >&2
echo "CLI parity: python scripts/build_tastemolnet_clean_policy_initializer.py build --config configs/hpc.yaml --set inference.fallback_to_heuristic=false" >&2
exit 78
