#!/usr/bin/env bash
#SBATCH --job-name=taste-gine-controller-static-only
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

echo "REFUSING_HPC_EXECUTION: TasteMolNet scoped authorization is AutoDL-only." >&2
echo "Documentation-only CLI parity: python scripts/autodl/run_tastemolnet_gine_controller.py status --controller-root ROOT --config configs/hpc.yaml --set inference.fallback_to_heuristic=false" >&2
exit 78
