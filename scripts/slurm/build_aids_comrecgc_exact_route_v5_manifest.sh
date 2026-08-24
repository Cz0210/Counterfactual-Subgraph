#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper only: AutoDL owns the production v5 controller.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'
echo "do not submit: AIDS exact-route v5 is AutoDL-only" >&2
echo "reference: python scripts/autodl/build_aids_comrecgc_exact_route_v5_manifest.py --config configs/hpc.yaml validate --spec /absolute/aids-v5-spec.json" >&2
exit 78
