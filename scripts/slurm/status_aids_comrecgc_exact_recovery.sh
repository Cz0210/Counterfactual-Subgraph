#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper only: production status is read on AutoDL.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'
echo "do not submit: status path belongs to the AutoDL recovery controller" >&2
echo "reference: python scripts/autodl/status_aids_comrecgc_exact_recovery.py --config configs/hpc.yaml --manifest /absolute/manifest.json" >&2
exit 78
