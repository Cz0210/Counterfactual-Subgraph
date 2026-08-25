#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Static parity wrapper only: this recovery is explicitly AutoDL-only/CPU-only.
set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'
echo "do not submit: AIDS disconnected-exact recovery is AutoDL-only" >&2
echo "reference: python scripts/autodl/build_aids_comrecgc_exact_recovery_v1.py --config configs/hpc.yaml validate --spec /absolute/spec.json" >&2
exit 78
