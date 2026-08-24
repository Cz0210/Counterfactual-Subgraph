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
mkdir -p logs
echo "[ENV] python=$(command -v python)"
python --version
python -c 'import torch; print(f"[ENV] cuda_available={torch.cuda.is_available()}")'
echo "do not submit: this selector adoption gate is owned by the AutoDL v5 controller" >&2
echo "reference: python scripts/autodl/write_aids_comrecgc_v5_selector_gate.py --config configs/hpc.yaml --thresholds /absolute/threshold.json --expected-sha256 SHA256 --output-dir /fresh/output" >&2
exit 78
