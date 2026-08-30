#!/usr/bin/env bash
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
python -c 'import torch; print("cuda_available=" + str(torch.cuda.is_available()))'
echo "Taste fixed-budget NeuroSED managed-v2 adoption is AutoDL-only; Slurm is static CLI parity." >&2
exit 78

# AutoDL controller-owned shape (intentionally unreachable on Slurm):
python -B scripts/autodl/adopt_tastemolnet_fixed_budget_neurosed_v2.py inspect \
  --config configs/hpc.yaml \
  --source-root /absolute/fixed-budget-pass \
  --vendored-gcf-root baselines/gcfexplainer_official \
  --expected-source-inventory-sha256 64-lowercase-hex
