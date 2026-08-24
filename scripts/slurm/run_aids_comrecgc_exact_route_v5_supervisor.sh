#!/bin/bash
# Static CLI parity for the AutoDL-only AIDS exact-route v5; do not submit.
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=aids-comrecgc-exact-v5

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export RUN_TASTEMOLNET=0

echo "This is static CLI parity for the AutoDL-only exact route; do not submit." >&2
echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
PY
exit 78

# Unreachable documentation-only parity command.  The frozen AutoDL manifest
# supplies every scientific/source/cache/lock environment field.
bash scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh \
  --config configs/hpc.yaml
