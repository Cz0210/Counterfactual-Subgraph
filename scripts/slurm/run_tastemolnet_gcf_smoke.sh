#!/usr/bin/env bash
# Static CLI parity only. TasteMolNet policy-v2 science is AutoDL-only.
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
echo "TasteMolNet T7 GCF is authorized only through the reviewed AutoDL controller." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python scripts/run_tastemolnet_gcf_smoke.py \
  --config configs/hpc.yaml \
  --stage T7_GCF_SMOKE \
  --output-dir /absolute/fresh/private/tastemolnet/t7-gcf-smoke \
  --set inference.fallback_to_heuristic=false
