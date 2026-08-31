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
echo "Taste T6 verification is published only by the managed AutoDL worker." >&2
exit 64

# Unreachable documentation-only CLI parity. Never submit this script.
python scripts/autodl/verify_tastemolnet_ours_ppo_smoke.py \
  --science-root /absolute/private/tastemolnet/t6-science \
  --verification-root /absolute/fresh/private/tastemolnet/t6-verification \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false
