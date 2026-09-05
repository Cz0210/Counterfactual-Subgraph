#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

: "${T12_OUTPUT_ROOT:?set the completed T12 production root}"
: "${T12_VERIFICATION_ROOT:?set one fresh absolute verification root}"

echo "python=$(command -v python)"
python --version
python -c 'import torch; print(f"cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")'

cadence_args=()
if [[ "${T12_FORMAL_CHECKPOINT_CADENCE:-0}" == "1" ]]; then
  cadence_args+=(--formal-checkpoint-cadence)
fi

python scripts/verify_tastemolnet_gcf_full_generation.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --production-root "$T12_OUTPUT_ROOT" \
  --output-root "$T12_VERIFICATION_ROOT" \
  "${cadence_args[@]}"
