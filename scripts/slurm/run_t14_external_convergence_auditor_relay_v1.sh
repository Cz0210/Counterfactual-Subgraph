#!/bin/bash
#SBATCH --job-name=t14-ro-relay
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
: "${T14_CHECKPOINT_ROOT:?set T14_CHECKPOINT_ROOT}"
: "${T14_CONVERGENCE_RELAY_ROOT:?set T14_CONVERGENCE_RELAY_ROOT}"
: "${T14_AUDITOR_EXECUTION_COMMIT:?set T14_AUDITOR_EXECUTION_COMMIT}"

# Repository workflow parity only.  The Python relay itself never acquires or
# uses a GPU; AutoDL production launches it through the no-GPU launcher.
CUDA_VISIBLE_DEVICES='' python scripts/autodl/run_t14_external_convergence_auditor_relay_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --checkpoint-root "$T14_CHECKPOINT_ROOT" \
  --relay-root "$T14_CONVERGENCE_RELAY_ROOT" \
  --execution-commit "$T14_AUDITOR_EXECUTION_COMMIT"
