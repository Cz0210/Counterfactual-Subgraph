#!/usr/bin/env bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=llm-early-gate-snapshot
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

exec python scripts/autodl/build_llm_early_launch_snapshot_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --matrix-authority "${MATRIX_AUTHORITY_ROOT:?}/state.json" \
  --owner-registry "${FINAL16_OWNER_REGISTRY:?}" \
  --runtime-observation "${LLM_EARLY_RUNTIME_OBSERVATION:?}" \
  --output "${LLM_ABLATION_MAIN_SNAPSHOT:?}"
