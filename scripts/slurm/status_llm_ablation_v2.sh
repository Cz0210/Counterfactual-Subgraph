#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=llm-ablation-v2-status
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

: "${LLM_ABLATION_MAIN_SNAPSHOT:?set the current main-state snapshot path}"
args=(
  python scripts/autodl/status_llm_ablation_v2.py
  --config configs/hpc.yaml
  --main-snapshot "$LLM_ABLATION_MAIN_SNAPSHOT"
)
[[ -n "${LLM_ABLATION_STATUS_OUTPUT:-}" ]] && args+=(--output "$LLM_ABLATION_STATUS_OUTPUT")
"${args[@]}"
