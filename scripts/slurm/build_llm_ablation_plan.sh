#!/bin/bash
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=8G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=llm-ablation-plan

set -euo pipefail

source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
PY

: "${LLM_ABLATION_SPEC:?set LLM_ABLATION_SPEC to a fully pinned JSON spec}"
: "${LLM_ABLATION_PLAN_OUTPUT:?set LLM_ABLATION_PLAN_OUTPUT to a fresh absolute directory}"

python scripts/ablations/llm/build_llm_ablation_plan.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --ablation-spec "$LLM_ABLATION_SPEC" \
  --output-dir "$LLM_ABLATION_PLAN_OUTPUT"
