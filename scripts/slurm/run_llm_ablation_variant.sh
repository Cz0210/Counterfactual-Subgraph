#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=llm-core-variant
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail
source ~/.bashrc
conda activate smiles_pip118
cd /share/home/u20526/czx/counterfactual-subgraph
export PYTHONPATH=$PWD

echo "python=$(command -v python)"
python --version
python -c 'import torch; print("cuda_available=", torch.cuda.is_available())'

: "${LLM_CORE_RUN_SPEC:?set exact core run-spec JSON}"
: "${LLM_CORE_RUN_SPEC_SHA256:?set exact core run-spec file SHA256}"
: "${LLM_CORE_LAUNCH_DECISION:?set exact authorized launch decision JSON}"
: "${LLM_CORE_LAUNCH_DECISION_SHA256:?set launch-decision file SHA256}"
args=(
  python scripts/autodl/run_llm_ablation_variant.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --run-spec "$LLM_CORE_RUN_SPEC"
  --run-spec-sha256 "$LLM_CORE_RUN_SPEC_SHA256"
  --launch-decision "$LLM_CORE_LAUNCH_DECISION"
  --launch-decision-sha256 "$LLM_CORE_LAUNCH_DECISION_SHA256"
)
[[ "${LLM_CORE_RESUME:-0}" == "1" ]] && args+=(--resume)
"${args[@]}"
