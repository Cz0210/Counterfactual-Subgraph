#!/bin/bash
#SBATCH --partition=A800
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --job-name=llm-core-status
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

: "${LLM_CORE_RUN_SPEC:?set exact core run-spec JSON}"
: "${LLM_CORE_RUN_SPEC_SHA256:?set exact core run-spec file SHA256}"
: "${LLM_ABLATION_MAIN_SNAPSHOT:?set current main-state snapshot JSON}"
: "${BACE_LLM_REFERENCE_CONTRACT:?set exact reference-v2 JSON}"
: "${BACE_LLM_REFERENCE_CONTRACT_SHA256:?set exact reference file SHA256}"
args=(
  python scripts/autodl/status_llm_ablation_core_v1.py
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --run-spec "$LLM_CORE_RUN_SPEC"
  --run-spec-sha256 "$LLM_CORE_RUN_SPEC_SHA256"
  --main-snapshot "$LLM_ABLATION_MAIN_SNAPSHOT"
  --reference-contract "$BACE_LLM_REFERENCE_CONTRACT"
  --reference-contract-sha256 "$BACE_LLM_REFERENCE_CONTRACT_SHA256"
)
[[ -n "${LLM_ABLATION_EARLY_RUN_RECEIPT:-}" ]] && args+=(--early-run-receipt "$LLM_ABLATION_EARLY_RUN_RECEIPT")
[[ -n "${LLM_CORE_STATUS_OUTPUT:-}" ]] && args+=(--output "$LLM_CORE_STATUS_OUTPUT")
[[ -n "${CHEMLLM_2B_SNAPSHOT_MANIFEST:-}" ]] && args+=(--two-b-snapshot-manifest "$CHEMLLM_2B_SNAPSHOT_MANIFEST" --two-b-snapshot-manifest-sha256 "${CHEMLLM_2B_SNAPSHOT_MANIFEST_SHA256:?}")
[[ -n "${CHEMLLM_2B_PARAMETER_REPORT:-}" ]] && args+=(--two-b-parameter-report "$CHEMLLM_2B_PARAMETER_REPORT" --two-b-parameter-report-sha256 "${CHEMLLM_2B_PARAMETER_REPORT_SHA256:?}")
[[ -n "${CHEMLLM_7B_PARAMETER_REPORT:-}" ]] && args+=(--seven-b-parameter-report "$CHEMLLM_7B_PARAMETER_REPORT" --seven-b-parameter-report-sha256 "${CHEMLLM_7B_PARAMETER_REPORT_SHA256:?}")
[[ -n "${CHEMLLM_20B_METADATA_MANIFEST:-}" ]] && args+=(--twenty-b-metadata-manifest "$CHEMLLM_20B_METADATA_MANIFEST" --twenty-b-metadata-manifest-sha256 "${CHEMLLM_20B_METADATA_MANIFEST_SHA256:?}")
"${args[@]}"
