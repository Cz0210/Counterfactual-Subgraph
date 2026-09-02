#!/usr/bin/env bash
# Config-only gate wrapper.  No science entrypoint exists in framework v2.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}

: "${LLM_ABLATION_MAIN_SNAPSHOT:?set LLM_ABLATION_MAIN_SNAPSHOT to current controller evidence}"
: "${BACE_LLM_REFERENCE_CONTRACT:?set exact BACE/Ours reference-v2 path}"
: "${BACE_LLM_REFERENCE_CONTRACT_SHA256:?set exact reference-v2 file SHA256}"

args=(
  "$PY" "$PROJECT_ROOT/scripts/autodl/status_llm_ablation_v2.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --main-snapshot "$LLM_ABLATION_MAIN_SNAPSHOT"
  --reference-contract "$BACE_LLM_REFERENCE_CONTRACT"
  --reference-contract-sha256 "$BACE_LLM_REFERENCE_CONTRACT_SHA256"
)
[[ -n "${LLM_ABLATION_EARLY_RUN_RECEIPT:-}" ]] && args+=(--early-run-receipt "$LLM_ABLATION_EARLY_RUN_RECEIPT")
[[ -n "${LLM_ABLATION_STATUS_OUTPUT:-}" ]] && args+=(--output "$LLM_ABLATION_STATUS_OUTPUT")
[[ -n "${CHEMLLM_2B_SNAPSHOT_MANIFEST:-}" ]] && args+=(--two-b-snapshot-manifest "$CHEMLLM_2B_SNAPSHOT_MANIFEST" --two-b-snapshot-manifest-sha256 "${CHEMLLM_2B_SNAPSHOT_MANIFEST_SHA256:?}")
[[ -n "${CHEMLLM_2B_PARAMETER_REPORT:-}" ]] && args+=(--two-b-parameter-report "$CHEMLLM_2B_PARAMETER_REPORT" --two-b-parameter-report-sha256 "${CHEMLLM_2B_PARAMETER_REPORT_SHA256:?}")
[[ -n "${CHEMLLM_7B_PARAMETER_REPORT:-}" ]] && args+=(--seven-b-parameter-report "$CHEMLLM_7B_PARAMETER_REPORT" --seven-b-parameter-report-sha256 "${CHEMLLM_7B_PARAMETER_REPORT_SHA256:?}")
[[ -n "${CHEMLLM_20B_METADATA_MANIFEST:-}" ]] && args+=(--twenty-b-metadata-manifest "$CHEMLLM_20B_METADATA_MANIFEST" --twenty-b-metadata-manifest-sha256 "${CHEMLLM_20B_METADATA_MANIFEST_SHA256:?}")

nice -n 10 "${args[@]}"
echo "BLOCKED_CONFIG_ONLY_NO_SCIENCE_ENTRYPOINT" >&2
exit 78
