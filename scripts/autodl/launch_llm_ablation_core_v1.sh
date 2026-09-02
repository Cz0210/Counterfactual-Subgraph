#!/usr/bin/env bash
# Launch exactly one core LLM-ablation variant after a fresh evidence gate PASS.

set -euo pipefail

: "${LLM_CORE_RUN_SPEC:?set exact core run-spec JSON}"
: "${LLM_CORE_RUN_SPEC_SHA256:?set exact core run-spec file SHA256}"
: "${LLM_ABLATION_MAIN_SNAPSHOT:?set current main-state snapshot JSON}"
: "${BACE_LLM_REFERENCE_CONTRACT:?set exact BACE/Ours reference-v2 JSON}"
: "${BACE_LLM_REFERENCE_CONTRACT_SHA256:?set exact reference file SHA256}"
: "${LLM_ABLATION_EARLY_RUN_RECEIPT:?set project-owner early-run receipt}"

decision="${LLM_CORE_LAUNCH_DECISION:-$(mktemp /tmp/llm-core-launch-decision.XXXXXX.json)}"
status_args=(
  python scripts/autodl/status_llm_ablation_core_v1.py
  --config configs/hpc.yaml
  --run-spec "$LLM_CORE_RUN_SPEC"
  --run-spec-sha256 "$LLM_CORE_RUN_SPEC_SHA256"
  --main-snapshot "$LLM_ABLATION_MAIN_SNAPSHOT"
  --reference-contract "$BACE_LLM_REFERENCE_CONTRACT"
  --reference-contract-sha256 "$BACE_LLM_REFERENCE_CONTRACT_SHA256"
  --early-run-receipt "$LLM_ABLATION_EARLY_RUN_RECEIPT"
  --output "$decision"
)
[[ -n "${CHEMLLM_2B_SNAPSHOT_MANIFEST:-}" ]] && status_args+=(--two-b-snapshot-manifest "$CHEMLLM_2B_SNAPSHOT_MANIFEST" --two-b-snapshot-manifest-sha256 "${CHEMLLM_2B_SNAPSHOT_MANIFEST_SHA256:?}")
[[ -n "${CHEMLLM_2B_PARAMETER_REPORT:-}" ]] && status_args+=(--two-b-parameter-report "$CHEMLLM_2B_PARAMETER_REPORT" --two-b-parameter-report-sha256 "${CHEMLLM_2B_PARAMETER_REPORT_SHA256:?}")
[[ -n "${CHEMLLM_7B_PARAMETER_REPORT:-}" ]] && status_args+=(--seven-b-parameter-report "$CHEMLLM_7B_PARAMETER_REPORT" --seven-b-parameter-report-sha256 "${CHEMLLM_7B_PARAMETER_REPORT_SHA256:?}")
[[ -n "${CHEMLLM_20B_METADATA_MANIFEST:-}" ]] && status_args+=(--twenty-b-metadata-manifest "$CHEMLLM_20B_METADATA_MANIFEST" --twenty-b-metadata-manifest-sha256 "${CHEMLLM_20B_METADATA_MANIFEST_SHA256:?}")

"${status_args[@]}"
allowed="$(python -c 'import json,sys; print("1" if json.load(open(sys.argv[1]))["science_launch_allowed"] else "0")' "$decision")"
if [[ "$allowed" != "1" ]]; then
  echo "BLOCKED_MAIN_PRIORITY_OR_RUNTIME_EVIDENCE"
  exit 78
fi
decision_sha="$(sha256sum "$decision" | awk '{print $1}')"
run_args=(
  python scripts/autodl/run_llm_ablation_variant.py
  --config configs/hpc.yaml
  --run-spec "$LLM_CORE_RUN_SPEC"
  --run-spec-sha256 "$LLM_CORE_RUN_SPEC_SHA256"
  --launch-decision "$decision"
  --launch-decision-sha256 "$decision_sha"
)
[[ "${LLM_CORE_RESUME:-0}" == "1" ]] && run_args+=(--resume)
exec "${run_args[@]}"
