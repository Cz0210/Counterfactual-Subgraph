#!/usr/bin/env bash
# Gate-only wrapper: it never acquires a GPU or starts model science itself.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}

: "${LLM_ABLATION_MAIN_SNAPSHOT:?set LLM_ABLATION_MAIN_SNAPSHOT to current controller evidence}"

args=(
  "$PY" "$PROJECT_ROOT/scripts/autodl/status_llm_ablation_v2.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --main-snapshot "$LLM_ABLATION_MAIN_SNAPSHOT"
)
[[ -n "${LLM_ABLATION_EARLY_RUN_RECEIPT:-}" ]] && args+=(--early-run-receipt "$LLM_ABLATION_EARLY_RUN_RECEIPT")
[[ -n "${LLM_ABLATION_STATUS_OUTPUT:-}" ]] && args+=(--output "$LLM_ABLATION_STATUS_OUTPUT")

exec nice -n 10 "${args[@]}"

