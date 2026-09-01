#!/usr/bin/env bash
# Gate-only launcher. It never starts ablation science by itself.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
AUTHORITY=${MATRIX_AUTHORITY_ROOT:-$RUNTIME/control/fast16_matrix_authority}

args=(
  "$PY" "$PROJECT_ROOT/scripts/autodl/status_gnn_ablation.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --common-config "$PROJECT_ROOT/configs/ablations/common_v1.yaml"
  --family gnn
  --matrix-authority "$AUTHORITY/state.json"
)
[[ -n "${MAIN_FINAL_AUDIT_RECEIPT:-}" ]] && args+=(--final-audit "$MAIN_FINAL_AUDIT_RECEIPT")
[[ -n "${FINAL_FIGURE3_RECEIPT:-}" ]] && args+=(--figure3-pass "$FINAL_FIGURE3_RECEIPT")
[[ -n "${FINAL_FIGURE4_RECEIPT:-}" ]] && args+=(--figure4-pass "$FINAL_FIGURE4_RECEIPT")
[[ -n "${FINAL_TABLE2_RECEIPT:-}" ]] && args+=(--table2-pass "$FINAL_TABLE2_RECEIPT")
[[ -n "${ABLATION_AUTHORIZATION_RECEIPT:-}" ]] && args+=(--authorization-receipt "$ABLATION_AUTHORIZATION_RECEIPT")
[[ "${RUN_GNN_ABLATION:-0}" == 1 ]] && args+=(--run-requested)
[[ -n "${GNN_ABLATION_STATUS_OUTPUT:-}" ]] && args+=(--output "$GNN_ABLATION_STATUS_OUTPUT")

exec nice -n 10 "${args[@]}"
