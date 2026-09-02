#!/usr/bin/env bash
# Gate and two-lane schedule emitter. Science execution remains owned by the
# ablation-aware sidecar so this entrypoint cannot create duplicate writers.
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
AUTHORITY=${MATRIX_AUTHORITY_ROOT:-$RUNTIME/control/fast16_matrix_authority}

args=(
  "$PY" "$PROJECT_ROOT/scripts/autodl/status_gnn_five_backbone_ablation_v1.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --ablation-config "$PROJECT_ROOT/configs/ablations/gnn/bace_ours_proposal_fixed_five_backbones_v1.yaml"
  --matrix-authority "$AUTHORITY/state.json"
)
[[ -n "${MAIN_FINAL_AUDIT_RECEIPT:-}" ]] && args+=(--final-audit "$MAIN_FINAL_AUDIT_RECEIPT")
[[ -n "${FINAL_FIGURE3_RECEIPT:-}" ]] && args+=(--figure3-pass "$FINAL_FIGURE3_RECEIPT")
[[ -n "${FINAL_FIGURE4_RECEIPT:-}" ]] && args+=(--figure4-pass "$FINAL_FIGURE4_RECEIPT")
[[ -n "${FINAL_TABLE2_RECEIPT:-}" ]] && args+=(--table2-pass "$FINAL_TABLE2_RECEIPT")
[[ -n "${GNN_ABLATION_AUTHORIZATION_RECEIPT:-}" ]] && args+=(--authorization-receipt "$GNN_ABLATION_AUTHORIZATION_RECEIPT")
[[ -n "${MAIN_READY_GPU_TASKS_RECEIPT:-}" ]] && args+=(--main-ready-gpu-tasks "$MAIN_READY_GPU_TASKS_RECEIPT")
[[ -n "${BACE_OURS_PROPOSAL_FIXED_MANIFEST:-}" ]] && args+=(--proposal-manifest "$BACE_OURS_PROPOSAL_FIXED_MANIFEST")
[[ "${ALLOW_GNN_ABLATION_RUN_AFTER_16:-0}" == 1 ]] && args+=(--allow-after-16)
[[ "${RUN_GNN_ABLATION:-0}" == 1 ]] && args+=(--run-requested)
[[ -n "${GNN_FIVE_BACKBONE_STATUS_OUTPUT:-}" ]] && args+=(--output "$GNN_FIVE_BACKBONE_STATUS_OUTPUT")

exec nice -n 10 "${args[@]}"
