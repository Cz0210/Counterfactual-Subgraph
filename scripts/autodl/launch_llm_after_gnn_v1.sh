#!/usr/bin/env bash
set -euo pipefail
gnn_project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$gnn_project_root"
: "${GNN_SEED7_EVALUATION_ROOT:?verified imported GNN seed7 root required}"
: "${ABLATION_MAIN_RESOURCE_EVIDENCE:?fresh main/GPU reservation evidence required}"
"${AUTODL_PYTHON:-python}" scripts/autodl/status_llm_after_gnn_v1.py \
  --gnn-evaluation-root "$GNN_SEED7_EVALUATION_ROOT" \
  --main-resource-evidence "$ABLATION_MAIN_RESOURCE_EVIDENCE" --require-pass
# The existing entrypoint independently rechecks exact model/run-spec/live owner
# evidence and acquires its own lease; a copied resource snapshot is insufficient.
exec bash scripts/autodl/launch_llm_ablation_core_v1.sh
