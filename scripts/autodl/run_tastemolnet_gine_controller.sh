#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
export AUTODL_MAX_GPUS=4
[[ "$PRIMARY_GNN_BACKBONE" == "gine" ]] || { echo "Taste controller freezes PRIMARY_GNN_BACKBONE=gine" >&2; exit 64; }
[[ "$PRIMARY_SEED" == "7" ]] || { echo "Taste controller freezes PRIMARY_SEED=7" >&2; exit 64; }
MIN_PERSISTENT_FREE_GB="${MIN_PERSISTENT_FREE_GB:-20}"
[[ "$MIN_PERSISTENT_FREE_GB" =~ ^[0-9]+$ ]] && (( MIN_PERSISTENT_FREE_GB >= 20 )) \
  || { echo "Taste controller requires MIN_PERSISTENT_FREE_GB>=20" >&2; exit 64; }
export MIN_PERSISTENT_FREE_GB
[[ "${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" == ":4096:8" ]] \
  || { echo "Taste controller freezes CUBLAS_WORKSPACE_CONFIG=:4096:8" >&2; exit 64; }
[[ "${PYTHONHASHSEED:-7}" == "7" ]] \
  || { echo "Taste controller freezes PYTHONHASHSEED=7" >&2; exit 64; }
[[ "${NVIDIA_TF32_OVERRIDE:-0}" == "0" ]] \
  || { echo "Taste controller freezes NVIDIA_TF32_OVERRIDE=0" >&2; exit 64; }
[[ "${CUDNN_DETERMINISTIC:-1}" == "1" ]] \
  || { echo "Taste controller freezes CUDNN_DETERMINISTIC=1" >&2; exit 64; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export NVIDIA_TF32_OVERRIDE=0
export CUDNN_DETERMINISTIC=1

[[ "$RUN_TASTEMOLNET" == "1" ]] || { echo "RUN_TASTEMOLNET must be 1" >&2; exit 64; }
[[ "${TASTE_UPSTREAM_LICENSE_STATUS:-}" == "NOT_EXPLICITLY_STATED" ]] || { echo "Taste upstream status must remain NOT_EXPLICITLY_STATED" >&2; exit 64; }
: "${TASTEMOLNET_GINE_CONTROLLER_CID:?TASTEMOLNET_GINE_CONTROLLER_CID is required}"
: "${TASTEMOLNET_GINE_CONTROLLER_ROOT:?TASTEMOLNET_GINE_CONTROLLER_ROOT is required}"
: "${TASTEMOLNET_GNN_FULL_OUTPUT:?TASTEMOLNET_GNN_FULL_OUTPUT is required}"
: "${TASTEMOLNET_GNN_TRAINING_STATE_ROOT:?TASTEMOLNET_GNN_TRAINING_STATE_ROOT is required}"
export TASTEMOLNET_PUBLISHED_OUTPUT_ADOPTION_RECEIPT="$TASTEMOLNET_GINE_CONTROLLER_ROOT/published_output_resume_adoption.json"

RESUME_ARGS=()
if [[ -e "$TASTEMOLNET_GINE_CONTROLLER_ROOT" ]]; then
  [[ -d "$TASTEMOLNET_GINE_CONTROLLER_ROOT" && ! -L "$TASTEMOLNET_GINE_CONTROLLER_ROOT" ]] \
    || { echo "Taste controller root is not one physical directory" >&2; exit 64; }
  RESUME_ARGS=(--resume-controller)
fi

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/run_tastemolnet_gine_controller.py" run \
  --cid "$TASTEMOLNET_GINE_CONTROLLER_CID" \
  --controller-root "$TASTEMOLNET_GINE_CONTROLLER_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$TASTEMOLNET_GNN_FULL_OUTPUT" \
  --training-state-root "$TASTEMOLNET_GNN_TRAINING_STATE_ROOT" \
  --worker-wrapper "$SCRIPT_DIR/run_tastemolnet_gnn_full.sh" \
  --poll-seconds "${TASTEMOLNET_GINE_CONTROLLER_POLL_SECONDS:-30}" \
  --terminal-stability-seconds "${TASTEMOLNET_GINE_TERMINAL_STABILITY_SECONDS:-2}" \
  "${RESUME_ARGS[@]}"
