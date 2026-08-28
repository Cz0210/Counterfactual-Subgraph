#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/counterfactual-subgraph}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD"

: "${AUTO_TERMINATE_UNCONTROLLED_CHILDREN:?must be explicitly set to 0}"
if [[ "$AUTO_TERMINATE_UNCONTROLLED_CHILDREN" != "0" ]]; then
  echo "T5_CLEAN_BASE_BLOCKED: AUTO_TERMINATE_UNCONTROLLED_CHILDREN must equal 0" >&2
  exit 75
fi
: "${TASTE_T5_ACTION:?inspect, worker, or verify is required}"
: "${TASTE_T5_SOURCE_MODEL:?absolute generic ChemLLM base is required}"

case "$TASTE_T5_ACTION" in
  inspect)
    python scripts/autodl/tastemolnet_t5_clean_base_worker_v2.py inspect \
      --config configs/hpc.yaml \
      --source-model "$TASTE_T5_SOURCE_MODEL"
    ;;
  worker)
    : "${TASTE_T5_STAGE_ROOT:?managed-v2 stage root is required}"
    : "${TASTE_CONTROLLER_ID:?controller id is required}"
    : "${TASTE_EXECUTION_COMMIT:?exact clean execution commit is required}"
    : "${TASTE_T5_SOURCE_INVENTORY_SHA256:?source inventory pin is required}"
    CONFIG_SHA256="$(sha256sum configs/hpc.yaml | awk '{print $1}')"
    python scripts/autodl/managed_worker_v2.py \
      --stage-root "$TASTE_T5_STAGE_ROOT" \
      --controller-id "$TASTE_CONTROLLER_ID" \
      --task-id T5_CLEAN_BASE_ADOPTION \
      --git-commit "$TASTE_EXECUTION_COMMIT" \
      --config-hash "$CONFIG_SHA256" \
      --input-hash "source_model_inventory=$TASTE_T5_SOURCE_INVENTORY_SHA256" \
      --cwd "$PROJECT_ROOT" \
      --config configs/hpc.yaml \
      -- python scripts/autodl/tastemolnet_t5_clean_base_worker_v2.py build \
        --config configs/hpc.yaml \
        --source-model "$TASTE_T5_SOURCE_MODEL" \
        --expected-source-inventory-sha256 "$TASTE_T5_SOURCE_INVENTORY_SHA256"
    ;;
  verify)
    : "${TASTE_T5_SEALED_ROOT:?SEALED worker root is required}"
    : "${TASTE_T5_FINAL_ROOT:?fresh adopted-clean-base-* final root is required}"
    : "${TASTE_T5_ATTEMPT_ID:?attempt UUID is required}"
    : "${TASTE_T5_GENERATION_TOKEN:?generation UUID is required}"
    : "${TASTE_CONTROLLER_ID:?controller id is required}"
    : "${TASTE_EXECUTION_COMMIT:?exact clean execution commit is required}"
    : "${TASTE_T5_SOURCE_INVENTORY_SHA256:?source inventory pin is required}"
    python scripts/autodl/tastemolnet_t5_clean_base_verifier_v2.py \
      --config configs/hpc.yaml \
      --sealed "$TASTE_T5_SEALED_ROOT" \
      --final-path "$TASTE_T5_FINAL_ROOT" \
      --source-model "$TASTE_T5_SOURCE_MODEL" \
      --expected-attempt-id "$TASTE_T5_ATTEMPT_ID" \
      --expected-generation-token "$TASTE_T5_GENERATION_TOKEN" \
      --expected-controller-id "$TASTE_CONTROLLER_ID" \
      --expected-git-commit "$TASTE_EXECUTION_COMMIT" \
      --expected-source-inventory-sha256 "$TASTE_T5_SOURCE_INVENTORY_SHA256"
    ;;
  *)
    echo "T5_CLEAN_BASE_BLOCKED: TASTE_T5_ACTION must be inspect, worker, or verify" >&2
    exit 75
    ;;
esac
