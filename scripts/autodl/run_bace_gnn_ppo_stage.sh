#!/usr/bin/env bash
# Foreground payload for the persistent AutoDL controller.  The controller,
# not this script, owns GPU locks, retries, nohup, PID files, and heartbeats.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
  echo "usage: $0 ACTION" >&2
  exit 64
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [[ -z "$OUTPUT_ROOT" || "$OUTPUT_ROOT" != /* ]]; then
  echo "OUTPUT_ROOT must be the controller's absolute expected_output" >&2
  exit 64
fi
case "$OUTPUT_ROOT" in
  "$AUTODL_RUNTIME_ROOT"/*) ;;
  *)
    echo "OUTPUT_ROOT must be under persistent AUTODL_RUNTIME_ROOT: $OUTPUT_ROOT" >&2
    exit 64
    ;;
esac
if [[ -d "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "OUTPUT_ROOT must be fresh and empty: $OUTPUT_ROOT" >&2
  exit 73
fi

CHEMLLM_MODEL_PATH="${CHEMLLM_MODEL_PATH:-$AUTODL_DATA_ROOT/incoming/counterfactual-subgraph-autodl-step0-20260820-141726/payload/project/pretrained_models/ChemLLM-7B-Chat}"
BACE_GNN_CHECKPOINT="${BACE_GNN_CHECKPOINT:-$AUTODL_RUNTIME_ROOT/outputs/gnn_oracles/bace/gine/seed7/calibrated-20260821T181039Z-97689}"
BACE_TRAIN_CSV="${BACE_TRAIN_CSV:-$BACE_SPLIT_ROOT/train.csv}"

case "$ACTION" in
  BACE_POLICY_PROVENANCE_AUDIT)
    mkdir -p "$OUTPUT_ROOT"
    AUDIT_CSV="${BACE_POLICY_AUDIT_CSV:-$OUTPUT_ROOT/bace_policy_initializer_provenance.csv}"
    if [[ "$AUDIT_CSV" != /* ]]; then
      echo "BACE_POLICY_AUDIT_CSV must be absolute: $AUDIT_CSV" >&2
      exit 64
    fi
    case "$AUDIT_CSV" in
      "$AUTODL_RUNTIME_ROOT"/*) ;;
      *)
        echo "BACE_POLICY_AUDIT_CSV must be under persistent AUTODL_RUNTIME_ROOT" >&2
        exit 64
        ;;
    esac
    CANDIDATE_ARGS=(--candidate "raw_base=$CHEMLLM_MODEL_PATH")
    LEGACY_UNKNOWN="${BACE_LEGACY_UNKNOWN_INITIALIZER:-$AUTODL_DATA_ROOT/worktrees/bace-tastemolnet-gnn-autodl-861ba55/trainer_output/checkpoint-3}"
    if [[ -e "$LEGACY_UNKNOWN" ]]; then
      CANDIDATE_ARGS+=(--candidate "adapter=$LEGACY_UNKNOWN")
    fi
    if [[ -n "${BACE_POLICY_CANDIDATES:-}" ]]; then
      IFS=',' read -r -a EXTRA_CANDIDATES <<< "$BACE_POLICY_CANDIDATES"
      for candidate in "${EXTRA_CANDIDATES[@]}"; do
        [[ -n "$candidate" ]] && CANDIDATE_ARGS+=(--candidate "$candidate")
      done
    fi
    exec "$AUTODL_PYTHON" "$PROJECT_ROOT/scripts/audit_bace_policy_initializers.py" \
      "${CANDIDATE_ARGS[@]}" \
      --output-csv "$AUDIT_CSV" \
      --selection-json "$OUTPUT_ROOT/initializer_selection.json" \
      --audit-manifest "$OUTPUT_ROOT/audit_manifest.json" \
      --pass-path "$OUTPUT_ROOT/PASS"
    ;;

  BACE_CLEAN_INITIALIZER_BUILD)
    MODE="${BACE_INITIALIZER_MODE:-raw-base}"
    BACE_POLICY_AUDIT_SELECTION="${BACE_POLICY_AUDIT_SELECTION:-}"
    if [[ -z "$BACE_POLICY_AUDIT_SELECTION" ]]; then
      echo "BACE_POLICY_AUDIT_SELECTION is required to reuse the one formal base hash" >&2
      exit 64
    fi
    BUILD_COMMAND=(
      "$AUTODL_PYTHON"
      "$PROJECT_ROOT/scripts/build_bace_clean_policy_initializer.py"
      --audit-selection "$BACE_POLICY_AUDIT_SELECTION"
    )
    if [[ -n "${BACE_SOURCE_MODEL_HASH:-}" ]]; then
      echo "Do not set BACE_SOURCE_MODEL_HASH when BACE_POLICY_AUDIT_SELECTION is used" >&2
      exit 64
    fi
    BUILD_COMMAND+=(
      "$MODE"
      --model-path "$CHEMLLM_MODEL_PATH"
      --output-dir "$OUTPUT_ROOT"
      --seed "${BACE_PPO_SEED:-7}"
    )
    if [[ "$MODE" == "raw-base" ]]; then
      exec "${BUILD_COMMAND[@]}"
    elif [[ "$MODE" == "oracle-neutral-sft" ]]; then
      BUILD_COMMAND+=(
        --train-csv "$BACE_TRAIN_CSV"
        --gnn-checkpoint "$BACE_GNN_CHECKPOINT"
        --max-steps "${BACE_POLICY_SFT_MAX_STEPS:-100}"
        --max-parents "${BACE_POLICY_SFT_MAX_PARENTS:-0}"
      )
      exec "${BUILD_COMMAND[@]}"
    else
      echo "Unsupported BACE_INITIALIZER_MODE: $MODE" >&2
      exit 64
    fi
    ;;

  BACE_GNN_PPO_ADAPTER_CANARY|B6_PPO_SMOKE_V2|B7_PPO_FULL)
    BACE_POLICY_INITIALIZER="${BACE_POLICY_INITIALIZER:-}"
    BACE_POLICY_PROVENANCE_MANIFEST="${BACE_POLICY_PROVENANCE_MANIFEST:-}"
    if [[ -z "$BACE_POLICY_INITIALIZER" || -z "$BACE_POLICY_PROVENANCE_MANIFEST" ]]; then
      echo "BACE_POLICY_INITIALIZER and BACE_POLICY_PROVENANCE_MANIFEST are required" >&2
      exit 64
    fi
    PPO_COMMAND=(
      "$AUTODL_PYTHON"
      "$PROJECT_ROOT/scripts/train_bace_gnn_ppo.py"
    )
    if [[ -n "${BACE_PPO_CONFIG:-}" ]]; then
      PPO_COMMAND+=(--config "$BACE_PPO_CONFIG")
    fi
    PPO_COMMAND+=(
      --stage "$ACTION"
      --model-path "$CHEMLLM_MODEL_PATH"
      --dataset-path "$BACE_TRAIN_CSV"
      --output-dir "$OUTPUT_ROOT"
      --gnn-checkpoint "$BACE_GNN_CHECKPOINT"
      --gnn-device "${BACE_GNN_DEVICE:-cuda}"
      --oracle-batch-size "${BACE_GNN_REWARD_BATCH_SIZE:-256}"
      --policy-initializer "$BACE_POLICY_INITIALIZER"
      --policy-provenance-manifest "$BACE_POLICY_PROVENANCE_MANIFEST"
      --batch-size "${BACE_PPO_BATCH_SIZE:-2}"
      --seed "${BACE_PPO_SEED:-7}"
    )
    if [[ "$ACTION" == "B6_PPO_SMOKE_V2" ]]; then
      PPO_COMMAND+=(
        --b6-updates "${BACE_B6_UPDATES:-5}"
        --b6-parent-count "${BACE_B6_PARENT_COUNT:-16}"
      )
    elif [[ "$ACTION" == "B7_PPO_FULL" ]]; then
      BACE_B6_V2_MANIFEST="${BACE_B6_V2_MANIFEST:-}"
      if [[ -z "$BACE_B6_V2_MANIFEST" ]]; then
        echo "BACE_B6_V2_MANIFEST is required for B7_PPO_FULL" >&2
        exit 64
      fi
      PPO_COMMAND+=(--b6-v2-manifest "$BACE_B6_V2_MANIFEST")
    fi
    exec "${PPO_COMMAND[@]}"
    ;;

  *)
    echo "Unsupported BACE GNN PPO action: $ACTION" >&2
    exit 64
    ;;
esac
