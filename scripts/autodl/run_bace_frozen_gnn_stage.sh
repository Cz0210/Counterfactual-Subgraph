#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

STAGE="${1:-${BACE_STAGE:-}}"
DRIVER="$PROJECT_ROOT/scripts/autodl/bace_frozen_gnn_route.py"
HPC_CONFIG="$PROJECT_ROOT/configs/hpc.yaml"
GNN_CONFIG="$PROJECT_ROOT/configs/gnn/${PRIMARY_GNN_BACKBONE}.yaml"
DATASET_CONFIG="$PROJECT_ROOT/configs/datasets/bace_gnn.yaml"
AUTODL_CONFIG="$PROJECT_ROOT/configs/autodl/bace_gine.yaml"
for required in "$DRIVER" "$HPC_CONFIG" "$GNN_CONFIG" "$DATASET_CONFIG" "$AUTODL_CONFIG"; do
  autodl_require_file "$required"
done

case "$STAGE" in
  B6_PPO_SMOKE|B7_PPO_FULL|B8_POOL_BASE|B9_POOL_HIGHTEMP|B10_POOL_MERGED|B11_CROSS_PARENT_VERIFIED|B12_SELECTOR|B13_FINAL_EVAL|B14_FROZEN)
    ;;
  *)
    echo "Usage: $0 B6_PPO_SMOKE|B7_PPO_FULL|B8_POOL_BASE|B9_POOL_HIGHTEMP|B10_POOL_MERGED|B11_CROSS_PARENT_VERIFIED|B12_SELECTOR|B13_FINAL_EVAL|B14_FROZEN" >&2
    exit 2
    ;;
esac

CHECKPOINT="$({
  autodl_passed_stage_output B4_GNN_CALIBRATED \
    --required-output-file model.pt \
    --required-output-file model_card.json \
    --required-output-file temperature_scaling.json \
    --required-output-file sha256sums.txt
})"

route_output_dir() {
  local stage_slug stamp
  stage_slug="$(printf '%s' "$STAGE" | tr '[:upper:]_' '[:lower:]-')"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  printf '%s\n' "$AUTODL_ARTIFACT_ROOT/bace/ours_gnn_${PRIMARY_GNN_BACKBONE}/${stage_slug}-${stamp}"
}

OUTPUT_DIR="${BACE_FROZEN_GNN_STAGE_OUTPUT:-$(route_output_dir)}"

COMMON_LAUNCH_ARGS=(
  --project-root "$PROJECT_ROOT"
  --data-root "$AUTODL_DATA_ROOT"
  launch
  --dataset bace
  --stage "$STAGE"
  --config-file "$HPC_CONFIG"
  --config-file "$GNN_CONFIG"
  --config-file "$DATASET_CONFIG"
  --config-file "$AUTODL_CONFIG"
  --expected-output "$OUTPUT_DIR"
)

COMMON_DRIVER_ARGS=(
  --config "$HPC_CONFIG"
  --config "$GNN_CONFIG"
  --config "$DATASET_CONFIG"
  --config "$AUTODL_CONFIG"
)

if [[ "$STAGE" == "B6_PPO_SMOKE" ]]; then
  PREDECESSOR="$({
    autodl_passed_stage_output B5_ORACLE_SMOKE \
      --required-output-file oracle_smoke.json \
      --required-output-file deletion_records.jsonl
  })"
  GPU_LINE="$(autodl_select_one_gpu)" || exit $?
  IFS=$'\t' read -r GPU_INDEX GPU_UUID <<< "$GPU_LINE"
  if [[ -z "${GPU_INDEX:-}" || -z "${GPU_UUID:-}" ]]; then
    echo "WAITING_FOR_IDLE_GPU" >&2
    exit 75
  fi
  exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
    "${COMMON_LAUNCH_ARGS[@]}" \
    --gpu-index "$GPU_INDEX" \
    --gpu-uuid "$GPU_UUID" \
    --gpu-required \
    --input-manifest "$PREDECESSOR/oracle_smoke.json" \
    --required-output-file b6_scoring_preflight.json \
    --required-output-file scored_candidates.jsonl \
    --required-output-file oracle_provenance.json \
    --required-output-file legacy_route_audit.json \
    --required-output-file blocker.json \
    --required-output-file stage_requirements.json \
    --required-log-marker "[BACE_GNN_STAGE_BLOCKED]" \
    -- \
    "$AUTODL_PYTHON" "$DRIVER" \
      "${COMMON_DRIVER_ARGS[@]}" \
      scoring-preflight \
      --checkpoint-dir "$CHECKPOINT" \
      --oracle-smoke-dir "$PREDECESSOR" \
      --output-dir "$OUTPUT_DIR" \
      --device cuda:0 \
      --batch-size 32 \
      --max-records 32
fi

PREDECESSOR_STAGE=""
PREDECESSOR_REQUIRED=()
EXTRA_DRIVER_ARGS=()
case "$STAGE" in
  B7_PPO_FULL)
    PREDECESSOR_STAGE=B6_PPO_SMOKE
    PREDECESSOR_REQUIRED=(ppo_smoke_manifest.json oracle_provenance.json)
    ;;
  B8_POOL_BASE)
    PREDECESSOR_STAGE=B7_PPO_FULL
    PREDECESSOR_REQUIRED=(ppo_manifest.json oracle_provenance.json)
    ;;
  B9_POOL_HIGHTEMP)
    PREDECESSOR_STAGE=B8_POOL_BASE
    PREDECESSOR_REQUIRED=(candidate_pool.jsonl pool_manifest.json)
    ;;
  B10_POOL_MERGED)
    PREDECESSOR_STAGE=B9_POOL_HIGHTEMP
    PREDECESSOR_REQUIRED=(candidate_pool.jsonl pool_manifest.json)
    BASE_POOL="$({
      autodl_passed_stage_output B8_POOL_BASE \
        --required-output-file candidate_pool.jsonl \
        --required-output-file pool_manifest.json
    })"
    EXTRA_DRIVER_ARGS+=(--base-pool-output "$BASE_POOL")
    ;;
  B11_CROSS_PARENT_VERIFIED)
    PREDECESSOR_STAGE=B10_POOL_MERGED
    PREDECESSOR_REQUIRED=(candidate_pool.jsonl merge_manifest.json)
    ;;
  B12_SELECTOR)
    PREDECESSOR_STAGE=B11_CROSS_PARENT_VERIFIED
    PREDECESSOR_REQUIRED=(matrix_manifest.json oracle_provenance.json)
    ;;
  B13_FINAL_EVAL)
    PREDECESSOR_STAGE=B12_SELECTOR
    PREDECESSOR_REQUIRED=(selected_top20.json frozen_selection_manifest.json)
    autodl_require_file "$BACE_SPLIT_ROOT/test.csv"
    EXTRA_DRIVER_ARGS+=(--test-csv "$BACE_SPLIT_ROOT/test.csv")
    ;;
  B14_FROZEN)
    PREDECESSOR_STAGE=B13_FINAL_EVAL
    PREDECESSOR_REQUIRED=(final_metrics.json test_evaluation_manifest.json)
    ;;
esac

STAGE_OUTPUT_ARGS=()
for required in "${PREDECESSOR_REQUIRED[@]}"; do
  STAGE_OUTPUT_ARGS+=(--required-output-file "$required")
done
PREDECESSOR="$({
  autodl_passed_stage_output "$PREDECESSOR_STAGE" "${STAGE_OUTPUT_ARGS[@]}"
})"

if [[ -n "${MOLCLR_CHECKPOINT:-}" ]]; then
  autodl_require_file "$MOLCLR_CHECKPOINT"
  EXTRA_DRIVER_ARGS+=(--molclr-checkpoint "$MOLCLR_CHECKPOINT")
fi

INPUT_MANIFEST="$PREDECESSOR/${PREDECESSOR_REQUIRED[0]}"
exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  "${COMMON_LAUNCH_ARGS[@]}" \
  --input-manifest "$INPUT_MANIFEST" \
  --required-output-file blocker.json \
  --required-output-file stage_requirements.json \
  --required-output-file oracle_provenance.json \
  --required-log-marker "[BACE_GNN_STAGE_BLOCKED]" \
  -- \
  "$AUTODL_PYTHON" "$DRIVER" \
    "${COMMON_DRIVER_ARGS[@]}" \
    stage-blocker \
    --stage "$STAGE" \
    --checkpoint-dir "$CHECKPOINT" \
    --predecessor-output "$PREDECESSOR" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_DRIVER_ARGS[@]}"
