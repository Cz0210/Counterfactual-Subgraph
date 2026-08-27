#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

if [[ "${RUN_TASTEMOLNET:-0}" != "1" ]]; then
  echo "RUN_TASTEMOLNET=1 is required by the active Taste downstream policy" >&2
  exit 64
fi

STAGE_SCRIPT="$PROJECT_ROOT/scripts/autodl/tastemolnet_gnn_stage.py"
BASE_POLICY="$PROJECT_ROOT/configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"
DOWNSTREAM_POLICY="$PROJECT_ROOT/configs/data_usage/tastemolnet_downstream_research_no_redistribution_v1.json"
SOURCE_CHECKPOINT="${TASTEMOLNET_T2_BUNDLE:?Set TASTEMOLNET_T2_BUNDLE to the verified immutable T2 bundle}"
T2_ADOPTION_ROOT="${TASTEMOLNET_T2_ADOPTION_ROOT:?Set TASTEMOLNET_T2_ADOPTION_ROOT to the fresh adopted PASS root}"
T2_ADOPTION_GATE_SHA256="${TASTEMOLNET_T2_ADOPTION_GATE_SHA256:?Set TASTEMOLNET_T2_ADOPTION_GATE_SHA256 to the reviewed gate digest}"
T2_ADOPTION_RECEIPT_SHA256="${TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256:?Set TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256 to the reviewed receipt digest}"
T2_SOURCE_EVIDENCE_SHA256="${TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256:?Set TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256 to the reviewed source-evidence digest}"
GRAPH_CACHE_ROOT="${TASTEMOLNET_GRAPH_CACHE_ROOT:?Set TASTEMOLNET_GRAPH_CACHE_ROOT to the immutable graph-cache root}"
for required in "$STAGE_SCRIPT" "$BASE_POLICY" "$DOWNSTREAM_POLICY" "$SOURCE_CHECKPOINT/sha256sums.txt" "$T2_ADOPTION_ROOT/gate.json" "$T2_ADOPTION_ROOT/manifest.json" "$GRAPH_CACHE_ROOT/manifest.json"; do
  autodl_require_file "$required"
done

OUTPUT_DIR="${TASTEMOLNET_T3_OUTPUT:-$(autodl_new_output_dir tastemolnet gine calibrated)}"

exec "$AUTODL_PYTHON" "$SCRIPT_DIR/exp_run.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  launch \
  --dataset tastemolnet \
  --stage T3_GINE_CALIBRATED \
  --max-gpus 4 \
  --gpu-hard-limit 4 \
  --config-file "$PROJECT_ROOT/configs/hpc.yaml" \
  --config-file "$PROJECT_ROOT/configs/gnn/gine.yaml" \
  --config-file "$PROJECT_ROOT/configs/autodl/tastemolnet_gine_research_v1.yaml" \
  --config-file "$DOWNSTREAM_POLICY" \
  --input-manifest "$T2_ADOPTION_ROOT/manifest.json" \
  --expected-output "$OUTPUT_DIR" \
  --required-output-file calibration_adoption.json \
  --required-output-file oracle_reference.json \
  --required-output-file policy_binding.json \
  --required-output-file gate.json \
  --required-output-file sha256sums.txt \
  --required-output-file TASTE_GINE_CALIBRATION_PASS \
  --required-log-marker "[TASTE_GINE_CALIBRATION_PASS]" \
  -- \
  "$AUTODL_PYTHON" -B "$STAGE_SCRIPT" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    t3-adopt \
    --checkpoint-dir "$SOURCE_CHECKPOINT" \
    --t2-adoption-root "$T2_ADOPTION_ROOT" \
    --t2-adoption-gate-sha256 "$T2_ADOPTION_GATE_SHA256" \
    --t2-adoption-receipt-sha256 "$T2_ADOPTION_RECEIPT_SHA256" \
    --t2-source-evidence-sha256 "$T2_SOURCE_EVIDENCE_SHA256" \
    --graph-cache-root "$GRAPH_CACHE_ROOT" \
    --artifact-root "$AUTODL_ARTIFACT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --downstream-policy "$DOWNSTREAM_POLICY" \
    --base-policy "$BASE_POLICY"
