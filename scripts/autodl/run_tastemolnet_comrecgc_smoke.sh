#!/usr/bin/env bash
set -euo pipefail

# Current T9 authority is intentionally narrow: the project owner runs one
# foreground chain as TRUSTED_SINGLE_OPERATOR_ROOT after T4. The standard UUID
# GPU lock still covers worker, SEALED handoff, verifier, and atomic publish.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "${RUN_TASTEMOLNET:-0}" == "1" ]] \
  || { echo "RUN_TASTEMOLNET=1 is required" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-0}" == "1" ]] \
  || { echo "Taste research compute is not authorized" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-0}" == "1" ]] \
  || { echo "Taste aggregate reporting is not authorized" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-1}" == "0" ]] \
  || { echo "Taste data redistribution must remain forbidden" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] \
  || { echo "GNN ablation is outside the Taste main route" >&2; exit 64; }

for variable in \
  TASTEMOLNET_T9_STAGE_ROOT \
  TASTEMOLNET_T9_OUTPUT \
  TASTEMOLNET_T9_RUN_ID \
  TASTEMOLNET_T2_ADOPTION_ROOT \
  TASTEMOLNET_T2_ADOPTION_GATE_SHA256 \
  TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256 \
  TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256 \
  TASTEMOLNET_T3_OUTPUT_ROOT \
  TASTEMOLNET_T4_OUTPUT_ROOT \
  TASTEMOLNET_TRAIN_CSV \
  COMRECGC_OFFICIAL_ROOT; do
  [[ -n "${!variable:-}" ]] \
    || { echo "$variable is required" >&2; exit 64; }
done

[[ -d "$TASTEMOLNET_T4_OUTPUT_ROOT" ]] \
  || { echo "T9 requires the completed T4 predecessor" >&2; exit 64; }
[[ ! -e "$TASTEMOLNET_T9_OUTPUT" && ! -L "$TASTEMOLNET_T9_OUTPUT" ]] \
  || { echo "T9 final output must be one fresh absent path" >&2; exit 64; }
install -d -m 700 "$TASTEMOLNET_T9_STAGE_ROOT"
install -d -m 700 "$(dirname "$TASTEMOLNET_T9_OUTPUT")"

GPU_JSON="$(
  "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$AUTODL_DATA_ROOT" \
    --max-gpus 4 \
    --gpu-hard-limit 4 \
    --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
    --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
    --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" \
    --format json
)"
GPU_UUID="$(
  printf '%s' "$GPU_JSON" | "$AUTODL_PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
rows = [row for row in payload["gpus"] if row["index"] == 1 and row["stable_idle"]]
if len(rows) != 1:
    raise SystemExit(75)
print(rows[0]["uuid"])
'
)" || {
  rc=$?
  [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU1_AFTER_T4" >&2
  exit "$rc"
}
[[ "$GPU_UUID" == GPU-* ]] \
  || { echo "T9 physical GPU1 UUID binding failed" >&2; exit 64; }

RUNNER="$PROJECT_ROOT/scripts/autodl/tastemolnet_t9_managed_runner_v2.py"
autodl_require_file "$RUNNER"
autodl_require_file "$PROJECT_ROOT/scripts/autodl/gpu_lock.py"
autodl_require_file "$PROJECT_ROOT/configs/hpc.yaml"

exec "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_lock.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  run \
  --gpu-index 1 \
  --gpu-uuid "$GPU_UUID" \
  --run-id "$TASTEMOLNET_T9_RUN_ID" \
  -- \
  "$AUTODL_PYTHON" -I -B "$RUNNER" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --stage-root "$TASTEMOLNET_T9_STAGE_ROOT" \
    --final-path "$TASTEMOLNET_T9_OUTPUT" \
    --run-id "$TASTEMOLNET_T9_RUN_ID" \
    --gpu-uuid "$GPU_UUID" \
    --t2-adoption-root "$TASTEMOLNET_T2_ADOPTION_ROOT" \
    --t2-adoption-gate-sha256 "$TASTEMOLNET_T2_ADOPTION_GATE_SHA256" \
    --t2-adoption-receipt-sha256 "$TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256" \
    --t2-source-evidence-sha256 "$TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256" \
    --t3-output-root "$TASTEMOLNET_T3_OUTPUT_ROOT" \
    --t4-output-root "$TASTEMOLNET_T4_OUTPUT_ROOT" \
    --checkpoint-dir "$TASTEMOLNET_T3_OUTPUT_ROOT/artifacts/checkpoint" \
    --train-csv "$TASTEMOLNET_TRAIN_CSV" \
    --official-root "$COMRECGC_OFFICIAL_ROOT" \
    --set inference.fallback_to_heuristic=false
