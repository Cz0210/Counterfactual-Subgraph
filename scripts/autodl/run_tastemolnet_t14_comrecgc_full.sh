#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

[[ "${RUN_TASTEMOLNET:-0}" == "1" ]] || { echo "RUN_TASTEMOLNET=1 is required" >&2; exit 64; }
[[ "${TASTE_RESEARCH_COMPUTE_ALLOWED:-0}" == "1" ]] || { echo "Taste research compute is not authorized" >&2; exit 64; }
[[ "${TASTE_PAPER_RESULTS_ALLOWED:-0}" == "1" ]] || { echo "Taste aggregate reporting is not authorized" >&2; exit 64; }
[[ "${TASTE_DATA_REDISTRIBUTION_ALLOWED:-1}" == "0" ]] || { echo "Taste redistribution must remain forbidden" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "GNN ablation is disabled before 16/16" >&2; exit 64; }

for variable in \
  TASTEMOLNET_T14_OUTPUT TASTEMOLNET_T14_RUN_ID \
  TASTEMOLNET_T2_ADOPTION_ROOT TASTEMOLNET_T2_ADOPTION_GATE_SHA256 \
  TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256 TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256 \
  TASTEMOLNET_T3_OUTPUT_ROOT TASTEMOLNET_T4_OUTPUT_ROOT \
  TASTEMOLNET_TRAIN_CSV COMRECGC_OFFICIAL_ROOT; do
  [[ -n "${!variable:-}" ]] || { echo "$variable is required" >&2; exit 64; }
done

[[ ! -e "$TASTEMOLNET_T14_OUTPUT" && ! -L "$TASTEMOLNET_T14_OUTPUT" ]] \
  || { echo "T14 requires a fresh absent output root" >&2; exit 64; }
install -d -m 700 "$(dirname "$TASTEMOLNET_T14_OUTPUT")"

GPU_JSON="$(
  "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_inventory.py" \
    --project-root "$PROJECT_ROOT" --data-root "$AUTODL_DATA_ROOT" \
    --max-gpus 4 --gpu-hard-limit 4 \
    --min-free-memory-mb "$AUTODL_MIN_FREE_MEMORY_MB" \
    --idle-util-threshold "$AUTODL_IDLE_UTIL_THRESHOLD" \
    --stable-seconds "$AUTODL_IDLE_STABLE_SECONDS" --format json
)"
GPU_UUID="$(printf '%s' "$GPU_JSON" | "$AUTODL_PYTHON" -c '
import json, sys
rows = [r for r in json.load(sys.stdin)["gpus"] if r["index"] == 1 and r["stable_idle"]]
if len(rows) != 1: raise SystemExit(75)
print(rows[0]["uuid"])
')" || { rc=$?; [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU1_FOR_T14" >&2; exit "$rc"; }

exec "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_lock.py" \
  --project-root "$PROJECT_ROOT" --data-root "$AUTODL_DATA_ROOT" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" run \
  --gpu-index 1 --gpu-uuid "$GPU_UUID" --run-id "$TASTEMOLNET_T14_RUN_ID" -- \
  "$AUTODL_PYTHON" -I -B "$PROJECT_ROOT/scripts/run_tastemolnet_comrecgc_full.py" \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --output-dir "$TASTEMOLNET_T14_OUTPUT" \
    --run-id "$TASTEMOLNET_T14_RUN_ID" --gpu-uuid "$GPU_UUID" \
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
