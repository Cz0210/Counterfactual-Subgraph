#!/usr/bin/env bash
set -euo pipefail

# Fresh T7 successor: validate the typed release before touching GPU state,
# then keep one physical-GPU0 UUID lock across worker, SEALED handoff, verifier,
# and terminal publication.
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
  || { echo "GNN ablation is outside the T7 main route" >&2; exit 64; }
[[ "${ALLOW_T7_TYPED_RELEASE_FROM_ADOPTED_NEUROSED:-0}" == "1" ]] \
  || { echo "typed T7 release authorization is required" >&2; exit 64; }

for variable in \
  TASTEMOLNET_T7_RELEASE_ROOT \
  TASTEMOLNET_T7_STAGE_ROOT \
  TASTEMOLNET_T7_OUTPUT \
  TASTEMOLNET_T7_RUN_ID; do
  [[ -n "${!variable:-}" ]] \
    || { echo "$variable is required" >&2; exit 64; }
done

RELEASE_CLI="$PROJECT_ROOT/scripts/autodl/tastemolnet_t7_typed_release_v1.py"
RUNNER="$PROJECT_ROOT/scripts/autodl/tastemolnet_t7_managed_runner_v3.py"
autodl_require_file "$RELEASE_CLI"
autodl_require_file "$RUNNER"
autodl_require_file "$PROJECT_ROOT/scripts/autodl/gpu_lock.py"
autodl_require_file "$PROJECT_ROOT/configs/hpc.yaml"

[[ ! -e "$TASTEMOLNET_T7_OUTPUT" && ! -L "$TASTEMOLNET_T7_OUTPUT" ]] \
  || { echo "T7 final output must be one fresh absent path" >&2; exit 64; }
install -d -m 700 "$TASTEMOLNET_T7_STAGE_ROOT"
install -d -m 700 "$(dirname "$TASTEMOLNET_T7_OUTPUT")"

# Fail closed on stale/missing pins and all source hashes before querying GPU0.
"$AUTODL_PYTHON" -I -B "$RELEASE_CLI" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  validate \
  --release-root "$TASTEMOLNET_T7_RELEASE_ROOT"

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
rows = [row for row in payload["gpus"] if row["index"] == 0 and row["stable_idle"]]
if len(rows) != 1:
    raise SystemExit(75)
print(rows[0]["uuid"])
'
)" || {
  rc=$?
  [[ $rc -ne 75 ]] || echo "WAITING_FOR_IDLE_GPU0_FOR_T7" >&2
  exit "$rc"
}
[[ "$GPU_UUID" == GPU-* ]] \
  || { echo "T7 physical GPU0 UUID binding failed" >&2; exit 64; }

exec "$AUTODL_PYTHON" -B "$PROJECT_ROOT/scripts/autodl/gpu_lock.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$AUTODL_DATA_ROOT" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  run \
  --gpu-index 0 \
  --gpu-uuid "$GPU_UUID" \
  --run-id "$TASTEMOLNET_T7_RUN_ID" \
  -- \
  "$AUTODL_PYTHON" -I -B "$RUNNER" \
    --mode run \
    --config "$PROJECT_ROOT/configs/hpc.yaml" \
    --release-root "$TASTEMOLNET_T7_RELEASE_ROOT" \
    --stage-root "$TASTEMOLNET_T7_STAGE_ROOT" \
    --final-path "$TASTEMOLNET_T7_OUTPUT" \
    --run-id "$TASTEMOLNET_T7_RUN_ID" \
    --gpu-uuid "$GPU_UUID" \
    --set inference.fallback_to_heuristic=false
