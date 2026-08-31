#!/usr/bin/env bash
# Wait for the exact BACE ComRecGC resource-cap handover, then launch its
# already-reviewed postprocess manifest exactly once.  This sidecar is CPU-only
# and owns no science process or GPU lock.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${BACE_CAP_SOURCE_FRAGMENT:?set the executor postprocess.tasks.json path}"
: "${BACE_CAP_QUEUE_ROOT:?set one fresh persistent sidecar root}"
: "${BACE_CAP_CONTROLLER_ID:?set one fresh controller ID}"

POLL_SECONDS="${BACE_CAP_POLL_SECONDS:-60}"
if [[ "$BACE_CAP_SOURCE_FRAGMENT" != /* || "$BACE_CAP_QUEUE_ROOT" != /* ]]; then
  echo "BACE cap sidecar paths must be absolute" >&2
  exit 64
fi
if ! [[ "$BACE_CAP_CONTROLLER_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Unsafe BACE cap postprocess controller ID" >&2
  exit 64
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "BACE_CAP_POLL_SECONDS must be a positive integer" >&2
  exit 64
fi

mkdir -p "$BACE_CAP_QUEUE_ROOT"
STATUS="$BACE_CAP_QUEUE_ROOT/status.json"
GENERIC="$BACE_CAP_QUEUE_ROOT/postprocess.generic.tasks.json"
MANIFEST="$BACE_CAP_QUEUE_ROOT/postprocess.manifest.json"
LAUNCH_LOG="$BACE_CAP_QUEUE_ROOT/controller-launch.log"
EXECUTOR_STATE="$(dirname "$BACE_CAP_SOURCE_FRAGMENT")/state.json"

write_status() {
  local state="$1" detail="$2"
  "$AUTODL_PYTHON" - "$STATUS" "$state" "$detail" "$$" \
    "$BACE_CAP_CONTROLLER_ID" "$BACE_CAP_SOURCE_FRAGMENT" "$MANIFEST" <<'PY'
import json, os, pathlib, sys, tempfile
from datetime import datetime, timezone

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": "bace_comrecgc_cap_postprocess_sidecar_v1",
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "pid": int(sys.argv[4]),
    "controller_id": sys.argv[5],
    "source_fragment": sys.argv[6],
    "manifest": sys.argv[7],
    "gnn_ablation_started": False,
    "heartbeat_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    try:
        os.unlink(name)
    except FileNotFoundError:
        pass
PY
}

on_error() {
  local rc=$?
  trap - ERR
  write_status "FAILED" "line=${BASH_LINENO[0]:-unknown};rc=$rc"
  exit "$rc"
}
trap on_error ERR

if [[ -s "$BACE_CAP_QUEUE_ROOT/LAUNCHED" ]]; then
  write_status "ALREADY_LAUNCHED" "immutable_launch_receipt_present"
  exit 0
fi
if [[ -e "$GENERIC" || -e "$MANIFEST" ]]; then
  write_status "BLOCKED" "partial_fresh_manifest_namespace_exists"
  exit 2
fi

while [[ ! -s "$BACE_CAP_SOURCE_FRAGMENT" ]]; do
  if [[ -s "$EXECUTOR_STATE" ]]; then
    executor_state="$($AUTODL_PYTHON - "$EXECUTOR_STATE" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("state", "UNKNOWN"))
PY
)"
    case "$executor_state" in
      SCIENTIFIC_FAILED_AT_ABSOLUTE_CAP|SIGTERM_TIMEOUT|FAILED|BLOCKED)
        write_status "BLOCKED" "executor_state=$executor_state"
        exit 2
        ;;
    esac
  else
    executor_state="MISSING"
  fi
  write_status "WAITING_FOR_POSTPROCESS_QUEUE" "executor_state=$executor_state"
  sleep "$POLL_SECONDS"
done

write_status "PREPARING_MANIFEST" "source_fragment_ready"
"$AUTODL_PYTHON" "$PROJECT_ROOT/scripts/autodl/prepare_bace_comrecgc_resource_cap_postprocess.py" \
  --config "$PROJECT_ROOT/configs/hpc.yaml" \
  --set inference.fallback_to_heuristic=false \
  --source-fragment "$BACE_CAP_SOURCE_FRAGMENT" \
  --generic-fragment-output "$GENERIC" \
  --manifest-output "$MANIFEST" \
  --controller-id "$BACE_CAP_CONTROLLER_ID" \
  >"$BACE_CAP_QUEUE_ROOT/prepare.log" 2>&1

write_status "LAUNCHING_CONTROLLER" "manifest_preparation_passed"
RUN_TASTEMOLNET=0 \
  "$PROJECT_ROOT/scripts/autodl/launch_four_by_four.sh" "$MANIFEST" \
  >"$LAUNCH_LOG" 2>&1

CONTROLLER_ROOT="$AUTODL_CONTROL_ROOT/four_methods_four_datasets_continuation/$BACE_CAP_CONTROLLER_ID"
for _ in 1 2 3 4 5; do
  if [[ -s "$CONTROLLER_ROOT/heartbeat.json" ]]; then
    break
  fi
  sleep 2
done
"$AUTODL_PYTHON" - "$CONTROLLER_ROOT/heartbeat.json" "$BACE_CAP_CONTROLLER_ID" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("controller_id") != sys.argv[2] or not isinstance(value.get("pid"), int):
    raise SystemExit("postprocess controller heartbeat identity mismatch")
PY

printf '%s\n' "$BACE_CAP_CONTROLLER_ID" >"$BACE_CAP_QUEUE_ROOT/LAUNCHED.tmp"
mv "$BACE_CAP_QUEUE_ROOT/LAUNCHED.tmp" "$BACE_CAP_QUEUE_ROOT/LAUNCHED"
write_status "CONTROLLER_LAUNCHED" "heartbeat_verified"
