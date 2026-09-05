#!/usr/bin/env bash
# Read-only status for the attempt-scoped T8 HPC -> Mac -> AutoDL relay.
set -uo pipefail

MAC_RELAY_ROOT=${MAC_RELAY_ROOT:-/Volumes/DireRaven/counterfactual-hpc-offload/t8-scoped-relay-v2}
RELAY_CONTROL_ROOT=${RELAY_CONTROL_ROOT:-$MAC_RELAY_ROOT/control}
state_path="$RELAY_CONTROL_ROOT/state.json"
pid="$(cat "$RELAY_CONTROL_ROOT/pid" 2>/dev/null || true)"
case "$pid" in
  ''|*[!0-9]*) alive=false;;
  *) if kill -0 "$pid" 2>/dev/null; then alive=true; else alive=false; fi;;
esac
printf 'relay_pid=%s\n' "${pid:-NONE}"
printf 'relay_alive=%s\n' "$alive"
if [[ -f "$state_path" ]]; then
  python3 - "$state_path" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("relay_state=MALFORMED")
    raise SystemExit(0)
fields = (
    "state", "heartbeat_at", "detail", "pid", "attempt_id",
    "hpc_package_root", "hpc_archive_path", "mac_partial_root",
    "mac_final_root", "autodl_partial_root", "autodl_final_root",
    "expected_archive_bytes", "expected_archive_sha256",
    "current_partial_bytes", "log_path", "matrix_write_enabled",
    "hpc_source_delete_enabled",
)
for key in fields:
    value = str(payload.get(key, "UNKNOWN")).replace("\n", " ")[:1024]
    print(f"relay_{key}={value}")
PY
else
  printf 'relay_state=MISSING\n'
fi
printf 'status_side_effects=NONE\n'
