#!/usr/bin/env bash
# Read-only, redacted status for the Mac T8 package relay.
set -uo pipefail
RELAY_CONTROL_ROOT=${RELAY_CONTROL_ROOT:-$HOME/.cache/t8-result-relay-v1}
state_path="$RELAY_CONTROL_ROOT/state.json"
pid="$(cat "$RELAY_CONTROL_ROOT/pid" 2>/dev/null || true)"
case "$pid" in ''|*[!0-9]*) alive=false;; *) if kill -0 "$pid" 2>/dev/null; then alive=true; else alive=false; fi;; esac
printf 'relay_pid=%s\n' "${pid:-NONE}"
printf 'relay_alive=%s\n' "$alive"
if [[ -f "$state_path" ]]; then
  python3 - "$state_path" <<'PY'
import json,sys
try: p=json.load(open(sys.argv[1]))
except Exception: print('relay_state=MALFORMED'); raise SystemExit(0)
for key in ('state','updated_at','detail','matrix_write_enabled'):
    value=str(p.get(key,'UNKNOWN')).replace('\n',' ')[:512]
    print(f'relay_{key}={value}')
PY
else
  printf 'relay_state=MISSING\n'
fi
