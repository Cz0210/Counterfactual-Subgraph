#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AUTODL_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}"
DATA_ROOT="${AUTODL_DATA_ROOT:-/autodl-fs/data}"
CONTROL_ROOT="${AUTODL_CONTROL_ROOT:-$DATA_ROOT/counterfactual-subgraph-runtime/control}"
NAMESPACE="${AUTODL_DASHBOARD_NAMESPACE:-four_methods_four_datasets_continuation}"
DASHBOARD_ID="${AUTODL_DASHBOARD_ID:-four_by_four_dashboard_v2}"
HOST="127.0.0.1"
PORT="${AUTODL_DASHBOARD_PORT:-8766}"
INTERVAL="${AUTODL_DASHBOARD_INTERVAL_SECONDS:-5}"
STALE_SECONDS="${AUTODL_DASHBOARD_STALE_SECONDS:-180}"
STATE_ROOT="$CONTROL_ROOT/dashboards/$DASHBOARD_ID"
PID_FILE="$STATE_ROOT/dashboard.pid"
LOG_FILE="$STATE_ROOT/dashboard.log"

if [[ ! -x "$PY" ]]; then
  echo "Dashboard Python is not executable: $PY" >&2
  exit 2
fi
if [[ ! -f "$SCRIPT_DIR/serve_four_by_four_dashboard.py" ]]; then
  echo "Dashboard entrypoint is missing" >&2
  exit 2
fi
case "$PORT" in
  ''|*[!0-9]*) echo "AUTODL_DASHBOARD_PORT must be numeric" >&2; exit 2 ;;
esac
if (( PORT < 1 || PORT > 65535 )); then
  echo "AUTODL_DASHBOARD_PORT must be in 1..65535" >&2
  exit 2
fi

mkdir -p "$STATE_ROOT"
if [[ -s "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$EXISTING_PID" && -r "/proc/$EXISTING_PID/cmdline" ]]; then
    EXISTING_COMMAND="$(tr '\0' ' ' < "/proc/$EXISTING_PID/cmdline")"
    if [[ "$EXISTING_COMMAND" == *"serve_four_by_four_dashboard.py"* && "$EXISTING_COMMAND" == *"--port $PORT"* ]]; then
      echo "[AUTODL_DASHBOARD_ALREADY_RUNNING] pid=$EXISTING_PID port=$PORT"
      exit 0
    fi
    echo "PID file points to another live process; choose a fresh AUTODL_DASHBOARD_ID" >&2
    exit 2
  fi
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
nohup "$PY" "$SCRIPT_DIR/serve_four_by_four_dashboard.py" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --control-root "$CONTROL_ROOT" \
  --namespace "$NAMESPACE" \
  --stale-seconds "$STALE_SECONDS" \
  serve \
  --host "$HOST" \
  --port "$PORT" \
  --interval "$INTERVAL" \
  >>"$LOG_FILE" 2>&1 </dev/null &
DASHBOARD_PID=$!

PID_TEMP="$PID_FILE.tmp.$DASHBOARD_PID"
printf '%s\n' "$DASHBOARD_PID" > "$PID_TEMP"
mv "$PID_TEMP" "$PID_FILE"

"$PY" -c "import time,urllib.request; u='http://127.0.0.1:$PORT/healthz'; last=None
for _ in range(20):
    try:
        with urllib.request.urlopen(u, timeout=2) as response:
            assert response.status == 200
        raise SystemExit(0)
    except Exception as exc:
        last=exc; time.sleep(0.5)
raise SystemExit('dashboard health check failed: '+repr(last))"

echo "[AUTODL_DASHBOARD_RUNNING] pid=$DASHBOARD_PID port=$PORT log=$LOG_FILE"
echo "SSH tunnel: ssh -N -T -L 18766:127.0.0.1:$PORT -p <SSH_PORT> <USER>@<AUTODL_HOST>"
echo "Open: http://127.0.0.1:18766"
