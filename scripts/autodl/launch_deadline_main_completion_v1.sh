#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-launch}"
case "$ACTION" in
  launch|run|status) ;;
  *) echo "usage: $0 [launch|run|status]" >&2; exit 64 ;;
esac

: "${DEADLINE_MAIN_COMPLETION_SPEC:?DEADLINE_MAIN_COMPLETION_SPEC is required}"
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "GNN ablation must remain disabled" >&2; exit 64; }
[[ "${SCHEDULER_POLL_SECONDS:-60}" == "60" ]] || { echo "poll interval must equal 60" >&2; exit 64; }

STATE_ROOT="$($AUTODL_PYTHON -B -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["state_root"])' "$DEADLINE_MAIN_COMPLETION_SPEC")"
if [[ "$ACTION" == "status" ]]; then
  exec "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/status_deadline_main_completion_v1.py" --state-root "$STATE_ROOT"
fi

PYTHON_ACTION="$ACTION"
if [[ "$ACTION" == "launch" ]]; then
  PYTHON_ACTION="run"
fi

COMMAND=(
  "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_deadline_main_completion_v1.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  "$PYTHON_ACTION"
  --spec "$DEADLINE_MAIN_COMPLETION_SPEC"
)
if [[ "$ACTION" == "run" ]]; then
  exec "${COMMAND[@]}"
fi

mkdir -p "$STATE_ROOT/logs"
nohup "${COMMAND[@]}" >"$STATE_ROOT/logs/controller.log" 2>&1 < /dev/null &
CONTROLLER_PID="$!"
for _ in $(seq 1 120); do
  if [[ -s "$STATE_ROOT/heartbeat.json" ]]; then
    printf 'deadline_controller_pid=%s\ndeadline_controller_heartbeat=%s\n' \
      "$CONTROLLER_PID" "$STATE_ROOT/heartbeat.json"
    exit 0
  fi
  sleep 0.25
done
echo "deadline sidecar did not publish a heartbeat" >&2
exit 75
