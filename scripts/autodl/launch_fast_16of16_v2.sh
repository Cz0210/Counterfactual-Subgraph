#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-launch}"
case "$ACTION" in launch|run|status|once) ;; *) echo "usage: $0 [launch|run|status|once]" >&2; exit 64 ;; esac
: "${FAST_16OF16_V2_SPEC:?FAST_16OF16_V2_SPEC is required}"
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "GNN ablation remains disabled" >&2; exit 64; }
[[ "${SCHEDULER_POLL_SECONDS:-60}" == "60" ]] || { echo "poll must equal 60" >&2; exit 64; }

STATE_ROOT="$("$AUTODL_PYTHON" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["state_root"])' "$FAST_16OF16_V2_SPEC")"
if [[ "$ACTION" == "status" ]]; then
  exec "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/status_fast_16of16_v2.py" --state-root "$STATE_ROOT"
fi

COMMAND=("$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_fast_16of16_v2.py" --config "$PROJECT_ROOT/configs/hpc.yaml" --spec "$FAST_16OF16_V2_SPEC")
[[ "$ACTION" != "once" ]] || COMMAND+=(--once)
if [[ "$ACTION" == "run" || "$ACTION" == "once" ]]; then exec "${COMMAND[@]}"; fi

install -d -m 700 "$STATE_ROOT/logs"
nohup "${COMMAND[@]}" >"$STATE_ROOT/logs/controller.log" 2>&1 < /dev/null &
PID="$!"
for _ in $(seq 1 120); do
  if [[ -s "$STATE_ROOT/heartbeat.json" ]]; then
    printf 'controller_pid=%s\ncontroller_heartbeat=%s\n' "$PID" "$STATE_ROOT/heartbeat.json"
    exit 0
  fi
  sleep 0.25
done
echo "fast-v2 did not publish a heartbeat" >&2
exit 75
