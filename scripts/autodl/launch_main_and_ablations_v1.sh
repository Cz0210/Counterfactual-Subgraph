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

RUNTIME="${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}"
CONTROL="${AUTODL_CONTROL_ROOT:-$RUNTIME/control}"
STATE_ROOT="${MAIN_AND_ABLATIONS_STATE_ROOT:-$CONTROL/main-and-ablations-v1}"
AUTHORITY_ROOT="${MATRIX_AUTHORITY_ROOT:-$CONTROL/fast16_matrix_authority}"
AUTHORITY_STATE="${MATRIX_AUTHORITY_STATE:-$AUTHORITY_ROOT/state.json}"

if [[ "$ACTION" == "status" ]]; then
  exec "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/status_main_and_ablations_v1.py" \
    --state-root "$STATE_ROOT"
fi

COMMAND=(
  "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_main_and_ablations_v1.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --set inference.fallback_to_heuristic=false
  --state-root "$STATE_ROOT"
  --matrix-authority "$AUTHORITY_STATE"
  --poll-seconds "${SCHEDULER_POLL_SECONDS:-30}"
)

if [[ "$ACTION" == "run" ]]; then
  exec "${COMMAND[@]}"
fi

mkdir -p "$STATE_ROOT/logs"
nohup "${COMMAND[@]}" >"$STATE_ROOT/logs/controller.log" 2>&1 < /dev/null &
CONTROLLER_PID=$!
for _ in $(seq 1 120); do
  if [[ -s "$STATE_ROOT/heartbeat.json" ]]; then
    printf 'controller_pid=%s\ncontroller_heartbeat=%s\n' \
      "$CONTROLLER_PID" "$STATE_ROOT/heartbeat.json"
    exit 0
  fi
  if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    tail -n 100 "$STATE_ROOT/logs/controller.log" >&2 || true
    exit 75
  fi
  sleep 0.25
done
echo "main-and-ablations sidecar did not publish a heartbeat" >&2
exit 75
