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

# Component launchers receive their complete environment from immutable task
# specs.  Preflight every supplied spec path here; the Python sidecar reports a
# missing component spec as DISPATCH_CONFIG_INVALID without blocking unrelated
# higher-priority recovery components.
for task_spec_variable in \
  MUT_CONTINUATION_TASK_SPEC T14_RESUME_TASK_SPEC \
  T8_ZERO_FINALIZER_TASK_SPEC LLM_ABLATION_TASK_SPEC GNN_ABLATION_TASK_SPEC; do
  task_spec_path="${!task_spec_variable:-}"
  [[ -z "$task_spec_path" ]] && continue
  [[ "$task_spec_path" == /* && -f "$task_spec_path" && ! -L "$task_spec_path" ]] || {
    echo "$task_spec_variable must name one absolute physical task spec" >&2
    exit 64
  }
done
unset task_spec_variable task_spec_path

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
    if "$AUTODL_PYTHON" -c \
      'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("controller_pid") == int(sys.argv[2]) else 1)' \
      "$STATE_ROOT/heartbeat.json" "$CONTROLLER_PID" 2>/dev/null; then
      printf 'controller_pid=%s\ncontroller_heartbeat=%s\n' \
        "$CONTROLLER_PID" "$STATE_ROOT/heartbeat.json"
      exit 0
    fi
  fi
  if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    tail -n 100 "$STATE_ROOT/logs/controller.log" >&2 || true
    exit 75
  fi
  sleep 0.25
done
echo "main-and-ablations sidecar did not publish a heartbeat" >&2
exit 75
