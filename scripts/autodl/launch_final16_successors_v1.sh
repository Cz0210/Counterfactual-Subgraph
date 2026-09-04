#!/usr/bin/env bash
# Launch/adopt the CPU-only final16 registry observer. Science remains owned by
# the already sealed one-shot binders named in final16-owner-registry/current.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ACTION="${1:-launch}"
case "$ACTION" in
  launch|run|status) ;;
  *) echo "usage: $0 [launch|run|status]" >&2; exit 64 ;;
esac

[[ "${RUN_LLM_ABLATION:-0}" == "0" ]] || {
  echo "final16 successors sidecar does not launch LLM ablation science" >&2
  exit 64
}
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || {
  echo "final16 successors sidecar does not launch GNN ablation science" >&2
  exit 64
}

MATRIX_ROOT="${MATRIX_AUTHORITY_ROOT:-$AUTODL_CONTROL_ROOT/fast16_matrix_authority}"
OWNER_REGISTRY="${FINAL16_OWNER_REGISTRY:-$AUTODL_CONTROL_ROOT/final16-owner-registry/current.json}"
STATE_ROOT="${FINAL16_SUCCESSORS_STATE_ROOT:-$AUTODL_CONTROL_ROOT/final16-successors-v1}"
POLL_SECONDS="${SCHEDULER_POLL_SECONDS:-60}"

for path in "$MATRIX_ROOT" "$OWNER_REGISTRY" "$STATE_ROOT"; do
  [[ "$path" == /* ]] || { echo "final16 paths must be absolute: $path" >&2; exit 64; }
  [[ ! -L "$path" ]] || { echo "final16 paths may not be symlinks: $path" >&2; exit 64; }
done
[[ -d "$MATRIX_ROOT" && -f "$MATRIX_ROOT/state.json" && ! -L "$MATRIX_ROOT/state.json" ]] || {
  echo "physical matrix authority is missing: $MATRIX_ROOT" >&2
  exit 66
}
[[ -f "$OWNER_REGISTRY" && ! -L "$OWNER_REGISTRY" ]] || {
  echo "canonical owner registry is missing: $OWNER_REGISTRY" >&2
  exit 66
}

STATUS_COMMAND=(
  "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/status_final16_successors_v1.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --set inference.fallback_to_heuristic=false
  --state-root "$STATE_ROOT"
)
if [[ "$ACTION" == "status" ]]; then
  exec "${STATUS_COMMAND[@]}"
fi

COMMAND=(
  "$AUTODL_PYTHON" -I -B "$SCRIPT_DIR/run_final16_successors_v1.py"
  --config "$PROJECT_ROOT/configs/hpc.yaml"
  --set inference.fallback_to_heuristic=false
  --state-root "$STATE_ROOT"
  --matrix-authority-root "$MATRIX_ROOT"
  --owner-registry "$OWNER_REGISTRY"
  --poll-seconds "$POLL_SECONDS"
)

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RUN_LLM_ABLATION=0
export RUN_GNN_ABLATION=0

if [[ "$ACTION" == "run" ]]; then
  cd "$PROJECT_ROOT"
  exec "${COMMAND[@]}"
fi

mkdir -p "$STATE_ROOT/logs"
chmod 700 "$STATE_ROOT"
if [[ -s "$STATE_ROOT/heartbeat.json" ]] && "${STATUS_COMMAND[@]}" >/dev/null 2>&1; then
  "$AUTODL_PYTHON" -c \
    'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8")); print(f"controller_id={v[\"controller_id\"]}"); print(f"controller_pid={v[\"controller_pid\"]}"); print("controller_adopted=true")' \
    "$STATE_ROOT/heartbeat.json"
  printf 'controller_heartbeat=%s\n' "$STATE_ROOT/heartbeat.json"
  exit 0
fi

LOG="$STATE_ROOT/logs/controller.log"
if command -v tmux >/dev/null 2>&1; then
  SESSION="final16-successors-v1"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session exists without a valid final16 heartbeat: $SESSION" >&2
    exit 75
  fi
  printf -v QUOTED_COMMAND '%q ' "${COMMAND[@]}"
  tmux new-session -d -s "$SESSION" \
    "cd $(printf '%q' "$PROJECT_ROOT") && exec $QUOTED_COMMAND >>$(printf '%q' "$LOG") 2>&1"
  printf 'launcher=tmux\nsession=%s\n' "$SESSION"
else
  cd "$PROJECT_ROOT"
  nohup "${COMMAND[@]}" >>"$LOG" 2>&1 </dev/null &
  printf 'launcher=nohup\nlauncher_pid=%s\n' "$!"
fi

for _ in $(seq 1 120); do
  if [[ -s "$STATE_ROOT/heartbeat.json" ]] && "${STATUS_COMMAND[@]}" >/dev/null 2>&1; then
    "$AUTODL_PYTHON" -c \
      'import json,sys; v=json.load(open(sys.argv[1], encoding="utf-8")); print(f"controller_id={v[\"controller_id\"]}"); print(f"controller_pid={v[\"controller_pid\"]}")' \
      "$STATE_ROOT/heartbeat.json"
    printf 'controller_heartbeat=%s\nstatus_command=%q status\n' \
      "$STATE_ROOT/heartbeat.json" "$0"
    exit 0
  fi
  sleep 0.25
done
tail -n 100 "$LOG" >&2 || true
echo "final16 successors controller did not publish a live heartbeat" >&2
exit 75
