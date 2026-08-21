#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_SPEC="$PROJECT_ROOT/ops/specs/autodl_three_lines_20260821.yaml"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 start [--spec PATH] [--lane LANE_ID ...]" >&2
  echo "       $0 resume [--spec PATH] [--lane LANE_ID ...]" >&2
  echo "       $0 {status|stop} [--spec PATH]" >&2
  exit 2
fi

ACTION="$1"
shift
case "$ACTION" in
  start|status|resume|stop) ;;
  *)
    echo "invalid action: $ACTION" >&2
    exit 2
    ;;
esac

HAS_SPEC=false
for argument in "$@"; do
  if [[ "$argument" == "--spec" || "$argument" == --spec=* ]]; then
    HAS_SPEC=true
  fi
done

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$HAS_SPEC" == true ]]; then
  exec python3 scripts/autodl/run_three_lines.py "$ACTION" "$@"
else
  exec python3 scripts/autodl/run_three_lines.py "$ACTION" --spec "$DEFAULT_SPEC" "$@"
fi
