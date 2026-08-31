#!/usr/bin/env bash
# Launch the narrow durable T12 paper continuation with a persistent heartbeat.

set -euo pipefail

: "${T12_PAPER_CONTROLLER_ROOT:?set one fresh paper controller root}"
: "${T12_REPO_ROOT:?set deployed repository root}"

mkdir -p "$T12_PAPER_CONTROLLER_ROOT"
if [[ -e "$T12_PAPER_CONTROLLER_ROOT/launcher.pid" ]]; then
  echo "T12 paper controller root is not fresh" >&2
  exit 73
fi

nohup bash "$T12_REPO_ROOT/scripts/autodl/run_tastemolnet_t12_paper_after_generation_v1.sh" \
  > "$T12_PAPER_CONTROLLER_ROOT/controller.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$T12_PAPER_CONTROLLER_ROOT/launcher.pid"
printf 't12_paper_controller_pid=%s\n' "$pid"
printf 't12_paper_controller_root=%s\n' "$T12_PAPER_CONTROLLER_ROOT"
