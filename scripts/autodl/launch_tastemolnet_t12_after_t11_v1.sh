#!/usr/bin/env bash
# Persist the narrow T12 10k -> 20k producer beyond the launching shell.

set -euo pipefail

: "${T12_REPO_ROOT:?set the immutable deployed repository root}"
: "${T12_WAIT_PID:?set the exact T11 manager PID}"
: "${T12_WAIT_PID_START_TICKS:?set the frozen T11 manager start ticks}"

RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
CONTROLLER_ID=tastemolnet-t12-gcf-after-t11-$(date -u +%Y%m%dT%H%M%SZ)-$("$PY" -c 'import uuid; print(str(uuid.uuid4())[:8])')
CONTROLLER_ROOT=$CONTROL/$CONTROLLER_ID
mkdir -p "$CONTROLLER_ROOT"

export T12_CONTROLLER_ROOT="$CONTROLLER_ROOT"
nohup bash "$T12_REPO_ROOT/scripts/autodl/run_tastemolnet_t12_after_t11_v1.sh" \
  > "$CONTROLLER_ROOT/controller.log" 2>&1 < /dev/null &
controller_pid=$!
printf '%s\n' "$controller_pid" > "$CONTROLLER_ROOT/launcher.pid"

printf 'controller_id=%s\n' "$CONTROLLER_ID"
printf 'controller_pid=%s\n' "$controller_pid"
printf 'controller_root=%s\n' "$CONTROLLER_ROOT"
printf 'status_command=cat %q; cat %q; tail -n 80 %q\n' \
  "$CONTROLLER_ROOT/state" "$CONTROLLER_ROOT/heartbeat.json" \
  "$CONTROLLER_ROOT/controller.log"
