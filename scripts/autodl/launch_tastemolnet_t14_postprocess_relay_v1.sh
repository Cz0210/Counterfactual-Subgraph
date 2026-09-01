#!/usr/bin/env bash
# Persist the narrow T14 generation-to-paper successor beyond this shell.

set -euo pipefail

: "${T14_RELAY_REPO_ROOT:?set one immutable deployed repository root}"

RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
CONTROLLER_ID=tastemolnet-t14-postprocess-relay-$($PY -I -B -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+str(uuid.uuid4())[:8])')
CONTROLLER_ROOT=$CONTROL/$CONTROLLER_ID
mkdir -p "$CONTROLLER_ROOT"

export T14_POSTPROCESS_CONTROLLER_ROOT=$CONTROLLER_ROOT
export T14_POSTPROCESS_GPU_INDEX=2
nohup bash "$T14_RELAY_REPO_ROOT/scripts/autodl/run_tastemolnet_t14_postprocess_relay_v1.sh" \
  >"$CONTROLLER_ROOT/controller.log" 2>&1 </dev/null &
controller_pid=$!
printf '%s\n' "$controller_pid" >"$CONTROLLER_ROOT/launcher.pid"

printf 'controller_id=%s\n' "$CONTROLLER_ID"
printf 'controller_pid=%s\n' "$controller_pid"
printf 'controller_root=%s\n' "$CONTROLLER_ROOT"
printf 'locator=%s\n' "$CONTROLLER_ROOT/cell_root_locator.json"
printf 'status_command=cat %q; cat %q; tail -n 80 %q\n' \
  "$CONTROLLER_ROOT/state" "$CONTROLLER_ROOT/heartbeat.json" \
  "$CONTROLLER_ROOT/controller.log"
