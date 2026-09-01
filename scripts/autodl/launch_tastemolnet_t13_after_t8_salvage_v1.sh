#!/usr/bin/env bash
# Persist one T13 relay after T8 salvage/adoption.

set -euo pipefail
: "${T13_REPO_ROOT:?set the immutable deployed repository root}"

RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
CONTROLLER_ID=tastemolnet-t13-after-t8-salvage-$($PY -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+str(uuid.uuid4())[:8])')
CONTROLLER_ROOT=$CONTROL/$CONTROLLER_ID
mkdir -p "$CONTROLLER_ROOT"
export T13_CONTROLLER_ROOT="$CONTROLLER_ROOT"
export T13_GPU_INDEX=1
nohup bash "$T13_REPO_ROOT/scripts/autodl/run_tastemolnet_t13_after_t8_salvage_v1.sh" \
  > "$CONTROLLER_ROOT/controller.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$CONTROLLER_ROOT/launcher.pid"
printf 'controller_id=%s\ncontroller_pid=%s\ncontroller_root=%s\n' "$CONTROLLER_ID" "$pid" "$CONTROLLER_ROOT"
