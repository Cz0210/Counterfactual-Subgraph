#!/usr/bin/env bash
# Persist the read-only T8 salvage/adoption route.

set -euo pipefail
: "${T8_REPO_ROOT:?set the immutable deployed repository root}"
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
ID=tastemolnet-t8-salvage-$($PY -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+str(uuid.uuid4())[:8])')
ROOT=$CONTROL/$ID
mkdir -p "$ROOT"
export T8_SALVAGE_CONTROLLER_ROOT="$ROOT"
nohup bash "$T8_REPO_ROOT/scripts/autodl/run_tastemolnet_t8_salvage_release_v1.sh" \
  > "$ROOT/controller.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$ROOT/launcher.pid"
printf 'controller_id=%s\ncontroller_pid=%s\ncontroller_root=%s\n' "$ID" "$pid" "$ROOT"
