#!/usr/bin/env bash
# Persist the fixed T8 dual-branch recovery relay.

set -euo pipefail
: "${T8_REPO_ROOT:?set the immutable deployed repository root}"

RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
CONTROLLER_ID=tastemolnet-t8-dual-branch-recovery-$($PY -c 'import datetime,uuid; print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+str(uuid.uuid4())[:8])')
CONTROLLER_ROOT=$CONTROL/$CONTROLLER_ID
[[ ! -e "$CONTROLLER_ROOT" && ! -L "$CONTROLLER_ROOT" ]] || { echo "T8 dual-branch launch root is not fresh" >&2; exit 73; }
mkdir -p "$CONTROLLER_ROOT"
export T8_DUAL_CONTROLLER_ROOT=$CONTROLLER_ROOT
export T8_DUAL_GPU_INDEX=1
export RUN_GNN_ABLATION=0
nohup bash "$T8_REPO_ROOT/scripts/autodl/run_tastemolnet_t8_dual_branch_recovery_v1.sh" \
  > "$CONTROLLER_ROOT/controller.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$CONTROLLER_ROOT/launcher.pid"
printf 'controller_id=%s\ncontroller_pid=%s\ncontroller_root=%s\n' "$CONTROLLER_ID" "$pid" "$CONTROLLER_ROOT"
