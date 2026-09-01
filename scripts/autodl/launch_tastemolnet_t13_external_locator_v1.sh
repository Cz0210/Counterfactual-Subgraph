#!/usr/bin/env bash
# Persist the read-only locator follower for one already-running T8/T13 chain.

set -euo pipefail

: "${T13_LOCATOR_REPO_ROOT:?set the immutable deployed repository root}"
: "${T8_DUAL_CONTROLLER_ROOT:?set the exact running T8 dual controller root}"
: "${T13_LOCATOR_CONTROLLER_ROOT:?set one fresh locator controller root}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
OUTPUT_BASE=${T13_OUTPUT_BASE:-$RUNTIME/outputs/autodl/tastemolnet/globalgce/t13-full}
POLL_SECONDS=${T13_LOCATOR_POLL_SECONDS:-60}
ROOT=$T13_LOCATOR_CONTROLLER_ROOT

[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "T13 locator refuses GNN ablation" >&2; exit 64; }
[[ ! -e "$ROOT" && ! -L "$ROOT" ]] || { echo "T13 locator controller root must be fresh" >&2; exit 73; }
mkdir -p "$ROOT" "$OUTPUT_BASE"

cd "$T13_LOCATOR_REPO_ROOT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1

nohup setsid "$PY" scripts/autodl/run_tastemolnet_t13_external_locator_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --t8-dual-controller-root "$T8_DUAL_CONTROLLER_ROOT" \
  --control-root "$CONTROL" \
  --t13-output-base "$OUTPUT_BASE" \
  --locator-path "$ROOT/cell_root_locator.json" \
  --heartbeat-path "$ROOT/heartbeat.json" \
  --poll-seconds "$POLL_SECONDS" \
  > "$ROOT/controller.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$ROOT/controller.pid"
{
  printf 'controller_pid=%s\n' "$pid"
  printf 'controller_root=%s\n' "$ROOT"
  printf 't8_dual_controller_root=%s\n' "$T8_DUAL_CONTROLLER_ROOT"
  printf 'locator=%s\n' "$ROOT/cell_root_locator.json"
  printf 'heartbeat=%s\n' "$ROOT/heartbeat.json"
} > "$ROOT/launch.env"

printf 'controller_pid=%s\n' "$pid"
printf 'controller_root=%s\n' "$ROOT"
printf 'locator=%s\n' "$ROOT/cell_root_locator.json"
printf 'heartbeat=%s\n' "$ROOT/heartbeat.json"
