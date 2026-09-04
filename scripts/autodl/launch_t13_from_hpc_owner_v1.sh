#!/usr/bin/env bash
# Predeploy one AutoDL T13 owner; it takes no GPU lease before import PASS.
set -euo pipefail

: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${T8_HPC_T13_REPO_ROOT:?set T8_HPC_T13_REPO_ROOT}"
: "${T8_HPC_T13_SPEC_ROOT:?set T8_HPC_T13_SPEC_ROOT}"
: "${T8_HPC_T13_RELEASE:?set T8_HPC_T13_RELEASE}"
: "${T13_HPC_OWNER_ROOT:?set one fresh owner root}"

[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "T13 owner refuses GNN ablation" >&2; exit 64; }
[[ "${RUN_LLM_ABLATION:-0}" == "0" ]] || { echo "T13 owner refuses LLM ablation" >&2; exit 64; }
[[ ! -e "$T13_HPC_OWNER_ROOT" && ! -L "$T13_HPC_OWNER_ROOT" ]] || { echo "T13 owner root must be fresh" >&2; exit 73; }
mkdir -p "$T13_HPC_OWNER_ROOT"
cd "$T8_HPC_T13_REPO_ROOT"
export PYTHONPATH=$PWD

nohup "$AUTODL_PYTHON" scripts/autodl/run_t13_from_hpc_owner_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --spec-root "$T8_HPC_T13_SPEC_ROOT" \
  --release "$T8_HPC_T13_RELEASE" \
  --heartbeat "$T13_HPC_OWNER_ROOT/heartbeat.json" \
  --owner-root "$T13_HPC_OWNER_ROOT" \
  --poll-seconds "${T13_HPC_OWNER_POLL_SECONDS:-30}" \
  >"$T13_HPC_OWNER_ROOT/owner.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$T13_HPC_OWNER_ROOT/owner.pid"
printf 'T13_HPC_OWNER_PREDEPLOYED pid=%s root=%s\n' "$pid" "$T13_HPC_OWNER_ROOT"
