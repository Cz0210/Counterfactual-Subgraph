#!/usr/bin/env bash
# Launch one CPU-only persistent T8 HPC import/T13 release owner.
set -euo pipefail

: "${AUTODL_PYTHON:?set AUTODL_PYTHON}"
: "${T8_HPC_T13_REPO_ROOT:?set T8_HPC_T13_REPO_ROOT}"
: "${T8_HPC_T13_SPEC_ROOT:?set T8_HPC_T13_SPEC_ROOT}"
: "${T8_HPC_IMPORT_OWNER_ROOT:?set one fresh owner root}"

[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "import owner refuses GNN ablation" >&2; exit 64; }
[[ "${RUN_LLM_ABLATION:-0}" == "0" ]] || { echo "import owner refuses LLM ablation" >&2; exit 64; }
[[ ! -e "$T8_HPC_IMPORT_OWNER_ROOT" ]] || { echo "owner root must be fresh" >&2; exit 73; }
mkdir -p "$T8_HPC_IMPORT_OWNER_ROOT"
cd "$T8_HPC_T13_REPO_ROOT"
export PYTHONPATH=$PWD

nohup "$AUTODL_PYTHON" scripts/autodl/run_t8_hpc_import_owner_v1.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --spec-root "$T8_HPC_T13_SPEC_ROOT" \
  --heartbeat "$T8_HPC_IMPORT_OWNER_ROOT/heartbeat.json" \
  --release "$T8_HPC_IMPORT_OWNER_ROOT/t13_release.json" \
  --poll-seconds "${T8_HPC_IMPORT_POLL_SECONDS:-300}" \
  >"$T8_HPC_IMPORT_OWNER_ROOT/owner.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$T8_HPC_IMPORT_OWNER_ROOT/owner.pid"
echo "T8_HPC_IMPORT_OWNER_LAUNCHED pid=$pid root=$T8_HPC_IMPORT_OWNER_ROOT"
