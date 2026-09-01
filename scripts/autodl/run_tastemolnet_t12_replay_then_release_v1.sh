#!/usr/bin/env bash
# Narrow AutoDL handoff: fresh exact replay gate, then the existing T12 relay.

set -euo pipefail

: "${T12_REPO_ROOT:?set the immutable deployed repository root}"
: "${T12_CANARY_CONTROLLER_ROOT:?set one fresh controller root}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the adopted managed NeuroSED root}"
: "${TASTE_T3_ROOT:?set the calibrated T3 root}"
: "${TASTE_T7_PASS_ROOT:?set the managed T7 smoke PASS root}"
: "${T12_MANAGED_RELEASE_ROOT:?set the typed managed release PASS root}"
: "${T12_RELEASE_VALIDATOR_ROOT:?set the pinned release validator root}"
: "${TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY:?set the threshold authority}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T12_GPU_INDEX:-3}
POLL_SECONDS=${T12_POLL_SECONDS:-30}
OFFICIAL_ROOT=${TASTE_OFFICIAL_GCF_ROOT:-$T12_REPO_ROOT/baselines/gcfexplainer_official}
CANARY_BASE=${T12_CANARY_OUTPUT_BASE:-$RUNTIME/outputs/autodl/tastemolnet/gcfexplainer/t12-replay/$(basename "$T12_CANARY_CONTROLLER_ROOT")}
HEARTBEAT=$T12_CANARY_CONTROLLER_ROOT/heartbeat.json
STATE=$T12_CANARY_CONTROLLER_ROOT/state

[[ "$GPU_INDEX" == "3" ]] || { echo "T12 recovery is pinned to GPU3" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "GNN ablation must remain disabled" >&2; exit 64; }
[[ -d "$T12_CANARY_CONTROLLER_ROOT" && ! -L "$T12_CANARY_CONTROLLER_ROOT" ]] || {
  echo "T12 canary controller root must be one physical directory" >&2
  exit 64
}
[[ ! -e "$T12_CANARY_CONTROLLER_ROOT/controller.pid" ]] || {
  echo "T12 canary controller root is not fresh" >&2
  exit 73
}
[[ ! -e "$CANARY_BASE" && ! -L "$CANARY_BASE" ]] || {
  echo "T12 canary output root is not fresh" >&2
  exit 73
}

write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  local temporary=$HEARTBEAT.tmp.$$
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; v={"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"written_at_unix":int(time.time())}; f=open(p,"x",encoding="utf-8"); json.dump(v,f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close()' "$temporary" "$phase" "$science_pid"
  mv "$temporary" "$HEARTBEAT"
  printf '%s\n' "$phase" > "$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

run_stage() {
  local phase=$1
  local log=$2
  shift 2
  "$@" > "$log" 2>&1 &
  local science_pid=$!
  write_heartbeat "$phase" "$science_pid"
  while kill -0 "$science_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    write_heartbeat "$phase" "$science_pid"
  done
  if ! wait "$science_pid"; then
    write_heartbeat "${phase}_FAILED" 0
    return 1
  fi
}

printf '%s\n' "$$" > "$T12_CANARY_CONTROLLER_ROOT/controller.pid"
exec 9>"$CONTROL/tastemolnet-t12-gcf-replay-canary.lock"
flock -n 9 || { write_heartbeat DUPLICATE_CANARY_BLOCKED 0; exit 73; }

cd "$T12_REPO_ROOT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$GPU_UUID" == GPU-* ]] || { write_heartbeat GPU_UUID_INVALID 0; exit 75; }
mkdir -p "$CANARY_BASE"
U_ATTEMPT=$($PY -c 'import uuid; print(uuid.uuid4())')
U_TOKEN=$($PY -c 'import secrets; print(secrets.token_hex(32))')
R_ATTEMPT=$($PY -c 'import uuid; print(uuid.uuid4())')
R_TOKEN=$($PY -c 'import secrets; print(secrets.token_hex(32))')

COMMON_ARGS=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --gpu-uuid "$GPU_UUID"
  --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT"
  --t3-root "$TASTE_T3_ROOT"
  --official-root "$OFFICIAL_ROOT"
  --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY"
)

run_stage CANARY_UNINTERRUPTED "$T12_CANARY_CONTROLLER_ROOT/uninterrupted.log" \
  "$PY" scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  "${COMMON_ARGS[@]}" --mode uninterrupted \
  --output-root "$CANARY_BASE/uninterrupted" \
  --observation "$CANARY_BASE/uninterrupted.json" \
  --attempt-id "$U_ATTEMPT" --generation-token "$U_TOKEN"

run_stage CANARY_CHECKPOINT "$T12_CANARY_CONTROLLER_ROOT/checkpoint.log" \
  "$PY" scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  "${COMMON_ARGS[@]}" --mode checkpoint \
  --output-root "$CANARY_BASE/resumable" \
  --attempt-id "$R_ATTEMPT" --generation-token "$R_TOKEN"

run_stage CANARY_RESUME "$T12_CANARY_CONTROLLER_ROOT/resume.log" \
  "$PY" scripts/run_tastemolnet_gcf_replay_canary_worker.py \
  "${COMMON_ARGS[@]}" --mode resume \
  --output-root "$CANARY_BASE/resumable" \
  --observation "$CANARY_BASE/resumed.json" \
  --checkpoint-manifest "$CANARY_BASE/resumable/checkpoints/checkpoint-00000008.manifest.json" \
  --attempt-id "$R_ATTEMPT" --generation-token "$R_TOKEN"

run_stage CANARY_GATE "$T12_CANARY_CONTROLLER_ROOT/gate.log" \
  "$PY" scripts/run_tastemolnet_gcf_replay_canary.py \
  --config configs/hpc.yaml --set inference.fallback_to_heuristic=false \
  --uninterrupted "$CANARY_BASE/uninterrupted.json" \
  --cross-process-resumed "$CANARY_BASE/resumed.json" \
  --checkpoint-prefix-receipt "$CANARY_BASE/resumable/prefix_receipt.json" \
  --output "$CANARY_BASE/replay_gate.json"

[[ -s "$CANARY_BASE/replay_gate.json" ]] || { write_heartbeat CANARY_GATE_MISSING 0; exit 76; }
printf '%s\n' "$CANARY_BASE/replay_gate.json" > "$T12_CANARY_CONTROLLER_ROOT/replay_gate_path"
export T12_EXACT_REPLAY_GATE="$CANARY_BASE/replay_gate.json"
write_heartbeat LAUNCHING_T12_RELEASE_SUCCESSOR 0
bash "$T12_REPO_ROOT/scripts/autodl/launch_tastemolnet_t12_release_relay_v1.sh" \
  > "$T12_CANARY_CONTROLLER_ROOT/release-launch.log" 2>&1
write_heartbeat T12_RELEASE_SUCCESSOR_LAUNCHED 0
