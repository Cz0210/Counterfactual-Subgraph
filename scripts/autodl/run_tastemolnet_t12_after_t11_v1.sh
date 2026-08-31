#!/usr/bin/env bash
# Narrow persistent T12 producer: wait for one exact T11 PID, then 10k -> 20k -> verify.

set -euo pipefail

: "${T12_REPO_ROOT:?set the immutable deployed repository root}"
: "${T12_CONTROLLER_ROOT:?set one fresh controller root}"
: "${T12_WAIT_PID:?set the exact T11 manager PID}"
: "${T12_WAIT_PID_START_TICKS:?set the frozen T11 /proc start ticks}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the managed NeuroSED root}"
: "${TASTE_T3_ROOT:?set the calibrated T3 root}"
: "${TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY:?set the typed NeuroSED threshold authority}"
: "${T12_EXACT_REPLAY_GATE:?set the exact gate-v2 JSON}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T12_GPU_INDEX:-1}
MIN_FREE_GB=${T12_MIN_FREE_GB:-220}
POLL_SECONDS=${T12_POLL_SECONDS:-60}
OFFICIAL_ROOT=${TASTE_OFFICIAL_GCF_ROOT:-$T12_REPO_ROOT/baselines/gcfexplainer_official}
OUTPUT_BASE=${T12_OUTPUT_BASE:-$RUNTIME/outputs/autodl/tastemolnet/gcfexplainer/t12-production}
LOCK_FILE=$CONTROL/tastemolnet-t12-gcf-production.lock

mkdir -p "$T12_CONTROLLER_ROOT" "$CONTROL" "$OUTPUT_BASE"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another T12 production sidecar holds $LOCK_FILE" >&2
  exit 73
fi

cd "$T12_REPO_ROOT"
export PYTHONPATH=$PWD
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

HEARTBEAT=$T12_CONTROLLER_ROOT/heartbeat.json
STATE=$T12_CONTROLLER_ROOT/state

write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  local temporary=$HEARTBEAT.tmp.$$
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; v={"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"written_at_unix":int(time.time())}; open(p,"w",encoding="utf-8").write(json.dumps(v,sort_keys=True)+"\n")' "$temporary" "$phase" "$science_pid"
  mv "$temporary" "$HEARTBEAT"
  printf '%s\n' "$phase" > "$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

process_start_ticks() {
  "$PY" -c 'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[19])' "$1"
}

process_state() {
  "$PY" -c 'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[0])' "$1"
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

printf '%s\n' "$$" > "$T12_CONTROLLER_ROOT/controller.pid"
write_heartbeat WAITING_FOR_T11 0

if [[ -r "/proc/$T12_WAIT_PID/stat" ]]; then
  observed_ticks=$(process_start_ticks "$T12_WAIT_PID")
  if [[ "$observed_ticks" != "$T12_WAIT_PID_START_TICKS" ]]; then
    echo "T11 PID start-ticks mismatch: expected=$T12_WAIT_PID_START_TICKS observed=$observed_ticks" >&2
    write_heartbeat T11_IDENTITY_MISMATCH 0
    exit 74
  fi
  while [[ -r "/proc/$T12_WAIT_PID/stat" ]]; do
    observed_ticks=$(process_start_ticks "$T12_WAIT_PID" 2>/dev/null || true)
    observed_state=$(process_state "$T12_WAIT_PID" 2>/dev/null || true)
    if [[ -z "$observed_ticks" ]]; then
      break
    fi
    if [[ "$observed_ticks" != "$T12_WAIT_PID_START_TICKS" || "$observed_state" == "Z" ]]; then
      break
    fi
    write_heartbeat WAITING_FOR_T11 0
    sleep "$POLL_SECONDS"
  done
fi

write_heartbeat WAITING_FOR_GPU1 0
while true; do
  gpu_processes=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -z "$gpu_processes" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GPU1 0
done

free_kb=$(df -Pk "$RUNTIME" | awk 'NR==2 {print $4}')
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if [[ -z "$free_kb" || "$free_kb" -lt "$required_kb" ]]; then
  echo "T12 needs at least ${MIN_FREE_GB} GiB free under $RUNTIME; observed_kb=${free_kb:-unknown}" >&2
  write_heartbeat INSUFFICIENT_STORAGE 0
  exit 75
fi

ATTEMPT_ID=${T12_ATTEMPT_ID:-$("$PY" -c 'import uuid; print(uuid.uuid4())')}
GENERATION_TOKEN=${T12_GENERATION_TOKEN:-$("$PY" -c 'import secrets; print(secrets.token_hex(32))')}
GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
OUTPUT_ROOT=${T12_OUTPUT_ROOT:-$OUTPUT_BASE/attempt-$ATTEMPT_ID}
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "T12 output root must be fresh: $OUTPUT_ROOT" >&2
  write_heartbeat OUTPUT_ROOT_NOT_FRESH 0
  exit 76
fi

{
  printf 'attempt_id=%s\n' "$ATTEMPT_ID"
  printf 'generation_token=%s\n' "$GENERATION_TOKEN"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
  printf 'gpu_uuid=%s\n' "$GPU_UUID"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
} > "$T12_CONTROLLER_ROOT/launch.env"

COMMON_ARGS=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --output-root "$OUTPUT_ROOT"
  --attempt-id "$ATTEMPT_ID"
  --generation-token "$GENERATION_TOKEN"
  --gpu-uuid "$GPU_UUID"
  --managed-neurosed-root "$TASTE_MANAGED_NEUROSED_ROOT"
  --t3-root "$TASTE_T3_ROOT"
  --official-root "$OFFICIAL_ROOT"
  --neurosed-threshold-authority "$TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY"
  --exact-replay-gate "$T12_EXACT_REPLAY_GATE"
)

run_stage T12_FRESH_10K "$T12_CONTROLLER_ROOT/fresh-10k.log" \
  "$PY" scripts/run_tastemolnet_gcf_full.py --mode fresh "${COMMON_ARGS[@]}"

CHECKPOINT_10K=$OUTPUT_ROOT/checkpoints/checkpoint-00010000.manifest.json
if [[ ! -s "$CHECKPOINT_10K" ]]; then
  echo "T12 fresh process exited without the committed 10k checkpoint" >&2
  write_heartbeat T12_FRESH_10K_INCOMPLETE 0
  exit 77
fi

run_stage T12_RESUME_20K "$T12_CONTROLLER_ROOT/resume-20k.log" \
  "$PY" scripts/run_tastemolnet_gcf_full.py --mode resume \
  --checkpoint-manifest "$CHECKPOINT_10K" "${COMMON_ARGS[@]}"

CHECKPOINT_20K=$OUTPUT_ROOT/checkpoints/checkpoint-00020000.manifest.json
if [[ ! -s "$CHECKPOINT_20K" ]]; then
  echo "T12 resume process exited without the committed 20k checkpoint" >&2
  write_heartbeat T12_RESUME_20K_INCOMPLETE 0
  exit 78
fi

run_stage T12_GENERATION_VERIFY "$T12_CONTROLLER_ROOT/generation-verify.log" \
  "$PY" scripts/verify_tastemolnet_gcf_full_generation.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --production-root "$OUTPUT_ROOT" \
  --output-root "$OUTPUT_ROOT/generation_verification"

if [[ ! -s "$OUTPUT_ROOT/generation_verification/GENERATION_PASS" ]]; then
  echo "T12 independent generation verifier emitted no PASS" >&2
  write_heartbeat T12_GENERATION_VERIFY_INCOMPLETE 0
  exit 79
fi

write_heartbeat GENERATION_PASS 0
printf '%s\n' "$OUTPUT_ROOT" > "$T12_CONTROLLER_ROOT/completed_output_root"
