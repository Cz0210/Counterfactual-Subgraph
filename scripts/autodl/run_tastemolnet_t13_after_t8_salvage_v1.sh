#!/usr/bin/env bash
# Persistent T13 relay released only by a typed managed-v2 T8 PASS.

set -euo pipefail

: "${T13_REPO_ROOT:?set the immutable deployed repository root}"
: "${T13_CONTROLLER_ROOT:?set one fresh controller root}"
: "${T8_PASS_ROOT:?set the published managed-v2 T8 final root}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set the frozen GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set the frozen train CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set the frozen calibration CSV}"
: "${TASTEMOLNET_TEST_CSV:?set the held-out test CSV}"
: "${GLOBALGCE_OFFICIAL_ROOT:?set the pinned official source root}"
: "${MOLCLR_ROOT:?set the pinned MolCLR source root}"
: "${MOLCLR_CHECKPOINT:?set the pinned MolCLR checkpoint}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set the frozen threshold contract}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T13_GPU_INDEX:-1}
POLL_SECONDS=${T13_POLL_SECONDS:-60}
OUTPUT_BASE=${T13_OUTPUT_BASE:-$RUNTIME/outputs/autodl/tastemolnet/globalgce/t13-full}
LOCK_FILE=$CONTROL/tastemolnet-t13-globalgce-salvage-relay.lock

[[ "$GPU_INDEX" == "1" ]] || { echo "T13 salvage relay is pinned to GPU1" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "T13 refuses GNN ablation" >&2; exit 64; }
[[ ! -e "$T13_CONTROLLER_ROOT/controller.pid" ]] || { echo "T13 controller root is not fresh" >&2; exit 73; }
mkdir -p "$T13_CONTROLLER_ROOT" "$CONTROL" "$OUTPUT_BASE"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "another T13 salvage relay is active" >&2; exit 73; }

cd "$T13_REPO_ROOT"
export PYTHONPATH=$PWD
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

HEARTBEAT=$T13_CONTROLLER_ROOT/heartbeat.json
STATE=$T13_CONTROLLER_ROOT/state
write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; q=p+".tmp."+str(os.getpid()); f=open(q,"x"); json.dump({"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"written_at_unix":int(time.time())},f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(q,p)' "$HEARTBEAT" "$phase" "$science_pid"
  printf '%s\n' "$phase" > "$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

printf '%s\n' "$$" > "$T13_CONTROLLER_ROOT/controller.pid"
write_heartbeat VALIDATING_T8_TYPED_PASS 0
"$PY" scripts/autodl/adopt_tastemolnet_t8_deadline_v2.py \
  --mode validate \
  --config "$T13_REPO_ROOT/configs/hpc.yaml" \
  --final-path "$T8_PASS_ROOT" \
  > "$T13_CONTROLLER_ROOT/t8-validation.log"

write_heartbeat WAITING_FOR_GPU1 0
while true; do
  gpu_processes=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  [[ -z "$gpu_processes" ]] && break
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GPU1 0
done

ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
OUTPUT_ROOT=$OUTPUT_BASE/attempt-$ATTEMPT_ID
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || { echo "T13 output is not fresh" >&2; exit 74; }
GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
WNODE_CACHE=${WNODE_CACHE_DB:-$RUNTIME/cache/tastemolnet/t13-globalgce-wnode.sqlite}
NODE_CACHE=${NODE_EMBEDDING_CACHE_DIR:-$RUNTIME/cache/tastemolnet/t13-globalgce-molclr-nodes}
mkdir -p "$(dirname "$WNODE_CACHE")" "$NODE_CACHE"
{
  printf 'attempt_id=%s\n' "$ATTEMPT_ID"
  printf 'output_root=%s\n' "$OUTPUT_ROOT"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
  printf 'gpu_uuid=%s\n' "$GPU_UUID"
  printf 't8_pass_root=%s\n' "$T8_PASS_ROOT"
} > "$T13_CONTROLLER_ROOT/launch.env"

COMMON=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --t8-pass-root "$T8_PASS_ROOT"
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT"
  --train-csv "$TASTEMOLNET_TRAIN_CSV"
  --calibration-csv "$TASTEMOLNET_CALIBRATION_CSV"
  --test-csv "$TASTEMOLNET_TEST_CSV"
  --official-root "$GLOBALGCE_OFFICIAL_ROOT"
  --molclr-root "$MOLCLR_ROOT"
  --molclr-checkpoint "$MOLCLR_CHECKPOINT"
  --wnode-cache-db "$WNODE_CACHE"
  --node-embedding-cache-dir "$NODE_CACHE"
  --threshold-contract "$TASTEMOLNET_THRESHOLD_CONTRACT"
  --output-dir "$OUTPUT_ROOT"
  --epochs "${T13_EPOCHS:-100}"
)

"$PY" scripts/run_tastemolnet_globalgce_full.py "${COMMON[@]}" \
  > "$T13_CONTROLLER_ROOT/science.log" 2>&1 &
SCIENCE_PID=$!
write_heartbeat T13_SCIENCE_RUNNING "$SCIENCE_PID"
printf '%s\n' '[TASTE_T13_GLOBALGCE_FULL_LAUNCHED]' | tee "$T13_CONTROLLER_ROOT/TASTE_T13_GLOBALGCE_FULL_LAUNCHED"
while kill -0 "$SCIENCE_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
  write_heartbeat T13_SCIENCE_RUNNING "$SCIENCE_PID"
done
if ! wait "$SCIENCE_PID"; then
  write_heartbeat T13_SCIENCE_FAILED 0
  exit 75
fi

write_heartbeat T13_INDEPENDENT_VERIFY 0
"$PY" scripts/run_tastemolnet_globalgce_full.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --output-dir "$OUTPUT_ROOT" \
  --verify-only > "$T13_CONTROLLER_ROOT/verification.log" 2>&1
printf '%s\n' "$OUTPUT_ROOT" > "$T13_CONTROLLER_ROOT/completed_output_root"
write_heartbeat PASS 0
