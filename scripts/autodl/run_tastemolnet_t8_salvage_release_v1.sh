#!/usr/bin/env bash
# Read-only T8 branch salvage -> managed adoption -> persistent T13 relay.

set -euo pipefail

: "${T8_REPO_ROOT:?set the immutable deployed repository root}"
: "${T8_SALVAGE_CONTROLLER_ROOT:?set one fresh controller root}"
: "${T8_SOURCE_ATTEMPT_ID:?set the completed source attempt UUID}"
: "${T8_TARGET_0_ROOT:?set the completed target-0 branch root}"
: "${T8_TARGET_2_ROOT:?set the completed target-2 branch root}"
: "${TASTEMOLNET_T3_OUTPUT:?set T3 PASS root}"
: "${TASTEMOLNET_T4_OUTPUT:?set T4 PASS root}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set frozen GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set frozen train CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set frozen calibration CSV for the persisted T13 relay}"
: "${TASTEMOLNET_TEST_CSV:?set held-out test CSV for the persisted T13 relay}"
: "${TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT:?set pinned official source root}"
: "${MOLCLR_ROOT:?set pinned MolCLR source root for the persisted T13 relay}"
: "${MOLCLR_CHECKPOINT:?set pinned MolCLR checkpoint for the persisted T13 relay}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set frozen threshold contract for the persisted T13 relay}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T8_SALVAGE_GPU_INDEX:-0}
POLL_SECONDS=${T8_SALVAGE_POLL_SECONDS:-60}
BASE=${T8_SALVAGE_BASE:-$RUNTIME/outputs/autodl/tastemolnet/globalgce/t8-salvage}
LOCK_FILE=$CONTROL/tastemolnet-t8-branch-salvage.lock

[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "T8 salvage refuses GNN ablation" >&2; exit 64; }
[[ ! -e "$T8_SALVAGE_CONTROLLER_ROOT/controller.pid" ]] || { echo "T8 salvage controller root is not fresh" >&2; exit 73; }
mkdir -p "$T8_SALVAGE_CONTROLLER_ROOT" "$CONTROL" "$BASE"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "another T8 salvage controller is active" >&2; exit 73; }
cd "$T8_REPO_ROOT"
export PYTHONPATH=$PWD
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

HEARTBEAT=$T8_SALVAGE_CONTROLLER_ROOT/heartbeat.json
write_heartbeat() {
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; q=p+".tmp."+str(os.getpid()); f=open(q,"x"); json.dump({"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"written_at_unix":int(time.time())},f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(q,p)' "$HEARTBEAT" "$1" "${2:-0}"
  printf '%s\n' "$1" > "$T8_SALVAGE_CONTROLLER_ROOT/state.tmp.$$"
  mv "$T8_SALVAGE_CONTROLLER_ROOT/state.tmp.$$" "$T8_SALVAGE_CONTROLLER_ROOT/state"
}
printf '%s\n' "$$" > "$T8_SALVAGE_CONTROLLER_ROOT/controller.pid"
write_heartbeat WAITING_FOR_SALVAGE_GPU 0
while true; do
  gpu_processes=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  [[ -z "$gpu_processes" ]] && break
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_SALVAGE_GPU 0
done

new_salvage_attempt() {
  SALVAGE_ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
  MANAGED_ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
  RUN_ROOT=$BASE/attempt-$SALVAGE_ATTEMPT_ID
  STATE_ROOT=$RUN_ROOT/private-state
  OUTPUT_ROOT=$RUN_ROOT/deadline-output
  RERUN_REQUEST=$RUN_ROOT/single-branch-rerun-request.json
  ADOPTION_STAGE=$RUN_ROOT/managed-stage
  ADOPTION_FINAL=$RUN_ROOT/managed-final
  mkdir -p "$RUN_ROOT" "$ADOPTION_STAGE"
  {
    printf 'salvage_attempt_id=%s\n' "$SALVAGE_ATTEMPT_ID"
    printf 'managed_attempt_id=%s\n' "$MANAGED_ATTEMPT_ID"
    printf 'target_0_root=%s\n' "$T8_TARGET_0_ROOT"
    printf 'target_2_root=%s\n' "$T8_TARGET_2_ROOT"
    printf 'state_root=%s\n' "$STATE_ROOT"
    printf 'deadline_output_root=%s\n' "$OUTPUT_ROOT"
    printf 'managed_final_root=%s\n' "$ADOPTION_FINAL"
    printf 'gpu_index=%s\n' "$GPU_INDEX"
  } > "$T8_SALVAGE_CONTROLLER_ROOT/launch.env"
}

run_salvage_attempt() {
  local log=$1
  "$PY" scripts/autodl/salvage_tastemolnet_t8_branches_v1.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --attempt-id "$SALVAGE_ATTEMPT_ID" \
    --source-attempt-id "$T8_SOURCE_ATTEMPT_ID" \
    --target-0-root "$T8_TARGET_0_ROOT" \
    --target-2-root "$T8_TARGET_2_ROOT" \
    --t3-output "$TASTEMOLNET_T3_OUTPUT" \
    --t4-output "$TASTEMOLNET_T4_OUTPUT" \
    --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
    --train-csv "$TASTEMOLNET_TRAIN_CSV" \
    --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
    --state-root "$STATE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --rerun-request "$RERUN_REQUEST" \
    --device cuda:0 > "$log" 2>&1 &
  SALVAGE_PID=$!
  write_heartbeat T8_SALVAGE_RUNNING "$SALVAGE_PID"
  while kill -0 "$SALVAGE_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    write_heartbeat T8_SALVAGE_RUNNING "$SALVAGE_PID"
  done
  wait "$SALVAGE_PID"
}

new_salvage_attempt
FIRST_ATTEMPT_ID=$SALVAGE_ATTEMPT_ID
FIRST_RERUN_REQUEST=$RERUN_REQUEST
if ! run_salvage_attempt "$T8_SALVAGE_CONTROLLER_ROOT/salvage-attempt-1.log"; then
  write_heartbeat T8_SINGLE_BRANCH_RERUN_REQUIRED 0
  INVALID_TARGET=$($PY -c 'import json,sys; v=json.load(open(sys.argv[1])); x=v.get("invalid_target_branches"); assert isinstance(x,list) and len(x)==1 and x[0] in (0,2); print(x[0])' "$FIRST_RERUN_REQUEST")
  BRANCH_ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
  BRANCH_BASE=$BASE/single-branch-$BRANCH_ATTEMPT_ID
  BRANCH_STATE=$BRANCH_BASE/state
  BRANCH_SCRATCH=$BRANCH_BASE/gspan-scratch
  mkdir -p "$BRANCH_BASE"
  {
    printf 'source_salvage_attempt_id=%s\n' "$FIRST_ATTEMPT_ID"
    printf 'branch_attempt_id=%s\n' "$BRANCH_ATTEMPT_ID"
    printf 'target=%s\n' "$INVALID_TARGET"
    printf 'state_root=%s\n' "$BRANCH_STATE"
  } > "$T8_SALVAGE_CONTROLLER_ROOT/single-branch-recovery.env"
  write_heartbeat "T8_TARGET_${INVALID_TARGET}_FRESH_RECOVERY" 0
  "$PY" scripts/autodl/rerun_tastemolnet_t8_single_branch_v1.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --attempt-id "$BRANCH_ATTEMPT_ID" \
    --source-attempt-id "$T8_SOURCE_ATTEMPT_ID" \
    --target "$INVALID_TARGET" \
    --t3-output "$TASTEMOLNET_T3_OUTPUT" \
    --t4-output "$TASTEMOLNET_T4_OUTPUT" \
    --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
    --train-csv "$TASTEMOLNET_TRAIN_CSV" \
    --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
    --state-root "$BRANCH_STATE" \
    --gspan-scratch-root "$BRANCH_SCRATCH" \
    --device cuda:0 > "$T8_SALVAGE_CONTROLLER_ROOT/single-branch-recovery.log" 2>&1
  if [[ "$INVALID_TARGET" == "0" ]]; then
    T8_TARGET_0_ROOT=$BRANCH_STATE/target-0
  else
    T8_TARGET_2_ROOT=$BRANCH_STATE/target-2
  fi
  new_salvage_attempt
  if ! run_salvage_attempt "$T8_SALVAGE_CONTROLLER_ROOT/salvage-attempt-2.log"; then
    write_heartbeat T8_BOUNDED_SINGLE_BRANCH_RECOVERY_FAILED 0
    exit 75
  fi
fi
FINAL_SALVAGE_LOG=$T8_SALVAGE_CONTROLLER_ROOT/salvage-attempt-2.log
[[ -s "$FINAL_SALVAGE_LOG" ]] || FINAL_SALVAGE_LOG=$T8_SALVAGE_CONTROLLER_ROOT/salvage-attempt-1.log
grep -Fx '[TASTE_T8_SALVAGE_PASS]' "$FINAL_SALVAGE_LOG" >/dev/null
grep -Fx '[TASTE_T8_GLOBALGCE_SMOKE_PASS]' "$FINAL_SALVAGE_LOG" >/dev/null

write_heartbeat T8_MANAGED_ADOPTION 0
AUTO_TERMINATE_UNCONTROLLED_CHILDREN=0 "$PY" \
  scripts/autodl/adopt_tastemolnet_t8_deadline_v2.py \
  --mode run \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --deadline-output-root "$OUTPUT_ROOT" \
  --deadline-state-root "$STATE_ROOT" \
  --deadline-attempt-id "$SALVAGE_ATTEMPT_ID" \
  --recovery-source-attempt-id "$T8_SOURCE_ATTEMPT_ID" \
  --t3-output "$TASTEMOLNET_T3_OUTPUT" \
  --t4-output "$TASTEMOLNET_T4_OUTPUT" \
  --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
  --train-csv "$TASTEMOLNET_TRAIN_CSV" \
  --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
  --stage-root "$ADOPTION_STAGE" \
  --final-path "$ADOPTION_FINAL" \
  --managed-attempt-id "$MANAGED_ATTEMPT_ID" \
  --run-id "tastemolnet-t8-salvage-$SALVAGE_ATTEMPT_ID" \
  > "$T8_SALVAGE_CONTROLLER_ROOT/adoption.log" 2>&1

export T8_PASS_ROOT="$ADOPTION_FINAL"
export T13_REPO_ROOT="$T8_REPO_ROOT"
export GLOBALGCE_OFFICIAL_ROOT="$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT"
bash scripts/autodl/launch_tastemolnet_t13_after_t8_salvage_v1.sh \
  > "$T8_SALVAGE_CONTROLLER_ROOT/t13-relay-launch.txt"
printf '%s\n' "$ADOPTION_FINAL" > "$T8_SALVAGE_CONTROLLER_ROOT/completed_t8_root"
write_heartbeat PASS_AND_T13_RELAY_PERSISTED 0
