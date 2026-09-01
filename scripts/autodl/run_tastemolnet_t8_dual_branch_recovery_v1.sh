#!/usr/bin/env bash
# Fixed TasteMolNet T8 dual-branch recovery from the sealed 7c8 0/0 receipt.

set -euo pipefail

: "${T8_REPO_ROOT:?set the immutable deployed repository root}"
: "${T8_DUAL_CONTROLLER_ROOT:?set one fresh dual-branch controller root}"
: "${TASTEMOLNET_T3_OUTPUT:?set T3 PASS root}"
: "${TASTEMOLNET_T4_OUTPUT:?set T4 PASS root}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set frozen GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set frozen train CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set frozen calibration CSV for T13}"
: "${TASTEMOLNET_TEST_CSV:?set held-out test CSV for T13}"
: "${TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT:?set pinned official source root}"
: "${MOLCLR_ROOT:?set pinned MolCLR source root for T13}"
: "${MOLCLR_CHECKPOINT:?set pinned MolCLR checkpoint for T13}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set frozen threshold contract for T13}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T8_DUAL_GPU_INDEX:-1}
POLL_SECONDS=${T8_DUAL_POLL_SECONDS:-60}
BASE=${T8_DUAL_BASE:-$RUNTIME/outputs/autodl/tastemolnet/globalgce/t8-dual-branch-recovery}
GSPAN_BASE=${T8_DUAL_GSPAN_BASE:-/dev/shm/tastemolnet-t8-dual-branch-recovery}
FAILED_ATTEMPT_ID=7c8cafa6-6679-49d7-bdc6-8d6259a0fbf4
FAILED_SALVAGE_ATTEMPT_ID=fadc2ac6-d1e8-4ede-b526-e06d0744eb8e
FAILED_STATE_ROOT=$RUNTIME/outputs/autodl/tastemolnet/globalgce/t8-smoke/state-$FAILED_ATTEMPT_ID
FAILURE_RECEIPT=$RUNTIME/outputs/autodl/tastemolnet/globalgce/t8-salvage/attempt-$FAILED_SALVAGE_ATTEMPT_ID/single-branch-rerun-request.json
LOCK_FILE=$CONTROL/tastemolnet-t8-dual-branch-recovery-v1.lock
HEARTBEAT=$T8_DUAL_CONTROLLER_ROOT/heartbeat.json
STATE=$T8_DUAL_CONTROLLER_ROOT/state
FAILURE_BINDING=$T8_DUAL_CONTROLLER_ROOT/source-failure-binding.json
INPUT_BINDING=$T8_DUAL_CONTROLLER_ROOT/science-input-binding.json

[[ "$GPU_INDEX" == "1" ]] || { echo "T8 dual-branch recovery is pinned to GPU1" >&2; exit 64; }
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || { echo "T8 dual-branch recovery refuses GNN ablation" >&2; exit 64; }
[[ ! -e "$T8_DUAL_CONTROLLER_ROOT/controller.pid" ]] || { echo "T8 dual-branch controller root is not fresh" >&2; exit 73; }
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "T8 dual-branch poll interval is invalid" >&2; exit 64; }
mkdir -p "$T8_DUAL_CONTROLLER_ROOT" "$CONTROL" "$BASE" "$GSPAN_BASE"
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "another T8 dual-branch recovery relay is active" >&2; exit 73; }

cd "$T8_REPO_ROOT"
export PYTHONPATH=$PWD
export PYTHONHASHSEED=7
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; q=p+".tmp."+str(os.getpid()); f=open(q,"x"); json.dump({"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"source_attempt_id":sys.argv[4],"gpu_index":1,"gnn_ablation_started":False,"written_at_unix":int(time.time())},f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(q,p)' "$HEARTBEAT" "$phase" "$science_pid" "$FAILED_ATTEMPT_ID"
  printf '%s\n' "$phase" > "$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

verify_failure_receipt() {
  local mode=$1
  "$PY" -c '
import hashlib,json,os,sys
from pathlib import Path

receipt=Path(sys.argv[1]); state=Path(sys.argv[2]); binding=Path(sys.argv[3]); mode=sys.argv[4]
if not receipt.is_absolute() or Path(os.path.abspath(receipt)) != receipt or receipt.resolve(strict=True) != receipt or not receipt.is_file():
    raise SystemExit("T8_DUAL_FAILURE_RECEIPT_NOT_ONE_REAL_FILE")
if not state.is_absolute() or Path(os.path.abspath(state)) != state or state.resolve(strict=True) != state or not state.is_dir():
    raise SystemExit("T8_DUAL_FAILED_STATE_NOT_ONE_REAL_DIRECTORY")
for target in (0,2):
    branch=state/f"target-{target}"
    if branch.resolve(strict=True) != branch or not branch.is_dir():
        raise SystemExit(f"T8_DUAL_FAILED_TARGET_{target}_MISSING")
payload=receipt.read_bytes(); value=json.loads(payload)
expected={
    "schema_version":"tastemolnet_t8_single_branch_rerun_request_v1",
    "status":"RERUN_REQUIRED",
    "invalid_target_branches":[0,2],
    "valid_target_branches_preserved":[],
    "rerun_both_branches":True,
    "source_artifacts_mutated":False,
}
for key,want in expected.items():
    if value.get(key) != want:
        raise SystemExit(f"T8_DUAL_FAILURE_RECEIPT_FIELD_MISMATCH:{key}")
reasons=value.get("reasons")
if type(reasons) is not dict or set(reasons) != {"0","2"}:
    raise SystemExit("T8_DUAL_FAILURE_RECEIPT_REASONS_MISSING")
needle="T8 salvage native rule application produced no candidates for [0, 2]"
if any(type(reasons[key]) is not str or needle not in reasons[key] for key in ("0","2")):
    raise SystemExit("T8_DUAL_FAILURE_RECEIPT_NOT_ZERO_ZERO")
sha=hashlib.sha256(payload).hexdigest()
record={
    "schema_version":"tastemolnet_t8_dual_branch_source_failure_binding_v1",
    "status":"PASS",
    "failed_attempt_id":sys.argv[5],
    "failed_state_root":str(state),
    "failure_receipt":str(receipt),
    "failure_receipt_sha256":sha,
    "invalid_target_branches":[0,2],
    "valid_candidate_counts":{"0":0,"2":0},
    "source_artifacts_mutated":False,
}
if mode == "create":
    if binding.exists() or binding.is_symlink():
        raise SystemExit("T8_DUAL_FAILURE_BINDING_NOT_FRESH")
    temporary=binding.with_name(f".{binding.name}.{os.getpid()}.tmp")
    with temporary.open("x",encoding="utf-8") as handle:
        json.dump(record,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,binding)
elif mode == "verify":
    if json.loads(binding.read_text(encoding="utf-8")) != record:
        raise SystemExit("T8_DUAL_FAILURE_BINDING_CHANGED")
else:
    raise SystemExit("T8_DUAL_FAILURE_BINDING_MODE_INVALID")
' "$FAILURE_RECEIPT" "$FAILED_STATE_ROOT" "$FAILURE_BINDING" "$mode" "$FAILED_ATTEMPT_ID"
}

capture_or_verify_science_inputs() {
  local mode=$1
  "$PY" -c '
import hashlib,json,os,sys
from pathlib import Path

binding=Path(sys.argv[1]); mode=sys.argv[2]
entries={
    "t3_verification":Path(sys.argv[3])/"verification.json",
    "t4_verification":Path(sys.argv[4])/"verification.json",
    "gine_model":Path(sys.argv[5])/"model.pt",
    "gine_feature_schema":Path(sys.argv[5])/"feature_schema.json",
    "train_csv":Path(sys.argv[6]),
}
record={"schema_version":"tastemolnet_t8_dual_branch_science_input_binding_v1","status":"PASS","test_loaded":False,"calibration_loaded":False,"files":{}}
for name,path in entries.items():
    if not path.is_absolute() or Path(os.path.abspath(path)) != path or path.resolve(strict=True) != path or not path.is_file():
        raise SystemExit(f"T8_DUAL_INPUT_NOT_ONE_REAL_FILE:{name}")
    payload=path.read_bytes()
    record["files"][name]={"path":str(path),"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}
official=Path(sys.argv[7])
if not official.is_absolute() or Path(os.path.abspath(official)) != official or official.resolve(strict=True) != official or not official.is_dir():
    raise SystemExit("T8_DUAL_OFFICIAL_ROOT_NOT_ONE_REAL_DIRECTORY")
record["official_root"]=str(official)
if mode == "create":
    if binding.exists() or binding.is_symlink():
        raise SystemExit("T8_DUAL_INPUT_BINDING_NOT_FRESH")
    temporary=binding.with_name(f".{binding.name}.{os.getpid()}.tmp")
    with temporary.open("x",encoding="utf-8") as handle:
        json.dump(record,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,binding)
elif mode == "verify":
    if json.loads(binding.read_text(encoding="utf-8")) != record:
        raise SystemExit("T8_DUAL_SCIENCE_INPUTS_CHANGED")
else:
    raise SystemExit("T8_DUAL_INPUT_BINDING_MODE_INVALID")
' "$INPUT_BINDING" "$mode" "$TASTEMOLNET_T3_OUTPUT" "$TASTEMOLNET_T4_OUTPUT" "$TASTEMOLNET_GNN_CHECKPOINT" "$TASTEMOLNET_TRAIN_CSV" "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT"
}

validate_branch_receipt() {
  local target=$1 attempt_id=$2 state_root=$3
  "$PY" -c '
import hashlib,json,os,sys
from pathlib import Path

target=int(sys.argv[1]); attempt=sys.argv[2]; source=sys.argv[3]; state=Path(sys.argv[4]); model=Path(sys.argv[5])/"model.pt"
receipt=state/"single_branch_recovery.json"; branch=state/f"target-{target}"
if state.resolve(strict=True) != state or branch.resolve(strict=True) != branch:
    raise SystemExit("T8_DUAL_RECOVERED_BRANCH_PATH_CHANGED")
value=json.loads(receipt.read_text(encoding="utf-8"))
expected_model=hashlib.sha256(model.read_bytes()).hexdigest()
if (
    value.get("schema_version") != "tastemolnet_t8_single_branch_recovery_v1"
    or value.get("status") != "PASS"
    or value.get("attempt_id") != attempt
    or value.get("recovery_source_attempt_id") != source
    or value.get("target_label") != target
    or value.get("source_label") != 1
    or value.get("state_root") != str(state)
    or value.get("branch_root") != str(branch)
    or value.get("oracle_checkpoint_hash") != expected_model
    or type(value.get("raw_generated_count")) is not int
    or value["raw_generated_count"] < 1
    or value.get("other_target_rerun") is not False
    or value.get("test_loaded") is not False
    or value.get("calibration_loaded") is not False
    or value.get("gnn_ablation_started") is not False
):
    raise SystemExit(f"T8_DUAL_TARGET_{target}_RECOVERY_RECEIPT_INVALID")
evidence=value.get("branch_evidence")
if type(evidence) is not dict or evidence.get("target_label") != target or evidence.get("source_label") != 1 or evidence.get("num_classes") != 3 or evidence.get("test_loaded") is not False or evidence.get("calibration_loaded") is not False or evidence.get("rf_oracle_used") is not False:
    raise SystemExit(f"T8_DUAL_TARGET_{target}_SCIENCE_BOUNDARY_INVALID")
' "$target" "$attempt_id" "$FAILED_ATTEMPT_ID" "$state_root" "$TASTEMOLNET_GNN_CHECKPOINT"
}

run_branch() {
  local target=$1 attempt_id=$2 state_root=$3 scratch_root=$4 log=$5
  verify_failure_receipt verify
  capture_or_verify_science_inputs verify
  "$PY" scripts/autodl/rerun_tastemolnet_t8_single_branch_v1.py \
    --config configs/hpc.yaml \
    --set inference.fallback_to_heuristic=false \
    --attempt-id "$attempt_id" \
    --source-attempt-id "$FAILED_ATTEMPT_ID" \
    --target "$target" \
    --t3-output "$TASTEMOLNET_T3_OUTPUT" \
    --t4-output "$TASTEMOLNET_T4_OUTPUT" \
    --gnn-checkpoint "$TASTEMOLNET_GNN_CHECKPOINT" \
    --train-csv "$TASTEMOLNET_TRAIN_CSV" \
    --official-root "$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT" \
    --state-root "$state_root" \
    --gspan-scratch-root "$scratch_root" \
    --device cuda:0 > "$log" 2>&1 &
  local branch_pid=$!
  write_heartbeat "T8_TARGET_${target}_FRESH_RECOVERY_RUNNING" "$branch_pid"
  while kill -0 "$branch_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    write_heartbeat "T8_TARGET_${target}_FRESH_RECOVERY_RUNNING" "$branch_pid"
  done
  if ! wait "$branch_pid"; then
    write_heartbeat "T8_TARGET_${target}_FRESH_RECOVERY_FAILED" 0
    return 75
  fi
  validate_branch_receipt "$target" "$attempt_id" "$state_root"
  grep -Fx '[TASTE_T8_SINGLE_BRANCH_RECOVERY_PASS]' "$log" >/dev/null
  printf '%s\n' "[TASTE_T8_TARGET_${target}_FRESH_RECOVERY_PASS]" > "$T8_DUAL_CONTROLLER_ROOT/TASTE_T8_TARGET_${target}_FRESH_RECOVERY_PASS"
  write_heartbeat "T8_TARGET_${target}_FRESH_RECOVERY_PASS" 0
}

printf '%s\n' "$$" > "$T8_DUAL_CONTROLLER_ROOT/controller.pid"
write_heartbeat VALIDATING_7C8_ZERO_ZERO_RECEIPT 0
verify_failure_receipt create
capture_or_verify_science_inputs create

TARGET_0_ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
TARGET_2_ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
[[ "$TARGET_0_ATTEMPT_ID" != "$TARGET_2_ATTEMPT_ID" ]] || { echo "T8 dual-branch UUID collision" >&2; exit 70; }
TARGET_0_BASE=$BASE/target-0-attempt-$TARGET_0_ATTEMPT_ID
TARGET_2_BASE=$BASE/target-2-attempt-$TARGET_2_ATTEMPT_ID
TARGET_0_STATE=$TARGET_0_BASE/state
TARGET_2_STATE=$TARGET_2_BASE/state
TARGET_0_SCRATCH=$GSPAN_BASE/target-0-attempt-$TARGET_0_ATTEMPT_ID
TARGET_2_SCRATCH=$GSPAN_BASE/target-2-attempt-$TARGET_2_ATTEMPT_ID
for path in "$TARGET_0_BASE" "$TARGET_2_BASE" "$TARGET_0_STATE" "$TARGET_2_STATE" "$TARGET_0_SCRATCH" "$TARGET_2_SCRATCH"; do
  [[ ! -e "$path" && ! -L "$path" ]] || { echo "T8 dual-branch path is not fresh: $path" >&2; exit 74; }
done
mkdir -p "$TARGET_0_BASE" "$TARGET_2_BASE"
{
  printf 'failed_attempt_id=%s\n' "$FAILED_ATTEMPT_ID"
  printf 'failure_receipt=%s\n' "$FAILURE_RECEIPT"
  printf 'target_0_attempt_id=%s\n' "$TARGET_0_ATTEMPT_ID"
  printf 'target_0_state_root=%s\n' "$TARGET_0_STATE"
  printf 'target_2_attempt_id=%s\n' "$TARGET_2_ATTEMPT_ID"
  printf 'target_2_state_root=%s\n' "$TARGET_2_STATE"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
  printf 'sequential_branches=true\n'
  printf 'gnn_ablation_started=false\n'
} > "$T8_DUAL_CONTROLLER_ROOT/launch.env"

write_heartbeat WAITING_FOR_GPU1 0
while true; do
  gpu_processes=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -z "$gpu_processes" ]]; then
    GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "$GPU_UUID" =~ ^GPU-[A-Za-z0-9-]+$ ]] || { echo "T8 dual-branch physical GPU UUID is invalid" >&2; exit 64; }
    mkdir -p "$RUNTIME/locks"
    exec 8>>"$RUNTIME/locks/gpu-$GPU_UUID.coordination.lock"
    if flock -n 8; then
      gpu_processes=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
      if [[ -z "$gpu_processes" ]]; then
        break
      fi
      flock -u 8
    fi
    exec 8>&-
  fi
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GPU1 0
done

run_branch 0 "$TARGET_0_ATTEMPT_ID" "$TARGET_0_STATE" "$TARGET_0_SCRATCH" "$T8_DUAL_CONTROLLER_ROOT/target-0.log"
run_branch 2 "$TARGET_2_ATTEMPT_ID" "$TARGET_2_STATE" "$TARGET_2_SCRATCH" "$T8_DUAL_CONTROLLER_ROOT/target-2.log"
verify_failure_receipt verify
capture_or_verify_science_inputs verify

# The existing salvage relay performs its own fresh GPU admission.  Release the
# dual-branch lock first so there is no self-deadlock and no preferential claim.
flock -u 8
exec 8>&-
write_heartbeat T8_DUAL_BRANCH_RECOVERY_PASS_HANDING_TO_EXISTING_CHAIN 0

DOWNSTREAM_ROOT=$T8_DUAL_CONTROLLER_ROOT/downstream-salvage
[[ ! -e "$DOWNSTREAM_ROOT" && ! -L "$DOWNSTREAM_ROOT" ]] || { echo "T8 downstream salvage root is not fresh" >&2; exit 74; }
export T8_SALVAGE_CONTROLLER_ROOT=$DOWNSTREAM_ROOT
export T8_SOURCE_ATTEMPT_ID=$FAILED_ATTEMPT_ID
export T8_TARGET_0_ROOT=$TARGET_0_STATE/target-0
export T8_TARGET_2_ROOT=$TARGET_2_STATE/target-2
export T8_SALVAGE_GPU_INDEX=1
export RUN_GNN_ABLATION=0
export GLOBALGCE_OFFICIAL_ROOT=$TASTEMOLNET_GLOBALGCE_OFFICIAL_ROOT

bash scripts/autodl/run_tastemolnet_t8_salvage_release_v1.sh \
  > "$T8_DUAL_CONTROLLER_ROOT/downstream-salvage.log" 2>&1 &
DOWNSTREAM_PID=$!
write_heartbeat EXISTING_SALVAGE_ADOPTION_T13_CHAIN_RUNNING "$DOWNSTREAM_PID"
while kill -0 "$DOWNSTREAM_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
  write_heartbeat EXISTING_SALVAGE_ADOPTION_T13_CHAIN_RUNNING "$DOWNSTREAM_PID"
done
if ! wait "$DOWNSTREAM_PID"; then
  write_heartbeat EXISTING_SALVAGE_ADOPTION_T13_CHAIN_FAILED 0
  exit 75
fi
[[ -s "$DOWNSTREAM_ROOT/completed_t8_root" ]] || { echo "T8 downstream chain did not publish its managed root locator" >&2; exit 75; }
"$PY" -c 'import os,sys; source=sys.argv[1]; destination=sys.argv[2]; payload=open(source,"rb").read(); temporary=destination+".tmp."+str(os.getpid()); handle=open(temporary,"xb"); handle.write(payload); handle.flush(); os.fsync(handle.fileno()); handle.close(); os.replace(temporary,destination); parent=os.open(os.path.dirname(destination),os.O_RDONLY|getattr(os,"O_DIRECTORY",0)); os.fsync(parent); os.close(parent)' \
  "$DOWNSTREAM_ROOT/completed_t8_root" \
  "$T8_DUAL_CONTROLLER_ROOT/completed_t8_root"
write_heartbeat PASS_AND_T13_RELAY_PERSISTED 0
