#!/usr/bin/env bash
# Narrow durable T14 queue: exact generation PASS -> postprocess -> locator.

set -euo pipefail
umask 077

: "${T14_RELAY_REPO_ROOT:?set one immutable deployed repository root}"
: "${T14_POSTPROCESS_CONTROLLER_ROOT:?set one fresh controller root}"
: "${T14_GENERATION_ROOT:?set the active fresh T14 generation root}"
: "${T14_GENERATION_LAUNCHER_JSON:?set its immutable launcher.json evidence}"
: "${T14_GENERATION_EXECUTION_COMMIT:?set the exact generation commit}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set the frozen Taste calibration CSV}"
: "${TASTEMOLNET_TEST_CSV:?set the frozen Taste held-out test CSV}"
: "${TASTEMOLNET_T3_OUTPUT_ROOT:?set the frozen T3 root}"
: "${MOLCLR_ROOT:?set the pinned MolCLR source root}"
: "${MOLCLR_CHECKPOINT:?set the pinned MolCLR checkpoint}"
: "${TASTEMOLNET_WNODE_THRESHOLD_JSON:?set the shared frozen threshold}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T14_POSTPROCESS_GPU_INDEX:-2}
POLL_SECONDS=${T14_POSTPROCESS_POLL_SECONDS:-60}
POSTPROCESS_BASE=${T14_POSTPROCESS_BASE:-$RUNTIME/outputs/autodl/tastemolnet/comrecgc/t14-postprocess}
LOCK_FILE=$CONTROL/tastemolnet-t14-comrecgc-postprocess-relay.lock
EXPECTED_GENERATION_PASS='[TASTE_T14_COMRECGC_FULL_GENERATION_PASS]'
EXPECTED_FINAL_PASS='[TASTE_COMRECGC_PASS]'

[[ "$GPU_INDEX" == "2" ]] || {
  echo "T14 postprocess relay is pinned to physical GPU2" >&2
  exit 64
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "T14_POSTPROCESS_POLL_SECONDS must be a positive integer" >&2
  exit 64
}
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || {
  echo "T14 relay refuses to run while GNN ablation is enabled" >&2
  exit 64
}
[[ "$T14_GENERATION_EXECUTION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "T14 generation commit must be one full lowercase SHA-1" >&2
  exit 64
}
for path in \
  "$T14_RELAY_REPO_ROOT" "$T14_POSTPROCESS_CONTROLLER_ROOT" \
  "$T14_GENERATION_ROOT" "$T14_GENERATION_LAUNCHER_JSON" \
  "$TASTEMOLNET_CALIBRATION_CSV" "$TASTEMOLNET_TEST_CSV" \
  "$TASTEMOLNET_T3_OUTPUT_ROOT" "$MOLCLR_ROOT" "$MOLCLR_CHECKPOINT" \
  "$TASTEMOLNET_WNODE_THRESHOLD_JSON" "$POSTPROCESS_BASE"; do
  [[ "$path" == /* ]] || { echo "absolute path required: $path" >&2; exit 64; }
done
[[ -x "$PY" && "$PY" == /* ]] || { echo "invalid AutoDL Python: $PY" >&2; exit 64; }
[[ -d "$T14_RELAY_REPO_ROOT" && ! -L "$T14_RELAY_REPO_ROOT" ]] \
  || { echo "relay repository is not one physical directory" >&2; exit 64; }
[[ -d "$T14_GENERATION_ROOT" && ! -L "$T14_GENERATION_ROOT" ]] \
  || { echo "generation root is not one physical directory" >&2; exit 64; }
[[ -f "$T14_GENERATION_LAUNCHER_JSON" && ! -L "$T14_GENERATION_LAUNCHER_JSON" ]] \
  || { echo "generation launcher evidence is not one physical file" >&2; exit 64; }

RELAY_COMMIT=$(git -C "$T14_RELAY_REPO_ROOT" rev-parse HEAD)
[[ "$RELAY_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "relay repository commit is malformed" >&2; exit 64; }
[[ -z "$(git -C "$T14_RELAY_REPO_ROOT" status --porcelain)" ]] \
  || { echo "relay repository is not immutable/clean" >&2; exit 64; }

mkdir -p "$T14_POSTPROCESS_CONTROLLER_ROOT" "$CONTROL" "$POSTPROCESS_BASE"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another T14 postprocess relay holds $LOCK_FILE" >&2
  exit 73
fi
[[ ! -e "$T14_POSTPROCESS_CONTROLLER_ROOT/controller.pid" ]] \
  || { echo "T14 relay controller root is not fresh" >&2; exit 73; }

HEARTBEAT=$T14_POSTPROCESS_CONTROLLER_ROOT/heartbeat.json
STATE=$T14_POSTPROCESS_CONTROLLER_ROOT/state
LOCATOR=$T14_POSTPROCESS_CONTROLLER_ROOT/cell_root_locator.json

atomic_json() {
  local destination=$1
  shift
  "$PY" -I -B - "$destination" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

destination = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
descriptor, name = tempfile.mkstemp(
    prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
)
temporary = Path(name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    temporary.unlink(missing_ok=True)
PY
}

write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  local now payload
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  payload=$("$PY" -I -B -c \
    'import json,os,sys; print(json.dumps({"schema_version":"tastemolnet_t14_postprocess_relay_heartbeat_v1","controller_pid":os.getppid(),"phase":sys.argv[1],"science_pid":int(sys.argv[2]),"generation_root":sys.argv[3],"postprocess_root":sys.argv[4],"final_root":sys.argv[5],"gpu_index":int(sys.argv[6]),"relay_commit":sys.argv[7],"written_at":sys.argv[8]},sort_keys=True))' \
    "$phase" "$science_pid" "$T14_GENERATION_ROOT" "$SCIENCE_ROOT" \
    "$FINAL_ROOT" "$GPU_INDEX" "$RELAY_COMMIT" "$now")
  atomic_json "$HEARTBEAT" "$payload"
  printf '%s\n' "$phase" >"$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

process_start_ticks() {
  "$PY" -I -B -c \
    'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[19])' \
    "$1" 2>/dev/null
}

process_cmdline() {
  "$PY" -I -B -c \
    'from pathlib import Path; import sys; print(Path(f"/proc/{sys.argv[1]}/cmdline").read_bytes().replace(b"\0",b" ").decode("utf-8","replace"))' \
    "$1" 2>/dev/null
}

file_sha256() {
  "$PY" -I -B -c \
    'import hashlib,sys; h=hashlib.sha256(); f=open(sys.argv[1],"rb"); [h.update(chunk) for chunk in iter(lambda:f.read(1048576),b"")]; print(h.hexdigest())' \
    "$1"
}

validate_pid_identity() {
  local pid=$1 expected_ticks=$2 required_token=$3 cmd observed_ticks
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$expected_ticks" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1
  observed_ticks=$(process_start_ticks "$pid") || return 1
  [[ "$observed_ticks" == "$expected_ticks" ]] || return 1
  cmd=$(process_cmdline "$pid") || return 1
  [[ "$cmd" == *"$required_token"* && "$cmd" == *"$T14_GENERATION_ROOT"* ]]
}

read_launcher_identity() {
  "$PY" -I -B - "$T14_GENERATION_LAUNCHER_JSON" "$T14_GENERATION_ROOT" \
    "$T14_GENERATION_EXECUTION_COMMIT" "$GPU_INDEX" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
root, commit, gpu = sys.argv[2], sys.argv[3], int(sys.argv[4])
value = json.loads(path.read_text(encoding="utf-8"))
if (
    value.get("state") != "STARTED"
    or value.get("output_root") != root
    or value.get("execution_commit") != commit
    or value.get("gpu_index") != gpu
    or type(value.get("launcher_pid")) is not int
    or type(value.get("start_ticks")) is not int
):
    raise SystemExit("generation launcher identity changed")
print(value["launcher_pid"], value["start_ticks"])
PY
}

read_science_identity() {
  "$PY" -I -B - "$T14_GENERATION_ROOT/progress.json" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("schema_version") != "tastemolnet_t14_progress_v1"
    or value.get("status") not in {"RUNNING", "PASS"}
    or type(value.get("pid")) is not int
    or type(value.get("completed_step")) is not int
    or not 0 <= value["completed_step"] <= 25000
):
    raise SystemExit("generation progress identity changed")
print(value["pid"], value["completed_step"])
PY
}

scan_no_writers() {
  "$PY" -I -B - "$T14_RELAY_REPO_ROOT" "$T14_GENERATION_ROOT" \
    "$T14_POSTPROCESS_CONTROLLER_ROOT/generation_writer_audit.json" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, sys.argv[1])
from src.eval.am_legacy_standardization import scan_live_writers

root = Path(sys.argv[2])
destination = Path(sys.argv[3])
payload = scan_live_writers(root, proc_root="/proc")
if payload.get("procfs_verified") is not True or payload.get("writable_fd_count") != 0:
    raise SystemExit("generation writer audit did not pass")
fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
temporary = Path(name)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
PY
}

run_owned_stage() {
  local phase=$1 log=$2
  shift 2
  "$@" >"$log" 2>&1 &
  local child_pid=$!
  write_heartbeat "$phase" "$child_pid"
  while kill -0 "$child_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    write_heartbeat "$phase" "$child_pid"
  done
  local rc=0
  wait "$child_pid" || rc=$?
  return "$rc"
}

printf '%s\n' "$$" >"$T14_POSTPROCESS_CONTROLLER_ROOT/controller.pid"
GENERATION_LAUNCHER_SHA256=$(file_sha256 "$T14_GENERATION_LAUNCHER_JSON")
read -r GENERATION_MANAGER_PID GENERATION_MANAGER_START_TICKS < <(read_launcher_identity)
validate_pid_identity "$GENERATION_MANAGER_PID" "$GENERATION_MANAGER_START_TICKS" \
  "run_tastemolnet_comrecgc_full.py" || {
  [[ -s "$T14_GENERATION_ROOT/GENERATION_PASS" ]] \
    || { echo "generation manager identity is not live before PASS" >&2; exit 75; }
}

SCIENCE_PID=0
SCIENCE_START_TICKS=0
if [[ -s "$T14_GENERATION_ROOT/progress.json" ]]; then
  read -r SCIENCE_PID _ < <(read_science_identity)
  SCIENCE_START_TICKS=$(process_start_ticks "$SCIENCE_PID")
  validate_pid_identity "$SCIENCE_PID" "$SCIENCE_START_TICKS" \
    "run_tastemolnet_comrecgc_full.py" \
    || { echo "generation science identity is not live" >&2; exit 75; }
fi

POSTPROCESS_ID=$("$PY" -I -B -c 'import uuid; print(uuid.uuid4())')
FINAL_ID=$("$PY" -I -B -c 'import uuid; print(uuid.uuid4())')
SCIENCE_ROOT=$POSTPROCESS_BASE/science-attempt-$POSTPROCESS_ID
FINAL_ROOT=$POSTPROCESS_BASE/final-attempt-$FINAL_ID
[[ ! -e "$SCIENCE_ROOT" && ! -L "$SCIENCE_ROOT" ]] \
  || { echo "fresh T14 postprocess science root exists" >&2; exit 76; }
[[ ! -e "$FINAL_ROOT" && ! -L "$FINAL_ROOT" ]] \
  || { echo "fresh T14 postprocess final root exists" >&2; exit 76; }

LAUNCH_PAYLOAD=$("$PY" -I -B -c \
  'import json,sys; print(json.dumps({"schema_version":"tastemolnet_t14_postprocess_relay_launch_v1","generation_root":sys.argv[1],"generation_launcher_json":sys.argv[2],"generation_launcher_sha256":sys.argv[3],"generation_execution_commit":sys.argv[4],"generation_manager_pid":int(sys.argv[5]),"generation_manager_start_ticks":int(sys.argv[6]),"generation_science_pid":int(sys.argv[7]),"generation_science_start_ticks":int(sys.argv[8]),"relay_commit":sys.argv[9],"postprocess_root":sys.argv[10],"final_root":sys.argv[11],"locator":sys.argv[12],"gpu_index":int(sys.argv[13]),"gnn_ablation_started":False},sort_keys=True))' \
  "$T14_GENERATION_ROOT" "$T14_GENERATION_LAUNCHER_JSON" \
  "$GENERATION_LAUNCHER_SHA256" "$T14_GENERATION_EXECUTION_COMMIT" "$GENERATION_MANAGER_PID" \
  "$GENERATION_MANAGER_START_TICKS" "$SCIENCE_PID" "$SCIENCE_START_TICKS" \
  "$RELAY_COMMIT" "$SCIENCE_ROOT" "$FINAL_ROOT" "$LOCATOR" "$GPU_INDEX")
atomic_json "$T14_POSTPROCESS_CONTROLLER_ROOT/launch.json" "$LAUNCH_PAYLOAD"

write_heartbeat WAITING_FOR_GENERATION_PASS 0
while [[ ! -s "$T14_GENERATION_ROOT/GENERATION_PASS" ]]; do
  validate_pid_identity "$GENERATION_MANAGER_PID" "$GENERATION_MANAGER_START_TICKS" \
    "run_tastemolnet_comrecgc_full.py" || {
    sleep 2
    [[ -s "$T14_GENERATION_ROOT/GENERATION_PASS" ]] || {
      write_heartbeat GENERATION_EXITED_WITHOUT_PASS 0
      exit 77
    }
  }
  if [[ "$SCIENCE_PID" -gt 0 ]]; then
    validate_pid_identity "$SCIENCE_PID" "$SCIENCE_START_TICKS" \
      "run_tastemolnet_comrecgc_full.py" || {
      sleep 2
      [[ -s "$T14_GENERATION_ROOT/GENERATION_PASS" ]] || {
        write_heartbeat GENERATION_SCIENCE_EXITED_WITHOUT_PASS 0
        exit 77
      }
    }
  fi
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GENERATION_PASS 0
done
[[ "$(cat "$T14_GENERATION_ROOT/GENERATION_PASS")" == "$EXPECTED_GENERATION_PASS" ]] \
  || { echo "T14 generation marker bytes changed" >&2; exit 78; }
[[ "$(file_sha256 "$T14_GENERATION_LAUNCHER_JSON")" == "$GENERATION_LAUNCHER_SHA256" ]] \
  || { echo "T14 generation launcher evidence changed" >&2; exit 78; }

write_heartbeat WAITING_FOR_GENERATION_WRITERS_TO_CLOSE 0
while validate_pid_identity "$GENERATION_MANAGER_PID" "$GENERATION_MANAGER_START_TICKS" \
  "run_tastemolnet_comrecgc_full.py" \
  || { [[ "$SCIENCE_PID" -gt 0 ]] && validate_pid_identity "$SCIENCE_PID" \
    "$SCIENCE_START_TICKS" "run_tastemolnet_comrecgc_full.py"; }; do
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GENERATION_WRITERS_TO_CLOSE 0
done
scan_no_writers

export AUTODL_DATA_ROOT=${AUTODL_DATA_ROOT:-/autodl-fs/data}
export AUTODL_RUNTIME_ROOT=$RUNTIME
export AUTODL_CONTROL_ROOT=$CONTROL
export AUTODL_PYTHON=$PY
export AUTODL_MAX_GPUS=4
export AUTODL_MIN_FREE_MEMORY_MB=${AUTODL_MIN_FREE_MEMORY_MB:-16000}
export AUTODL_IDLE_UTIL_THRESHOLD=${AUTODL_IDLE_UTIL_THRESHOLD:-10}
export AUTODL_IDLE_STABLE_SECONDS=${AUTODL_IDLE_STABLE_SECONDS:-60}
export RUN_TASTEMOLNET=1
export TASTE_RESEARCH_COMPUTE_ALLOWED=1
export TASTE_PAPER_RESULTS_ALLOWED=1
export TASTE_DATA_REDISTRIBUTION_ALLOWED=0
export RUN_GNN_ABLATION=0
export TASTEMOLNET_T14_GENERATION_ROOT=$T14_GENERATION_ROOT
export TASTEMOLNET_T14_POSTPROCESS_ROOT=$SCIENCE_ROOT
export TASTEMOLNET_T14_FINAL_ROOT=$FINAL_ROOT
export WNODE_CACHE_DB=${WNODE_CACHE_DB:-$RUNTIME/cache/tastemolnet/t14-postprocess/wnode.sqlite}
export NODE_EMBEDDING_CACHE_DIR=${NODE_EMBEDDING_CACHE_DIR:-$RUNTIME/cache/tastemolnet/t14-postprocess/molclr_nodes}
export TASTEMOLNET_T14_POSTPROCESS_RUN_ID=taste-t14-comrecgc-postprocess-$POSTPROCESS_ID
export TASTEMOLNET_T14_POSTPROCESS_GPU_INDEX=$GPU_INDEX
export TASTEMOLNET_T14_POSTPROCESS_RESUME=0
mkdir -p "$(dirname "$WNODE_CACHE_DB")" "$NODE_EMBEDDING_CACHE_DIR"

cd "$T14_RELAY_REPO_ROOT"
while true; do
  set +e
  run_owned_stage T14_POSTPROCESS \
    "$T14_POSTPROCESS_CONTROLLER_ROOT/postprocess.log" \
    bash scripts/autodl/run_tastemolnet_t14_comrecgc_postprocess.sh
  rc=$?
  set -e
  if [[ "$rc" -eq 75 && ! -e "$SCIENCE_ROOT" && ! -e "$FINAL_ROOT" ]]; then
    write_heartbeat WAITING_FOR_IDLE_GPU2 0
    sleep "$POLL_SECONDS"
    continue
  fi
  [[ "$rc" -eq 0 ]] || {
    write_heartbeat T14_POSTPROCESS_FAILED 0
    exit "$rc"
  }
  break
done

[[ -f "$FINAL_ROOT/PASS" && ! -L "$FINAL_ROOT/PASS" ]] \
  || { echo "T14 final PASS is absent" >&2; exit 79; }
[[ "$(cat "$FINAL_ROOT/PASS")" == "$EXPECTED_FINAL_PASS" ]] \
  || { echo "T14 final PASS bytes changed" >&2; exit 79; }
[[ -s "$FINAL_ROOT/final_artifact_audit.json" && -s "$FINAL_ROOT/run_manifest.json" ]] \
  || { echo "T14 final verification closure is incomplete" >&2; exit 79; }

LOCATOR_PAYLOAD=$("$PY" -I -B -c \
  'import json,sys; print(json.dumps({"schema_version":"fast16_matrix_cell_root_locator_v1","status":"READY","dataset":"TasteMolNet","method":"ComRecGC","terminal_root":sys.argv[1]},sort_keys=True))' \
  "$FINAL_ROOT")
atomic_json "$LOCATOR" "$LOCATOR_PAYLOAD"
printf '%s\n' "$FINAL_ROOT" >"$T14_POSTPROCESS_CONTROLLER_ROOT/completed_output_root"
write_heartbeat PAPER_CELL_PASS 0
