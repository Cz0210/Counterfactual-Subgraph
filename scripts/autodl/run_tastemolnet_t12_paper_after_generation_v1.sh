#!/usr/bin/env bash
# Narrow durable T12 queue: generation PASS -> postprocess -> independent verifier.

set -euo pipefail

: "${T12_REPO_ROOT:?set deployed repository root}"
: "${T12_GENERATION_CONTROLLER_ROOT:?set the running generation controller root}"
: "${T12_PAPER_CONTROLLER_ROOT:?set one fresh paper controller root}"
: "${T12_TRAIN_CSV:?set frozen Taste train CSV}"
: "${T12_CALIBRATION_CSV:?set frozen Taste calibration CSV}"
: "${T12_TEST_CSV:?set frozen Taste held-out test CSV}"
: "${T12_GNN_CHECKPOINT:?set T3 artifacts/checkpoint root}"
: "${T12_MOLCLR_ROOT:?set pinned MolCLR source root}"
: "${T12_MOLCLR_CHECKPOINT:?set pinned MolCLR checkpoint}"
: "${T12_WNODE_THRESHOLD_CONTRACT:?set shared frozen WNode threshold contract}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T12_GPU_INDEX:-1}
POLL_SECONDS=${T12_POLL_SECONDS:-60}
LOCK_FILE=$CONTROL/tastemolnet-t12-gcf-paper-postprocess.lock

mkdir -p "$T12_PAPER_CONTROLLER_ROOT" "$CONTROL"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
  echo "another T12 paper sidecar holds $LOCK_FILE" >&2
  exit 73
fi

cd "$T12_REPO_ROOT"
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
export PYTHONHASHSEED=7
export CUBLAS_WORKSPACE_CONFIG=:4096:8

HEARTBEAT=$T12_PAPER_CONTROLLER_ROOT/heartbeat.json
STATE=$T12_PAPER_CONTROLLER_ROOT/state
write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  local temporary=$HEARTBEAT.tmp.$$
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; v={"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"written_at_unix":int(time.time())}; open(p,"w",encoding="utf-8").write(json.dumps(v,sort_keys=True)+"\n")' "$temporary" "$phase" "$science_pid"
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

printf '%s\n' "$$" > "$T12_PAPER_CONTROLLER_ROOT/controller.pid"
write_heartbeat WAITING_FOR_GENERATION_ROOT 0
LAUNCH_ENV=$T12_GENERATION_CONTROLLER_ROOT/launch.env
while [[ ! -s "$LAUNCH_ENV" ]]; do
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GENERATION_ROOT 0
done
GENERATION_ROOT=$(awk -F= '$1 == "output_root" {print substr($0,index($0,"=")+1)}' "$LAUNCH_ENV")
if [[ -z "$GENERATION_ROOT" || "$GENERATION_ROOT" != /* ]]; then
  echo "generation controller launch.env lacks an absolute output_root" >&2
  exit 74
fi

GENERATION_PID_FILE=$T12_GENERATION_CONTROLLER_ROOT/controller.pid
while [[ ! -s "$GENERATION_PID_FILE" ]]; do
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GENERATION_CONTROLLER_PID 0
done
GENERATION_CONTROLLER_PID=$(cat "$GENERATION_PID_FILE")
if [[ ! "$GENERATION_CONTROLLER_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "generation controller PID is malformed" >&2
  exit 75
fi
GENERATION_CONTROLLER_START_TICKS=""
if [[ -r "/proc/$GENERATION_CONTROLLER_PID/stat" ]]; then
  GENERATION_CONTROLLER_START_TICKS=$("$PY" -c 'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[19])' "$GENERATION_CONTROLLER_PID")
fi

write_heartbeat WAITING_FOR_GENERATION_PASS 0
GENERATION_PASS=$GENERATION_ROOT/generation_verification/GENERATION_PASS
while [[ ! -s "$GENERATION_PASS" ]]; do
  if [[ -s "$T12_GENERATION_CONTROLLER_ROOT/state" ]] && grep -q '_FAILED$' "$T12_GENERATION_CONTROLLER_ROOT/state"; then
    echo "T12 generation controller reached a failed state" >&2
    write_heartbeat GENERATION_FAILED 0
    exit 76
  fi
  observed_ticks=""
  if [[ -r "/proc/$GENERATION_CONTROLLER_PID/stat" ]]; then
    observed_ticks=$("$PY" -c 'from pathlib import Path; import sys; raw=Path(f"/proc/{sys.argv[1]}/stat").read_text(); print(raw[raw.rfind(")")+2:].split()[19])' "$GENERATION_CONTROLLER_PID" 2>/dev/null || true)
  fi
  if [[ -z "$observed_ticks" || ( -n "$GENERATION_CONTROLLER_START_TICKS" && "$observed_ticks" != "$GENERATION_CONTROLLER_START_TICKS" ) ]]; then
    recorded_state=$(cat "$T12_GENERATION_CONTROLLER_ROOT/state" 2>/dev/null || printf 'UNKNOWN')
    echo "generation controller exited before PASS: pid=$GENERATION_CONTROLLER_PID state=$recorded_state" >&2
    write_heartbeat GENERATION_CONTROLLER_EXITED_WITHOUT_PASS 0
    exit 77
  fi
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GENERATION_PASS 0
done
if [[ "$(cat "$GENERATION_PASS")" != "[TASTE_T12_GCF_GENERATION_PASS]" ]]; then
  echo "T12 generation marker bytes changed" >&2
  exit 78
fi

PAPER_ROOT=${T12_PAPER_ROOT:-$GENERATION_ROOT/paper_cell}
VERIFY_ROOT=${T12_TERMINAL_VERIFICATION_ROOT:-$GENERATION_ROOT/paper_terminal_verification}
WNODE_CACHE=${T12_WNODE_CACHE_DB:-$RUNTIME/cache/tastemolnet/t12-gcf-full-wnode.sqlite}
NODE_CACHE=${T12_NODE_EMBEDDING_CACHE_DIR:-$RUNTIME/cache/tastemolnet/t12-gcf-full-molclr-nodes}
mkdir -p "$(dirname "$WNODE_CACHE")" "$NODE_CACHE"
{
  printf 'generation_root=%s\n' "$GENERATION_ROOT"
  printf 'paper_root=%s\n' "$PAPER_ROOT"
  printf 'verification_root=%s\n' "$VERIFY_ROOT"
  printf 'gpu_index=%s\n' "$GPU_INDEX"
} > "$T12_PAPER_CONTROLLER_ROOT/launch.env"

COMMON=(
  --config configs/hpc.yaml
  --set inference.fallback_to_heuristic=false
  --generation-root "$GENERATION_ROOT"
  --generation-verification-root "$GENERATION_ROOT/generation_verification"
  --train-csv "$T12_TRAIN_CSV"
  --calibration-csv "$T12_CALIBRATION_CSV"
  --test-csv "$T12_TEST_CSV"
  --gnn-checkpoint "$T12_GNN_CHECKPOINT"
  --molclr-root "$T12_MOLCLR_ROOT"
  --molclr-checkpoint "$T12_MOLCLR_CHECKPOINT"
  --threshold-contract "$T12_WNODE_THRESHOLD_CONTRACT"
  --output-root "$PAPER_ROOT"
)
RESUME=()
if [[ -s "$PAPER_ROOT/checkpoint.json" ]]; then
  RESUME=(--resume)
fi
run_stage T12_PAPER_POSTPROCESS "$T12_PAPER_CONTROLLER_ROOT/postprocess.log" \
  "$PY" scripts/run_tastemolnet_gcf_full_postprocess.py \
  "${COMMON[@]}" --wnode-cache-db "$WNODE_CACHE" \
  --node-embedding-cache-dir "$NODE_CACHE" "${RESUME[@]}"

if [[ ! -s "$PAPER_ROOT/SEALED" ]]; then
  echo "T12 postprocess did not durably seal its paper root" >&2
  exit 79
fi
run_stage T12_TERMINAL_VERIFY "$T12_PAPER_CONTROLLER_ROOT/terminal-verify.log" \
  "$PY" scripts/verify_tastemolnet_gcf_full.py \
  "${COMMON[@]}" --verification-root "$VERIFY_ROOT"

if [[ "$(cat "$PAPER_ROOT/PASS")" != "[TASTE_GCF_PASS]" ]]; then
  echo "T12 independent verifier emitted no exact paper PASS" >&2
  exit 80
fi
printf '%s\n' "$PAPER_ROOT" > "$T12_PAPER_CONTROLLER_ROOT/completed_output_root"
LOCATOR=$T12_PAPER_CONTROLLER_ROOT/cell_root_locator.json
"$PY" -c 'import json,os,sys,tempfile; target=sys.argv[1]; root=sys.argv[2]; parent=os.path.dirname(target); payload={"schema_version":"fast16_matrix_cell_root_locator_v1","status":"READY","dataset":"TasteMolNet","method":"GCFExplainer","terminal_root":root}; fd,tmp=tempfile.mkstemp(prefix=".cell-root-locator.",suffix=".tmp",dir=parent); f=os.fdopen(fd,"w",encoding="utf-8"); json.dump(payload,f,sort_keys=True,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(tmp,target); d=os.open(parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)); os.fsync(d); os.close(d)' "$LOCATOR" "$PAPER_ROOT"
write_heartbeat PAPER_CELL_PASS 0
