#!/usr/bin/env bash
# One-shot T11 relay: wait for one physical GPU, then resume the durable science root.

set -euo pipefail

: "${T11_REPO_ROOT:?set the immutable T11 repair worktree}"
: "${T11_RELAY_ROOT:?set one fresh persistent relay root}"
: "${T11_SCIENCE_ROOT:?set the existing resumable T11 science root}"
: "${T11_FINAL_ROOT:?set the fresh T11 verifier/publisher root}"
: "${T11_PPO_OUTPUT_ROOT:?set the existing exact T11 PPO root}"
: "${T6_OUTPUT_ROOT:?set the independently verified T6 root}"
: "${TASTEMOLNET_BASE_MODEL:?set the exact T6 base model}"
: "${TASTEMOLNET_GNN_CHECKPOINT:?set the frozen three-class GINE checkpoint}"
: "${TASTEMOLNET_TRAIN_CSV:?set the frozen train CSV}"
: "${TASTEMOLNET_CALIBRATION_CSV:?set the frozen calibration CSV}"
: "${TASTEMOLNET_TEST_CSV:?set the held-out test CSV}"
: "${MOLCLR_ROOT:?set the pinned MolCLR source root}"
: "${MOLCLR_CHECKPOINT:?set the pinned MolCLR checkpoint}"
: "${TASTEMOLNET_THRESHOLD_CONTRACT:?set the adopted WNode threshold contract}"
: "${WNODE_CACHE_DB:?set the persistent T11 WNode cache}"
: "${NODE_EMBEDDING_CACHE_DIR:?set the persistent MolCLR node cache}"
: "${T11_GPU_UUID:?set the immutable physical GPU UUID}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T11_GPU_INDEX:-0}
POLL_SECONDS=${T11_RELAY_POLL_SECONDS:-30}
MIN_FREE_MEMORY_MB=${T11_MIN_FREE_MEMORY_MB:-16000}
LOCK_FILE=$CONTROL/tastemolnet-t11-checkpoint-resume-relay.lock
HEARTBEAT=$T11_RELAY_ROOT/heartbeat.json
STATE=$T11_RELAY_ROOT/state
LOG=$T11_RELAY_ROOT/science.log

mkdir -p "$T11_RELAY_ROOT" "$CONTROL"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another T11 checkpoint-resume relay holds $LOCK_FILE" >&2
  exit 73
fi

cd "$T11_REPO_ROOT"
export PYTHONPATH=$PWD
export RUN_GNN_ABLATION=0
export RUN_TASTEMOLNET=1
export TOKENIZERS_PARALLELISM=false

write_heartbeat() {
  local phase=$1
  local manager_pid=${2:-0}
  "$PY" -c '
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(path.name + f".tmp.{os.getppid()}")
payload = {
    "schema_version": "tastemolnet_t11_checkpoint_resume_relay_v1",
    "controller_pid": os.getppid(),
    "phase": sys.argv[2],
    "gpu_index": int(sys.argv[3]),
    "gpu_uuid": sys.argv[4],
    "manager_pid": int(sys.argv[5]),
    "science_root": sys.argv[6],
    "final_root": sys.argv[7],
    "written_at": datetime.now(timezone.utc).isoformat(),
}
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, path)
' "$HEARTBEAT" "$phase" "$GPU_INDEX" "$T11_GPU_UUID" "$manager_pid" \
    "$T11_SCIENCE_ROOT" "$T11_FINAL_ROOT"
  printf '%s\n' "$phase" > "$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

required_final_files=(
  PASS
  summary.json
  run_manifest.json
  final_artifact_audit.json
  freeze_manifest.json
)

if [[ ! -s "$T11_SCIENCE_ROOT/checkpoint.json" ]]; then
  write_heartbeat BLOCKED_MISSING_CHECKPOINT 0
  echo "T11 durable checkpoint is missing" >&2
  exit 74
fi

if [[ -e "$T11_FINAL_ROOT" ]]; then
  for name in "${required_final_files[@]}"; do
    if [[ ! -s "$T11_FINAL_ROOT/$name" ]]; then
      write_heartbeat BLOCKED_PARTIAL_FINAL_ROOT 0
      echo "T11 final root exists without complete terminal files: $name" >&2
      exit 75
    fi
  done
  write_heartbeat ALREADY_PASS 0
  exit 0
fi

printf '%s\n' "$$" > "$T11_RELAY_ROOT/controller.pid"
while true; do
  write_heartbeat WAITING_FOR_GPU 0
  while true; do
    gpu_pids=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
    free_memory_mb=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
    if [[ -z "$gpu_pids" && "$free_memory_mb" =~ ^[0-9]+$ ]] \
      && (( free_memory_mb >= MIN_FREE_MEMORY_MB )) \
      && "$PY" scripts/autodl/gpu_lock.py \
        --project-root "$T11_REPO_ROOT" \
        --data-root /autodl-fs/data \
        --config configs/hpc.yaml \
        probe --gpu-index "$GPU_INDEX" --gpu-uuid "$T11_GPU_UUID" \
        >/dev/null 2>&1; then
      break
    fi
    sleep "$POLL_SECONDS"
    write_heartbeat WAITING_FOR_GPU 0
  done

  run_id=taste-t11-ours-full-wnode-resume-7fcf9ba-relay
  write_heartbeat ACQUIRING_GPU 0
  printf '\n[%s] attempting exact GPU lock and T11 resume\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  "$PY" scripts/autodl/gpu_lock.py \
    --project-root "$T11_REPO_ROOT" \
    --data-root /autodl-fs/data \
    --config configs/hpc.yaml \
    run --gpu-index "$GPU_INDEX" --gpu-uuid "$T11_GPU_UUID" --run-id "$run_id" \
    -- /bin/bash scripts/autodl/run_tastemolnet_ours_full.sh \
    >> "$LOG" 2>&1 &
  manager_pid=$!
  write_heartbeat RUNNING "$manager_pid"

  while kill -0 "$manager_pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
    write_heartbeat RUNNING "$manager_pid"
  done

  set +e
  wait "$manager_pid"
  exit_code=$?
  set -e
  if [[ "$exit_code" -eq 2 ]] \
    && tail -n 20 "$LOG" | grep -Fq "is project-locked"; then
    write_heartbeat GPU_LOCK_RACE_RETRY 0
    sleep "$POLL_SECONDS"
    continue
  fi
  break
done

if [[ "$exit_code" -ne 0 ]]; then
  write_heartbeat SCIENCE_FAILED 0
  echo "T11 resumed worker failed with exit code $exit_code" >&2
  exit "$exit_code"
fi

for name in "${required_final_files[@]}"; do
  if [[ ! -s "$T11_FINAL_ROOT/$name" ]]; then
    write_heartbeat TERMINAL_OUTPUT_INCOMPLETE 0
    echo "T11 worker exited successfully without terminal file: $name" >&2
    exit 76
  fi
done

write_heartbeat PASS 0
