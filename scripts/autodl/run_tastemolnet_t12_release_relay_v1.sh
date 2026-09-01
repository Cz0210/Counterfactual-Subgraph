#!/usr/bin/env bash
# Narrow T12 successor: verified release dependencies -> fresh GPU3 science.

set -euo pipefail

: "${T12_REPO_ROOT:?set the immutable deployed repository root}"
: "${T12_CONTROLLER_ROOT:?set one fresh controller root}"
: "${TASTE_T3_ROOT:?set the calibrated T3 terminal root}"
: "${TASTE_T7_PASS_ROOT:?set the managed T7 smoke PASS root}"
: "${TASTE_MANAGED_NEUROSED_ROOT:?set the managed NeuroSED PASS root}"
: "${T12_MANAGED_RELEASE_ROOT:?set the typed managed release PASS root}"
: "${T12_RELEASE_VALIDATOR_ROOT:?set the pinned typed-release implementation root}"
: "${TASTE_T7_NEUROSED_THRESHOLD_AUTHORITY:?set the typed NeuroSED threshold authority}"
: "${T12_EXACT_REPLAY_GATE:?set the exact gate-v2 JSON}"

PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
GPU_INDEX=${T12_GPU_INDEX:-3}
MIN_FREE_GB=${T12_MIN_FREE_GB:-100}
POLL_SECONDS=${T12_POLL_SECONDS:-60}
OFFICIAL_ROOT=${TASTE_OFFICIAL_GCF_ROOT:-$T12_REPO_ROOT/baselines/gcfexplainer_official}
OUTPUT_BASE=${T12_OUTPUT_BASE:-$RUNTIME/outputs/autodl/tastemolnet/gcfexplainer/t12-production}
LOCK_FILE=$CONTROL/tastemolnet-t12-gcf-release-relay.lock

[[ "$GPU_INDEX" == "3" ]] || {
  echo "T12 release successor is pinned to physical GPU3" >&2
  exit 64
}
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || {
  echo "T12 refuses to run while GNN ablation is enabled" >&2
  exit 64
}
[[ ! -e "$T12_CONTROLLER_ROOT/controller.pid" ]] || {
  echo "T12 controller root is not fresh" >&2
  exit 73
}

mkdir -p "$T12_CONTROLLER_ROOT" "$CONTROL" "$OUTPUT_BASE"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another T12 release successor holds $LOCK_FILE" >&2
  exit 73
fi

cd "$T12_REPO_ROOT"
export PYTHONPATH=$PWD
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=7
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

HEARTBEAT=$T12_CONTROLLER_ROOT/heartbeat.json
STATE=$T12_CONTROLLER_ROOT/state

write_heartbeat() {
  local phase=$1
  local science_pid=${2:-0}
  local temporary=$HEARTBEAT.tmp.$$
  "$PY" -c 'import json,os,sys,time; p=sys.argv[1]; v={"controller_pid":os.getppid(),"phase":sys.argv[2],"science_pid":int(sys.argv[3]),"written_at_unix":int(time.time())}; f=open(p,"x",encoding="utf-8"); json.dump(v,f,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno()); f.close()' "$temporary" "$phase" "$science_pid"
  mv "$temporary" "$HEARTBEAT"
  printf '%s\n' "$phase" > "$STATE.tmp.$$"
  mv "$STATE.tmp.$$" "$STATE"
}

validate_terminal_pass() {
  local root=$1
  local label=$2
  "$PY" - "$root" "$label" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
label = sys.argv[2]
if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
    raise SystemExit(f"{label} root is not one normalized directory: {root}")
marker = root / "PASS"
if not marker.is_file() or not marker.read_bytes().strip():
    raise SystemExit(f"{label} has no terminal PASS marker")
for name in ("gate.json", "verification.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"{label} lacks {name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise SystemExit(f"{label} {name} is not PASS")
PY
}

run_stage() {
  local phase=$1
  local log=$2
  shift 2
  "$@" > "$log" 2>&1 &
  local science_pid=$!
  write_heartbeat "$phase" "$science_pid"
  if [[ "$phase" == "T12_FRESH_10K" ]]; then
    printf '%s\n' '[TASTE_T12_GCF_FULL_LAUNCHED]' | tee "$T12_CONTROLLER_ROOT/TASTE_T12_GCF_FULL_LAUNCHED"
  fi
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
write_heartbeat VALIDATING_RELEASE_DEPENDENCIES 0
validate_terminal_pass "$TASTE_T3_ROOT" T3
validate_terminal_pass "$TASTE_T7_PASS_ROOT" T7
validate_terminal_pass "$TASTE_MANAGED_NEUROSED_ROOT" NeuroSED
PYTHONPATH="$T12_RELEASE_VALIDATOR_ROOT" \
"$PY" "$T12_RELEASE_VALIDATOR_ROOT/scripts/autodl/tastemolnet_t7_typed_release_v1.py" \
  --config "$T12_RELEASE_VALIDATOR_ROOT/configs/hpc.yaml" validate \
  --release-root "$T12_MANAGED_RELEASE_ROOT" \
  > "$T12_CONTROLLER_ROOT/managed-release-validation.log"

"$PY" - "$T12_CONTROLLER_ROOT/dependency_receipt.json" \
  "$TASTE_T3_ROOT" "$TASTE_T7_PASS_ROOT" "$TASTE_MANAGED_NEUROSED_ROOT" \
  "$T12_MANAGED_RELEASE_ROOT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

destination = Path(sys.argv[1])
names = ("t3", "t7", "neurosed", "managed_release")
roots = [Path(value) for value in sys.argv[2:]]
payload = {
    "schema_version": "tastemolnet_t12_release_dependency_receipt_v1",
    "status": "PASS",
    "dependencies": {
        name: {
            "root": str(root),
            "pass_sha256": hashlib.sha256((root / "PASS").read_bytes()).hexdigest(),
        }
        for name, root in zip(names, roots, strict=True)
    },
    "depends_on_taste_full_task": False,
    "gpu_index": 3,
}
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
PY

printf '%s\n' '[TASTE_T12_DEPENDENCY_DECOUPLED]' | tee "$T12_CONTROLLER_ROOT/TASTE_T12_DEPENDENCY_DECOUPLED"
write_heartbeat WAITING_FOR_GPU3 0
while true; do
  gpu_processes=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)
  [[ -z "$gpu_processes" ]] && break
  sleep "$POLL_SECONDS"
  write_heartbeat WAITING_FOR_GPU3 0
done

free_kb=$(df -Pk "$RUNTIME" | awk 'NR==2 {print $4}')
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if [[ -z "$free_kb" || "$free_kb" -lt "$required_kb" ]]; then
  echo "T12 needs at least ${MIN_FREE_GB} GiB free; observed_kb=${free_kb:-unknown}" >&2
  write_heartbeat INSUFFICIENT_STORAGE 0
  exit 75
fi

ATTEMPT_ID=$($PY -c 'import uuid; print(uuid.uuid4())')
GENERATION_TOKEN=$($PY -c 'import secrets; print(secrets.token_hex(32))')
GPU_UUID=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
OUTPUT_ROOT=$OUTPUT_BASE/attempt-$ATTEMPT_ID
[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
  echo "fresh T12 output root unexpectedly exists: $OUTPUT_ROOT" >&2
  exit 76
}

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
[[ -s "$CHECKPOINT_10K" ]] || {
  write_heartbeat T12_FRESH_10K_INCOMPLETE 0
  exit 77
}
run_stage T12_RESUME_20K "$T12_CONTROLLER_ROOT/resume-20k.log" \
  "$PY" scripts/run_tastemolnet_gcf_full.py --mode resume \
  --checkpoint-manifest "$CHECKPOINT_10K" "${COMMON_ARGS[@]}"

CHECKPOINT_20K=$OUTPUT_ROOT/checkpoints/checkpoint-00020000.manifest.json
[[ -s "$CHECKPOINT_20K" ]] || {
  write_heartbeat T12_RESUME_20K_INCOMPLETE 0
  exit 78
}
run_stage T12_GENERATION_VERIFY "$T12_CONTROLLER_ROOT/generation-verify.log" \
  "$PY" scripts/verify_tastemolnet_gcf_full_generation.py \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --production-root "$OUTPUT_ROOT" \
  --output-root "$OUTPUT_ROOT/generation_verification"

[[ -s "$OUTPUT_ROOT/generation_verification/GENERATION_PASS" ]] || {
  write_heartbeat T12_GENERATION_VERIFY_INCOMPLETE 0
  exit 79
}
printf '%s\n' "$OUTPUT_ROOT" > "$T12_CONTROLLER_ROOT/completed_output_root"
write_heartbeat GENERATION_PASS 0
