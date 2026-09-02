#!/usr/bin/env bash
# Launch one persistent, read-only T14 convergence relay without a GPU lease.

set -euo pipefail
umask 077

: "${T14_AUDITOR_REPO_ROOT:?set one immutable deployed repository root}"
: "${T14_CHECKPOINT_ROOT:?set the active T14 checkpoints directory}"
: "${T14_AUDITOR_EXECUTION_COMMIT:?set the exact immutable auditor commit}"

RUNTIME=${AUTODL_RUNTIME_ROOT:-/autodl-fs/data/counterfactual-subgraph-runtime}
CONTROL=${AUTODL_CONTROL_ROOT:-$RUNTIME/control}
PY=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
POLL_SECONDS=${T14_CONVERGENCE_RELAY_POLL_SECONDS:-60}
HEARTBEAT_SECONDS=${T14_CONVERGENCE_RELAY_HEARTBEAT_SECONDS:-60}

[[ "$T14_AUDITOR_REPO_ROOT" == /* && -d "$T14_AUDITOR_REPO_ROOT" \
  && ! -L "$T14_AUDITOR_REPO_ROOT" ]] || {
  echo "T14_AUDITOR_REPO_ROOT must be one absolute physical directory" >&2
  exit 64
}
[[ "$T14_CHECKPOINT_ROOT" == /* && -d "$T14_CHECKPOINT_ROOT" \
  && ! -L "$T14_CHECKPOINT_ROOT" ]] || {
  echo "T14_CHECKPOINT_ROOT must be one absolute physical directory" >&2
  exit 64
}
[[ "$T14_AUDITOR_EXECUTION_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "T14_AUDITOR_EXECUTION_COMMIT must be one exact lowercase Git SHA" >&2
  exit 64
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ && "$HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "relay poll/heartbeat seconds must be positive integers" >&2
  exit 64
}
[[ "${RUN_GNN_ABLATION:-0}" == "0" ]] || {
  echo "T14 main-table relay refuses a concurrent GNN ablation request" >&2
  exit 64
}

observed_commit=$(git -C "$T14_AUDITOR_REPO_ROOT" rev-parse HEAD)
[[ "$observed_commit" == "$T14_AUDITOR_EXECUTION_COMMIT" ]] || {
  echo "deployed repository commit does not match T14 auditor commit" >&2
  exit 65
}
[[ -z "$(git -C "$T14_AUDITOR_REPO_ROOT" status --porcelain)" ]] || {
  echo "deployed T14 auditor repository must be clean" >&2
  exit 65
}

controller_id=$(
  "$PY" -I -B -c \
    'import datetime,uuid; print("t14-external-convergence-relay-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+str(uuid.uuid4()))'
)
controller_root=$CONTROL/$controller_id
mkdir -p "$controller_root"

nohup nice -n 10 ionice -c 2 -n 7 \
  "$PY" "$T14_AUDITOR_REPO_ROOT/scripts/autodl/run_t14_external_convergence_auditor_relay_v1.py" \
  --config configs/hpc.yaml \
  --set inference.fallback_to_heuristic=false \
  --checkpoint-root "$T14_CHECKPOINT_ROOT" \
  --relay-root "$controller_root" \
  --execution-commit "$T14_AUDITOR_EXECUTION_COMMIT" \
  --one-shot-script "$T14_AUDITOR_REPO_ROOT/scripts/autodl/run_t14_external_convergence_auditor_v1.py" \
  --poll-seconds "$POLL_SECONDS" \
  --heartbeat-seconds "$HEARTBEAT_SECONDS" \
  >"$controller_root/controller.log" 2>&1 </dev/null &
controller_pid=$!
printf '%s\n' "$controller_pid" >"$controller_root/launcher.pid"

for _ in $(seq 1 20); do
  [[ -s "$controller_root/heartbeat.json" ]] && break
  if ! kill -0 "$controller_pid" 2>/dev/null; then
    echo "T14 convergence relay exited before its first heartbeat" >&2
    tail -n 80 "$controller_root/controller.log" >&2 || true
    exit 70
  fi
  sleep 0.5
done
[[ -s "$controller_root/heartbeat.json" ]] || {
  echo "T14 convergence relay did not publish its first heartbeat" >&2
  exit 70
}

printf 'controller_id=%s\n' "$controller_id"
printf 'controller_pid=%s\n' "$controller_pid"
printf 'controller_root=%s\n' "$controller_root"
printf 'heartbeat=%s\n' "$controller_root/heartbeat.json"
printf 'convergence_receipt=%s\n' "$controller_root/t14_convergence_relay_receipt.json"
printf 'status_command=cat %q; tail -n 80 %q\n' \
  "$controller_root/heartbeat.json" "$controller_root/controller.log"
