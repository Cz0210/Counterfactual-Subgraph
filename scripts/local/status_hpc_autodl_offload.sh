#!/usr/bin/env bash
# Read-only, credential-free status summary for the Mac -> HPC -> AutoDL T8
# CPU-offload route.  This script deliberately does not print SSH diagnostics,
# process command lines, environment dumps, model/data payloads, or SSH config.

set -uo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd -P)
DEFAULT_REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." 2>/dev/null && pwd -P)

LOCAL_REPO_ROOT=${LOCAL_REPO_ROOT:-$DEFAULT_REPO_ROOT}
LOCAL_TRANSFER_ROOT=${LOCAL_TRANSFER_ROOT:-/Volumes/DireRaven/counterfactual-hpc-offload}
HPC_ALIAS=${HPC_ALIAS:-tongji-hpc}
AUTODL_ALIAS=${AUTODL_ALIAS:-autodl-a800}
HPC_CONTROL_SOCKET=${HPC_CONTROL_SOCKET:-/tmp/tongji-codex.sock}
HPC_RUNTIME_ROOT=${HPC_RUNTIME_ROOT:-/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime}
HPC_EXECUTION_WORKTREE=${HPC_EXECUTION_WORKTREE:-/share/home/u20526/czx/worktrees/t8-hpc-481475c3}
HPC_T8_CURRENT_POINTER=${HPC_T8_CURRENT_POINTER:-$HPC_RUNTIME_ROOT/control/t8-hpc-current-chain/current.json}
HPC_PYTHON=${HPC_PYTHON:-/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python}
AUTODL_MATRIX_AUTHORITY=${AUTODL_MATRIX_AUTHORITY:-/autodl-fs/data/counterfactual-subgraph-runtime/control/fast16_matrix_authority}
AUTODL_PYTHON=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
AUTODL_T8_STATUS_PATH=${AUTODL_T8_STATUS_PATH:-}
AUTODL_T12_STATUS_PATH=${AUTODL_T12_STATUS_PATH:-}
AUTODL_T14_STATUS_PATH=${AUTODL_T14_STATUS_PATH:-}
AUTODL_MUT_STATUS_PATH=${AUTODL_MUT_STATUS_PATH:-}
STATUS_CONNECT_TIMEOUT_SECONDS=${STATUS_CONNECT_TIMEOUT_SECONDS:-8}
STATUS_SSH_BIN=${STATUS_SSH_BIN:-ssh}

safe_alias() {
  case "$1" in
    ''|*[!A-Za-z0-9_.-]*) return 1 ;;
    *) return 0 ;;
  esac
}

safe_path_or_empty() {
  case "$1" in
    '') return 0 ;;
    /*[!A-Za-z0-9_./:+-]*|*[!A-Za-z0-9_./:+-]*) return 1 ;;
    /*) return 0 ;;
    *) return 1 ;;
  esac
}

single_line() {
  # Status values are restricted to printable, non-control output.  This is
  # defense in depth for unexpected Git/hostname filesystem metadata.
  LC_ALL=C tr '\r\n\t' '   ' | LC_ALL=C tr -cd '[:print:]' | cut -c 1-512
}

print_kv() {
  printf '%s=%s\n' "$1" "$(printf '%s' "$2" | single_line)"
}

if ! safe_alias "$HPC_ALIAS" || ! safe_alias "$AUTODL_ALIAS"; then
  printf '%s\n' 'status_error=INVALID_SSH_ALIAS'
  exit 2
fi

for status_path in \
  "$LOCAL_REPO_ROOT" \
  "$LOCAL_TRANSFER_ROOT" \
  "$HPC_CONTROL_SOCKET" \
  "$HPC_EXECUTION_WORKTREE" \
  "$HPC_RUNTIME_ROOT" \
  "$HPC_T8_CURRENT_POINTER" \
  "$HPC_PYTHON" \
  "$AUTODL_MATRIX_AUTHORITY" \
  "$AUTODL_PYTHON" \
  "$AUTODL_T8_STATUS_PATH" \
  "$AUTODL_T12_STATUS_PATH" \
  "$AUTODL_T14_STATUS_PATH" \
  "$AUTODL_MUT_STATUS_PATH"
do
  if ! safe_path_or_empty "$status_path"; then
    printf '%s\n' 'status_error=INVALID_STATUS_PATH'
    exit 2
  fi
done

case "$STATUS_CONNECT_TIMEOUT_SECONDS" in
  ''|*[!0-9]*) printf '%s\n' 'status_error=INVALID_CONNECT_TIMEOUT'; exit 2 ;;
esac

print_kv status_schema_version 1
print_kv checked_at_utc "$(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || printf UNKNOWN)"
print_kv status_mode READ_ONLY_REDACTED
print_kv local_repo_root "$LOCAL_REPO_ROOT"

if [ -d "$LOCAL_REPO_ROOT/.git" ] || GIT_OPTIONAL_LOCKS=0 git -C "$LOCAL_REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  local_commit=$(GIT_OPTIONAL_LOCKS=0 git -C "$LOCAL_REPO_ROOT" rev-parse HEAD 2>/dev/null || printf UNKNOWN)
  local_dirty_count=$(GIT_OPTIONAL_LOCKS=0 git -C "$LOCAL_REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l | tr -d ' ')
  print_kv local_repo_commit "$local_commit"
  print_kv local_repo_tracked_dirty_count "${local_dirty_count:-UNKNOWN}"
else
  print_kv local_repo_commit NOT_A_GIT_WORKTREE
  print_kv local_repo_tracked_dirty_count UNKNOWN
fi

if [ -d "$LOCAL_TRANSFER_ROOT" ]; then
  local_transfer_free_kib=$(df -k "$LOCAL_TRANSFER_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
  print_kv local_transfer_root "$LOCAL_TRANSFER_ROOT"
  print_kv local_transfer_state AVAILABLE
  print_kv local_transfer_free_kib "${local_transfer_free_kib:-UNKNOWN}"
else
  print_kv local_transfer_root "$LOCAL_TRANSFER_ROOT"
  print_kv local_transfer_state NOT_MOUNTED
  print_kv local_transfer_free_kib UNKNOWN
fi

if [ -S "$HPC_CONTROL_SOCKET" ] && "$STATUS_SSH_BIN" -q -S "$HPC_CONTROL_SOCKET" -O check "$HPC_ALIAS" >/dev/null 2>&1; then
  print_kv hpc_control_master_state ALIVE
elif "$STATUS_SSH_BIN" -q -O check "$HPC_ALIAS" >/dev/null 2>&1; then
  # Let the user's SSH config resolve its own ControlPath when it is not the
  # conventional /tmp socket. This remains a read-only control operation.
  print_kv hpc_control_master_state ALIVE_CONFIGURED_PATH
else
  print_kv hpc_control_master_state UNAVAILABLE
fi

hpc_status=$(
  "$STATUS_SSH_BIN" -q \
    -o BatchMode=yes \
    -o "ConnectTimeout=$STATUS_CONNECT_TIMEOUT_SECONDS" \
    "$HPC_ALIAS" \
    "bash -s -- '$HPC_EXECUTION_WORKTREE' '$HPC_RUNTIME_ROOT' '$HPC_PYTHON' '$HPC_T8_CURRENT_POINTER'" \
    2>/dev/null <<'HPC_STATUS_REMOTE'
set -u
worktree=$1
runtime_root=$2
python_bin=$3
current_pointer=$4
printf 'hpc_ssh_state=PASS\n'
printf 'hpc_hostname=%s\n' "$(hostname -s 2>/dev/null || printf UNKNOWN)"
if [ -d "$worktree" ] && git -C "$worktree" rev-parse HEAD >/dev/null 2>&1; then
  printf 'hpc_execution_worktree_state=AVAILABLE\n'
  printf 'hpc_execution_commit=%s\n' "$(git -C "$worktree" rev-parse HEAD 2>/dev/null || printf UNKNOWN)"
else
  printf 'hpc_execution_worktree_state=NOT_AVAILABLE\n'
  printf 'hpc_execution_commit=UNKNOWN\n'
fi
if [ -x "$python_bin" ]; then
  printf 'hpc_python_state=AVAILABLE\n'
  printf 'hpc_python_version=%s\n' "$($python_bin -V 2>&1 | tr ' ' '_')"
else
  printf 'hpc_python_state=NOT_AVAILABLE\n'
  printf 'hpc_python_version=UNKNOWN\n'
fi
probe_root=$runtime_root
if [ ! -e "$probe_root" ]; then
  probe_root=$(dirname "$probe_root")
fi
printf 'hpc_runtime_free_kib=%s\n' "$(df -k "$probe_root" 2>/dev/null | awk 'NR==2 {print $4}' || printf UNKNOWN)"
if command -v squeue >/dev/null 2>&1; then
  jobs=$(squeue -h -u "$(id -un)" -o '%i,%j,%T,%M,%R' 2>/dev/null \
    | awk 'BEGIN {IGNORECASE=1} $0 ~ /t8|globalgce/ {print}' \
    | head -n 20 \
    | paste -sd ';' -)
  printf 'hpc_slurm_state=AVAILABLE\n'
  printf 'hpc_t8_jobs=%s\n' "${jobs:-NONE}"
else
  printf 'hpc_slurm_state=NOT_AVAILABLE\n'
  printf 'hpc_t8_jobs=UNKNOWN\n'
fi
if [ ! -x "$python_bin" ]; then
  printf 'hpc_t8_current_pointer_state=PYTHON_UNAVAILABLE\n'
  printf 'hpc_t8_chain_state=UNKNOWN\n'
  printf 'hpc_t8_chain_jobs=UNKNOWN\n'
else
  pointer_status=$(
    "$python_bin" - "$current_pointer" <<'PY_POINTER'
import hashlib
import json
import re
import sys
from pathlib import Path


def emit(key, value):
    text = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    print(f"{key}={''.join(ch for ch in text if ch.isprintable())[:256]}")


path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    emit("hpc_t8_current_pointer_state", "MISSING")
    emit("hpc_t8_chain_state", "UNKNOWN")
    emit("hpc_t8_chain_jobs", "NONE")
    raise SystemExit(0)
try:
    raw = path.read_bytes()
    payload = json.loads(raw)
    claimed = payload.pop("current_sha256")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    detached = path.with_suffix(path.suffix + ".sha256").read_text(
        encoding="ascii"
    ).strip()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(claimed))
        or hashlib.sha256(canonical).hexdigest() != claimed
        or detached != hashlib.sha256(raw).hexdigest()
        or payload.get("schema_version") != "t8_hpc_current_chain_v1"
    ):
        raise ValueError("pointer hash/schema mismatch")
    jobs = []
    for key in (
        "canary_job_id",
        "followup_job_id",
        "array_job_id",
        "merge_job_id",
        "package_job_id",
    ):
        value = payload.get(key)
        if value is not None:
            if not re.fullmatch(r"[0-9]+", str(value)):
                raise ValueError("pointer job id is invalid")
            jobs.append(str(value))
    dependencies = [
        str(payload[key])
        for key in (
            "followup_dependency",
            "merge_dependency",
            "package_dependency",
        )
        if payload.get(key)
    ]
    emit("hpc_t8_current_pointer_state", "PASS")
    emit("hpc_t8_chain_state", payload.get("state", "UNKNOWN"))
    emit("hpc_t8_chain_stage", payload.get("active_stage", "UNKNOWN"))
    emit("hpc_t8_chain_refinement_depth", payload.get("refinement_depth", "NONE"))
    emit("hpc_t8_chain_job_ids", ",".join(jobs) if jobs else "NONE")
    emit("hpc_t8_chain_dependency", ",".join(dependencies) if dependencies else "NONE")
    emit("hpc_t8_chain_jobs", ",".join(jobs) if jobs else "NONE")
except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
    emit("hpc_t8_current_pointer_state", "INVALID")
    emit("hpc_t8_chain_state", "UNKNOWN")
    emit("hpc_t8_chain_jobs", "NONE")
PY_POINTER
  )
  printf '%s\n' "$pointer_status"
  chain_job_ids=$(printf '%s\n' "$pointer_status" | awk -F= '$1 == "hpc_t8_chain_job_ids" {print $2}')
  valid_chain_ids=$(printf '%s\n' "$chain_job_ids" \
    | awk '/^[0-9]+(,[0-9]+)*$/ {print "yes"}')
  if [ "$valid_chain_ids" != yes ]; then
    chain_rows=NONE
  else
    chain_rows=$(squeue -h -j "$chain_job_ids" -o '%i,%T,%M,%R,%E' 2>/dev/null \
      | head -n 20 \
      | paste -sd ';' -)
    chain_rows=${chain_rows:-NO_ACTIVE_ROWS}
  fi
  printf 'hpc_t8_chain_slurm=%s\n' "$chain_rows"
fi
HPC_STATUS_REMOTE
)
hpc_rc=$?
if [ "$hpc_rc" -eq 0 ]; then
  printf '%s\n' "$hpc_status" | while IFS='=' read -r key value; do
    case "$key" in
      hpc_ssh_state|hpc_hostname|hpc_execution_worktree_state|hpc_execution_commit|hpc_python_state|hpc_python_version|hpc_runtime_free_kib|hpc_slurm_state|hpc_t8_jobs|hpc_t8_current_pointer_state|hpc_t8_chain_state|hpc_t8_chain_stage|hpc_t8_chain_refinement_depth|hpc_t8_chain_job_ids|hpc_t8_chain_dependency|hpc_t8_chain_jobs|hpc_t8_chain_slurm)
        print_kv "$key" "$value"
        ;;
    esac
  done
else
  print_kv hpc_ssh_state UNREACHABLE
  print_kv hpc_execution_worktree_state UNKNOWN
  print_kv hpc_execution_commit UNKNOWN
  print_kv hpc_slurm_state UNKNOWN
  print_kv hpc_t8_jobs UNKNOWN
  print_kv hpc_t8_current_pointer_state UNKNOWN
  print_kv hpc_t8_chain_state UNKNOWN
  print_kv hpc_t8_chain_jobs UNKNOWN
  print_kv hpc_t8_chain_slurm UNKNOWN
fi

autodl_status=$(
  "$STATUS_SSH_BIN" -q \
    -o BatchMode=yes \
    -o "ConnectTimeout=$STATUS_CONNECT_TIMEOUT_SECONDS" \
    "$AUTODL_ALIAS" \
    "bash -s -- '$AUTODL_MATRIX_AUTHORITY' '$AUTODL_PYTHON' '$AUTODL_T8_STATUS_PATH' '$AUTODL_T12_STATUS_PATH' '$AUTODL_T14_STATUS_PATH' '$AUTODL_MUT_STATUS_PATH'" \
    2>/dev/null <<'AUTODL_STATUS_REMOTE'
set -u
matrix_root=$1
python_bin=$2
t8_status=$3
t12_status=$4
t14_status=$5
mut_status=$6
printf 'autodl_ssh_state=PASS\n'
printf 'autodl_hostname=%s\n' "$(hostname -s 2>/dev/null || printf UNKNOWN)"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_rows=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' | sed 's/;$//')
  printf 'autodl_gpu_summary=index_usedMiB_freeMiB_utilPct:%s\n' "${gpu_rows:-UNKNOWN}"
else
  printf 'autodl_gpu_summary=UNAVAILABLE\n'
fi
if [ ! -x "$python_bin" ]; then
  printf 'autodl_matrix_state=PYTHON_UNAVAILABLE\n'
  printf 'autodl_matrix_complete_cells=UNKNOWN\n'
  printf 'autodl_matrix_total_cells=16\n'
  exit 0
fi
"$python_bin" - "$matrix_root" "$t8_status" "$t12_status" "$t14_status" "$mut_status" <<'PY_STATUS'
import json
import os
import sys
from pathlib import Path


def clean(value):
    text = str(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return "".join(ch for ch in text if ch.isprintable())[:256]


def load_json(path):
    try:
        candidate = Path(path)
        if not candidate.is_file() or candidate.is_symlink():
            return None
        with candidate.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def recursive_numbers(value, keys):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, int) and not isinstance(child, bool):
                yield child
            yield from recursive_numbers(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_numbers(child, keys)


root = Path(sys.argv[1])
matrix_docs = []
if root.is_file():
    loaded = load_json(root)
    if loaded is not None:
        matrix_docs.append(loaded)
elif root.is_dir() and not root.is_symlink():
    preferred = (
        "state.json",
        "matrix_status.json",
        "current.json",
        "pointer.json",
        "manifest.json",
    )
    for name in preferred:
        loaded = load_json(root / name)
        if loaded is not None:
            matrix_docs.append(loaded)
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)[-8:]
    except OSError:
        children = []
    for child in children:
        if child.is_dir() and not child.is_symlink():
            for name in preferred[:2]:
                loaded = load_json(child / name)
                if loaded is not None:
                    matrix_docs.append(loaded)

complete_keys = {
    "complete_cells", "completed_cells", "matrix_complete_cells",
    "complete_cell_count", "completed_count", "pass_count", "latest_count",
}
total_keys = {"total_cells", "matrix_total_cells", "cell_count"}
complete_values = [n for doc in matrix_docs for n in recursive_numbers(doc, complete_keys)]
for doc in matrix_docs:
    applied = doc.get("applied_cells")
    if isinstance(applied, list):
        complete_values.append(len(applied))
total_values = [n for doc in matrix_docs for n in recursive_numbers(doc, total_keys)]
complete = max((n for n in complete_values if 0 <= n <= 16), default=None)
total = max((n for n in total_values if 1 <= n <= 64), default=16)
print("autodl_matrix_state=" + ("READABLE" if matrix_docs else "UNREADABLE"))
print("autodl_matrix_complete_cells=" + (str(complete) if complete is not None else "UNKNOWN"))
print("autodl_matrix_total_cells=" + str(total))


def status_line(label, raw_path):
    if not raw_path:
        print(f"autodl_{label}_state=STATUS_PATH_NOT_CONFIGURED")
        return
    doc = load_json(raw_path)
    if doc is None:
        print(f"autodl_{label}_state=STATUS_UNREADABLE")
        return
    state = "UNKNOWN"
    for key in ("state", "status", "phase", "terminal_state"):
        value = doc.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            state = clean(value)
            break
    print(f"autodl_{label}_state={state}")


for label, raw_path in zip(("t8", "t12", "t14", "mut"), sys.argv[2:]):
    status_line(label, raw_path)
PY_STATUS
AUTODL_STATUS_REMOTE
)
autodl_rc=$?
if [ "$autodl_rc" -eq 0 ]; then
  printf '%s\n' "$autodl_status" | while IFS='=' read -r key value; do
    case "$key" in
      autodl_ssh_state|autodl_hostname|autodl_gpu_summary|autodl_matrix_state|autodl_matrix_complete_cells|autodl_matrix_total_cells|autodl_t8_state|autodl_t12_state|autodl_t14_state|autodl_mut_state)
        print_kv "$key" "$value"
        ;;
    esac
  done
else
  print_kv autodl_ssh_state UNREACHABLE
  print_kv autodl_matrix_state UNKNOWN
  print_kv autodl_matrix_complete_cells UNKNOWN
  print_kv autodl_matrix_total_cells 16
  print_kv autodl_t8_state UNKNOWN
  print_kv autodl_t12_state UNKNOWN
  print_kv autodl_t14_state UNKNOWN
  print_kv autodl_mut_state UNKNOWN
fi

print_kv status_side_effects NONE
