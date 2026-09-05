#!/usr/bin/env bash
# One-shot, credential-free, content-addressed T8 relay:
# tongji-hpc -> Mac external disk -> fresh AutoDL import staging.
# This script never removes an HPC source and never writes the matrix authority.
set -euo pipefail

HPC_ALIAS=${HPC_ALIAS:-tongji-hpc}
AUTODL_ALIAS=${AUTODL_ALIAS:-autodl-a800}
HPC_HIERARCHICAL_ROOT=${HPC_HIERARCHICAL_ROOT:-/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/continuations/stress-2535373-e8be657a-2223-476b-94a3-fd14997e48ad/hierarchical-08a63955-20260904T181200Z}
MAC_RELAY_ROOT=${MAC_RELAY_ROOT:-/Volumes/DireRaven/counterfactual-hpc-offload/t8-scoped-relay-v2}
AUTODL_IMPORT_PARENT=${AUTODL_IMPORT_PARENT:-/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/imports/t8-hpc}
AUTODL_PYTHON=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RELAY_CONTROL_ROOT=${RELAY_CONTROL_ROOT:-$MAC_RELAY_ROOT/control}
RELAY_ATTEMPT_ID=${RELAY_ATTEMPT_ID:-}
RELAY_HEARTBEAT_SECONDS=${RELAY_HEARTBEAT_SECONDS:-15}
RELAY_TRANSFER_RETRIES=${RELAY_TRANSFER_RETRIES:-3}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-20}
RELAY_LOG_PATH=${RELAY_LOG_PATH:-}
SSH_BIN=${SSH_BIN:-ssh}
RSYNC_BIN=${RSYNC_BIN:-rsync}

EXPECTED_ARCHIVE_NAME=t8_exact_result_bundle.tar.gz
EXPECTED_ARCHIVE_BYTES=6103923589
EXPECTED_ARCHIVE_SHA256=06702fdc97ae2bb3661855497a336d19c6ceb33fd53f2304f41471781629346e

safe_alias() { case "$1" in ''|*[!A-Za-z0-9_.-]*) return 1;; *) return 0;; esac; }
safe_path() { case "$1" in ''|*[!A-Za-z0-9_./:+-]*) return 1;; /*) return 0;; *) return 1;; esac; }
safe_attempt() { case "$1" in ''|*[!A-Za-z0-9_.-]*) return 1;; *) return 0;; esac; }
is_uint() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }

safe_alias "$HPC_ALIAS" && safe_alias "$AUTODL_ALIAS" || { echo "invalid SSH alias" >&2; exit 64; }
for value in "$HPC_HIERARCHICAL_ROOT" "$MAC_RELAY_ROOT" "$AUTODL_IMPORT_PARENT" "$AUTODL_PYTHON" "$RELAY_CONTROL_ROOT"; do
  safe_path "$value" || { echo "invalid relay path: $value" >&2; exit 64; }
done
if [[ -n "$RELAY_LOG_PATH" ]]; then
  safe_path "$RELAY_LOG_PATH" || { echo "invalid relay log path" >&2; exit 64; }
fi
is_uint "$RELAY_HEARTBEAT_SECONDS" && (( RELAY_HEARTBEAT_SECONDS >= 5 )) || { echo "invalid heartbeat interval" >&2; exit 64; }
is_uint "$RELAY_TRANSFER_RETRIES" && (( RELAY_TRANSFER_RETRIES >= 1 && RELAY_TRANSFER_RETRIES <= 5 )) || { echo "invalid retry count" >&2; exit 64; }
is_uint "$SSH_CONNECT_TIMEOUT" || { echo "invalid SSH timeout" >&2; exit 64; }
case "$SSH_BIN:$RSYNC_BIN" in *[!A-Za-z0-9_./:+-]*) echo "invalid tool path" >&2; exit 64;; esac
for required_rsync_option in --append-verify --protect-args --info=progress2; do
  if ! "$RSYNC_BIN" "$required_rsync_option" --version >/dev/null 2>&1; then
    echo "rsync lacks required $required_rsync_option support: $RSYNC_BIN" >&2
    exit 69
  fi
done

if [[ -z "$RELAY_ATTEMPT_ID" ]]; then
  RELAY_ATTEMPT_ID="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
fi
safe_attempt "$RELAY_ATTEMPT_ID" || { echo "invalid relay attempt ID" >&2; exit 64; }

mkdir -p "$RELAY_CONTROL_ROOT" "$MAC_RELAY_ROOT"
chmod 700 "$RELAY_CONTROL_ROOT"
lock_dir="$RELAY_CONTROL_ROOT/lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  old_pid="$(cat "$RELAY_CONTROL_ROOT/pid" 2>/dev/null || true)"
  if is_uint "$old_pid" && kill -0 "$old_pid" 2>/dev/null; then
    echo "relay already running pid=$old_pid" >&2
    exit 73
  fi
  echo "stale relay lock requires inspection: $lock_dir" >&2
  exit 73
fi
printf '%s\n' "$$" > "$RELAY_CONTROL_ROOT/pid"
printf '%s\n' "$RELAY_ATTEMPT_ID" > "$RELAY_CONTROL_ROOT/attempt_id"

state=STARTING
detail="validating pinned HPC package"
hpc_package_root=
hpc_archive_path=
local_partial=
local_final=
autodl_partial=
autodl_final=
current_partial_path=
active_child=
error_written=false

partial_bytes() {
  if [[ -n "$current_partial_path" && -f "$current_partial_path" ]]; then
    wc -c < "$current_partial_path" | tr -d ' '
  else
    printf '0\n'
  fi
}

write_state() {
  state=$1
  detail=${2:-}
  python3 - "$RELAY_CONTROL_ROOT/state.json" "$state" "$detail" "$$" "$RELAY_ATTEMPT_ID" \
    "$HPC_HIERARCHICAL_ROOT" "$hpc_package_root" "$hpc_archive_path" \
    "$local_partial" "$local_final" "$autodl_partial" "$autodl_final" \
    "$EXPECTED_ARCHIVE_BYTES" "$EXPECTED_ARCHIVE_SHA256" "$(partial_bytes)" "$RELAY_LOG_PATH" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "t8_scoped_relay_state_v2",
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "pid": int(sys.argv[4]),
    "attempt_id": sys.argv[5],
    "hpc_hierarchical_root": sys.argv[6],
    "hpc_package_root": sys.argv[7] or None,
    "hpc_archive_path": sys.argv[8] or None,
    "mac_partial_root": sys.argv[9] or None,
    "mac_final_root": sys.argv[10] or None,
    "autodl_partial_root": sys.argv[11] or None,
    "autodl_final_root": sys.argv[12] or None,
    "expected_archive_bytes": int(sys.argv[13]),
    "expected_archive_sha256": sys.argv[14],
    "current_partial_bytes": int(sys.argv[15]),
    "log_path": sys.argv[16] or None,
    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
    "matrix_write_enabled": False,
    "hpc_source_delete_enabled": False,
}
data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd, tmp = tempfile.mkstemp(prefix=".state.", dir=path.parent)
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
finally:
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
PY
}

cleanup() {
  active_child=
  if [[ "$(cat "$RELAY_CONTROL_ROOT/pid" 2>/dev/null || true)" == "$$" ]]; then
    rm -f "$RELAY_CONTROL_ROOT/pid"
  fi
  rmdir "$lock_dir" 2>/dev/null || true
}

on_error() {
  rc=$?
  trap - ERR
  if [[ "$error_written" != true ]]; then
    error_written=true
    write_state FAILED "exit=$rc state=$state" || true
  fi
  exit "$rc"
}

on_signal() {
  trap - TERM INT
  if is_uint "${active_child:-}" && kill -0 "$active_child" 2>/dev/null; then
    kill -TERM "$active_child" 2>/dev/null || true
    wait "$active_child" 2>/dev/null || true
  fi
  error_written=true
  write_state INTERRUPTED "relay received signal" || true
  exit 143
}

trap cleanup EXIT
trap on_error ERR
trap on_signal TERM INT
write_state "$state" "$detail"

# Resolve package and archive paths on the HPC from the sealed ready/manifest pair.
# This is deliberately one-shot: absence or mismatch is a terminal relay failure.
metadata_json="$($SSH_BIN -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$HPC_ALIAS" \
  "python3 - '$HPC_HIERARCHICAL_ROOT' '$EXPECTED_ARCHIVE_NAME' '$EXPECTED_ARCHIVE_BYTES' '$EXPECTED_ARCHIVE_SHA256'" <<'PY_REMOTE'
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
expected_name, expected_bytes, expected_sha = sys.argv[2], int(sys.argv[3]), sys.argv[4]
package = (root / "artifacts" / "package").resolve(strict=True)
if package != root / "artifacts" / "package":
    raise SystemExit("non-canonical package path")
ready_path = package / "HIERARCHICAL_PACKAGE_READY.json"
manifest_path = package / "result_manifest.json"
evidence_manifest_path = package / "hierarchical_evidence_manifest.json"
ready = json.loads(ready_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
evidence = json.loads(evidence_manifest_path.read_text(encoding="utf-8"))
if ready.get("status") != "PASS" or manifest.get("status") != "PASS" or evidence.get("status") != "PASS":
    raise SystemExit("package receipts are not PASS")
if manifest.get("archive_name") != expected_name:
    raise SystemExit("unexpected archive name")
if int(manifest.get("archive_bytes", -1)) != expected_bytes:
    raise SystemExit("unexpected manifest archive size")
if manifest.get("archive_sha256") != expected_sha or ready.get("result_archive_sha256") != expected_sha:
    raise SystemExit("unexpected archive SHA")
archive = (package / manifest["archive_name"]).resolve(strict=True)
if archive.parent != package or archive.name != expected_name or archive.stat().st_size != expected_bytes:
    raise SystemExit("archive path or physical size mismatch")
evidence_name = "t8_hierarchical_evidence.tar.gz"
evidence_archive = (package / evidence_name).resolve(strict=True)
evidence_sha = str(evidence.get("archive_sha256", ""))
if evidence_archive.parent != package or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha):
    raise SystemExit("invalid evidence archive metadata")
if ready.get("evidence_archive_sha256") != evidence_sha:
    raise SystemExit("evidence SHA mismatch")
print(json.dumps({
    "archive_bytes": expected_bytes,
    "archive_path": str(archive),
    "archive_sha256": expected_sha,
    "evidence_archive_path": str(evidence_archive),
    "evidence_archive_sha256": evidence_sha,
    "evidence_manifest_path": str(evidence_manifest_path.resolve(strict=True)),
    "package_ready_path": str(ready_path.resolve(strict=True)),
    "package_root": str(package),
    "result_manifest_path": str(manifest_path.resolve(strict=True)),
}, sort_keys=True, separators=(",", ":")))
PY_REMOTE
)"

metadata_dir="$RELAY_CONTROL_ROOT/metadata"
mkdir -p "$metadata_dir"
metadata_path="$metadata_dir/hpc-package-$RELAY_ATTEMPT_ID.json"
python3 - "$metadata_path" "$metadata_json" <<'PY'
import json, os, sys, tempfile
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd, tmp = tempfile.mkstemp(prefix=".metadata.", dir=path.parent)
with os.fdopen(fd, "wb") as stream:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(tmp, path)
PY

json_field() {
  python3 - "$metadata_path" "$1" <<'PY'
import json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())[sys.argv[2]]
if not isinstance(value, (str, int)):
    raise SystemExit("invalid metadata field")
print(value)
PY
}

hpc_package_root="$(json_field package_root)"
hpc_archive_path="$(json_field archive_path)"
evidence_archive_path="$(json_field evidence_archive_path)"
result_manifest_path="$(json_field result_manifest_path)"
evidence_manifest_path="$(json_field evidence_manifest_path)"
package_ready_path="$(json_field package_ready_path)"
archive_sha="$(json_field archive_sha256)"
archive_bytes="$(json_field archive_bytes)"
evidence_sha="$(json_field evidence_archive_sha256)"
for value in "$hpc_package_root" "$hpc_archive_path" "$evidence_archive_path" "$result_manifest_path" "$evidence_manifest_path" "$package_ready_path"; do
  safe_path "$value" || { echo "invalid canonical HPC path" >&2; exit 65; }
done
[[ "$archive_sha" == "$EXPECTED_ARCHIVE_SHA256" && "$archive_bytes" == "$EXPECTED_ARCHIVE_BYTES" ]] || { echo "pinned archive mismatch" >&2; exit 65; }

local_partial="$MAC_RELAY_ROOT/.t8-result-$archive_sha.partial-$RELAY_ATTEMPT_ID"
local_final="$MAC_RELAY_ROOT/t8-result-$archive_sha"
autodl_partial="$AUTODL_IMPORT_PARENT/.t8-result-$archive_sha.partial-$RELAY_ATTEMPT_ID"
autodl_final="$AUTODL_IMPORT_PARENT/t8-result-$archive_sha"
for value in "$local_partial" "$local_final" "$autodl_partial" "$autodl_final"; do
  safe_path "$value" || { echo "invalid derived relay path" >&2; exit 65; }
done
adopt_existing_mac_final=false
if [[ -e "$local_final" ]]; then
  # A network interruption may happen after the content-addressed Mac relay was
  # independently verified and atomically sealed but before the AutoDL leg
  # finishes.  Re-adopt that immutable result instead of downloading 6.1 GB
  # from the HPC again.  The marker is necessary but not sufficient: re-hash
  # both archives and re-check the pinned package receipts below.
  [[ -d "$local_final" && ! -L "$local_final" ]] || { echo "existing Mac final is not a regular directory" >&2; exit 67; }
  [[ -f "$local_final/MAC_RELAY_READY.json" ]] || { echo "existing Mac final lacks MAC_RELAY_READY" >&2; exit 67; }
  state=VERIFYING_EXISTING_MAC_FINAL
  detail="$local_final"
  current_partial_path="$local_final/$EXPECTED_ARCHIVE_NAME"
  write_state "$state" "$detail"
  python3 - "$local_final" "$EXPECTED_ARCHIVE_BYTES" "$EXPECTED_ARCHIVE_SHA256" "$evidence_sha" <<'PY_EXISTING_MAC'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
expected_bytes, expected_sha, evidence_sha = int(sys.argv[2]), sys.argv[3], sys.argv[4]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

archive = root / "t8_exact_result_bundle.tar.gz"
evidence_archive = root / "t8_hierarchical_evidence.tar.gz"
manifest = json.loads((root / "result_manifest.json").read_text(encoding="utf-8"))
ready = json.loads((root / "HIERARCHICAL_PACKAGE_READY.json").read_text(encoding="utf-8"))
mac_ready = json.loads((root / "MAC_RELAY_READY.json").read_text(encoding="utf-8"))
if not archive.is_file() or archive.is_symlink() or archive.stat().st_size != expected_bytes:
    raise SystemExit("existing Mac result archive shape/size mismatch")
if digest(archive) != expected_sha:
    raise SystemExit("existing Mac result archive SHA mismatch")
if not evidence_archive.is_file() or evidence_archive.is_symlink() or digest(evidence_archive) != evidence_sha:
    raise SystemExit("existing Mac evidence archive SHA mismatch")
if manifest.get("status") != "PASS" or manifest.get("archive_bytes") != expected_bytes or manifest.get("archive_sha256") != expected_sha:
    raise SystemExit("existing Mac result manifest mismatch")
if ready.get("status") != "PASS" or ready.get("result_archive_sha256") != expected_sha or ready.get("evidence_archive_sha256") != evidence_sha:
    raise SystemExit("existing Mac package-ready receipt mismatch")
if mac_ready.get("state") != "MAC_RELAY_READY" or mac_ready.get("archive_bytes") != expected_bytes or mac_ready.get("archive_sha256") != expected_sha:
    raise SystemExit("existing Mac relay-ready receipt mismatch")
PY_EXISTING_MAC
  adopt_existing_mac_final=true
  local_partial=
  current_partial_path=
  write_state MAC_RELAY_READY "re-adopted verified content-addressed Mac final"
else
  mkdir -p "$local_partial"
fi

run_transfer() {
  transfer_state=$1
  transfer_detail=$2
  transfer_destination=$3
  shift 3
  transfer_attempt=1
  while (( transfer_attempt <= RELAY_TRANSFER_RETRIES )); do
    state=$transfer_state
    detail="$transfer_detail attempt=$transfer_attempt"
    current_partial_path=$transfer_destination
    write_state "$state" "$detail"
    "$@" &
    active_child=$!
    while kill -0 "$active_child" 2>/dev/null; do
      write_state "$state" "$detail"
      sleep "$RELAY_HEARTBEAT_SECONDS"
    done
    if wait "$active_child"; then
      active_child=
      write_state "$state" "$transfer_detail complete"
      return 0
    fi
    active_child=
    if (( transfer_attempt == RELAY_TRANSFER_RETRIES )); then
      return 1
    fi
    detail="$transfer_detail retry_pending=$((transfer_attempt + 1))"
    write_state RETRYING_TRANSFER "$detail"
    transfer_attempt=$((transfer_attempt + 1))
  done
  return 1
}

ssh_transport="ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT"
rsync_common_args=(-a --partial --append-verify --protect-args --info=progress2 -e "$ssh_transport")
if [[ "$adopt_existing_mac_final" != true ]]; then
  run_transfer COPYING_HPC_TO_MAC "result archive $archive_bytes bytes" "$local_partial/$EXPECTED_ARCHIVE_NAME" \
    "$RSYNC_BIN" "${rsync_common_args[@]}" \
    "$HPC_ALIAS:$hpc_archive_path" "$local_partial/$EXPECTED_ARCHIVE_NAME"
  run_transfer COPYING_HPC_TO_MAC "result manifest" "$local_partial/result_manifest.json" \
    "$RSYNC_BIN" "${rsync_common_args[@]}" \
    "$HPC_ALIAS:$result_manifest_path" "$local_partial/result_manifest.json"
  run_transfer COPYING_HPC_TO_MAC "hierarchical evidence" "$local_partial/t8_hierarchical_evidence.tar.gz" \
    "$RSYNC_BIN" "${rsync_common_args[@]}" \
    "$HPC_ALIAS:$evidence_archive_path" "$local_partial/t8_hierarchical_evidence.tar.gz"
  run_transfer COPYING_HPC_TO_MAC "evidence manifest" "$local_partial/hierarchical_evidence_manifest.json" \
    "$RSYNC_BIN" "${rsync_common_args[@]}" \
    "$HPC_ALIAS:$evidence_manifest_path" "$local_partial/hierarchical_evidence_manifest.json"
  run_transfer COPYING_HPC_TO_MAC "package ready receipt" "$local_partial/HIERARCHICAL_PACKAGE_READY.json" \
    "$RSYNC_BIN" "${rsync_common_args[@]}" \
    "$HPC_ALIAS:$package_ready_path" "$local_partial/HIERARCHICAL_PACKAGE_READY.json"

  state=VERIFYING_MAC_SHA256
  detail="$local_partial/$EXPECTED_ARCHIVE_NAME"
  current_partial_path="$local_partial/$EXPECTED_ARCHIVE_NAME"
  write_state "$state" "$detail"
  local_sha="$(shasum -a 256 "$local_partial/$EXPECTED_ARCHIVE_NAME" | awk '{print $1}')"
  local_size="$(wc -c < "$local_partial/$EXPECTED_ARCHIVE_NAME" | tr -d ' ')"
  local_evidence_sha="$(shasum -a 256 "$local_partial/t8_hierarchical_evidence.tar.gz" | awk '{print $1}')"
  [[ "$local_sha" == "$EXPECTED_ARCHIVE_SHA256" && "$local_size" == "$EXPECTED_ARCHIVE_BYTES" ]] || { echo "Mac result archive size/SHA mismatch" >&2; exit 66; }
  [[ "$local_evidence_sha" == "$evidence_sha" ]] || { echo "Mac evidence archive SHA mismatch" >&2; exit 66; }
  python3 - "$local_partial" "$EXPECTED_ARCHIVE_BYTES" "$EXPECTED_ARCHIVE_SHA256" "$evidence_sha" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
expected_bytes, expected_sha, evidence_sha = int(sys.argv[2]), sys.argv[3], sys.argv[4]
manifest = json.loads((root / "result_manifest.json").read_text(encoding="utf-8"))
ready = json.loads((root / "HIERARCHICAL_PACKAGE_READY.json").read_text(encoding="utf-8"))
if manifest.get("status") != "PASS" or manifest.get("archive_name") != "t8_exact_result_bundle.tar.gz":
    raise SystemExit("Mac manifest contract mismatch")
if manifest.get("archive_bytes") != expected_bytes or manifest.get("archive_sha256") != expected_sha:
    raise SystemExit("Mac manifest size/SHA mismatch")
if ready.get("status") != "PASS" or ready.get("result_archive_sha256") != expected_sha or ready.get("evidence_archive_sha256") != evidence_sha:
    raise SystemExit("Mac ready receipt mismatch")
payload = {
    "schema_version": "t8_scoped_mac_relay_ready_v2",
    "state": "MAC_RELAY_READY",
    "archive_bytes": expected_bytes,
    "archive_sha256": expected_sha,
    "evidence_archive_sha256": evidence_sha,
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "matrix_write_enabled": False,
    "hpc_source_delete_enabled": False,
}
data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd, tmp = tempfile.mkstemp(prefix=".ready.", dir=root)
with os.fdopen(fd, "wb") as stream:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(tmp, root / "MAC_RELAY_READY.json")
for path in root.iterdir():
    if path.is_file():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
dir_fd = os.open(root, os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
PY
  mv "$local_partial" "$local_final"
  local_partial=
  current_partial_path=
  write_state MAC_RELAY_READY "$local_final"
fi

# Create or resume only this attempt's fresh AutoDL staging directory.
$SSH_BIN -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$AUTODL_ALIAS" \
  "'$AUTODL_PYTHON' - '$autodl_partial' '$autodl_final'" <<'PY_AUTODL_PREPARE'
import os, sys
from pathlib import Path

partial, final = Path(sys.argv[1]), Path(sys.argv[2])
if final.exists():
    raise SystemExit("AutoDL final target already exists")
partial.mkdir(parents=True, exist_ok=True)
if partial.resolve().parent != final.resolve(strict=False).parent:
    raise SystemExit("AutoDL staging/final parent mismatch")
PY_AUTODL_PREPARE

run_transfer COPYING_MAC_TO_AUTODL "result archive $archive_bytes bytes" "$local_final/$EXPECTED_ARCHIVE_NAME" \
  "$RSYNC_BIN" "${rsync_common_args[@]}" \
  "$local_final/$EXPECTED_ARCHIVE_NAME" "$AUTODL_ALIAS:$autodl_partial/$EXPECTED_ARCHIVE_NAME"
run_transfer COPYING_MAC_TO_AUTODL "result manifest" "$local_final/result_manifest.json" \
  "$RSYNC_BIN" "${rsync_common_args[@]}" \
  "$local_final/result_manifest.json" "$AUTODL_ALIAS:$autodl_partial/result_manifest.json"
run_transfer COPYING_MAC_TO_AUTODL "hierarchical evidence" "$local_final/t8_hierarchical_evidence.tar.gz" \
  "$RSYNC_BIN" "${rsync_common_args[@]}" \
  "$local_final/t8_hierarchical_evidence.tar.gz" "$AUTODL_ALIAS:$autodl_partial/t8_hierarchical_evidence.tar.gz"
run_transfer COPYING_MAC_TO_AUTODL "evidence manifest" "$local_final/hierarchical_evidence_manifest.json" \
  "$RSYNC_BIN" "${rsync_common_args[@]}" \
  "$local_final/hierarchical_evidence_manifest.json" "$AUTODL_ALIAS:$autodl_partial/hierarchical_evidence_manifest.json"
run_transfer COPYING_MAC_TO_AUTODL "package ready receipt" "$local_final/HIERARCHICAL_PACKAGE_READY.json" \
  "$RSYNC_BIN" "${rsync_common_args[@]}" \
  "$local_final/HIERARCHICAL_PACKAGE_READY.json" "$AUTODL_ALIAS:$autodl_partial/HIERARCHICAL_PACKAGE_READY.json"

state=VERIFYING_AUTODL_SHA256
detail="$autodl_partial/$EXPECTED_ARCHIVE_NAME"
current_partial_path="$local_final/$EXPECTED_ARCHIVE_NAME"
write_state "$state" "$detail"
$SSH_BIN -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$AUTODL_ALIAS" \
  "'$AUTODL_PYTHON' - '$autodl_partial' '$autodl_final' '$EXPECTED_ARCHIVE_BYTES' '$EXPECTED_ARCHIVE_SHA256' '$evidence_sha' '$RELAY_ATTEMPT_ID'" <<'PY_AUTODL_VERIFY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

partial, final = Path(sys.argv[1]), Path(sys.argv[2])
expected_bytes, expected_sha, evidence_sha, attempt_id = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
if final.exists():
    raise SystemExit("AutoDL final target appeared during transfer")

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

archive = partial / "t8_exact_result_bundle.tar.gz"
evidence_archive = partial / "t8_hierarchical_evidence.tar.gz"
if archive.stat().st_size != expected_bytes or digest(archive) != expected_sha:
    raise SystemExit("AutoDL result archive size/SHA mismatch")
if digest(evidence_archive) != evidence_sha:
    raise SystemExit("AutoDL evidence archive SHA mismatch")
manifest = json.loads((partial / "result_manifest.json").read_text(encoding="utf-8"))
ready = json.loads((partial / "HIERARCHICAL_PACKAGE_READY.json").read_text(encoding="utf-8"))
if manifest.get("status") != "PASS" or manifest.get("archive_bytes") != expected_bytes or manifest.get("archive_sha256") != expected_sha:
    raise SystemExit("AutoDL result manifest mismatch")
if ready.get("status") != "PASS" or ready.get("result_archive_sha256") != expected_sha or ready.get("evidence_archive_sha256") != evidence_sha:
    raise SystemExit("AutoDL package-ready receipt mismatch")
marker = {
    "schema_version": "t8_hpc_package_ready_v2",
    "state": "HPC_PACKAGE_READY",
    "relay_attempt_id": attempt_id,
    "archive_bytes": expected_bytes,
    "archive_sha256": expected_sha,
    "hierarchical_evidence_sha256": evidence_sha,
    "received_at": datetime.now(timezone.utc).isoformat(),
    "independent_autodl_sha256_verified": True,
    "matrix_write_enabled": False,
}
data = (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd, tmp = tempfile.mkstemp(prefix=".ready.", dir=partial)
with os.fdopen(fd, "wb") as stream:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(tmp, partial / "HPC_PACKAGE_READY.json")
for path in partial.iterdir():
    if path.is_file():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
dir_fd = os.open(partial, os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
os.rename(partial, final)
parent_fd = os.open(final.parent, os.O_RDONLY)
try:
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
PY_AUTODL_VERIFY

autodl_partial=
current_partial_path=
state=PASS
detail="$autodl_final"
write_state "$state" "$detail"
trap - ERR TERM INT
cleanup
trap - EXIT
echo "T8_SCOPED_RELAY_PASS autodl_import_root=$autodl_final"
