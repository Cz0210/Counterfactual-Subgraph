#!/usr/bin/env bash
# Credential-free, resumable HPC -> Mac external disk -> AutoDL T8 relay.
# Run under `caffeinate -dimsu`; this script never publishes the matrix.
set -euo pipefail

HPC_ALIAS=${HPC_ALIAS:-tongji-hpc}
AUTODL_ALIAS=${AUTODL_ALIAS:-autodl-a800}
HPC_PACKAGE_ROOT=${HPC_PACKAGE_ROOT:?HPC_PACKAGE_ROOT is required}
MAC_RELAY_ROOT=${MAC_RELAY_ROOT:-/Volumes/DireRaven/counterfactual-hpc-offload}
AUTODL_IMPORT_PARENT=${AUTODL_IMPORT_PARENT:-/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/imports/t8-hpc}
AUTODL_PYTHON=${AUTODL_PYTHON:-/root/miniconda3/envs/smiles_pip118/bin/python}
RELAY_CONTROL_ROOT=${RELAY_CONTROL_ROOT:-$HOME/.cache/t8-result-relay-v1}
RELAY_POLL_SECONDS=${RELAY_POLL_SECONDS:-300}
RELAY_TRANSFER_RETRIES=${RELAY_TRANSFER_RETRIES:-3}
SSH_CONNECT_TIMEOUT=${SSH_CONNECT_TIMEOUT:-15}

safe_alias() { case "$1" in ''|*[!A-Za-z0-9_.-]*) return 1;; *) return 0;; esac; }
safe_path() { case "$1" in /*[!A-Za-z0-9_./:+-]*|*[!A-Za-z0-9_./:+-]*|'') return 1;; /*) return 0;; *) return 1;; esac; }
is_uint() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }
safe_alias "$HPC_ALIAS" && safe_alias "$AUTODL_ALIAS" || { echo "invalid SSH alias" >&2; exit 64; }
for value in "$HPC_PACKAGE_ROOT" "$MAC_RELAY_ROOT" "$AUTODL_IMPORT_PARENT" "$AUTODL_PYTHON" "$RELAY_CONTROL_ROOT"; do
  safe_path "$value" || { echo "invalid relay path" >&2; exit 64; }
done
is_uint "$RELAY_POLL_SECONDS" && is_uint "$RELAY_TRANSFER_RETRIES" && is_uint "$SSH_CONNECT_TIMEOUT" || { echo "invalid numeric relay setting" >&2; exit 64; }

mkdir -p "$RELAY_CONTROL_ROOT" "$MAC_RELAY_ROOT"
chmod 700 "$RELAY_CONTROL_ROOT"
lock_dir="$RELAY_CONTROL_ROOT/lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  old_pid="$(cat "$RELAY_CONTROL_ROOT/pid" 2>/dev/null || true)"
  if is_uint "$old_pid" && kill -0 "$old_pid" 2>/dev/null; then
    echo "relay already running pid=$old_pid" >&2
    exit 73
  fi
  rmdir "$lock_dir" 2>/dev/null || { echo "stale relay lock requires inspection" >&2; exit 73; }
  mkdir "$lock_dir"
fi
printf '%s\n' "$$" > "$RELAY_CONTROL_ROOT/pid"

write_state() {
  state=$1
  detail=${2:-}
  python3 - "$RELAY_CONTROL_ROOT/state.json" "$state" "$detail" "$$" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {"schema_version":"t8_result_relay_state_v1","state":sys.argv[2],"detail":sys.argv[3],"pid":int(sys.argv[4]),"updated_at":datetime.now(timezone.utc).isoformat(),"matrix_write_enabled":False}
data = (json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True)+"\n").encode()
fd, tmp = tempfile.mkstemp(prefix=".state.", dir=path.parent)
try:
    with os.fdopen(fd,"wb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp,path)
finally:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
PY
}

cleanup() {
  rm -f "$RELAY_CONTROL_ROOT/pid"
  rmdir "$lock_dir" 2>/dev/null || true
}
interrupted() { write_state INTERRUPTED "relay received signal" || true; cleanup; exit 143; }
trap cleanup EXIT
trap interrupted TERM INT

retry_transfer() {
  attempt=1
  while ! "$@"; do
    if (( attempt >= RELAY_TRANSFER_RETRIES )); then return 1; fi
    delay=$((30 * attempt))
    write_state RETRYING_TRANSFER "attempt=$attempt delay_seconds=$delay"
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

write_state WAITING_HPC_PACKAGE "$HPC_PACKAGE_ROOT"
while true; do
  metadata="$(ssh -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$HPC_ALIAS" \
    "python3 - '$HPC_PACKAGE_ROOT'" 2>/dev/null <<'PY_REMOTE' || true
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=root/'result_manifest.json'
archive=root/'t8_exact_result_bundle.tar.gz'
evidence_manifest=root/'hierarchical_evidence_manifest.json'
evidence_archive=root/'t8_hierarchical_evidence.tar.gz'
ready=root/'HIERARCHICAL_PACKAGE_READY.json'
if not all(path.is_file() for path in (manifest,archive,evidence_manifest,evidence_archive,ready)): raise SystemExit(0)
try: p=json.loads(manifest.read_text()); e=json.loads(evidence_manifest.read_text()); r=json.loads(ready.read_text())
except Exception: raise SystemExit(0)
sha=p.get('archive_sha256','')
evidence_sha=e.get('archive_sha256','')
if p.get('status')=='PASS' and e.get('status')=='PASS' and r.get('status')=='PASS' and re.fullmatch(r'[0-9a-f]{64}',str(sha)) and re.fullmatch(r'[0-9a-f]{64}',str(evidence_sha)) and p.get('archive_name')==archive.name and r.get('result_archive_sha256')==sha and r.get('evidence_archive_sha256')==evidence_sha:
    print(f"{sha}|{archive.stat().st_size}|{evidence_sha}")
PY_REMOTE
)"
  case "$metadata" in *'|'*'|'*) break;; esac
  write_state WAITING_HPC_PACKAGE "$HPC_PACKAGE_ROOT"
  sleep "$RELAY_POLL_SECONDS"
done

archive_sha=${metadata%%|*}
remainder=${metadata#*|}
archive_bytes=${remainder%%|*}
evidence_sha=${remainder#*|}
case "$archive_sha" in *[!0-9a-f]*|'') echo "invalid package SHA" >&2; exit 65;; esac
is_uint "$archive_bytes" || { echo "invalid package size" >&2; exit 65; }
case "$evidence_sha" in *[!0-9a-f]*|'') echo "invalid evidence SHA" >&2; exit 65;; esac
local_final="$MAC_RELAY_ROOT/t8-result-$archive_sha"
local_partial="$MAC_RELAY_ROOT/.t8-result-$archive_sha.partial"
mkdir -p "$local_partial"
write_state COPYING_HPC_TO_MAC "$archive_bytes bytes"
retry_transfer rsync -a --partial --append-verify -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$HPC_ALIAS:$HPC_PACKAGE_ROOT/t8_exact_result_bundle.tar.gz" "$local_partial/t8_exact_result_bundle.tar.gz"
retry_transfer rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$HPC_ALIAS:$HPC_PACKAGE_ROOT/result_manifest.json" "$local_partial/result_manifest.json"
retry_transfer rsync -a --partial --append-verify -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$HPC_ALIAS:$HPC_PACKAGE_ROOT/t8_hierarchical_evidence.tar.gz" "$local_partial/t8_hierarchical_evidence.tar.gz"
retry_transfer rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$HPC_ALIAS:$HPC_PACKAGE_ROOT/hierarchical_evidence_manifest.json" "$local_partial/hierarchical_evidence_manifest.json"
retry_transfer rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$HPC_ALIAS:$HPC_PACKAGE_ROOT/HIERARCHICAL_PACKAGE_READY.json" "$local_partial/HIERARCHICAL_PACKAGE_READY.json"
local_sha="$(shasum -a 256 "$local_partial/t8_exact_result_bundle.tar.gz" | awk '{print $1}')"
[[ "$local_sha" == "$archive_sha" ]] || { write_state FAILED "Mac archive SHA mismatch"; exit 66; }
local_evidence_sha="$(shasum -a 256 "$local_partial/t8_hierarchical_evidence.tar.gz" | awk '{print $1}')"
[[ "$local_evidence_sha" == "$evidence_sha" ]] || { write_state FAILED "Mac evidence SHA mismatch"; exit 66; }
if [[ ! -d "$local_final" ]]; then
  mv "$local_partial" "$local_final"
else
  existing_sha="$(shasum -a 256 "$local_final/t8_exact_result_bundle.tar.gz" 2>/dev/null | awk '{print $1}')"
  existing_evidence_sha="$(shasum -a 256 "$local_final/t8_hierarchical_evidence.tar.gz" 2>/dev/null | awk '{print $1}')"
  [[ "$existing_sha" == "$archive_sha" && "$existing_evidence_sha" == "$evidence_sha" ]] || { write_state FAILED "existing Mac relay target differs"; exit 67; }
  rm -rf "$local_partial"
fi

remote_final="$AUTODL_IMPORT_PARENT/t8-result-$archive_sha"
remote_partial="$AUTODL_IMPORT_PARENT/.t8-result-$archive_sha.partial"
write_state COPYING_MAC_TO_AUTODL "$remote_final"
if ssh -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$AUTODL_ALIAS" \
  "'$AUTODL_PYTHON' - '$remote_final' '$archive_sha' '$evidence_sha'" <<'PY_AUTODL_EXISTING'
import hashlib,json,sys
from pathlib import Path
root,expected,expected_evidence=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
marker=root/'HPC_PACKAGE_READY.json'
if not marker.is_file(): raise SystemExit(1)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()
payload=json.loads(marker.read_text())
if digest(root/'t8_exact_result_bundle.tar.gz')!=expected or digest(root/'t8_hierarchical_evidence.tar.gz')!=expected_evidence or payload.get('archive_sha256')!=expected or payload.get('hierarchical_evidence_sha256')!=expected_evidence: raise SystemExit(1)
PY_AUTODL_EXISTING
then
  write_state PASS "$remote_final"
  trap - EXIT TERM INT
  cleanup
  echo "T8_RESULT_RELAY_PASS autodl_import_root=$remote_final"
  exit 0
fi
ssh -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$AUTODL_ALIAS" \
  "test ! -e '$remote_final' && mkdir -p '$remote_partial'"
retry_transfer rsync -a --partial --append-verify -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$local_final/t8_exact_result_bundle.tar.gz" "$AUTODL_ALIAS:$remote_partial/t8_exact_result_bundle.tar.gz"
retry_transfer rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$local_final/result_manifest.json" "$AUTODL_ALIAS:$remote_partial/result_manifest.json"
retry_transfer rsync -a --partial --append-verify -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$local_final/t8_hierarchical_evidence.tar.gz" "$AUTODL_ALIAS:$remote_partial/t8_hierarchical_evidence.tar.gz"
retry_transfer rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$local_final/hierarchical_evidence_manifest.json" "$AUTODL_ALIAS:$remote_partial/hierarchical_evidence_manifest.json"
retry_transfer rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=$SSH_CONNECT_TIMEOUT" \
  "$local_final/HIERARCHICAL_PACKAGE_READY.json" "$AUTODL_ALIAS:$remote_partial/HIERARCHICAL_PACKAGE_READY.json"
ssh -q -o BatchMode=yes -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT" "$AUTODL_ALIAS" \
  "'$AUTODL_PYTHON' - '$remote_partial' '$remote_final' '$archive_sha' '$evidence_sha'" <<'PY_AUTODL'
import hashlib,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
partial,final=map(Path,sys.argv[1:3]); expected=sys.argv[3]; expected_evidence=sys.argv[4]
archive=partial/'t8_exact_result_bundle.tar.gz'
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
if h.hexdigest()!=str(expected): raise SystemExit('AutoDL archive SHA mismatch')
evidence=partial/'t8_hierarchical_evidence.tar.gz'; h=hashlib.sha256()
with evidence.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
if h.hexdigest()!=expected_evidence: raise SystemExit('AutoDL evidence SHA mismatch')
manifest=json.loads((partial/'result_manifest.json').read_text())
if manifest.get('status')!='PASS' or manifest.get('archive_sha256')!=str(expected): raise SystemExit('AutoDL manifest mismatch')
hierarchy=json.loads((partial/'HIERARCHICAL_PACKAGE_READY.json').read_text())
if hierarchy.get('status')!='PASS' or hierarchy.get('result_archive_sha256')!=expected or hierarchy.get('evidence_archive_sha256')!=expected_evidence: raise SystemExit('AutoDL hierarchy marker mismatch')
marker={"schema_version":"t8_hpc_package_ready_v1","state":"HPC_PACKAGE_READY","archive_sha256":str(expected),"hierarchical_evidence_sha256":expected_evidence,"received_at":datetime.now(timezone.utc).isoformat(),"matrix_write_enabled":False}
data=(json.dumps(marker,sort_keys=True,separators=(",",":"))+"\n").encode()
fd,tmp=tempfile.mkstemp(prefix='.ready.',dir=partial)
with os.fdopen(fd,'wb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
os.replace(tmp,partial/'HPC_PACKAGE_READY.json')
os.rename(partial,final)
PY_AUTODL

write_state PASS "$remote_final"
trap - EXIT TERM INT
rm -f "$RELAY_CONTROL_ROOT/pid"
rmdir "$lock_dir"
echo "T8_RESULT_RELAY_PASS autodl_import_root=$remote_final"
