"""Small CM-CReM file/receipt helpers; no controller, GPU lock or main registry."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_json(path: str | Path, value: Any, *, immutable: bool = False) -> None:
    path = Path(path)
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and immutable:
        if path.read_bytes() != body:
            raise ValueError(f"immutable receipt conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("xb") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())
    if immutable:
        os.link(tmp, path)  # no replacement if a competing writer published first
        tmp.unlink()
    else:
        os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require_compute_node() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CM heavy science requires a Slurm allocation; login inference forbidden")


def checked_root(root: str | Path, allowroot: str | Path) -> Path:
    path, base = Path(root).absolute(), Path(allowroot).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("task root must be a strict child of the authorized root")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"symlink in write path: {parent}")
    return path


def storage_probe(root: str | Path, allowroot: str | Path, timeout_s: int = 40) -> dict:
    """One 64-KiB fresh-file durability check; never opens an experiment DB."""
    root = checked_root(root, allowroot)
    code = r'''
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=False)
b=bytes(range(256))*256
def event(name): print(json.dumps({'operation':name}),flush=True)
event('open'); f=(p/'probe.tmp').open('xb')
event('write'); f.write(b); f.flush()
event('fsync'); os.fsync(f.fileno())
event('close'); f.close()
event('rename'); os.rename(p/'probe.tmp',p/'probe.bin')
event('directory_fsync'); d=os.open(p,os.O_RDONLY); os.fsync(d); os.close(d)
event('reopen'); actual=(p/'probe.bin').read_bytes()
assert actual==b
s=os.statvfs(p)
print(json.dumps({'status':'PASS_BOUNDED_IO','bytes_checked':len(b),
'sha256':hashlib.sha256(actual).hexdigest(),'available_bytes':s.f_bavail*s.f_frsize,
'f_files':s.f_files,'f_favail':s.f_favail}),flush=True)
'''
    start = time.monotonic()
    receipt = {"schema": "cm_crem_bounded_io_v1", "root": str(root),
               "created_at": utc_now(), "timeout_s": timeout_s}
    try:
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root)],
                                text=True, capture_output=True, timeout=timeout_s)
        receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        receipt["status"] = "PASS_BOUNDED_IO" if result.returncode == 0 else "FAILED_IO"
        if result.returncode == 0:
            receipt.update(json.loads(result.stdout.splitlines()[-1]))
    except subprocess.TimeoutExpired as exc:
        def decode(s): return s.decode(errors="replace") if isinstance(s, bytes) else s
        receipt.update(status="BLOCKED_IO_TIMEOUT", stdout=decode(exc.stdout), stderr=decode(exc.stderr))
    receipt["elapsed_seconds"] = time.monotonic() - start
    if receipt.get("f_files", 0) >= 2**60:
        receipt["inode_quota_state"] = "UNAVAILABLE_PLACEHOLDER_NOT_USER_QUOTA"
        receipt["effective_available_file_slots"] = None
    elif "f_favail" in receipt:
        receipt["inode_quota_state"] = "CLIENT_REPORTED"
        receipt["effective_available_file_slots"] = receipt["f_favail"]
    return receipt


def cgroup_memory() -> dict:
    def first(paths):
        for p in paths:
            if Path(p).exists():
                text = Path(p).read_text().strip()
                number = int(text) if text.isdecimal() else None
                return {"path": p, "bytes": number if number is not None and number < 2**60 else None}
        return {"path": None, "bytes": None}
    limit = first(["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"])
    usage = first(["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"])
    headroom = None if None in (limit["bytes"], usage["bytes"]) else limit["bytes"] - usage["bytes"]
    return {"limit": limit, "usage": usage, "headroom_bytes": headroom}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--allowroot", required=True)
    args = parser.parse_args()
    print(json.dumps({"storage": storage_probe(args.root, args.allowroot),
                      "cgroup": cgroup_memory()}, sort_keys=True))
