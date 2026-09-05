"""One campaign's verified GNN package: HPC -> Mac external disk -> AutoDL.

No scheduler, science launcher, GPU lease or matrix publisher lives here.
Only the fixed fd98c5f2 campaign is accepted. Transport failures are terminal;
source files and partial destinations are retained for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import time
from typing import Any
import uuid


HPC_HOST = "tongji-hpc"
AUTODL_HOST = "autodl-a800"
HPC_ROOT = Path("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/gnn/runs/bace-seed7-20260905T105800Z/exact-parent-closeout-fd98c5f2/verified")
HPC_RECEIPT = HPC_ROOT / "result_package.json"
HPC_JOBS = ("2558894", "2558895", "2558896", "2558897", "2558898", "2558899", "2558901")
HPC_PYTHON = "/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python"
VOLUME = Path("/Volumes/DireRaven")
MAC_PARENT = VOLUME / "counterfactual-hpc-offload/gnn-seed7-closeout-fd98c5f2"
AUTODL_PARENT = Path("/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/ablations/gnn/seed7-import-fd98c5f2-20260905")
AUTODL_WORKTREE = Path("/root/autodl-tmp/worktrees/bace-gnn-closeout-fd98c5f2")
AUTODL_PYTHON = "/root/miniconda3/envs/smiles_pip118/bin/python"
ARCHIVE_NAME = "bace_gnn_seed7_verified.tar.gz"
SCIENTIFIC_ENGINE_COMMIT = "532e83733971701b0709086469d2ed8955a96e25"
PUBLICATION_DRIVER_COMMIT = "fd98c5f23bf835f2b68799d03b7a2fd8b8b713f7"
EXPECTED_BACKBONES = ("gine", "gin", "gcn", "gatv2", "gatedgcn_plus")
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
HEARTBEAT_SECONDS = 60
MAX_WAIT_SECONDS = 7 * 24 * 3600


class RelayError(RuntimeError):
    pass


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RelayError(f"Archive is not a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RelayError("Archive changed while hashing")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


@dataclass(frozen=True)
class RelayPlan:
    attempt_id: str

    def __post_init__(self):
        parsed = uuid.UUID(self.attempt_id)
        if parsed.version != 4 or str(parsed) != self.attempt_id:
            raise RelayError("A canonical fresh UUIDv4 is required")

    @property
    def mac_root(self):
        return MAC_PARENT / self.attempt_id

    @property
    def control(self):
        return self.mac_root / "control"

    @property
    def local_partial(self):
        return self.mac_root / (ARCHIVE_NAME + ".partial")

    @property
    def local_final(self):
        return self.mac_root / ARCHIVE_NAME

    @property
    def incoming(self):
        return AUTODL_PARENT / ("incoming-" + self.attempt_id)

    @property
    def remote_partial(self):
        return self.incoming / (ARCHIVE_NAME + ".partial")

    @property
    def remote_final(self):
        return self.incoming / ARCHIVE_NAME

    @property
    def import_root(self):
        return AUTODL_PARENT / ("import-" + self.attempt_id)

    def to_dict(self):
        return {"attempt_id": self.attempt_id, "hpc_receipt": str(HPC_RECEIPT),
            "hpc_job_chain": list(HPC_JOBS), "mac_root": str(self.mac_root),
            "control_root": str(self.control), "autodl_incoming": str(self.incoming),
            "autodl_import_root": str(self.import_root), "poll_seconds": HEARTBEAT_SECONDS,
            "max_wait_seconds": MAX_WAIT_SECONDS, "transfer_retries": 0,
            "scientific_engine_commit": SCIENTIFIC_ENGINE_COMMIT,
            "publication_driver_commit": PUBLICATION_DRIVER_COMMIT,
            "backbones": list(EXPECTED_BACKBONES),
            "main_matrix_write": False, "llm_started": False, "gpu_requested": False}


def validate_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("state") != "PASS" or value.get("main_matrix_write") is not False:
        raise RelayError("HPC package receipt is not an independent non-matrix PASS")
    if value.get("path") != str(HPC_ROOT / ARCHIVE_NAME):
        raise RelayError("Receipt path escapes the fixed campaign archive")
    if (value.get("scientific_engine_commit") != SCIENTIFIC_ENGINE_COMMIT
            or value.get("publication_driver_commit") != PUBLICATION_DRIVER_COMMIT):
        raise RelayError("Receipt scientific engine/publication driver commit differs")
    size, digest = value.get("bytes"), value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise RelayError("Invalid receipt bytes")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RelayError("Invalid receipt SHA256")
    return value


def ssh_python(host: str, python: str, code: str, *args: str) -> list[str]:
    if host not in (HPC_HOST, AUTODL_HOST):
        raise RelayError("Host outside this campaign")
    return ["ssh", *SSH_OPTIONS, host, shlex.join([python, "-c", code, *map(str, args)])]


RECEIPT_QUERY = r'''
import json,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);jobs=sys.argv[2].split(',')
def receipt():
    if p.is_symlink():raise RuntimeError('Receipt symlink is forbidden')
    if p.is_file():
        if p.stat().st_size>1048576:raise RuntimeError('Oversized package receipt')
        return json.loads(p.read_text())
    return None
value=receipt()
if value is not None:
    print(json.dumps({'state':'RECEIPT_READY','receipt':value}));raise SystemExit(0)
r=subprocess.run(['sacct','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True)
rows=[]
for line in r.stdout.splitlines():
    fields=line.strip().split('|')
    if len(fields)>=3 and '.' not in fields[0]:rows.append({'job':fields[0],'state':fields[1],'exit_code':fields[2]})
bad=('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED')
failed=[row for row in rows if row['state'].split()[0].rstrip('+') in bad]
completed=any(row['job']==jobs[-1] and row['state']=='COMPLETED' for row in rows)
if failed or completed:
    value=receipt()
    if value is not None:print(json.dumps({'state':'RECEIPT_READY','receipt':value}));raise SystemExit(0)
    print(json.dumps({'state':'CHAIN_FAILED' if failed else 'PACKAGE_MISSING_AFTER_PUBLICATION_EXIT','jobs':rows}));raise SystemExit(0)
print(json.dumps({'state':'WAITING_PACKAGE','jobs':rows}))
'''

VERIFY_ARCHIVE = r'''
import hashlib,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);expected_size=int(sys.argv[2]);expected_sha=sys.argv[3]
if p.is_symlink() or not p.is_file() or p.stat().st_size!=expected_size:raise RuntimeError('Archive bytes/type mismatch')
before=p.stat();digest=hashlib.sha256()
with p.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
after=p.stat()
if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise RuntimeError('Archive changed during hash')
if digest.hexdigest()!=expected_sha:raise RuntimeError('Archive SHA mismatch')
if len(sys.argv)>4:
    destination=pathlib.Path(sys.argv[4])
    if destination.parent!=p.parent or destination.exists() or destination.is_symlink():raise RuntimeError('Final destination not fresh sibling')
    with p.open('rb') as stream:os.fsync(stream.fileno())
    os.replace(p,destination)
    directory=os.open(destination.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    p=destination
print(json.dumps({'state':'PASS','path':str(p),'bytes':after.st_size,'sha256':digest.hexdigest()}))
'''

PREPARE_AUTODL = r'''
import json,pathlib,sys
parent,incoming,output=map(pathlib.Path,sys.argv[1:])
if incoming.parent!=parent or output.parent!=parent or incoming==output:raise RuntimeError('Invalid fixed incoming/import roots')
if any(p.is_symlink() for target in (parent,incoming,output) for p in (target,*target.parents)):raise RuntimeError('Symlink destination')
if incoming.exists() or output.exists():raise RuntimeError('AutoDL incoming/import root must be fresh')
parent.mkdir(parents=True,exist_ok=True);incoming.mkdir(exist_ok=False)
print(json.dumps({'state':'READY','incoming':str(incoming),'import_root':str(output)}))
'''


def transfer_command(source: str, destination: str) -> list[str]:
    # All options are supported by the Mac's system rsync 2.6.9. No deletion,
    # in-place archive mutation, append-verify, protect-args or --info options.
    return ["rsync", "-a", "--partial", "--timeout=300", "-e", shlex.join(["ssh", *SSH_OPTIONS]), source, destination]


def import_command(plan: RelayPlan, receipt: dict[str, Any]) -> list[str]:
    argv = [AUTODL_PYTHON, str(AUTODL_WORKTREE / "scripts/hpc/gnn/import_bace_gnn_verified.py"),
        "--config", "configs/hpc.yaml", "--archive-path", str(plan.remote_final),
        "--expected-sha256", receipt["sha256"], "--output-root", str(plan.import_root)]
    remote = "cd " + shlex.quote(str(AUTODL_WORKTREE)) + " && " + shlex.join([
        "env", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", *argv])
    return ["ssh", *SSH_OPTIONS, AUTODL_HOST, remote]


class Heartbeat:
    def __init__(self, plan: RelayPlan):
        self.plan = plan
        self.state = {**plan.to_dict(), "schema_version": "gnn_seed7_mac_relay_v1", "pid": os.getpid(), "state": "STARTING", "active_child_pid": None}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
            self.state["local_partial_bytes"] = self.plan.local_partial.stat().st_size if self.plan.local_partial.is_file() else None
            atomic_json(self.plan.control / "heartbeat.json", dict(self.state))

    def _loop(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            self.update()

    def __enter__(self):
        self.update()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)


class CommandRunner:
    def __init__(self, control: Path, heartbeat: Heartbeat):
        self.control, self.heartbeat, self.sequence = control, heartbeat, 0

    def run(self, argv: list[str], stage: str) -> str:
        self.sequence += 1
        prefix = f"{self.sequence:05d}-{stage}"
        self.heartbeat.update(state=stage, command=argv)
        with (self.control / (prefix + ".stdout.log")).open("xb") as stdout, (self.control / (prefix + ".stderr.log")).open("xb") as stderr:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            self.heartbeat.update(active_child_pid=child.pid)
            try:
                code = child.wait()
            except BaseException:
                # Only this relay's precise local transport child. No remote
                # process signal, process group kill or automatic retry.
                child.terminate()
                raise
            finally:
                self.heartbeat.update(active_child_pid=None)
        if code != 0:
            raise RelayError(f"{stage} exit={code}; logs retained at {prefix}")
        path = self.control / (prefix + ".stdout.log")
        if path.stat().st_size > 1024 * 1024:
            raise RelayError(f"{stage} unexpected oversized command output")
        return path.read_text()


def _object(text: str) -> dict[str, Any]:
    try:
        result = json.loads(text.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RelayError("Malformed remote JSON; no retry") from exc
    if not isinstance(result, dict):
        raise RelayError("Remote output is not an object")
    return result


def transfer_and_import(plan: RelayPlan, receipt: dict[str, Any], runner: CommandRunner, pulse: Heartbeat) -> dict[str, Any]:
    receipt = validate_receipt(receipt)
    atomic_json(plan.control / "source_result_package.json", receipt)
    source = _object(runner.run(ssh_python(HPC_HOST, HPC_PYTHON, VERIFY_ARCHIVE, receipt["path"], str(receipt["bytes"]), receipt["sha256"]), "VERIFY_HPC_ARCHIVE"))
    if source.get("sha256") != receipt["sha256"] or source.get("bytes") != receipt["bytes"]:
        raise RelayError("HPC transport identity differs")
    atomic_json(plan.control / "hpc_transport_receipt.json", source)
    runner.run(transfer_command(HPC_HOST + ":" + receipt["path"], str(plan.local_partial)), "HPC_TO_MAC")
    pulse.update(state="VERIFY_MAC_ARCHIVE")
    local = file_identity(plan.local_partial)
    if local["sha256"] != receipt["sha256"] or local["bytes"] != receipt["bytes"]:
        raise RelayError("Mac archive bytes/SHA differs; partial preserved")
    if plan.local_final.exists() or plan.local_final.is_symlink():
        raise RelayError("Mac final archive already exists")
    with plan.local_partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(plan.local_partial, plan.local_final)
    directory = os.open(plan.mac_root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    local["path"] = str(plan.local_final)
    atomic_json(plan.control / "mac_transport_receipt.json", local)
    runner.run(ssh_python(AUTODL_HOST, AUTODL_PYTHON, PREPARE_AUTODL, str(AUTODL_PARENT), str(plan.incoming), str(plan.import_root)), "PREPARE_AUTODL_FRESH_ROOTS")
    runner.run(transfer_command(str(plan.local_final), AUTODL_HOST + ":" + str(plan.remote_partial)), "MAC_TO_AUTODL")
    remote = _object(runner.run(ssh_python(AUTODL_HOST, AUTODL_PYTHON, VERIFY_ARCHIVE, str(plan.remote_partial), str(receipt["bytes"]), receipt["sha256"], str(plan.remote_final)), "VERIFY_AUTODL_ARCHIVE"))
    if remote.get("sha256") != receipt["sha256"] or remote.get("bytes") != receipt["bytes"] or remote.get("path") != str(plan.remote_final):
        raise RelayError("AutoDL transport identity differs")
    atomic_json(plan.control / "autodl_transport_receipt.json", remote)
    imported = _object(runner.run(import_command(plan, receipt), "AUTODL_INDEPENDENT_IMPORT"))
    if (imported.get("state") != "PASS"
            or imported.get("archive_sha256") != receipt["sha256"]
            or imported.get("evaluation_root") != str(plan.import_root / "evaluation")
            or imported.get("model_roots") != {name: str(plan.import_root / "classifiers" / name) for name in EXPECTED_BACKBONES}
            or imported.get("scientific_engine_commit") != SCIENTIFIC_ENGINE_COMMIT
            or imported.get("publication_driver_commit") != PUBLICATION_DRIVER_COMMIT
            or any(imported.get(key) is not False for key in (
                "main_matrix_write", "classifier_inference_rerun", "ot_recomputed", "historical_hpc_paths_opened"))
            or imported.get("original_manifest_paths_preserved") is not True):
        raise RelayError("AutoDL independent import did not PASS")
    atomic_json(plan.control / "autodl_import_receipt.json", imported)
    return {"state": "VERIFIED_PACKAGE_IMPORTED", "archive_sha256": receipt["sha256"], "archive_bytes": receipt["bytes"],
        "hpc_source": receipt["path"], "mac_archive": str(plan.local_final), "autodl_archive": str(plan.remote_final),
        "autodl_import_root": str(plan.import_root), "source_files_preserved": True, "main_matrix_write": False,
        "llm_started": False, "gpu_requested": False}


def run_relay(plan: RelayPlan) -> dict[str, Any]:
    if not VOLUME.is_mount():
        raise RelayError("External disk is not mounted; refusing local-disk fallback")
    if any(p.is_symlink() for p in (plan.mac_root, *plan.mac_root.parents)):
        raise RelayError("Mac staging ancestor may not be a symlink")
    MAC_PARENT.mkdir(parents=True, exist_ok=True)
    plan.mac_root.mkdir(exist_ok=False)
    plan.control.mkdir()
    atomic_json(plan.control / "plan.json", plan.to_dict())
    atomic_json(plan.control / "pid.json", {"pid": os.getpid(), "attempt_id": plan.attempt_id})
    with Heartbeat(plan) as pulse:
        runner = CommandRunner(plan.control, pulse)
        started = time.monotonic()
        try:
            while True:
                result = _object(runner.run(ssh_python(HPC_HOST, HPC_PYTHON, RECEIPT_QUERY, str(HPC_RECEIPT), ",".join(HPC_JOBS)), "READ_HPC_PACKAGE_STATUS"))
                atomic_json(plan.control / "last_hpc_status.json", result)
                if result.get("state") == "RECEIPT_READY":
                    terminal = transfer_and_import(plan, result["receipt"], runner, pulse)
                    break
                if result.get("state") != "WAITING_PACKAGE":
                    raise RelayError(f"HPC chain terminal: {result.get('state')}")
                if time.monotonic() - started >= MAX_WAIT_SECONDS:
                    raise RelayError("PACKAGE_WAIT_DEADLINE_EXCEEDED_7_DAYS")
                pulse.update(state="WAITING_PACKAGE")
                time.sleep(HEARTBEAT_SECONDS)
        except BaseException as exc:
            terminal = {"state": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "partial_files_preserved": True, "automatic_retry": False, "main_matrix_write": False}
            atomic_json(plan.control / "terminal.json", terminal)
            pulse.update(**terminal)
            raise
        atomic_json(plan.control / "terminal.json", terminal)
        pulse.update(**terminal)
        return terminal
