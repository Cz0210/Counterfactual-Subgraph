#!/usr/bin/env python3
"""One bounded Mac CM-CReM relay: fixed stage DAG, no Codex/API calls or GPU actions."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cm_crem_runtime import atomic_json, utc_now, read_json, file_sha

STAGES = ["pilot-oracle", "pilot-generate", "pilot-filter", "pilot-closeout", "attribution",
          "generate", "filter", "encode", "calibrate", "select", "test", "audit", "export", "package"]


def ssh_read(alias: str, argv: list[str], timeout: int = 90) -> str:
    run = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias,
                          shlex.join(argv)], capture_output=True, text=True, timeout=timeout)
    if run.returncode:
        raise RuntimeError(f"SSH returncode={run.returncode}: {run.stderr[-2000:]}")
    return run.stdout


def require_import_identity(receipt: dict, manifest: dict) -> None:
    """A science contract can produce several packages; do not conflate them."""
    if receipt.get("status") != "CM_RESULT_IMPORT_VERIFIED" or any(
        receipt.get(key) != manifest.get(key)
        for key in ("science_hash", "package_sha256", "package_bytes")
    ) or not manifest.get("package_sha256") or not isinstance(manifest.get("package_bytes"), int):
        raise ValueError("Completed import identity does not match this exact CM package")


def transfer_completed(args, local: Path, python: str) -> dict:
    """Transport only this accepted CM package; no models/DB/main authority."""
    from src.baselines.cm_crem_release import verify_import
    code = "import pathlib,sys; print((pathlib.Path(sys.argv[1])/'release/package_receipt.json').read_text())"
    package = json.loads(ssh_read(args.hpc_alias, [python, "-I", "-B", "-c", code, args.hpc_run_root]))
    remote_files = [package["package_path"], package["manifest_path"]]
    if any(Path(p).parent != Path(args.hpc_run_root)/"release" for p in remote_files):
        raise ValueError("Package producer pointed outside this run's release")
    transfer = local/"transfer"
    transfer.mkdir(exist_ok=True)
    for remote in remote_files:
        subprocess.run(["rsync", "-t", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                        args.hpc_alias+":"+remote, str(transfer/Path(remote).name)], check=True, timeout=3600)
    local_package, local_manifest = [transfer/Path(p).name for p in remote_files]
    expected = read_json(local_manifest)
    imported = local/"import"
    if (imported/"cm_import_receipt.json").exists():
        local_receipt = read_json(imported/"cm_import_receipt.json")
    else:
        local_receipt = verify_import(local_package, local_manifest, imported)
    require_import_identity(local_receipt, expected)
    # A tiny source-pinned importer, not an entire runtime or a new controller.
    tools_root = local/"delivery_tools"
    copied = {}
    for relative in ("scripts/import_cm_crem.py", "src/baselines/cm_crem_release.py",
                     "src/baselines/cm_crem_runtime.py", "src/baselines/cm_crem_export.py",
                     "src/baselines/cm_crem_selection.py", "src/__init__.py", "src/baselines/__init__.py"):
        target = tools_root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha(target) != file_sha(ROOT/relative):
                raise ValueError("Existing CM delivery tool identity differs")
        else:
            shutil.copyfile(ROOT/relative, target)
        copied[relative] = file_sha(target)
    atomic_json(tools_root/"source_manifest.json", {"files": copied, "purpose": "CM_RECORD_ONLY_IMPORT_AND_REPLOT"}, immutable=True)
    remote_base = "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/autodl/baselines/cm_crem_global_v1"
    remote_root = remote_base+"/"+Path(args.hpc_run_root).name
    intent = {"hpc_root": args.hpc_run_root, "autodl_root": remote_root,
              "package_manifest_sha256": file_sha(local_manifest),
              "package_sha256": expected["package_sha256"], "main_matrix_written": False}
    atomic_json(local/"autodl_transfer_intent.json", intent, immutable=True)
    mkdir_code = """from pathlib import Path
import json, os, sys
b=Path(sys.argv[1]); r=Path(sys.argv[2]); expected=json.loads(sys.argv[3])
assert b.is_dir() and b.resolve()==b and r.parent==b and not r.is_symlink()
marker=r/'transfer_intent.json'
if r.exists():
    assert marker.is_file() and json.loads(marker.read_text())==expected, 'Existing transfer root is not bound to this package'
else:
    r.mkdir()
    with marker.open('x') as f:
        json.dump(expected,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
for name in ('transfer','delivery_tools'):
    p=r/name; assert not p.is_symlink(); p.mkdir(exist_ok=True)
"""
    remote_python = "/root/miniconda3/envs/smiles_pip118/bin/python"
    ssh_read("autodl-a800", [remote_python, "-I", "-B", "-c", mkdir_code, remote_base, remote_root,
                            json.dumps(intent, sort_keys=True)])
    for path in (local_package, local_manifest):
        subprocess.run(["rsync", "-t", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                        str(path), "autodl-a800:"+remote_root+"/transfer/"+path.name], check=True, timeout=3600)
    subprocess.run(["rsync", "-rt", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=20",
                    str(tools_root)+"/", "autodl-a800:"+remote_root+"/delivery_tools/"], check=True, timeout=300)
    remote_receipt = remote_root+"/import/cm_import_receipt.json"
    exists_code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); print(p.read_text() if p.exists() else 'null')"
    prior = json.loads(ssh_read("autodl-a800", [remote_python, "-I", "-B", "-c", exists_code, remote_receipt]))
    if prior is None:
        imported_remote = json.loads(ssh_read("autodl-a800", [remote_python, "-I", "-B",
            remote_root+"/delivery_tools/scripts/import_cm_crem.py", "--package", remote_root+"/transfer/"+local_package.name,
            "--manifest", remote_root+"/transfer/"+local_manifest.name, "--destination", remote_root+"/import"], timeout=3600))
    else:
        imported_remote = prior
    for receipt in (local_receipt, imported_remote):
        require_import_identity(receipt, expected)
    result = {"status": "BACE_CM_RESULT_DELIVERED", "local_import": local_receipt,
              "autodl_import": imported_remote, "main_matrix_written": False,
              "source_files_preserved": True, "completed_at": utc_now()}
    atomic_json(local/"delivery_receipt.json", result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpc-alias", default="tongji-hpc", choices=["tongji-hpc"])
    parser.add_argument("--hpc-run-root", required=True)
    parser.add_argument("--hpc-execution-root", required=True)
    parser.add_argument("--hpc-spec", help="Fresh immutable successor spec within this run; original pilot spec stays unchanged")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--start-time-utc", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if any(not re.fullmatch(r"/[A-Za-z0-9_./-]+", p) for p in (args.hpc_run_root, args.hpc_execution_root)):
        raise ValueError("CM relay remote paths must be literal safe absolute paths")
    spec_path = args.hpc_spec or args.hpc_run_root+"/spec.json"
    if not Path(spec_path).is_absolute() or Path(spec_path).parent != Path(args.hpc_run_root):
        raise ValueError("Relay spec must be an absolute direct child of the exact run root")
    base = Path("/Volumes/DireRaven/counterfactual-hpc-offload/cm-crem-global-v1")
    local = args.local_root.resolve()
    if local == base or not local.is_relative_to(base) or not Path(args.hpc_run_root).is_relative_to("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1"):
        raise ValueError("Relay is restricted to the current CM package and task roots")
    if not os.path.ismount("/Volumes/DireRaven") or base.resolve() != base:
        raise ValueError("The authorized external drive must be mounted; do not write a substitute local directory")
    local.mkdir(parents=True, exist_ok=True)
    lock = (local/"relay.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t0 = datetime.fromisoformat(args.start_time_utc.replace("Z", "+00:00"))
    atomic_json(local/"relay_identity.json", {"pid": os.getpid(), "created_at": utc_now(),
                "hpc_run_root": args.hpc_run_root, "execution_root": args.hpc_execution_root,
                "spec": spec_path,
                "max_lifetime_hours": 168, "model_api_calls": False, "poll_seconds": 300}, immutable=True)
    failures, blocker_since = 0, None
    python = "/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python"
    while (datetime.now(timezone.utc)-t0).total_seconds() < 168*3600:
        snapshot = {"updated_at": utc_now(), "pid": os.getpid(), "hpc_run_root": args.hpc_run_root}
        try:
            # Read only this task's tiny submission/status documents, not package SHAs.
            code = "import json,pathlib; p=pathlib.Path(__import__('sys').argv[1]); print(json.dumps({f.stem:json.loads(f.read_text()) for f in (p/'submissions').glob('*.json') if not f.name.endswith('.intent.json')}))"
            receipts = json.loads(ssh_read(args.hpc_alias, [python, "-I", "-B", "-c", code, args.hpc_run_root]))
            active, completed = [], set()
            for stage, record in receipts.items():
                job = record.get("job_id")
                if not job:
                    raise ValueError(f"Uncertain submission {stage}; no automatic duplicate")
                accounting = ssh_read(args.hpc_alias, ["sacct", "-X", "-j", job, "--noheader", "--parsable2", "--format=JobID,State,ExitCode"])
                states = [row.split("|") for row in accounting.splitlines() if row.strip()]
                if not states:
                    active.append({"stage": stage, "job_id": job, "state": "ACCOUNTING_PENDING"})
                elif all(row[1] == "COMPLETED" and row[2] == "0:0" for row in states):
                    completed.add(stage)
                elif any(row[1].startswith(("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")) for row in states):
                    snapshot.update(status="BLOCKED_FAILED_STAGE", failed_stage=stage, accounting=accounting)
                    atomic_json(local/"state.json", snapshot)
                    return 2
                else:
                    active.append({"stage": stage, "job_id": job, "state": accounting.strip()})
            if active:
                snapshot.update(status="WAITING_SLURM", active=active)
            else:
                next_stage = next((s for s in STAGES if s not in completed), None)
                if next_stage is None:
                    snapshot.update(transfer_completed(args, local, python))
                    atomic_json(local/"state.json", snapshot)
                    return 0
                status_text = ssh_read(args.hpc_alias, [python, "-I", "-B", args.hpc_execution_root+"/scripts/run_cm_crem.py", "--spec", spec_path, "--run-root", args.hpc_run_root, "--action", "status"])
                status = json.loads(status_text)
                if next_stage == "pilot-generate" and status["assets"]["missing_assets"]:
                    snapshot.update(status="WAITING_OFFICIAL_DATABASE_OR_ENV", next_stage=next_stage,
                                    missing_assets=status["assets"]["missing_assets"])
                    blocker_since = blocker_since or time.monotonic()
                    if time.monotonic()-blocker_since > 6*3600:
                        snapshot["status"] = "BLOCKED_ASSET_OVER_6H"
                        atomic_json(local/"state.json", snapshot)
                        return 3
                else:
                    result = ssh_read(args.hpc_alias, [python, "-I", "-B", args.hpc_execution_root+"/scripts/submit_cm_crem_stage.py", "--spec", spec_path, "--run-root", args.hpc_run_root, "--stage", next_stage], timeout=120)
                    snapshot.update(status="STAGE_SUBMITTED", stage=next_stage, submission=json.loads(result))
                    blocker_since = None
            failures = 0
        except ValueError as exc:
            snapshot.update(status="BLOCKED_CONTRACT_OR_UNCERTAIN_SUBMISSION", error=str(exc))
            atomic_json(local/"state.json", snapshot)
            return 6
        except (OSError, RuntimeError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            failures += 1
            snapshot.update(status="WAITING_TRANSPORT", transport_failures=failures, error=str(exc))
            if failures >= 2:
                atomic_json(local/"state.json", snapshot)
                return 4
        atomic_json(local/"state.json", snapshot)
        if args.once:
            return 0
        time.sleep(300)
    atomic_json(local/"state.json", {"status": "PLANNING_HORIZON_EXPIRED", "updated_at": utc_now(), "pid": os.getpid()})
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
