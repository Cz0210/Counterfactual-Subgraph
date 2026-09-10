#!/usr/bin/env python3
"""One bounded Mac CM-CReM relay: fixed stage DAG, no Codex/API calls or GPU actions."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cm_crem_runtime import atomic_json, utc_now, read_json, file_sha

STAGES = ["pilot-oracle", "pilot-generate", "pilot-filter", "pilot-closeout", "attribution",
          "generate", "filter", "encode", "calibrate", "select", "test", "audit", "export", "package"]


def _utc_start(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Relay campaign start must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def _argv_value(argv: list[str], flag: str) -> str | None:
    values = []
    for index, value in enumerate(argv):
        if value == flag:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"Original launch intent has no value for {flag}")
            values.append(argv[index + 1])
        elif value.startswith(flag + "="):
            values.append(value.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"Original launch intent repeats {flag}")
    return values[0] if values else None


def claim_relay_identity(args, local: Path, spec_path: str, lock) -> datetime:
    """Claim the current identity under the existing campaign flock only.

    A terminal asset wait can resume; a live/reused PID or any other terminal
    cannot. Preserve exact old bytes before changing the derived current view.
    Submission receipts and the stage DAG are not changed by this operation.
    """
    if (local/"relay.lock").is_symlink():
        raise ValueError("Relay lock must not be replaced by a symlink")
    lock_stat, path_stat = os.fstat(lock.fileno()), (local/"relay.lock").stat()
    if (lock_stat.st_dev, lock_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
        raise ValueError("Relay lock FD is not the original campaign lock file")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t0 = _utc_start(args.start_time_utc)
    now = datetime.now(timezone.utc)
    identity_path = local/"relay_identity.json"
    identity = {"pid": os.getpid(), "created_at": utc_now(),
                "hpc_run_root": args.hpc_run_root, "execution_root": args.hpc_execution_root,
                "spec": spec_path, "start_time_utc": t0.isoformat().replace("+00:00", "Z"),
                "planning_deadline_utc": (t0+timedelta(hours=168)).isoformat().replace("+00:00", "Z"),
                "max_lifetime_hours": 168, "model_api_calls": False, "poll_seconds": 300}
    if not getattr(args, "resume_after_terminal", False):
        if identity_path.exists():
            raise ValueError("Existing relay identity; only explicit terminal-resume may replace it")
        atomic_json(identity_path, identity, immutable=True)
        return t0

    snapshots = {}
    source_names = ["relay_identity.json", "state.json", "launch_intent.json"]
    if (local/"launch_receipt.json").exists():
        source_names.append("launch_receipt.json")
    for name in source_names:
        source = local/name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Terminal resume requires the original regular {name}")
        snapshots[name] = source.read_bytes()
    old = json.loads(snapshots["relay_identity.json"])
    state = json.loads(snapshots["state.json"])
    launch = json.loads(snapshots["launch_intent.json"])
    if launch.get("max_hours", 168) != 168:
        raise ValueError("Original launch intent lifetime differs from168h")
    diagnostic_resume = (getattr(args, "diagnostic_attempt", None) is not None and
                         state.get("status") == "BLOCKED_FAILED_STAGE" and
                         state.get("failed_stage") == "audit")
    context_resume = (getattr(args, 'audit_successor_stage', None) == 'audit-context' and
                      state.get('status') == 'BLOCKED_DIAGNOSTIC_REVIEW_REQUIRED' and
                      state.get('diagnostic',{}).get('status') == 'DIAGNOSTIC_CAPTURE_COMPLETE_NOT_ACCEPTANCE')
    if state.get("status") != "BLOCKED_ASSET_OVER_6H" and not diagnostic_resume and not context_resume:
        raise ValueError("Only BLOCKED_ASSET_OVER_6H permits this explicit relay resume")
    if old.get("hpc_run_root") != args.hpc_run_root or state.get("hpc_run_root") != args.hpc_run_root:
        raise ValueError("Terminal identity/state belong to another HPC campaign")
    if old.get("max_lifetime_hours") != 168:
        raise ValueError("Original relay lifetime contract is not the fixed168h")
    pid = old.get("pid")
    if type(pid) is not int or pid <= 1 or state.get("pid") != pid:
        raise ValueError("Terminal relay PID identities are incomplete or inconsistent")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        raise ValueError("Old relay PID still exists or is inaccessible; no takeover") from error
    else:
        raise ValueError("Old relay PID still exists; no takeover or signal is permitted")

    argv = launch.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError("Original launch intent argv is not an explicit argument vector")
    if _argv_value(argv, "--hpc-run-root") != args.hpc_run_root:
        raise ValueError("Original launch intent belongs to another HPC campaign")
    old_local = _argv_value(argv, "--local-root")
    if old_local is None or Path(old_local).resolve() != local.resolve():
        raise ValueError("Original launch intent belongs to another local campaign")
    launch_t0 = _argv_value(argv, "--start-time-utc")
    start_evidence = {"source": "launch_intent.json:argv", "start_time_utc": launch_t0}
    if launch_t0 is None:
        # Legacy argv may lack T0; inspect only this bound spec's small campaign
        # field. A freshly supplied CLI timestamp alone is never enough.
        python = "/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python"
        code = "import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps({'spec':sys.argv[1], 'start_time_utc':s['campaign']['start_time_utc'], 'planning_horizon_hours':s['campaign']['planning_horizon_hours']}))"
        bound_spec = old.get("spec") or args.hpc_run_root+"/spec.json"
        if Path(bound_spec).parent != Path(args.hpc_run_root):
            raise ValueError("Original spec evidence is not a direct child of this campaign")
        start_evidence = json.loads(ssh_read(args.hpc_alias, [python, "-I", "-B", "-c", code, bound_spec]))
        if start_evidence.get("spec") != bound_spec or start_evidence.get("planning_horizon_hours") != 168:
            raise ValueError("Original spec start/lifetime evidence is not bound")
        launch_t0 = start_evidence.get("start_time_utc")
    if not isinstance(launch_t0, str) or _utc_start(launch_t0) != t0:
        raise ValueError("Resume cannot reset the original campaign start time")
    if old.get("start_time_utc") is not None and _utc_start(old["start_time_utc"]) != t0:
        raise ValueError("Original identity and launch/spec T0 disagree")
    if not 0 <= (now-t0).total_seconds() < 168*3600:
        raise ValueError("Original168h campaign horizon is expired or has a future start")

    attempts = local/"relay_attempts"
    if attempts.is_symlink() or (attempts.exists() and not attempts.is_dir()):
        raise ValueError("Relay attempt archive is not a local regular directory")
    archive = attempts/("terminal-"+uuid.uuid4().hex)
    archive.mkdir(parents=True, exist_ok=False)
    archived = {}
    for name, body in snapshots.items():
        target = archive/name
        with target.open("xb") as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        target.chmod(0o444)
        archived[name] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    atomic_json(archive/"preservation_receipt.json", {
        "schema": "cm_relay_terminal_preservation_v1", "files": archived,
        "old_pid": pid, "old_pid_absent": True, "preserved_at": utc_now(),
        "start_evidence": start_evidence, "start_time_utc": identity["start_time_utc"],
        "planning_deadline_utc": identity["planning_deadline_utc"],
        "submission_receipts_modified": False, "science_resubmitted": False}, immutable=True)
    for name, body in snapshots.items():
        if (local/name).read_bytes() != body:
            raise ValueError("Prior relay evidence changed during preservation; no current-view update")
    identity.update(resume_after_terminal=True, prior_terminal_archive=str(archive),
                    prior_pid=pid, start_evidence=start_evidence)
    atomic_json(archive/"resume_identity.json", identity, immutable=True)
    atomic_json(identity_path, identity)
    atomic_json(local/"state.json", {"status": "RESUMING_CONTEXT_AUDIT" if context_resume else "RESUMING_AUDIT_DIAGNOSTIC" if diagnostic_resume else "RESUMING_AFTER_ASSET_TERMINAL", "updated_at": utc_now(),
                "pid": os.getpid(), "hpc_run_root": args.hpc_run_root,
                "start_time_utc": identity["start_time_utc"], "prior_terminal_archive": str(archive)})
    return t0


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


def collect_diagnostic(args, local, t0, python):
    """Finite same-lock consumer; cannot infer acceptance or resubmit science."""
    attempt = Path(args.diagnostic_attempt)
    if attempt.parent != Path(args.hpc_run_root) or not attempt.name.startswith('audit-recovery-'):
        raise ValueError('Diagnostic must be an explicit direct child of this campaign')
    failures = 0
    while (datetime.now(timezone.utc)-t0).total_seconds() < 168*3600:
        state = {'updated_at':utc_now(), 'pid':os.getpid(), 'hpc_run_root':args.hpc_run_root,
                 'diagnostic_attempt':str(attempt), 'scientific_pass_claimed':False,
                 'planning_deadline_utc':(t0+timedelta(hours=168)).isoformat()}
        try:
            code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); print((p/'submission.json').read_text())"
            submission = json.loads(ssh_read(args.hpc_alias,[python,'-I','-B','-c',code,str(attempt)]))
            jobs = submission.get('job_ids',[])
            if submission.get('status') != 'SUBMITTED_NOT_SCIENCE_PASS' or len(jobs)!=1 or not jobs[0].isdigit():
                raise ValueError('Diagnostic submission uncertain; no resubmit')
            job=jobs[0]; state['job_id']=job
            accounting=ssh_read(args.hpc_alias,['sacct','-X','-j',job,'--noheader','--parsable2','--format=JobID,State,ExitCode'])
            rows=[r.split('|') for r in accounting.splitlines() if r.strip()]
            terminal=bool(rows) and all(r[1].split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL') for r in rows)
            state.update(status='WAITING_DIAGNOSTIC_SLURM',accounting=accounting)
            if terminal:
                evidence_code="import pathlib,sys; p=pathlib.Path(sys.argv[1])/'evidence/diagnostic.json'; print(p.read_text() if p.exists() else 'null')"
                evidence=json.loads(ssh_read(args.hpc_alias,[python,'-I','-B','-c',evidence_code,str(attempt)]))
                dest=local/'audit-recovery'/attempt.name
                dest.mkdir(parents=True,exist_ok=True)
                atomic_json(dest/'submission.json',submission,immutable=True)
                atomic_json(dest/'diagnostic.json',evidence,immutable=True)
                state.update(status='BLOCKED_DIAGNOSTIC_REVIEW_REQUIRED',diagnostic=evidence,
                             next_action='REVIEW_CAPTURE_THEN_NARROW_AUDITOR_OR_PRODUCER_REPAIR_NO_GENERATION',
                             diagnostic_local_root=str(dest))
                atomic_json(local/'state.json',state)
                return 2
            atomic_json(local/'state.json',state); failures=0
        except (OSError,RuntimeError,subprocess.TimeoutExpired) as exc:
            failures+=1;state.update(status='WAITING_TRANSPORT',error=str(exc),transport_failures=failures)
            atomic_json(local/'state.json',state)
            if failures>=2:return 4
        if args.once:return 0
        time.sleep(300)
    atomic_json(local/'state.json',{'status':'PLANNING_HORIZON_EXPIRED','pid':os.getpid(),'updated_at':utc_now()})
    return 5


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
    parser.add_argument("--resume-after-terminal", action="store_true",
                        help="Resume only an exited BLOCKED_ASSET_OVER_6H relay, retaining original T0 and168h horizon")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--diagnostic-attempt", help="Existing audit-recovery submission; same lock/T0, no pilot or inferred PASS")
    parser.add_argument('--audit-successor-stage', choices=['audit-context'], help='Adopt the explicitly submitted original-batch audit successor; preserve failed audit receipt')
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
    t0 = claim_relay_identity(args, local, spec_path, lock)
    failures, blocker_since = 0, None
    python = "/share/home/u20526/anaconda3/envs/smiles_pip118/bin/python"
    if args.diagnostic_attempt:
        return collect_diagnostic(args, local, t0, python)
    while (datetime.now(timezone.utc)-t0).total_seconds() < 168*3600:
        snapshot = {"updated_at": utc_now(), "pid": os.getpid(), "hpc_run_root": args.hpc_run_root}
        try:
            # Read only this task's tiny submission/status documents, not package SHAs.
            code = "import json,pathlib; p=pathlib.Path(__import__('sys').argv[1]); print(json.dumps({f.stem:json.loads(f.read_text()) for f in (p/'submissions').glob('*.json') if not f.name.endswith('.intent.json')}))"
            receipts = json.loads(ssh_read(args.hpc_alias, [python, "-I", "-B", "-c", code, args.hpc_run_root]))
            if args.audit_successor_stage:
                if receipts.get('audit',{}).get('job_id') != '2659067' or 'audit-context' not in receipts:
                    raise ValueError('Missing bound failed audit or actual context successor')
                receipts['audit'] = receipts.pop('audit-context')
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
