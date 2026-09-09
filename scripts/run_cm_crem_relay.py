#!/usr/bin/env python3
"""One bounded Mac CM-CReM relay: fixed stage DAG, no Codex/API calls or GPU actions."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines.cm_crem_runtime import atomic_json, utc_now

STAGES = ["pilot-oracle", "pilot-generate", "pilot-filter", "pilot-closeout", "attribution",
          "generate", "filter", "encode", "calibrate", "select", "test", "audit", "export"]


def ssh_read(alias: str, argv: list[str], timeout: int = 90) -> str:
    run = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", alias,
                          shlex.join(argv)], capture_output=True, text=True, timeout=timeout)
    if run.returncode:
        raise RuntimeError(f"SSH returncode={run.returncode}: {run.stderr[-2000:]}")
    return run.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpc-alias", default="tongji-hpc", choices=["tongji-hpc"])
    parser.add_argument("--hpc-run-root", required=True)
    parser.add_argument("--hpc-execution-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--start-time-utc", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    base = Path("/Volumes/DireRaven/counterfactual-hpc-offload/cm-crem-global-v1")
    local = args.local_root.resolve()
    if local == base or not local.is_relative_to(base) or not Path(args.hpc_run_root).is_relative_to("/share/home/u20526/czx/counterfactual-subgraph-hpc-runtime/baselines/cm_crem_global_v1"):
        raise ValueError("Relay is restricted to the current CM package and task roots")
    local.mkdir(parents=True, exist_ok=True)
    lock = (local/"relay.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t0 = datetime.fromisoformat(args.start_time_utc.replace("Z", "+00:00"))
    atomic_json(local/"relay_identity.json", {"pid": os.getpid(), "created_at": utc_now(),
                "hpc_run_root": args.hpc_run_root, "execution_root": args.hpc_execution_root,
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
                    snapshot.update(status="EXPORT_COMPLETE_PORTABLE_ACCEPTANCE_AND_TRANSFER_REQUIRED")
                    atomic_json(local/"state.json", snapshot)
                    return 0
                status_text = ssh_read(args.hpc_alias, [python, "-I", "-B", args.hpc_execution_root+"/scripts/run_cm_crem.py", "--spec", args.hpc_run_root+"/spec.json", "--run-root", args.hpc_run_root, "--action", "status"])
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
                    result = ssh_read(args.hpc_alias, [python, "-I", "-B", args.hpc_execution_root+"/scripts/submit_cm_crem_stage.py", "--spec", args.hpc_run_root+"/spec.json", "--run-root", args.hpc_run_root, "--stage", next_stage], timeout=120)
                    snapshot.update(status="STAGE_SUBMITTED", stage=next_stage, submission=json.loads(result))
                    blocker_since = None
            failures = 0
        except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
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
