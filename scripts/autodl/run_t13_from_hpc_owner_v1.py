#!/usr/bin/env python3
"""Own one AutoDL T13 successor after the exact HPC import is released."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.t8_hpc_t13_successor_v1 import (  # noqa: E402
    atomic_json,
    publish_verified_t13_locator,
    validate_spec_set,
    validate_t13_release,
)


HEARTBEAT_SCHEMA = "t13_from_hpc_owner_heartbeat_v1"


class T13FromHPCOwnerError(RuntimeError):
    """The sealed release, GPU ownership, or science subprocess is invalid."""


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def _heartbeat(
    path: Path,
    *,
    state: str,
    science_started: bool,
    science_pid: int = 0,
    detail: Any = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": HEARTBEAT_SCHEMA,
        "owner_pid": os.getpid(),
        "state": state,
        "science_started": science_started,
        "science_pid": science_pid,
        "detail": detail,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "hpc_matrix_write_enabled": False,
        "autodl_locator_only": True,
    }
    atomic_json(path, payload)
    return payload


@contextmanager
def _nonblocking_lease(path: Path) -> Iterator[int | None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            pass
        yield descriptor if acquired else None
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _gpu_observation(index: int, expected_uuid: str | None) -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    identities: dict[int, str] = {}
    for line in query.splitlines():
        left, right = (part.strip() for part in line.split(",", 1))
        identities[int(left)] = right
    if index not in identities:
        raise T13FromHPCOwnerError(f"configured GPU {index} is absent")
    observed_uuid = identities[index]
    if expected_uuid is not None and observed_uuid != expected_uuid:
        raise T13FromHPCOwnerError("configured GPU UUID changed")
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    processes: list[dict[str, Any]] = []
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", 2)]
            if len(fields) == 3 and fields[0] == observed_uuid:
                try:
                    pid = int(fields[1])
                except ValueError:
                    continue
                processes.append({"pid": pid, "process_name": fields[2]})
    return {"gpu_index": index, "gpu_uuid": observed_uuid, "processes": processes}


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    heartbeat: Path,
    state: str,
    poll_seconds: int,
) -> None:
    with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open(
        "ab", buffering=0
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
        )
        while process.poll() is None:
            _heartbeat(
                heartbeat,
                state=state,
                science_started=True,
                science_pid=process.pid,
                detail={"command": command, "cwd": str(cwd)},
            )
            time.sleep(poll_seconds)
        if process.returncode != 0:
            raise T13FromHPCOwnerError(
                f"{state} subprocess exited with code {process.returncode}"
            )


def run_once(
    *,
    spec_root: Path,
    release_path: Path,
    heartbeat: Path,
    owner_root: Path,
    poll_seconds: int,
) -> dict[str, Any]:
    specs = validate_spec_set(spec_root, check_files=True)
    if not release_path.is_file():
        return _heartbeat(
            heartbeat,
            state="WAITING_HPC_IMPORT_PASS",
            science_started=False,
            detail={"release_path": str(release_path)},
        )
    release = validate_t13_release(spec_root=spec_root, release_path=release_path)
    t13 = specs["t13"]
    output = Path(t13["output_root"])
    locator = Path(specs["publisher"]["terminal_root_locator"])
    if (output / "PASS").is_file():
        published = publish_verified_t13_locator(
            spec_root=spec_root, terminal_root=output
        )
        return _heartbeat(
            heartbeat,
            state="PASS_LOCATOR_READY",
            science_started=True,
            detail={"locator": str(locator), "terminal_root": published["terminal_root"]},
        )
    lease_path = Path(t13["gpu_lease_path"])
    with _nonblocking_lease(lease_path) as lease:
        if lease is None:
            return _heartbeat(
                heartbeat,
                state="READY_WAITING_T13_GPU_LEASE",
                science_started=False,
                detail={"gpu_lease_path": str(lease_path)},
            )
        gpu = _gpu_observation(int(t13["gpu_index"]), t13.get("gpu_uuid"))
        if gpu["processes"]:
            return _heartbeat(
                heartbeat,
                state="READY_WAITING_T13_GPU_IDLE",
                science_started=False,
                detail=gpu,
            )
        command = [str(value) for value in t13["command"]]
        checkpoint = output / "checkpoint.json"
        if output.exists():
            if output.is_symlink() or not output.is_dir() or not checkpoint.is_file():
                raise T13FromHPCOwnerError(
                    "existing T13 output cannot be safely resumed"
                )
            if "--resume" not in command:
                command.append("--resume")
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(t13["gpu_index"]),
                "PYTHONPATH": str(t13["repo_root"]),
                "RUN_LLM_ABLATION": "0",
                "RUN_GNN_ABLATION": "0",
            }
        )
        _run_process(
            command,
            cwd=Path(t13["repo_root"]),
            env=env,
            stdout_path=owner_root / "science.stdout.log",
            stderr_path=owner_root / "science.stderr.log",
            heartbeat=heartbeat,
            state="T13_AUTODL_SCIENCE_RUNNING",
            poll_seconds=poll_seconds,
        )
        verifier = [
            str(t13["python"]),
            str(Path(t13["repo_root"]) / "scripts/autodl/run_t13_from_hpc_import_v1.py"),
            "--config",
            str(Path(t13["repo_root"]) / "configs/hpc.yaml"),
            "--set",
            "inference.fallback_to_heuristic=false",
            "--output-dir",
            str(output),
            "--spec-root",
            str(spec_root),
            "--verify-only",
        ]
        _run_process(
            verifier,
            cwd=Path(t13["repo_root"]),
            env=env,
            stdout_path=owner_root / "verifier.stdout.log",
            stderr_path=owner_root / "verifier.stderr.log",
            heartbeat=heartbeat,
            state="T13_AUTODL_INDEPENDENT_VERIFY_RUNNING",
            poll_seconds=poll_seconds,
        )
    if not locator.is_file():
        raise T13FromHPCOwnerError("independent verifier produced no publisher locator")
    return _heartbeat(
        heartbeat,
        state="PASS_LOCATOR_READY",
        science_started=True,
        detail={
            "release_sha256": release["release_sha256"],
            "terminal_root": str(output),
            "locator": str(locator),
        },
    )


def run(
    *,
    spec_root: Path,
    release_path: Path,
    heartbeat: Path,
    owner_root: Path,
    poll_seconds: int,
    once: bool,
) -> dict[str, Any]:
    while True:
        result = run_once(
            spec_root=spec_root,
            release_path=release_path,
            heartbeat=heartbeat,
            owner_root=owner_root,
            poll_seconds=poll_seconds,
        )
        if once or result["state"] == "PASS_LOCATOR_READY":
            return result
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--spec-root", type=_absolute, required=True)
    parser.add_argument("--release", type=_absolute, required=True)
    parser.add_argument("--heartbeat", type=_absolute, required=True)
    parser.add_argument("--owner-root", type=_absolute, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml when supplied")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("unsupported --set override")
    if not 5 <= args.poll_seconds <= 3600:
        raise SystemExit("--poll-seconds must be in [5,3600]")
    result = run(
        spec_root=args.spec_root,
        release_path=args.release,
        heartbeat=args.heartbeat,
        owner_root=args.owner_root,
        poll_seconds=args.poll_seconds,
        once=args.once,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
