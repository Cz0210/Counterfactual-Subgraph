#!/usr/bin/env python3
"""Run release-v3 authority using compatibility Taste main-v2 paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_tastemolnet_main_v2 import (  # noqa: E402
    HEARTBEAT_INTERVAL_SECONDS,
    LAUNCHER_READY_NAME,
    TasteMainV2AuthorityError,
    capture_policy_facts,
    create_launcher_receipt,
    create_controller_receipt,
    create_gpu_lease_request,
    ensure_controller_namespace_parents,
    hold_taste_main_v2_controller_authority,
    immutable_authority_sha256,
    initial_heartbeat_path,
    inspect_clean_git,
    publish_launcher_ready,
    read_launcher_policy_facts,
    run_controller_loop,
)
from src.utils.process_identity_v2 import (  # noqa: E402
    ProcessSnapshotV2,
    ProcessIdentityV2Error,
    capture_process_snapshot,
    require_auto_termination_disabled,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _uuid4(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a UUID") from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise argparse.ArgumentTypeError("value must be canonical UUIDv4")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="paired-launcher parity")
    subparsers = parser.add_subparsers(dest="action", required=True)

    launch = subparsers.add_parser(
        "launch", help="externally supervise and attest a controller child"
    )
    launch.add_argument("--control-root", type=_absolute, required=True)
    launch.add_argument("--controller-root", type=_absolute, required=True)
    launch.add_argument("--launcher-root", type=_absolute, required=True)
    launch.add_argument("--controller-id", required=True)
    launch.add_argument("--controller-uuid", type=_uuid4, required=True)
    launch.add_argument("--project-root", type=_absolute, default=PROJECT_ROOT)
    launch.add_argument("--persistent-storage-root", type=_absolute, required=True)
    launch.add_argument("--expected-git-commit", required=True)
    launch.add_argument("--expected-git-tree", required=True)
    launch.add_argument("--controller-log", type=_absolute, required=True)
    launch.add_argument("--readiness-timeout-seconds", type=int, default=60)
    launch.add_argument("--heartbeat-count", type=int, default=0)

    run = subparsers.add_parser("run", help="run the controller in foreground")
    run.add_argument("--controller-root", type=_absolute, required=True)
    run.add_argument("--controller-id", required=True)
    run.add_argument("--controller-uuid", type=_uuid4, required=True)
    run.add_argument("--project-root", type=_absolute, default=PROJECT_ROOT)
    run.add_argument("--persistent-storage-root", type=_absolute, required=True)
    run.add_argument("--expected-git-commit", required=True)
    run.add_argument("--expected-git-tree", required=True)
    run.add_argument("--launcher-receipt", type=_absolute, required=True)
    run.add_argument("--launcher-handshake-fd", type=int, required=True)
    run.add_argument("--launcher-registration-fd", type=int, required=True)
    run.add_argument("--heartbeat-count", type=int, default=0)
    run.add_argument(
        "--heartbeat-interval-seconds",
        type=int,
        default=HEARTBEAT_INTERVAL_SECONDS,
    )

    lease = subparsers.add_parser(
        "request-lease", help="write an immutable lease request for acknowledgement"
    )
    lease.add_argument("--controller-receipt", type=_absolute, required=True)
    lease.add_argument("--task-id", required=True)
    lease.add_argument("--physical-gpu-index", type=int, required=True)
    lease.add_argument("--physical-gpu-uuid", required=True)
    lease.add_argument("--lease-uuid", type=_uuid4)
    lease.add_argument("--lifetime-seconds", type=int, default=21600)
    return parser.parse_args(argv)


def _read_launcher_handshake(descriptor: int) -> dict[str, str]:
    if descriptor < 3:
        raise TasteMainV2AuthorityError("launcher handshake FD is invalid")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            block = os.read(descriptor, 4096)
            if not block:
                break
            total += len(block)
            if total > 16 * 1024:
                raise TasteMainV2AuthorityError("launcher handshake is too large")
            chunks.append(block)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteMainV2AuthorityError("launcher handshake is malformed") from exc
    if type(payload) is not dict or set(payload) != {
        "launcher_receipt_path",
        "launcher_receipt_sha256",
    }:
        raise TasteMainV2AuthorityError("launcher handshake fields changed")
    return {str(name): str(value) for name, value in payload.items()}


def _write_controller_registration(descriptor: int) -> ProcessSnapshotV2:
    if descriptor < 3:
        raise TasteMainV2AuthorityError("launcher registration FD is invalid")
    snapshot = capture_process_snapshot(os.getpid())
    data = json.dumps(snapshot.to_dict(), sort_keys=True).encode("utf-8")
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise TasteMainV2AuthorityError("launcher registration short write")
            offset += written
    finally:
        os.close(descriptor)
    return snapshot


def _read_controller_registration(descriptor: int) -> ProcessSnapshotV2:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            block = os.read(descriptor, 4096)
            if not block:
                break
            total += len(block)
            if total > 16 * 1024:
                raise TasteMainV2AuthorityError("launcher registration is too large")
            chunks.append(block)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteMainV2AuthorityError("launcher registration is malformed") from exc
    if type(payload) is not dict:
        raise TasteMainV2AuthorityError("launcher registration is not an object")
    return ProcessSnapshotV2.from_mapping(payload)


def _launch(args: argparse.Namespace, *, policy_facts: dict[str, object]) -> int:
    if args.persistent_storage_root != Path(
        str(policy_facts["persistent_storage_root"])
    ):
        raise TasteMainV2AuthorityError(
            "launcher persistent root differs from canonical policy"
        )
    if args.control_root != Path(str(policy_facts["persistent_control_root"])):
        raise TasteMainV2AuthorityError(
            "launcher control root differs from canonical policy"
        )
    controllers, launches = ensure_controller_namespace_parents(args.control_root)
    if args.controller_root.parent != controllers:
        raise TasteMainV2AuthorityError("controller root is outside canonical namespace")
    if args.launcher_root.parent != launches:
        raise TasteMainV2AuthorityError("launcher root is outside canonical namespace")
    if args.readiness_timeout_seconds != 60:
        raise TasteMainV2AuthorityError("production readiness timeout must be 60 seconds")
    commit, tree = inspect_clean_git(args.project_root)
    if commit != args.expected_git_commit or tree != args.expected_git_tree:
        raise TasteMainV2AuthorityError("launcher Git identity differs from authority")
    if args.controller_log.parent.resolve(strict=True) != args.controller_log.parent:
        raise TasteMainV2AuthorityError("controller log parent must be physical")
    read_fd, write_fd = os.pipe()
    registration_read_fd, registration_write_fd = os.pipe()
    command = [
        sys.executable,
        "-I",
        "-B",
        str(Path(__file__).resolve(strict=True)),
        "--config",
        str(Path(args.config).resolve(strict=True)),
        "run",
        "--controller-root",
        str(args.controller_root),
        "--controller-id",
        args.controller_id,
        "--controller-uuid",
        args.controller_uuid,
        "--project-root",
        str(args.project_root),
        "--persistent-storage-root",
        str(args.persistent_storage_root),
        "--expected-git-commit",
        commit,
        "--expected-git-tree",
        tree,
        "--launcher-receipt",
        str(args.launcher_root / "launcher_receipt.json"),
        "--launcher-handshake-fd",
        str(read_fd),
        "--launcher-registration-fd",
        str(registration_write_fd),
        "--heartbeat-count",
        str(args.heartbeat_count),
    ]
    log_descriptor = os.open(
        args.controller_log,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=args.project_root,
            env={**os.environ, "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0"},
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=log_descriptor,
            pass_fds=(read_fd, registration_write_fd),
        )
    finally:
        os.close(log_descriptor)
        os.close(read_fd)
        os.close(registration_write_fd)
    if process is None:
        os.close(write_fd)
        raise TasteMainV2AuthorityError("controller child was not spawned")
    registered_controller = _read_controller_registration(registration_read_fd)
    if registered_controller.pid != process.pid or registered_controller.ppid != os.getpid():
        os.close(write_fd)
        raise TasteMainV2AuthorityError("registered controller lineage changed")
    if sys.platform.startswith("linux"):
        observed_controller = capture_process_snapshot(process.pid)
        if not registered_controller.same_runtime_identity(observed_controller):
            os.close(write_fd)
            raise TasteMainV2AuthorityError("controller registration differs from /proc")
    launcher = create_launcher_receipt(
        launcher_root=args.launcher_root,
        controller_id=args.controller_id,
        controller_uuid=args.controller_uuid,
        controller_snapshot=registered_controller,
        project_root=args.project_root,
        git_identity=(commit, tree),
        policy_facts=policy_facts,
    )
    handshake = json.dumps(
        {
            "launcher_receipt_path": str(launcher.receipt_path),
            "launcher_receipt_sha256": launcher.receipt_sha256,
        },
        sort_keys=True,
    ).encode("utf-8")
    try:
        os.write(write_fd, handshake)
    finally:
        os.close(write_fd)
    deadline = time.monotonic() + args.readiness_timeout_seconds
    ready_path = launcher.launcher_root / LAUNCHER_READY_NAME
    last_readiness_error = "controller has not published sequence-1 heartbeat"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return int(process.returncode or 75)
        receipt_path = args.controller_root / "controller_receipt.json"
        try:
            anchor_path = initial_heartbeat_path(args.controller_root)
            receipt_sha = immutable_authority_sha256(
                receipt_path, label="controller receipt"
            )
            anchor_sha = immutable_authority_sha256(
                anchor_path, label="controller heartbeat"
            )
            with hold_taste_main_v2_controller_authority(
                receipt_path,
                anchor_path,
                args.controller_id,
                commit,
                tree,
                35,
                expected_launcher_receipt_path=launcher.receipt_path,
                expected_launcher_receipt_sha256=launcher.receipt_sha256,
                expected_receipt_sha256=receipt_sha,
                expected_heartbeat_sha256=anchor_sha,
            ) as authority:
                evidence = authority.revalidate()
                publish_launcher_ready(
                    launcher_receipt_path=launcher.receipt_path,
                    controller_receipt_path=receipt_path,
                    controller_anchor_heartbeat_path=anchor_path,
                    authority_evidence=evidence,
                )
                print(
                    json.dumps(
                        {
                            "state": "RUNNING",
                            "ready_path": str(ready_path),
                            **evidence,
                            "science_released": False,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            break
        except (
            OSError,
            ValueError,
            ProcessIdentityV2Error,
            TasteMainV2AuthorityError,
        ) as exc:
            last_readiness_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25)
    else:
        print(
            json.dumps(
                {
                    "state": "QUARANTINED",
                    "reason": (
                        "controller readiness was not proven within 60 seconds; "
                        f"last_error={last_readiness_error}"
                    ),
                    "controller_pid": process.pid,
                    "science_released": False,
                    "manual_review_required": True,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return int(process.wait())


def _run(args: argparse.Namespace) -> int:
    config = Path(args.config)
    expected_config = PROJECT_ROOT / "configs/hpc.yaml"
    if config.resolve(strict=True) != expected_config:
        raise TasteMainV2AuthorityError(
            "--config must be this checkout's configs/hpc.yaml"
        )
    require_auto_termination_disabled()
    if args.action == "request-lease":
        fixed = {"T4_ORACLE_SMOKE": 1, "TASTE_GCF_NEUROSED": 2}
        if fixed.get(args.task_id) != args.physical_gpu_index:
            raise TasteMainV2AuthorityError(
                "request-lease accepts only fixed T4/GPU1 or NeuroSED/GPU2"
            )
        lease = create_gpu_lease_request(
            controller_receipt_path=args.controller_receipt,
            task_id=args.task_id,
            physical_gpu_index=args.physical_gpu_index,
            physical_gpu_uuid=args.physical_gpu_uuid,
            lease_uuid=args.lease_uuid,
            lifetime_seconds=args.lifetime_seconds,
        )
        print(
            json.dumps(
                {
                    "state": "WAITING_CONTROLLER_ACKNOWLEDGEMENT",
                    "lease_path": str(lease.path),
                    "lease_sha256": lease.sha256,
                    "lease_uuid": lease.lease_uuid,
                    "task_id": lease.payload["task_id"],
                    "physical_gpu_index": lease.payload["physical_gpu_index"],
                    "physical_gpu_uuid": lease.payload["physical_gpu_uuid"],
                    "science_released": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.action == "launch":
        policy_facts = capture_policy_facts(
            persistent_storage_root=args.persistent_storage_root
        )
        return _launch(args, policy_facts=policy_facts)
    controller_snapshot = _write_controller_registration(
        args.launcher_registration_fd
    )
    handshake = _read_launcher_handshake(args.launcher_handshake_fd)
    if Path(handshake["launcher_receipt_path"]) != args.launcher_receipt:
        raise TasteMainV2AuthorityError("launcher handshake path changed")
    if immutable_authority_sha256(
        args.launcher_receipt, label="external launcher receipt"
    ) != handshake["launcher_receipt_sha256"]:
        raise TasteMainV2AuthorityError("launcher handshake SHA changed")
    policy_facts = read_launcher_policy_facts(
        args.launcher_receipt,
        expected_sha256=handshake["launcher_receipt_sha256"],
    )
    if (
        policy_facts.get("persistent_storage_root")
        != str(args.persistent_storage_root)
    ):
        raise TasteMainV2AuthorityError("controller storage root differs from launcher")
    created = create_controller_receipt(
        controller_root=args.controller_root,
        project_root=args.project_root,
        controller_id=args.controller_id,
        controller_uuid=args.controller_uuid,
        expected_git_commit=args.expected_git_commit,
        expected_git_tree=args.expected_git_tree,
        launcher_receipt_path=args.launcher_receipt,
        expected_launcher_receipt_sha256=handshake[
            "launcher_receipt_sha256"
        ],
        process_snapshot=controller_snapshot,
        git_identity=(args.expected_git_commit, args.expected_git_tree),
        policy_facts=policy_facts,
    )
    print(
        json.dumps(
            {
                "state": "RUNNING",
                "controller_id": args.controller_id,
                "controller_uuid": args.controller_uuid,
                "controller_root": str(created.controller_root),
                "controller_receipt": str(created.receipt_path),
                "controller_receipt_sha256": created.receipt_sha256,
                "git_commit": created.payload["git_commit"],
                "git_tree": created.payload["git_tree"],
                "science_released": False,
                "auto_terminate_uncontrolled_children": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return run_controller_loop(
        receipt_path=created.receipt_path,
        heartbeat_count=args.heartbeat_count,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(parse_args(argv))
    except (OSError, ValueError, ProcessIdentityV2Error, TasteMainV2AuthorityError) as exc:
        print(
            json.dumps(
                {
                    "state": "QUARANTINED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "science_released": False,
                    "manual_review_required": True,
                    "auto_terminate_uncontrolled_children": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
