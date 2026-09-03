#!/usr/bin/env python3
"""Persistently relay committed T14 checkpoints to the read-only auditor.

This process is deliberately dataset-specific.  It never opens T14 SQLite,
never writes below the T14 generation/checkpoint root, and has no process
signalling API.  A fresh one-shot audit is spawned only when a new committed
checkpoint at or after step 12,500 appears.  A convergence receipt is left for
the existing exact-PID owner; this relay does not perform the handover.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import (  # noqa: E402
    stable_json_sha256,
    write_json,
)
from src.eval.tastemolnet_t14_external_convergence import (  # noqa: E402
    ALLOWED_STEPS,
    T14ExternalConvergenceError,
    discover_committed_checkpoints,
)


RELAY_SCHEMA = "tastemolnet_t14_external_convergence_relay_v1"
STATE_SCHEMA = "tastemolnet_t14_external_convergence_relay_state_v1"
HEARTBEAT_SCHEMA = "tastemolnet_t14_external_convergence_relay_heartbeat_v1"
RECEIPT_SCHEMA = "tastemolnet_t14_external_convergence_relay_receipt_v1"
MINIMUM_AUDIT_STEP = 12_500
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class T14ExternalConvergenceRelayError(RuntimeError):
    """The relay identity or a one-shot audit result is invalid."""


class T14FullStateConsumerBusy(T14ExternalConvergenceRelayError):
    """Science or another auditor owns the dataset-wide full-state lock."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise T14ExternalConvergenceRelayError(
            f"{label} must be one absolute physical directory: {path}"
        )
    return path.resolve(strict=True)


def _relay_directory(path: Path, checkpoint_root: Path) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise T14ExternalConvergenceRelayError(
            f"relay root must be absolute and non-symlink: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    resolved = _physical_directory(path, "relay root")
    try:
        resolved.relative_to(checkpoint_root)
    except ValueError:
        return resolved
    raise T14ExternalConvergenceRelayError(
        "external relay root must not be inside the active T14 checkpoint root"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_hashed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(payload)
    value[field] = stable_json_sha256(value)
    return value


def _read_self_hashed(path: Path, *, field: str, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14ExternalConvergenceRelayError(f"invalid relay JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise T14ExternalConvergenceRelayError(f"relay schema changed: {path}")
    digest = value.get(field)
    unsigned = {key: item for key, item in value.items() if key != field}
    if digest != stable_json_sha256(unsigned):
        raise T14ExternalConvergenceRelayError(f"relay digest changed: {path}")
    return value


def _initial_state(checkpoint_root: Path, execution_commit: str) -> dict[str, Any]:
    return _self_hashed(
        {
            "schema_version": STATE_SCHEMA,
            "status": "WAITING_FOR_12500",
            "checkpoint_root": str(checkpoint_root),
            "execution_commit": execution_commit,
            "audited_through_step": 0,
            "audit_attempts": [],
            "converged": False,
            "stop_action_performed": False,
            "stop_action_pending_exact_pid_handover": False,
            "active_sqlite_opened": False,
            "checkpoint_sqlite_opened": False,
            "signal_sent": False,
            "updated_at": _utc_now(),
        },
        "state_sha256",
    )


def _load_or_initialize_state(
    relay_root: Path,
    *,
    checkpoint_root: Path,
    execution_commit: str,
) -> dict[str, Any]:
    path = relay_root / "relay_state.json"
    if path.exists():
        value = _read_self_hashed(path, field="state_sha256", schema=STATE_SCHEMA)
        if (
            value.get("checkpoint_root") != str(checkpoint_root)
            or value.get("execution_commit") != execution_commit
            or value.get("active_sqlite_opened") is not False
            or value.get("checkpoint_sqlite_opened") is not False
            or value.get("signal_sent") is not False
        ):
            raise T14ExternalConvergenceRelayError("relay resume identity changed")
        return value
    value = _initial_state(checkpoint_root, execution_commit)
    write_json(path, value)
    return value


def choose_relay_action(
    available_steps: Sequence[int],
    *,
    audited_through_step: int,
    converged: bool,
) -> tuple[str, int | None]:
    """Choose the next metadata-only relay action.

    The newest committed checkpoint is sufficient because every one-shot audit
    reopens the complete eligible checkpoint sequence.  If multiple checkpoints
    arrived while the relay was down, one fresh audit therefore covers all of
    them without replaying redundant tens-of-GiB loads.
    """

    if converged:
        return "CONVERGED_STOP_ACTION_PENDING_EXACT_PID_HANDOVER", None
    normalized = sorted({int(step) for step in available_steps})
    if any(step not in ALLOWED_STEPS for step in normalized):
        raise T14ExternalConvergenceRelayError("off-cadence checkpoint reached relay")
    eligible = [step for step in normalized if step >= MINIMUM_AUDIT_STEP]
    if not eligible:
        return "WAITING_FOR_12500", None
    newest = max(eligible)
    if newest <= int(audited_through_step):
        return "WAITING_FOR_NEXT_COMMITTED_CHECKPOINT", None
    return "AUDIT_NEW_COMMITTED_CHECKPOINT", newest


def _heartbeat(
    relay_root: Path,
    state: Mapping[str, Any],
    *,
    phase: str,
    sequence: int,
    available_steps: Sequence[int],
    audit_child_pid: int | None = None,
    detail: str | None = None,
) -> None:
    payload = _self_hashed(
        {
            "schema_version": HEARTBEAT_SCHEMA,
            "controller_pid": os.getpid(),
            "sequence": sequence,
            "phase": phase,
            "checkpoint_root": state["checkpoint_root"],
            "execution_commit": state["execution_commit"],
            "available_steps": list(available_steps),
            "audited_through_step": int(state["audited_through_step"]),
            "audit_child_pid": audit_child_pid,
            "converged": bool(state["converged"]),
            "stop_action_performed": False,
            "stop_action_pending_exact_pid_handover": bool(
                state["stop_action_pending_exact_pid_handover"]
            ),
            "active_sqlite_opened": False,
            "checkpoint_sqlite_opened": False,
            "signal_sent": False,
            "detail": detail,
            "written_at": _utc_now(),
        },
        "heartbeat_sha256",
    )
    write_json(relay_root / "heartbeat.json", payload)


def _validate_one_shot_audit(attempt_root: Path) -> dict[str, Any]:
    audit_path = attempt_root / "t14_external_convergence_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T14ExternalConvergenceRelayError("one-shot audit output is absent") from exc
    if not isinstance(audit, dict):
        raise T14ExternalConvergenceRelayError("one-shot audit is not a JSON object")
    digest = audit.get("audit_sha256")
    unsigned = {key: item for key, item in audit.items() if key != "audit_sha256"}
    if digest != stable_json_sha256(unsigned):
        raise T14ExternalConvergenceRelayError("one-shot audit digest changed")
    if (
        audit.get("active_t14_root_modified") is not False
        or audit.get("active_sqlite_opened") is not False
        or audit.get("checkpoint_sqlite_opened") is not False
        or audit.get("signal_sent") is not False
        or audit.get("status")
        not in {"WAITING_FOR_12500", "CONTINUE_T14", "CONVERGED_EARLY_STOP"}
    ):
        raise T14ExternalConvergenceRelayError("one-shot safety contract changed")
    return audit


def _run_one_shot(
    *,
    script: Path,
    checkpoint_root: Path,
    attempt_root: Path,
    execution_commit: str,
    relay_root: Path,
    state: Mapping[str, Any],
    available_steps: Sequence[int],
    sequence: int,
    heartbeat_seconds: float,
) -> dict[str, Any]:
    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    if attempt_root.exists():
        raise T14ExternalConvergenceRelayError("one-shot attempt root is not fresh")
    log_path = attempt_root.parent / f"{attempt_root.name}.log"
    command = [
        sys.executable,
        str(script),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output-root",
        str(attempt_root),
        "--execution-commit",
        execution_commit,
    ]
    full_state_lock_path = checkpoint_root / ".t14-full-state-consumer.lock"
    with full_state_lock_path.open("a+b") as full_state_lock, log_path.open(
        "xb"
    ) as log_handle:
        try:
            fcntl.flock(
                full_state_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            log_path.unlink(missing_ok=True)
            raise T14FullStateConsumerBusy(
                "T14 science owns the full-state lock; auditor remains metadata-only"
            ) from exc
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        while child.poll() is None:
            _heartbeat(
                relay_root,
                state,
                phase="AUDIT_RUNNING",
                sequence=sequence,
                available_steps=available_steps,
                audit_child_pid=child.pid,
            )
            try:
                child.wait(timeout=heartbeat_seconds)
            except subprocess.TimeoutExpired:
                pass
        return_code = child.wait()
    if return_code != 0:
        raise T14ExternalConvergenceRelayError(
            f"one-shot auditor failed with exit status {return_code}; see {log_path}"
        )
    return _validate_one_shot_audit(attempt_root)


def _commit_audit_result(
    relay_root: Path,
    state: Mapping[str, Any],
    *,
    attempt_root: Path,
    trigger_step: int,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    available = [int(item) for item in audit.get("available_steps", [])]
    audited_through = max(available, default=0)
    if audited_through < trigger_step:
        raise T14ExternalConvergenceRelayError(
            "one-shot audit did not observe its triggering committed checkpoint"
        )
    attempts = list(state.get("audit_attempts", []))
    attempts.append(
        {
            "trigger_step": trigger_step,
            "audited_through_step": audited_through,
            "attempt_root": str(attempt_root),
            "audit_path": str(attempt_root / "t14_external_convergence_audit.json"),
            "audit_sha256": audit["audit_sha256"],
            "status": audit["status"],
            "converged": audit.get("converged") is True,
            "completed_at": _utc_now(),
        }
    )
    converged = audit.get("converged") is True
    updated = {
        **{key: item for key, item in state.items() if key != "state_sha256"},
        "status": (
            "CONVERGED_STOP_ACTION_PENDING_EXACT_PID_HANDOVER"
            if converged
            else "WAITING_FOR_NEXT_COMMITTED_CHECKPOINT"
        ),
        "audited_through_step": audited_through,
        "audit_attempts": attempts,
        "converged": converged,
        "stop_action_performed": False,
        "stop_action_pending_exact_pid_handover": converged,
        "active_sqlite_opened": False,
        "checkpoint_sqlite_opened": False,
        "signal_sent": False,
        "updated_at": _utc_now(),
    }
    updated = _self_hashed(updated, "state_sha256")
    write_json(relay_root / "relay_state.json", updated)
    if converged:
        one_shot_receipt = attempt_root / "t14_convergence_early_stop_receipt.json"
        source = _read_self_hashed(
            one_shot_receipt,
            field="receipt_sha256",
            schema="tastemolnet_t14_external_convergence_receipt_v1",
        )
        if (
            source.get("status") != "PASS"
            or source.get("stop_action_performed") is not False
            or source.get("next_safe_checkpoint_exact_pid_handover_required") is not True
        ):
            raise T14ExternalConvergenceRelayError("one-shot convergence receipt changed")
        receipt = _self_hashed(
            {
                "schema_version": RECEIPT_SCHEMA,
                "status": "PASS",
                "checkpoint_root": state["checkpoint_root"],
                "execution_commit": state["execution_commit"],
                "audited_through_step": audited_through,
                "one_shot_attempt_root": str(attempt_root),
                "one_shot_audit_sha256": audit["audit_sha256"],
                "one_shot_receipt_path": str(one_shot_receipt),
                "one_shot_receipt_file_sha256": _sha256_file(one_shot_receipt),
                "stop_action_performed": False,
                "stop_action_pending_exact_pid_handover": True,
                "required_handover_checks": [
                    "pid",
                    "pid_start_ticks",
                    "command_hash",
                    "cwd",
                    "generation_root",
                    "checkpoint_root",
                    "next_safe_checkpoint",
                ],
                "active_sqlite_opened": False,
                "checkpoint_sqlite_opened": False,
                "signal_sent": False,
                "created_at": _utc_now(),
            },
            "receipt_sha256",
        )
        write_json(relay_root / "t14_convergence_relay_receipt.json", receipt)
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--relay-root", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument(
        "--one-shot-script",
        type=Path,
        default=Path(__file__).with_name("run_t14_external_convergence_auditor_v1.py"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="Testing/recovery bound; zero means persistent operation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _GIT_SHA.fullmatch(str(args.execution_commit)) is None:
        raise T14ExternalConvergenceRelayError(
            "execution commit must be one exact lowercase Git SHA"
        )
    if args.poll_seconds <= 0 or args.heartbeat_seconds <= 0 or args.max_polls < 0:
        raise T14ExternalConvergenceRelayError("relay timing values are invalid")
    checkpoint_root = _physical_directory(args.checkpoint_root, "checkpoint root")
    relay_root = _relay_directory(args.relay_root, checkpoint_root)
    raw_script = args.one_shot_script
    if not raw_script.is_absolute() or raw_script.is_symlink() or not raw_script.is_file():
        raise T14ExternalConvergenceRelayError("one-shot auditor is not a physical file")
    script = raw_script.resolve(strict=True)
    identity_path = relay_root / "relay_identity.json"
    identity = _self_hashed(
        {
            "schema_version": RELAY_SCHEMA,
            "checkpoint_root": str(checkpoint_root),
            "relay_root": str(relay_root),
            "execution_commit": args.execution_commit,
            "one_shot_script": str(script),
            "one_shot_script_sha256": _sha256_file(script),
            "poll_seconds": args.poll_seconds,
            "heartbeat_seconds": args.heartbeat_seconds,
            "signals_enabled": False,
            "sqlite_access_enabled": False,
        },
        "identity_sha256",
    )
    if identity_path.exists():
        observed = _read_self_hashed(
            identity_path, field="identity_sha256", schema=RELAY_SCHEMA
        )
        if observed != identity:
            raise T14ExternalConvergenceRelayError("relay identity changed on resume")
    else:
        write_json(identity_path, identity)

    # The relay-root lock supports maintenance resume.  The parent-directory
    # lock prevents two fresh controller UUIDs from loading the same enormous
    # checkpoint sequence concurrently.
    global_lock_path = relay_root.parent / "tastemolnet-t14-external-convergence-relay.lock"
    lock_path = relay_root / ".relay.lock"
    with global_lock_path.open("a+b") as global_lock_handle, lock_path.open(
        "a+b"
    ) as lock_handle:
        try:
            fcntl.flock(
                global_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise T14ExternalConvergenceRelayError(
                "another T14 convergence relay is already running"
            ) from exc
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise T14ExternalConvergenceRelayError(
                "another T14 convergence relay owns this root"
            ) from exc
        write_json(
            relay_root / "relay_pid.json",
            {
                "schema_version": "tastemolnet_t14_external_convergence_relay_pid_v1",
                "pid": os.getpid(),
                "started_at": _utc_now(),
            },
        )
        state = _load_or_initialize_state(
            relay_root,
            checkpoint_root=checkpoint_root,
            execution_commit=args.execution_commit,
        )
        sequence = 0
        polls = 0
        while True:
            sequence += 1
            polls += 1
            checkpoints = discover_committed_checkpoints(checkpoint_root)
            available_steps = [checkpoint.step for checkpoint in checkpoints]
            action, trigger_step = choose_relay_action(
                available_steps,
                audited_through_step=int(state["audited_through_step"]),
                converged=bool(state["converged"]),
            )
            _heartbeat(
                relay_root,
                state,
                phase=action,
                sequence=sequence,
                available_steps=available_steps,
            )
            if action == "AUDIT_NEW_COMMITTED_CHECKPOINT":
                assert trigger_step is not None
                attempt_root = (
                    relay_root
                    / "audits"
                    / f"step-{trigger_step:012d}"
                    / f"attempt-{uuid.uuid4()}"
                )
                try:
                    audit = _run_one_shot(
                        script=script,
                        checkpoint_root=checkpoint_root,
                        attempt_root=attempt_root,
                        execution_commit=args.execution_commit,
                        relay_root=relay_root,
                        state=state,
                        available_steps=available_steps,
                        sequence=sequence,
                        heartbeat_seconds=args.heartbeat_seconds,
                    )
                except T14FullStateConsumerBusy:
                    _heartbeat(
                        relay_root,
                        state,
                        phase="WAITING_FULL_STATE_CONSUMER_SERIALIZATION",
                        sequence=sequence,
                        available_steps=available_steps,
                    )
                    if args.max_polls and polls >= args.max_polls:
                        return 0
                    time.sleep(args.poll_seconds)
                    continue
                state = _commit_audit_result(
                    relay_root,
                    state,
                    attempt_root=attempt_root,
                    trigger_step=trigger_step,
                    audit=audit,
                )
                _heartbeat(
                    relay_root,
                    state,
                    phase=str(state["status"]),
                    sequence=sequence,
                    available_steps=audit.get("available_steps", available_steps),
                )
            if args.max_polls and polls >= args.max_polls:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (T14ExternalConvergenceRelayError, T14ExternalConvergenceError) as exc:
        print(f"T14_CONVERGENCE_RELAY_FAILED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
