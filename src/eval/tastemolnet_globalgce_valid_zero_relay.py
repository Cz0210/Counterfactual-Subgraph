"""Read-only waiting relay for the sole Taste GlobalGCE recovery.

The relay observes only small JSON/procfs metadata.  It never opens the gSpan
SQLite database, never signals a process, and never starts training.  A
non-zero completed rule catalog is handed back to the normal T13 path.  The
valid-zero finalizer is invoked only after both branches are complete, the
source has no live writer, and the independent zero-source replay passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Any, Callable, Mapping

from src.baselines.tastemolnet_globalgce_full import read_json, stable_sha256
from src.eval.am_legacy_standardization import (
    LegacyStandardizationError,
    scan_live_writers,
)
from src.eval.tastemolnet_globalgce_valid_zero import (
    OBSERVATION_SCHEMA,
    TasteGlobalGCEValidZeroError,
    publish_valid_zero_result,
    validate_attempt_receipt,
    validate_authorization_receipt,
    validate_terminal_observation,
    validate_valid_zero_source,
)


HEARTBEAT_SCHEMA = "tastemolnet_globalgce_valid_zero_relay_heartbeat_v1"
TERMINAL_SCHEMA = "tastemolnet_globalgce_valid_zero_relay_terminal_v1"
GSPAN_SCHEMA = "globalgce_gspan_sqlite_chunks_v2"
class TasteGlobalGCEValidZeroRelayError(RuntimeError):
    """The waiting relay encountered ambiguous or corrupt terminal evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _physical_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TasteGlobalGCEValidZeroRelayError(f"missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise TasteGlobalGCEValidZeroRelayError(f"invalid {label}: {path}")
    return path


def _small_json(path: Path, *, label: str) -> dict[str, Any]:
    target = _physical_file(path, label=label)
    if target.stat().st_size <= 0 or target.stat().st_size > 16 * 1024 * 1024:
        raise TasteGlobalGCEValidZeroRelayError(f"invalid small JSON size: {target}")
    return read_json(target)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    payload = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _process_sample(
    proc_root: Path,
    *,
    pid: int,
    expected_start_ticks: int,
    previous: Mapping[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    root = proc_root / str(pid)
    try:
        raw = (root / "stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        start_ticks = int(fields[19])
        cpu_ticks = int(fields[11]) + int(fields[12])
        status = (root / "status").read_text(encoding="utf-8").splitlines()
        rss_line = next(line for line in status if line.startswith("VmRSS:"))
        rss_bytes = int(rss_line.split()[1]) * 1024
    except (OSError, StopIteration, ValueError, IndexError):
        return {
            "alive": False,
            "pid": pid,
            "start_ticks": expected_start_ticks,
            "rss_bytes": int((previous or {}).get("rss_bytes") or 0),
            "cpu_percent": float((previous or {}).get("cpu_percent") or 0.0),
            "cpu_ticks": int((previous or {}).get("cpu_ticks") or 0),
            "sampled_at_unix": now,
        }
    if start_ticks != expected_start_ticks:
        raise TasteGlobalGCEValidZeroRelayError("science PID start ticks changed")
    cpu_percent = 0.0
    if previous and previous.get("alive") is True:
        elapsed = now - float(previous.get("sampled_at_unix") or now)
        delta = cpu_ticks - int(previous.get("cpu_ticks") or cpu_ticks)
        if elapsed > 0.0 and delta >= 0:
            cpu_percent = delta / float(os.sysconf("SC_CLK_TCK")) / elapsed * 100.0
    return {
        "alive": True,
        "pid": pid,
        "start_ticks": start_ticks,
        "rss_bytes": rss_bytes,
        "cpu_percent": cpu_percent,
        "cpu_ticks": cpu_ticks,
        "sampled_at_unix": now,
    }


def read_gspan_progress(source_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in (0, 2):
        roots = sorted(
            (source_root / "raw" / f"target_{target}" / "globalgce_training_checkpoints" / "gspan").glob(
                "support_*/heartbeat.json"
            )
        )
        if len(roots) > 1:
            raise TasteGlobalGCEValidZeroRelayError(
                f"target-{target} has multiple gSpan heartbeat authorities"
            )
        if not roots:
            continue
        payload = _small_json(roots[0], label=f"target-{target} gSpan heartbeat")
        integers = {
            name: payload.get(name)
            for name in (
                "root_count",
                "completed_root_count",
                "frequent_subgraph_count",
            )
        }
        if (
            payload.get("schema_version") != GSPAN_SCHEMA
            or any(type(value) is not int or value < 0 for value in integers.values())
            or integers["completed_root_count"] > integers["root_count"]
            or isinstance(payload.get("heartbeat_epoch_seconds"), bool)
            or not isinstance(payload.get("heartbeat_epoch_seconds"), (int, float))
            or not math.isfinite(float(payload["heartbeat_epoch_seconds"]))
        ):
            raise TasteGlobalGCEValidZeroRelayError(
                f"target-{target} gSpan heartbeat is malformed"
            )
        rows.append(
            {
                "target_label": target,
                "path": str(roots[0]),
                **integers,
                "stage": payload.get("stage"),
                "current_root_index": payload.get("current_root_index"),
                "peak_rss_mib": payload.get("peak_rss_mib"),
                "heartbeat_epoch_seconds": float(payload["heartbeat_epoch_seconds"]),
                "sqlite_opened": False,
            }
        )
    return {
        "branches": rows,
        "root_total_count": sum(row["root_count"] for row in rows),
        "root_completed_count": sum(row["completed_root_count"] for row in rows),
        "patterns_seen": sum(row["frequent_subgraph_count"] for row in rows),
        "last_progress_unix": max(
            (row["heartbeat_epoch_seconds"] for row in rows), default=0.0
        ),
        "sqlite_opened": False,
    }


def completed_branch_rule_counts(source_root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    complete = True
    for target in (0, 2):
        path = source_root / "raw" / f"target_{target}" / "branch_manifest.json"
        if not path.is_file():
            complete = False
            continue
        payload = _small_json(path, label=f"target-{target} branch manifest")
        count = payload.get("valid_native_rule_count")
        if payload.get("status") != "PASS" or type(count) is not int or count < 0:
            raise TasteGlobalGCEValidZeroRelayError(
                f"target-{target} completed branch is malformed"
            )
        counts[str(target)] = count
    return {
        "both_branches_complete": complete and set(counts) == {"0", "2"},
        "valid_rule_counts": counts,
        "valid_rule_count": sum(counts.values()),
    }


@dataclass(slots=True)
class RelayRuntime:
    previous_process: dict[str, Any] | None = None
    previous_progress: dict[str, Any] | None = None
    last_live_process: dict[str, Any] | None = None


def observe_once(
    *,
    source_root: Path,
    proc_root: Path,
    science_pid: int,
    science_start_ticks: int,
    runtime: RelayRuntime,
    now: float | None = None,
) -> dict[str, Any]:
    sampled = time.time() if now is None else float(now)
    checkpoint = _small_json(source_root / "checkpoint.json", label="T13 checkpoint")
    process = _process_sample(
        proc_root,
        pid=science_pid,
        expected_start_ticks=science_start_ticks,
        previous=runtime.previous_process,
        now=sampled,
    )
    runtime.previous_process = process
    if process["alive"]:
        runtime.last_live_process = dict(process)
    progress = read_gspan_progress(source_root)
    previous = runtime.previous_progress
    patterns_delta = 0
    patterns_per_minute = 0.0
    if previous is not None:
        elapsed = sampled - float(previous.get("sampled_at_unix") or sampled)
        patterns_delta = max(
            0, int(progress["patterns_seen"]) - int(previous.get("patterns_seen") or 0)
        )
        if elapsed > 0.0:
            patterns_per_minute = patterns_delta / elapsed * 60.0
    runtime.previous_progress = {**progress, "sampled_at_unix": sampled}
    branches = completed_branch_rule_counts(source_root)
    phase = str(checkpoint.get("phase") or "")
    state = "WAITING_RECOVERY"
    if phase == "TARGET_2_COMPLETE" and branches["both_branches_complete"]:
        state = "NORMAL_PATH" if branches["valid_rule_count"] >= 1 else "ZERO_CANDIDATE"
    return {
        "schema_version": HEARTBEAT_SCHEMA,
        "state": state,
        "source_root": str(source_root),
        "checkpoint_phase": phase,
        "checkpoint_resume_identity_sha256": checkpoint.get(
            "resume_identity_sha256"
        ),
        "science_process": process,
        "last_live_process": runtime.last_live_process,
        "gspan_progress": progress,
        "patterns_delta": patterns_delta,
        "patterns_per_minute": patterns_per_minute,
        "branches": branches,
        "sqlite_opened": False,
        "signal_sent": False,
        "training_started": False,
        "observed_at": datetime.fromtimestamp(sampled, timezone.utc).isoformat(),
    }


def build_terminal_observation(
    heartbeat: Mapping[str, Any],
    *,
    attempt_id: str,
    source_root: Path,
    output_bytes: int,
) -> dict[str, Any]:
    if heartbeat.get("state") != "ZERO_CANDIDATE":
        raise TasteGlobalGCEValidZeroRelayError(
            "terminal observation requires a completed zero-candidate source"
        )
    progress = heartbeat.get("gspan_progress")
    process = heartbeat.get("last_live_process") or heartbeat.get("science_process")
    if not isinstance(progress, Mapping) or not isinstance(process, Mapping):
        raise TasteGlobalGCEValidZeroRelayError("terminal progress evidence is absent")
    root_total = progress.get("root_total_count")
    root_complete = progress.get("root_completed_count")
    branch_progress = progress.get("branches")
    if type(root_total) is not int or root_total <= 0 or root_complete != root_total:
        raise TasteGlobalGCEValidZeroRelayError("gSpan roots are not fully complete")
    if (
        not isinstance(branch_progress, list)
        or {row.get("target_label") for row in branch_progress} != {0, 2}
        or any(
            row.get("stage") != "complete"
            or row.get("completed_root_count") != row.get("root_count")
            for row in branch_progress
        )
    ):
        raise TasteGlobalGCEValidZeroRelayError(
            "both target gSpan authorities must be individually complete"
        )
    last_progress = float(progress.get("last_progress_unix") or 0.0)
    if not math.isfinite(last_progress) or last_progress <= 0.0:
        raise TasteGlobalGCEValidZeroRelayError("terminal progress timestamp is absent")
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source_root": str(source_root),
        "attempt_id": attempt_id,
        "training_complete": True,
        "branch0_complete": True,
        "branch2_complete": True,
        "no_engineering_failure": True,
        "active_database_opened": False,
        "root_completed_count": root_complete,
        "root_total_count": root_total,
        "patterns_seen": int(progress.get("patterns_seen") or 0),
        "patterns_delta": int(heartbeat.get("patterns_delta") or 0),
        "patterns_per_minute": float(heartbeat.get("patterns_per_minute") or 0.0),
        "output_bytes": int(output_bytes),
        "rss_bytes": int(process.get("rss_bytes") or 0),
        "cpu_percent": float(process.get("cpu_percent") or 0.0),
        "last_progress_time": datetime.fromtimestamp(
            last_progress, timezone.utc
        ).isoformat(),
        "sqlite_opened": False,
        "signal_sent": False,
        "training_started": False,
        "written_at": _utc_now(),
    }


def source_tree_bytes(source_root: Path) -> int:
    total = 0
    for current, directories, files in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        if any((current_path / name).is_symlink() for name in directories):
            raise TasteGlobalGCEValidZeroRelayError("source contains a directory symlink")
        for name in files:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise TasteGlobalGCEValidZeroRelayError("source contains a non-physical file")
            total += path.stat().st_size
    return total


def wait_and_finalize(
    *,
    source_root: Path,
    attempt_receipt_path: Path,
    authorization_receipt_path: Path,
    test_csv: Path,
    threshold_contract: Path,
    output_root: Path,
    control_root: Path,
    execution_commit: str,
    science_pid: int,
    science_start_ticks: int,
    poll_seconds: int,
    proc_root: Path = Path("/proc"),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if poll_seconds < 30 or poll_seconds > 60:
        raise TasteGlobalGCEValidZeroRelayError("relay poll interval must be 30--60s")
    attempt = validate_attempt_receipt(attempt_receipt_path, source_root=source_root)
    authorization = validate_authorization_receipt(
        authorization_receipt_path,
        source_root=source_root,
        attempt_receipt=attempt,
    )
    runtime = RelayRuntime()
    heartbeat_path = control_root / "heartbeat.json"
    terminal_path = control_root / "terminal.json"
    while True:
        heartbeat = observe_once(
            source_root=source_root,
            proc_root=proc_root,
            science_pid=science_pid,
            science_start_ticks=science_start_ticks,
            runtime=runtime,
        )
        heartbeat.update(
            {
                "attempt_id": attempt["attempt_id"],
                "relay_pid": os.getpid(),
                "poll_seconds": poll_seconds,
                "authorization_receipt": str(authorization_receipt_path),
                "valid_zero_output_root": str(output_root),
            }
        )
        _atomic_json(heartbeat_path, heartbeat)
        if heartbeat["state"] == "NORMAL_PATH":
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": "NORMAL_PATH",
                "source_root": str(source_root),
                "attempt_id": attempt["attempt_id"],
                "valid_rule_count": heartbeat["branches"]["valid_rule_count"],
                "valid_zero_fallback_used": False,
                "normal_t13_path_modified": False,
                "sqlite_opened": False,
                "signal_sent": False,
                "training_started": False,
                "written_at": _utc_now(),
            }
            _atomic_json(terminal_path, terminal)
            return terminal
        if heartbeat["state"] == "ZERO_CANDIDATE":
            # A process may close its current file between writes.  Requiring the
            # exact science identity to exit closes that race before the fd scan.
            if heartbeat["science_process"]["alive"] is True:
                heartbeat["state"] = "WAITING_WRITER_RELEASE"
                _atomic_json(heartbeat_path, heartbeat)
                sleep(float(poll_seconds))
                continue
            try:
                writers = scan_live_writers(source_root, proc_root=proc_root)
            except LegacyStandardizationError:
                heartbeat["state"] = "WAITING_WRITER_RELEASE"
                _atomic_json(heartbeat_path, heartbeat)
                sleep(float(poll_seconds))
                continue
            if writers.get("writable_fd_count") != 0:
                raise TasteGlobalGCEValidZeroRelayError("writer audit changed")
            try:
                source_audit = validate_valid_zero_source(
                    source_root, proc_root=proc_root
                )
            except (TasteGlobalGCEValidZeroError, LegacyStandardizationError) as exc:
                terminal = {
                    "schema_version": TERMINAL_SCHEMA,
                    "status": "BLOCKED_SCIENCE_CRITICAL",
                    "source_root": str(source_root),
                    "attempt_id": attempt["attempt_id"],
                    "reason": f"{type(exc).__name__}:{exc}",
                    "valid_zero_fallback_used": False,
                    "sqlite_opened": False,
                    "signal_sent": False,
                    "training_started": False,
                    "written_at": _utc_now(),
                }
                _atomic_json(terminal_path, terminal)
                return terminal
            observation = build_terminal_observation(
                heartbeat,
                attempt_id=attempt["attempt_id"],
                source_root=source_root,
                output_bytes=source_tree_bytes(source_root),
            )
            observation_path = control_root / "terminal_observation.json"
            _atomic_json(observation_path, observation)
            observation = validate_terminal_observation(
                observation_path,
                source_root=source_root,
                attempt_id=attempt["attempt_id"],
            )
            result = publish_valid_zero_result(
                source_audit=source_audit,
                attempt_receipt=attempt,
                authorization=authorization,
                observation=observation,
                test_csv=test_csv,
                threshold_contract=threshold_contract,
                output_root=output_root,
                execution_commit=execution_commit,
            )
            terminal = {
                "schema_version": TERMINAL_SCHEMA,
                "status": "PASS",
                "source_root": str(source_root),
                "attempt_id": attempt["attempt_id"],
                "valid_zero_fallback_used": True,
                "output_root": result["output_root"],
                "result_type": result["result_type"],
                "matrix_append_ready": result["matrix_append_ready"],
                "observation_sha256": stable_sha256(observation),
                "sqlite_opened": False,
                "signal_sent": False,
                "training_started": False,
                "written_at": _utc_now(),
            }
            _atomic_json(terminal_path, terminal)
            return terminal
        sleep(float(poll_seconds))


__all__ = [
    "GSPAN_SCHEMA",
    "HEARTBEAT_SCHEMA",
    "RelayRuntime",
    "TERMINAL_SCHEMA",
    "TasteGlobalGCEValidZeroRelayError",
    "build_terminal_observation",
    "completed_branch_rule_counts",
    "observe_once",
    "read_gspan_progress",
    "source_tree_bytes",
    "wait_and_finalize",
]
