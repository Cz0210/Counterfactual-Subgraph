#!/usr/bin/env python3
"""Fail-closed one-shot worker for the authorized Mut trace-on 50k adoption.

This is intentionally a dataset-specific continuation worker, not a second
controller.  It attaches to the live ``mut_fast_accurate_v2`` controller and
the existing fast16 matrix authority.  Before touching a GPU it measures real
protected-task throughput for five minutes.  Under one exclusive GPU lock it
then runs, sequentially:

1. the reviewed historical->checkpoint-instrumented 500-step gate; and
2. the checkpoint-instrumented trace-on->trace-off 500-step/reload gate.

Any uncertain identity, memory event, incomplete five-minute throughput
window, semantic mismatch, or reload mismatch stops the exact fresh process
group with SIGTERM only and prevents adoption.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.run_mut_checkpoint_instrumentation_equivalence import (  # noqa: E402
    INSTRUMENTATION_COMMIT,
    INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
    LEGACY_SOURCE_INVENTORY_SHA256,
    SOURCE_COMMIT,
)
from scripts.autodl.run_mut_fast_accurate_v2 import (  # noqa: E402
    _publish_matrix_queue,
    load_spec,
    publish_adoption,
    publish_inventory,
)
from scripts.autodl.run_mut_trace_mode_equivalence import (  # noqa: E402
    HISTORICAL_DISTANCE_SHA256,
    HISTORICAL_GNN_SHA256,
    HISTORICAL_RF_ORACLE_SHA256,
)
from src.baselines.comrecgc.contracts import (  # noqa: E402
    sha256_file,
    stable_json_sha256,
)
from src.utils.autodl_mut_trace_on_adoption_v1 import (  # noqa: E402
    AUDIT_SCHEMA as TRACE_AUDIT_SCHEMA,
    CANARY_HEADROOM_STOP_BYTES,
    CANARY_REQUIRED_HEADROOM_BYTES,
    CANARY_RSS_STOP_BYTES,
    EXPECTED_CANDIDATE_UNIVERSE_SHA256,
    ProtectedThroughputGate,
    atomic_json,
    audit_trace_semantics,
    establish_protected_throughput_baseline,
    load_protected_throughput_manifest,
    validate_authorization_receipt,
    verify_mut_candidate_pair_dbscan_binding,
    write_authorization_receipt,
)
from src.utils.autodl_mut_traceoff_parity_v1 import (  # noqa: E402
    SOURCE_PAYLOAD_SHA256,
    validate_instrumentation_equivalence_gate,
)
from src.utils.autodl_runtime import (  # noqa: E402
    GPUFileLock,
    GPULockError,
    gpu_lock_available,
    query_gpu_inventory,
)


SCHEMA = "mut_trace_on_adoption_worker_v1"
MEMORY_SCHEMA = "mut_trace_mode_canary_memory_v1"
VERIFICATION_SCHEMA = "mut_trace_on_50k_adoption_verification_v1"
GUARD_SCRIPT = "run_mut_checkpoint_instrumentation_equivalence.py"
GUARD_ACTION = "run-pair"
REQUIRED_TRACE_PHASES = (
    "trace_on_continuous",
    "trace_on_reload",
    "trace_off_continuous",
    "trace_off_reload",
)
SAMPLE_SECONDS = 10
PROTECTED_WINDOW_SECONDS = 300
HEADROOM_WAIT_SECONDS = 60
TRACE_BRANCH_ALLOWLIST = frozenset(
    {"OBSERVATIONAL_WRITE_ONLY", "CHECKPOINT_SERIALIZATION_ONLY"}
)
BINDING_SCHEMA = "mut_candidate_pair_dbscan_binding_v1"
BINDING_KIND = "transitive_generation_pair_store_vectors_dbscan_v1"


class MutTraceWorkerError(RuntimeError):
    """A one-shot adoption gate failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(value: str | Path, *, exists: bool = True, label: str = "path") -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise MutTraceWorkerError(f"{label} must be absolute and non-symlink: {path}")
    try:
        return path.resolve(strict=exists)
    except OSError as exc:
        raise MutTraceWorkerError(f"{label} is absent: {path}") from exc


def _physical_json(path: Path, *, label: str = "JSON") -> dict[str, Any]:
    resolved = _absolute(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        raw = b""
        while len(raw) < metadata.st_size:
            chunk = os.read(descriptor, metadata.st_size - len(raw))
            if not chunk:
                break
            raw += chunk
        if len(raw) != metadata.st_size or metadata.st_size <= 0:
            raise MutTraceWorkerError(f"{label} changed while read: {resolved}")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MutTraceWorkerError(f"{label} is invalid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise MutTraceWorkerError(f"{label} must be one object: {resolved}")
    return dict(value)


def _write_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _process_start_ticks(proc_root: Path, pid: int) -> int | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        return int(raw[closing + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _wait_process_start_ticks(
    proc_root: Path, process: subprocess.Popen[Any], *, timeout_seconds: float = 5.0
) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ticks = _process_start_ticks(proc_root, process.pid)
        if ticks is not None:
            return ticks
        if process.poll() is not None:
            return None
        time.sleep(0.05)
    return _process_start_ticks(proc_root, process.pid)


def _process_cmdline(proc_root: Path, pid: int) -> str:
    try:
        return (
            (proc_root / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", errors="replace")
            .strip()
        )
    except OSError as exc:
        raise MutTraceWorkerError(f"Cannot read process command: pid={pid}") from exc


def _verify_controller(
    *, spec: Mapping[str, Any], controller_pid: int, controller_start_ticks: int
) -> dict[str, Any]:
    proc_root = _absolute(spec["proc_root"], label="proc root")
    if _process_start_ticks(proc_root, controller_pid) != controller_start_ticks:
        raise MutTraceWorkerError("Successor controller PID/start-ticks changed")
    command = _process_cmdline(proc_root, controller_pid)
    if "run_mut_fast_accurate_v2.py" not in command or " run " not in f" {command} ":
        raise MutTraceWorkerError("Successor controller command is not the frozen runner")
    control = (
        _absolute(spec["control_root"], label="control root")
        / "mut_fast_accurate_v2"
        / str(spec["controller_id"])
    )
    heartbeat_path = _absolute(control / "heartbeat.json", label="controller heartbeat")
    heartbeat = _physical_json(heartbeat_path, label="controller heartbeat")
    if (
        heartbeat.get("controller_id") != spec["controller_id"]
        or int(heartbeat.get("pid", -1)) != controller_pid
    ):
        raise MutTraceWorkerError("Successor heartbeat identity does not match process")
    try:
        heartbeat_text = str(heartbeat["heartbeat_at"])
        heartbeat_time = datetime.fromisoformat(
            heartbeat_text[:-1] + "+00:00"
            if heartbeat_text.endswith("Z")
            else heartbeat_text
        )
        if heartbeat_time.tzinfo is None:
            raise ValueError("heartbeat has no timezone")
        heartbeat_age = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        raise MutTraceWorkerError("Successor heartbeat timestamp is invalid") from exc
    if heartbeat_age < -30 or heartbeat_age > max(300, 5 * int(spec["poll_seconds"])):
        raise MutTraceWorkerError(
            f"Successor heartbeat is stale or future-dated: age={heartbeat_age:.1f}s"
        )
    return {
        "controller_id": spec["controller_id"],
        "pid": controller_pid,
        "start_ticks": controller_start_ticks,
        "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_at": heartbeat.get("heartbeat_at"),
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_state": heartbeat.get("state"),
        "control_dir": str(control),
    }


def _validate_worker_guard(args: argparse.Namespace) -> None:
    if (
        args.successor_guard_script != GUARD_SCRIPT
        or args.successor_guard_action != GUARD_ACTION
    ):
        raise MutTraceWorkerError("Successor science-child guard tokens changed")
    command = _process_cmdline(Path("/proc"), os.getpid())
    if GUARD_SCRIPT not in command or GUARD_ACTION not in command:
        raise MutTraceWorkerError("Worker command line does not expose successor guard tokens")


def _validate_frozen_replay_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    replay = dict(spec["replay"])
    expected_numbers = {
        "parent_limit": 1448,
        "batch_size": 128,
        "steps": 500,
        "seed": 0,
    }
    changed = [
        key for key, expected in expected_numbers.items() if int(replay.get(key, -1)) != expected
    ]
    if changed:
        raise MutTraceWorkerError(f"Frozen Mut replay settings changed: {changed}")
    upstream = _absolute(replay["upstream_root"], label="frozen COMRECGC checkout")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
        tracked_status = subprocess.check_output(
            [
                "git",
                "-C",
                str(upstream),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MutTraceWorkerError("Cannot verify frozen COMRECGC checkout") from exc
    expected_upstream = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
    if commit != expected_upstream or tracked_status.strip():
        raise MutTraceWorkerError("Frozen COMRECGC commit is changed or tracked-dirty")
    identities = {
        "gnn": (
            _absolute(replay["gnn_checkpoint"], label="Mut GNN checkpoint"),
            HISTORICAL_GNN_SHA256,
        ),
        "distance": (
            _absolute(replay["distance_checkpoint"], label="Mut distance checkpoint"),
            HISTORICAL_DISTANCE_SHA256,
        ),
        "rf_oracle": (
            _absolute(spec["standardization"]["teacher_path"], label="Mut RF oracle"),
            HISTORICAL_RF_ORACLE_SHA256,
        ),
    }
    mismatches = [
        name for name, (path, expected) in identities.items() if sha256_file(path) != expected
    ]
    if mismatches:
        raise MutTraceWorkerError(f"Frozen classifier/distance identity changed: {mismatches}")
    return {
        "status": "PASS",
        "upstream_root": str(upstream),
        "upstream_commit": commit,
        "tracked_tree_clean": True,
        "parent_limit": 1448,
        "batch_size": 128,
        "steps": 500,
        "seed": 0,
        "identities": {
            name: {"path": str(path), "sha256": expected}
            for name, (path, expected) in identities.items()
        },
    }


@contextmanager
def _worker_lease(path: Path, *, controller_id: str) -> Iterator[TextIO]:
    target = _absolute(path, exists=False, label="worker lease")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise MutTraceWorkerError("Worker lease path is a symlink")
    handle = target.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MutTraceWorkerError("Another Mut trace-on one-shot worker is live") from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "schema_version": "mut_trace_on_worker_lease_v1",
                "controller_id": controller_id,
                "pid": os.getpid(),
                "start_ticks": _process_start_ticks(Path("/proc"), os.getpid()),
                "acquired_at": _utc_now(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _cgroup_snapshot(root: Path) -> dict[str, Any]:
    try:
        limit = int((root / "memory.limit_in_bytes").read_text(encoding="utf-8"))
        current = int((root / "memory.usage_in_bytes").read_text(encoding="utf-8"))
        failcnt = int((root / "memory.failcnt").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MutTraceWorkerError("Required cgroup-v1 memory counters are unavailable") from exc
    oom: dict[str, int] = {}
    for line in (root / "memory.oom_control").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in {"oom_kill", "under_oom"}:
            oom[fields[0]] = int(fields[1])
    return {
        "limit_bytes": limit,
        "current_bytes": current,
        "headroom_bytes": max(0, limit - current),
        "failcnt": failcnt,
        "oom_kill": int(oom.get("oom_kill", 0)),
        "under_oom": int(oom.get("under_oom", 0)),
        "sampled_at": _utc_now(),
    }


def _process_tree(proc_root: Path, root_pid: int) -> dict[str, Any]:
    parents: dict[int, int] = {}
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        try:
            raw = (child / "stat").read_text(encoding="utf-8")
            closing = raw.rfind(")")
            parents[int(child.name)] = int(raw[closing + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    rss_kib = pss_kib = cpu_ticks = 0
    live: list[int] = []
    for pid in sorted(selected):
        status = proc_root / str(pid) / "status"
        if not status.is_file():
            continue
        live.append(pid)
        try:
            raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
            closing = raw.rfind(")")
            stat_fields = raw[closing + 2 :].split()
            cpu_ticks += int(stat_fields[11]) + int(stat_fields[12])
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
                    break
            rollup = proc_root / str(pid) / "smaps_rollup"
            if rollup.is_file():
                for line in rollup.read_text(encoding="utf-8").splitlines():
                    if line.startswith("Pss:"):
                        pss_kib += int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            continue
    return {
        "root_pid": root_pid,
        "live_pids": live,
        "aggregate_rss_bytes": rss_kib * 1024,
        "aggregate_pss_bytes": pss_kib * 1024,
        "aggregate_cpu_ticks": cpu_ticks,
    }


def _process_group_members(proc_root: Path, pgid: int) -> list[int]:
    members: list[int] = []
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        try:
            raw = (child / "stat").read_text(encoding="utf-8")
            closing = raw.rfind(")")
            fields = raw[closing + 2 :].split()
            if int(fields[2]) == pgid:
                members.append(int(child.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members)


def _gpu_vram_bytes(gpu_uuid: str, pids: set[int]) -> int:
    matches = [row for row in query_gpu_inventory() if row.uuid == gpu_uuid]
    if len(matches) != 1:
        raise MutTraceWorkerError("Assigned GPU UUID disappeared or duplicated")
    foreign = sorted(
        int(process.pid)
        for process in matches[0].processes
        if int(process.pid) not in pids
    )
    if foreign:
        raise MutTraceWorkerError(
            f"Uncoordinated process appeared on exclusively locked GPU: {foreign}"
        )
    return sum(
        int(process.used_memory_mb) * 1024**2
        for process in matches[0].processes
        if int(process.pid) in pids
    )


def _safe_stop_group(
    process: subprocess.Popen[Any], *, start_ticks: int, proc_root: Path
) -> None:
    """SIGTERM only the verified fresh PGID; never escalate to SIGKILL."""

    if process.poll() is not None:
        return
    if _process_start_ticks(proc_root, process.pid) != start_ticks:
        raise MutTraceWorkerError("Fresh gate PID identity changed before SIGTERM")
    pgid = os.getpgid(process.pid)
    if pgid != process.pid:
        raise MutTraceWorkerError("Fresh gate is not its own isolated process group")
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise MutTraceWorkerError(
            "Exact canary PGID ignored SIGTERM; SIGKILL is forbidden"
        ) from exc


def _active_trace_phase(path: Path) -> str:
    if not path.is_file():
        return "trace_pair_startup"
    value = _physical_json(path, label="active trace arm")
    mode = str(value.get("trace_mode") or "")
    phase = str(value.get("phase") or "")
    if mode not in {"on", "off"} or phase not in {"continuous", "reload"}:
        raise MutTraceWorkerError("Active trace arm descriptor is invalid")
    return f"trace_{mode}_{phase}"


class CanaryMonitor:
    """Collect process/cgroup/GPU evidence and enforce every stop threshold."""

    def __init__(
        self,
        *,
        cgroup_root: Path,
        proc_root: Path,
        gpu_uuid: str,
        monitor_path: Path,
        throughput_gate: ProtectedThroughputGate,
        heartbeat: Any,
    ) -> None:
        self.cgroup_root = cgroup_root
        self.proc_root = proc_root
        self.gpu_uuid = gpu_uuid
        self.monitor_path = monitor_path
        self.throughput_gate = throughput_gate
        self.heartbeat = heartbeat
        self.initial = _cgroup_snapshot(cgroup_root)
        self.samples = 0
        self.phase_stats: dict[str, dict[str, Any]] = {}
        self.failures: list[str] = []
        self._last_cpu_ticks: int | None = None
        self._last_cpu_sample_monotonic: float | None = None

    def _sample(
        self,
        *,
        process: subprocess.Popen[Any] | None,
        phase: str,
        checkpoint_event: bool = False,
    ) -> None:
        snapshot = _cgroup_snapshot(self.cgroup_root)
        tree = (
            _process_tree(self.proc_root, process.pid)
            if process is not None and process.poll() is None
            else {
                "root_pid": None,
                "live_pids": [],
                "aggregate_rss_bytes": 0,
                "aggregate_pss_bytes": 0,
                "aggregate_cpu_ticks": 0,
            }
        )
        now_monotonic = time.monotonic()
        cpu_percent: float | None = None
        if (
            self._last_cpu_ticks is not None
            and self._last_cpu_sample_monotonic is not None
            and now_monotonic > self._last_cpu_sample_monotonic
        ):
            cpu_percent = (
                max(0, tree["aggregate_cpu_ticks"] - self._last_cpu_ticks)
                / max(1, os.sysconf("SC_CLK_TCK"))
                / (now_monotonic - self._last_cpu_sample_monotonic)
                * 100.0
            )
        self._last_cpu_ticks = int(tree["aggregate_cpu_ticks"])
        self._last_cpu_sample_monotonic = now_monotonic
        vram = _gpu_vram_bytes(self.gpu_uuid, set(tree["live_pids"]))
        protected = self.throughput_gate.sample()
        row = {
            "schema_version": "mut_trace_canary_memory_sample_v1",
            "sample": self.samples,
            "phase": phase,
            "sampled_at": _utc_now(),
            "process_tree": tree,
            "process_cpu_percent": cpu_percent,
            "process_gpu_vram_bytes": vram,
            "checkpoint_event": checkpoint_event,
            "parent_cgroup": snapshot,
            "protected_throughput": protected,
        }
        _write_jsonl(self.monitor_path, row)
        self.samples += 1
        stats = self.phase_stats.setdefault(
            phase,
            {
                "sample_count": 0,
                "peak_rss_bytes": 0,
                "peak_pss_bytes": 0,
                "peak_gpu_vram_bytes": 0,
                "minimum_parent_headroom_bytes": snapshot["headroom_bytes"],
                "peak_parent_current_bytes": 0,
                "checkpoint_event_count": 0,
                "checkpoint_event_peak_rss_bytes": 0,
            },
        )
        stats["sample_count"] += 1
        stats["peak_rss_bytes"] = max(
            stats["peak_rss_bytes"], tree["aggregate_rss_bytes"]
        )
        stats["peak_pss_bytes"] = max(
            stats["peak_pss_bytes"], tree["aggregate_pss_bytes"]
        )
        stats["peak_gpu_vram_bytes"] = max(stats["peak_gpu_vram_bytes"], vram)
        stats["minimum_parent_headroom_bytes"] = min(
            stats["minimum_parent_headroom_bytes"], snapshot["headroom_bytes"]
        )
        stats["peak_parent_current_bytes"] = max(
            stats["peak_parent_current_bytes"], snapshot["current_bytes"]
        )
        if checkpoint_event:
            stats["checkpoint_event_count"] += 1
            stats["checkpoint_event_peak_rss_bytes"] = max(
                stats["checkpoint_event_peak_rss_bytes"],
                tree["aggregate_rss_bytes"],
            )
        current_failures: list[str] = []
        if tree["aggregate_rss_bytes"] > CANARY_RSS_STOP_BYTES:
            current_failures.append(f"rss_gt_24gib:{phase}")
        if snapshot["headroom_bytes"] < CANARY_HEADROOM_STOP_BYTES:
            current_failures.append(f"parent_headroom_lt_32gib:{phase}")
        if snapshot["failcnt"] > self.initial["failcnt"]:
            current_failures.append(f"cgroup_failcnt_increased:{phase}")
        if snapshot["oom_kill"] > self.initial["oom_kill"]:
            current_failures.append(f"cgroup_oom_kill_increased:{phase}")
        if snapshot["under_oom"] > 0:
            current_failures.append(f"cgroup_under_oom:{phase}")
        current_failures.extend(str(item) for item in protected.get("failures", []))
        self.failures.extend(item for item in current_failures if item not in self.failures)
        self.heartbeat(
            "CANARY_RUNNING",
            phase=phase,
            sample_count=self.samples,
            process_pid=process.pid if process is not None else None,
            cgroup_headroom_bytes=snapshot["headroom_bytes"],
            process_rss_bytes=tree["aggregate_rss_bytes"],
            failures=list(self.failures),
        )

    def run_process(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        log_path: Path,
        static_phase: str | None = None,
        active_arm: Path | None = None,
        checkpoint_scan_root: Path | None = None,
    ) -> dict[str, Any]:
        if (static_phase is None) == (active_arm is None):
            raise MutTraceWorkerError("Exactly one phase source is required")
        admission = _cgroup_snapshot(self.cgroup_root)
        if admission["headroom_bytes"] < CANARY_REQUIRED_HEADROOM_BYTES:
            raise MutTraceWorkerError("64-GiB parent headroom admission was lost")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            start_ticks = _wait_process_start_ticks(self.proc_root, process)
            if start_ticks is None:
                returncode = process.wait(timeout=5)
                raise MutTraceWorkerError(
                    f"Fresh gate exited before process identity capture: {returncode}"
                )
            if os.getpgid(process.pid) != process.pid:
                _safe_stop_group(process, start_ticks=start_ticks, proc_root=self.proc_root)
                raise MutTraceWorkerError("Fresh gate lacks exact isolated PID/PGID")
            last_sample = 0.0
            previous_phase: str | None = None
            checkpoint_markers: set[str] = set()
            try:
                while process.poll() is None:
                    phase = (
                        str(static_phase)
                        if static_phase is not None
                        else _active_trace_phase(Path(active_arm))
                    )
                    now = time.monotonic()
                    current_markers = (
                        {
                            str(path.resolve())
                            for path in checkpoint_scan_root.glob(
                                "*checkpoint-mirror/step-*/_CHECKPOINT_MIRRORED.json"
                            )
                        }
                        if checkpoint_scan_root is not None
                        and checkpoint_scan_root.is_dir()
                        else set()
                    )
                    checkpoint_event = bool(current_markers - checkpoint_markers)
                    checkpoint_markers.update(current_markers)
                    if (
                        phase != previous_phase
                        or checkpoint_event
                        or now - last_sample >= SAMPLE_SECONDS
                    ):
                        # A new trace arm must independently satisfy the 64-GiB
                        # launch threshold, even though all arms share this lock.
                        if phase in REQUIRED_TRACE_PHASES and phase != previous_phase:
                            if _cgroup_snapshot(self.cgroup_root)["headroom_bytes"] < CANARY_REQUIRED_HEADROOM_BYTES:
                                self.failures.append(f"arm_admission_lt_64gib:{phase}")
                        self._sample(
                            process=process,
                            phase=phase,
                            checkpoint_event=checkpoint_event,
                        )
                        previous_phase = phase
                        last_sample = now
                    if self.failures:
                        _safe_stop_group(
                            process, start_ticks=start_ticks, proc_root=self.proc_root
                        )
                        raise MutTraceWorkerError(
                            "Canary watchdog stopped exact gate: " + ",".join(self.failures)
                        )
                    time.sleep(1)
            except BaseException:
                if process.poll() is None:
                    _safe_stop_group(
                        process, start_ticks=start_ticks, proc_root=self.proc_root
                    )
                raise
            returncode = int(process.returncode or 0)
        if returncode != 0:
            orphaned = _process_group_members(self.proc_root, process.pid)
            raise MutTraceWorkerError(
                "Sequential equivalence gate exited nonzero: "
                f"returncode={returncode}, remaining_exact_pgid_members={orphaned}; "
                "no post-root signal was sent"
            )
        orphaned = _process_group_members(self.proc_root, process.pid)
        if orphaned:
            raise MutTraceWorkerError(
                "Sequential gate root exited with remaining exact-PGID members; "
                f"no post-root signal was sent: {orphaned}"
            )
        return {
            "pid": process.pid,
            "start_ticks": start_ticks,
            "pgid": process.pid,
            "started_at_unix": started,
            "completed_at_unix": time.time(),
            "returncode": returncode,
            "command_sha256": hashlib.sha256(
                "\0".join(command).encode("utf-8")
            ).hexdigest(),
            "log_path": str(log_path),
            "log_sha256": sha256_file(log_path),
        }

    def finish_protected_window(self) -> dict[str, Any]:
        deadline = time.monotonic() + PROTECTED_WINDOW_SECONDS + 30
        while True:
            self._sample(process=None, phase="post_canary_protected_window")
            receipt = self.throughput_gate.receipt()
            if receipt.get("status") == "PASS":
                return receipt
            if self.failures:
                raise MutTraceWorkerError(
                    "Protected throughput gate failed: " + ",".join(self.failures)
                )
            if time.monotonic() >= deadline:
                raise MutTraceWorkerError(
                    "No complete five-minute protected-task window was observed"
                )
            time.sleep(SAMPLE_SECONDS)

    def receipt(
        self,
        *,
        protected_baseline: Mapping[str, Any],
        protected_gate: Mapping[str, Any],
        gate_runs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        final = _cgroup_snapshot(self.cgroup_root)
        missing = [
            phase
            for phase in REQUIRED_TRACE_PHASES
            if int((self.phase_stats.get(phase) or {}).get("sample_count", 0)) <= 0
        ]
        phase_failures: list[str] = []
        for phase in REQUIRED_TRACE_PHASES:
            stats = self.phase_stats.get(phase) or {}
            if int(stats.get("peak_rss_bytes", CANARY_RSS_STOP_BYTES + 1)) > CANARY_RSS_STOP_BYTES:
                phase_failures.append(f"rss_limit:{phase}")
            if int(stats.get("minimum_parent_headroom_bytes", 0)) < CANARY_HEADROOM_STOP_BYTES:
                phase_failures.append(f"headroom_stop:{phase}")
        failures = [*self.failures, *[f"missing_phase:{item}" for item in missing], *phase_failures]
        if self.initial["headroom_bytes"] < CANARY_REQUIRED_HEADROOM_BYTES:
            failures.append("initial_parent_headroom_lt_64gib")
        if self.initial["under_oom"] != 0:
            failures.append("initial_cgroup_under_oom")
        if final["failcnt"] != self.initial["failcnt"]:
            failures.append("failcnt_delta_nonzero")
        if final["oom_kill"] != self.initial["oom_kill"]:
            failures.append("oom_kill_delta_nonzero")
        if protected_baseline.get("status") != "PASS":
            failures.append("protected_baseline_not_pass")
        if protected_gate.get("status") != "PASS":
            failures.append("protected_five_minute_window_not_pass")
        required_phase_rows = {
            phase: {
                **dict(self.phase_stats.get(phase) or {}),
                "status": (
                    "PASS"
                    if phase not in missing
                    and not any(item.endswith(f":{phase}") for item in phase_failures)
                    else "FAIL"
                ),
            }
            for phase in REQUIRED_TRACE_PHASES
        }
        all_stats = list(self.phase_stats.values())
        payload = {
            "schema_version": MEMORY_SCHEMA,
            "status": "PASS" if not failures else "FAIL",
            "sample_interval_seconds": SAMPLE_SECONDS,
            "sample_count": self.samples,
            "arms_sequential": True,
            "max_concurrent_arms": 1,
            "required_trace_phases": list(REQUIRED_TRACE_PHASES),
            "phase_stats": self.phase_stats,
            "phases": required_phase_rows,
            "rss_stop_bytes": CANARY_RSS_STOP_BYTES,
            "parent_headroom_stop_bytes": CANARY_HEADROOM_STOP_BYTES,
            "parent_headroom_admission_bytes": CANARY_REQUIRED_HEADROOM_BYTES,
            "initial_parent_headroom_admission_pass": (
                self.initial["headroom_bytes"] >= CANARY_REQUIRED_HEADROOM_BYTES
            ),
            "initial_parent_headroom_bytes": self.initial["headroom_bytes"],
            "process_rss_peak_bytes": max(
                (int(row.get("peak_rss_bytes", 0)) for row in all_stats),
                default=0,
            ),
            "process_pss_peak_bytes": max(
                (int(row.get("peak_pss_bytes", 0)) for row in all_stats),
                default=0,
            ),
            "parent_headroom_min_bytes": min(
                (
                    int(row.get("minimum_parent_headroom_bytes", self.initial["headroom_bytes"]))
                    for row in all_stats
                ),
                default=self.initial["headroom_bytes"],
            ),
            "initial_cgroup": self.initial,
            "final_cgroup": final,
            "failcnt_delta": final["failcnt"] - self.initial["failcnt"],
            "oom_kill_delta": final["oom_kill"] - self.initial["oom_kill"],
            "cgroup_failcnt_delta": final["failcnt"] - self.initial["failcnt"],
            "cgroup_oom_delta": final["under_oom"] - self.initial["under_oom"],
            "cgroup_oom_kill_delta": final["oom_kill"] - self.initial["oom_kill"],
            "protected_baseline": dict(protected_baseline),
            "protected_gate": dict(protected_gate),
            "protected_throughput_gate": dict(protected_gate),
            "gate_runs": [dict(item) for item in gate_runs],
            "monitor_path": str(self.monitor_path),
            "monitor_sha256": sha256_file(self.monitor_path),
            "failures": sorted(set(failures)),
            "completed_at": _utc_now(),
        }
        payload["summary_sha256"] = stable_json_sha256(payload)
        return payload


def _eligible_gpu(spec: Mapping[str, Any]) -> Any | None:
    lock_root = _absolute(spec["runtime_root"], label="runtime root") / "locks"
    for gpu in sorted(query_gpu_inventory(), key=lambda row: row.index):
        if (
            len(gpu.processes) == 0
            and gpu.memory_free_mb >= int(spec["gpu_min_free_memory_mb"])
            and gpu.utilization_gpu_percent <= int(spec["gpu_max_utilization_percent"])
            and gpu_lock_available(lock_root, gpu.uuid)
        ):
            return gpu
    return None


@contextmanager
def _fresh_idle_gpu_lock(
    spec: Mapping[str, Any], *, cgroup: Path, heartbeat: Any
) -> Iterator[Any | None]:
    """Select and lock an idle physical GPU only after the baseline finishes.

    The availability probe and advisory-lock acquisition cannot be atomic with
    ``nvidia-smi``.  Reopen the physical UUID under the acquired lock and
    discard a raced observation rather than launching from a five-minute-old
    selection.
    """

    lock_root = _absolute(spec["runtime_root"], label="runtime root") / "locks"
    poll_seconds = int(spec["poll_seconds"])
    while True:
        headroom = _cgroup_snapshot(cgroup)
        if headroom["headroom_bytes"] < CANARY_REQUIRED_HEADROOM_BYTES:
            heartbeat(
                "WAITING_FOR_64G_PARENT_HEADROOM",
                wait_reason="headroom_dropped_while_waiting_for_gpu",
                parent_headroom_bytes=headroom["headroom_bytes"],
                required_parent_headroom_bytes=CANARY_REQUIRED_HEADROOM_BYTES,
                gpu_lock_held=False,
            )
            yield None
            return
        candidate = _eligible_gpu(spec)
        if candidate is None:
            heartbeat("WAITING_FOR_NATURALLY_IDLE_GPU_AFTER_BASELINE")
            time.sleep(poll_seconds)
            continue
        lock = GPUFileLock(
            lock_root,
            gpu_index=int(candidate.index),
            gpu_uuid=str(candidate.uuid),
            owner={
                "controller_id": spec["controller_id"],
                "worker_pid": os.getpid(),
                "worker_start_ticks": _process_start_ticks(Path("/proc"), os.getpid()),
                "purpose": "mut_trace_on_adoption_sequential_500_step_gates",
            },
        )
        try:
            lock.acquire()
        except GPULockError:
            heartbeat(
                "GPU_LOCK_RACED_AFTER_BASELINE",
                raced_gpu_index=int(candidate.index),
                raced_gpu_uuid=str(candidate.uuid),
            )
            time.sleep(poll_seconds)
            continue
        try:
            refreshed = [
                row for row in query_gpu_inventory() if row.uuid == candidate.uuid
            ]
            current = refreshed[0] if len(refreshed) == 1 else None
            current_is_exact_idle_gpu = bool(
                current is not None
                and int(current.index) == int(candidate.index)
                and str(current.uuid) == str(candidate.uuid)
                and len(current.processes) == 0
                and int(current.memory_free_mb)
                >= int(spec["gpu_min_free_memory_mb"])
                and int(current.utilization_gpu_percent)
                <= int(spec["gpu_max_utilization_percent"])
            )
            if not current_is_exact_idle_gpu:
                heartbeat(
                    "GPU_INVENTORY_RACED_AFTER_EXACT_LOCK",
                    raced_gpu_index=int(candidate.index),
                    raced_gpu_uuid=str(candidate.uuid),
                )
            else:
                heartbeat("GPU_EXACT_LOCK_ACQUIRED", gpu=current.as_json())
                yield current
                return
        finally:
            lock.release()
        time.sleep(poll_seconds)


def _wait_for_64g_parent_headroom(
    cgroup: Path, *, heartbeat: Any
) -> dict[str, Any]:
    """Wait persistently without selecting or locking a GPU."""

    while True:
        observation = _cgroup_snapshot(cgroup)
        if observation["headroom_bytes"] >= CANARY_REQUIRED_HEADROOM_BYTES:
            return observation
        heartbeat(
            "WAITING_FOR_64G_PARENT_HEADROOM",
            parent_headroom_bytes=observation["headroom_bytes"],
            required_parent_headroom_bytes=CANARY_REQUIRED_HEADROOM_BYTES,
            gpu_lock_held=False,
        )
        time.sleep(HEADROOM_WAIT_SECONDS)


def _instrumentation_command(
    spec: Mapping[str, Any], *, run_root: Path, output_dir: Path
) -> list[str]:
    replay = dict(spec["replay"])
    return [
        str(spec["python"]),
        str(PROJECT_ROOT / "scripts/autodl/run_mut_checkpoint_instrumentation_equivalence.py"),
        "--config", "configs/hpc.yaml",
        "--set", "inference.fallback_to_heuristic=false",
        "run-pair",
        "--python", str(spec["python"]),
        "--legacy-project-root", str(spec["legacy_project_root"]),
        "--execution-project-root", str(spec["instrumentation_project_root"]),
        "--execution-commit", INSTRUMENTATION_COMMIT,
        "--expected-legacy-inventory-sha256", LEGACY_SOURCE_INVENTORY_SHA256,
        "--expected-instrumentation-inventory-sha256", INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
        "--run-root", str(run_root),
        "--output-dir", str(output_dir),
        "--upstream-root", str(replay["upstream_root"]),
        "--dataset-dir", str(replay["dataset_dir"]),
        "--gnn-checkpoint", str(replay["gnn_checkpoint"]),
        "--distance-checkpoint", str(replay["distance_checkpoint"]),
        "--parent-limit", str(int(replay.get("parent_limit", 1448))),
        "--device", "cuda:0",
        "--batch-size", str(int(replay.get("batch_size", 128))),
    ]


def _trace_mode_command(
    spec: Mapping[str, Any], *, run_root: Path, output_dir: Path
) -> list[str]:
    replay = dict(spec["replay"])
    return [
        str(spec["python"]),
        str(PROJECT_ROOT / "scripts/autodl/run_mut_trace_mode_equivalence.py"),
        "--config", "configs/hpc.yaml",
        "--set", "inference.fallback_to_heuristic=false",
        "run-pair",
        "--python", str(spec["python"]),
        "--legacy-project-root", str(spec["legacy_project_root"]),
        "--execution-project-root", str(spec["instrumentation_project_root"]),
        "--run-root", str(run_root),
        "--output-dir", str(output_dir),
        "--active-arm-path", str(run_root / "active_arm.json"),
        "--historical-artifact-root", str(spec["historical_source_root"]),
        "--rf-oracle", str(spec["standardization"]["teacher_path"]),
        "--upstream-root", str(replay["upstream_root"]),
        "--dataset-dir", str(replay["dataset_dir"]),
        "--gnn-checkpoint", str(replay["gnn_checkpoint"]),
        "--distance-checkpoint", str(replay["distance_checkpoint"]),
        "--parent-limit", str(int(replay.get("parent_limit", 1448))),
        "--device", "cuda:0",
        "--batch-size", str(int(replay.get("batch_size", 128))),
    ]


def _validate_memory_receipt(path: Path) -> dict[str, Any]:
    value = _physical_json(path, label="canary memory receipt")
    unhashed = {key: item for key, item in value.items() if key != "summary_sha256"}
    failures: list[str] = []
    if value.get("schema_version") != MEMORY_SCHEMA or value.get("status") != "PASS":
        failures.append("status")
    if value.get("failures") != []:
        failures.append("failures")
    if value.get("summary_sha256") != stable_json_sha256(unhashed):
        failures.append("summary_sha256")
    if value.get("arms_sequential") is not True or value.get("max_concurrent_arms") != 1:
        failures.append("sequential_arms")
    if value.get("initial_parent_headroom_admission_pass") is not True:
        failures.append("initial_headroom")
    initial = value.get("initial_cgroup")
    if (
        not isinstance(initial, Mapping)
        or int(initial.get("headroom_bytes", 0)) < CANARY_REQUIRED_HEADROOM_BYTES
        or int(initial.get("under_oom", -1)) != 0
    ):
        failures.append("initial_headroom_bytes")
    if int(value.get("initial_parent_headroom_bytes", 0)) < CANARY_REQUIRED_HEADROOM_BYTES:
        failures.append("initial_parent_headroom")
    if int(value.get("process_rss_peak_bytes", CANARY_RSS_STOP_BYTES + 1)) > CANARY_RSS_STOP_BYTES:
        failures.append("global_rss_peak")
    if int(value.get("parent_headroom_min_bytes", 0)) < CANARY_HEADROOM_STOP_BYTES:
        failures.append("global_headroom_min")
    phases = value.get("phase_stats") if isinstance(value.get("phase_stats"), Mapping) else {}
    for phase in REQUIRED_TRACE_PHASES:
        stats = phases.get(phase) if isinstance(phases, Mapping) else None
        if not isinstance(stats, Mapping) or int(stats.get("sample_count", 0)) <= 0:
            failures.append(f"phase:{phase}")
        elif (
            int(stats.get("peak_rss_bytes", CANARY_RSS_STOP_BYTES + 1)) > CANARY_RSS_STOP_BYTES
            or int(stats.get("minimum_parent_headroom_bytes", 0)) < CANARY_HEADROOM_STOP_BYTES
        ):
            failures.append(f"phase_threshold:{phase}")
    required_rows = value.get("phases")
    if not isinstance(required_rows, Mapping):
        failures.append("phases")
    else:
        for phase in REQUIRED_TRACE_PHASES:
            row = required_rows.get(phase)
            if (
                not isinstance(row, Mapping)
                or row.get("status") != "PASS"
                or int(row.get("sample_count", 0)) <= 0
            ):
                failures.append(f"phase_status:{phase}")
    protected = value.get("protected_throughput_gate")
    if not isinstance(protected, Mapping) or protected.get("status") != "PASS":
        failures.append("protected_gate")
    if protected and protected.get("missing_complete_five_minute_windows") != []:
        failures.append("protected_windows")
    if (
        int(value.get("cgroup_failcnt_delta", -1)) != 0
        or int(value.get("cgroup_oom_delta", -1)) != 0
        or int(value.get("cgroup_oom_kill_delta", -1)) != 0
    ):
        failures.append("cgroup_events")
    if failures:
        raise MutTraceWorkerError(f"Canary memory receipt failed closed: {failures}")
    return value


def _validate_trace_code_audit(path: Path) -> dict[str, Any]:
    value = _physical_json(path, label="trace code audit")
    unhashed = {key: item for key, item in value.items() if key != "audit_sha256"}
    failures: list[str] = []
    required = {
        "schema_version": TRACE_AUDIT_SCHEMA,
        "status": "PASS",
        "trace_is_observational": True,
        "trace_candidate_selection_is_observational": True,
        "trace_rng_mutation_found": False,
        "trace_algorithm_state_mutation_found": False,
        "trace_control_flow_mutation_found": False,
        "trace_operational_side_effects_found": True,
        "trace_post_walk_payload_serialization_mutation_found": True,
        "trace_post_walk_graph_closure_only": True,
        "static_audit_sufficient_for_adoption": False,
        "dynamic_500_step_equivalence_required": True,
        "full_trace_on_off_parity_claimed": False,
        "failures": [],
    }
    failures.extend(key for key, expected in required.items() if value.get(key) != expected)
    for key, expected_commit in (
        ("historical", SOURCE_COMMIT),
        ("instrumentation", INSTRUMENTATION_COMMIT),
    ):
        tree = value.get(key)
        if not isinstance(tree, Mapping):
            failures.append(key)
            continue
        branches = tree.get("branches")
        assertions = tree.get("scientific_assertions")
        if (
            tree.get("status") != "PASS"
            or tree.get("commit") != expected_commit
            or tree.get("unknown_branches") != []
            or tree.get("failed_scientific_assertions") != []
            or not isinstance(assertions, Mapping)
            or not assertions
            or any(assertion is not True for assertion in assertions.values())
            or not isinstance(branches, list)
            or not branches
            or any(
                not isinstance(row, Mapping)
                or row.get("classification") not in TRACE_BRANCH_ALLOWLIST
                for row in branches
            )
        ):
            failures.append(f"{key}_branch_inventory")
    if value.get("audit_sha256") != stable_json_sha256(unhashed):
        failures.append("audit_sha256")
    if failures:
        raise MutTraceWorkerError(f"Trace code audit binding failed: {sorted(set(failures))}")
    return value


def _binding_self_sha256(value: Mapping[str, Any]) -> str:
    unhashed = {key: item for key, item in value.items() if key != "binding_sha256"}
    payload = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_transitive_candidate_universe_binding(
    *,
    spec: Mapping[str, Any],
    adoption: Mapping[str, Any],
    adoption_root: Path,
) -> dict[str, Any]:
    """Independently reopen the exact source->pair->vectors->DBSCAN receipt.

    DBSCAN has no native candidate-universe identity.  Its only valid universe
    claim is the transitive one established by replaying the historical source
    candidate selection, checking every pair-store slice and concatenated vector
    byte, and reopening the exact DBSCAN consumer/output manifests.
    """

    source_root = _absolute(spec["historical_source_root"], label="historical source")
    source_payload = _absolute(
        source_root / "counterfactuals.pt", label="historical counterfactual payload"
    )
    common_root = _absolute(spec["completed_common_root"], label="completed common")
    pair_adoption_path = (
        common_root / "external_memory/pair_store_adoption/run_manifest.json"
    )
    pair_adoption = _physical_json(
        pair_adoption_path, label="pair-store adoption manifest"
    )
    pair_manifest_path = _absolute(
        str(pair_adoption.get("source_manifest_path") or ""),
        label="source pair-store manifest",
    )
    dbscan_manifest_path = _absolute(
        common_root / "external_memory/dbscan/run_manifest.json",
        label="DBSCAN manifest",
    )

    try:
        reopened = verify_mut_candidate_pair_dbscan_binding(
            source_payload_path=source_payload,
            pair_manifest_path=pair_manifest_path,
            dbscan_manifest_path=dbscan_manifest_path,
            expected_candidate_universe_sha256=(
                EXPECTED_CANDIDATE_UNIVERSE_SHA256
            ),
            expected_source_payload_sha256=SOURCE_PAYLOAD_SHA256,
            expected_candidate_count=50_620,
            candidate_capacity=100_000,
        )
    except Exception as exc:
        raise MutTraceWorkerError(
            f"Independent source/pair/vector/DBSCAN replay failed: {exc}"
        ) from exc

    embedded = adoption.get("candidate_pair_dbscan_binding_receipt")
    if not isinstance(embedded, Mapping):
        raise MutTraceWorkerError("Adoption lacks embedded candidate binding receipt")
    embedded_value = dict(embedded)
    external_path = _absolute(
        str(adoption.get("candidate_pair_dbscan_binding_path") or ""),
        label="external candidate binding receipt",
    )
    expected_external_path = (
        _absolute(adoption_root, label="adoption staging root")
        / "candidate_universe_binding.json"
    )
    external_value = _physical_json(
        external_path, label="external candidate binding receipt"
    )
    external_file_sha = sha256_file(external_path)

    expected = EXPECTED_CANDIDATE_UNIVERSE_SHA256
    failures: list[str] = []
    required_receipt = {
        "schema_version": BINDING_SCHEMA,
        "status": "PASS",
        "binding_kind": BINDING_KIND,
        "source_payload_path": str(source_payload),
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "candidate_count": 50_620,
        "source_native_candidate_universe_sha": expected,
        "pair_store_source_candidate_universe_sha": expected,
        "pair_store_manifest_path": str(pair_manifest_path),
        "pair_store_manifest_sha256": sha256_file(pair_manifest_path),
        "dbscan_manifest_path": str(dbscan_manifest_path),
        "dbscan_manifest_sha256": sha256_file(dbscan_manifest_path),
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_transitively_bound_candidate_universe_sha": expected,
        "dbscan_approximation_used": False,
        "candidate_universe_binding_state": "PASS",
    }
    failures.extend(
        f"receipt.{key}"
        for key, expected_value in required_receipt.items()
        if reopened.get(key) != expected_value
    )
    if reopened.get("binding_sha256") != _binding_self_sha256(reopened):
        failures.append("receipt.binding_sha256")
    if embedded_value != reopened:
        failures.append("adoption.embedded_receipt")
    if external_path != expected_external_path:
        failures.append("adoption.external_receipt_path")
    if external_value != reopened:
        failures.append("adoption.external_receipt_content")
    if (
        adoption.get("candidate_pair_dbscan_binding_sha256")
        != reopened.get("binding_sha256")
    ):
        failures.append("adoption.embedded_receipt_sha256")
    if (
        adoption.get("candidate_pair_dbscan_binding_file_sha256")
        != external_file_sha
    ):
        failures.append("adoption.external_receipt_file_sha256")

    required_adoption = {
        "source_payload_sha256": SOURCE_PAYLOAD_SHA256,
        "candidate_universe_sha": expected,
        "source_native_candidate_universe_sha": expected,
        "pair_store_source_candidate_universe_sha": expected,
        "dbscan_native_candidate_universe_sha": None,
        "dbscan_transitively_bound_candidate_universe_sha": expected,
        "candidate_universe_binding_state": "PASS",
        "transitive_binding_kind": BINDING_KIND,
        "dbscan_native_candidate_universe_field_present": False,
        "dbscan_universe_binding_via_pair_vectors": True,
        "pair_store_manifest_sha256": reopened.get(
            "pair_store_manifest_sha256"
        ),
        "dbscan_manifest_sha256": reopened.get("dbscan_manifest_sha256"),
    }
    failures.extend(
        f"adoption.{key}"
        for key, expected_value in required_adoption.items()
        if adoption.get(key) != expected_value
    )
    adoption_unhashed = {
        key: item for key, item in adoption.items() if key != "binding_sha256"
    }
    if adoption.get("binding_sha256") != stable_json_sha256(adoption_unhashed):
        failures.append("adoption.binding_sha256")
    if failures:
        raise MutTraceWorkerError(
            "Candidate-universe transitive binding failed: "
            + ",".join(sorted(set(failures)))
        )
    return reopened


def _verify_adoption(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    memory = _validate_memory_receipt(_absolute(args.memory_receipt, label="memory receipt"))
    _validate_trace_code_audit(_absolute(args.trace_code_audit, label="trace code audit"))
    instrumentation = validate_instrumentation_equivalence_gate(
        gate_path=_absolute(args.instrumentation_gate, label="instrumentation gate"),
        expected_legacy_inventory_sha256=LEGACY_SOURCE_INVENTORY_SHA256,
        expected_instrumentation_inventory_sha256=INSTRUMENTATION_SOURCE_INVENTORY_SHA256,
    )
    final = _absolute(args.output_dir, exists=False, label="adoption output")
    if final.exists():
        raise FileExistsError(f"Adoption output must be fresh: {final}")
    staging = final.parent / f".{final.name}.independent-verifier-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"Adoption verifier staging exists: {staging}")
    try:
        adoption = publish_adoption(
            spec=spec,
            inventory_gate=_absolute(args.inventory_gate, label="inventory gate"),
            equivalence_gate=_absolute(args.trace_mode_gate, label="trace-mode gate"),
            output_dir=staging,
            authorization_receipt=_absolute(args.authorization_receipt, label="authorization"),
            trace_code_audit=_absolute(args.trace_code_audit, label="trace audit"),
            instrumentation_equivalence_gate=_absolute(
                args.instrumentation_gate, label="instrumentation gate"
            ),
            canary_memory_receipt=_absolute(
                args.memory_receipt, label="canary memory receipt"
            ),
        )
        binding = _validate_transitive_candidate_universe_binding(
            spec=spec,
            adoption=adoption,
            adoption_root=staging,
        )
        verification = {
            "schema_version": VERIFICATION_SCHEMA,
            "status": "PASS",
            "independent_verifier_pid": os.getpid(),
            "source_algorithm_commit": SOURCE_COMMIT,
            "instrumentation_commit": INSTRUMENTATION_COMMIT,
            "instrumentation_gate_path": instrumentation["path"],
            "instrumentation_gate_sha256": instrumentation["sha256"],
            "trace_mode_gate_path": str(_absolute(args.trace_mode_gate)),
            "trace_mode_gate_sha256": sha256_file(_absolute(args.trace_mode_gate)),
            "memory_receipt_path": str(_absolute(args.memory_receipt)),
            "memory_receipt_sha256": sha256_file(_absolute(args.memory_receipt)),
            "candidate_universe_binding": binding,
            "pair_store_reused": adoption.get("pair_store_reused") is True,
            "dbscan_reused": adoption.get("dbscan_reused") is True,
            "pair_store_recompute_performed": adoption.get("pair_store_recompute_performed"),
            "dbscan_recompute_performed": adoption.get("dbscan_recompute_performed"),
            "trace_off_full_rerun_performed": False,
            "full_trace_on_off_parity_claimed": False,
            "verified_at": _utc_now(),
        }
        if (
            verification["pair_store_reused"] is not True
            or verification["dbscan_reused"] is not True
            or verification["pair_store_recompute_performed"] is not False
            or verification["dbscan_recompute_performed"] is not False
        ):
            raise MutTraceWorkerError("Pair-store/DBSCAN reuse disclosure changed")
        verification["verification_sha256"] = stable_json_sha256(verification)
        atomic_json(staging / "verification.json", verification)
        os.replace(staging, final)
        directory = os.open(final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if staging.exists():
            atomic_json(
                staging / "INDEPENDENT_VERIFIER_REJECTED.json",
                {"status": "REJECTED", "at": _utc_now()},
            )
        raise
    print(json.dumps(verification, sort_keys=True))
    print("[MUT_TRACE_ON_50K_ADOPTION_PASS]", flush=True)
    print("[MUT_CANDIDATE_UNIVERSE_BINDING_PASS]", flush=True)
    print("[MUT_PAIR_STORE_REUSED]", flush=True)
    print("[MUT_DBSCAN_REUSED]", flush=True)
    return 0


def _run_checked(command: Sequence[str], *, cwd: Path, log: Path, env: Mapping[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab", buffering=0) as handle:
        result = subprocess.run(
            list(command), cwd=cwd, env=dict(env), stdout=handle, stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise MutTraceWorkerError(f"Downstream stage failed ({result.returncode}): {command[1]}")


def _downstream_close(
    *, spec: Mapping[str, Any], root: Path, adoption_root: Path, heartbeat: Any
) -> dict[str, Any]:
    python = str(spec["python"])
    standard = dict(spec["standardization"])
    replay = dict(spec["replay"])
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RUN_GNN_ABLATION": "0",
    }
    threshold = root / "threshold-freeze"
    heartbeat("THRESHOLD_FREEZE_RUNNING")
    _run_checked(
        [
            python,
            str(PROJECT_ROOT / "scripts/autodl/verify_frozen_threshold_contract.py"),
            "--config", "configs/hpc.yaml",
            "--dataset", "mutagenicity",
            "--source", str(standard["thresholds_path"]),
            "--output", str(threshold),
        ],
        cwd=PROJECT_ROOT,
        log=root / "logs/threshold-freeze.log",
        env=environment,
    )
    for relative in (
        "frozen_threshold_contract.json",
        "threshold_adoption_audit.json",
        "PASS",
    ):
        _absolute(threshold / relative, label=f"threshold output {relative}")
    standardized = root / "standardized"
    heartbeat("STANDARDIZATION_RUNNING", standardized_root=str(standardized))
    _run_checked(
        [
            python,
            str(PROJECT_ROOT / "scripts/autodl/run_mut_comrecgc_parity_standardization.py"),
            "--config", "configs/hpc.yaml",
            "--set", "inference.fallback_to_heuristic=false",
            "--source-generation-root", str(spec["historical_source_root"]),
            "--upstream-root", str(replay["upstream_root"]),
            "--dataset-dir", str(replay["dataset_dir"]),
            "--distance-checkpoint", str(replay["distance_checkpoint"]),
            "--dataset-csv", str(standard["dataset_csv"]),
            "--teacher-path", str(standard["teacher_path"]),
            "--molclr-root", str(standard["molclr_root"]),
            "--molclr-checkpoint", str(standard["molclr_checkpoint"]),
            "--thresholds-path", str(threshold / "frozen_threshold_contract.json"),
            "--historical-adoption", str(adoption_root / "historical_adoption.json"),
            "--output-root", str(standardized),
            "--device", "cpu",
        ],
        cwd=PROJECT_ROOT,
        log=root / "logs/standardization.log",
        env=environment,
    )
    for relative in (
        "standardized/_FINALIZED.json",
        "standardized/run_manifest.json",
        "run_manifest.json",
        "final_gate.json",
        "_RUN_COMPLETE.json",
        "PASS",
    ):
        _absolute(standardized / relative, label=f"standardized output {relative}")
    heartbeat("MATRIX_PUBLICATION_SUBMITTING", standardized_root=str(standardized))
    matrix = _publish_matrix_queue(spec, standardized)
    return {
        "threshold_root": str(threshold),
        "standardized_root": str(standardized),
        "matrix_publication": matrix,
    }


def _write_fallback(root: Path, *, reason: str, science_gate_failed: bool) -> None:
    atomic_json(
        root / "fallback_route_required.json",
        {
            "schema_version": "mut_trace_on_adoption_fallback_v1",
            "fallback_route_required": bool(science_gate_failed),
            "fallback_route": "fresh_trace_off_route_b" if science_gate_failed else None,
            "fresh_generation_launched": False,
            "pair_store_recomputed": False,
            "dbscan_recomputed": False,
            "reason": reason,
            "recorded_at": _utc_now(),
        },
    )


def _run(args: argparse.Namespace) -> int:
    if os.environ.get("RUN_GNN_ABLATION", "0") != "0":
        raise MutTraceWorkerError("RUN_GNN_ABLATION must remain 0")
    _validate_worker_guard(args)
    spec = load_spec(args.spec)
    controller = _verify_controller(
        spec=spec,
        controller_pid=int(args.controller_pid),
        controller_start_ticks=int(args.controller_start_ticks),
    )
    control = Path(controller["control_dir"])
    output = _absolute(args.output_root, exists=False, label="worker output")
    repairs = (
        _absolute(spec["runtime_root"], label="runtime root")
        / "outputs/autodl/paper_matrix/four_methods_four_datasets_v1/repairs"
    )
    try:
        output.relative_to(repairs)
    except ValueError as exc:
        raise MutTraceWorkerError("Worker output escapes matrix repairs root") from exc
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Worker output must be fresh: {output}")
    output.mkdir(parents=True)
    heartbeat_path = control / "trace_on_adoption_worker_heartbeat.json"

    def heartbeat(state: str, **extra: Any) -> None:
        current = _verify_controller(
            spec=spec,
            controller_pid=int(args.controller_pid),
            controller_start_ticks=int(args.controller_start_ticks),
        )
        atomic_json(
            heartbeat_path,
            {
                "schema_version": SCHEMA,
                "state": state,
                "worker_pid": os.getpid(),
                "worker_start_ticks": _process_start_ticks(Path("/proc"), os.getpid()),
                "controller": current,
                "output_root": str(output),
                "run_gnn_ablation": False,
                "fresh_50k_launched": False,
                "pair_store_recomputed": False,
                "dbscan_recomputed": False,
                "heartbeat_at": _utc_now(),
                **extra,
            },
        )

    lease = control / "trace_on_adoption_worker.lock"
    with _worker_lease(lease, controller_id=str(spec["controller_id"])):
        try:
            heartbeat("AUTHORIZATION_LOADING")
            authorization, authorization_file_sha = validate_authorization_receipt(
                _absolute(args.authorization_receipt, label="authorization receipt"),
                expected_controller_id=str(spec["controller_id"]),
                expected_source_root=_absolute(
                    spec["historical_source_root"], label="historical artifact"
                ),
            )
            atomic_json(
                output / "authorization_adoption.json",
                {
                    "status": "PASS",
                    "authorization": authorization,
                    "authorization_file_sha256": authorization_file_sha,
                    "state": "TRACE_ON_ADOPTION_EVALUATION_READY",
                },
            )
            heartbeat("TRACE_ON_ADOPTION_EVALUATION_READY")
            replay_contract = _validate_frozen_replay_contract(spec)
            atomic_json(output / "frozen_replay_contract.json", replay_contract)
            historical_review = _absolute(
                args.historical_project_root, label="historical review worktree"
            )
            instrumentation_review = _absolute(
                args.instrumentation_project_root, label="instrumentation worktree"
            )
            if historical_review != _absolute(
                spec["legacy_project_root"], label="spec historical worktree"
            ) or instrumentation_review != _absolute(
                spec["instrumentation_project_root"],
                label="spec instrumentation worktree",
            ):
                raise MutTraceWorkerError(
                    "Static-audit worktrees differ from the execution spec"
                )
            audit_root = output / "trace-code-audit"
            audit = audit_trace_semantics(
                historical_root=historical_review,
                instrumentation_root=instrumentation_review,
                output_dir=audit_root,
            )
            _validate_trace_code_audit(audit_root / "trace_semantics_audit.json")
            heartbeat("TRACE_CODE_AUDIT_PASS", trace_code_audit=str(audit_root))
            matrix_state = _physical_json(
                _absolute(
                    Path(spec["control_root"]) / "fast16_matrix_authority/state.json",
                    label="unique matrix authority",
                ),
                label="unique matrix authority",
            )
            latest_authority = _absolute(
                str(matrix_state.get("latest_authority_root") or ""),
                label="latest matrix authority root",
            )
            atomic_json(
                output / "matrix_authority_binding.json",
                {
                    "status": "PASS",
                    "state_path": str(
                        Path(spec["control_root"]) / "fast16_matrix_authority/state.json"
                    ),
                    "latest_authority_root": str(latest_authority),
                    "single_authority_used": True,
                },
            )

            protected_manifest = load_protected_throughput_manifest(
                _absolute(args.protected_manifest, label="protected throughput manifest")
            )
            proc_root = _absolute(spec["proc_root"], label="proc root")
            cgroup = _absolute(spec["cgroup_memory_root"], label="cgroup memory root")
            inventory_root = output / "historical-inventory"
            inventory: dict[str, Any] | None = None
            baseline_attempt = 0
            while True:
                prebaseline_admission = _wait_for_64g_parent_headroom(
                    cgroup, heartbeat=heartbeat
                )
                atomic_json(
                    output / "parent_headroom_observation.json",
                    prebaseline_admission,
                )
                if inventory is None:
                    # The inventory streams multi-GiB artifacts.  It is a
                    # one-time science gate, but it must not add I/O pressure
                    # while the parent cgroup has only a few GiB free.
                    heartbeat(
                        "HISTORICAL_INVENTORY_RUNNING",
                        parent_headroom_bytes=prebaseline_admission[
                            "headroom_bytes"
                        ],
                        gpu_lock_held=False,
                    )
                    inventory = publish_inventory(
                        spec=spec, output_dir=inventory_root
                    )
                    if inventory.get("status") != "PASS":
                        raise MutTraceWorkerError(
                            "Historical 50k inventory gate failed"
                        )
                    heartbeat(
                        "HISTORICAL_INVENTORY_PASS",
                        inventory_gate=str(
                            inventory_root / "historical_inventory.json"
                        ),
                        gpu_lock_held=False,
                    )
                postinventory_admission = _cgroup_snapshot(cgroup)
                if (
                    postinventory_admission["headroom_bytes"]
                    < CANARY_REQUIRED_HEADROOM_BYTES
                ):
                    heartbeat(
                        "WAITING_FOR_64G_PARENT_HEADROOM",
                        wait_reason="headroom_dropped_after_inventory",
                        parent_headroom_bytes=postinventory_admission[
                            "headroom_bytes"
                        ],
                        required_parent_headroom_bytes=(
                            CANARY_REQUIRED_HEADROOM_BYTES
                        ),
                        gpu_lock_held=False,
                    )
                    time.sleep(HEADROOM_WAIT_SECONDS)
                    continue

                baseline_attempt += 1
                # The five-minute baseline intentionally holds no GPU
                # selection.  A candidate observed here would be stale by
                # launch time.
                heartbeat(
                    "PROTECTED_BASELINE_RUNNING",
                    baseline_attempt=baseline_attempt,
                    gpu_lock_held=False,
                )
                baseline = establish_protected_throughput_baseline(
                    protected_manifest,
                    proc_root=proc_root,
                    baseline_seconds=PROTECTED_WINDOW_SECONDS,
                    poll_seconds=SAMPLE_SECONDS,
                    progress_callback=lambda elapsed, _first: heartbeat(
                        "PROTECTED_BASELINE_RUNNING",
                        baseline_attempt=baseline_attempt,
                        baseline_elapsed_seconds=elapsed,
                        gpu_lock_held=False,
                    ),
                )
                atomic_json(
                    output
                    / f"protected_throughput_baseline_attempt_{baseline_attempt:04d}.json",
                    baseline,
                )
                atomic_json(output / "protected_throughput_baseline.json", baseline)
                if baseline.get("status") != "PASS":
                    raise MutTraceWorkerError(
                        "Protected-task five-minute baseline failed: "
                        + ",".join(
                            str(item) for item in baseline.get("failures", [])
                        )
                    )
                postbaseline_admission = _cgroup_snapshot(cgroup)
                if (
                    postbaseline_admission["headroom_bytes"]
                    < CANARY_REQUIRED_HEADROOM_BYTES
                ):
                    heartbeat(
                        "WAITING_FOR_64G_PARENT_HEADROOM",
                        wait_reason="headroom_dropped_after_baseline",
                        parent_headroom_bytes=postbaseline_admission["headroom_bytes"],
                        required_parent_headroom_bytes=CANARY_REQUIRED_HEADROOM_BYTES,
                        gpu_lock_held=False,
                    )
                    time.sleep(HEADROOM_WAIT_SECONDS)
                    continue

                # Re-query only after baseline and hold the exact physical UUID
                # lock before accepting the refreshed nvidia-smi observation.
                headroom_retry = False
                with _fresh_idle_gpu_lock(
                    spec, cgroup=cgroup, heartbeat=heartbeat
                ) as selected:
                    if selected is None:
                        headroom_retry = True
                    else:
                        admission = _cgroup_snapshot(cgroup)
                        atomic_json(output / "canary_admission.json", admission)
                        if (
                            admission["headroom_bytes"]
                            < CANARY_REQUIRED_HEADROOM_BYTES
                        ):
                            headroom_retry = True
                            heartbeat(
                                "WAITING_FOR_64G_PARENT_HEADROOM",
                                wait_reason=(
                                    "headroom_dropped_after_exact_gpu_lock"
                                ),
                                parent_headroom_bytes=admission[
                                    "headroom_bytes"
                                ],
                                required_parent_headroom_bytes=(
                                    CANARY_REQUIRED_HEADROOM_BYTES
                                ),
                                gpu_index=int(selected.index),
                                gpu_uuid=str(selected.uuid),
                                gpu_lock_release_pending=True,
                            )
                        else:
                            gate = ProtectedThroughputGate(
                                protected_manifest,
                                baseline,
                                proc_root=proc_root,
                                window_seconds=PROTECTED_WINDOW_SECONDS,
                                maximum_slowdown=0.10,
                            )
                            monitor = CanaryMonitor(
                                cgroup_root=cgroup,
                                proc_root=proc_root,
                                gpu_uuid=str(selected.uuid),
                                monitor_path=output / "canary_memory_monitor.jsonl",
                                throughput_gate=gate,
                                heartbeat=heartbeat,
                            )
                            environment = {
                                **os.environ,
                                "CUDA_VISIBLE_DEVICES": str(selected.index),
                                "PYTHONDONTWRITEBYTECODE": "1",
                                "PYTHONHASHSEED": "0",
                                "OMP_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1",
                                "TOKENIZERS_PARALLELISM": "false",
                                "RUN_GNN_ABLATION": "0",
                            }
                            gate_runs: list[dict[str, Any]] = []
                            instrumentation_run = output / "instrumentation-equivalence-run"
                            instrumentation_output = output / "instrumentation-equivalence"
                            gate_runs.append(
                                monitor.run_process(
                                    _instrumentation_command(
                                        spec,
                                        run_root=instrumentation_run,
                                        output_dir=instrumentation_output,
                                    ),
                                    cwd=PROJECT_ROOT,
                                    environment=environment,
                                    log_path=output / "logs/instrumentation-equivalence.log",
                                    static_phase="instrumentation_equivalence",
                                    checkpoint_scan_root=instrumentation_run,
                                )
                            )
                            instrumentation_gate = (
                                instrumentation_output / "equivalence.json"
                            )
                            validate_instrumentation_equivalence_gate(
                                gate_path=instrumentation_gate,
                                expected_legacy_inventory_sha256=(
                                    LEGACY_SOURCE_INVENTORY_SHA256
                                ),
                                expected_instrumentation_inventory_sha256=(
                                    INSTRUMENTATION_SOURCE_INVENTORY_SHA256
                                ),
                            )
                            trace_run = output / "trace-mode-equivalence-run"
                            trace_output = output / "trace-mode-equivalence"
                            gate_runs.append(
                                monitor.run_process(
                                    _trace_mode_command(
                                        spec,
                                        run_root=trace_run,
                                        output_dir=trace_output,
                                    ),
                                    cwd=PROJECT_ROOT,
                                    environment=environment,
                                    log_path=output / "logs/trace-mode-equivalence.log",
                                    active_arm=trace_run / "active_arm.json",
                                    checkpoint_scan_root=trace_run,
                                )
                            )
                            trace_gate = (
                                trace_output / "trace_on_off_500_step_equivalence.json"
                            )
                            trace_value = _physical_json(
                                trace_gate, label="trace-mode equivalence"
                            )
                            if trace_value.get("status") != "PASS":
                                raise MutTraceWorkerError(
                                    "Trace-on/off semantic equivalence failed"
                                )
                            protected_receipt = monitor.finish_protected_window()
                            atomic_json(
                                output / "protected_throughput_gate.json",
                                protected_receipt,
                            )
                            memory_receipt = monitor.receipt(
                                protected_baseline=baseline,
                                protected_gate=protected_receipt,
                                gate_runs=gate_runs,
                            )
                            atomic_json(
                                output / "mut_trace_mode_canary_memory.json",
                                memory_receipt,
                            )
                            _validate_memory_receipt(
                                output / "mut_trace_mode_canary_memory.json"
                            )
                            heartbeat(
                                "SEQUENTIAL_EQUIVALENCE_GATES_PASS",
                                gpu_index=selected.index,
                                instrumentation_gate=str(instrumentation_gate),
                                trace_mode_gate=str(trace_gate),
                            )
                if headroom_retry:
                    heartbeat(
                        "WAITING_FOR_64G_PARENT_HEADROOM",
                        wait_reason="exact_gpu_lock_released_for_headroom_wait",
                        gpu_lock_held=False,
                    )
                    time.sleep(HEADROOM_WAIT_SECONDS)
                    continue
                break

            adoption_root = output / "trace-on-50k-adoption"
            verify_command = [
                str(spec["python"]),
                str(Path(__file__).resolve()),
                "verify-adoption",
                "--spec", str(args.spec),
                "--inventory-gate", str(inventory_root / "historical_inventory.json"),
                "--instrumentation-gate", str(instrumentation_gate),
                "--trace-mode-gate", str(trace_gate),
                "--memory-receipt", str(output / "mut_trace_mode_canary_memory.json"),
                "--authorization-receipt", str(args.authorization_receipt),
                "--trace-code-audit", str(audit_root / "trace_semantics_audit.json"),
                "--output-dir", str(adoption_root),
            ]
            heartbeat("ADOPTION_INDEPENDENT_VERIFIER_RUNNING")
            _run_checked(
                verify_command,
                cwd=PROJECT_ROOT,
                log=output / "logs/adoption-verifier.log",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "RUN_GNN_ABLATION": "0"},
            )
            _absolute(adoption_root / "verification.json", label="adoption verification")
            closure = _downstream_close(
                spec=spec,
                root=output / "cell-closure",
                adoption_root=adoption_root,
                heartbeat=heartbeat,
            )
            atomic_json(
                output / "worker_terminal.json",
                {
                    "schema_version": SCHEMA,
                    "status": "PASS_MATRIX_PUBLISHER_SUBMITTED",
                    "controller": controller,
                    "authorization_receipt": str(args.authorization_receipt),
                    "instrumentation_gate": str(instrumentation_gate),
                    "trace_mode_gate": str(trace_gate),
                    "memory_receipt": str(output / "mut_trace_mode_canary_memory.json"),
                    "adoption_root": str(adoption_root),
                    "closure": closure,
                    "fresh_50k_launched": False,
                    "pair_store_recomputed": False,
                    "dbscan_recomputed": False,
                    "run_gnn_ablation": False,
                    "completed_at": _utc_now(),
                },
            )
            heartbeat("PASS_MATRIX_PUBLISHER_SUBMITTED", **closure)
            print("[MUT_TRACE_ON_50K_ADOPTION_PASS]", flush=True)
            print("[MUT_PAIR_STORE_REUSED]", flush=True)
            print("[MUT_DBSCAN_REUSED]", flush=True)
            print("[GNN_BACKBONE_ABLATION_NOT_STARTED_BY_POLICY]", flush=True)
            return 0
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            science_failure_tokens = (
                "equivalence failed",
                "semantic",
                "Trace code audit",
                "instrumentation gate",
                "candidate-universe",
                "checkpoint",
                "Historical 50k inventory",
            )
            science_failed = any(
                token.lower() in reason.lower() for token in science_failure_tokens
            )
            _write_fallback(output, reason=reason, science_gate_failed=science_failed)
            heartbeat(
                "BLOCKED",
                error=reason,
                fallback_route_required=science_failed,
                fresh_generation_launched=False,
            )
            raise


def _authorize(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    result = write_authorization_receipt(
        path=_absolute(args.output, exists=False, label="authorization output"),
        controller_id=str(spec["controller_id"]),
        source_root=_absolute(spec["historical_source_root"], label="historical artifact"),
    )
    print(json.dumps(result, sort_keys=True))
    print("[MUT_TRACE_ON_ADOPTION_AUTHORIZED]", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="action", required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--spec", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-adoption")
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--inventory-gate", type=Path, required=True)
    verify.add_argument("--instrumentation-gate", type=Path, required=True)
    verify.add_argument("--trace-mode-gate", type=Path, required=True)
    verify.add_argument("--memory-receipt", type=Path, required=True)
    verify.add_argument("--authorization-receipt", type=Path, required=True)
    verify.add_argument("--trace-code-audit", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--authorization-receipt", type=Path, required=True)
    run.add_argument("--protected-manifest", type=Path, required=True)
    run.add_argument("--historical-project-root", type=Path, required=True)
    run.add_argument("--instrumentation-project-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--controller-pid", type=int, required=True)
    run.add_argument("--controller-start-ticks", type=int, required=True)
    run.add_argument("--successor-guard-script", required=True, help=argparse.SUPPRESS)
    run.add_argument("--successor-guard-action", required=True, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "authorize":
        return _authorize(args)
    if args.action == "verify-adoption":
        return _verify_adoption(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
