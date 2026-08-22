"""Small, fail-closed runtime primitives for local AutoDL experiments.

This module is shared by the frozen-GNN launchers and the manifest-driven
four-GPU recovery controller.  Scientific policy remains outside these small
runtime primitives.

The helpers here keep policy separate from scientific code:

* a GPU is eligible only after every sample in a stability window is idle;
* ordinary callers select at most two GPUs; the recovery controller must opt in
  explicitly to the audited four-GPU hard ceiling;
* a UUID-scoped advisory lock serializes project-owned GPU workers;
* registry and stage documents are written atomically and durably; and
* BACE stages form a fixed, predecessor-gated state machine.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, TextIO


# Frozen-GNN launchers retain the conservative two-GPU default.  The recovery
# controller may opt in to the separately audited four-GPU ceiling by passing
# ``hard_limit=FOUR_GPU_RECOVERY_LIMIT``.  Keeping the higher limit explicit
# prevents an unrelated call site from silently expanding its resource scope.
MAX_AUTODL_GPUS = 2
FOUR_GPU_RECOVERY_LIMIT = 4

BACE_STAGES: tuple[str, ...] = (
    "B0_AUDIT",
    "B1_DATA_READY",
    "B2_GNN_SMOKE",
    "B3_GNN_FULL",
    "B4_GNN_CALIBRATED",
    "B5_ORACLE_SMOKE",
    "B6_PPO_SMOKE",
    "B7_PPO_FULL",
    "B8_POOL_BASE",
    "B9_POOL_HIGHTEMP",
    "B10_POOL_MERGED",
    "B11_CROSS_PARENT_VERIFIED",
    "B12_SELECTOR",
    "B13_FINAL_EVAL",
    "B14_FROZEN",
)
BACE_PREDECESSOR: dict[str, str | None] = {
    stage: (BACE_STAGES[index - 1] if index else None)
    for index, stage in enumerate(BACE_STAGES)
}

STAGE_STATES = {
    "NOT_STARTED",
    "WAITING_FOR_GPU",
    "STARTING",
    "RUNNING",
    "PASS",
    "FAILED",
    "BLOCKED",
    "STOPPED",
}

_SECRET_KEY = re.compile(
    r"(?i)(password|passwd|secret|token|authorization|api[_-]?key|"
    r"credential|private[_-]?key)"
)
_SAFE_LOCK_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


class AutoDLRuntimeError(RuntimeError):
    """A fail-closed AutoDL runtime error."""


class GPUInventoryError(AutoDLRuntimeError):
    """GPU state could not be determined safely."""


class GPULockError(AutoDLRuntimeError):
    """A physical GPU lock could not be acquired or validated."""


class StageTransitionError(AutoDLRuntimeError):
    """A requested stage transition violates the frozen BACE order."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one JSON object and fsync its directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # corruption must not be interpreted as NOT_STARTED
        raise AutoDLRuntimeError(f"Invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise AutoDLRuntimeError(f"Expected JSON object: {path}")
    return payload


def append_jsonl_locked(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one registry event under an advisory lock and fsync it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            fsync_directory(path.parent)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[Path]) -> str | None:
    """Hash path names and bytes without scanning unrelated directory trees."""

    normalized = sorted({path.resolve(strict=True) for path in paths if path.is_file()})
    if not normalized:
        return None
    digest = hashlib.sha256()
    for path in normalized:
        encoded = str(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def command_digest(command: Sequence[str]) -> str:
    body = json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def sanitized_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Do not leak login/API credentials into long-lived scientific workers."""

    source = os.environ if environment is None else environment
    return {str(key): str(value) for key, value in source.items() if not _SECRET_KEY.search(str(key))}


def resolve_project_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve(strict=True)
    else:
        try:
            value = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AutoDLRuntimeError("Current directory is not inside a Git worktree") from exc
        root = Path(value).resolve(strict=True)
    if not (root / ".git").exists():
        # Linked worktrees have a .git file, primary worktrees have a directory.
        raise AutoDLRuntimeError(f"Not a Git worktree root: {root}")
    return root


def _write_probe(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".autodl-write-probe.", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"autodl-runtime-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        Path(name).unlink(missing_ok=True)


def select_data_root(
    project_root: Path,
    *,
    explicit: Path | None = None,
    min_free_bytes: int = 1,
) -> Path:
    """Select one writable AutoDL data root using the documented priority."""

    env_value = os.environ.get("AUTODL_DATA_ROOT")
    explicit_value = explicit or (Path(env_value).expanduser() if env_value else None)
    if explicit_value is not None:
        candidates = [explicit_value]
        strict = True
    else:
        candidates = [
            Path("/autodl-fs/data"),
            Path("/root/autodl-fs"),
            Path("/root/autodl-tmp"),
            project_root,
        ]
        strict = False

    reasons: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve(strict=False)
        if not strict and candidate != project_root and not candidate.is_dir():
            continue
        try:
            _write_probe(candidate)
            free_bytes = shutil.disk_usage(candidate).free
            if free_bytes < min_free_bytes:
                reasons.append(f"{candidate}: free={free_bytes} < {min_free_bytes}")
                continue
            return candidate
        except OSError as exc:
            reasons.append(f"{candidate}: {exc}")
    raise AutoDLRuntimeError(
        "No writable AutoDL data root passed the write/capacity probe: "
        + "; ".join(reasons)
    )


@dataclass(frozen=True)
class RuntimeLayout:
    project_root: Path
    data_root: Path
    runtime_root: Path
    data_dir: Path
    checkpoints_dir: Path
    cache_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    locks_dir: Path
    control_root: Path
    registry_path: Path
    stages_root: Path
    runs_root: Path

    def ensure(self) -> "RuntimeLayout":
        for path in (
            self.runtime_root,
            self.data_dir,
            self.checkpoints_dir,
            self.cache_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.locks_dir,
            self.control_root,
            self.registry_path.parent,
            self.stages_root,
            self.runs_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def as_json(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def build_runtime_layout(
    *,
    project_root: Path,
    data_root: Path,
    control_root: Path | None = None,
) -> RuntimeLayout:
    project_root = project_root.resolve(strict=True)
    data_root = data_root.resolve(strict=True)
    runtime_root = data_root / "counterfactual-subgraph-runtime"
    configured_control_root: Path
    if control_root is not None:
        configured_control_root = control_root.expanduser()
    else:
        environment_value = os.environ.get("AUTODL_CONTROL_ROOT")
        configured_control_root = (
            Path(environment_value).expanduser()
            if environment_value is not None
            else runtime_root / "control"
        )
    if not configured_control_root.is_absolute():
        raise AutoDLRuntimeError(
            "AUTODL_CONTROL_ROOT must be an absolute path, got "
            f"{configured_control_root}"
        )
    if configured_control_root.is_symlink():
        raise AutoDLRuntimeError(
            f"AUTODL_CONTROL_ROOT must not be a symlink: {configured_control_root}"
        )
    resolved_control_root = configured_control_root.resolve(strict=False)
    try:
        resolved_control_root.relative_to(data_root)
    except ValueError as exc:
        raise AutoDLRuntimeError(
            "AUTODL_CONTROL_ROOT must be contained by the selected persistent "
            f"data root {data_root}, got {resolved_control_root}"
        ) from exc
    try:
        resolved_control_root.relative_to(project_root)
    except ValueError:
        pass
    else:
        raise AutoDLRuntimeError(
            "AUTODL_CONTROL_ROOT must not be inside the code worktree: "
            f"{resolved_control_root}"
        )
    return RuntimeLayout(
        project_root=project_root,
        data_root=data_root,
        runtime_root=runtime_root,
        data_dir=runtime_root / "data",
        checkpoints_dir=runtime_root / "checkpoints",
        cache_dir=runtime_root / "cache",
        artifacts_dir=runtime_root / "outputs",
        logs_dir=runtime_root / "logs",
        locks_dir=runtime_root / "locks",
        control_root=resolved_control_root,
        registry_path=resolved_control_root / "experiment_registry" / "runs.jsonl",
        stages_root=resolved_control_root / "bace" / "stages",
        runs_root=resolved_control_root / "experiment_registry" / "run_state",
    )


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    process_name: str
    used_memory_mb: int


@dataclass(frozen=True)
class GPUObservation:
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_gpu_percent: int
    processes: tuple[GPUProcess, ...] = ()

    @property
    def process_count(self) -> int:
        return len(self.processes)

    def is_idle(self, *, min_free_memory_mb: int, max_utilization_percent: int) -> bool:
        return (
            not self.processes
            and self.memory_free_mb >= min_free_memory_mb
            and self.utilization_gpu_percent <= max_utilization_percent
        )

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["processes"] = [asdict(process) for process in self.processes]
        payload["process_count"] = self.process_count
        return payload


def _integer(value: str, *, label: str) -> int:
    match = re.search(r"-?\d+", value)
    if match is None:
        raise GPUInventoryError(f"nvidia-smi returned non-integer {label}: {value!r}")
    return int(match.group(0))


def parse_gpu_inventory(
    gpu_rows: str,
    process_rows: str = "",
) -> list[GPUObservation]:
    """Parse nounits CSV output from two nvidia-smi queries."""

    process_map: dict[str, list[GPUProcess]] = {}
    for row in csv.reader(line for line in process_rows.splitlines() if line.strip()):
        if len(row) < 4:
            raise GPUInventoryError(f"Malformed compute-process row: {row!r}")
        uuid, pid, process_name, used = (value.strip() for value in row[:4])
        process_map.setdefault(uuid, []).append(
            GPUProcess(
                pid=_integer(pid, label="pid"),
                process_name=process_name,
                used_memory_mb=_integer(used, label="used_gpu_memory"),
            )
        )

    observations: list[GPUObservation] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for row in csv.reader(line for line in gpu_rows.splitlines() if line.strip()):
        if len(row) < 7:
            raise GPUInventoryError(f"Malformed GPU row: {row!r}")
        index_value, uuid, name, total, used, free, utilization = (
            value.strip() for value in row[:7]
        )
        index = _integer(index_value, label="index")
        if index in seen_indices or uuid in seen_uuids:
            raise GPUInventoryError(f"Duplicate GPU index/UUID: index={index}, uuid={uuid}")
        seen_indices.add(index)
        seen_uuids.add(uuid)
        observations.append(
            GPUObservation(
                index=index,
                uuid=uuid,
                name=name,
                memory_total_mb=_integer(total, label="memory.total"),
                memory_used_mb=_integer(used, label="memory.used"),
                memory_free_mb=_integer(free, label="memory.free"),
                utilization_gpu_percent=_integer(utilization, label="utilization.gpu"),
                processes=tuple(process_map.get(uuid, ())),
            )
        )
    return sorted(observations, key=lambda observation: observation.index)


def query_gpu_inventory(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[GPUObservation]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise GPUInventoryError("nvidia-smi is unavailable")
    gpu_command = [
        executable,
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        executable,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = runner(gpu_command, text=True, capture_output=True, check=False)
    if gpu_result.returncode != 0:
        raise GPUInventoryError(
            f"nvidia-smi GPU query failed ({gpu_result.returncode}): "
            f"{gpu_result.stderr.strip()}"
        )
    process_result = runner(process_command, text=True, capture_output=True, check=False)
    if process_result.returncode == 0:
        process_rows = process_result.stdout
    elif "no running processes found" in (
        process_result.stdout + "\n" + process_result.stderr
    ).lower():
        process_rows = ""
    else:
        raise GPUInventoryError(
            f"nvidia-smi compute-process query failed ({process_result.returncode}): "
            f"{process_result.stderr.strip()}"
        )
    observations = parse_gpu_inventory(gpu_result.stdout, process_rows)
    if not observations:
        raise GPUInventoryError("nvidia-smi returned no physical GPUs")
    return observations


def validate_max_gpus(value: int, *, hard_limit: int = MAX_AUTODL_GPUS) -> int:
    if hard_limit < 1 or hard_limit > FOUR_GPU_RECOVERY_LIMIT:
        raise GPUInventoryError(
            "GPU hard limit must be in "
            f"[1, {FOUR_GPU_RECOVERY_LIMIT}], got {hard_limit}"
        )
    if value < 1 or value > hard_limit:
        raise GPUInventoryError(
            f"AUTODL_MAX_GPUS must be in [1, {hard_limit}], got {value}"
        )
    return value


@dataclass(frozen=True)
class StableGPUInventory:
    sampled_at: str
    stable_seconds: float
    samples: int
    observations: tuple[GPUObservation, ...]
    stable_idle_uuids: frozenset[str]

    def selected(
        self,
        *,
        max_gpus: int,
        lock_root: Path | None = None,
        hard_limit: int = MAX_AUTODL_GPUS,
    ) -> list[GPUObservation]:
        limit = validate_max_gpus(max_gpus, hard_limit=hard_limit)
        if lock_root is not None:
            limit = min(
                limit,
                available_project_gpu_slots(
                    lock_root, limit, hard_limit=hard_limit
                ),
            )
        if limit <= 0:
            return []
        selected: list[GPUObservation] = []
        for observation in self.observations:
            if observation.uuid not in self.stable_idle_uuids:
                continue
            if lock_root is not None and not gpu_lock_available(lock_root, observation.uuid):
                continue
            selected.append(observation)
            if len(selected) >= limit:
                break
        return selected


def observe_stable_idle_gpus(
    *,
    stable_seconds: float,
    sample_interval_seconds: float,
    min_free_memory_mb: int,
    max_utilization_percent: int,
    sampler: Callable[[], list[GPUObservation]] = query_gpu_inventory,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> StableGPUInventory:
    """Intersect idle UUIDs across the whole requested stability window."""

    if stable_seconds < 0:
        raise GPUInventoryError("stable_seconds cannot be negative")
    if sample_interval_seconds <= 0:
        raise GPUInventoryError("sample_interval_seconds must be positive")

    started = monotonic()
    observations = sampler()
    stable = {
        observation.uuid
        for observation in observations
        if observation.is_idle(
            min_free_memory_mb=min_free_memory_mb,
            max_utilization_percent=max_utilization_percent,
        )
    }
    samples = 1
    while stable and monotonic() - started < stable_seconds:
        remaining = stable_seconds - (monotonic() - started)
        sleep(min(sample_interval_seconds, max(0.0, remaining)))
        observations = sampler()
        idle_now = {
            observation.uuid
            for observation in observations
            if observation.is_idle(
                min_free_memory_mb=min_free_memory_mb,
                max_utilization_percent=max_utilization_percent,
            )
        }
        stable.intersection_update(idle_now)
        samples += 1

    # Keep the last physical index/name/memory observation for selected UUIDs.
    return StableGPUInventory(
        sampled_at=utc_now(),
        stable_seconds=stable_seconds,
        samples=samples,
        observations=tuple(observations),
        stable_idle_uuids=frozenset(stable),
    )


def _gpu_lock_path(lock_root: Path, gpu_uuid: str) -> Path:
    component = _SAFE_LOCK_COMPONENT.sub("_", gpu_uuid).strip("._")
    if not component:
        raise GPULockError(f"Unsafe or empty GPU UUID: {gpu_uuid!r}")
    return lock_root / f"gpu-{component}.lock"


class GPUFileLock(AbstractContextManager["GPUFileLock"]):
    """Nonblocking advisory lock keyed by immutable physical GPU UUID."""

    def __init__(
        self,
        lock_root: Path,
        *,
        gpu_index: int,
        gpu_uuid: str,
        owner: Mapping[str, Any] | None = None,
    ) -> None:
        self.lock_root = lock_root
        self.gpu_index = int(gpu_index)
        self.gpu_uuid = str(gpu_uuid)
        self.owner = dict(owner or {})
        self.path = _gpu_lock_path(lock_root, gpu_uuid)
        self._handle: TextIO | None = None

    def acquire(self) -> "GPUFileLock":
        self.lock_root.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise GPULockError(f"GPU lock path must not be a symlink: {self.path}")
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            detail = handle.read().strip() or "owner metadata unavailable"
            handle.close()
            raise GPULockError(
                f"GPU {self.gpu_index} ({self.gpu_uuid}) is project-locked: {detail}"
            ) from exc
        payload = {
            "schema_version": 1,
            "state": "LOCKED",
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "pid": os.getpid(),
            "acquired_at": utc_now(),
            **self.owner,
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        payload = {
            "schema_version": 1,
            "state": "RELEASED",
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "pid": os.getpid(),
            "released_at": utc_now(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._handle = None

    def __enter__(self) -> "GPUFileLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


class ProjectGPUSlotLock(AbstractContextManager["ProjectGPUSlotLock"]):
    """Bound aggregate frozen-GNN concurrency independently of GPU UUIDs."""

    def __init__(
        self,
        lock_root: Path,
        *,
        max_slots: int,
        hard_limit: int = MAX_AUTODL_GPUS,
        owner: Mapping[str, Any] | None = None,
    ) -> None:
        self.lock_root = lock_root
        self.hard_limit = hard_limit
        self.max_slots = validate_max_gpus(max_slots, hard_limit=hard_limit)
        self.owner = dict(owner or {})
        self.slot: int | None = None
        self.path: Path | None = None
        self._handle: TextIO | None = None

    def acquire(self) -> "ProjectGPUSlotLock":
        self.lock_root.mkdir(parents=True, exist_ok=True)
        for slot in range(self.max_slots):
            path = self.lock_root / f"project-gpu-slot-{slot}.lock"
            if path.is_symlink():
                continue
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            payload = {
                "schema_version": 1,
                "state": "LOCKED",
                "slot": slot,
                "pid": os.getpid(),
                "acquired_at": utc_now(),
                **self.owner,
            }
            handle.seek(0)
            handle.truncate()
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            self.slot = slot
            self.path = path
            self._handle = handle
            return self
        raise GPULockError(
            f"All {self.max_slots} project GPU slots are already locked"
        )

    def release(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "schema_version": 1,
                "state": "RELEASED",
                "slot": self.slot,
                "pid": os.getpid(),
                "released_at": utc_now(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._handle = None

    def __enter__(self) -> "ProjectGPUSlotLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def available_project_gpu_slots(
    lock_root: Path,
    max_slots: int,
    *,
    hard_limit: int = MAX_AUTODL_GPUS,
) -> int:
    """Count currently acquirable project slots without changing metadata."""

    maximum = validate_max_gpus(max_slots, hard_limit=hard_limit)
    lock_root.mkdir(parents=True, exist_ok=True)
    handles: list[TextIO] = []
    try:
        for slot in range(maximum):
            path = lock_root / f"project-gpu-slot-{slot}.lock"
            if path.is_symlink():
                continue
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            handles.append(handle)
        return len(handles)
    finally:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def gpu_lock_available(lock_root: Path, gpu_uuid: str) -> bool:
    """Probe a lock without rewriting its owner metadata."""

    lock_root.mkdir(parents=True, exist_ok=True)
    path = _gpu_lock_path(lock_root, gpu_uuid)
    if path.is_symlink():
        return False
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return True


def assert_tastemolnet_launch_allowed(
    *, dataset: str,
    heavy: bool,
    run_tastemolnet: str | int | bool | None = None,
) -> None:
    normalized = dataset.strip().lower()
    enabled = str(
        os.environ.get("RUN_TASTEMOLNET", "0")
        if run_tastemolnet is None
        else run_tastemolnet
    ).strip().lower() in {"1", "true", "yes", "on"}
    if normalized in {"tastemolnet", "taste", "bst", "bitter_sweet_tasteless"} and heavy and not enabled:
        raise AutoDLRuntimeError(
            "TASTEMOLNET_HEAVY_RUN_DISABLED: set RUN_TASTEMOLNET=1 only after "
            "explicit authorization"
        )


def stage_paths(layout: RuntimeLayout, stage: str) -> dict[str, Path]:
    root = layout.stages_root / stage
    return {
        "root": root,
        "state": root / "state.json",
        "manifest": root / "manifest.json",
        "gate": root / "gate.json",
    }


def initialize_bace_stage_tree(layout: RuntimeLayout) -> None:
    """Create the B0--B14 templates without overwriting existing evidence."""

    layout.ensure()
    for stage in BACE_STAGES:
        paths = stage_paths(layout, stage)
        paths["root"].mkdir(parents=True, exist_ok=True)
        predecessor = BACE_PREDECESSOR[stage]
        if not paths["state"].exists():
            atomic_write_json(
                paths["state"],
                {
                    "schema_version": 1,
                    "dataset": "bace",
                    "stage": stage,
                    "predecessor": predecessor,
                    "state": "NOT_STARTED",
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "run_id": None,
                    "pid": None,
                    "tmux_session": None,
                    "gpu_index": None,
                    "gpu_uuid": None,
                },
            )
        if not paths["manifest"].exists():
            atomic_write_json(
                paths["manifest"],
                {
                    "schema_version": 1,
                    "dataset": "bace",
                    "stage": stage,
                    "status": "NOT_AVAILABLE",
                    "command": None,
                    "config_hash": None,
                    "input_hash": None,
                    "expected_output": None,
                },
            )
        if not paths["gate"].exists():
            atomic_write_json(
                paths["gate"],
                {
                    "schema_version": 1,
                    "dataset": "bace",
                    "stage": stage,
                    "status": "NOT_EVALUATED",
                    "checked_at": None,
                    "evidence": [],
                    "reason": None,
                },
            )


def read_bace_stage(layout: RuntimeLayout, stage: str) -> dict[str, dict[str, Any]]:
    if stage not in BACE_PREDECESSOR:
        raise StageTransitionError(f"Unknown BACE stage: {stage}")
    paths = stage_paths(layout, stage)
    return {
        "state": read_json_object(paths["state"]),
        "manifest": read_json_object(paths["manifest"]),
        "gate": read_json_object(paths["gate"]),
    }


def assert_bace_stage_can_start(layout: RuntimeLayout, stage: str) -> None:
    current = read_bace_stage(layout, stage)
    state = str(current["state"].get("state"))
    if state in {"STARTING", "RUNNING", "PASS"}:
        raise StageTransitionError(f"BACE stage {stage} is already {state}")
    predecessor = BACE_PREDECESSOR[stage]
    if predecessor is None:
        return
    previous = read_bace_stage(layout, predecessor)
    if previous["state"].get("state") != "PASS" or previous["gate"].get("status") != "PASS":
        raise StageTransitionError(
            f"BACE stage {stage} requires {predecessor} state=PASS and gate=PASS"
        )


def resolve_passed_bace_stage_output(
    layout: RuntimeLayout,
    stage: str,
    *,
    required_relative: Sequence[str] = (),
) -> Path:
    """Resolve one immutable upstream output only after its state and gate PASS.

    The path comes from the frozen stage manifest rather than a directory scan.
    It must be absolute, exist below the persistent artifact root, and contain
    every explicitly required file.  This is the only supported B3 -> B4 and
    B4 -> B5 hand-off contract.
    """

    documents = read_bace_stage(layout, stage)
    if documents["state"].get("state") != "PASS":
        raise StageTransitionError(f"BACE stage {stage} state is not PASS")
    if documents["gate"].get("status") != "PASS":
        raise StageTransitionError(f"BACE stage {stage} gate is not PASS")
    manifest = documents["manifest"]
    if manifest.get("status") != "FROZEN":
        raise StageTransitionError(f"BACE stage {stage} manifest is not FROZEN")
    raw_output = manifest.get("expected_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise StageTransitionError(
            f"BACE stage {stage} manifest has no expected_output"
        )
    candidate = Path(raw_output).expanduser()
    if not candidate.is_absolute():
        raise StageTransitionError(
            f"BACE stage {stage} expected_output is not absolute: {candidate}"
        )
    output = candidate.resolve(strict=True)
    if not output.is_dir():
        raise StageTransitionError(
            f"BACE stage {stage} expected_output is not a directory: {output}"
        )
    try:
        output.relative_to(layout.artifacts_dir.resolve(strict=True))
    except ValueError as exc:
        raise StageTransitionError(
            f"BACE stage {stage} expected_output escapes persistent artifacts: {output}"
        ) from exc
    failures = verify_required_outputs(output, required_relative)
    if failures:
        raise StageTransitionError("; ".join(failures))
    return output


def update_bace_stage_state(
    layout: RuntimeLayout,
    stage: str,
    state: str,
    **fields: Any,
) -> dict[str, Any]:
    if state not in STAGE_STATES:
        raise StageTransitionError(f"Unknown BACE stage state: {state}")
    document = read_bace_stage(layout, stage)["state"]
    document.update(fields)
    document["state"] = state
    document["updated_at"] = utc_now()
    atomic_write_json(stage_paths(layout, stage)["state"], document)
    return document


def mark_bace_stage_pass(
    layout: RuntimeLayout,
    *,
    stage: str,
    evidence: Sequence[Path],
    note: str,
) -> None:
    """Manually publish an evidence-bound audit/data stage PASS."""

    if stage not in {"B0_AUDIT", "B1_DATA_READY"}:
        raise StageTransitionError("Manual PASS is restricted to B0_AUDIT/B1_DATA_READY")
    assert_bace_stage_can_start(layout, stage)
    evidence_rows: list[dict[str, str]] = []
    for path in evidence:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise StageTransitionError(f"PASS evidence is missing or empty: {resolved}")
        evidence_rows.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    if not evidence_rows:
        raise StageTransitionError("At least one nonempty evidence file is required")
    now = utc_now()
    atomic_write_json(
        stage_paths(layout, stage)["manifest"],
        {
            "schema_version": 1,
            "dataset": "bace",
            "stage": stage,
            "status": "FROZEN",
            "evidence": evidence_rows,
            "note": note,
            "published_at": now,
        },
    )
    atomic_write_json(
        stage_paths(layout, stage)["gate"],
        {
            "schema_version": 1,
            "dataset": "bace",
            "stage": stage,
            "status": "PASS",
            "checked_at": now,
            "evidence": evidence_rows,
            "reason": None,
        },
    )
    update_bace_stage_state(layout, stage, "PASS", completed_at=now)


def verify_required_outputs(expected_output: Path, required_relative: Sequence[str]) -> list[str]:
    """Return explicit validation failures for a scientific output bundle."""

    failures: list[str] = []
    if not expected_output.exists():
        return [f"expected output is absent: {expected_output}"]
    root = expected_output.resolve(strict=True)
    for value in required_relative:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe required output path: {value}")
            continue
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"required output is missing or empty: {path}")
    return failures


def verify_required_output_alternatives(
    expected_output: Path,
    required_groups: Sequence[Sequence[str]],
) -> list[str]:
    """Require at least one nonempty file from each declared alternative group."""

    failures: list[str] = []
    if not required_groups:
        return failures
    if not expected_output.exists():
        return [f"expected output is absent: {expected_output}"]
    root = expected_output.resolve(strict=True)
    for group in required_groups:
        if not group:
            failures.append("required output alternative group is empty")
            continue
        safe: list[Path] = []
        unsafe = False
        for value in group:
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                failures.append(f"unsafe required output path: {value}")
                unsafe = True
                continue
            safe.append(root / relative)
        if unsafe:
            continue
        if not any(path.is_file() and path.stat().st_size > 0 for path in safe):
            failures.append(
                "none of the required output alternatives is nonempty: "
                + " | ".join(str(path) for path in safe)
            )
    return failures


def verify_required_absolute_outputs(
    required_paths: Sequence[Path], *, allowed_root: Path
) -> list[str]:
    """Require nonempty physical files under one persistent output root.

    Most result evidence belongs below ``expected_output`` and should use
    :func:`verify_required_outputs`.  This narrow companion exists for
    append-only audit artifacts whose externally prescribed absolute path is
    outside that task directory.  Symlinks and paths escaping ``allowed_root``
    fail closed.
    """

    failures: list[str] = []
    root = allowed_root.expanduser().resolve(strict=True)
    for raw_path in required_paths:
        candidate = raw_path.expanduser()
        if not candidate.is_absolute():
            failures.append(f"required absolute output path is relative: {candidate}")
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError):
            failures.append(
                f"required absolute output is absent or outside {root}: {candidate}"
            )
            continue
        if candidate.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
            failures.append(
                f"required absolute output is not a nonempty physical file: {candidate}"
            )
    return failures


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AutoDLRuntimeError(f"Corrupt registry row {line_number}: {path}") from exc
        if not isinstance(value, dict):
            raise AutoDLRuntimeError(f"Registry row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def latest_registry_events(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id", ""))
        if run_id:
            latest[run_id] = dict(row)
    return sorted(latest.values(), key=lambda row: str(row.get("timestamp", "")), reverse=True)
