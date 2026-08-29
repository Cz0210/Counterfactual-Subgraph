"""Minimal persistent continuation sidecar for the active AutoDL main table.

This module deliberately owns no scientific implementation.  It observes four
already-running lines, persists a small fixed queue, and may launch only the
existing TasteMolNet T9 wrapper or one explicitly supplied NeuroSED trainer.
It has no process-termination operation and no matrix publication operation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from typing import Any, TextIO

from src.utils.autodl_runtime import (
    GPUObservation,
    append_jsonl_locked,
    atomic_write_json,
    gpu_lock_available,
    query_gpu_inventory,
    sanitized_environment,
    sha256_file,
)


SPEC_SCHEMA = "autodl_main_table_continuation_spec_v1"
STATE_SCHEMA = "autodl_main_table_continuation_state_v1"
QUEUE_SCHEMA = "autodl_main_table_continuation_queue_v1"
HEARTBEAT_SCHEMA = "autodl_main_table_continuation_heartbeat_v1"
ATTEMPT_SCHEMA = "autodl_main_table_continuation_attempt_v1"
TERMINAL_SCHEMA = "autodl_main_table_continuation_child_terminal_v1"
CONVERGENCE_SCHEMA = "bace_comrecgc_convergence_registration_v1"
RELEASE_BLOCKED_TASKS = ("T6", "T7", "T8")
SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
HASH = re.compile(r"[0-9a-f]{64}")


class ContinuationSidecarError(RuntimeError):
    """The bounded sidecar cannot safely continue."""


@dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    ppid: int
    start_ticks: int
    command: str


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    process_pids: tuple[int, ...]
    project_lock_available: bool

    @property
    def available(self) -> bool:
        return not self.process_pids and self.project_lock_available


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _absolute(value: Any, *, label: str, existing: bool = False) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise ContinuationSidecarError(f"{label} must be absolute: {value!r}")
    try:
        return path.resolve(strict=existing)
    except FileNotFoundError as exc:
        raise ContinuationSidecarError(f"{label} does not exist: {path}") from exc


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuationSidecarError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuationSidecarError(f"{label} must be one JSON object: {path}")
    return dict(value)


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    if SAFE_ID.fullmatch(text) is None:
        raise ContinuationSidecarError(f"{label} is not a safe identifier: {value!r}")
    return text


def _sha256_json(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_process_snapshot(pid: int) -> ProcessSnapshot | None:
    """Read one Linux process identity without sending it any signal."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = stat.rfind(")")
        fields = stat[closing + 2 :].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return ProcessSnapshot(
            pid=pid,
            ppid=int(fields[1]),
            start_ticks=int(fields[19]),
            command=command.decode("utf-8", errors="replace").strip(),
        )
    except (IndexError, OSError, ValueError):
        return None


def read_descendants(pid: int) -> tuple[ProcessSnapshot, ...]:
    """Return current descendants using procfs parent identities only."""

    snapshots: dict[int, ProcessSnapshot] = {}
    proc = Path("/proc")
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        snapshot = read_process_snapshot(int(entry.name))
        if snapshot is not None:
            snapshots[snapshot.pid] = snapshot
    result: list[ProcessSnapshot] = []
    frontier = [pid]
    seen = {pid}
    while frontier:
        parent = frontier.pop(0)
        children = sorted(
            (item for item in snapshots.values() if item.ppid == parent),
            key=lambda item: item.pid,
        )
        for child in children:
            if child.pid in seen:
                continue
            seen.add(child.pid)
            result.append(child)
            frontier.append(child.pid)
    return tuple(result)


def _exact_process_state(
    binding: Mapping[str, Any],
    *,
    process_reader: Callable[[int], ProcessSnapshot | None],
) -> dict[str, Any]:
    pid = int(binding.get("pid", 0))
    expected_ticks = int(binding.get("start_ticks", 0))
    observed = process_reader(pid)
    alive = observed is not None and observed.start_ticks == expected_ticks
    return {
        "pid": pid,
        "expected_start_ticks": expected_ticks,
        "alive": alive,
        "observed": asdict(observed) if observed is not None else None,
    }


def _json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ContinuationSidecarError(f"JSON pointer must start with '/': {pointer}")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise KeyError(pointer)
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            current = current[int(token)]
        else:
            raise KeyError(pointer)
    return current


def _format_tokens(values: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = str(raw)
        for key, replacement in replacements.items():
            value = value.replace("{" + key + "}", replacement)
        if "{" in value or "}" in value:
            raise ContinuationSidecarError(f"unresolved command placeholder: {value}")
        result.append(value)
    return result


def load_continuation_spec(path: str | Path) -> dict[str, Any]:
    spec_path = _absolute(path, label="spec", existing=True)
    value = _json_object(spec_path, label="continuation spec")
    if value.get("schema_version") != SPEC_SCHEMA:
        raise ContinuationSidecarError("unsupported continuation spec schema")
    if value.get("run_gnn_ablation") is not False:
        raise ContinuationSidecarError("RUN_GNN_ABLATION must remain exactly false")
    _safe_id(value.get("controller_id"), label="controller_id")
    for key in ("state_root", "project_root", "runtime_root", "data_root", "python"):
        _absolute(value.get(key), label=key, existing=key in {"project_root", "python"})
    config = _absolute(value.get("config"), label="config", existing=True)
    if not config.is_file():
        raise ContinuationSidecarError("config must be a physical file")
    entrypoint = _absolute(value.get("entrypoint"), label="entrypoint", existing=True)
    if entrypoint.name != "run_main_table_continuation_sidecar.py":
        raise ContinuationSidecarError("entrypoint must be the paired sidecar CLI")
    poll_seconds = value.get("poll_seconds", 60)
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)):
        raise ContinuationSidecarError("poll_seconds must be numeric")
    if float(poll_seconds) < 1 or float(poll_seconds) > 3600:
        raise ContinuationSidecarError("poll_seconds must be in [1, 3600]")

    raw_gpus = value.get("gpus")
    if not isinstance(raw_gpus, list) or len(raw_gpus) != 2:
        raise ContinuationSidecarError("gpus must bind exactly physical GPU0 and GPU1")
    indices: set[int] = set()
    for row in raw_gpus:
        if not isinstance(row, Mapping):
            raise ContinuationSidecarError("each GPU binding must be an object")
        index = row.get("index")
        uuid_value = str(row.get("uuid") or "")
        if isinstance(index, bool) or index not in {0, 1} or not uuid_value.startswith("GPU-"):
            raise ContinuationSidecarError(f"invalid GPU binding: {row}")
        indices.add(int(index))
    if indices != {0, 1}:
        raise ContinuationSidecarError("GPU bindings must be exactly {0,1}")
    _absolute(value.get("lock_root"), label="lock_root")

    aids = value.get("aids_exact")
    if not isinstance(aids, Mapping):
        raise ContinuationSidecarError("aids_exact binding is required")
    if aids.get("state") != "BLOCKED" or aids.get("handover_allowed") is not False:
        raise ContinuationSidecarError("AIDS exact must remain BLOCKED with handover forbidden")
    _safe_id(aids.get("controller_id"), label="aids_exact.controller_id")
    for key in ("controller_pid", "science_pid"):
        raw = aids.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ContinuationSidecarError(f"aids_exact.{key} must be a positive integer")
    _absolute(aids.get("checkpoint"), label="aids_exact.checkpoint")
    if not str(aids.get("blocker") or "").strip():
        raise ContinuationSidecarError("aids_exact.blocker is required")

    observers = value.get("observers")
    if not isinstance(observers, Mapping):
        raise ContinuationSidecarError("observers object is required")
    for name in ("bace_gcf", "bace_globalgce", "bace_comrecgc"):
        if not isinstance(observers.get(name), Mapping):
            raise ContinuationSidecarError(f"observers.{name} is required")
    for name in ("bace_gcf", "bace_globalgce", "bace_comrecgc"):
        binding = observers[name]
        for key in ("pid", "start_ticks"):
            raw = binding.get(key)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ContinuationSidecarError(
                    f"observers.{name}.{key} must be a positive integer"
                )
    markers = observers["bace_gcf"].get("final_markers")
    if not isinstance(markers, list) or not markers:
        raise ContinuationSidecarError("bace_gcf.final_markers must be non-empty")
    for marker in markers:
        _absolute(marker, label="bace_gcf.final_marker")
    comrec = observers["bace_comrecgc"]
    if comrec.get("trigger_step") != 17500:
        raise ContinuationSidecarError("BACE ComRecGC trigger_step must remain 17500")
    if comrec.get("progress_json") is not None:
        _absolute(comrec.get("progress_json"), label="bace_comrecgc.progress_json")
        if not str(comrec.get("progress_pointer") or "").startswith("/"):
            raise ContinuationSidecarError("bace_comrecgc.progress_pointer is required")

    blocked = value.get("blocked_taste")
    if not isinstance(blocked, Mapping) or set(blocked) != set(RELEASE_BLOCKED_TASKS):
        raise ContinuationSidecarError("blocked_taste must contain exactly T6/T7/T8")
    for task in RELEASE_BLOCKED_TASKS:
        row = blocked[task]
        if not isinstance(row, Mapping) or row.get("state") != "BLOCKED_RELEASE":
            raise ContinuationSidecarError(f"{task} must remain BLOCKED_RELEASE")
        if not str(row.get("reason") or "").strip():
            raise ContinuationSidecarError(f"{task} release blocker is required")

    t9 = value.get("t9")
    if not isinstance(t9, Mapping) or t9.get("enabled") is not True:
        raise ContinuationSidecarError("the fixed T9 route must be explicitly enabled")
    wrapper = _absolute(t9.get("wrapper"), label="t9.wrapper", existing=True)
    if wrapper.name != "run_tastemolnet_comrecgc_smoke.sh":
        raise ContinuationSidecarError("T9 must invoke the existing managed wrapper")
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise ContinuationSidecarError("T9 wrapper must be an executable physical file")
    for key in ("stage_parent", "final_parent"):
        _absolute(t9.get(key), label=f"t9.{key}")
    _safe_id(t9.get("run_id_prefix"), label="t9.run_id_prefix")
    fixed_env = t9.get("fixed_environment")
    if not isinstance(fixed_env, Mapping):
        raise ContinuationSidecarError("t9.fixed_environment must be an object")
    expected_authority = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "RUN_GNN_ABLATION": "0",
    }
    for key, expected in expected_authority.items():
        if str(fixed_env.get(key)) != expected:
            raise ContinuationSidecarError(f"T9 authority {key} must equal {expected}")
    required_t9 = {
        "AUTODL_PYTHON",
        "AUTODL_DATA_ROOT",
        "TASTEMOLNET_T2_ADOPTION_ROOT",
        "TASTEMOLNET_T2_ADOPTION_GATE_SHA256",
        "TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256",
        "TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256",
        "TASTEMOLNET_T3_OUTPUT_ROOT",
        "TASTEMOLNET_T4_OUTPUT_ROOT",
        "TASTEMOLNET_TRAIN_CSV",
        "COMRECGC_OFFICIAL_ROOT",
    }
    missing = sorted(key for key in required_t9 if not str(fixed_env.get(key) or ""))
    if missing:
        raise ContinuationSidecarError(f"T9 fixed authority is incomplete: {missing}")
    for generated in (
        "TASTEMOLNET_T9_STAGE_ROOT",
        "TASTEMOLNET_T9_OUTPUT",
        "TASTEMOLNET_T9_RUN_ID",
    ):
        if generated in fixed_env:
            raise ContinuationSidecarError(f"T9 generated value may not be pinned: {generated}")

    neurosed = value.get("neurosed")
    if not isinstance(neurosed, Mapping):
        raise ContinuationSidecarError("neurosed object is required")
    trainer = neurosed.get("trainer_argv")
    if trainer is not None:
        if not isinstance(trainer, list) or not trainer or not all(
            isinstance(item, str) and item for item in trainer
        ):
            raise ContinuationSidecarError("neurosed.trainer_argv must be a non-empty argv")
        combined = "\n".join(trainer) + json.dumps(neurosed.get("fixed_environment", {}))
        if "{attempt_root}" not in combined:
            raise ContinuationSidecarError("NeuroSED trainer must receive its fresh attempt_root")
        _absolute(neurosed.get("label_manifest"), label="neurosed.label_manifest")
        assertions = neurosed.get("label_assertions")
        if not isinstance(assertions, Mapping) or not assertions:
            raise ContinuationSidecarError("NeuroSED label_assertions are required")
        expected_label_contract = {
            "train_success_count": 5000,
            "validation_success_count": 1000,
            "ged_backend": "branch",
            "calibration_loaded": False,
            "test_loaded": False,
        }
        asserted_by_leaf = {
            str(pointer).rsplit("/", 1)[-1]: expected
            for pointer, expected in assertions.items()
        }
        if any(
            asserted_by_leaf.get(key) != expected
            for key, expected in expected_label_contract.items()
        ):
            raise ContinuationSidecarError(
                "NeuroSED labels must assert 5000/1000 branch and no calibration/test"
            )
        marker = str(neurosed.get("success_marker_template") or "")
        if "{attempt_root}" not in marker:
            raise ContinuationSidecarError("NeuroSED success marker must bind attempt_root")
        if not str(neurosed.get("science_process_token") or "").strip():
            raise ContinuationSidecarError("NeuroSED science_process_token is required")
        fixed_neurosed_env = neurosed.get("fixed_environment", {})
        if not isinstance(fixed_neurosed_env, Mapping):
            raise ContinuationSidecarError("NeuroSED fixed_environment must be an object")
        if str(fixed_neurosed_env.get("RUN_GNN_ABLATION", "0")) != "0":
            raise ContinuationSidecarError("NeuroSED may not enable GNN ablation")
    return value


def _attempt_paths(state_root: Path, task: str, attempt_uuid: str) -> dict[str, Path]:
    root = state_root / "attempts" / task.lower() / attempt_uuid
    return {
        "root": root,
        "attempt": root / "attempt.json",
        "terminal": root / "terminal.json",
        "stdout": root / "stdout.log",
        "stderr": root / "stderr.log",
    }


class ContinuationSidecar:
    """One finite, restart-readable queue for the current AutoDL continuation."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        gpu_reader: Callable[[], Sequence[GPUObservation]] = query_gpu_inventory,
        lock_reader: Callable[[Path, str], bool] = gpu_lock_available,
        process_reader: Callable[[int], ProcessSnapshot | None] = read_process_snapshot,
        descendants_reader: Callable[[int], Sequence[ProcessSnapshot]] = read_descendants,
        launcher: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        uuid_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        convergence_auditor: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> None:
        self.spec_path = _absolute(spec_path, label="spec", existing=True)
        self.spec = load_continuation_spec(self.spec_path)
        self.state_root = _absolute(self.spec["state_root"], label="state_root")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.gpu_reader = gpu_reader
        self.lock_reader = lock_reader
        self.process_reader = process_reader
        self.descendants_reader = descendants_reader
        self.launcher = launcher
        self.command_runner = command_runner
        self.uuid_factory = uuid_factory
        # The convergence algorithm is maintained as a separate, read-only
        # library.  This optional hook is the only integration seam: the
        # sidecar persists its result but deliberately performs no handover.
        self.convergence_auditor = convergence_auditor
        self._lock_handle: TextIO | None = None
        self._heartbeat_sequence = self._read_heartbeat_sequence()
        self._last_event_digest: str | None = None
        self.tasks = self._load_or_initialize_tasks()
        self._initialize_control_files()

    def _read_heartbeat_sequence(self) -> int:
        path = self.state_root / "heartbeat.json"
        if not path.is_file():
            return 0
        value = _json_object(path, label="heartbeat")
        sequence = value.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ContinuationSidecarError("persisted heartbeat sequence is invalid")
        return sequence

    @property
    def queue_path(self) -> Path:
        return self.state_root / "queue.json"

    @property
    def state_path(self) -> Path:
        return self.state_root / "state.json"

    @property
    def heartbeat_path(self) -> Path:
        return self.state_root / "heartbeat.json"

    def __enter__(self) -> "ContinuationSidecar":
        lock_path = self.state_root / "controller.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise ContinuationSidecarError("another continuation sidecar owns this root") from exc
        self._lock_handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None

    def _load_or_initialize_tasks(self) -> dict[str, dict[str, Any]]:
        if self.state_path.is_file():
            state = _json_object(self.state_path, label="sidecar state")
            raw = state.get("tasks")
            if isinstance(raw, Mapping):
                return {
                    str(key): dict(value)
                    for key, value in raw.items()
                    if isinstance(value, Mapping)
                }
        now = utc_now()
        tasks: dict[str, dict[str, Any]] = {}
        for task in RELEASE_BLOCKED_TASKS:
            tasks[task] = {
                "state": "BLOCKED_RELEASE",
                "reason": str(self.spec["blocked_taste"][task]["reason"]),
                "launch_attempts": 0,
                "updated_at": now,
            }
        tasks["T9"] = {
            "state": "WAITING_GPU1",
            "launch_attempts": 0,
            "active_attempt_uuid": None,
            "updated_at": now,
        }
        tasks["T10"] = {
            "state": "WAITING_SMOKES",
            "launch_attempts": 0,
            "updated_at": now,
        }
        tasks["NEUROSED"] = {
            "state": "WAITING_INPUT",
            "launch_attempts": 0,
            "active_attempt_uuid": None,
            "updated_at": now,
        }
        return tasks

    def _initialize_control_files(self) -> None:
        spec_sha = sha256_file(self.spec_path)
        receipt_path = self.state_root / "controller_receipt.json"
        if receipt_path.exists():
            receipt = _json_object(receipt_path, label="controller receipt")
            if receipt.get("spec_sha256") != spec_sha:
                raise ContinuationSidecarError("state root is bound to a different spec")
        else:
            atomic_write_json(
                receipt_path,
                {
                    "schema_version": STATE_SCHEMA,
                    "controller_id": self.spec["controller_id"],
                    "created_at": utc_now(),
                    "spec_path": str(self.spec_path),
                    "spec_sha256": spec_sha,
                    "scope": [
                        "READ_ONLY_ACTIVE_LINE_OBSERVATION",
                        "T9_FRESH_RETRIGGER",
                        "NEUROSED_SPEC_DRIVEN_TRAINER",
                    ],
                    "matrix_publication_allowed": False,
                    "process_termination_allowed": False,
                    "run_gnn_ablation": False,
                },
            )
        convergence = self.state_root / "bace_comrecgc_convergence_registration.json"
        if not convergence.exists():
            binding = self.spec["observers"]["bace_comrecgc"]
            atomic_write_json(
                convergence,
                {
                    "schema_version": CONVERGENCE_SCHEMA,
                    "registered_at": utc_now(),
                    "trigger_step": 17500,
                    "status": "DURABLE_PENDING",
                    "pid": binding["pid"],
                    "start_ticks": binding["start_ticks"],
                    "external_convergence_audit_required": True,
                    "sidecar_may_compute_convergence": False,
                    "sidecar_may_stop_worker": False,
                },
            )

    def _gpu_states(self) -> dict[int, GPUState]:
        expected = {int(row["index"]): str(row["uuid"]) for row in self.spec["gpus"]}
        observed = {gpu.index: gpu for gpu in self.gpu_reader()}
        result: dict[int, GPUState] = {}
        lock_root = _absolute(self.spec["lock_root"], label="lock_root")
        for index in (0, 1):
            gpu = observed.get(index)
            if gpu is None or gpu.uuid != expected[index]:
                result[index] = GPUState(index, expected[index], (-1,), False)
                continue
            result[index] = GPUState(
                index=index,
                uuid=gpu.uuid,
                process_pids=tuple(sorted(process.pid for process in gpu.processes)),
                project_lock_available=bool(self.lock_reader(lock_root, gpu.uuid)),
            )
        return result

    def _observe_active_lines(self) -> dict[str, Any]:
        observers = self.spec["observers"]
        gcf_process = _exact_process_state(
            observers["bace_gcf"], process_reader=self.process_reader
        )
        markers = [
            {
                "path": str(_absolute(path, label="bace_gcf.final_marker")),
                "exists": _absolute(path, label="bace_gcf.final_marker").is_file(),
            }
            for path in observers["bace_gcf"]["final_markers"]
        ]
        gcf_state = "FINAL_PASS_OBSERVED" if all(row["exists"] for row in markers) else (
            "RUNNING" if gcf_process["alive"] else "NOT_FINAL_PROCESS_NOT_ALIVE"
        )
        global_process = _exact_process_state(
            observers["bace_globalgce"], process_reader=self.process_reader
        )
        comrec_binding = observers["bace_comrecgc"]
        comrec_process = _exact_process_state(
            comrec_binding, process_reader=self.process_reader
        )
        progress: int | None = None
        progress_error: str | None = None
        progress_path_raw = comrec_binding.get("progress_json")
        if progress_path_raw:
            progress_path = _absolute(progress_path_raw, label="bace_comrecgc.progress_json")
            if progress_path.is_file():
                try:
                    raw = _json_pointer(
                        _json_object(progress_path, label="BACE ComRecGC progress"),
                        str(comrec_binding["progress_pointer"]),
                    )
                    if isinstance(raw, bool):
                        raise ValueError("boolean is not progress")
                    progress = int(raw)
                except (ContinuationSidecarError, KeyError, TypeError, ValueError) as exc:
                    progress_error = f"{type(exc).__name__}: {exc}"
        registration_path = self.state_root / "bace_comrecgc_convergence_registration.json"
        registration = _json_object(registration_path, label="convergence registration")
        registration["observed_at"] = utc_now()
        registration["latest_observed_progress"] = progress
        registration["progress_error"] = progress_error
        if progress is not None and progress >= 17500:
            registration["status"] = "READY_FOR_EXTERNAL_CONVERGENCE_CHECK"
            if self.convergence_auditor is not None:
                hook_input = {
                    "trigger_step": 17500,
                    "latest_observed_progress": progress,
                    "pid": comrec_binding["pid"],
                    "start_ticks": comrec_binding["start_ticks"],
                    "progress_json": comrec_binding.get("progress_json"),
                    "progress_pointer": comrec_binding.get("progress_pointer"),
                    "sidecar_may_stop_worker": False,
                }
                hook_path = (
                    self.state_root
                    / "bace_comrecgc_convergence_audits"
                    / f"step-{progress}.json"
                )
                if hook_path.is_file():
                    hook_receipt = _json_object(
                        hook_path, label="BACE convergence hook receipt"
                    )
                    hook_result = hook_receipt.get("result")
                    if not isinstance(hook_result, Mapping):
                        raise ContinuationSidecarError(
                            "BACE convergence hook receipt has no result mapping"
                        )
                else:
                    hook_result = self.convergence_auditor(hook_input)
                    if not isinstance(hook_result, Mapping):
                        raise ContinuationSidecarError(
                            "BACE convergence audit hook must return one mapping"
                        )
                    atomic_write_json(
                        hook_path,
                        {
                            "schema_version": CONVERGENCE_SCHEMA,
                            "written_at": utc_now(),
                            "input": hook_input,
                            "result": dict(hook_result),
                            "handover_implemented_by_sidecar": False,
                        },
                    )
                registration["hook_result_status"] = hook_result.get("status")
                registration["hook_result_path"] = str(hook_path)
                registration["status"] = (
                    "AUDIT_PASS_AWAITING_SEPARATE_HANDOVER"
                    if hook_result.get("status") == "PASS"
                    else "AUDIT_CONTINUE"
                )
        else:
            registration["status"] = "DURABLE_PENDING"
        registration["sidecar_may_compute_convergence"] = False
        registration["sidecar_may_stop_worker"] = False
        atomic_write_json(registration_path, registration)
        aids = dict(self.spec["aids_exact"])
        checkpoint = _absolute(aids["checkpoint"], label="aids_exact.checkpoint")
        return {
            "aids_exact": {
                "state": "BLOCKED",
                "blocker": aids["blocker"],
                "controller_id": aids["controller_id"],
                "controller_pid": aids["controller_pid"],
                "science_pid": aids["science_pid"],
                "checkpoint": str(checkpoint),
                "checkpoint_exists": checkpoint.is_file(),
                "handover_allowed": False,
                "action": "OBSERVE_ONLY",
            },
            "bace_gcf": {
                "state": gcf_state,
                "process": gcf_process,
                "final_markers": markers,
                "action": "OBSERVE_ONLY",
            },
            "bace_globalgce": {
                "state": "ALIVE" if global_process["alive"] else "NOT_ALIVE",
                "process": global_process,
                "action": "OBSERVE_ONLY",
            },
            "bace_comrecgc": {
                "state": registration["status"],
                "process": comrec_process,
                "latest_observed_progress": progress,
                "registration": str(registration_path),
                "external_convergence_audit_required": True,
                "sidecar_may_stop_worker": False,
                "action": "OBSERVE_ONLY",
            },
        }

    def _attempt_record(self, task: str, attempt_uuid: str) -> dict[str, Any]:
        path = _attempt_paths(self.state_root, task, attempt_uuid)["attempt"]
        return _json_object(path, label=f"{task} attempt")

    def _write_attempt(self, task: str, attempt_uuid: str, value: Mapping[str, Any]) -> None:
        paths = _attempt_paths(self.state_root, task, attempt_uuid)
        paths["root"].mkdir(parents=True, exist_ok=False)
        atomic_write_json(paths["attempt"], value)

    def _child_supervisor_argv(
        self, *, task: str, terminal_path: Path, command: Sequence[str]
    ) -> list[str]:
        return [
            str(self.spec["python"]),
            "-B",
            str(self.spec["entrypoint"]),
            "--config",
            str(self.spec["config"]),
            "_child",
            "--task",
            task,
            "--terminal-receipt",
            str(terminal_path),
            "--",
            *command,
        ]

    def _spawn(
        self,
        *,
        task: str,
        attempt_uuid: str,
        command: Sequence[str],
        environment: Mapping[str, str],
    ) -> ProcessSnapshot:
        paths = _attempt_paths(self.state_root, task, attempt_uuid)
        supervisor = self._child_supervisor_argv(
            task=task, terminal_path=paths["terminal"], command=command
        )
        env = sanitized_environment()
        env.update({str(key): str(value) for key, value in environment.items()})
        env["RUN_GNN_ABLATION"] = "0"
        with paths["stdout"].open("a", encoding="utf-8") as stdout, paths[
            "stderr"
        ].open("a", encoding="utf-8") as stderr:
            process = self.launcher(
                supervisor,
                cwd=str(self.spec["project_root"]),
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
        snapshot = self.process_reader(int(process.pid))
        if snapshot is None:
            # A preflight refusal can finish before procfs is reopened. Keep
            # its exact PID and let the durable terminal receipt decide it on
            # the next tick; start_ticks=0 is never treated as a live identity.
            snapshot = ProcessSnapshot(
                pid=int(process.pid),
                ppid=0,
                start_ticks=0,
                command=" ".join(supervisor),
            )
        return snapshot

    def _fresh_uuid(self, task: str, extra_paths: Callable[[str], Sequence[Path]]) -> str:
        for _ in range(32):
            value = str(self.uuid_factory())
            if not re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                value,
            ):
                raise ContinuationSidecarError("uuid_factory did not return UUIDv4")
            paths = [
                _attempt_paths(self.state_root, task, value)["root"],
                *extra_paths(value),
            ]
            if not any(path.exists() or path.is_symlink() for path in paths):
                return value
        raise ContinuationSidecarError(f"unable to allocate a fresh {task} UUID")

    def _launch_t9(self) -> None:
        t9 = self.spec["t9"]
        stage_parent = _absolute(t9["stage_parent"], label="t9.stage_parent")
        final_parent = _absolute(t9["final_parent"], label="t9.final_parent")
        prefix = str(t9["run_id_prefix"])

        def science_paths(value: str) -> Sequence[Path]:
            return (
                stage_parent / f"stage-m500-{value}",
                final_parent / f"smoke-m500-{value}",
            )

        attempt_uuid = self._fresh_uuid("T9", science_paths)
        stage_root, final_root = science_paths(attempt_uuid)
        run_id = f"{prefix}-{attempt_uuid}"
        paths = _attempt_paths(self.state_root, "T9", attempt_uuid)
        attempt = {
            "schema_version": ATTEMPT_SCHEMA,
            "task": "T9",
            "attempt_uuid": attempt_uuid,
            "uuid_version": 4,
            "created_at": utc_now(),
            "state": "STARTING_PREFLIGHT",
            "reuse_allowed": False,
            "preflight_rc75_policy": "ABANDON_ALL_IDENTITIES_AND_ALLOCATE_NEW_UUID",
            "stage_root": str(stage_root),
            "final_root": str(final_root),
            "run_id": run_id,
            "terminal_receipt": str(paths["terminal"]),
            "gpu_index": 1,
            "supervisor_process": None,
            "science_child_pid": None,
        }
        self._write_attempt("T9", attempt_uuid, attempt)
        environment = {
            **{str(key): str(value) for key, value in t9["fixed_environment"].items()},
            "TASTEMOLNET_T9_STAGE_ROOT": str(stage_root),
            "TASTEMOLNET_T9_OUTPUT": str(final_root),
            "TASTEMOLNET_T9_RUN_ID": run_id,
            "RUN_GNN_ABLATION": "0",
        }
        snapshot = self._spawn(
            task="T9",
            attempt_uuid=attempt_uuid,
            command=[str(t9["wrapper"])],
            environment=environment,
        )
        attempt["supervisor_process"] = asdict(snapshot)
        atomic_write_json(paths["attempt"], attempt)
        task = self.tasks["T9"]
        task.update(
            {
                "state": "STARTING_PREFLIGHT",
                "launch_attempts": int(task.get("launch_attempts", 0)) + 1,
                "active_attempt_uuid": attempt_uuid,
                "stage_root": str(stage_root),
                "final_root": str(final_root),
                "run_id": run_id,
                "supervisor_pid": snapshot.pid,
                "supervisor_start_ticks": snapshot.start_ticks,
                "science_child_pid": None,
                "updated_at": utc_now(),
            }
        )

    def _labels_ready(self) -> tuple[bool, str]:
        neurosed = self.spec["neurosed"]
        if neurosed.get("trainer_argv") is None:
            return False, "trainer_argv is absent from the immutable spec"
        path = _absolute(neurosed["label_manifest"], label="neurosed.label_manifest")
        if not path.is_file() or path.is_symlink():
            return False, f"label manifest is not a physical file: {path}"
        expected_sha = neurosed.get("label_manifest_sha256")
        if expected_sha is not None:
            expected = str(expected_sha).lower()
            if HASH.fullmatch(expected) is None or sha256_file(path) != expected:
                return False, "label manifest SHA-256 mismatch"
        try:
            payload = _json_object(path, label="NeuroSED label manifest")
            for pointer, expected in neurosed["label_assertions"].items():
                if _json_pointer(payload, str(pointer)) != expected:
                    return False, f"label assertion failed: {pointer}"
        except (ContinuationSidecarError, KeyError) as exc:
            return False, f"label manifest check failed: {type(exc).__name__}: {exc}"
        return True, "fixed-budget label manifest PASS"

    def _launch_neurosed(self, gpu: GPUState) -> None:
        neurosed = self.spec["neurosed"]
        attempt_parent = _absolute(
            neurosed.get("attempt_parent") or (self.state_root / "neurosed-runs"),
            label="neurosed.attempt_parent",
        )

        def science_paths(value: str) -> Sequence[Path]:
            return (attempt_parent / value,)

        attempt_uuid = self._fresh_uuid("NEUROSED", science_paths)
        attempt_root = science_paths(attempt_uuid)[0]
        replacements = {
            "attempt_uuid": attempt_uuid,
            "attempt_root": str(attempt_root),
            "gpu_index": str(gpu.index),
            "gpu_uuid": gpu.uuid,
        }
        trainer = _format_tokens(neurosed["trainer_argv"], replacements)
        success_marker = _format_tokens(
            [str(neurosed["success_marker_template"])], replacements
        )[0]
        run_id = f"taste-neurosed-{attempt_uuid}"
        paths = _attempt_paths(self.state_root, "NEUROSED", attempt_uuid)
        attempt = {
            "schema_version": ATTEMPT_SCHEMA,
            "task": "NEUROSED",
            "attempt_uuid": attempt_uuid,
            "uuid_version": 4,
            "created_at": utc_now(),
            "state": "STARTING",
            "reuse_allowed": False,
            "attempt_root": str(attempt_root),
            "success_marker": success_marker,
            "gpu_index": gpu.index,
            "gpu_uuid": gpu.uuid,
            "run_id": run_id,
            "terminal_receipt": str(paths["terminal"]),
            "trainer_argv": trainer,
            "supervisor_process": None,
            "science_child_pid": None,
        }
        self._write_attempt("NEUROSED", attempt_uuid, attempt)
        gpu_lock_command = [
            str(self.spec["python"]),
            "-B",
            str(Path(self.spec["project_root"]) / "scripts/autodl/gpu_lock.py"),
            "--project-root",
            str(self.spec["project_root"]),
            "--data-root",
            str(self.spec["data_root"]),
            "--config",
            str(self.spec["config"]),
            "run",
            "--gpu-index",
            str(gpu.index),
            "--gpu-uuid",
            gpu.uuid,
            "--run-id",
            run_id,
            "--",
            *trainer,
        ]
        environment = {
            **{
                str(key): _format_tokens([str(value)], replacements)[0]
                for key, value in neurosed.get("fixed_environment", {}).items()
            },
            "RUN_GNN_ABLATION": "0",
        }
        snapshot = self._spawn(
            task="NEUROSED",
            attempt_uuid=attempt_uuid,
            command=gpu_lock_command,
            environment=environment,
        )
        attempt["supervisor_process"] = asdict(snapshot)
        atomic_write_json(paths["attempt"], attempt)
        task = self.tasks["NEUROSED"]
        task.update(
            {
                "state": "STARTING",
                "launch_attempts": int(task.get("launch_attempts", 0)) + 1,
                "active_attempt_uuid": attempt_uuid,
                "attempt_root": str(attempt_root),
                "gpu_index": gpu.index,
                "gpu_uuid": gpu.uuid,
                "supervisor_pid": snapshot.pid,
                "supervisor_start_ticks": snapshot.start_ticks,
                "science_child_pid": None,
                "updated_at": utc_now(),
            }
        )

    def _refresh_attempt(self, task_name: str) -> None:
        task = self.tasks[task_name]
        attempt_uuid = task.get("active_attempt_uuid")
        if not attempt_uuid:
            return
        attempt = self._attempt_record(task_name, str(attempt_uuid))
        paths = _attempt_paths(self.state_root, task_name, str(attempt_uuid))
        terminal: dict[str, Any] | None = None
        if paths["terminal"].is_file():
            terminal = _json_object(paths["terminal"], label=f"{task_name} terminal")
            if terminal.get("schema_version") != TERMINAL_SCHEMA:
                raise ContinuationSidecarError(f"invalid {task_name} terminal schema")
        supervisor = attempt.get("supervisor_process")
        live = False
        descendants: Sequence[ProcessSnapshot] = ()
        if isinstance(supervisor, Mapping):
            snapshot = self.process_reader(int(supervisor.get("pid", 0)))
            live = (
                snapshot is not None
                and snapshot.start_ticks == int(supervisor.get("start_ticks", -1))
            )
            if live:
                descendants = self.descendants_reader(snapshot.pid)
        marker = (
            "tastemolnet_t9_managed_runner_v2.py"
            if task_name == "T9"
            else str(self.spec["neurosed"].get("science_process_token") or "")
        )
        science = next((row for row in descendants if marker and marker in row.command), None)
        attempt["last_observed_at"] = utc_now()
        attempt["descendant_pids"] = [row.pid for row in descendants]
        if science is not None:
            attempt["science_child_pid"] = science.pid
            attempt["science_child_start_ticks"] = science.start_ticks
            attempt["state"] = "RUNNING"
            task["science_child_pid"] = science.pid
            task["science_child_start_ticks"] = science.start_ticks
            task["state"] = "RUNNING"
        elif live:
            attempt["state"] = "STARTING_PREFLIGHT" if task_name == "T9" else "STARTING"
            task["state"] = attempt["state"]

        if terminal is not None:
            rc = terminal.get("returncode")
            if isinstance(rc, bool) or not isinstance(rc, int):
                raise ContinuationSidecarError(f"invalid {task_name} terminal returncode")
            attempt["terminal"] = terminal
            if task_name == "T9" and rc == 75:
                attempt["state"] = "ABANDONED_PREFLIGHT_RC75"
                attempt["reuse_allowed"] = False
                task.update(
                    {
                        "state": "WAITING_GPU1",
                        "last_abandoned_attempt_uuid": attempt_uuid,
                        "last_terminal_returncode": 75,
                        "active_attempt_uuid": None,
                        "science_child_pid": None,
                    }
                )
            elif rc != 0:
                attempt["state"] = "FAILED_TERMINAL"
                task.update(
                    {
                        "state": "FAILED_TERMINAL",
                        "last_terminal_returncode": rc,
                        "active_attempt_uuid": None,
                    }
                )
            elif task_name == "T9":
                final_root = Path(str(attempt["final_root"]))
                validation = self.command_runner(
                    [
                        str(self.spec["python"]),
                        "-B",
                        str(
                            Path(self.spec["project_root"])
                            / "scripts/run_tastemolnet_comrecgc_smoke.py"
                        ),
                        "--config",
                        str(self.spec["config"]),
                        "--output-dir",
                        str(final_root),
                        "--validate-only",
                    ],
                    cwd=str(self.spec["project_root"]),
                    env={**sanitized_environment(), "RUN_GNN_ABLATION": "0"},
                    text=True,
                    capture_output=True,
                    check=False,
                )
                atomic_write_json(
                    paths["root"] / "final_validation.json",
                    {
                        "returncode": int(validation.returncode),
                        "stdout": validation.stdout,
                        "stderr": validation.stderr,
                        "validated_at": utc_now(),
                    },
                )
                if validation.returncode == 0:
                    attempt["state"] = "PASS"
                    task.update({"state": "PASS", "active_attempt_uuid": None})
                else:
                    attempt["state"] = "FAILED_FINAL_VALIDATION"
                    task.update(
                        {
                            "state": "FAILED_FINAL_VALIDATION",
                            "active_attempt_uuid": None,
                        }
                    )
            else:
                marker_path = Path(str(attempt["success_marker"]))
                if marker_path.is_file() and not marker_path.is_symlink():
                    attempt["state"] = "PASS"
                    task.update({"state": "PASS", "active_attempt_uuid": None})
                else:
                    attempt["state"] = "FAILED_SUCCESS_MARKER_MISSING"
                    task.update(
                        {
                            "state": "FAILED_SUCCESS_MARKER_MISSING",
                            "active_attempt_uuid": None,
                        }
                    )
        elif not live:
            attempt["state"] = "BLOCKED_LOST_TERMINAL_RECEIPT"
            task.update(
                {
                    "state": "BLOCKED_LOST_TERMINAL_RECEIPT",
                    "active_attempt_uuid": None,
                }
            )
        task["updated_at"] = utc_now()
        atomic_write_json(paths["attempt"], attempt)

    def _write_state(
        self, *, observations: Mapping[str, Any], gpus: Mapping[int, GPUState]
    ) -> dict[str, Any]:
        now = utc_now()
        payload = {
            "schema_version": STATE_SCHEMA,
            "controller_id": self.spec["controller_id"],
            "written_at": now,
            "controller_pid": os.getpid(),
            "run_gnn_ablation": False,
            "tasks": self.tasks,
            "observations": observations,
            "gpus": {
                str(index): asdict(value) | {"available": value.available}
                for index, value in gpus.items()
            },
            "process_termination_allowed": False,
            "matrix_publication_allowed": False,
        }
        atomic_write_json(self.state_path, payload)
        queue = {
            "schema_version": QUEUE_SCHEMA,
            "controller_id": self.spec["controller_id"],
            "written_at": now,
            "order": ["NEUROSED", "T9", "T6", "T7", "T8", "T10"],
            "tasks": self.tasks,
            "fixed_policy": {
                "first_free_gpu": "NEUROSED_IF_LABELS_AND_ARGV_READY",
                "gpu1_after_neurosed": "T9_FRESH_UUID",
                "release_blocked_never_invoked": list(RELEASE_BLOCKED_TASKS),
                "run_gnn_ablation": False,
            },
        }
        atomic_write_json(self.queue_path, queue)
        self._heartbeat_sequence += 1
        heartbeat = {
            "schema_version": HEARTBEAT_SCHEMA,
            "controller_id": self.spec["controller_id"],
            "pid": os.getpid(),
            "sequence": self._heartbeat_sequence,
            "written_at": now,
            "state": "RUNNING",
            "active_tasks": [
                name
                for name, task in self.tasks.items()
                if task.get("state") in {"STARTING", "STARTING_PREFLIGHT", "RUNNING"}
            ],
            "run_gnn_ablation": False,
        }
        atomic_write_json(self.heartbeat_path, heartbeat)
        event_value = {
            "tasks": {
                name: {
                    key: task.get(key)
                    for key in (
                        "state",
                        "active_attempt_uuid",
                        "last_abandoned_attempt_uuid",
                        "science_child_pid",
                    )
                }
                for name, task in self.tasks.items()
            },
            "observation_states": {
                name: {
                    "state": value.get("state"),
                    "latest_observed_progress": value.get(
                        "latest_observed_progress"
                    ),
                }
                for name, value in observations.items()
            },
            "gpus": {str(index): asdict(value) for index, value in gpus.items()},
        }
        digest = _sha256_json(event_value)
        if digest != self._last_event_digest:
            append_jsonl_locked(
                self.state_root / "events.jsonl",
                {"written_at": now, "digest": digest, **event_value},
            )
            self._last_event_digest = digest
        return payload

    def tick(self) -> dict[str, Any]:
        for task in RELEASE_BLOCKED_TASKS:
            self.tasks[task].update(
                {
                    "state": "BLOCKED_RELEASE",
                    "reason": self.spec["blocked_taste"][task]["reason"],
                    "launch_attempts": 0,
                    "updated_at": utc_now(),
                }
            )
        self._refresh_attempt("NEUROSED")
        self._refresh_attempt("T9")
        observations = self._observe_active_lines()
        gpus = self._gpu_states()

        neurosed_task = self.tasks["NEUROSED"]
        if neurosed_task.get("active_attempt_uuid") is None and neurosed_task.get("state") not in {
            "PASS",
            "FAILED_TERMINAL",
            "FAILED_SUCCESS_MARKER_MISSING",
            "BLOCKED_LOST_TERMINAL_RECEIPT",
        }:
            labels_ready, reason = self._labels_ready()
            if not labels_ready:
                neurosed_task.update(
                    {
                        "state": "WAITING_INPUT"
                        if self.spec["neurosed"].get("trainer_argv") is None
                        else "WAITING_LABELS",
                        "reason": reason,
                        "updated_at": utc_now(),
                    }
                )
            else:
                selected = next((gpus[index] for index in (0, 1) if gpus[index].available), None)
                if selected is None:
                    neurosed_task.update(
                        {
                            "state": "WAITING_GPU",
                            "reason": "GPU0/GPU1 unavailable",
                            "updated_at": utc_now(),
                        }
                    )
                else:
                    self._launch_neurosed(selected)
                    gpus[selected.index] = GPUState(
                        selected.index,
                        selected.uuid,
                        (self.tasks["NEUROSED"]["supervisor_pid"],),
                        False,
                    )

        t9_task = self.tasks["T9"]
        if t9_task.get("active_attempt_uuid") is None and t9_task.get("state") not in {
            "PASS",
            "FAILED_TERMINAL",
            "FAILED_FINAL_VALIDATION",
            "BLOCKED_LOST_TERMINAL_RECEIPT",
        }:
            if gpus[1].available:
                self._launch_t9()
                gpus[1] = GPUState(1, gpus[1].uuid, (self.tasks["T9"]["supervisor_pid"],), False)
            else:
                t9_task.update(
                    {
                        "state": "WAITING_GPU1",
                        "reason": "GPU1 process or project lock is active",
                        "updated_at": utc_now(),
                    }
                )
        return self._write_state(observations=observations, gpus=gpus)

    def run(self, *, once: bool = False) -> int:
        while True:
            self.tick()
            if once:
                return 0
            time.sleep(float(self.spec.get("poll_seconds", 60)))


def run_child_with_terminal_receipt(
    *, task: str, terminal_receipt: str | Path, command: Sequence[str]
) -> int:
    """Run one sidecar-owned child and durably record its exact return code."""

    if task not in {"T9", "NEUROSED"}:
        raise ContinuationSidecarError(f"unsupported child task: {task}")
    receipt_path = _absolute(terminal_receipt, label="terminal_receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ContinuationSidecarError(f"fresh terminal receipt required: {receipt_path}")
    argv = list(command)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise ContinuationSidecarError("child command is empty")
    started = utc_now()
    try:
        completed = subprocess.run(argv, check=False)
        returncode = int(completed.returncode)
        launch_error = None
    except OSError as exc:
        returncode = 127
        launch_error = f"{type(exc).__name__}: {exc}"
    atomic_write_json(
        receipt_path,
        {
            "schema_version": TERMINAL_SCHEMA,
            "task": task,
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": utc_now(),
            "returncode": returncode,
            "command_sha256": hashlib.sha256(
                json.dumps(argv, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "launch_error": launch_error,
        },
    )
    return returncode


def read_sidecar_status(state_root: str | Path) -> dict[str, Any]:
    root = _absolute(state_root, label="state_root", existing=True)
    result: dict[str, Any] = {}
    for name in (
        "controller_receipt.json",
        "queue.json",
        "state.json",
        "heartbeat.json",
        "bace_comrecgc_convergence_registration.json",
    ):
        path = root / name
        result[name] = _json_object(path, label=name) if path.is_file() else None
    return result
