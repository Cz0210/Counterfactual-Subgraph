"""Resource-safe continuation policy for the one authorized Mut worker.

The helpers in this module do not launch science by themselves.  They make
the existing dataset-specific Mut adoption worker safe to resume while the
Taste ComRecGC job is protected:

* exactly one Mut writer may exist;
* the continuation inherits a two-CPU, low-priority execution policy; and
* a protected-task slowdown is actionable only after a comparable 20-minute
  window and corroborating memory or I/O contention evidence.

Checkpoint, fsync/materialization, and otherwise active-but-coarse progress
windows are deliberately excluded from throughput comparisons.  This avoids
mistaking a legitimate T14 checkpoint for interference from Mut.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "mut_throttled_continuation_policy_v1"
ROBUST_GATE_SCHEMA = "mut_robust_protected_throughput_gate_v1"
BASELINE_SECONDS = 1_800
EVALUATION_SECONDS = 1_200
MAXIMUM_SLOWDOWN = 0.15

_CHECKPOINT_PHASE_TOKENS = (
    "checkpoint",
    "commit",
    "flush",
    "fsync",
    "rename",
    "materializ",
    "serialize",
    "snapshot",
    "db_flush",
)

# These are the only command entrypoints which can write Mut ComRecGC
# scientific state in the current recovery route.  The scan is intentionally
# narrow: generic Python processes and read-only evaluators are not writers.
_MUT_WRITER_ENTRYPOINTS = frozenset(
    {
        "run_mut_trace_on_adoption_worker.py",
        "run_mut_checkpoint_instrumentation_equivalence.py",
        "run_mut_trace_mode_equivalence.py",
        "run_mut_fast_accurate_v2.py",
        "run_mut_throttled_continuation_v1.py",
    }
)


class MutContinuationPolicyError(RuntimeError):
    """The continuation ownership or resource policy failed closed."""


@dataclass(frozen=True)
class MutThrottlePolicy:
    workers: int = 2
    nice: int = 10
    ionice_class: int = 2
    ionice_priority: int = 7
    prefetch: int = 1
    baseline_seconds: int = BASELINE_SECONDS
    evaluation_seconds: int = EVALUATION_SECONDS
    maximum_slowdown: float = MAXIMUM_SLOWDOWN

    def validate(self) -> "MutThrottlePolicy":
        if (
            self.workers != 2
            or self.nice != 10
            or self.ionice_class != 2
            or self.ionice_priority != 7
            or self.prefetch != 1
            or self.baseline_seconds != BASELINE_SECONDS
            or self.evaluation_seconds != EVALUATION_SECONDS
            or self.maximum_slowdown != MAXIMUM_SLOWDOWN
        ):
            raise MutContinuationPolicyError(
                "Mut throttling policy differs from the authorized contract"
            )
        return self

    def as_receipt(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux cpulist syntax (for example ``0-3,8,10-11``)."""

    cpus: set[int] = set()
    for token in value.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start < 0 or stop < start:
                raise MutContinuationPolicyError(f"Invalid CPU range: {token}")
            cpus.update(range(start, stop + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise MutContinuationPolicyError(f"Invalid CPU ID: {token}")
            cpus.add(cpu)
    if not cpus:
        raise MutContinuationPolicyError("Current cpuset is empty")
    return tuple(sorted(cpus))


def allowed_cpus(proc_root: Path = Path("/proc"), pid: int | str = "self") -> tuple[int, ...]:
    status = (proc_root / str(pid) / "status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return parse_cpu_list(line.split(":", 1)[1])
    raise MutContinuationPolicyError("Cpus_allowed_list is absent")


def read_cpu_times(proc_stat: Path = Path("/proc/stat")) -> dict[int, tuple[int, int]]:
    """Return ``cpu -> (busy_ticks, total_ticks)`` without changing affinity."""

    result: dict[int, tuple[int, int]] = {}
    for line in proc_stat.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu\d+", fields[0]):
            continue
        values = [int(item) for item in fields[1:]]
        if len(values) < 4:
            continue
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        result[int(fields[0][3:])] = (total - idle, total)
    if not result:
        raise MutContinuationPolicyError("No per-CPU counters found in /proc/stat")
    return result


def select_two_least_busy_cpus(
    before: Mapping[int, tuple[int, int]],
    after: Mapping[int, tuple[int, int]],
    *,
    candidates: Sequence[int],
    sibling_groups: Mapping[int, Sequence[int]] | None = None,
) -> tuple[int, int]:
    """Select two stable least-busy CPUs from the current allowed cpuset."""

    rows: list[tuple[float, int]] = []
    for cpu in sorted(set(int(item) for item in candidates)):
        if cpu not in before or cpu not in after:
            continue
        busy_delta = int(after[cpu][0]) - int(before[cpu][0])
        total_delta = int(after[cpu][1]) - int(before[cpu][1])
        if busy_delta < 0 or total_delta <= 0 or busy_delta > total_delta:
            continue
        rows.append((busy_delta / total_delta, cpu))
    rows.sort(key=lambda item: (item[0], item[1]))
    if len(rows) < 2:
        raise MutContinuationPolicyError(
            "Fewer than two stable CPU counters are available in the cpuset"
        )
    first = rows[0][1]
    siblings = set(int(item) for item in (sibling_groups or {}).get(first, (first,)))
    for _, second in rows[1:]:
        if second not in siblings:
            return first, second
    raise MutContinuationPolicyError(
        "No two non-SMT-sibling CPUs are available in the cpuset"
    )


def read_thread_sibling_groups(
    sys_cpu_root: Path = Path("/sys/devices/system/cpu"),
    *,
    candidates: Sequence[int],
) -> dict[int, tuple[int, ...]]:
    groups: dict[int, tuple[int, ...]] = {}
    for cpu in sorted(set(int(item) for item in candidates)):
        path = sys_cpu_root / f"cpu{cpu}" / "topology/thread_siblings_list"
        try:
            siblings = parse_cpu_list(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise MutContinuationPolicyError(
                f"Cannot verify SMT siblings for CPU {cpu}"
            ) from exc
        groups[cpu] = siblings
    return groups


def sample_and_select_two_cpus(
    *,
    proc_root: Path = Path("/proc"),
    sample_seconds: float = 1.0,
) -> tuple[int, int]:
    candidates = allowed_cpus(proc_root)
    sibling_groups = read_thread_sibling_groups(candidates=candidates)
    before = read_cpu_times(proc_root / "stat")
    time.sleep(sample_seconds)
    after = read_cpu_times(proc_root / "stat")
    return select_two_least_busy_cpus(
        before,
        after,
        candidates=candidates,
        sibling_groups=sibling_groups,
    )


def throttled_environment(
    base: Mapping[str, str], policy: MutThrottlePolicy | None = None
) -> dict[str, str]:
    policy = (policy or MutThrottlePolicy()).validate()
    result = dict(base)
    result.update(
        {
            "MUT_EXACT_WORKERS": str(policy.workers),
            "MUT_CPU_WORKERS": str(policy.workers),
            "MUT_PREFETCH": str(policy.prefetch),
            "MUT_PREFETCH_FACTOR": str(policy.prefetch),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return result


def _cmdline(proc_root: Path, pid: int) -> tuple[str, ...]:
    try:
        return tuple(
            item.decode("utf-8", errors="replace")
            for item in (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0")
            if item
        )
    except OSError:
        return ()


def find_live_mut_writers(
    proc_root: Path = Path("/proc"), *, excluded_pids: Sequence[int] = ()
) -> list[dict[str, Any]]:
    excluded = {int(item) for item in excluded_pids}
    writers: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        arguments = _cmdline(proc_root, int(entry.name))
        names = {Path(item).name for item in arguments}
        matched = sorted(names & _MUT_WRITER_ENTRYPOINTS)
        if not matched:
            # The shared upstream generation entrypoint is a writer only when
            # its command is explicitly bound to Mutagenicity.
            if (
                "run_generation.py" not in names
                or not any("mutagenicity" in item.lower() for item in arguments)
            ):
                continue
            matched = ["run_generation.py"]
        writers.append(
            {
                "pid": int(entry.name),
                "entrypoints": matched,
                "command": list(arguments),
            }
        )
    return sorted(writers, key=lambda row: int(row["pid"]))


def process_descendants(proc_root: Path, root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent == root_pid or parent in descendants:
                if pid not in descendants:
                    descendants.add(pid)
                    changed = True
    return descendants


def assert_single_continuation_owner(
    proc_root: Path = Path("/proc"),
    *,
    current_pid: int | None = None,
    attached_controller_pid: int | None = None,
) -> dict[str, Any]:
    excluded = [
        pid
        for pid in (current_pid, attached_controller_pid)
        if pid is not None
    ]
    if current_pid is not None:
        excluded.extend(process_descendants(proc_root, current_pid))
    writers = find_live_mut_writers(proc_root, excluded_pids=excluded)
    if writers:
        raise MutContinuationPolicyError(
            "A competing live Mut scientific writer already exists: "
            + ",".join(str(row["pid"]) for row in writers)
        )
    return {
        "schema_version": "mut_single_continuation_owner_preflight_v1",
        "status": "PASS",
        "current_pid": current_pid,
        "attached_controller_pid": attached_controller_pid,
        "competing_writer_count": 0,
        "competing_writers": [],
        "sampled_at_unix": time.time(),
    }


def _nested_number(value: Mapping[str, Any], dotted: str) -> float:
    current: Any = value
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise MutContinuationPolicyError(f"Missing progress field: {dotted}")
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise MutContinuationPolicyError(f"Non-numeric progress field: {dotted}")
    return float(current)


def _process_cpu_ticks(proc_root: Path, pid: int) -> int:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[11]) + int(fields[12])
    except (OSError, ValueError, IndexError):
        return 0


def _direct_output_bytes(path: Path | None) -> int:
    if path is None or not path.is_dir() or path.is_symlink():
        return 0
    total = 0
    try:
        for item in path.iterdir():
            if item.is_file() and not item.is_symlink():
                total += int(item.stat().st_size)
    except OSError:
        return total
    return total


def _read_phase(task: Mapping[str, Any]) -> str:
    phase_path_text = str(task.get("phase_path") or "")
    phase_field = str(task.get("phase_field") or "")
    if not phase_path_text or not phase_field:
        return "SCIENCE"
    path = Path(phase_path_text)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        current: Any = value
        for component in phase_field.split("."):
            current = current[component]
        return str(current or "UNKNOWN").upper()
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return "UNKNOWN"


def _pressure_value(path: Path, key: str = "full", field: str = "avg10") -> float:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            tokens = line.split()
            if tokens and tokens[0] == key:
                values = dict(token.split("=", 1) for token in tokens[1:])
                return float(values[field])
    except (OSError, ValueError):
        pass
    return 0.0


def read_protected_activity(
    task: Mapping[str, Any], *, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    progress_path = Path(str(task["progress_path"]))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    counter = _nested_number(progress, str(task["counter_field"]))
    terminal = float(task.get("terminal_value", -1))
    pid = int(task["pid"])
    output_text = str(task.get("output_root") or "")
    output_root = Path(output_text) if output_text else progress_path.parent
    phase = _read_phase(task)
    lower_phase = phase.lower()
    checkpoint_active = any(token in lower_phase for token in _CHECKPOINT_PHASE_TOKENS)
    mem_available_kib = 0
    try:
        for line in (proc_root / "meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_kib = int(line.split()[1])
                break
    except (OSError, ValueError, IndexError):
        pass
    io_full_avg10 = _pressure_value(proc_root / "pressure/io")
    memory_full_avg10 = _pressure_value(proc_root / "pressure/memory")
    return {
        "task_id": str(task["task_id"]),
        "pid": pid,
        "alive": (proc_root / str(pid)).exists(),
        "completed": terminal >= 0 and counter >= terminal,
        "counter": counter,
        "sampled_at_unix": time.time(),
        "phase": phase,
        "checkpoint_or_flush_active": checkpoint_active,
        "cpu_ticks": _process_cpu_ticks(proc_root, pid),
        "output_bytes": _direct_output_bytes(output_root),
        "memory_pressure": memory_full_avg10 > 0.0 or (
            mem_available_kib > 0 and mem_available_kib < 32 * 1024**2
        ),
        "io_pressure": io_full_avg10 > 0.0,
        "memory_full_avg10": memory_full_avg10,
        "io_full_avg10": io_full_avg10,
    }


class RobustProtectedThroughputGate:
    """Evaluate only sustained, comparable T14 science windows.

    A slowdown alone is never a stop condition.  It must exceed 15% for one
    complete 1,200-second comparable window *and* that same window must carry
    memory or I/O contention evidence.
    """

    def __init__(
        self,
        manifest: Mapping[str, Any],
        baseline: Mapping[str, Any],
        *,
        proc_root: Path = Path("/proc"),
        sample_reader: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        window_seconds: int = EVALUATION_SECONDS,
        maximum_slowdown: float = MAXIMUM_SLOWDOWN,
    ) -> None:
        if baseline.get("status") != "PASS":
            raise MutContinuationPolicyError("Robust gate requires a PASS baseline")
        if int(window_seconds) != EVALUATION_SECONDS:
            raise MutContinuationPolicyError("Evaluation window must remain 1200 seconds")
        if float(maximum_slowdown) != MAXIMUM_SLOWDOWN:
            raise MutContinuationPolicyError("Slowdown threshold must remain 15 percent")
        self.tasks = list(manifest.get("tasks") or [])
        self.baseline = dict(baseline.get("tasks") or {})
        self.proc_root = proc_root
        self.sample_reader = sample_reader or (
            lambda task: read_protected_activity(task, proc_root=proc_root)
        )
        self.window_seconds = int(window_seconds)
        self.maximum_slowdown = float(maximum_slowdown)
        self.windows: dict[str, list[dict[str, Any]]] = {}
        self.checked_windows: list[dict[str, Any]] = []
        self.excluded_windows: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.completed: set[str] = set()
        self.step_baseline_unavailable: set[str] = set()
        for task in self.tasks:
            key = str(task.get("task_id") or "")
            row = self.baseline.get(key)
            if not key or not isinstance(row, Mapping):
                raise MutContinuationPolicyError(f"Missing baseline task: {key}")
            if row.get("units_per_second") is None and row.get("state") != (
                "COMPLETED_DURING_BASELINE"
            ):
                self.step_baseline_unavailable.add(key)

    @staticmethod
    def _phase_comparable(first: Mapping[str, Any], last: Mapping[str, Any]) -> bool:
        if first.get("checkpoint_or_flush_active") or last.get(
            "checkpoint_or_flush_active"
        ):
            return False
        first_phase = str(first.get("phase") or "UNKNOWN")
        last_phase = str(last.get("phase") or "UNKNOWN")
        if first_phase == "UNKNOWN" or first_phase != last_phase:
            return False
        return not any(
            token in first_phase.lower() for token in _CHECKPOINT_PHASE_TOKENS
        )

    def sample(self) -> dict[str, Any]:
        failures: list[str] = []
        task_rows: dict[str, Any] = {}
        for task in self.tasks:
            key = str(task["task_id"])
            current = dict(self.sample_reader(task))
            task_rows[key] = current
            if not current.get("alive") and not current.get("completed"):
                failures.append(f"protected_task_exited:{key}")
                continue
            if current.get("completed"):
                self.completed.add(key)
                continue
            base = self.baseline[key]
            if base.get("state") == "COMPLETED_DURING_BASELINE":
                continue
            baseline_rate = base.get("units_per_second")
            if baseline_rate is None:
                # A coarse baseline observed during a checkpoint is explicitly
                # non-comparable.  Liveness and strict memory/RSS gates remain
                # active in the outer canary monitor.
                continue
            window = self.windows.setdefault(key, [])
            window.append(current)
            now = float(current["sampled_at_unix"])
            window[:] = [
                row
                for row in window
                if now - float(row["sampled_at_unix"])
                <= self.window_seconds + 30
            ]
            if len(window) < 2:
                continue
            first = window[0]
            elapsed = now - float(first["sampled_at_unix"])
            if elapsed < self.window_seconds:
                continue
            delta = float(current["counter"]) - float(first["counter"])
            cpu_delta = int(current.get("cpu_ticks", 0)) - int(
                first.get("cpu_ticks", 0)
            )
            output_delta = int(current.get("output_bytes", 0)) - int(
                first.get("output_bytes", 0)
            )
            phase_comparable = self._phase_comparable(first, current)
            active_without_step = delta <= 0 and (cpu_delta > 0 or output_delta > 0)
            if not phase_comparable or active_without_step:
                self.excluded_windows.append(
                    {
                        "task_id": key,
                        "elapsed_seconds": elapsed,
                        "reason": (
                            "ACTIVE_CHECKPOINT_OR_MATERIALIZATION"
                            if active_without_step
                            else "NON_COMPARABLE_PHASE"
                        ),
                        "phase_start": first.get("phase"),
                        "phase_end": current.get("phase"),
                        "counter_delta": delta,
                        "cpu_tick_delta": cpu_delta,
                        "output_bytes_delta": output_delta,
                    }
                )
                window.clear()
                window.append(current)
                continue
            rate = delta / elapsed if elapsed > 0 else 0.0
            baseline_rate_float = float(baseline_rate)
            slowdown = (
                1.0 - rate / baseline_rate_float
                if baseline_rate_float > 0
                else 1.0
            )
            contention = any(
                bool(row.get("memory_pressure") or row.get("io_pressure"))
                for row in window
            )
            actionable = slowdown > self.maximum_slowdown and contention
            check = {
                "task_id": key,
                "elapsed_seconds": elapsed,
                "counter_delta": delta,
                "baseline_units_per_second": baseline_rate_float,
                "observed_units_per_second": rate,
                "slowdown_fraction": slowdown,
                "memory_or_io_contention": contention,
                "comparable_phase": True,
                "actionable_pause": actionable,
                "pass": not actionable,
            }
            self.checked_windows.append(check)
            window.clear()
            window.append(current)
            if actionable:
                failures.append(f"protected_slowdown_gt_15_percent_with_contention:{key}")
        for failure in failures:
            if failure not in self.failures:
                self.failures.append(failure)
        return {
            "status": "PASS" if not failures else "FAIL",
            "tasks": task_rows,
            "failures": failures,
            "checked_window_count": len(self.checked_windows),
            "excluded_window_count": len(self.excluded_windows),
        }

    def receipt(self) -> dict[str, Any]:
        active = {
            key
            for key, row in self.baseline.items()
            if row.get("state") == "ACTIVE" and row.get("units_per_second") is not None
        }
        checked = {str(row["task_id"]) for row in self.checked_windows}
        missing = sorted(active - checked - self.completed)
        return {
            "schema_version": ROBUST_GATE_SCHEMA,
            "status": "PASS" if not missing and not self.failures else "FAIL",
            "window_seconds": self.window_seconds,
            "maximum_slowdown": self.maximum_slowdown,
            "slowdown_requires_memory_or_io_contention": True,
            "checkpoint_flush_windows_excluded": True,
            "active_without_step_windows_excluded": True,
            "missing_complete_comparable_windows": missing,
            # Backward-compatible key consumed by the current memory receipt.
            "missing_complete_five_minute_windows": missing,
            "checked_windows": list(self.checked_windows),
            "excluded_windows": list(self.excluded_windows),
            "completed_during_canary_task_ids": sorted(self.completed),
            "step_baseline_unavailable_task_ids": sorted(
                self.step_baseline_unavailable
            ),
            "step_baseline_unavailable_warning": (
                "PROTECTED_STEP_BASELINE_UNAVAILABLE_DURING_CHECKPOINT"
                if self.step_baseline_unavailable
                else None
            ),
            "strict_resource_gates_retained": True,
            "failures": list(self.failures),
        }
