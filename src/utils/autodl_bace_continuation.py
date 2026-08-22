"""Build and guard a fresh BACE B11--B14 continuation controller.

The original four-GPU controller represents B8 and B9 as two four-instance
tasks.  ``exp_run`` adoption is deliberately one-run-to-one-task, so a later
controller cannot safely bind either aggregate task to one historical run.
This module flattens the eight passing shard runs into ordinary, non-sharded
evidence tasks and keeps B11--B14 as fresh writers under a new namespace.
"""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from src.utils.autodl_runtime import (
    build_runtime_layout,
    sha256_file,
    sha256_paths,
    verify_required_absolute_outputs,
    verify_required_output_alternatives,
    verify_required_outputs,
)


CONTINUATION_KIND = "bace_b11_b14_continuation_v1"
ACTIVE_STATES = frozenset({"STARTING", "RUNNING"})
SOURCE_MAIN_TASKS = (
    "bace_b6_ppo_smoke",
    "bace_b7_ppo_full",
    "bace_b7_prep_gnn_before",
    "bace_b7_prep_shard_manifests",
    "bace_b7_prep_output_preflight",
    "bace_b10_pool_merged",
)
SOURCE_SHARDED_TASKS = (
    "bace_b8_pool_base",
    "bace_b9_pool_hightemp",
)
SOURCE_REQUIRED_TASKS = (*SOURCE_MAIN_TASKS, *SOURCE_SHARDED_TASKS)
DOWNSTREAM_TASKS = (
    "bace_b11_verification_shards",
    "bace_b11_cross_parent_verified",
    "bace_b12_selector",
    "bace_b13_test_parent_manifest",
    "bace_b13_verification_shards",
    "bace_b13_final_eval",
    "bace_b14_frozen",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class BaceContinuationError(RuntimeError):
    """Continuation evidence is incomplete, mutable, or internally inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BaceContinuationError(f"Required physical JSON file is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaceContinuationError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BaceContinuationError(f"Expected one JSON object: {path}")
    return payload


def _absolute_path(value: Any, *, label: str, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise BaceContinuationError(f"{label} must be a nonempty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise BaceContinuationError(f"{label} must be absolute: {path}")
    return path.resolve(strict=must_exist)


def _safe_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or SAFE_ID.fullmatch(value) is None:
        raise BaceContinuationError(f"{label} is not a safe identifier: {value!r}")
    if len(value) > 120:
        raise BaceContinuationError(f"{label} exceeds 120 characters")
    return value


def _lock_available_read_only(path: Path) -> bool:
    """Probe an existing controller lock without creating or mutating it."""

    if not path.exists():
        return True
    if not path.is_file() or path.is_symlink():
        raise BaceContinuationError(f"Controller lock is not a physical file: {path}")
    with path.open("rb") as handle:
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


class PredecessorControllerGuard:
    """Hold the predecessor lock so it cannot restart beside continuation."""

    def __init__(
        self, policy: Mapping[str, Any], *, require_fresh_targets: bool
    ) -> None:
        self.policy = policy
        self.require_fresh_targets = require_fresh_targets
        self._handle: Any = None

    def __enter__(self) -> "PredecessorControllerGuard":
        source_root = _absolute_path(
            self.policy.get("source_controller_root"),
            label="continuation.source_controller_root",
            must_exist=True,
        )
        lock = source_root / "controller.lock"
        if not lock.is_file() or lock.is_symlink():
            raise BaceContinuationError(
                "Source controller lock evidence is absent or non-physical"
            )
        handle = lock.open("rb")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise BaceContinuationError(
                "Source controller still owns its controller lock; "
                "continuation would duplicate writers"
            ) from exc
        self._handle = handle
        try:
            _assert_source_state_and_targets(
                self.policy,
                require_fresh_targets=self.require_fresh_targets,
                probe_lock=False,
            )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def validate_continuation_policy(
    raw: Any, *, controller_id: str
) -> dict[str, Any] | None:
    """Validate optional top-level continuation metadata during manifest load."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise BaceContinuationError("continuation must be one object")
    if raw.get("kind") != CONTINUATION_KIND:
        raise BaceContinuationError("Unsupported continuation kind")
    source_id = _safe_identifier(
        raw.get("source_controller_id"), label="source_controller_id"
    )
    if source_id == controller_id:
        raise BaceContinuationError("Continuation requires a fresh controller_id")
    for key in (
        "source_controller_root",
        "source_manifest",
        "fresh_output_root",
        "fresh_wnode_cache_db",
        "molclr_node_embedding_cache",
    ):
        _absolute_path(raw.get(key), label=f"continuation.{key}")
    source_sha = raw.get("source_manifest_sha256")
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha) is None:
        raise BaceContinuationError("continuation.source_manifest_sha256 is invalid")
    repair_run = _safe_identifier(
        raw.get("molclr_repair_run_id"), label="molclr_repair_run_id"
    )
    adopted = raw.get("adopted_run_ids")
    if not isinstance(adopted, list) or any(
        not isinstance(value, str) or SAFE_ID.fullmatch(value) is None
        for value in adopted
    ):
        raise BaceContinuationError("continuation.adopted_run_ids is invalid")
    if len(adopted) != len(set(adopted)) or repair_run not in adopted:
        raise BaceContinuationError(
            "continuation adopted run IDs must be unique and include the MolCLR repair"
        )
    required = raw.get("required_source_task_ids")
    if required != list(SOURCE_REQUIRED_TASKS):
        raise BaceContinuationError(
            "continuation.required_source_task_ids does not match the frozen contract"
        )
    return dict(raw)


def assert_continuation_predecessor_quiescent(
    policy: Mapping[str, Any], *, require_fresh_targets: bool
) -> None:
    """Fail closed unless the predecessor has stopped all of its writers."""

    _assert_source_state_and_targets(
        policy,
        require_fresh_targets=require_fresh_targets,
        probe_lock=True,
    )


def _assert_source_state_and_targets(
    policy: Mapping[str, Any], *, require_fresh_targets: bool, probe_lock: bool
) -> None:
    """Validate source state while an optional caller-owned lock is held."""

    source_root = _absolute_path(
        policy.get("source_controller_root"),
        label="continuation.source_controller_root",
        must_exist=True,
    )
    source_manifest = _absolute_path(
        policy.get("source_manifest"),
        label="continuation.source_manifest",
        must_exist=True,
    )
    if source_manifest.is_symlink() or not source_manifest.is_file():
        raise BaceContinuationError("Source manifest must be a physical file")
    if sha256_file(source_manifest) != policy.get("source_manifest_sha256"):
        raise BaceContinuationError("Source controller manifest SHA256 changed")
    lock = source_root / "controller.lock"
    if not lock.is_file() or lock.is_symlink():
        raise BaceContinuationError(
            "Source controller lock evidence is absent or non-physical"
        )
    if probe_lock and not _lock_available_read_only(lock):
        raise BaceContinuationError(
            "Source controller still owns its controller lock; "
            "continuation would duplicate writers"
        )

    task_root = source_root / "tasks"
    if not task_root.is_dir():
        raise BaceContinuationError("Source controller has no frozen task state")
    for state_path in sorted(task_root.glob("*/state.json")):
        state = _read_json(state_path)
        if state.get("state") in ACTIVE_STATES:
            raise BaceContinuationError(
                f"Source controller task is still active: {state_path.parent.name}"
            )
        instances = state.get("instances")
        if not isinstance(instances, dict):
            raise BaceContinuationError(f"Malformed source task state: {state_path}")
        active = [
            instance_id
            for instance_id, instance in instances.items()
            if isinstance(instance, dict) and instance.get("state") in ACTIVE_STATES
        ]
        if active:
            raise BaceContinuationError(
                "Source controller still has active instances in "
                f"{state_path.parent.name}: {active}"
            )
    b10_state = _read_json(task_root / "bace_b10_pool_merged" / "state.json")
    b10_gate = _read_json(task_root / "bace_b10_pool_merged" / "gate.json")
    if b10_state.get("state") != "PASS" or b10_gate.get("status") != "PASS":
        raise BaceContinuationError("Source B10 task and gate must both be PASS")

    cache = _absolute_path(
        policy.get("molclr_node_embedding_cache"),
        label="continuation.molclr_node_embedding_cache",
        must_exist=True,
    )
    if not cache.is_dir() or cache.is_symlink():
        raise BaceContinuationError("MolCLR repair cache must be a physical directory")
    if require_fresh_targets:
        for key in ("fresh_output_root", "fresh_wnode_cache_db"):
            target = _absolute_path(policy.get(key), label=f"continuation.{key}")
            if target.exists() or target.is_symlink():
                raise BaceContinuationError(
                    f"Continuation first-launch target is not fresh: {target}"
                )


def _source_task_documents(
    source_root: Path, task_id: str, instance_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _read_json(source_root / "tasks" / task_id / "state.json")
    gate = _read_json(source_root / "tasks" / task_id / "gate.json")
    if state.get("state") != "PASS" or gate.get("status") != "PASS":
        raise BaceContinuationError(f"Source task/gate is not PASS: {task_id}")
    instances = state.get("instances")
    instance = instances.get(instance_id) if isinstance(instances, dict) else None
    if not isinstance(instance, dict) or instance.get("state") != "PASS":
        raise BaceContinuationError(
            f"Source instance is not PASS: {task_id}/{instance_id}"
        )
    return state, instance


def _validate_completed_run(
    runs_root: Path, run_id: str, *, expected_output: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_root = runs_root / run_id
    spec = _read_json(run_root / "launch_spec.json")
    state = _read_json(run_root / "state.json")
    if spec.get("run_id") != run_id or state.get("run_id") != run_id:
        raise BaceContinuationError(f"Run identity mismatch: {run_id}")
    if state.get("state") != "PASS":
        raise BaceContinuationError(f"Adopted run is not PASS: {run_id}")
    for key in ("dataset", "stage", "gpu_index", "gpu_uuid"):
        if state.get(key) != spec.get(key):
            raise BaceContinuationError(f"Run state {key} mismatch: {run_id}")
    output_value = spec.get("expected_output")
    if expected_output is not None and output_value != expected_output:
        raise BaceContinuationError(f"Run output identity mismatch: {run_id}")
    output = _absolute_path(output_value, label=f"{run_id}.expected_output", must_exist=True)
    failures = verify_required_outputs(output, spec.get("required_output_files", []))
    failures.extend(
        verify_required_output_alternatives(
            output, spec.get("required_output_any", [])
        )
    )
    required_absolute = [
        Path(value) for value in spec.get("required_absolute_output_files", [])
    ]
    if required_absolute:
        evidence_layout = build_runtime_layout(
            project_root=_absolute_path(
                spec.get("project_root"),
                label=f"{run_id}.project_root",
                must_exist=True,
            ),
            data_root=_absolute_path(
                spec.get("data_root"), label=f"{run_id}.data_root", must_exist=True
            ),
            control_root=_absolute_path(
                spec.get("control_root"),
                label=f"{run_id}.control_root",
                must_exist=True,
            ),
        )
        failures.extend(
            verify_required_absolute_outputs(
                required_absolute, allowed_root=evidence_layout.artifacts_dir
            )
        )
    input_manifest = _absolute_path(
        spec.get("input_manifest"), label=f"{run_id}.input_manifest", must_exist=True
    )
    if spec.get("input_hash") != sha256_file(input_manifest):
        failures.append("input hash mismatch")
    config_files = [
        _absolute_path(path, label=f"{run_id}.config_file", must_exist=True)
        for path in spec.get("config_files", [])
    ]
    if spec.get("config_hash") != sha256_paths(config_files):
        failures.append("config hash mismatch")
    log = _absolute_path(spec.get("log_path"), label=f"{run_id}.log", must_exist=True)
    marker = spec.get("required_log_marker")
    if not isinstance(marker, str) or not _file_contains(log, marker):
        failures.append("required log marker missing")
    if failures:
        raise BaceContinuationError(
            f"Adopted run evidence failed for {run_id}: " + "; ".join(failures)
        )
    return spec, state


def _file_contains(path: Path, marker: str, *, chunk_bytes: int = 1024 * 1024) -> bool:
    """Search a potentially large worker log without loading it into memory."""

    needle = marker.encode("utf-8")
    overlap = max(0, len(needle) - 1)
    previous = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return False
            combined = previous + chunk
            if needle in combined:
                return True
            previous = combined[-overlap:] if overlap else b""


def _adopted_task(
    *,
    task_id: str,
    stage: str,
    dataset: str,
    priority: int,
    depends_on: Sequence[str],
    spec: Mapping[str, Any],
    run_state: Mapping[str, Any],
    data_splits: Sequence[str],
    manifest_only: bool,
    external_bace_stage: str | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": task_id,
        "dataset": dataset,
        "stage": stage,
        "runner_dataset": spec["dataset"],
        "runner_stage": spec["stage"],
        "depends_on": list(depends_on),
        "resource": "gpu" if spec.get("gpu_index") is not None else "cpu",
        "priority": priority,
        "adopt_existing_run_id": spec["run_id"],
        "adopt_gpu_index": spec.get("gpu_index"),
        "adopt_gpu_uuid": spec.get("gpu_uuid"),
        "adopt_project_root": spec["project_root"],
        "adopt_git_commit": spec["git_commit"],
        "adopt_max_gpus": spec["max_gpus"],
        "adopt_heavy": bool(spec.get("heavy", False)),
        "data_splits": list(data_splits),
        "manifest_only": bool(manifest_only),
        "command": list(spec["command"]),
        "config_files": list(spec.get("config_files", [])),
        "input_manifest": spec["input_manifest"],
        "expected_output": spec["expected_output"],
        "required_output_files": list(spec.get("required_output_files", [])),
        "required_output_any": list(spec.get("required_output_any", [])),
        "required_absolute_output_files": list(
            spec.get("required_absolute_output_files", [])
        ),
        "required_log_marker": spec["required_log_marker"],
        "environment": dict(spec.get("environment", {})),
        "publish_bace_stage": False,
    }
    if external_bace_stage is not None:
        task["external_bace_stage"] = external_bace_stage
    if task["resource"] == "gpu" and (
        run_state.get("gpu_index") != spec.get("gpu_index")
        or run_state.get("gpu_uuid") != spec.get("gpu_uuid")
    ):
        raise BaceContinuationError(f"Adopted GPU identity changed for {spec['run_id']}")
    return task


def _command_option(command: Sequence[Any], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise BaceContinuationError(f"Expected exactly one {option} in repaired command")
    value = command[positions[0] + 1]
    if not isinstance(value, str) or not value:
        raise BaceContinuationError(f"Invalid {option} value in repaired command")
    return value


def _validate_molclr_cache_inventory(
    *, repair_output: Path, repair_manifest: Mapping[str, Any], cache_root: Path
) -> None:
    inventory = repair_output / "calibration_parent_molclr_cache.jsonl"
    if not inventory.is_file() or inventory.is_symlink():
        raise BaceContinuationError("MolCLR repair cache inventory is absent")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        inventory.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaceContinuationError(
                f"Invalid MolCLR cache inventory JSON at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise BaceContinuationError("MolCLR cache inventory row is not an object")
        cache_path = _absolute_path(
            row.get("cache_path"),
            label=f"MolCLR cache inventory line {line_number}",
            must_exist=True,
        )
        try:
            cache_path.relative_to(cache_root)
        except ValueError as exc:
            raise BaceContinuationError(
                f"MolCLR cache inventory escapes repaired cache root: {cache_path}"
            ) from exc
        if cache_path.is_symlink() or not cache_path.is_file() or cache_path.stat().st_size <= 0:
            raise BaceContinuationError(
                f"MolCLR cache inventory entry is not a nonempty physical file: {cache_path}"
            )
        rows.append(row)
    expected_count = repair_manifest.get("parent_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(rows) != expected_count
    ):
        raise BaceContinuationError(
            "MolCLR repair cache inventory count differs from prep_manifest"
        )


def _replace_command_option(task: dict[str, Any], option: str, replacement: str) -> None:
    command = task.get("command")
    if not isinstance(command, list):
        raise BaceContinuationError(f"{task.get('id')} has no command")
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise BaceContinuationError(f"{task.get('id')} must contain one {option}")
    command[positions[0] + 1] = replacement


def build_bace_continuation_payload(
    *,
    source_manifest: Path,
    source_controller_root: Path,
    runs_root: Path,
    molclr_repair_run_id: str,
    controller_id: str,
    output_root: Path,
    wnode_cache_db: Path,
) -> dict[str, Any]:
    """Return one validated-by-construction continuation manifest payload."""

    controller_id = _safe_identifier(controller_id, label="controller_id")
    repair_run_id = _safe_identifier(
        molclr_repair_run_id, label="molclr_repair_run_id"
    )
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    source_controller_root = source_controller_root.expanduser().resolve(strict=True)
    runs_root = runs_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    wnode_cache_db = wnode_cache_db.expanduser().resolve(strict=False)
    source_payload = _read_json(source_manifest)
    source_id = _safe_identifier(
        source_payload.get("controller_id"), label="source controller_id"
    )
    if source_id == controller_id:
        raise BaceContinuationError("Continuation controller ID must be fresh")
    source_tasks_raw = source_payload.get("tasks")
    if not isinstance(source_tasks_raw, list):
        raise BaceContinuationError("Source manifest has no tasks")
    source_tasks = {
        task.get("id"): task
        for task in source_tasks_raw
        if isinstance(task, dict) and isinstance(task.get("id"), str)
    }
    missing = sorted(set((*SOURCE_REQUIRED_TASKS, *DOWNSTREAM_TASKS)) - set(source_tasks))
    if missing:
        raise BaceContinuationError(f"Source manifest misses required tasks: {missing}")

    adopted: list[dict[str, Any]] = []
    adopted_run_ids: list[str] = []

    def adopt_source_main(
        source_task_id: str,
        *,
        priority: int,
        depends_on: Sequence[str],
        stage: str | None = None,
        dataset: str | None = None,
        external_bace_stage: str | None = None,
    ) -> None:
        _state, instance = _source_task_documents(
            source_controller_root, source_task_id, "main"
        )
        run_id = _safe_identifier(instance.get("run_id"), label="source run_id")
        spec, run_state = _validate_completed_run(
            runs_root,
            run_id,
            expected_output=str(instance.get("expected_output")),
        )
        raw = source_tasks[source_task_id]
        adopted.append(
            _adopted_task(
                task_id=source_task_id,
                stage=stage or str(raw["stage"]),
                dataset=dataset or str(raw["dataset"]),
                priority=priority,
                depends_on=depends_on,
                spec=spec,
                run_state=run_state,
                data_splits=raw.get("data_splits", []),
                manifest_only=bool(raw.get("manifest_only", False)),
                external_bace_stage=external_bace_stage,
            )
        )
        adopted_run_ids.append(run_id)

    adopt_source_main(
        "bace_b6_ppo_smoke",
        priority=10,
        depends_on=(),
        external_bace_stage="B5_ORACLE_SMOKE",
    )
    adopt_source_main(
        "bace_b7_ppo_full", priority=20, depends_on=("bace_b6_ppo_smoke",)
    )
    adopt_source_main(
        "bace_b7_prep_gnn_before",
        priority=21,
        depends_on=("bace_b6_ppo_smoke",),
        dataset="bace-evidence",
    )
    adopt_source_main(
        "bace_b7_prep_shard_manifests",
        priority=22,
        depends_on=("bace_b6_ppo_smoke",),
        dataset="bace-evidence",
    )
    adopt_source_main(
        "bace_b7_prep_output_preflight",
        priority=23,
        depends_on=("bace_b6_ppo_smoke",),
        dataset="bace-evidence",
    )

    repair_spec, repair_state = _validate_completed_run(runs_root, repair_run_id)
    repair_output = _absolute_path(
        repair_spec.get("expected_output"), label="MolCLR repair output", must_exist=True
    )
    repair_manifest = _read_json(repair_output / "prep_manifest.json")
    repair_contract = {
        "status": "PASS",
        "dataset": "bace",
        "action": "CALIBRATION_MOLCLR_PARENT_CACHE",
        "rf_oracle_used": False,
        "calibration_loaded": True,
        "test_loaded": False,
        "policy_checkpoint_loaded": False,
        "candidate_generation_performed": False,
        "selector_fitted": False,
    }
    mismatches = [
        key for key, expected in repair_contract.items() if repair_manifest.get(key) != expected
    ]
    if mismatches:
        raise BaceContinuationError(
            "MolCLR repair provenance mismatch: " + ", ".join(mismatches)
        )
    node_embedding_cache = _absolute_path(
        _command_option(repair_spec.get("command", []), "--node-embedding-cache-dir"),
        label="MolCLR repair node cache",
        must_exist=True,
    )
    _validate_molclr_cache_inventory(
        repair_output=repair_output,
        repair_manifest=repair_manifest,
        cache_root=node_embedding_cache,
    )
    adopted.append(
        _adopted_task(
            task_id="bace_b7_prep_molclr_parent",
            stage="B7_PREP_MOLCLR_PARENT_REPAIRED",
            dataset="bace-evidence",
            priority=24,
            depends_on=("bace_b6_ppo_smoke",),
            spec=repair_spec,
            run_state=repair_state,
            data_splits=("calibration",),
            manifest_only=False,
        )
    )
    adopted_run_ids.append(repair_run_id)

    flattened_ids: dict[str, list[str]] = {}
    for source_task_id, base_priority in (
        ("bace_b8_pool_base", 30),
        ("bace_b9_pool_hightemp", 40),
    ):
        raw = source_tasks[source_task_id]
        flattened_ids[source_task_id] = []
        for index in range(4):
            instance_id = f"shard-{index:03d}"
            new_task_id = f"{source_task_id}_shard_{index:03d}"
            _state, instance = _source_task_documents(
                source_controller_root, source_task_id, instance_id
            )
            run_id = _safe_identifier(instance.get("run_id"), label="source shard run_id")
            spec, run_state = _validate_completed_run(
                runs_root,
                run_id,
                expected_output=str(instance.get("expected_output")),
            )
            canonical = index == 0
            adopted.append(
                _adopted_task(
                    task_id=new_task_id,
                    stage=(
                        str(raw["stage"])
                        if canonical
                        else f"{raw['stage']}_ADOPTED_SHARD_{index:03d}"
                    ),
                    dataset="bace" if canonical else "bace-evidence",
                    priority=base_priority + index,
                    depends_on=(
                        "bace_b7_ppo_full",
                        "bace_b7_prep_shard_manifests",
                        "bace_b7_prep_output_preflight",
                    ),
                    spec=spec,
                    run_state=run_state,
                    data_splits=raw.get("data_splits", []),
                    manifest_only=False,
                )
            )
            flattened_ids[source_task_id].append(new_task_id)
            adopted_run_ids.append(run_id)

    b10_dependencies = (
        *flattened_ids["bace_b8_pool_base"],
        *flattened_ids["bace_b9_pool_hightemp"],
    )
    adopt_source_main(
        "bace_b10_pool_merged", priority=50, depends_on=b10_dependencies
    )

    fresh_preflight_id = "bace_continuation_output_preflight"
    planned_subroots = (
        "b11-shards",
        "b11-merged",
        "b12-selector",
    )
    preflight_command = [
        "bash",
        "{project_root}/scripts/autodl/run_bace_frozen_gnn_downstream.sh",
        "prep",
        "--prep-action",
        "OUTPUT_PREFLIGHT",
        "--b6-output",
        "{dep_bace_b6_ppo_smoke_output}",
    ]
    for name in planned_subroots:
        preflight_command.extend(("--planned-output-root", str(output_root / name)))
    preflight_command.extend(
        (
            "--planned-output-root",
            str(wnode_cache_db),
            "--output-dir",
            "{task_output}",
        )
    )
    fresh_preflight = {
        "id": fresh_preflight_id,
        "dataset": "bace-continuation",
        "stage": "B11_CONTINUATION_OUTPUT_PREFLIGHT",
        "runner_dataset": "bace-gnn-clean-prep-continuation",
        "runner_stage": "B11_CONTINUATION_OUTPUT_PREFLIGHT",
        "depends_on": ["bace_b6_ppo_smoke", "bace_b10_pool_merged"],
        "resource": "cpu",
        "priority": 60,
        "data_splits": [],
        "manifest_only": True,
        "command": preflight_command,
        "input_manifest": "{dep_bace_b6_ppo_smoke_output}/ppo_smoke_manifest.json",
        "expected_output": str(output_root / "continuation-preflight" / "attempt-{attempt}"),
        "required_output_files": ["output_preflight.jsonl", "prep_manifest.json", "PASS"],
        "required_log_marker": "[BACE_B7_PARALLEL_PREP_PASS]",
        "environment": {
            "AUTODL_DATA_ROOT": str(repair_spec.get("data_root")),
            "AUTODL_CONTROL_ROOT": str(repair_spec.get("control_root")),
            "AUTODL_PYTHON": str(repair_spec.get("python_executable")),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    }

    fresh_tasks: list[dict[str, Any]] = []
    output_suffix = {
        "bace_b11_verification_shards": "b11-shards/{shard_id}/attempt-{attempt}",
        "bace_b11_cross_parent_verified": "b11-merged/attempt-{attempt}",
        "bace_b12_selector": "b12-selector/attempt-{attempt}",
        "bace_b13_test_parent_manifest": "b13-test-parent-manifest/attempt-{attempt}",
        "bace_b13_verification_shards": "b13-shards/{shard_id}/attempt-{attempt}",
        "bace_b13_final_eval": "b13-final/attempt-{attempt}",
        "bace_b14_frozen": "b14-frozen/attempt-{attempt}",
    }
    for task_id in DOWNSTREAM_TASKS:
        task = deepcopy(source_tasks[task_id])
        task["expected_output"] = str(output_root / output_suffix[task_id])
        for key in tuple(task):
            if key.startswith("adopt_"):
                task.pop(key, None)
        if task_id == "bace_b11_verification_shards":
            task["depends_on"] = [
                "bace_b10_pool_merged",
                "bace_b7_prep_gnn_before",
                "bace_b7_prep_molclr_parent",
                "bace_b7_prep_shard_manifests",
                fresh_preflight_id,
            ]
        if task_id in {
            "bace_b11_verification_shards",
            "bace_b13_verification_shards",
        }:
            _replace_command_option(task, "--wnode-cache-db", str(wnode_cache_db))
            _replace_command_option(
                task,
                "--node-embedding-cache-dir",
                str(node_embedding_cache),
            )
        fresh_tasks.append(task)

    taste_task = deepcopy(source_tasks.get("tastemolnet_foundation"))
    if not isinstance(taste_task, dict):
        taste_task = {
            "id": "tastemolnet_foundation",
            "dataset": "tastemolnet",
            "stage": "FOUNDATION_ONLY",
            "depends_on": [],
            "resource": "cpu",
            "priority": 1000,
            "data_splits": ["train", "validation", "calibration"],
            "command": None,
            "blocked_reason": "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW",
        }
    taste_task["command"] = None
    taste_task["blocked_reason"] = "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW"

    policy = {
        "kind": CONTINUATION_KIND,
        "source_controller_id": source_id,
        "source_controller_root": str(source_controller_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "required_source_task_ids": list(SOURCE_REQUIRED_TASKS),
        "molclr_repair_run_id": repair_run_id,
        "molclr_repair_output": str(repair_output),
        "molclr_node_embedding_cache": str(node_embedding_cache),
        "adopted_run_ids": adopted_run_ids,
        "fresh_output_root": str(output_root),
        "fresh_wnode_cache_db": str(wnode_cache_db),
        "old_failed_molclr_task_mutated": False,
        "paper_frozen": True,
        "run_tastemolnet": False,
    }
    validate_continuation_policy(policy, controller_id=controller_id)
    runtime = deepcopy(source_payload.get("runtime", {}))
    runtime["keep_alive_when_blocked"] = True
    return {
        "schema_version": 1,
        "controller_id": controller_id,
        "paper_frozen": True,
        "continuation": policy,
        "runtime": runtime,
        "resource_gates": deepcopy(source_payload.get("resource_gates", {})),
        "tasks": [*adopted, fresh_preflight, *fresh_tasks, taste_task],
    }
