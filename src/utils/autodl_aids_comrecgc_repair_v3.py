"""Build the CPU-only, host-memory-exclusive AIDS ComRecGC repair v3.

Repair v2 proved its two AIDS source gates and then lost the scientific child
to the AutoDL container OOM killer while Mutagenicity common-recourse was
running concurrently.  This module adopts only those two immutable source
gates and publishes one fresh AIDS continuation.  It intentionally contains no
Mutagenicity scientific task: the latter failed its independent trace-parity
gate and is not an engineering retry.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerManifest,
    TaskSpec,
    load_controller_manifest,
)
from src.utils.autodl_four_by_four_am_repair import (
    STANDARDIZED_REQUIRED_FILES,
    VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
    verify_fix_ancestry,
)
from src.utils.autodl_four_by_four_repair import (
    RepairManifestError,
    sha256_file,
    verify_controller_terminal,
)


SPEC_SCHEMA = "aids_comrecgc_cpu_highmem_repair_spec_v3"
CONTROLLER_ID = "four_methods_four_datasets_aids_comrecgc_repair_v3"
SOURCE_CONTROLLER_ID = "four_methods_four_datasets_am_repair_v2"
SOURCE_NAMESPACE = "four_methods_four_datasets_continuation"
GENERATION_SOURCE_KEY = "aids_generation"
THRESHOLD_SOURCE_KEY = "aids_threshold"
GENERATION_GATE_TASK_ID = "am_v3_source_aids_comrec_generation"
THRESHOLD_GATE_TASK_ID = "am_v3_source_aids_comrec_threshold"
STANDARDIZATION_TASK_ID = "aids_comrecgc_standardized_cpu_highmem"
SOURCE_TASK_IDS = {
    GENERATION_SOURCE_KEY: "am_v2_source_aids_comrec_generation",
    THRESHOLD_SOURCE_KEY: "am_v2_source_aids_comrec_threshold",
}
MINIMUM_HEADROOM_BYTES = 400 * 1024**3
HEX64 = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: str | Path) -> dict[str, Any]:
    logical = Path(path).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise RepairManifestError(f"JSON path must be absolute and physical: {logical}")
    source = logical.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RepairManifestError(f"JSON path must be a nonempty file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairManifestError(f"Invalid JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise RepairManifestError(f"Expected JSON object: {source}")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairManifestError(f"{label} must be one object")
    return value


def _absolute(value: Any, *, label: str, kind: str) -> Path:
    logical = Path(str(value or "")).expanduser()
    if not logical.is_absolute() or logical.is_symlink():
        raise RepairManifestError(f"{label} must be absolute and physical: {logical}")
    if kind == "fresh":
        return logical.resolve(strict=False)
    resolved = logical.resolve(strict=True)
    if kind == "dir" and not resolved.is_dir():
        raise RepairManifestError(f"{label} must be a directory: {resolved}")
    if kind == "file" and (
        not resolved.is_file() or resolved.stat().st_size <= 0
    ):
        raise RepairManifestError(f"{label} must be a nonempty file: {resolved}")
    if kind == "readable" and (not resolved.is_file() or not os.access(resolved, os.R_OK)):
        raise RepairManifestError(f"{label} must be a readable file: {resolved}")
    return resolved


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _source_controller(
    *, manifest_path: Path, controller_root: Path, control_root: Path
) -> ControllerManifest:
    expected_namespace = control_root / SOURCE_NAMESPACE
    if manifest_path.parent != (expected_namespace / "manifests").resolve(strict=True):
        raise RepairManifestError("repair-v2 manifest is outside its exact namespace")
    if controller_root != (expected_namespace / SOURCE_CONTROLLER_ID).resolve(strict=True):
        raise RepairManifestError("repair-v2 controller root is not exact")
    manifest = load_controller_manifest(manifest_path)
    if manifest.controller_id != SOURCE_CONTROLLER_ID:
        raise RepairManifestError("source controller is not AM repair-v2")
    return manifest


def _source_semantic(source_key: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    if source_key not in SOURCE_TASK_IDS:
        raise RepairManifestError(f"unknown AIDS source key: {source_key}")
    failures: list[str] = []
    if gate.get("schema_version") != "four_by_four_am_repair_source_gate_v2":
        failures.append("schema_version")
    if gate.get("status") != "PASS" or gate.get("source_key") != source_key:
        failures.append("status_or_source_key")
    evidence = _mapping(gate.get("evidence"), label="source gate evidence")
    if evidence.get("status") != "PASS" or evidence.get("dataset") != "aids":
        failures.append("evidence_identity")
    semantic = _mapping(evidence.get("semantic"), label="source gate semantic")
    if semantic.get("dataset") != "aids":
        failures.append("semantic.dataset")
    if source_key == GENERATION_SOURCE_KEY:
        if semantic.get("kind") != "generation_adoption":
            failures.append("semantic.kind")
        generation_root = _absolute(
            semantic.get("generation_root"), label="generation_root", kind="dir"
        )
        payload_hash = str(semantic.get("generation_payload_claimed_sha256") or "")
        if HEX64.fullmatch(payload_hash) is None:
            failures.append("generation_payload_claimed_sha256")
        result = {
            "kind": "generation_adoption",
            "dataset": "aids",
            "generation_root": str(generation_root),
            "generation_payload_claimed_sha256": payload_hash,
        }
    else:
        if semantic.get("kind") != "threshold":
            failures.append("semantic.kind")
        threshold = _absolute(
            semantic.get("threshold_contract"),
            label="threshold_contract",
            kind="file",
        )
        if int(semantic.get("threshold_count", -1)) != 601:
            failures.append("threshold_count")
        if semantic.get("test_used_for_selection") is not False:
            failures.append("test_used_for_selection")
        if not math.isclose(float(semantic.get("theta_star", -1.0)), 0.05, abs_tol=1e-15):
            failures.append("theta_star")
        if not math.isclose(float(semantic.get("cost_cap", -1.0)), 0.0535, abs_tol=1e-15):
            failures.append("cost_cap")
        result = {
            "kind": "threshold",
            "dataset": "aids",
            "threshold_contract": str(threshold),
            "threshold_contract_sha256": sha256_file(threshold),
            "threshold_count": 601,
            "theta_star": 0.05,
            "cost_cap": 0.0535,
            "test_used_for_selection": False,
        }
    if failures:
        raise RepairManifestError(f"repair-v2 {source_key} gate is invalid: {failures}")
    return result


def verify_v2_source(
    *,
    source_key: str,
    source_manifest: str | Path,
    source_controller_root: str | Path,
    control_root: str | Path,
    expected_output_root: str | Path,
    project_root: str | Path,
    required_fix_commit: str,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    manifest_path = _absolute(source_manifest, label="source_manifest", kind="file")
    controller_root = _absolute(
        source_controller_root, label="source_controller_root", kind="dir"
    )
    control = _absolute(control_root, label="control_root", kind="dir")
    source = _source_controller(
        manifest_path=manifest_path,
        controller_root=controller_root,
        control_root=control,
    )
    fix = verify_fix_ancestry(
        project_root=project_root, required_fix_commit=required_fix_commit
    )
    output = _absolute(expected_output_root, label="source output", kind="dir")
    task_id = SOURCE_TASK_IDS[source_key]
    terminal = verify_controller_terminal(
        source_manifest=manifest_path,
        source_controller_root=controller_root,
        task_id=task_id,
        expected_output_root=output,
        required_files=("source_gate.json", "PASS"),
        proc_root=proc_root,
    )
    semantic = _source_semantic(source_key, _read_object(output / "source_gate.json"))
    return {
        "schema_version": "aids_comrecgc_repair_v3_source_terminal",
        "status": "PASS",
        "source_key": source_key,
        "dataset": "aids",
        "source_controller_id": source.controller_id,
        "controller_terminal": terminal,
        "semantic": semantic,
        "execution_fix_gate": fix,
        "verified_at": _utc_now(),
    }


def publish_source_gate(
    *, source_key: str, evidence: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    if evidence.get("status") != "PASS" or evidence.get("source_key") != source_key:
        raise RepairManifestError("cannot publish a mismatched/non-PASS source")
    destination = _absolute(output_dir, label="source output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"source output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o755)
    payload = {
        "schema_version": "aids_comrecgc_repair_v3_source_gate",
        "status": "PASS",
        "source_key": source_key,
        "evidence": dict(evidence),
        "published_at": _utc_now(),
    }
    _atomic_json(destination / "source_gate.json", payload)
    descriptor = os.open(destination / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _cgroup_oom_evidence(root: Path) -> dict[str, Any]:
    def integer(name: str) -> int:
        source = _absolute(root / name, label=f"cgroup.{name}", kind="readable")
        try:
            return int(source.read_text(encoding="utf-8").strip())
        except ValueError as exc:
            raise RepairManifestError(f"malformed cgroup counter: {source}") from exc

    limit = integer("memory.limit_in_bytes")
    maximum = integer("memory.max_usage_in_bytes")
    fail_count = integer("memory.failcnt")
    oom_path = _absolute(root / "memory.oom_control", label="cgroup.oom", kind="readable")
    oom_fields: dict[str, int] = {}
    for line in oom_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            oom_fields[fields[0]] = int(fields[1])
    if limit <= 0 or maximum < limit or fail_count <= 0 or oom_fields.get("oom_kill", 0) < 1:
        raise RepairManifestError("cgroup does not prove a memory-limit OOM kill")
    return {
        "memory_limit_in_bytes": limit,
        "memory_max_usage_in_bytes": maximum,
        "memory_failcnt": fail_count,
        "oom_kill_count": oom_fields["oom_kill"],
        "peak_reached_limit": True,
    }


def _failed_aids_task(
    *, manifest: ControllerManifest, controller_root: Path, expected_root: Path
) -> tuple[TaskSpec, dict[str, Any]]:
    task_id = "aids_comrecgc_standardized"
    if task_id not in manifest.by_id:
        raise RepairManifestError("repair-v2 has no AIDS standardization task")
    task = manifest.by_id[task_id]
    state = _read_object(controller_root / "tasks" / task_id / "state.json")
    instance = _mapping(
        _mapping(state.get("instances"), label="task instances").get("main"),
        label="task main instance",
    )
    failures: list[str] = []
    if task.dataset != "aids" or task.stage != "AM_COMRECGC_HELDOUT_EVAL":
        failures.append("task identity")
    if task.resource != "gpu" or task.data_splits != ("test",):
        failures.append("old resource/split contract")
    if state.get("state") != "FAILED" or instance.get("state") != "FAILED":
        failures.append("terminal state")
    if int(instance.get("exit_code", -1)) != 1:
        failures.append("exit_code")
    if Path(str(instance.get("expected_output") or "")).resolve(strict=True) != expected_root:
        failures.append("failed output identity")
    failure = _read_object(expected_root / "FAILED.json")
    message = str(failure.get("message") or "")
    if (
        failure.get("status") != "FAILED"
        or failure.get("dataset") != "aids"
        or "run_common_recourse.py" not in message
        or "Signals.SIGKILL: 9" not in message
    ):
        failures.append("SIGKILL evidence")
    if failures:
        raise RepairManifestError(f"repair-v2 AIDS failure is not the reviewed OOM: {failures}")
    return task, {
        "source_task_id": task_id,
        "source_failed_output": str(expected_root),
        "source_failed_json_sha256": sha256_file(expected_root / "FAILED.json"),
        "source_exit_code": 1,
        "source_signal": "SIGKILL",
        "scientific_failure": False,
    }


def _scientific_environment(
    *, source_task: TaskSpec, generation_root: Path, threshold_contract: Path
) -> dict[str, str]:
    directories = ("COMRECGC_UPSTREAM_ROOT", "DATASET_DIR", "MOLCLR_ROOT")
    files = (
        "DATASET_CSV",
        "SOURCE_CSV",
        "TEACHER_PATH",
        "DISTANCE_CHECKPOINT",
        "MOLCLR_CHECKPOINT",
    )
    environment: dict[str, str] = {
        "AUTODL_PYTHON": "{python}",
        "DATASET": "aids",
        "SOURCE_GENERATION_ROOT": str(generation_root),
        "THRESHOLDS_PATH": str(threshold_contract),
        "OUTPUT_ROOT": "{task_output}",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "RUN_TASTEMOLNET": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
    }
    for key in directories:
        environment[key] = str(
            _absolute(source_task.environment.get(key), label=key, kind="dir")
        )
    for key in files:
        environment[key] = str(
            _absolute(source_task.environment.get(key), label=key, kind="file")
        )
    source_generation = _absolute(
        source_task.environment.get("SOURCE_GENERATION_ROOT"),
        label="source task generation root",
        kind="dir",
    )
    if source_generation != generation_root:
        raise RepairManifestError("source task generation root differs from adopted gate")
    return environment


def _source_gate_task(
    *,
    source_key: str,
    source_output: Path,
    source_manifest: Path,
    source_controller_root: Path,
    control_root: Path,
    project_root: Path,
    required_fix_commit: str,
    fresh_root: Path,
    priority: int,
) -> dict[str, Any]:
    task_id = (
        GENERATION_GATE_TASK_ID
        if source_key == GENERATION_SOURCE_KEY
        else THRESHOLD_GATE_TASK_ID
    )
    is_threshold = source_key == THRESHOLD_SOURCE_KEY
    return {
        "id": task_id,
        "dataset": "aids" if is_threshold else "repair-source-audit",
        "stage": (
            "AM_COMRECGC_THRESHOLD_FREEZE"
            if is_threshold
            else "FOUR_BY_FOUR_AM_REPAIR_SOURCE_ADOPTION"
        ),
        "runner_dataset": f"am-v3-source-{source_key}",
        "runner_stage": "FOUR_BY_FOUR_AM_REPAIR_SOURCE_GATE",
        "depends_on": [],
        "resource": "cpu",
        "priority": priority,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": is_threshold,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/build_aids_comrecgc_repair_v3_manifest.py",
            "--config",
            "configs/hpc.yaml",
            "verify-source",
            "--source-key",
            source_key,
            "--source-manifest",
            str(source_manifest),
            "--source-controller-root",
            str(source_controller_root),
            "--control-root",
            str(control_root),
            "--expected-output-root",
            str(source_output),
            "--project-root",
            str(project_root),
            "--required-fix-commit",
            required_fix_commit,
            "--output-dir",
            "{task_output}",
        ],
        "input_manifest": str(source_output / "source_gate.json"),
        "config_files": [str(source_output / "source_gate.json")],
        "expected_output": str(fresh_root / f"source-adoptions/{source_key}/attempt-{{attempt}}"),
        "required_output_files": ["source_gate.json", "PASS"],
        "required_log_marker": f"[AIDS_COMRECGC_REPAIR_V3_SOURCE_PASS] source={source_key}",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        },
    }


def _load_spec(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = _absolute(path, label="spec", kind="file")
    spec = _read_object(source)
    if spec.get("schema_version") != SPEC_SCHEMA or spec.get("controller_id") != CONTROLLER_ID:
        raise RepairManifestError("invalid AIDS repair-v3 spec identity")
    if spec.get("paper_frozen") is not True or spec.get("run_tastemolnet") != 0:
        raise RepairManifestError("paper must stay frozen and Taste disabled")
    if any(key in spec for key in ("mutagenicity", "bace", "taste", "paper", "continuation")):
        raise RepairManifestError("AIDS repair-v3 spec contains a forbidden route")
    return source, spec


def build_payload(
    *, spec_path: str | Path, proc_root_override: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path_resolved, spec = _load_spec(spec_path)
    runtime_root = _absolute(spec.get("runtime_root"), label="runtime_root", kind="dir")
    control_root = _absolute(spec.get("control_root"), label="control_root", kind="dir")
    project_root = _absolute(spec.get("project_root"), label="project_root", kind="dir")
    python = _absolute(spec.get("python"), label="python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError("configured Python is not executable")
    fresh_root = _absolute(spec.get("fresh_output_root"), label="fresh output", kind="fresh")
    if fresh_root.exists() or fresh_root.is_symlink():
        raise RepairManifestError(f"fresh output already exists: {fresh_root}")
    try:
        fresh_root.relative_to((runtime_root / "outputs/autodl").resolve(strict=False))
    except ValueError as exc:
        raise RepairManifestError("fresh output must stay below runtime outputs/autodl") from exc
    controller_destination = (control_root / SOURCE_NAMESPACE / CONTROLLER_ID).resolve(strict=False)
    if controller_destination.exists() or controller_destination.is_symlink():
        raise RepairManifestError("fresh repair-v3 controller root already exists")
    required_fix = str(spec.get("verify_comrecgc_checkout_safe_git_fix_commit") or "")
    fix = verify_fix_ancestry(project_root=project_root, required_fix_commit=required_fix)
    proc_root = _absolute(
        proc_root_override if proc_root_override is not None else spec.get("proc_root", "/proc"),
        label="proc_root",
        kind="dir",
    )
    cgroup_root = _absolute(spec.get("cgroup_memory_root"), label="cgroup memory root", kind="dir")
    min_free_bytes = int(spec.get("min_cgroup_free_bytes", 0))
    if min_free_bytes < MINIMUM_HEADROOM_BYTES:
        raise RepairManifestError(
            f"min_cgroup_free_bytes must be at least {MINIMUM_HEADROOM_BYTES}"
        )
    oom_evidence = _cgroup_oom_evidence(cgroup_root)

    source_config = _mapping(spec.get("source_controller"), label="source_controller")
    source_manifest = _absolute(source_config.get("manifest"), label="source manifest", kind="file")
    source_controller_root = _absolute(source_config.get("root"), label="source root", kind="dir")
    source_manifest_value = _source_controller(
        manifest_path=source_manifest,
        controller_root=source_controller_root,
        control_root=control_root,
    )
    source_outputs = _mapping(spec.get("source_outputs"), label="source_outputs")
    if set(source_outputs) != {GENERATION_SOURCE_KEY, THRESHOLD_SOURCE_KEY}:
        raise RepairManifestError("source_outputs must contain exactly two AIDS gates")
    adopted: dict[str, dict[str, Any]] = {}
    resolved_source_outputs: dict[str, Path] = {}
    for key in (GENERATION_SOURCE_KEY, THRESHOLD_SOURCE_KEY):
        output = _absolute(source_outputs[key], label=f"source_outputs.{key}", kind="dir")
        evidence = verify_v2_source(
            source_key=key,
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            control_root=control_root,
            expected_output_root=output,
            project_root=project_root,
            required_fix_commit=required_fix,
            proc_root=proc_root,
        )
        adopted[key] = evidence
        resolved_source_outputs[key] = output
    generation_root = _absolute(
        _mapping(
            adopted[GENERATION_SOURCE_KEY]["semantic"],
            label="generation semantic",
        ).get("generation_root"),
        label="generation root",
        kind="dir",
    )
    threshold_contract = _absolute(
        _mapping(
            adopted[THRESHOLD_SOURCE_KEY]["semantic"],
            label="threshold semantic",
        ).get("threshold_contract"),
        label="threshold contract",
        kind="file",
    )
    failed_root = _absolute(
        spec.get("failed_aids_output_root"),
        label="failed AIDS root",
        kind="dir",
    )
    source_task, failure_evidence = _failed_aids_task(
        manifest=source_manifest_value,
        controller_root=source_controller_root,
        expected_root=failed_root,
    )
    environment = _scientific_environment(
        source_task=source_task,
        generation_root=generation_root,
        threshold_contract=threshold_contract,
    )
    lock_path = _absolute(
        spec.get("highmem_lock_path"), label="highmem lock path", kind="fresh"
    )
    expected_lock = (
        runtime_root / "locks/comrecgc_common_recourse_highmem.lock"
    ).resolve(strict=False)
    if lock_path != expected_lock:
        raise RepairManifestError(f"highmem lock path must be exact: {expected_lock}")
    flock_bin = _absolute(spec.get("flock_bin"), label="flock_bin", kind="file")
    if not os.access(flock_bin, os.X_OK):
        raise RepairManifestError(f"flock_bin must be executable: {flock_bin}")
    environment.update(
        {
            "COMRECGC_HIGHMEM_LOCK_PATH": str(lock_path),
            "COMRECGC_FLOCK_BIN": str(flock_bin),
            "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup_root),
            "COMRECGC_MIN_CGROUP_FREE_BYTES": str(min_free_bytes),
            "COMRECGC_PROC_ROOT": str(proc_root),
        }
    )
    tasks = [
        _source_gate_task(
            source_key=key,
            source_output=resolved_source_outputs[key],
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            control_root=control_root,
            project_root=project_root,
            required_fix_commit=required_fix,
            fresh_root=fresh_root,
            priority=index + 1,
        )
        for index, key in enumerate((GENERATION_SOURCE_KEY, THRESHOLD_SOURCE_KEY))
    ]
    tasks.append(
        {
            "id": STANDARDIZATION_TASK_ID,
            "dataset": "aids",
            "stage": "AM_COMRECGC_HELDOUT_EVAL",
            "runner_dataset": "paper-cell-aids-comrecgc-am-repair-v3",
            "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
            "depends_on": [GENERATION_GATE_TASK_ID, THRESHOLD_GATE_TASK_ID],
            "resource": "cpu",
            "priority": 20,
            "data_splits": ["test"],
            "manifest_only": False,
            "selector_parameters_frozen": True,
            "read_only_test": True,
            "command": [
                "bash",
                (
                    "{project_root}/scripts/autodl/"
                    "run_comrecgc_standardized_continuation_cpu_highmem.sh"
                ),
            ],
            "input_manifest": "{dep_" + GENERATION_GATE_TASK_ID + "_output}/source_gate.json",
            "config_files": [
                "{dep_" + THRESHOLD_GATE_TASK_ID + "_output}/source_gate.json",
                str(threshold_contract),
            ],
            "expected_output": str(
                fresh_root / "cells/aids/comrecgc/standardized/attempt-{attempt}"
            ),
            "required_output_files": list(STANDARDIZED_REQUIRED_FILES),
            "required_log_marker": "[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset=aids",
            "environment": environment,
            "semantic_failure_markers": [
                "source_closure_changed",
                "live_writer_detected",
                "graph_hash_collision_or_corruption",
                "test leakage",
                "dubious ownership",
            ],
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "controller_id": CONTROLLER_ID,
        "paper_frozen": True,
        "runtime": {
            "max_gpus": 4,
            "stable_idle_seconds": 60,
            "sample_interval_seconds": 5,
            "poll_seconds": 60,
            "min_free_memory_mb": 16000,
            "idle_util_threshold": 10,
            "worker_launcher": "auto",
            "max_cpu_tasks": 1,
            "launch_grace_seconds": 180,
            "max_transient_retries": 0,
            "keep_alive_when_blocked": True,
        },
        "resource_gates": {
            "min_available_ram_gb": 32,
            "min_free_disk_gb": 20,
            "max_cpu_load_fraction": 0.9,
        },
        "aids_comrecgc_repair_v3_contract": {
            "schema_version": SPEC_SCHEMA,
            "spec_path": str(spec_path_resolved),
            "spec_sha256": sha256_file(spec_path_resolved),
            "execution_project_root": str(project_root),
            "execution_commit": fix["execution_head"],
            "verify_comrecgc_checkout_safe_git_fix_commit": required_fix,
            "fresh_output_root": str(fresh_root),
            "source_controller_id": SOURCE_CONTROLLER_ID,
            "source_controller_manifest": str(source_manifest),
            "source_controller_manifest_sha256": source_manifest_value.sha256,
            "source_controller_root": str(source_controller_root),
            "source_evidence": adopted,
            "failed_task_evidence": failure_evidence,
            "cgroup_oom_evidence": oom_evidence,
            "gpu_required": False,
            "device": "cpu",
            "max_cpu_tasks": 1,
            "highmem_lock_path": str(lock_path),
            "flock_bin": str(flock_bin),
            "min_cgroup_free_bytes": min_free_bytes,
            "common_recourse_colocation_forbidden": True,
            "mutagenicity_scientific_task_present": False,
            "mutagenicity_block_reason": "MISSING_TRUE_TRACE_ON_OFF_PARITY",
            "bace_tasks_present": False,
            "taste_tasks_present": False,
            "paper_tasks_present": False,
        },
        "tasks": tasks,
    }
    validation = validate_payload(payload)
    return payload, {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "task_count": validation["task_count"],
        "fresh_output_root": str(fresh_root),
        "gpu_required": False,
        "max_cpu_tasks": 1,
        "scientific_failure": False,
    }


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aids-comrecgc-repair-v3-") as directory:
        manifest_path = Path(directory) / "manifest.json"
        _atomic_json(manifest_path, payload)
        manifest = load_controller_manifest(manifest_path)
    expected = {GENERATION_GATE_TASK_ID, THRESHOLD_GATE_TASK_ID, STANDARDIZATION_TASK_ID}
    if (
        manifest.controller_id != CONTROLLER_ID
        or {task.task_id for task in manifest.tasks} != expected
    ):
        raise RepairManifestError("repair-v3 must contain exactly its three AIDS tasks")
    if len(manifest.tasks) != 3 or int(manifest.runtime.get("max_cpu_tasks", 0)) != 1:
        raise RepairManifestError("repair-v3 max_cpu_tasks/task count is invalid")
    if any(task.resource != "cpu" for task in manifest.tasks):
        raise RepairManifestError("repair-v3 is not CPU-only")
    standard = manifest.by_id[STANDARDIZATION_TASK_ID]
    if (
        standard.environment.get("DEVICE") != "cpu"
        or standard.environment.get("GPU_REQUIRED") != "0"
    ):
        raise RepairManifestError("repair-v3 GPU guard is absent")
    if standard.environment.get("CUDA_VISIBLE_DEVICES") != "":
        raise RepairManifestError("repair-v3 must clear CUDA visibility")
    if not any(
        value.endswith("run_comrecgc_standardized_continuation_cpu_highmem.sh")
        for value in standard.command or ()
    ):
        raise RepairManifestError("repair-v3 does not use the high-memory wrapper")
    contract = _mapping(payload.get("aids_comrecgc_repair_v3_contract"), label="repair contract")
    if (
        contract.get("gpu_required") is not False
        or contract.get("common_recourse_colocation_forbidden") is not True
    ):
        raise RepairManifestError("repair-v3 resource contract is incomplete")
    return {
        "status": "PASS",
        "controller_id": manifest.controller_id,
        "task_count": len(manifest.tasks),
        "task_ids": [task.task_id for task in manifest.tasks],
        "manifest_sha256": manifest.sha256,
        "gpu_required": False,
        "max_cpu_tasks": 1,
    }


def build_manifest(
    *, spec_path: str | Path, output_path: str | Path, proc_root_override: str | Path | None = None
) -> dict[str, Any]:
    destination = _absolute(output_path, label="manifest output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"manifest output must be fresh: {destination}")
    payload, summary = build_payload(spec_path=spec_path, proc_root_override=proc_root_override)
    control_root = Path(str(_read_object(spec_path)["control_root"])).resolve(strict=True)
    expected = (
        control_root / SOURCE_NAMESPACE / "manifests" / f"{CONTROLLER_ID}.json"
    ).resolve(strict=False)
    if destination != expected:
        raise RepairManifestError(f"manifest output must be exact: {expected}")
    validation = validate_payload(payload)
    _atomic_json(destination, payload)
    frozen = load_controller_manifest(destination)
    if frozen.sha256 != validation["manifest_sha256"]:
        destination.unlink(missing_ok=True)
        raise RepairManifestError("published manifest differs from validated bytes")
    return {**summary, "manifest": str(destination), "manifest_sha256": frozen.sha256}


__all__ = [
    "CONTROLLER_ID",
    "GENERATION_SOURCE_KEY",
    "MINIMUM_HEADROOM_BYTES",
    "SOURCE_CONTROLLER_ID",
    "SPEC_SCHEMA",
    "THRESHOLD_SOURCE_KEY",
    "build_manifest",
    "build_payload",
    "publish_source_gate",
    "validate_payload",
    "verify_v2_source",
]
