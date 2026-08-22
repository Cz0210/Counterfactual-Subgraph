"""Build the fresh exact external-memory AIDS ComRecGC repair-v4 controller."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from scripts.autodl.run_four_gpu_recovery_controller import (
    ControllerManifest,
    TaskSpec,
    load_controller_manifest,
)
from src.utils import autodl_aids_comrecgc_repair_v3 as v3
from src.utils.autodl_four_by_four_am_repair import (
    STANDARDIZED_REQUIRED_FILES,
    VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
    verify_fix_ancestry,
)
from src.utils.autodl_four_by_four_repair import RepairManifestError, sha256_file


SPEC_SCHEMA = "aids_comrecgc_external_memory_repair_spec_v4"
CONTROLLER_ID = "four_methods_four_datasets_aids_comrecgc_repair_v4"
SOURCE_NAMESPACE = "four_methods_four_datasets_continuation"
SOURCE_CONTROLLER_ID = v3.SOURCE_CONTROLLER_ID
FAILED_CONTROLLER_ID = v3.CONTROLLER_ID
GENERATION_SOURCE_KEY = v3.GENERATION_SOURCE_KEY
THRESHOLD_SOURCE_KEY = v3.THRESHOLD_SOURCE_KEY
GENERATION_GATE_TASK_ID = "am_v4_source_aids_comrec_generation"
THRESHOLD_GATE_TASK_ID = "am_v4_source_aids_comrec_threshold"
STANDARDIZATION_TASK_ID = "aids_comrecgc_standardized_external_memory"
FAILED_SOURCE_TASK_ID = v3.STANDARDIZATION_TASK_ID
FAILED_SOURCE_RUN_ID = (
    "four_methods_four_datasets_aids_comrecgc_repair_v3-"
    "aids_comrecgc_standardized_cpu_highmem-main-a0"
)
EXTERNAL_MEMORY_FIX_COMMIT = "d5c1d67339df4b9642beaf2b10908ed92bac30de"
EXPECTED_SKLEARN_VERSION = "1.7.2"
EXTERNAL_MAX_RSS_GB = 96
EXTERNAL_QUERY_BLOCK_SIZE = 8
MINIMUM_HEADROOM_BYTES = 128 * 1024**3
PROCESS_TRANSIENT_SIGNALS = ("Signals.SIGKILL: 9", "Signals.SIGTERM: 15")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairManifestError(f"{label} must be one object")
    return value


def _failed_controller(
    *, manifest_path: Path, controller_root: Path, control_root: Path
) -> ControllerManifest:
    namespace = (control_root / SOURCE_NAMESPACE).resolve(strict=True)
    if manifest_path.parent != (namespace / "manifests").resolve(strict=True):
        raise RepairManifestError("repair-v3 manifest is outside its namespace")
    if controller_root != (namespace / FAILED_CONTROLLER_ID).resolve(strict=True):
        raise RepairManifestError("repair-v3 controller root is not exact")
    controller_copy = v3._read_object(controller_root / "controller_manifest.json")
    if (
        controller_copy.get("controller_id") != FAILED_CONTROLLER_ID
        or controller_copy.get("source_manifest") != str(manifest_path)
        or controller_copy.get("source_manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RepairManifestError("repair-v3 controller manifest binding is invalid")
    manifest = load_controller_manifest(manifest_path)
    if manifest.controller_id != FAILED_CONTROLLER_ID:
        raise RepairManifestError("failed controller is not AIDS repair-v3")
    return manifest


def _verify_v3_oom_failure(
    *,
    manifest_path: Path,
    controller_root: Path,
    control_root: Path,
    expected_output_root: Path,
    cgroup_root: Path,
) -> tuple[TaskSpec, dict[str, Any]]:
    manifest = _failed_controller(
        manifest_path=manifest_path,
        controller_root=controller_root,
        control_root=control_root,
    )
    if FAILED_SOURCE_TASK_ID not in manifest.by_id:
        raise RepairManifestError("repair-v3 has no external-memory source failure")
    task = manifest.by_id[FAILED_SOURCE_TASK_ID]
    task_root = controller_root / "tasks" / FAILED_SOURCE_TASK_ID
    state = v3._read_object(task_root / "state.json")
    gate = v3._read_object(task_root / "gate.json")
    instances = _mapping(state.get("instances"), label="failed instances")
    instance = _mapping(instances.get("main"), label="failed main instance")
    gate_runs = gate.get("runs")
    gate_run = (
        gate_runs[0]
        if isinstance(gate_runs, list)
        and len(gate_runs) == 1
        and isinstance(gate_runs[0], Mapping)
        else {}
    )
    run_id = str(instance.get("run_id") or "")
    registry = control_root / "experiment_registry/run_state" / run_id
    exp_state_path = registry / "state.json"
    launch_spec_path = registry / "launch_spec.json"
    exp_state = v3._read_object(exp_state_path)
    launch_spec = v3._read_object(launch_spec_path)
    launch_environment = _mapping(
        launch_spec.get("environment"), label="failed launch environment"
    )
    launch_command = launch_spec.get("command")
    failure_path = expected_output_root / "FAILED.json"
    failure = v3._read_object(failure_path)
    oom = v3._cgroup_oom_evidence(cgroup_root)
    failures: list[str] = []
    if (
        task.dataset != "aids"
        or task.resource != "cpu"
        or task.stage != "AM_COMRECGC_HELDOUT_EVAL"
        or task.data_splits != ("test",)
    ):
        failures.append("task identity/resource")
    if set(instances) != {"main"}:
        failures.append("instance set")
    if (
        state.get("state") != "FAILED"
        or instance.get("state") != "FAILED"
        or instance.get("failure_class") != "EXECUTION"
        or "scientific command exited 1" not in str(instance.get("failure_reason"))
    ):
        failures.append("controller terminal")
    if (
        run_id != FAILED_SOURCE_RUN_ID
        or instance.get("attempt") != 0
        or gate.get("status") != "FAILED"
        or gate.get("task_id") != FAILED_SOURCE_TASK_ID
        or gate.get("reason") != "main:EXECUTION"
        or gate_run.get("run_id") != run_id
        or gate_run.get("state") != "FAILED"
        or gate_run.get("attempt") != 0
    ):
        failures.append("gate/run identity")
    for actual in (
        instance.get("expected_output"),
        gate_run.get("expected_output"),
        launch_spec.get("expected_output"),
    ):
        try:
            if Path(str(actual)).resolve(strict=True) != expected_output_root:
                failures.append("failed output identity")
                break
        except Exception:
            failures.append("failed output identity")
            break
    if (
        exp_state.get("state") != "FAILED"
        or exp_state.get("run_id") != run_id
        or exp_state.get("exit_code") != 1
        or exp_state.get("dataset") != task.runner_dataset
        or exp_state.get("stage") != task.runner_stage
        or exp_state.get("pid") != instance.get("worker_pid")
        or exp_state.get("child_pid") != instance.get("child_pid")
        or exp_state.get("log_path") != instance.get("log_path")
    ):
        failures.append("exp-run terminal")
    if (
        launch_spec.get("run_id") != run_id
        or launch_spec.get("dataset") != task.runner_dataset
        or launch_spec.get("stage") != task.runner_stage
        or not isinstance(launch_command, list)
        or len(launch_command) != 2
        or launch_command[0] != "bash"
        or not str(launch_command[1]).endswith(
            "/scripts/autodl/run_comrecgc_standardized_continuation_cpu_highmem.sh"
        )
        or launch_environment.get("DATASET") != "aids"
        or launch_environment.get("DEVICE") != "cpu"
        or launch_environment.get("GPU_REQUIRED") != "0"
        or launch_environment.get("OUTPUT_ROOT") != str(expected_output_root)
    ):
        failures.append("launch identity")
    message = str(failure.get("message") or "")
    if (
        failure.get("status") != "FAILED"
        or failure.get("dataset") != "aids"
        or failure.get("error_class") != "CalledProcessError"
        or failure.get("output_root") != str(expected_output_root)
        or "run_common_recourse.py" not in message
        or "Signals.SIGKILL: 9" not in message
    ):
        failures.append("SIGKILL evidence")
    if (
        oom.get("peak_reached_limit") is not True
        or int(oom.get("oom_kill_count", 0)) < 2
        or int(oom.get("memory_failcnt", 0)) < 1
        or int(oom.get("memory_max_usage_in_bytes", 0))
        < int(oom.get("memory_limit_in_bytes", 0))
    ):
        failures.append("second cgroup OOM evidence")
    if failures:
        raise RepairManifestError(
            f"repair-v3 failure is not the reviewed second OOM: {failures}"
        )
    return task, {
        "status": "PASS",
        "source_controller_id": FAILED_CONTROLLER_ID,
        "source_controller_manifest": str(manifest_path),
        "source_controller_manifest_sha256": manifest.sha256,
        "source_task_id": FAILED_SOURCE_TASK_ID,
        "source_run_id": run_id,
        "source_failed_output": str(expected_output_root),
        "source_failed_json_sha256": sha256_file(failure_path),
        "source_exp_run_state": str(exp_state_path.resolve(strict=True)),
        "source_exp_run_state_sha256": sha256_file(exp_state_path),
        "source_launch_spec": str(launch_spec_path.resolve(strict=True)),
        "source_launch_spec_sha256": sha256_file(launch_spec_path),
        "source_exit_code": 1,
        "source_signal": "SIGKILL",
        "cgroup_oom_jointly_verified": True,
        "oom_kill_count_at_v4_build": int(oom["oom_kill_count"]),
        "scientific_failure": False,
    }


def _source_gate_task(
    *,
    source_key: str,
    source_output: Path,
    source_manifest: Path,
    source_controller_root: Path,
    control_root: Path,
    project_root: Path,
    fresh_root: Path,
    priority: int,
) -> dict[str, Any]:
    is_threshold = source_key == THRESHOLD_SOURCE_KEY
    task_id = THRESHOLD_GATE_TASK_ID if is_threshold else GENERATION_GATE_TASK_ID
    return {
        "id": task_id,
        "dataset": "aids" if is_threshold else "repair-source-audit",
        "stage": (
            "AM_COMRECGC_THRESHOLD_FREEZE"
            if is_threshold
            else "FOUR_BY_FOUR_AM_REPAIR_SOURCE_ADOPTION"
        ),
        "runner_dataset": f"am-v4-source-{source_key}",
        "runner_stage": "FOUR_BY_FOUR_AM_REPAIR_SOURCE_GATE",
        "depends_on": [],
        "resource": "cpu",
        "priority": priority,
        "data_splits": [],
        "manifest_only": True,
        "freezes_selector": is_threshold,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/build_aids_comrecgc_repair_v4_manifest.py",
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
            "--output-dir",
            "{task_output}",
        ],
        "input_manifest": str(source_output / "source_gate.json"),
        "config_files": [str(source_output / "source_gate.json")],
        "expected_output": str(
            fresh_root / f"source-adoptions/{source_key}/attempt-{{attempt}}"
        ),
        "required_output_files": ["source_gate.json", "PASS"],
        "required_log_marker": (
            f"[AIDS_COMRECGC_REPAIR_V4_SOURCE_PASS] source={source_key}"
        ),
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        },
    }


def verify_source(
    *,
    source_key: str,
    source_manifest: str | Path,
    source_controller_root: str | Path,
    control_root: str | Path,
    expected_output_root: str | Path,
    project_root: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    evidence = v3.verify_v2_source(
        source_key=source_key,
        source_manifest=source_manifest,
        source_controller_root=source_controller_root,
        control_root=control_root,
        expected_output_root=expected_output_root,
        project_root=project_root,
        required_fix_commit=VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
        proc_root=proc_root,
    )
    external_fix = verify_fix_ancestry(
        project_root=project_root,
        required_fix_commit=EXTERNAL_MEMORY_FIX_COMMIT,
    )
    return {
        "schema_version": "aids_comrecgc_repair_v4_source_terminal",
        "status": "PASS",
        "source_key": source_key,
        "dataset": "aids",
        "v3_source_evidence": evidence,
        "semantic": evidence["semantic"],
        "external_memory_fix_gate": external_fix,
        "verified_at": _utc_now(),
    }


def publish_source_gate(
    *, source_key: str, evidence: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    if evidence.get("status") != "PASS" or evidence.get("source_key") != source_key:
        raise RepairManifestError("cannot publish mismatched v4 source evidence")
    destination = v3._absolute(output_dir, label="source output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"source output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o755)
    payload = {
        "schema_version": "aids_comrecgc_repair_v4_source_gate",
        "status": "PASS",
        "source_key": source_key,
        "evidence": dict(evidence),
        "published_at": _utc_now(),
    }
    v3._atomic_json(destination / "source_gate.json", payload)
    descriptor = os.open(
        destination / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def verify_same_root_resume_failure(
    *, output_root: str | Path, exit_code: int
) -> dict[str, Any]:
    """Authorize one supervisor retry only for a lost common-recourse child."""

    root = v3._absolute(output_root, label="resume output", kind="dir")
    if root.is_symlink() or (root / "PASS").exists():
        raise RepairManifestError("completed or symlinked output cannot resume")
    contract = v3._read_object(root / "continuation_resume_contract.json")
    if (
        contract.get("schema_version")
        != "comrecgc_standardized_stage_resume_v1"
        or contract.get("dataset") != "aids"
        or contract.get("output_root") != str(root)
        or contract.get("common_recourse_engine") != "external_memory_exact_v1"
        or int(contract.get("external_query_block_size", -1))
        != EXTERNAL_QUERY_BLOCK_SIZE
        or float(contract.get("external_max_rss_gb", -1)) != EXTERNAL_MAX_RSS_GB
        or contract.get("expected_sklearn_version") != EXPECTED_SKLEARN_VERSION
    ):
        raise RepairManifestError("same-root resume contract is not exact v4")
    checkpoint = v3._read_object(
        root / "stage_checkpoints/common_recourse.json"
    )
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("stage") != "common_recourse"
        or checkpoint.get("status") not in {"RUNNING", "FAILED"}
        or not str(checkpoint.get("argv_sha256") or "")
    ):
        raise RepairManifestError("common-recourse stage is not interrupted")
    failure_path = root / "FAILED.json"
    failure_message = ""
    if failure_path.exists():
        failure = v3._read_object(failure_path)
        failure_message = str(failure.get("message") or "")
        if (
            failure.get("status") != "FAILED"
            or failure.get("dataset") != "aids"
            or not any(signal in failure_message for signal in PROCESS_TRANSIENT_SIGNALS)
        ):
            raise RepairManifestError("failure is semantic/non-process and cannot resume")
    elif int(exit_code) not in {137, 143}:
        raise RepairManifestError("missing signal failure evidence for same-root resume")
    forbidden = (
        "RSS_BUDGET_EXCEEDED",
        "SKLEARN_VERSION_MISMATCH",
        "checkpoint scientific identity mismatch",
        "SOURCE_CLOSURE_CHANGED",
        "test leakage",
    )
    if any(marker.lower() in failure_message.lower() for marker in forbidden):
        raise RepairManifestError("semantic/contract failure cannot be supervised")
    return {
        "schema_version": "aids_comrecgc_repair_v4_same_root_resume_gate",
        "status": "PASS",
        "output_root": str(root),
        "exit_code": int(exit_code),
        "failure_json": str(failure_path) if failure_path.exists() else None,
        "failure_signal_verified": (
            next(
                (
                    signal
                    for signal in PROCESS_TRANSIENT_SIGNALS
                    if signal in failure_message
                ),
                f"shell_exit_{int(exit_code)}",
            )
        ),
        "stage_checkpoint_sha256": sha256_file(
            root / "stage_checkpoints/common_recourse.json"
        ),
        "resume_contract_sha256": sha256_file(
            root / "continuation_resume_contract.json"
        ),
        "bounded_same_root_resume": True,
        "verified_at": _utc_now(),
    }


def _load_spec(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = v3._absolute(path, label="spec", kind="file")
    spec = v3._read_object(source)
    if spec.get("schema_version") != SPEC_SCHEMA or spec.get("controller_id") != CONTROLLER_ID:
        raise RepairManifestError("invalid repair-v4 spec identity")
    if spec.get("paper_frozen") is not True or spec.get("run_tastemolnet") != 0:
        raise RepairManifestError("paper must remain frozen and Taste disabled")
    if any(key in spec for key in ("mutagenicity", "bace", "taste", "paper")):
        raise RepairManifestError("repair-v4 spec contains a forbidden route")
    return source, spec


def build_payload(
    *, spec_path: str | Path, proc_root_override: str | Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path_resolved, spec = _load_spec(spec_path)
    runtime_root = v3._absolute(spec.get("runtime_root"), label="runtime", kind="dir")
    control_root = v3._absolute(spec.get("control_root"), label="control", kind="dir")
    project_root = v3._absolute(spec.get("project_root"), label="project", kind="dir")
    python = v3._absolute(spec.get("python"), label="python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError("configured Python is not executable")
    fresh_root = v3._absolute(spec.get("fresh_output_root"), label="fresh root", kind="fresh")
    if fresh_root.exists() or fresh_root.is_symlink():
        raise RepairManifestError("repair-v4 output root must be fresh")
    try:
        fresh_root.relative_to((runtime_root / "outputs/autodl").resolve(strict=False))
    except ValueError as exc:
        raise RepairManifestError("repair-v4 root must stay below runtime outputs") from exc
    destination = (control_root / SOURCE_NAMESPACE / CONTROLLER_ID).resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise RepairManifestError("repair-v4 controller root already exists")
    safe_fix = verify_fix_ancestry(
        project_root=project_root,
        required_fix_commit=VERIFY_COMRECGC_SAFE_GIT_FIX_COMMIT,
    )
    external_fix = verify_fix_ancestry(
        project_root=project_root,
        required_fix_commit=EXTERNAL_MEMORY_FIX_COMMIT,
    )
    proc_root = v3._absolute(
        proc_root_override if proc_root_override is not None else spec.get("proc_root", "/proc"),
        label="proc_root",
        kind="dir",
    )
    cgroup_root = v3._absolute(
        spec.get("cgroup_memory_root"), label="cgroup", kind="dir"
    )
    min_free = int(spec.get("min_cgroup_free_bytes", 0))
    if min_free < MINIMUM_HEADROOM_BYTES:
        raise RepairManifestError("repair-v4 cgroup headroom is below 128 GiB")
    if int(spec.get("external_max_rss_gb", -1)) != EXTERNAL_MAX_RSS_GB:
        raise RepairManifestError("repair-v4 RSS budget must be exactly 96 GiB")
    if int(spec.get("external_query_block_size", -1)) != EXTERNAL_QUERY_BLOCK_SIZE:
        raise RepairManifestError("repair-v4 query block must be exactly 8")
    if spec.get("expected_sklearn_version") != EXPECTED_SKLEARN_VERSION:
        raise RepairManifestError("repair-v4 sklearn version is not frozen")

    source = _mapping(spec.get("source_controller"), label="source controller")
    source_manifest = v3._absolute(source.get("manifest"), label="source manifest", kind="file")
    source_controller_root = v3._absolute(source.get("root"), label="source root", kind="dir")
    source_outputs = _mapping(spec.get("source_outputs"), label="source outputs")
    if set(source_outputs) != {GENERATION_SOURCE_KEY, THRESHOLD_SOURCE_KEY}:
        raise RepairManifestError("repair-v4 requires exactly two source outputs")
    adopted: dict[str, dict[str, Any]] = {}
    resolved_outputs: dict[str, Path] = {}
    for key in (GENERATION_SOURCE_KEY, THRESHOLD_SOURCE_KEY):
        output = v3._absolute(source_outputs[key], label=f"source.{key}", kind="dir")
        adopted[key] = verify_source(
            source_key=key,
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            control_root=control_root,
            expected_output_root=output,
            project_root=project_root,
            proc_root=proc_root,
        )
        resolved_outputs[key] = output
    generation_root = v3._absolute(
        adopted[GENERATION_SOURCE_KEY]["semantic"].get("generation_root"),
        label="generation root",
        kind="dir",
    )
    threshold_contract = v3._absolute(
        adopted[THRESHOLD_SOURCE_KEY]["semantic"].get("threshold_contract"),
        label="threshold contract",
        kind="file",
    )

    failed = _mapping(spec.get("failed_controller"), label="failed controller")
    failed_manifest = v3._absolute(failed.get("manifest"), label="failed manifest", kind="file")
    failed_controller_root = v3._absolute(failed.get("root"), label="failed root", kind="dir")
    failed_output = v3._absolute(
        spec.get("failed_aids_output_root"), label="failed output", kind="dir"
    )
    failed_task, failure_evidence = _verify_v3_oom_failure(
        manifest_path=failed_manifest,
        controller_root=failed_controller_root,
        control_root=control_root,
        expected_output_root=failed_output,
        cgroup_root=cgroup_root,
    )
    environment = v3._scientific_environment(
        source_task=failed_task,
        generation_root=generation_root,
        threshold_contract=threshold_contract,
    )
    lock_path = v3._absolute(spec.get("highmem_lock_path"), label="lock", kind="fresh")
    expected_lock = (runtime_root / "locks/comrecgc_common_recourse_highmem.lock").resolve(
        strict=False
    )
    if lock_path != expected_lock:
        raise RepairManifestError("repair-v4 highmem lock path differs")
    flock_bin = v3._absolute(spec.get("flock_bin"), label="flock", kind="file")
    environment.update(
        {
            "COMRECGC_HIGHMEM_LOCK_PATH": str(lock_path),
            "COMRECGC_FLOCK_BIN": str(flock_bin),
            "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup_root),
            "COMRECGC_MIN_CGROUP_FREE_BYTES": str(min_free),
            "COMRECGC_PROC_ROOT": str(proc_root),
            "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
            "COMRECGC_EXTERNAL_MAX_RSS_GB": str(EXTERNAL_MAX_RSS_GB),
            "COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE": str(EXTERNAL_QUERY_BLOCK_SIZE),
            "COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS": "1",
            "COMRECGC_EXPECTED_SKLEARN_VERSION": EXPECTED_SKLEARN_VERSION,
            # Enabled from the first invocation: a fresh root is still mandatory
            # initially, while an interrupted attempt may later re-enter only
            # through its hash-bound stage and external-memory checkpoints.
            "COMRECGC_COMMON_RECOURSE_RESUME": "1",
            "AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES": "1",
            "AIDS_COMRECGC_V4_TEST_MODE": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    tasks = [
        _source_gate_task(
            source_key=key,
            source_output=resolved_outputs[key],
            source_manifest=source_manifest,
            source_controller_root=source_controller_root,
            control_root=control_root,
            project_root=project_root,
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
            "runner_dataset": "paper-cell-aids-comrecgc-repair-v4",
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
                "{project_root}/scripts/autodl/run_aids_comrecgc_repair_v4_supervisor.sh",
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
                "SOURCE_CLOSURE_CHANGED",
                "LIVE_WRITER_DETECTED",
                "SKLEARN_VERSION_MISMATCH",
                "RSS_BUDGET_EXCEEDED",
                "checkpoint scientific identity mismatch",
                "test leakage",
            ],
        }
    )
    payload = {
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
            "min_available_ram_gb": 128,
            "min_free_disk_gb": 100,
            "max_cpu_load_fraction": 0.95,
        },
        "aids_comrecgc_repair_v4_contract": {
            "schema_version": SPEC_SCHEMA,
            "spec_path": str(spec_path_resolved),
            "spec_sha256": sha256_file(spec_path_resolved),
            "execution_project_root": str(project_root),
            "execution_commit": external_fix["execution_head"],
            "safe_git_fix_gate": safe_fix,
            "external_memory_fix_gate": external_fix,
            "fresh_output_root": str(fresh_root),
            "source_controller_id": SOURCE_CONTROLLER_ID,
            "source_evidence": adopted,
            "failed_controller_id": FAILED_CONTROLLER_ID,
            "failed_task_evidence": failure_evidence,
            "gpu_required": False,
            "device": "cpu",
            "max_cpu_tasks": 1,
            "highmem_lock_path": str(lock_path),
            "min_cgroup_free_bytes": min_free,
            "common_recourse_engine": "external_memory_exact_v1",
            "external_max_rss_gb": EXTERNAL_MAX_RSS_GB,
            "external_query_block_size": EXTERNAL_QUERY_BLOCK_SIZE,
            "external_checkpoint_interval_blocks": 1,
            "expected_sklearn_version": EXPECTED_SKLEARN_VERSION,
            "parameters": {
                "theta": 0.1,
                "delta": 0.02,
                "recourse_size": 100,
                "cf_size": 100000,
                "cluster_size": 3,
                "seed": 0,
            },
            "scientific_budget_reduced": False,
            "legacy_roots_mutated": False,
            "common_recourse_colocation_forbidden": True,
            "mutagenicity_tasks_present": False,
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
        "common_recourse_engine": "external_memory_exact_v1",
    }


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aids-comrecgc-repair-v4-") as directory:
        path = Path(directory) / "manifest.json"
        v3._atomic_json(path, payload)
        manifest = load_controller_manifest(path)
    expected = {
        GENERATION_GATE_TASK_ID,
        THRESHOLD_GATE_TASK_ID,
        STANDARDIZATION_TASK_ID,
    }
    if manifest.controller_id != CONTROLLER_ID or set(manifest.by_id) != expected:
        raise RepairManifestError("repair-v4 must contain exactly three AIDS tasks")
    if len(manifest.tasks) != 3 or int(manifest.runtime.get("max_cpu_tasks", 0)) != 1:
        raise RepairManifestError("repair-v4 task/CPU concurrency mismatch")
    if any(task.resource != "cpu" for task in manifest.tasks):
        raise RepairManifestError("repair-v4 is not CPU-only")
    standard = manifest.by_id[STANDARDIZATION_TASK_ID]
    environment = standard.environment
    required = {
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_EXTERNAL_MAX_RSS_GB": "96",
        "COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE": "8",
        "COMRECGC_EXPECTED_SKLEARN_VERSION": "1.7.2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES": "1",
        "AIDS_COMRECGC_V4_TEST_MODE": "0",
    }
    if any(environment.get(key) != value for key, value in required.items()):
        raise RepairManifestError("repair-v4 external-memory environment is incomplete")
    contract = _mapping(
        payload.get("aids_comrecgc_repair_v4_contract"), label="v4 contract"
    )
    if (
        contract.get("scientific_budget_reduced") is not False
        or contract.get("legacy_roots_mutated") is not False
        or contract.get("common_recourse_colocation_forbidden") is not True
        or contract.get("parameters")
        != {
            "theta": 0.1,
            "delta": 0.02,
            "recourse_size": 100,
            "cf_size": 100000,
            "cluster_size": 3,
            "seed": 0,
        }
    ):
        raise RepairManifestError("repair-v4 scientific contract changed")
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
    *,
    spec_path: str | Path,
    output_path: str | Path,
    proc_root_override: str | Path | None = None,
) -> dict[str, Any]:
    destination = v3._absolute(output_path, label="manifest output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"manifest output must be fresh: {destination}")
    payload, summary = build_payload(
        spec_path=spec_path, proc_root_override=proc_root_override
    )
    control_root = Path(str(v3._read_object(spec_path)["control_root"])).resolve(
        strict=True
    )
    expected = (
        control_root / SOURCE_NAMESPACE / "manifests" / f"{CONTROLLER_ID}.json"
    ).resolve(strict=False)
    if destination != expected:
        raise RepairManifestError(f"manifest output must be exact: {expected}")
    validation = validate_payload(payload)
    v3._atomic_json(destination, payload)
    frozen = load_controller_manifest(destination)
    if frozen.sha256 != validation["manifest_sha256"]:
        destination.unlink(missing_ok=True)
        raise RepairManifestError("published repair-v4 manifest changed after validation")
    return {**summary, "manifest": str(destination), "manifest_sha256": frozen.sha256}


__all__ = [
    "CONTROLLER_ID",
    "EXTERNAL_MEMORY_FIX_COMMIT",
    "GENERATION_SOURCE_KEY",
    "MINIMUM_HEADROOM_BYTES",
    "SPEC_SCHEMA",
    "THRESHOLD_SOURCE_KEY",
    "build_manifest",
    "build_payload",
    "publish_source_gate",
    "validate_payload",
    "verify_source",
    "verify_same_root_resume_failure",
]
