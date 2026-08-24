"""Build the fresh terminal-only AIDS ComRecGC exact-route v5 controller."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from scripts.autodl.run_four_gpu_recovery_controller import (
    load_controller_manifest,
)
from scripts.autodl.verify_aids_comrecgc_v5_process_set import verify_process_set
from src.baselines.comrecgc.external_memory_recourse import (
    PAIR_STORE_SCHEMA,
    _file_stat_identity,
    _pair_store_source_root_guard,
    _validate_pair_store_manifest,
)
from src.utils import autodl_aids_comrecgc_repair_v3 as v3
from src.utils.autodl_aids_comrecgc_repair_v4 import (
    CONTROLLER_ID as V4_CONTROLLER_ID,
    STANDARDIZATION_TASK_ID as V4_TASK_ID,
)
from src.utils.autodl_four_by_four_am_repair import STANDARDIZED_REQUIRED_FILES
from src.utils.autodl_four_by_four_repair import (
    RepairManifestError,
    sha256_file,
)


SPEC_SCHEMA = "aids_comrecgc_exact_route_v5_spec_v1"
CONTROLLER_ID = "four_methods_four_datasets_aids_comrecgc_exact_route_v5"
TASK_ID = "aids_comrecgc_standardized_exact_route_v5"
SELECTOR_TASK_ID = "aids_comrecgc_exact_route_v5_selector_freeze"
SOURCE_NAMESPACE = "four_methods_four_datasets_continuation"
REVIEWED_CORE_COMMIT = "645c6e51b7abcdc5dd4a9e0a1226d71d020880da"
ROUTE_RELEASE_COMMIT = "e75b6e8160e07c869c558080259b4b05695f76d7"
MINIMUM_CGROUP_FREE_BYTES = 128 * 1024**3
EXPECTED_PARENT_COUNT = 1_283
EXPECTED_CANDIDATE_COUNT = 71_642
EXPECTED_PAIR_COUNT = EXPECTED_PARENT_COUNT * EXPECTED_CANDIDATE_COUNT
EXPECTED_VECTOR_DIM = 64
EXPECTED_PARAMETERS = {
    "theta": 0.1,
    "delta": 0.02,
    "recourse_size": 100,
    "cf_size": 100000,
    "cluster_size": 3,
    "seed": 0,
}


def _read_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise RepairManifestError(f"{label} must be a physical nonempty file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RepairManifestError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RepairManifestError(f"{label} must be one JSON object")
    return payload


def _absolute(value: Any, *, label: str, kind: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        raise RepairManifestError(f"{label} must be absolute")
    if kind == "fresh":
        return path.resolve(strict=False)
    if path.is_symlink():
        raise RepairManifestError(f"{label} may not be a symlink")
    resolved = path.resolve(strict=True)
    if kind == "file" and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise RepairManifestError(f"{label} must be a nonempty file")
    if kind == "dir" and not resolved.is_dir():
        raise RepairManifestError(f"{label} must be a directory")
    return resolved


def _git_head(project_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepairManifestError("cannot resolve execution HEAD") from exc
    if len(value) != 40:
        raise RepairManifestError("execution HEAD is not a full SHA")
    return value


def _require_ancestor(project_root: Path, commit: str) -> dict[str, str]:
    head = _git_head(project_root)
    completed = subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", commit, head],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RepairManifestError(
            f"required AIDS exact-route commit is not an ancestor: {commit}"
        )
    return {"required_commit": commit, "execution_head": head, "is_ancestor": "true"}


def _terminal_source_evidence(
    *, root: Path, expected_sha256: str, proc_root: Path
) -> dict[str, Any]:
    if root.is_symlink() or not root.resolve(strict=True).is_dir():
        raise RepairManifestError("terminal pair-store root must be a physical directory")
    root = root.resolve(strict=True)
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_symlink():
        raise RepairManifestError("terminal pair-store manifest may not be a symlink")
    manifest = _read_object(manifest_path, label="terminal pair-store manifest")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != expected_sha256:
        raise RepairManifestError("terminal pair-store manifest SHA256 mismatch")
    if (
        manifest.get("schema_version") != PAIR_STORE_SCHEMA
        or manifest.get("run_complete") is not True
        or manifest.get("candidate_major_parent_minor_order") is not True
        or int(manifest.get("row_count", -1)) != EXPECTED_PAIR_COUNT
        or int(manifest.get("vector_dim", -1)) != EXPECTED_VECTOR_DIM
        or manifest.get("vectors_dtype") != "float32"
    ):
        raise RepairManifestError("terminal pair-store scientific shape is not AIDS v5")
    identity = manifest.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise RepairManifestError("terminal pair-store scientific identity is missing")
    parameters = identity.get("parameters")
    if (
        identity.get("dataset") != "aids"
        or identity.get("mode") != "full"
        or int(identity.get("parent_count", -1)) != EXPECTED_PARENT_COUNT
        or int(identity.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or identity.get("pair_order") != "candidate_major_parent_minor"
        or identity.get("device") != "cpu"
        or parameters != EXPECTED_PARAMETERS
    ):
        raise RepairManifestError("terminal pair-store scientific identity changed")
    guard = _pair_store_source_root_guard(
        pair_store_root=root,
        source_owner_root=root,
        proc_root=proc_root,
        reject_partial_files=True,
    )
    _validate_pair_store_manifest(manifest_path, manifest)
    pairs_path = Path(str(manifest["pairs_path"])).resolve(strict=True)
    vectors_path = Path(str(manifest["vectors_path"])).resolve(strict=True)
    return {
        "status": "PASS",
        "source_root": str(root),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha,
        "pairs_path": str(pairs_path),
        "pairs_sha256": str(manifest["pairs_sha256"]),
        "pairs_stat": _file_stat_identity(pairs_path),
        "vectors_path": str(vectors_path),
        "vectors_sha256": str(manifest["vectors_sha256"]),
        "vectors_stat": _file_stat_identity(vectors_path),
        "row_count": EXPECTED_PAIR_COUNT,
        "parent_count": EXPECTED_PARENT_COUNT,
        "candidate_count": EXPECTED_CANDIDATE_COUNT,
        "vector_dim": EXPECTED_VECTOR_DIM,
        "source_guard": guard,
    }


def _base_environment(
    *, manifest_path: Path, expected_sha256: str
) -> tuple[dict[str, str], dict[str, Any]]:
    if sha256_file(manifest_path) != expected_sha256:
        raise RepairManifestError("base repair-v4 manifest SHA256 mismatch")
    manifest = load_controller_manifest(manifest_path)
    if manifest.controller_id != V4_CONTROLLER_ID or V4_TASK_ID not in manifest.by_id:
        raise RepairManifestError("base manifest is not the exact AIDS repair-v4 owner")
    task = manifest.by_id[V4_TASK_ID]
    environment = dict(task.environment)
    required = {
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
    }
    if any(environment.get(key) != value for key, value in required.items()):
        raise RepairManifestError("base repair-v4 scientific environment changed")
    for key in (
        "SOURCE_GENERATION_ROOT",
        "COMRECGC_UPSTREAM_ROOT",
        "DATASET_DIR",
        "SOURCE_CSV",
        "DISTANCE_CHECKPOINT",
        "DATASET_CSV",
        "TEACHER_PATH",
        "MOLCLR_ROOT",
        "MOLCLR_CHECKPOINT",
        "THRESHOLDS_PATH",
    ):
        value = Path(environment.get(key, "")).expanduser()
        if not value.is_absolute() or value.is_symlink() or not value.resolve(strict=True).exists():
            raise RepairManifestError(f"base scientific input is unavailable: {key}")
    environment.pop("AIDS_COMRECGC_V4_MAX_SAME_ROOT_RESUMES", None)
    environment.pop("AIDS_COMRECGC_V4_TEST_MODE", None)
    return environment, {
        "controller_id": manifest.controller_id,
        "task_id": task.task_id,
        "manifest": str(manifest_path),
        "manifest_sha256": expected_sha256,
    }


def build_payload(*, spec_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _absolute(spec_path, label="v5 spec", kind="file")
    spec = _read_object(source, label="v5 spec")
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("controller_id") != CONTROLLER_ID
        or spec.get("paper_frozen") is not True
        or spec.get("run_tastemolnet") != 0
    ):
        raise RepairManifestError("invalid AIDS exact-route v5 spec identity")
    project_root = _absolute(spec.get("project_root"), label="project root", kind="dir")
    execution_commit = str(spec.get("execution_commit") or "")
    if _git_head(project_root) != execution_commit:
        raise RepairManifestError("v5 spec is not bound to execution HEAD")
    core_gate = _require_ancestor(project_root, REVIEWED_CORE_COMMIT)
    release_gate = _require_ancestor(project_root, ROUTE_RELEASE_COMMIT)
    runtime_root = _absolute(spec.get("runtime_root"), label="runtime root", kind="dir")
    control_root = _absolute(spec.get("control_root"), label="control root", kind="dir")
    proc_root = _absolute(spec.get("proc_root"), label="proc root", kind="dir")
    cgroup_root = _absolute(
        spec.get("cgroup_memory_root"), label="cgroup memory root", kind="dir"
    )
    minimum_free = int(spec.get("min_cgroup_free_bytes", 0))
    if minimum_free < MINIMUM_CGROUP_FREE_BYTES:
        raise RepairManifestError("AIDS v5 cgroup headroom is below 128 GiB")
    limit = int((cgroup_root / "memory.limit_in_bytes").read_text().strip())
    usage = int((cgroup_root / "memory.usage_in_bytes").read_text().strip())
    if usage >= limit or limit - usage < minimum_free:
        raise RepairManifestError("AIDS v5 live cgroup headroom gate failed")
    python = _absolute(spec.get("python"), label="python", kind="file")
    if not os.access(python, os.X_OK):
        raise RepairManifestError("configured Python is not executable")
    flock_bin = _absolute(spec.get("flock_bin"), label="flock", kind="file")
    if not os.access(flock_bin, os.X_OK):
        raise RepairManifestError("configured flock is not executable")
    scratch_root = _absolute(
        spec.get("local_scratch_root"), label="local scratch root", kind="dir"
    )
    route_lock = _absolute(spec.get("route_lock_path"), label="route lock", kind="fresh")
    try:
        route_lock.relative_to(scratch_root)
    except ValueError as exc:
        raise RepairManifestError("route lock must stay below local scratch root") from exc
    fresh_root = _absolute(spec.get("fresh_output_root"), label="fresh output", kind="fresh")
    if fresh_root.exists() or fresh_root.is_symlink():
        raise RepairManifestError("AIDS v5 output root must be fresh")
    try:
        fresh_root.relative_to((runtime_root / "outputs/autodl").resolve(strict=False))
    except ValueError as exc:
        raise RepairManifestError("AIDS v5 output must stay below runtime outputs") from exc
    controller_root = (
        control_root / SOURCE_NAMESPACE / CONTROLLER_ID
    ).resolve(strict=False)
    if controller_root.exists() or controller_root.is_symlink():
        raise RepairManifestError("AIDS v5 controller root already exists")
    base_manifest = _absolute(
        spec.get("base_v4_manifest"), label="base repair-v4 manifest", kind="file"
    )
    base_sha = str(spec.get("base_v4_manifest_sha256") or "")
    environment, base_evidence = _base_environment(
        manifest_path=base_manifest, expected_sha256=base_sha
    )
    terminal_root = _absolute(
        spec.get("terminal_pair_store_root"), label="terminal pair-store root", kind="dir"
    )
    terminal = _terminal_source_evidence(
        root=terminal_root,
        expected_sha256=str(spec.get("terminal_pair_store_manifest_sha256") or ""),
        proc_root=proc_root,
    )
    allowed_old_raw = spec.get("allowed_old_read_only_process")
    if not isinstance(allowed_old_raw, Mapping):
        raise RepairManifestError("allowed old read-only process identity is missing")
    old_pid = int(allowed_old_raw.get("pid", 0))
    old_start_ticks = int(allowed_old_raw.get("start_ticks", 0))
    old_cmdline_sha256 = str(allowed_old_raw.get("cmdline_sha256") or "")
    old_output_root = _absolute(
        allowed_old_raw.get("output_root"), label="old v4 output root", kind="dir"
    )
    old_project_root = _absolute(
        allowed_old_raw.get("project_root"), label="old v4 project root", kind="dir"
    )
    try:
        terminal_root.relative_to(old_output_root)
    except ValueError as exc:
        raise RepairManifestError("terminal pair store is outside the bound old output") from exc
    process_gate = verify_process_set(
        proc_root=proc_root,
        allowed_pid=old_pid,
        allowed_start_ticks=old_start_ticks,
        allowed_cmdline_sha256=old_cmdline_sha256,
        allowed_output_root=old_output_root,
        allowed_project_root=old_project_root,
    )
    if (
        process_gate.get("process_set_status")
        != "ALLOWED_OLD_READ_ONLY_PROCESS_PRESENT"
        or int(process_gate.get("active_common_recourse_count", -1)) != 1
    ):
        raise RepairManifestError("old read-only process must be present at v5 build")
    generation_manifest = (
        Path(environment["SOURCE_GENERATION_ROOT"]) / "run_manifest.json"
    ).resolve(strict=True)
    pair_manifest = _read_object(
        terminal["source_manifest"], label="terminal pair-store manifest"
    )
    pair_identity = pair_manifest["scientific_identity"]
    if (
        pair_identity.get("generation_manifest_sha256")
        != sha256_file(generation_manifest)
        or pair_identity.get("distance_checkpoint_sha256")
        != sha256_file(environment["DISTANCE_CHECKPOINT"])
    ):
        raise RepairManifestError("terminal pair store does not match frozen v4 inputs")
    environment.update(
        {
            "AUTODL_PYTHON": "{python}",
            "OUTPUT_ROOT": "{task_output}",
            "DEVICE": "cpu",
            "GPU_REQUIRED": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
            "COMRECGC_COMMON_RECOURSE_RESUME": "1",
            "COMRECGC_EXTERNAL_MAX_RSS_GB": "96",
            "COMRECGC_EXTERNAL_QUERY_BLOCK_SIZE": "8",
            "COMRECGC_EXTERNAL_CHECKPOINT_INTERVAL_BLOCKS": "1",
            "COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE": (
                "all_core_one_component_adaptive_anchor_v1"
            ),
            "COMRECGC_EXTERNAL_SHORTCUT_SEED_COUNT": "3",
            "COMRECGC_EXTERNAL_SHORTCUT_FAILURE_CAP": "4096",
            "COMRECGC_EXTERNAL_SHORTCUT_QUERY_BLOCK_SIZE": "65536",
            "COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES": "0",
            "COMRECGC_EXTERNAL_SUMMARY_BLOCK_SIZE": "65536",
            "COMRECGC_EXPECTED_SKLEARN_VERSION": "1.7.2",
            "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL": "1",
            "COMRECGC_EXTERNAL_PAIR_STORE_AUTO_ROOT": str(terminal_root),
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT": str(terminal_root),
            "COMRECGC_EXTERNAL_VECTOR_CACHE_MIN_FREE_GB": "3",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_PROC_ROOT": str(proc_root),
            "COMRECGC_EXTERNAL_ROUTE_LOCK": str(route_lock),
            "COMRECGC_CGROUP_MEMORY_ROOT": str(cgroup_root),
            "AIDS_COMRECGC_V5_MIN_CGROUP_FREE_BYTES": str(minimum_free),
            "AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES": "1",
            "AIDS_COMRECGC_V5_ALLOWED_OLD_PID": str(old_pid),
            "AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS": str(old_start_ticks),
            "AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256": old_cmdline_sha256,
            "AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT": str(old_output_root),
            "AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT": str(old_project_root),
            "COMRECGC_FLOCK_BIN": str(flock_bin),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        }
    )
    for forbidden in (
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST",
        "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK",
        "COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK",
    ):
        environment.pop(forbidden, None)
    expected_output = fresh_root / "cells/aids/comrecgc/standardized/attempt-{attempt}"
    threshold_path = Path(environment["THRESHOLDS_PATH"]).resolve(strict=True)
    threshold_sha256 = sha256_file(threshold_path)
    selector_task = {
        "id": SELECTOR_TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "runner_dataset": "paper-cell-aids-comrecgc-exact-route-v5",
        "runner_stage": "AM_COMRECGC_THRESHOLD_FREEZE",
        "depends_on": [],
        "resource": "cpu",
        "priority": 1,
        "manifest_only": True,
        "freezes_selector": True,
        "command": [
            "{python}",
            "{project_root}/scripts/autodl/write_aids_comrecgc_v5_selector_gate.py",
            "--config",
            "configs/hpc.yaml",
            "--thresholds",
            str(threshold_path),
            "--expected-sha256",
            threshold_sha256,
            "--output-dir",
            "{task_output}",
        ],
        "input_manifest": str(base_manifest),
        "config_files": [str(threshold_path)],
        "expected_output": str(fresh_root / "gates/selector/attempt-{attempt}"),
        "required_output_files": ["selector_gate.json", "PASS"],
        "required_log_marker": "[AIDS_COMRECGC_EXACT_ROUTE_V5_SELECTOR_PASS]",
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
        },
    }
    task = {
        "id": TASK_ID,
        "dataset": "aids",
        "stage": "AM_COMRECGC_HELDOUT_EVAL",
        "runner_dataset": "paper-cell-aids-comrecgc-exact-route-v5",
        "runner_stage": "AM_COMRECGC_HELDOUT_EVAL",
        "depends_on": [SELECTOR_TASK_ID],
        "resource": "cpu",
        "priority": 1,
        "data_splits": ["test"],
        "manifest_only": False,
        "selector_parameters_frozen": True,
        "read_only_test": True,
        "command": [
            "bash",
            "{project_root}/scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh",
        ],
        "input_manifest": str(terminal["source_manifest"]),
        "config_files": [str(base_manifest), environment["THRESHOLDS_PATH"]],
        "expected_output": str(expected_output),
        "required_output_files": list(STANDARDIZED_REQUIRED_FILES),
        "required_log_marker": "[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset=aids",
        "environment": environment,
        "semantic_failure_markers": [
            "source closure changed",
            "live writer",
            "sklearn version mismatch",
            "rss budget exceeded",
            "exact_dbscan_complexity_blocked",
            "test leakage",
        ],
    }
    contract = {
        "schema_version": SPEC_SCHEMA,
        "spec_path": str(source),
        "spec_sha256": sha256_file(source),
        "execution_project_root": str(project_root),
        "execution_commit": execution_commit,
        "reviewed_core_gate": core_gate,
        "route_release_gate": release_gate,
        "base_v4": base_evidence,
        "terminal_pair_store": terminal,
        "fresh_output_root": str(fresh_root),
        "gpu_required": False,
        "parameters": EXPECTED_PARAMETERS,
        "strict_flip": True,
        "test_loaded_after_selector_freeze": True,
        "old_v4_mutated": False,
        "old_v4_signal_authorized": False,
        "allowed_old_read_only_process": process_gate,
        "highmem_exclusion": {
            "v5_global_highmem_lock_acquired": False,
            "reason": (
                "the hash-bound old v4 read-only DBSCAN owns that lock until "
                "a separately authorized graceful switch"
            ),
            "old_read_only_consumer_is_the_only_colocation_exception": True,
            "v5_route_lock_held": True,
            "per_attempt_cgroup_headroom_gate_bytes": minimum_free,
            "v5_rss_budget_gib": 96,
            "process_set_revalidated_before_every_attempt": True,
            "mut_dependency_blocks_until_v5_pass": True,
        },
        "mut_dependency": {
            "controller_id": CONTROLLER_ID,
            "task_id": TASK_ID,
            "expected_output": str(fresh_root / "cells/aids/comrecgc/standardized/attempt-0"),
            "terminal_marker": "PASS",
        },
    }
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
            "min_free_disk_gb": 20,
            "max_cpu_load_fraction": 0.95,
        },
        "aids_comrecgc_exact_route_v5_contract": contract,
        "tasks": [selector_task, task],
    }
    validation = validate_payload(payload)
    return payload, {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "task_id": TASK_ID,
        "fresh_output_root": str(fresh_root),
        "expected_output_attempt0": contract["mut_dependency"]["expected_output"],
        "gpu_required": False,
        "task_count": validation["task_count"],
    }


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aids-comrecgc-exact-v5-") as directory:
        path = Path(directory) / "manifest.json"
        v3._atomic_json(path, payload)
        manifest = load_controller_manifest(path)
    if manifest.controller_id != CONTROLLER_ID or set(manifest.by_id) != {
        SELECTOR_TASK_ID,
        TASK_ID,
    }:
        raise RepairManifestError("AIDS v5 must contain exactly selector gate and terminal-only task")
    task = manifest.by_id[TASK_ID]
    selector = manifest.by_id[SELECTOR_TASK_ID]
    if (
        selector.stage != "AM_COMRECGC_THRESHOLD_FREEZE"
        or not selector.freezes_selector
        or task.depends_on != (SELECTOR_TASK_ID,)
    ):
        raise RepairManifestError("AIDS v5 selector-freeze dependency changed")
    if task.resource != "cpu" or task.command != (
        "bash",
        "{project_root}/scripts/autodl/run_aids_comrecgc_exact_route_v5_supervisor.sh",
    ):
        raise RepairManifestError("AIDS v5 task launch contract changed")
    required = {
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_EXTERNAL_REQUIRE_PROMOTED_FINAL": "1",
        "COMRECGC_EXTERNAL_EXACT_FALLBACK_MAX_SAMPLES": "0",
        "COMRECGC_EXTERNAL_DBSCAN_SHORTCUT_MODE": (
            "all_core_one_component_adaptive_anchor_v1"
        ),
        "AIDS_COMRECGC_V5_MAX_SAME_ROOT_RESUMES": "1",
        "RUN_TASTEMOLNET": "0",
    }
    if any(task.environment.get(key) != value for key, value in required.items()):
        raise RepairManifestError("AIDS v5 production environment is incomplete")
    if any(
        key in task.environment
        for key in (
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_MANIFEST",
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_CHECKPOINT",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_ROOT",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_LOCK",
            "COMRECGC_EXTERNAL_VECTOR_CACHE_ROUTE_LOCK",
        )
    ):
        raise RepairManifestError("AIDS v5 production task contains a fallback bypass")
    contract = payload.get("aids_comrecgc_exact_route_v5_contract")
    if not isinstance(contract, Mapping):
        raise RepairManifestError("AIDS v5 contract is missing")
    process_contract = contract.get("allowed_old_read_only_process")
    if not isinstance(process_contract, Mapping):
        raise RepairManifestError("AIDS v5 old-process contract is missing")
    process_environment = {
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PID": str(process_contract.get("allowed_pid")),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_START_TICKS": str(
            process_contract.get("allowed_start_ticks")
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_CMDLINE_SHA256": str(
            process_contract.get("allowed_cmdline_sha256")
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_OUTPUT_ROOT": str(
            process_contract.get("allowed_output_root")
        ),
        "AIDS_COMRECGC_V5_ALLOWED_OLD_PROJECT_ROOT": str(
            process_contract.get("allowed_project_root")
        ),
    }
    if any(
        task.environment.get(key) != value
        for key, value in process_environment.items()
    ):
        raise RepairManifestError("AIDS v5 old-process environment drifted")
    if (
        contract.get("parameters") != EXPECTED_PARAMETERS
        or contract.get("gpu_required") is not False
        or contract.get("old_v4_mutated") is not False
        or contract.get("old_v4_signal_authorized") is not False
        or process_contract.get("status") != "PASS"
        or not isinstance(contract.get("highmem_exclusion"), Mapping)
        or contract["highmem_exclusion"].get(
            "process_set_revalidated_before_every_attempt"
        )
        is not True
    ):
        raise RepairManifestError("AIDS v5 scientific/safety contract changed")
    terminal = contract.get("terminal_pair_store")
    if not isinstance(terminal, Mapping) or terminal.get("status") != "PASS":
        raise RepairManifestError("AIDS v5 terminal-source evidence is missing")
    return {
        "status": "PASS",
        "controller_id": CONTROLLER_ID,
        "task_id": TASK_ID,
        "task_count": 2,
        "gpu_required": False,
        "manifest_sha256": manifest.sha256,
    }


def build_manifest(*, spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    destination = _absolute(output_path, label="manifest output", kind="fresh")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"manifest output must be fresh: {destination}")
    payload, summary = build_payload(spec_path=spec_path)
    spec = _read_object(spec_path, label="v5 spec")
    control_root = _absolute(spec.get("control_root"), label="control root", kind="dir")
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
        raise RepairManifestError("published AIDS v5 manifest changed")
    return {
        **summary,
        "manifest": str(destination),
        "manifest_sha256": frozen.sha256,
    }


__all__ = [
    "CONTROLLER_ID",
    "EXPECTED_PAIR_COUNT",
    "REVIEWED_CORE_COMMIT",
    "ROUTE_RELEASE_COMMIT",
    "SELECTOR_TASK_ID",
    "SPEC_SCHEMA",
    "TASK_ID",
    "build_manifest",
    "build_payload",
    "validate_payload",
]
