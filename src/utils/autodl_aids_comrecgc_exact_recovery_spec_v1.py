"""Code-generate the production AIDS disconnected-exact recovery spec.

The only production authority accepted here is a terminal, canonical-validated
c766 recovery-evidence receipt.  Source paths and continuation inputs are
derived from that receipt plus its frozen controller task environment; users do
not hand-author the scientific JSON contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (
    ADOPTION_STAGE,
    ADOPTION_VALIDATOR_API,
    CONTROLLER_ID,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_MAX_RSS_BYTES,
    DEFAULT_SAFETY_FLOOR_BYTES,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_THREAD_COUNT,
    MAX_THREAD_COUNT,
    MIN_THREAD_COUNT,
    DEPENDENCIES,
    DOWNSTREAM_STAGE,
    EXACT_STAGE,
    EXACT_MONOTONIC_PROGRESS_FIELD,
    EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256,
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_PARENT_COUNT,
    EXPECTED_ROWS,
    EXPECTED_SUBSET_NAMES,
    EXPECTED_VECTOR_DIM,
    FINAL_STAGE,
    PARTIAL_STAGE_ARCHIVE_COUNT,
    PARTIAL_STAGE_ARCHIVE_MAX_BYTES,
    REQUIRED_RELEASE_PINS,
    SCIENCE_RELEASE_COMMIT,
    SPEC_SCHEMA,
    STAGE_KINDS,
    STAGE_ORDER,
    STARTUP_BARRIER_MAX_GENERATIONS,
    STARTUP_BARRIER_RECORD_MAX_BYTES,
    STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER,
    SUBSET_MAX_ATTEMPTS,
    SUBSET_STAGE,
    derive_output_budget,
    sha256_file,
    stable_json_sha256,
)


ADOPTION_MODULE = "src.baselines.comrecgc.failed_selection_adoption"
ADOPTION_CALLABLE = "verify_aids_c766_failed_selection_recovery_evidence"
ADOPTION_ENTRYPOINT = "scripts/autodl/adopt_aids_c766_failed_selection.py"
STAGE_ENTRYPOINT = "scripts/autodl/run_aids_comrecgc_exact_recovery_stage.py"


class RecoverySpecError(RuntimeError):
    """The frozen production authority cannot form an executable spec."""


def _physical_file(path: str | Path, *, label: str) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise RecoverySpecError(f"{label} may not be a symlink")
    resolved = value.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RecoverySpecError(f"{label} must be a nonempty physical file")
    return resolved


def _physical_dir(path: str | Path, *, label: str) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise RecoverySpecError(f"{label} may not be a symlink")
    resolved = value.resolve(strict=True)
    if not resolved.is_dir():
        raise RecoverySpecError(f"{label} must be a physical directory")
    return resolved


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve(strict=True) != path.parent
    ):
        raise RecoverySpecError("production spec parent must already be physical")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RecoverySpecError(f"production spec already exists: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_by_role(receipt: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    rows = receipt.get("source_artifacts")
    if not isinstance(rows, list):
        raise RecoverySpecError("adoption receipt source artifacts are absent")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("roles"), list)
        and role in row["roles"]
    ]
    if len(matches) != 1:
        raise RecoverySpecError(f"adoption source role is not unique: {role}")
    row = matches[0]
    _physical_file(str(row.get("path") or ""), label=role)
    if sha256_file(row["path"]) != row.get("sha256"):
        raise RecoverySpecError(f"adoption source role SHA changed: {role}")
    return row


def _derive_source_authority(receipt: Mapping[str, Any]) -> dict[str, Any]:
    authority = receipt.get("authority")
    close = receipt.get("close_authority")
    failed = receipt.get("failed_selection")
    if not all(isinstance(value, Mapping) for value in (authority, close, failed)):
        raise RecoverySpecError("adoption receipt science authorities are absent")
    assert isinstance(authority, Mapping)
    assert isinstance(close, Mapping)
    assert isinstance(failed, Mapping)

    role_fields = {
        "anchor_indices": "adaptive selected anchor indices",
        "anchor_rows": "adaptive selected anchor rows",
        "failure_indices": "adaptive first-pass failure indices",
        "anchor_edges": "failed disconnected anchor edges",
        "pair_store_manifest": "physical pair-store manifest",
        "normalized_distances": "normalized distance authority",
        "close_bitmap": "close bitmap",
    }
    role_rows = {
        field: _artifact_by_role(receipt, role) for field, role in role_fields.items()
    }
    result: dict[str, Any] = {
        "source_controller_manifest_path": authority["source_manifest"],
        "source_controller_manifest_sha256": authority["source_manifest_sha256"],
        "close_pass_gate_path": authority["close_gate"],
        "close_pass_gate_sha256": authority["close_gate_sha256"],
        "failed_final_gate_path": authority["final_gate"],
        "failed_final_gate_sha256": authority["final_gate_sha256"],
        "failed_shortcut_artifact_path": failed["failure_artifact"],
        "failed_shortcut_artifact_sha256": failed["failure_artifact_sha256"],
        "failed_checkpoint_path": failed["checkpoint"],
        "failed_checkpoint_sha256": failed["checkpoint_sha256"],
        "adaptive_selection_path": failed["selection_manifest"],
        "adaptive_selection_sha256": failed["selection_manifest_sha256"],
        "close_pair_manifest_path": close["manifest"],
        "close_pair_manifest_sha256": close["manifest_sha256"],
        "pair_semantics_receipt_path": close["pair_semantics_contract"],
        "pair_semantics_receipt_sha256": close[
            "pair_semantics_contract_sha256"
        ],
        "physical_pairs_path": close["pair_path"],
        "physical_pairs_sha256": close["pair_sha256"],
        "source_vectors_path": close["vector_path"],
        "source_vectors_sha256": close["vector_sha256"],
        "physical_pair_count": EXPECTED_ROWS,
        "pair_store_regenerated": False,
        "seed_failure_scan_reexecuted": False,
        "source_pair_store_access": "read_only_zero_copy",
        "failed_final_gate_status": "FAILED",
        "failed_final_reason": "anchor_epsilon_graph_disconnected",
        "failed_final_gate_ordinary_pass_eligible": False,
    }
    for field, row in role_rows.items():
        result[f"{field}_path"] = row["path"]
        result[f"{field}_sha256"] = row["sha256"]
    if (
        int(close.get("physical_rows", -1)) != EXPECTED_ROWS
        or int(close.get("logical_close_rows", -1)) != EXPECTED_ROWS
        or close.get("all_pairs_close") is not True
        or int(failed.get("anchor_count", -1)) != 266
        or failed.get("unique_seed_component") is not True
        or failed.get("dbscan_partition_proven") is not False
    ):
        raise RecoverySpecError("adoption receipt production science contract changed")
    for field, value in tuple(result.items()):
        if field.endswith("_sha256"):
            path = result[field[: -len("_sha256")] + "_path"]
            if sha256_file(path) != value:
                raise RecoverySpecError(f"source authority changed: {field}")
    return result


def _derive_runtime_inputs(
    receipt: Mapping[str, Any],
    *,
    manifest_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    authority = receipt["authority"]
    final_task = receipt["final_task"]
    manifest_path = _physical_file(
        authority["source_manifest"], label="source controller manifest"
    )
    if sha256_file(manifest_path) != authority["source_manifest_sha256"]:
        raise RecoverySpecError("source controller manifest changed")
    if manifest_loader is None:
        from scripts.autodl.run_four_gpu_recovery_controller import (
            load_controller_manifest,
        )

        manifest_loader = load_controller_manifest
    controller = manifest_loader(manifest_path)
    task_id = str(final_task.get("task_id") or "")
    try:
        task = controller.by_id[task_id]
    except (AttributeError, KeyError) as exc:
        raise RecoverySpecError("failed final task is absent from source manifest") from exc
    environment = dict(task.environment)
    required_fixed = {
        "DATASET": "aids",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "CUDA_VISIBLE_DEVICES": "",
        "COMMON_RECOURSE_ENGINE": "external_memory_exact_v1",
        "COMRECGC_COMMON_RECOURSE_RESUME": "1",
        "THETA_STAR": "0.05",
        "COST_CAP": "0.0535",
        "COMRECGC_EXPECTED_SKLEARN_VERSION": "1.7.2",
    }
    if any(environment.get(key) != value for key, value in required_fixed.items()):
        raise RecoverySpecError("source final scientific environment changed")
    mapping = {
        "source_generation_root": ("SOURCE_GENERATION_ROOT", "dir"),
        "upstream_root": ("COMRECGC_UPSTREAM_ROOT", "dir"),
        "dataset_dir": ("DATASET_DIR", "dir"),
        "source_csv": ("SOURCE_CSV", "file"),
        "distance_checkpoint": ("DISTANCE_CHECKPOINT", "file"),
        "dataset_csv": ("DATASET_CSV", "file"),
        "teacher_path": ("TEACHER_PATH", "file"),
        "molclr_root": ("MOLCLR_ROOT", "dir"),
        "molclr_checkpoint": ("MOLCLR_CHECKPOINT", "file"),
        "thresholds_path": ("THRESHOLDS_PATH", "file"),
        "pair_store_owner_root": (
            "COMRECGC_EXTERNAL_PAIR_STORE_SOURCE_OWNER_ROOT",
            "dir",
        ),
    }
    result: dict[str, Any] = {}
    for field, (environment_key, kind) in mapping.items():
        value = environment.get(environment_key)
        result[field] = str(
            _physical_dir(value, label=environment_key)
            if kind == "dir"
            else _physical_file(value, label=environment_key)
        )
    result.update(
        {
            "expected_sklearn_version": "1.7.2",
            "theta_star": 0.05,
            "cost_cap": 0.0535,
        }
    )
    return result


def _stage_spec(
    *,
    stage_id: str,
    python: Path,
    entrypoint: Path,
    output_dir: Path,
    terminal_path: Path,
    controller_manifest_path: Path,
    controller_root: Path,
) -> dict[str, Any]:
    gates = controller_root / "gates"
    bindings: dict[str, tuple[str, str]]
    prefix: list[str]
    if stage_id == ADOPTION_STAGE:
        prefix = [
            str(python),
            str(entrypoint),
            "--config",
            str(entrypoint.parents[2] / "configs/hpc.yaml"),
        ]
        bindings = {"output": ("--output-dir", str(output_dir))}
        suffix = ["--validate-only"]
    else:
        prefix = [
            str(python),
            str(entrypoint),
            "--config",
            str(entrypoint.parents[2] / "configs/hpc.yaml"),
            {
                SUBSET_STAGE: "subset",
                EXACT_STAGE: "exact",
                DOWNSTREAM_STAGE: "downstream",
                FINAL_STAGE: "final",
            }[stage_id],
        ]
        suffix = []
        bindings = {
            "output": ("--output-dir", str(output_dir)),
            "controller_manifest": (
                "--controller-manifest",
                str(controller_manifest_path),
            ),
        }
        if stage_id in {SUBSET_STAGE, EXACT_STAGE, FINAL_STAGE}:
            bindings["adoption_gate"] = (
                "--adoption-gate",
                str(gates / "01_failed_selection_adoption.json"),
            )
        if stage_id in {EXACT_STAGE, FINAL_STAGE}:
            bindings["subset_gate"] = (
                "--subset-gate",
                str(gates / "02_production_subset_equivalence.json"),
            )
        if stage_id in {DOWNSTREAM_STAGE, FINAL_STAGE}:
            bindings["exact_gate"] = (
                "--exact-gate",
                str(gates / "03_exact_component_recovery.json"),
            )
        if stage_id == FINAL_STAGE:
            bindings["downstream_gate"] = (
                "--downstream-gate",
                str(gates / "04_component_downstream_radius_ab.json"),
            )
    fresh = list(prefix)
    for flag, value in bindings.values():
        fresh.extend([flag, value])
    fresh.extend(suffix)
    resume = (
        [*fresh, "--resume"]
        if stage_id in {SUBSET_STAGE, EXACT_STAGE, DOWNSTREAM_STAGE, FINAL_STAGE}
        else None
    )
    return {
        "stage_id": stage_id,
        "kind": STAGE_KINDS[stage_id],
        "dependencies": list(DEPENDENCIES[stage_id]),
        "output_dir": str(output_dir),
        "terminal_path": str(terminal_path),
        "terminal_schema": {
            ADOPTION_STAGE: "aids_comrecgc_c766_failed_selection_adoption_v3",
            SUBSET_STAGE: "aids_comrecgc_production_subset_stage_v1",
            EXACT_STAGE: "aids_comrecgc_exact_component_recovery_stage_v1",
            DOWNSTREAM_STAGE: "comrecgc_all_core_component_summary_v1",
            FINAL_STAGE: "aids_comrecgc_recovered_standardized_freeze_v1",
        }[stage_id],
        "entrypoint_sha256": sha256_file(entrypoint),
        "commands": {"fresh": fresh, "resume": resume},
        "argv_bindings": {
            role: {"flag": flag, "value": value}
            for role, (flag, value) in bindings.items()
        },
        "progress_checkpoint_path": (
            str(output_dir / "dbscan/checkpoint.json")
            if stage_id == EXACT_STAGE
            else None
        ),
        "progress_field": (
            EXACT_MONOTONIC_PROGRESS_FIELD if stage_id == EXACT_STAGE else None
        ),
    }


def generate_production_spec(
    *,
    adoption_output: str | Path,
    controller_parent: str | Path,
    python: str | Path,
    project_root: str | Path,
    controller_manifest_path: str | Path,
    timestamp: str | None = None,
    thread_count: int = DEFAULT_THREAD_COUNT,
    adoption_validator: Callable[..., Mapping[str, Any]] | None = None,
    manifest_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    project = _physical_dir(project_root, label="project root")
    if not MIN_THREAD_COUNT <= int(thread_count) <= MAX_THREAD_COUNT:
        raise RecoverySpecError("thread_count must remain between 8 and 12")
    interpreter = _physical_file(python, label="AutoDL Python")
    adoption_root = _physical_dir(adoption_output, label="adoption output")
    parent = _physical_dir(controller_parent, label="controller parent")
    manifest_path = Path(controller_manifest_path).expanduser().resolve(strict=False)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise RecoverySpecError("controller manifest path already exists")
    if adoption_validator is None:
        module = __import__(ADOPTION_MODULE, fromlist=[ADOPTION_CALLABLE])
        adoption_validator = getattr(module, ADOPTION_CALLABLE)
    receipt = dict(adoption_validator(output_dir=adoption_root))
    receipt_path = adoption_root / "failed_selection_adoption_receipt.json"
    if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
        raise RecoverySpecError("canonical adoption validator did not return receipt")
    source = _derive_source_authority(receipt)
    runtime = _derive_runtime_inputs(receipt, manifest_loader=manifest_loader)
    when = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = stable_json_sha256(
        {"adoption_receipt_sha256": sha256_file(receipt_path), "timestamp": when}
    )[:8]
    cid = f"aids_comrecgc_exact_recovery_v1_{when}_{suffix}"
    controller_root = parent / cid
    if controller_root.exists() or controller_root.is_symlink():
        raise RecoverySpecError("fresh controller CID/root already exists")
    science = controller_root / "science"
    subset = controller_root / "subset_preflight"
    exact = science / "common_recourse/external_memory"
    downstream = exact / "all_core_component_summary"
    outputs = {
        ADOPTION_STAGE: adoption_root,
        SUBSET_STAGE: subset,
        EXACT_STAGE: exact,
        DOWNSTREAM_STAGE: downstream,
        FINAL_STAGE: science,
    }
    terminals = {
        ADOPTION_STAGE: receipt_path,
        SUBSET_STAGE: subset / "subset_stage_receipt.json",
        EXACT_STAGE: exact / "exact_recovery_receipt.json",
        DOWNSTREAM_STAGE: downstream / "run_manifest.json",
        FINAL_STAGE: science / "exact_recovery_freeze_receipt.json",
    }
    adoption_entrypoint = _physical_file(
        project / ADOPTION_ENTRYPOINT, label="adoption entrypoint"
    )
    stage_entrypoint = _physical_file(
        project / STAGE_ENTRYPOINT, label="recovery stage entrypoint"
    )
    stages = [
        _stage_spec(
            stage_id=stage_id,
            python=interpreter,
            entrypoint=(
                adoption_entrypoint if stage_id == ADOPTION_STAGE else stage_entrypoint
            ),
            output_dir=outputs[stage_id],
            terminal_path=terminals[stage_id],
            controller_manifest_path=manifest_path,
            controller_root=controller_root,
        )
        for stage_id in STAGE_ORDER
    ]
    module_file = _physical_file(
        Path(str(__import__(ADOPTION_MODULE, fromlist=["__file__"]).__file__)),
        label="adoption validator module",
    )
    task_state = receipt["task_state_authority"]
    projections = {
        "close": task_state["close_projection_sha256"],
        "final": task_state["final_projection_sha256"],
    }
    if projections != EXPECTED_ADOPTION_TASK_STATE_PROJECTION_SHA256:
        raise RecoverySpecError("adoption production state projections changed")
    allowed_parent_entries = sorted(
        entry.name for entry in adoption_root.parent.iterdir() if entry != adoption_root
    )
    budget = derive_output_budget(
        row_count=EXPECTED_ROWS,
        vector_dim=EXPECTED_VECTOR_DIM,
        subset_size=DEFAULT_SUBSET_SIZE,
        subset_count=len(EXPECTED_SUBSET_NAMES),
        block_size=DEFAULT_BLOCK_SIZE,
        safety_floor_bytes=DEFAULT_SAFETY_FLOOR_BYTES,
    )
    pins = {name: None for name in REQUIRED_RELEASE_PINS}
    pins["science_commit"] = SCIENCE_RELEASE_COMMIT
    return {
        "schema_version": SPEC_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "cid": cid,
        "project_root": str(project),
        "controller_root": str(controller_root),
        "controller_manifest_path": str(manifest_path),
        "adoption_authority_parent": str(adoption_root.parent),
        "production_deployment_authorized": False,
        "stages": stages,
        "adoption_contract": {
            "receipt_schema": receipt["schema_version"],
            "receipt_name": receipt_path.name,
            "ready_marker_name": receipt["terminal_marker"],
            "receipt_status": receipt["status"],
            "artifact_kind": receipt["artifact_kind"],
            "projection_profile": "canonical_scientific_task_state_projection_v3",
            "validator_module": ADOPTION_MODULE,
            "validator_callable": ADOPTION_CALLABLE,
            "validator_module_sha256": sha256_file(module_file),
            "validator_api": ADOPTION_VALIDATOR_API,
            "authority_profile_sha256": receipt["authority_profile_sha256"],
            "expected_task_state_projection_sha256": projections,
            "recovery_only": True,
            "ordinary_pass_dependency_eligible": False,
            "dbscan_partition_proven": False,
            "authority_parent_allowed_entries": allowed_parent_entries,
        },
        "source_authority": source,
        "runtime_inputs": runtime,
        "resources": {
            "row_count": EXPECTED_ROWS,
            "vector_dim": EXPECTED_VECTOR_DIM,
            "subset_size": DEFAULT_SUBSET_SIZE,
            "subset_max_attempts": SUBSET_MAX_ATTEMPTS,
            "block_size": DEFAULT_BLOCK_SIZE,
            "partial_stage_archive_count": PARTIAL_STAGE_ARCHIVE_COUNT,
            "partial_stage_archive_max_bytes_each": (
                PARTIAL_STAGE_ARCHIVE_MAX_BYTES
            ),
            "startup_barrier_max_generations": STARTUP_BARRIER_MAX_GENERATIONS,
            "startup_barrier_record_max_bytes": STARTUP_BARRIER_RECORD_MAX_BYTES,
            "startup_barrier_publication_file_multiplier": (
                STARTUP_BARRIER_PUBLICATION_FILE_MULTIPLIER
            ),
            "safety_floor_bytes": DEFAULT_SAFETY_FLOOR_BYTES,
            "budget": budget,
            "max_rss_bytes": DEFAULT_MAX_RSS_BYTES,
            "max_rss_scope": (
                "exact_dbscan_process_with_native_peak_certificate"
            ),
            "thread_count": int(thread_count),
            "cpu_only": True,
            "gpu_lock_required": False,
            "proc_root": "/proc",
            "coexistence_probe": {
                "min_progress_rows": DEFAULT_BLOCK_SIZE,
                "max_load_per_cpu": 0.8,
                "max_iowait_fraction": 0.35,
                "timeout_seconds": 1800,
            },
        },
        "release_pins": pins,
    }


def write_production_spec(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    path = Path(output).expanduser().resolve(strict=False)
    payload = generate_production_spec(**kwargs)
    _write_new_json(path, payload)
    return payload


__all__ = [
    "RecoverySpecError",
    "generate_production_spec",
    "write_production_spec",
]
