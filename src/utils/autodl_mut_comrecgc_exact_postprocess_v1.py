"""Read-only Mutagenicity exact adoption and standardized continuation.

This module deliberately has no common-recourse entry point.  It accepts one
hash-closed, already completed exact result and can only run the downstream
chemistry, WNode evaluation, freeze, and strict matrix-append stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Mapping, Sequence

from scripts.autodl.append_bace_gcf_matrix_authority import (
    _git_identity,
    _inventory,
    _json_bytes,
    _read_json,
    _verify_authority,
)
from scripts.autodl.run_comrecgc_standardized_continuation import (
    ContinuationInputs,
    _archive_noncheckpointed_partial_stage,
    _archive_previous_failure,
    _git_head,
    _load_object,
    _require_completed_stage_writer_quiescence,
    _require_file,
    _resume_contract,
    _run_stage,
    _scan_live_source_writers,
    _snapshot_frozen_file,
    _utc_now,
    _validate_common_recourse_completion,
    _validate_completed_stage,
    _verify_adopted_generation_integrity,
    validate_adopted_generation,
)
from scripts.autodl.run_mut_comrecgc_parity_standardization import (
    SOURCE_CANDIDATE_COUNT,
    SOURCE_PAYLOAD_SHA256,
    _commands as _downstream_commands,
)
from scripts.verify_comrecgc_checkout import verify_checkout
from src.baselines.comrecgc.contracts import (
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from src.eval.am_legacy_standardization import scan_live_writers
from src.eval.four_by_four_registry import (
    AuditConfig,
    CellStatus,
    PASS_STATUSES,
    audit_registry,
    write_registry_outputs,
)
from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.autodl_mut_traceoff_parity_v1 import _validate_parity_gate


ADOPTION_SCHEMA = "mut_comrecgc_exact_multicomponent_adoption_v1"
RUN_SCHEMA = "mut_comrecgc_exact_postprocess_v1"
MATRIX_APPEND_SCHEMA = "mut_comrecgc_matrix_authority_append_v1"
TARGET_DATASET = "Mutagenicity"
TARGET_METHOD = "ComRecGC"
EXPECTED_COMMON_RECOURSES = 100
EXPECTED_REMAINING_STAGES = ["WNode", "standardized_export", "matrix_append"]
EXPECTED_RECOURSE_PARAMETERS = {
    "theta": 0.1,
    "delta": 0.02,
    "recourse_size": 100,
    "cf_size": 100_000,
    "cluster_size": 3,
    "seed": 0,
}
EXPECTED_DBSCAN_CONTRACT = {
    "eps": 0.02,
    "min_samples": 3,
    "query_block_size": 4,
    "checkpoint_interval_blocks": 1,
    "max_rss_bytes": 96 * 1024**3,
    "expected_sklearn_version": "1.7.2",
    "shortcut_mode": "sklearn_float64_exact_multi_component_v1",
    "shortcut_anchor_count": 64,
    "shortcut_seed_count": 3,
    "shortcut_failure_cap": 4_096,
    "shortcut_query_block_size": 65_536,
    "exact_fallback_max_samples": 100_000,
}
EXPECTED_DBSCAN_SHORTCUT_CONTRACT = {
    "mode": "sklearn_float64_exact_multi_component_v1",
    "single_component_assumed": False,
    "failure_cap_used": False,
    "multi_component_labels_materialized": True,
    "reference_semantics": "SKLEARN_FLOAT64",
    "comparison": "distance <= eps",
    "query_block_size": 4,
    "worker_count": 4,
}
EXPECTED_CONTROLLER_STAGE = "MUT_EXACT_MULTICOMPONENT_FAST16"
SHARED_MUT_IDENTITY_FIELDS = (
    "dataset_hash",
    "split_hash",
    "oracle_backend",
    "oracle_checkpoint",
    "oracle_hash",
    "molclr_checkpoint_hash",
    "distance_line",
    "cf_mode",
    "threshold_config_hash",
)
PASS_MARKER = "[MUT_COMRECGC_EXACT_POSTPROCESS_PASS]"


class MutExactPostprocessError(RuntimeError):
    """The completed exact result cannot be continued without ambiguity."""


def _physical_file(path: str | Path, *, label: str) -> Path:
    logical = Path(path).expanduser()
    if not logical.is_absolute():
        raise MutExactPostprocessError(f"{label} must be an absolute path")
    if logical.is_symlink():
        raise MutExactPostprocessError(f"{label} may not be a symlink: {logical}")
    try:
        resolved = logical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MutExactPostprocessError(f"{label} is absent: {logical}") from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise MutExactPostprocessError(f"{label} is not a non-empty file: {resolved}")
    return resolved


def _physical_directory(path: str | Path, *, label: str) -> Path:
    logical = Path(path).expanduser()
    if not logical.is_absolute():
        raise MutExactPostprocessError(f"{label} must be an absolute path")
    if logical.is_symlink():
        raise MutExactPostprocessError(f"{label} may not be a symlink: {logical}")
    try:
        resolved = logical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MutExactPostprocessError(f"{label} is absent: {logical}") from exc
    if not resolved.is_dir():
        raise MutExactPostprocessError(f"{label} is not a directory: {resolved}")
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_output_path_isolation(
    *,
    output_paths: Mapping[str, Path],
    protected_paths: Mapping[str, Path],
) -> dict[str, str]:
    """Reject every output/source containment relationship before writing.

    Resolving a missing output with ``strict=False`` still resolves all existing
    ancestors, so a symlinked parent cannot hide an output below a frozen root.
    Outputs are also mutually disjoint: in particular the matrix authority may
    not be published inside the standardized cell it describes.
    """

    outputs: dict[str, Path] = {}
    for label, raw in output_paths.items():
        logical = Path(raw).expanduser()
        if not logical.is_absolute() or logical.is_symlink():
            raise MutExactPostprocessError(
                f"{label} must be an absolute non-symlink path"
            )
        outputs[label] = logical.resolve(strict=False)
    protected: dict[str, Path] = {}
    for label, raw in protected_paths.items():
        logical = Path(raw).expanduser()
        if not logical.is_absolute() or logical.is_symlink():
            raise MutExactPostprocessError(
                f"Protected {label} must be an absolute physical path"
            )
        protected[label] = logical.resolve(strict=True)
    for output_label, output in outputs.items():
        for protected_label, source in protected.items():
            if _paths_overlap(output, source):
                raise MutExactPostprocessError(
                    "OUTPUT_SOURCE_PATH_OVERLAP:"
                    f"output={output_label}:{output}:"
                    f"protected={protected_label}:{source}"
                )
    output_items = sorted(outputs.items())
    for index, (left_label, left) in enumerate(output_items):
        for right_label, right in output_items[index + 1 :]:
            if _paths_overlap(left, right):
                raise MutExactPostprocessError(
                    "OUTPUT_PATH_OVERLAP:"
                    f"left={left_label}:{left}:right={right_label}:{right}"
                )
    return {label: str(path) for label, path in sorted(outputs.items())}


def _validate_controller_terminal(
    *,
    controller_state_path: Path,
    receipt: Mapping[str, Any],
    proc_root: Path,
) -> dict[str, Any]:
    state = _load_object(controller_state_path)
    worker_pid = receipt.get("source_worker_pid")
    controller_pid = state.get("pid")
    expected = {
        "schema_version": 1,
        "state": "PASS",
        "dataset": "mutagenicity",
        "stage": EXPECTED_CONTROLLER_STAGE,
        "exit_code": 0,
        "failures": [],
        "child_pid": worker_pid,
    }
    failures = [name for name, value in expected.items() if state.get(name) != value]
    if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid <= 0:
        failures.append("source_worker_pid")
    if (
        not isinstance(controller_pid, int)
        or isinstance(controller_pid, bool)
        or controller_pid <= 0
    ):
        failures.append("controller_pid")
    if controller_pid == worker_pid:
        failures.append("controller_worker_pid_alias")
    if not str(state.get("run_id") or "").startswith("mut-exact-multicomponent-"):
        failures.append("run_id")
    if not str(state.get("completed_at") or "").strip():
        failures.append("completed_at")
    if failures:
        raise MutExactPostprocessError(
            "Source controller is not the completed Mut exact terminal: "
            + ", ".join(sorted(set(failures)))
        )
    proc = _physical_directory(proc_root, label="procfs root")
    live = [pid for pid in (worker_pid, controller_pid) if (proc / str(pid)).exists()]
    if live:
        raise MutExactPostprocessError(
            f"Source exact controller/worker PID still exists: {live}"
        )
    return {
        "schema_version": state["schema_version"],
        "state": state["state"],
        "run_id": state["run_id"],
        "controller_pid": controller_pid,
        "worker_pid": worker_pid,
        "worker_exit_code": state["exit_code"],
        "completed_at": state["completed_at"],
        "controller_and_worker_absent_from_procfs": True,
    }


def _one_of(payload: Mapping[str, Any], names: Sequence[str], *, label: str) -> Any:
    present = [name for name in names if name in payload]
    if not present:
        raise MutExactPostprocessError(
            f"{label} requires a frozen field from {list(names)}"
        )
    values = [payload[name] for name in present]
    if any(value != values[0] for value in values[1:]):
        raise MutExactPostprocessError(
            f"{label} aliases disagree: {present}"
        )
    return payload[present[0]]


def _common_artifact_path(root: Path, relative: str) -> Path:
    aliases = {
        "dbscan/run_manifest.json": "external_memory/dbscan/run_manifest.json",
        "dbscan/labels.npy": "external_memory/dbscan/labels.npy",
    }
    translated = aliases.get(relative, relative)
    path = _physical_file(root / translated, label=f"exact artifact {relative}")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MutExactPostprocessError(
            f"Exact artifact escapes completed root: {relative}"
        ) from exc
    return path


def validate_exact_adoption(
    *,
    adoption_receipt_path: str | Path,
    common_root: str | Path,
    source_generation_root: str | Path,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Reopen the exact terminal and prove that it is a read-only input."""

    receipt_path = _physical_file(adoption_receipt_path, label="adoption receipt")
    common = _physical_directory(common_root, label="completed exact full root")
    generation = _physical_directory(
        source_generation_root, label="frozen generation root"
    )
    receipt = _load_object(receipt_path)
    expected = {
        "schema_version": ADOPTION_SCHEMA,
        "status": "PASS",
        "state": "ADOPTED_COMPLETED_SCIENCE",
        "source_worker_active": False,
        "source_worker_exit_code": 0,
        "exactly_zero_active_exact_writers": True,
        "active_exact_writer_pids": [],
        "second_writer_started": False,
        "labels_partition_centroid_radius_coverage_greedy_complete": True,
        "remaining_stages": EXPECTED_REMAINING_STAGES,
    }
    failures = [name for name, value in expected.items() if receipt.get(name) != value]
    try:
        claimed_root = Path(str(receipt.get("full_root") or "")).resolve(strict=True)
    except FileNotFoundError:
        claimed_root = Path("/")
    if claimed_root != common:
        failures.append("full_root")
    try:
        claimed_source_root = Path(
            str(receipt.get("source_root") or "")
        ).resolve(strict=True)
    except FileNotFoundError:
        claimed_source_root = Path("/")
    if claimed_source_root != common.parent:
        failures.append("source_root")
    if failures:
        raise MutExactPostprocessError(
            "Exact adoption receipt contract mismatch: " + ", ".join(failures)
        )

    controller_state = _physical_file(
        str(receipt.get("source_controller_state") or ""),
        label="source controller terminal state",
    )
    if sha256_file(controller_state) != receipt.get("source_controller_state_sha256"):
        raise MutExactPostprocessError("Source controller terminal state hash changed")
    controller_terminal = _validate_controller_terminal(
        controller_state_path=controller_state,
        receipt=receipt,
        proc_root=Path(proc_root),
    )

    artifact_claims = receipt.get("artifact_sha256")
    if not isinstance(artifact_claims, Mapping):
        raise MutExactPostprocessError("Adoption receipt has no artifact hash closure")
    required_artifacts = {
        "_RUN_COMPLETE.json",
        "run_manifest.json",
        "representative_counterfactuals.pt",
        "selected_common_recourses.csv",
        "selected_common_recourses.json",
        "dbscan/run_manifest.json",
        "dbscan/labels.npy",
    }
    if not required_artifacts.issubset(artifact_claims):
        missing = sorted(required_artifacts - set(artifact_claims))
        raise MutExactPostprocessError(
            f"Adoption receipt artifact closure is incomplete: {missing}"
        )
    artifact_paths: dict[str, Path] = {}
    for relative in sorted(required_artifacts):
        path = _common_artifact_path(common, relative)
        if sha256_file(path) != str(artifact_claims[relative]):
            raise MutExactPostprocessError(
                f"Adopted exact artifact hash changed: {relative}"
            )
        artifact_paths[relative] = path

    terminal_path = artifact_paths["_RUN_COMPLETE.json"]
    terminal = _load_object(terminal_path)
    try:
        _validate_common_recourse_completion(marker=terminal_path, terminal=terminal)
    except Exception as exc:
        raise MutExactPostprocessError(
            f"Completed exact common-recourse closure failed: {exc}"
        ) from exc

    manifest = _load_object(artifact_paths["run_manifest.json"])
    manifest_expected = {
        "dataset": "mutagenicity",
        "mode": "full",
        "method": METHOD,
        "cf_mode": CF_MODE,
        "test_loaded": False,
        "calibration_loaded": False,
        "run_complete": True,
        "common_recourse_engine": "external_memory_exact_v1",
        "common_recourse_count": EXPECTED_COMMON_RECOURSES,
    }
    changed = [
        name for name, value in manifest_expected.items() if manifest.get(name) != value
    ]
    if changed:
        raise MutExactPostprocessError(
            "Completed exact scientific identity changed: " + ", ".join(changed)
        )
    if manifest.get("parameters") != EXPECTED_RECOURSE_PARAMETERS:
        raise MutExactPostprocessError(
            "Completed exact common-recourse parameters changed"
        )
    selected = json.loads(
        artifact_paths["selected_common_recourses.json"].read_text(encoding="utf-8")
    )
    if not isinstance(selected, list) or len(selected) != EXPECTED_COMMON_RECOURSES:
        raise MutExactPostprocessError(
            "Completed exact result does not contain exactly 100 selected recourses"
        )
    external = manifest.get("external_memory_artifacts")
    dbscan_manifest_path = artifact_paths["dbscan/run_manifest.json"]
    dbscan_manifest = _load_object(dbscan_manifest_path)
    dbscan_identity = dbscan_manifest.get("scientific_identity")
    dbscan_next_offset = receipt.get("dbscan_next_offset")
    labels_path = artifact_paths["dbscan/labels.npy"]
    try:
        claimed_labels_path = Path(
            str(dbscan_manifest.get("labels_path") or "")
        ).resolve(strict=True)
    except FileNotFoundError:
        claimed_labels_path = Path("/")
    if (
        not isinstance(dbscan_next_offset, int)
        or isinstance(dbscan_next_offset, bool)
        or dbscan_next_offset <= 0
    ):
        raise MutExactPostprocessError("Adoption receipt has no terminal DBSCAN cursor")
    if (
        not isinstance(external, Mapping)
        or external.get("engine") != "external_memory_exact_v1"
        or external.get("dbscan_shortcut_mode")
        != "sklearn_float64_exact_multi_component_v1"
        or Path(str(external.get("dbscan_manifest") or "")).resolve(strict=True)
        != dbscan_manifest_path
        or external.get("dbscan_manifest_sha256")
        != sha256_file(dbscan_manifest_path)
        or receipt.get("dbscan_phase") != "complete"
        or receipt.get("dbscan_run_manifest_sha256")
        != sha256_file(dbscan_manifest_path)
        or dbscan_manifest.get("schema_version")
        != "comrecgc_external_memory_dbscan_v3"
        or dbscan_manifest.get("run_complete") is not True
        or dbscan_manifest.get("clustering_path")
        != "sklearn_float64_exact_multi_component_v1"
        or dbscan_manifest.get("distance_reference_dtype") != "float64"
        or int(dbscan_manifest.get("exact_worker_count", -1)) != 4
        or dbscan_manifest.get("single_component_shortcut_used") is not False
        or dbscan_manifest.get("failure_cap_used") is not False
        or dbscan_manifest.get("approximation_used") is not False
        or dbscan_manifest.get("sklearn_dbscan_label_semantics_preserved") is not True
        or dbscan_manifest.get("num_samples") != dbscan_next_offset
        or claimed_labels_path != labels_path
        or dbscan_manifest.get("labels_sha256") != sha256_file(labels_path)
        or dbscan_manifest.get("neighbor_counts_available") is not True
        or dbscan_manifest.get("all_neighborhoods_materialized_simultaneously")
        is not False
        or dbscan_manifest.get("passes")
        != ["neighbor_counts", "core_union", "border_assignment"]
        or dbscan_manifest.get("max_rss_bytes")
        != EXPECTED_DBSCAN_CONTRACT["max_rss_bytes"]
        or not isinstance(dbscan_identity, Mapping)
    ):
        raise MutExactPostprocessError(
            "Completed exact DBSCAN route is not sklearn float64 multi-component"
        )
    assert isinstance(dbscan_identity, Mapping)
    dbscan_identity_failures = []
    dbscan_identity_expected = {
        "schema_version": "comrecgc_external_memory_dbscan_v3",
        "contract": EXPECTED_DBSCAN_CONTRACT,
        "sklearn_version": "1.7.2",
        "nearest_neighbors_fit_method": "brute",
        "nearest_neighbors_metric": "euclidean",
        "nearest_neighbors_algorithm": "brute",
        "border_assignment": "minimum_cluster_label_of_adjacent_core_component",
        "shortcut_contract": EXPECTED_DBSCAN_SHORTCUT_CONTRACT,
        "distance_reference_dtype": "float64",
        "exact_worker_count": 4,
    }
    for name, expected_value in dbscan_identity_expected.items():
        if dbscan_identity.get(name) != expected_value:
            dbscan_identity_failures.append(name)
    if dbscan_manifest.get("scientific_identity_sha256") != stable_json_sha256(
        dbscan_identity
    ):
        dbscan_identity_failures.append("scientific_identity_sha256")
    if dbscan_identity_failures:
        raise MutExactPostprocessError(
            "Completed exact DBSCAN scientific identity changed: "
            + ", ".join(sorted(dbscan_identity_failures))
        )

    generation_manifest = _physical_file(
        generation / "run_manifest.json", label="source generation manifest"
    )
    claimed_generation_path = Path(
        str(
            _one_of(
                manifest,
                ("generation_manifest_path", "source_generation_manifest_path"),
                label="common generation manifest path",
            )
        )
    ).resolve(strict=True)
    if claimed_generation_path != generation_manifest:
        raise MutExactPostprocessError("Common result references another generation manifest")
    claimed_generation_sha = str(
        _one_of(
            manifest,
            ("generation_manifest_sha256", "source_generation_manifest_sha256"),
            label="common generation manifest hash",
        )
    )
    if claimed_generation_sha != sha256_file(generation_manifest):
        raise MutExactPostprocessError("Common result generation manifest hash changed")
    generation_payload = _load_object(generation_manifest)
    generation_counterfactual_sha = str(
        generation_payload.get("counterfactuals_sha256") or ""
    )
    claimed_counterfactual_sha = str(
        _one_of(
            manifest,
            ("counterfactuals_sha256", "source_counterfactuals_sha256"),
            label="common source counterfactual hash",
        )
    )
    if (
        generation_counterfactual_sha != SOURCE_PAYLOAD_SHA256
        or claimed_counterfactual_sha != SOURCE_PAYLOAD_SHA256
    ):
        raise MutExactPostprocessError(
            "Completed exact result is not bound to the frozen Mut generation payload"
        )

    protected = [
        _snapshot_frozen_file(path, include_sha256=False)
        for path in (
            receipt_path,
            controller_state,
            generation_manifest,
            *artifact_paths.values(),
        )
    ]
    writer_audit = _scan_live_source_writers(
        common,
        protected_snapshots=protected,
        proc_root=Path(proc_root),
    )
    return {
        "schema_version": "mut_comrecgc_exact_read_only_adoption_v1",
        "status": "PASS",
        "adoption_receipt_path": str(receipt_path),
        "adoption_receipt_sha256": sha256_file(receipt_path),
        "source_controller_state_path": str(controller_state),
        "source_controller_state_sha256": sha256_file(controller_state),
        "controller_terminal": controller_terminal,
        "common_root": str(common),
        "common_terminal_sha256": sha256_file(terminal_path),
        "common_manifest_sha256": sha256_file(artifact_paths["run_manifest.json"]),
        "source_generation_root": str(generation),
        "source_generation_manifest_sha256": sha256_file(generation_manifest),
        "source_counterfactuals_sha256": SOURCE_PAYLOAD_SHA256,
        "common_recourse_count": EXPECTED_COMMON_RECOURSES,
        "common_recourse_parameters": dict(EXPECTED_RECOURSE_PARAMETERS),
        "dbscan_scientific_identity_sha256": dbscan_manifest[
            "scientific_identity_sha256"
        ],
        "dbscan_next_offset": dbscan_next_offset,
        "labels_partition_centroid_radius_coverage_greedy_complete": True,
        "common_recourse_rerun": False,
        "dbscan_rerun": False,
        "pair_store_rerun": False,
        "second_exact_writer_started": False,
        "writer_audit": writer_audit,
        "validated_at": _utc_now(),
    }


def build_postprocess_commands(
    inputs: ContinuationInputs,
    *,
    common_root: Path,
    parity_path: Path,
    project_commit: str,
    teacher_sha256: str,
) -> list[tuple[str, list[str], Path, str]]:
    commands = _downstream_commands(
        inputs,
        common_root=common_root,
        parity_path=parity_path,
        project_commit=project_commit,
        teacher_sha256=teacher_sha256,
    )
    expected_names = ["chemistry", "unified_eval", "full_gate", "freeze"]
    if [stage for stage, *_ in commands] != expected_names:
        raise MutExactPostprocessError("Downstream stage ordering changed")
    result: list[tuple[str, list[str], Path, str]] = []
    for stage, raw_argv, marker, required_field in commands:
        argv = list(raw_argv)
        if stage == "chemistry":
            if "--expected-medoid-count" in argv:
                raise MutExactPostprocessError("Chemistry medoid gate is ambiguous")
            argv.extend(
                ["--expected-medoid-count", str(EXPECTED_COMMON_RECOURSES)]
            )
        if any(
            forbidden in " ".join(argv)
            for forbidden in (
                "run_common_recourse.py",
                "pair_store",
                "run_external_dbscan",
            )
        ):
            raise MutExactPostprocessError(
                f"Read-only continuation attempted a forbidden science stage: {stage}"
            )
        result.append((stage, argv, marker, required_field))
    return result


def _append_mut_matrix_authority(
    *,
    prior_authority_root: str | Path,
    standardized_root: str | Path,
    output_root: str | Path,
    proc_root: str | Path = "/proc",
    require_writer_audit: bool = True,
    git_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prior = _verify_authority(prior_authority_root)
    target_key = (TARGET_DATASET, TARGET_METHOD)
    prior_rows = prior["rows"]
    passing = {status.value for status in PASS_STATUSES}
    if str(prior_rows[target_key].get("status") or "") in passing:
        raise MutExactPostprocessError("Prior authority already passes Mut/ComRecGC")
    reference = prior_rows[(TARGET_DATASET, "Ours")]
    if str(reference.get("status") or "") not in passing:
        raise MutExactPostprocessError(
            "Mut/ComRecGC append requires the frozen Mut/Ours identity reference"
        )

    cell_root = _physical_directory(standardized_root, label="standardized cell root")
    inventory = _inventory(cell_root)
    writer_audit = (
        scan_live_writers(cell_root, proc_root=proc_root)
        if require_writer_audit
        else {
            "procfs_verified": False,
            "scanned_process_count": 0,
            "writable_fd_count": 0,
            "writers": [],
        }
    )
    explicit_cells = {
        f"{dataset}/{method}": str(
            Path(str(row["standardized_output_root"])).resolve(strict=True)
        )
        for (dataset, method), row in prior_rows.items()
        if str(row.get("status") or "") in passing
    }
    explicit_cells[f"{TARGET_DATASET}/{TARGET_METHOD}"] = str(cell_root)
    destination_logical = Path(output_root).expanduser()
    if not destination_logical.is_absolute() or destination_logical.is_symlink():
        raise MutExactPostprocessError(
            "Matrix append output root must be an absolute non-symlink path"
        )
    destination = destination_logical.resolve(strict=False)
    if destination.exists():
        raise MutExactPostprocessError(
            f"Matrix append output root must be absent: {destination}"
        )
    _require_output_path_isolation(
        output_paths={"matrix_output_root": destination},
        protected_paths={
            "prior_matrix_root": Path(prior["root"]),
            "standardized_root": cell_root,
        },
    )
    result = audit_registry(
        AuditConfig(scan_roots=(), output_root=destination, explicit_cells=explicit_cells)
    )
    current_rows = {
        (str(row["dataset"]), str(row["method"])): dict(row)
        for row in result.matrix_rows
    }
    for key, old_row in prior_rows.items():
        if key != target_key and current_rows.get(key) != old_row:
            raise MutExactPostprocessError(
                f"Non-target matrix row drifted during Mut append: {key}"
            )
    expected_complete = int(prior["complete"]) + 1
    if result.matrix_complete_cells != expected_complete:
        raise MutExactPostprocessError(
            "Mut append must add exactly one cell: "
            f"prior={prior['complete']}, new={result.matrix_complete_cells}"
        )
    target = current_rows[target_key]
    if (
        target.get("status") != CellStatus.FROZEN_PASS.value
        or Path(str(target.get("standardized_output_root") or "")).resolve(
            strict=True
        )
        != cell_root
        or target.get("k_max") != 20
        or target.get("table2_k") != 10
    ):
        raise MutExactPostprocessError(
            "Mut/ComRecGC standardized artifact failed the frozen-cell gate"
        )
    shared_fields = SHARED_MUT_IDENTITY_FIELDS
    if any(target.get(field) != reference.get(field) for field in shared_fields):
        raise MutExactPostprocessError(
            "Mut/ComRecGC is not identity-compatible with frozen Mut/Ours"
        )

    execution = dict(git_identity or _git_identity())
    if set(execution) != {"commit", "tree"} or any(
        not re.fullmatch(r"[0-9a-f]{40}", str(execution[field]))
        for field in ("commit", "tree")
    ):
        raise MutExactPostprocessError("Execution Git identity is incomplete")
    marker = f"[MATRIX_{expected_complete}_OF_16_PASS]"
    append_manifest = {
        "schema_version": MATRIX_APPEND_SCHEMA,
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution": execution,
        "prior_authority_root": str(prior["root"]),
        "prior_matrix_complete_cells": prior["complete"],
        "prior_matrix_status_sha256": prior["matrix_sha256"],
        "prior_combined_audit_sha256": prior["combined_sha256"],
        "appended_cell": {
            "dataset": TARGET_DATASET,
            "method": TARGET_METHOD,
            "standardized_output_root": str(cell_root),
            "status": target["status"],
            "registry_row": target,
            "source_inventory": inventory,
            "source_inventory_sha256": stable_json_sha256(inventory),
            "writer_audit": writer_audit,
        },
        "unchanged_non_target_rows": True,
        "new_matrix_complete_cells": expected_complete,
        "new_matrix_total_cells": 16,
        "new_authority_root": str(destination),
        "shared_mutagenicity_identity_fields": list(shared_fields),
        "scientific_metrics_recomputed": False,
        "raw_test_opened": False,
        "numeric_imputation_used": False,
        "marker": marker,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.staging-{secrets.token_hex(16)}"
    )
    if staging.exists():  # pragma: no cover - cryptographic collision guard
        raise MutExactPostprocessError(
            f"Matrix publication staging root already exists: {staging}"
        )
    write_registry_outputs(
        result,
        staging,
        supplemental_outputs={
            "append_authority.json": _json_bytes(append_manifest),
        },
    )
    staged = _verify_authority(staging, expected_complete=expected_complete)
    if staged["rows"] != current_rows:
        raise MutExactPostprocessError("Staged matrix rows changed before publication")
    if _read_json(staging / "append_authority.json") != append_manifest:
        raise MutExactPostprocessError(
            "Staged Mut append receipt changed before publication"
        )
    atomic_rename_directory_noreplace(staging, destination)
    return _reopen_existing_matrix_append(
        prior_authority_root=Path(prior["root"]),
        standardized_root=cell_root,
        output_root=destination,
        proc_root=Path(proc_root),
        require_writer_audit=require_writer_audit,
        adopted_after_interruption=False,
    )


def _reopen_existing_matrix_append(
    *,
    prior_authority_root: Path,
    standardized_root: Path,
    output_root: Path,
    proc_root: Path = Path("/proc"),
    require_writer_audit: bool = True,
    adopted_after_interruption: bool = True,
) -> dict[str, Any]:
    prior = _verify_authority(prior_authority_root)
    cell_root = _physical_directory(
        standardized_root, label="standardized cell root on matrix reopen"
    )
    destination = _physical_directory(
        output_root, label="published matrix authority on reopen"
    )
    _require_output_path_isolation(
        output_paths={"matrix_output_root": destination},
        protected_paths={
            "prior_matrix_root": Path(prior["root"]),
            "standardized_root": cell_root,
        },
    )
    expected_complete = int(prior["complete"]) + 1
    current = _verify_authority(destination, expected_complete=expected_complete)
    append = _read_json(destination / "append_authority.json")
    target_key = (TARGET_DATASET, TARGET_METHOD)
    target = current["rows"].get(target_key)
    reference = prior["rows"].get((TARGET_DATASET, "Ours"))
    passing = {status.value for status in PASS_STATUSES}
    failures: list[str] = []
    if not isinstance(target, Mapping):
        failures.append("target_row")
        target = {}
    if not isinstance(reference, Mapping) or str(reference.get("status") or "") not in passing:
        failures.append("reference_row")
        reference = {}
    for key, prior_row in prior["rows"].items():
        if key != target_key and current["rows"].get(key) != prior_row:
            failures.append(f"non_target_row:{key[0]}/{key[1]}")
    try:
        target_root = Path(
            str(target.get("standardized_output_root") or "")
        ).resolve(strict=True)
    except FileNotFoundError:
        target_root = Path("/")
    if (
        target.get("status") != CellStatus.FROZEN_PASS.value
        or target_root != cell_root
        or target.get("k_max") != 20
        or target.get("table2_k") != 10
    ):
        failures.append("target_frozen_contract")
    if any(
        target.get(field) != reference.get(field)
        for field in SHARED_MUT_IDENTITY_FIELDS
    ):
        failures.append("shared_mutagenicity_identity")

    expected_scalars = {
        "schema_version": MATRIX_APPEND_SCHEMA,
        "status": "PASS",
        "prior_authority_root": str(prior["root"]),
        "prior_matrix_complete_cells": prior["complete"],
        "prior_matrix_status_sha256": prior["matrix_sha256"],
        "prior_combined_audit_sha256": prior["combined_sha256"],
        "unchanged_non_target_rows": True,
        "new_matrix_complete_cells": expected_complete,
        "new_matrix_total_cells": 16,
        "new_authority_root": str(destination),
        "shared_mutagenicity_identity_fields": list(SHARED_MUT_IDENTITY_FIELDS),
        "scientific_metrics_recomputed": False,
        "raw_test_opened": False,
        "numeric_imputation_used": False,
        "marker": f"[MATRIX_{expected_complete}_OF_16_PASS]",
    }
    failures.extend(
        name for name, expected in expected_scalars.items() if append.get(name) != expected
    )
    execution = append.get("execution")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != {"commit", "tree"}
        or any(
            not re.fullmatch(r"[0-9a-f]{40}", str(execution.get(field) or ""))
            for field in ("commit", "tree")
        )
    ):
        failures.append("execution")
    if not str(append.get("created_at") or "").strip():
        failures.append("created_at")

    appended_cell = append.get("appended_cell")
    inventory = _inventory(cell_root)
    if not isinstance(appended_cell, Mapping):
        failures.append("appended_cell")
        appended_cell = {}
    appended_expected = {
        "dataset": TARGET_DATASET,
        "method": TARGET_METHOD,
        "standardized_output_root": str(cell_root),
        "status": CellStatus.FROZEN_PASS.value,
        "registry_row": target,
        "source_inventory": inventory,
        "source_inventory_sha256": stable_json_sha256(inventory),
    }
    failures.extend(
        f"appended_cell.{name}"
        for name, expected in appended_expected.items()
        if appended_cell.get(name) != expected
    )
    writer_audit = appended_cell.get("writer_audit")
    if not isinstance(writer_audit, Mapping):
        failures.append("appended_cell.writer_audit")
    elif require_writer_audit:
        if (
            writer_audit.get("procfs_verified") is not True
            or writer_audit.get("writable_fd_count") != 0
            or writer_audit.get("writers") != []
        ):
            failures.append("appended_cell.writer_audit")
        current_writer_audit = scan_live_writers(cell_root, proc_root=proc_root)
        if (
            current_writer_audit.get("procfs_verified") is not True
            or current_writer_audit.get("writable_fd_count") != 0
            or current_writer_audit.get("writers") != []
        ):
            failures.append("current_writer_audit")
    else:
        if (
            writer_audit.get("procfs_verified") is not False
            or writer_audit.get("writable_fd_count") != 0
            or writer_audit.get("writers") != []
        ):
            failures.append("appended_cell.writer_audit_test_contract")
    if failures:
        raise MutExactPostprocessError(
            "Existing matrix append cannot be adopted: "
            + ", ".join(sorted(set(failures)))
        )
    return {
        "status": "PASS",
        "output_root": str(destination),
        "matrix_status_path": str(destination / "matrix_status.json"),
        "matrix_status_sha256": current["matrix_sha256"],
        "combined_audit_sha256": current["combined_sha256"],
        "matrix_complete_cells": current["complete"],
        "matrix_total_cells": 16,
        "appended_cell": f"{TARGET_DATASET}/{TARGET_METHOD}",
        "appended_standardized_root": str(cell_root),
        "marker": append.get("marker"),
        "adopted_after_interruption": adopted_after_interruption,
    }


def run_mut_exact_postprocess(
    *,
    inputs: ContinuationInputs,
    exact_adoption_receipt: str | Path,
    common_root: str | Path,
    trace_parity_path: str | Path,
    prior_matrix_root: str | Path,
    matrix_output_root: str | Path,
    resume: bool,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Run only downstream science, then atomically append Mut/ComRecGC."""

    if inputs.dataset != "mutagenicity" or inputs.device != "cpu":
        raise MutExactPostprocessError("This continuation is Mutagenicity CPU-only")
    output_logical = inputs.output_root.expanduser()
    if not output_logical.is_absolute() or output_logical.is_symlink():
        raise MutExactPostprocessError(
            "Postprocess output root must be an absolute non-symlink path"
        )
    output = output_logical.resolve(strict=False)
    if resume:
        if not output.is_dir() or (output / "PASS").exists():
            raise MutExactPostprocessError(
                "Resume requires one incomplete postprocess root without PASS"
            )
    elif output.exists():
        raise MutExactPostprocessError("Fresh postprocess output root already exists")

    # All science authorities are validated before a fresh output directory is
    # created.  In particular, a missing Mut trace-parity receipt fails here.
    parity_file = _physical_file(trace_parity_path, label="Mut trace parity receipt")
    parity = _validate_parity_gate(parity_file)
    source_generation = _physical_directory(
        inputs.source_generation_root, label="frozen generation root"
    )
    parity_generation_root = _physical_directory(
        str(parity.get("traced_source_root") or ""),
        label="trace parity traced source root",
    )
    if parity_generation_root != source_generation:
        raise MutExactPostprocessError(
            "Trace parity gate is not bound to the adopted generation root"
        )
    exact = validate_exact_adoption(
        adoption_receipt_path=exact_adoption_receipt,
        common_root=common_root,
        source_generation_root=inputs.source_generation_root,
        proc_root=proc_root,
    )
    generation = validate_adopted_generation(inputs)
    if (
        int(generation["counterfactual_candidate_count"]) != SOURCE_CANDIDATE_COUNT
        or generation["counterfactuals_sha256_actual"] != SOURCE_PAYLOAD_SHA256
    ):
        raise MutExactPostprocessError("Frozen generation identity changed")
    checkout = verify_checkout(
        inputs.upstream_root,
        expected_commit=UPSTREAM_COMMIT,
        validate_imports=True,
    )
    prior_matrix = _verify_authority(prior_matrix_root)
    if str(
        prior_matrix["rows"][(TARGET_DATASET, TARGET_METHOD)].get("status") or ""
    ) in {status.value for status in PASS_STATUSES}:
        raise MutExactPostprocessError("Prior authority already passes Mut/ComRecGC")
    matrix_output_logical = Path(matrix_output_root).expanduser()
    if not matrix_output_logical.is_absolute() or matrix_output_logical.is_symlink():
        raise MutExactPostprocessError(
            "Matrix output root must be an absolute non-symlink path"
        )
    matrix_output = matrix_output_logical.resolve(strict=False)
    if not resume and matrix_output.exists():
        raise MutExactPostprocessError(
            f"Fresh matrix output root already exists: {matrix_output}"
        )
    protected_paths = {
        "source_generation_root": source_generation,
        "exact_common_root": Path(exact["common_root"]),
        "exact_adoption_receipt": Path(exact["adoption_receipt_path"]),
        "exact_controller_state": Path(exact["source_controller_state_path"]),
        "trace_parity_receipt": parity_file,
        "upstream_root": _physical_directory(
            inputs.upstream_root, label="upstream checkout"
        ),
        "dataset_dir": _physical_directory(
            inputs.dataset_dir, label="Mutagenicity dataset directory"
        ),
        "molclr_root": _physical_directory(
            inputs.molclr_root, label="MolCLR checkout"
        ),
        "distance_checkpoint": _physical_file(
            inputs.distance_checkpoint, label="distance checkpoint"
        ),
        "dataset_csv": _physical_file(inputs.dataset_csv, label="dataset CSV"),
        "teacher_path": _physical_file(inputs.teacher_path, label="teacher model"),
        "molclr_checkpoint": _physical_file(
            inputs.molclr_checkpoint, label="MolCLR checkpoint"
        ),
        "thresholds_path": _physical_file(
            inputs.thresholds_path, label="threshold contract"
        ),
        "prior_matrix_root": Path(prior_matrix["root"]),
    }
    if inputs.source_csv is not None:
        protected_paths["source_csv"] = _physical_file(
            inputs.source_csv, label="explicit source CSV"
        )
    output_path_isolation = _require_output_path_isolation(
        output_paths={
            "postprocess_output_root": output,
            "matrix_output_root": matrix_output,
        },
        protected_paths=protected_paths,
    )
    project_commit = _git_head()
    teacher_sha256 = sha256_file(inputs.teacher_path)
    common = Path(exact["common_root"])
    commands = build_postprocess_commands(
        inputs,
        common_root=common,
        parity_path=parity_file,
        project_commit=project_commit,
        teacher_sha256=teacher_sha256,
    )

    if not resume:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o755)
        write_json(output / "generation_adoption_manifest.json", generation)
        write_json(output / "exact_common_adoption_manifest.json", exact)
        write_json(output / "trace_parity_adoption_manifest.json", parity)
        write_json(output / "upstream_checkout_audit.json", checkout)

    base_contract = _resume_contract(
        inputs=inputs,
        adoption=generation,
        checkout=_load_object(output / "upstream_checkout_audit.json"),
        project_commit=project_commit,
        teacher_sha256=teacher_sha256,
        commands=commands,
    )
    contract = {
        **base_contract,
        "schema_version": "mut_comrecgc_exact_postprocess_resume_v1",
        "exact_adoption_receipt_path": str(
            Path(exact["adoption_receipt_path"])
        ),
        "exact_adoption_receipt_sha256": exact["adoption_receipt_sha256"],
        "adopted_common_root": str(common),
        "adopted_common_terminal_sha256": exact["common_terminal_sha256"],
        "adopted_common_manifest_sha256": exact["common_manifest_sha256"],
        "expected_common_recourse_count": EXPECTED_COMMON_RECOURSES,
        "expected_common_recourse_parameters": dict(EXPECTED_RECOURSE_PARAMETERS),
        "dbscan_scientific_identity_sha256": exact[
            "dbscan_scientific_identity_sha256"
        ],
        "source_controller_state_sha256": exact[
            "source_controller_state_sha256"
        ],
        "trace_parity_path": str(parity_file),
        "trace_parity_sha256": parity["sha256"],
        "trace_parity_traced_source_root": str(parity_generation_root),
        "prior_matrix_root": str(prior_matrix["root"]),
        "prior_matrix_status_sha256": prior_matrix["matrix_sha256"],
        "prior_matrix_combined_audit_sha256": prior_matrix["combined_sha256"],
        "matrix_output_root": str(matrix_output),
        "output_path_isolation": output_path_isolation,
        "forbidden_reruns": ["pair_store", "DBSCAN", "common_recourse"],
    }
    contract_path = output / "continuation_resume_contract.json"
    if resume:
        frozen = _load_object(_require_file(contract_path))
        if frozen != contract:
            changed = sorted(
                key
                for key in set(frozen) | set(contract)
                if frozen.get(key) != contract.get(key)
            )
            raise MutExactPostprocessError(
                f"RESUME_SCIENTIFIC_CONTRACT_MISMATCH:fields={changed}"
            )
        _archive_previous_failure(output)
    else:
        write_json(contract_path, contract)

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "DEVICE": "cpu",
    }
    checkpoints = output / "stage_checkpoints"
    try:
        for stage, argv, marker, required_field in commands:
            checkpoint = checkpoints / f"{stage}.json"
            if resume and marker.exists():
                _require_completed_stage_writer_quiescence(
                    output_root=output,
                    stage=stage,
                    argv=argv,
                    checkpoint_path=checkpoint,
                )
                _validate_completed_stage(
                    stage=stage,
                    argv=argv,
                    marker=marker,
                    required_field=required_field,
                    checkpoint_path=checkpoint,
                )
                continue
            if resume and marker.parent.exists():
                _archive_noncheckpointed_partial_stage(
                    stage=stage,
                    argv=argv,
                    marker=marker,
                    required_field=required_field,
                    checkpoint_path=checkpoint,
                    output_root=output,
                )
            _run_stage(
                stage=stage,
                argv=argv,
                marker=marker,
                required_field=required_field,
                environment=environment,
                output_root=output,
                checkpoint_path=checkpoint,
            )

        standardized = _physical_directory(
            output / "standardized", label="standardized Mut/ComRecGC output"
        )
        standardized_manifest = _load_object(
            _require_file(standardized / "run_manifest.json")
        )
        freeze_manifest = _load_object(
            _require_file(standardized / "freeze_manifest.json")
        )
        failures: list[str] = []
        if standardized_manifest.get("dataset_key") != "mutagenicity":
            failures.append("dataset")
        if standardized_manifest.get("cf_mode") != CF_MODE:
            failures.append("cf_mode")
        if standardized_manifest.get("distance_line") != DISTANCE_LINE:
            failures.append("distance_line")
        if standardized_manifest.get("teacher_sha256") != teacher_sha256:
            failures.append("teacher")
        if freeze_manifest.get("dataset_key") != "mutagenicity":
            failures.append("freeze_dataset")
        if failures:
            raise MutExactPostprocessError(
                f"Standardized output identity mismatch: {failures}"
            )

        generation_final = _verify_adopted_generation_integrity(generation)
        exact_final = validate_exact_adoption(
            adoption_receipt_path=exact_adoption_receipt,
            common_root=common,
            source_generation_root=inputs.source_generation_root,
            proc_root=proc_root,
        )
        write_json(output / "source_generation_integrity_final.json", generation_final)
        write_json(output / "source_exact_integrity_final.json", exact_final)
        science = {
            "schema_version": "mut_comrecgc_exact_postprocess_science_v1",
            "status": "PASS",
            "dataset": "mutagenicity",
            "method": METHOD,
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "trace_parity_passed": True,
            "generation_adopted": True,
            "generation_rerun": False,
            "common_recourse_adopted": True,
            "common_recourse_rerun": False,
            "dbscan_rerun": False,
            "pair_store_rerun": False,
            "chemistry_rerun": True,
            "wnode_evaluation_rerun": True,
            "expected_common_recourse_count": EXPECTED_COMMON_RECOURSES,
            "standardized_output_root": str(standardized),
            "teacher_sha256": teacher_sha256,
            "calibration_loaded": False,
            "test_loaded_only_in_unified_evaluation": True,
            "completed_at": _utc_now(),
        }
        write_json(output / "science_manifest.json", science)
        write_json(output / "_SCIENCE_COMPLETE.json", {**science, "run_complete": True})

        if matrix_output.exists():
            matrix = _reopen_existing_matrix_append(
                prior_authority_root=Path(prior_matrix_root),
                standardized_root=standardized,
                output_root=matrix_output,
                proc_root=Path(proc_root),
            )
        else:
            matrix = _append_mut_matrix_authority(
                prior_authority_root=prior_matrix_root,
                standardized_root=standardized,
                output_root=matrix_output,
                proc_root=proc_root,
            )
        write_json(output / "matrix_append_receipt.json", matrix)
        final = {
            **science,
            "schema_version": RUN_SCHEMA,
            "matrix_append_status": "PASS",
            "matrix_output_root": matrix["output_root"],
            "matrix_complete_cells": matrix["matrix_complete_cells"],
            "matrix_total_cells": 16,
            "matrix_status_sha256": matrix["matrix_status_sha256"],
            "run_complete": True,
            "completed_at": _utc_now(),
        }
        write_json(output / "run_manifest.json", final)
        write_json(output / "_RUN_COMPLETE.json", final)
        atomic_write_bytes(output / "PASS", b"PASS\n")
        return final
    except Exception as exc:
        write_json(
            output / "FAILED.json",
            {
                "schema_version": "mut_comrecgc_exact_postprocess_failure_v1",
                "status": "FAILED",
                "error_class": type(exc).__name__,
                "message": str(exc),
                "common_recourse_rerun": False,
                "dbscan_rerun": False,
                "pair_store_rerun": False,
                "failed_at": _utc_now(),
            },
        )
        raise
