"""Fresh policy-v2 control plane for the TasteMolNet main-table route.

The historical four-dataset controller remains immutable.  This module reads
its blocked Taste task as evidence, proves that no heavy Taste science was
started, publishes one independent policy-adoption receipt, and then delegates
T2 to the reviewed persistent GINE controller.  Later method stages remain
visible in the queue but cannot run until their own typed implementations pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import tempfile
import threading
from typing import Any, Mapping

from src.utils.autodl_tastemolnet_gine_controller_v1 import (
    TasteGINEControllerSpec,
    inspect_tastemolnet_gine_controller,
    run_tastemolnet_gine_controller,
)
from src.utils.tastemolnet_research_policy import (
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    sha256_file,
    stable_json_sha256,
    validate_tastemolnet_local_authority,
    validate_tastemolnet_policy_receipt,
)


MAIN_SCHEMA = "autodl_tastemolnet_main_controller_v1"
ADOPTION_SCHEMA = "tastemolnet_policy_adoption_v2"
STAGE_SCHEMA = "tastemolnet_main_stage_evidence_v1"
QUEUE_SCHEMA = "tastemolnet_main_queue_v1"
STATE_SCHEMA = "tastemolnet_main_state_v1"
HEARTBEAT_SCHEMA = "tastemolnet_main_heartbeat_v1"
NAMESPACE_NAME = "tastemolnet-main-v1"
OLD_TASK_ID = "tastemolnet_foundation"
OLD_BLOCKER = "TASTEMOLNET_FOUNDATION_BLOCKED_LICENSE_REVIEW"
OLD_STATE = "BLOCKED_LICENSE_REVIEW"
SUPERSEDED_STATE = "SUPERSEDED_POLICY_V1"
CURRENT_ROUTE_STATE = "READY_FOR_MAIN_ROUTE"
POLICY_MARKER = "TASTE_RESEARCH_POLICY_V2_PASS"
NO_REDISTRIBUTION_MARKER = "TASTE_NO_DATA_REDISTRIBUTION_GUARD_PASS"
SUPERSEDED_MARKER = "TASTE_OLD_LICENSE_BLOCK_SUPERSEDED"
CONTROLLER_ID_PATTERN = re.compile(
    r"^tastemolnet-main-v1-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)
STAGES = (
    "T0_POLICY_MIGRATION",
    "T1_DATA_READY",
    "T2_GINE_FULL",
    "T3_GINE_CALIBRATED",
    "T4_ORACLE_SMOKE",
    "T5_CLEAN_POLICY_READY",
    "T6_OURS_SMOKE",
    "T7_GCF_SMOKE",
    "T8_GLOBALGCE_SMOKE",
    "T9_COMRECGC_SMOKE",
    "T10_METHOD_SMOKES_PASS",
    "T11_OURS_FULL",
    "T12_GCF_FULL",
    "T13_GLOBALGCE_FULL",
    "T14_COMRECGC_FULL",
    "T15_UNIFIED_EVAL",
    "T16_FROZEN",
)


class TasteMainControllerError(RuntimeError):
    """The new Taste main route failed a fail-closed authority gate."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _absolute(value: str | Path, *, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise TasteMainControllerError(f"absolute path required: {value}")
    return path.resolve(strict=must_exist)


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _directory_identity(info: os.stat_result) -> dict[str, int]:
    if not stat.S_ISDIR(info.st_mode):
        raise TasteMainControllerError("controller root must remain a directory")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
    }


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes, dict[str, int]]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteMainControllerError(f"{label} must be one regular single-link file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(after) != _identity(named):
            raise TasteMainControllerError(f"{label} changed while read")
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteMainControllerError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TasteMainControllerError(f"{label} must contain one JSON object")
    return payload, data, _identity(after)


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    data = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_new(path: Path, value: str) -> None:
    data = (value.rstrip("\n") + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(payload), sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_main_lock(namespace_root: Path) -> int:
    descriptor = os.open(
        namespace_root / ".main-controller.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return "NON_LINUX_BOOT_ID_UNAVAILABLE"


def _validate_runtime_environment() -> None:
    expected = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "TASTE_UPSTREAM_LICENSE_STATUS": "NOT_EXPLICITLY_STATED",
        "PRIMARY_TASTE_SOURCE_LABEL": "1",
        "RUN_GNN_ABLATION": "0",
        "MAX_CONCURRENT_TASTE_FULL": "2",
        "MIN_FREE_AFTER_RESERVATIONS_GB": "100",
        "TASTEMOLNET_GPU_INDEX": "1",
        "TASTEMOLNET_STORAGE_RESERVATION_GB": "20",
    }
    drift = [key for key, value in expected.items() if os.environ.get(key) != value]
    if drift:
        raise TasteMainControllerError(
            "Taste main runtime environment changed: " + ",".join(sorted(drift))
        )


def _old_state_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    instances = payload.get("instances")
    main = instances.get("main") if isinstance(instances, Mapping) else None
    if not isinstance(main, Mapping):
        raise TasteMainControllerError("old Taste state lost instances.main")
    projection = {
        "schema_version": payload.get("schema_version"),
        "task_id": payload.get("task_id"),
        "dataset": payload.get("dataset"),
        "stage": payload.get("stage"),
        "state": payload.get("state"),
        "reason": payload.get("reason"),
        "instance": {
            key: main.get(key)
            for key in (
                "state",
                "run_id",
                "gpu_index",
                "gpu_uuid",
                "launcher_pid",
                "worker_pid",
                "child_pid",
            )
        },
    }
    if (
        projection["schema_version"] != 1
        or projection["task_id"] != OLD_TASK_ID
        or projection["dataset"] != "tastemolnet"
        or projection["state"] != "BLOCKED"
        or projection["reason"] != OLD_BLOCKER
        or projection["instance"]
        != {
            "state": "NOT_STARTED",
            "run_id": None,
            "gpu_index": None,
            "gpu_uuid": None,
            "launcher_pid": None,
            "worker_pid": None,
            "child_pid": None,
        }
    ):
        raise TasteMainControllerError("old Taste task contains science or changed state")
    return projection


def _validate_old_block(
    *, source_manifest: Path, task_root: Path
) -> dict[str, Any]:
    state_before, state_before_data, state_before_identity = _read_json(
        task_root / "state.json", label="old Taste state"
    )
    projection_before = _old_state_projection(state_before)
    source, source_data, source_identity = _read_json(
        source_manifest, label="old controller source manifest"
    )
    manifest, manifest_data, manifest_identity = _read_json(
        task_root / "manifest.json", label="old Taste manifest"
    )
    gate, gate_data, gate_identity = _read_json(
        task_root / "gate.json", label="old Taste gate"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("task_id") != OLD_TASK_ID
        or manifest.get("dataset") != "tastemolnet"
        or manifest.get("status") != "FROZEN"
        or manifest.get("blocked_reason") != OLD_BLOCKER
        or manifest.get("command") is not None
        or manifest.get("expected_output") is not None
        or manifest.get("adopt_existing_run_id") is not None
        or manifest.get("adopt_gpu_index") is not None
        or manifest.get("adopt_gpu_uuid") is not None
        or manifest.get("controller_manifest_sha256") != _sha256(source_data)
    ):
        raise TasteMainControllerError("old Taste manifest is not the frozen no-science block")
    runs = gate.get("runs")
    if (
        gate.get("schema_version") != 1
        or gate.get("task_id") != OLD_TASK_ID
        or gate.get("status") != "BLOCKED"
        or gate.get("reason") != OLD_BLOCKER
        or not isinstance(runs, list)
        or len(runs) != 1
        or not isinstance(runs[0], Mapping)
        or runs[0].get("state") != "NOT_STARTED"
        or any(
            runs[0].get(key) is not None
            for key in ("run_id", "gpu_index", "gpu_uuid", "expected_output")
        )
    ):
        raise TasteMainControllerError("old Taste gate is not blocked-before-science")
    state_after, state_after_data, state_after_identity = _read_json(
        task_root / "state.json", label="old Taste state after authority scan"
    )
    projection_after = _old_state_projection(state_after)
    if projection_after != projection_before:
        raise TasteMainControllerError("old Taste stable state changed during adoption")
    return {
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": _sha256(source_data),
            "identity": source_identity,
        },
        "task_manifest": {
            "path": str(task_root / "manifest.json"),
            "sha256": _sha256(manifest_data),
            "identity": manifest_identity,
        },
        "task_gate": {
            "path": str(task_root / "gate.json"),
            "sha256": _sha256(gate_data),
            "identity": gate_identity,
        },
        "task_state": {
            "path": str(task_root / "state.json"),
            "before_sha256": _sha256(state_before_data),
            "after_sha256": _sha256(state_after_data),
            "before_identity": state_before_identity,
            "after_identity": state_after_identity,
            "stable_projection": projection_after,
            "stable_projection_sha256": stable_json_sha256(projection_after),
            "mutable_fields": ["created_at", "updated_at"],
        },
    }


def _stage_payloads(
    *, stage: str, controller_id: str, status: str, evidence: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    now = _utc()
    base = {
        "schema_version": STAGE_SCHEMA,
        "stage": stage,
        "controller_id": controller_id,
        "status": status,
        "updated_at": now,
    }
    manifest = {**base, "artifact_kind": "manifest", "evidence": dict(evidence)}
    state = {**base, "artifact_kind": "state"}
    gate = {
        **base,
        "artifact_kind": "gate",
        "passed": status == "PASS",
    }
    inputs = {
        **base,
        "artifact_kind": "input_hashes",
        "evidence_sha256": stable_json_sha256(dict(evidence)),
    }
    outputs = {
        **base,
        "artifact_kind": "output_hashes",
        "manifest_sha256": stable_json_sha256(manifest),
        "gate_sha256": stable_json_sha256(gate),
    }
    return {
        "manifest.json": manifest,
        "state.json": state,
        "gate.json": gate,
        "input_hashes.json": inputs,
        "output_hashes.json": outputs,
    }


@dataclass(frozen=True, slots=True)
class TasteMainSpec:
    controller_id: str
    control_root: Path
    runtime_root: Path
    controller_root: Path
    old_source_manifest: Path
    old_task_root: Path
    policy_path: Path
    policy_receipt: Path
    prepared_root: Path
    graph_cache_root: Path
    project_root: Path
    gine_controller_root: Path
    gine_output_root: Path
    gine_training_state_root: Path
    reservation_gb: int = 20
    minimum_free_after_reservations_gb: int = 100

    @property
    def namespace_root(self) -> Path:
        return self.control_root / NAMESPACE_NAME


def _queue(spec: TasteMainSpec, *, t2_status: str) -> dict[str, Any]:
    dependencies = {
        "T0_POLICY_MIGRATION": [],
        "T1_DATA_READY": ["T0_POLICY_MIGRATION"],
        "T2_GINE_FULL": ["T1_DATA_READY"],
        "T3_GINE_CALIBRATED": ["T2_GINE_FULL"],
        "T4_ORACLE_SMOKE": ["T3_GINE_CALIBRATED"],
        "T5_CLEAN_POLICY_READY": ["T1_DATA_READY"],
        "T6_OURS_SMOKE": ["T4_ORACLE_SMOKE", "T5_CLEAN_POLICY_READY"],
        "T7_GCF_SMOKE": ["T4_ORACLE_SMOKE"],
        "T8_GLOBALGCE_SMOKE": ["T4_ORACLE_SMOKE"],
        "T9_COMRECGC_SMOKE": ["T4_ORACLE_SMOKE"],
        "T10_METHOD_SMOKES_PASS": [
            "T6_OURS_SMOKE",
            "T7_GCF_SMOKE",
            "T8_GLOBALGCE_SMOKE",
            "T9_COMRECGC_SMOKE",
        ],
        "T11_OURS_FULL": ["T10_METHOD_SMOKES_PASS"],
        "T12_GCF_FULL": ["T10_METHOD_SMOKES_PASS"],
        "T13_GLOBALGCE_FULL": ["T10_METHOD_SMOKES_PASS"],
        "T14_COMRECGC_FULL": ["T10_METHOD_SMOKES_PASS"],
        "T15_UNIFIED_EVAL": [
            "T11_OURS_FULL",
            "T12_GCF_FULL",
            "T13_GLOBALGCE_FULL",
            "T14_COMRECGC_FULL",
        ],
        "T16_FROZEN": ["T15_UNIFIED_EVAL"],
    }
    rows = []
    for index, stage in enumerate(STAGES):
        if index < 2:
            status = "PASS"
        elif index == 2:
            status = t2_status
        elif stage == "T3_GINE_CALIBRATED" and t2_status == "PASS":
            status = "READY"
        elif stage == "T5_CLEAN_POLICY_READY":
            status = "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT"
        else:
            status = "WAITING_DEPENDENCY"
        rows.append(
            {
                "stage": stage,
                "status": status,
                "depends_on": dependencies[stage],
                "gnn_backbone_ablation": False,
            }
        )
    return {
        "schema_version": QUEUE_SCHEMA,
        "controller_id": spec.controller_id,
        "scheduler": "LONGEST_READY_FIRST_AFTER_BOUNDED_SMOKES",
        "max_concurrent_taste_full": 2,
        "gnn_ablation_started": False,
        "gnn_ablation_marker": "GNN_BACKBONE_ABLATION_NOT_STARTED_BY_POLICY",
        "gnn_ablation_recommended_start_time": "AFTER_MAIN_16_16",
        "resource_lanes": {
            "gpu1_taste_gine_full": {
                "gpu_index": 1,
                "status": t2_status,
                "classifier_dependent": True,
            },
            "gpu2_classifier_independent_precompute": {
                "gpu_index": 2,
                "status": "READY_CLASSIFIER_INDEPENDENT_PRECOMPUTE",
                "allowed_splits": ["train"],
                "initializer_data_split_used": "none",
                "taste_split_access_max": "train_only",
                "t5_release_enabled": False,
                "t5_release_state": "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT",
                "test_loaded": False,
                "classifier_dependent": False,
                "science_started": False,
            },
            "gpu0_bace_protected": {"gpu_index": 0, "status": "EXCLUDED"},
            "gpu3_bace_protected": {"gpu_index": 3, "status": "EXCLUDED"},
        },
        "stages": rows,
        "updated_at": _utc(),
    }


def _write_stage(root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, payload in payloads.items():
        _atomic_json(root / name, payload, replace=False)


def _replace_stage(root: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    for name, payload in payloads.items():
        _atomic_json(root / name, payload, replace=True)


def _matrix_evidence() -> dict[str, Any]:
    raw = os.environ.get("TASTE_MATRIX_STATUS_PATH", "")
    path = Path(raw)
    if not raw or not path.is_absolute():
        raise TasteMainControllerError("TASTE_MATRIX_STATUS_PATH must be absolute")
    if not path.exists():
        raise TasteMainControllerError(
            "main matrix status is required for read-only nonmutation evidence"
        )
    payload, data, identity = _read_json(path, label="main matrix status")
    cells = payload.get("cells")
    if (
        payload.get("schema_version") != "four_methods_four_datasets_registry_v1"
        or type(payload.get("matrix_complete_cells")) is not int
        or type(payload.get("matrix_total_cells")) is not int
        or payload.get("matrix_total_cells") != 16
        or not isinstance(cells, list)
        or len(cells) != 16
        or any(not isinstance(row, Mapping) for row in cells)
    ):
        raise TasteMainControllerError("main matrix status schema changed")
    taste_rows = [row for row in cells if row.get("dataset") == "TasteMolNet"]
    expected_methods = {"Ours", "GCFExplainer", "GlobalGCE", "ComRecGC"}
    if (
        len(taste_rows) != 4
        or {row.get("method") for row in taste_rows} != expected_methods
        or any(row.get("method") == "GINE" for row in cells)
    ):
        raise TasteMainControllerError("main matrix Taste method projection changed")
    return {
        "path": str(path),
        "status": "PRESENT_READ_ONLY",
        "sha256": _sha256(data),
        "identity": identity,
        "schema_version": payload.get("schema_version"),
        "matrix_complete_cells": payload["matrix_complete_cells"],
        "matrix_total_cells": payload["matrix_total_cells"],
        "taste_method_statuses": {
            str(row["method"]): row.get("status") for row in taste_rows
        },
        "taste_gine_counts_as_method_cell": False,
    }


def prepare_tastemolnet_main(spec: TasteMainSpec) -> dict[str, Any]:
    _validate_runtime_environment()
    if not CONTROLLER_ID_PATTERN.fullmatch(spec.controller_id):
        raise TasteMainControllerError("Taste main controller ID is malformed")
    if spec.namespace_root != spec.control_root / NAMESPACE_NAME:
        raise TasteMainControllerError("Taste main namespace escaped its fixed control root")
    for path in (
        spec.control_root,
        spec.runtime_root,
        spec.old_source_manifest,
        spec.old_task_root,
        spec.policy_path,
        spec.policy_receipt,
        spec.prepared_root,
        spec.graph_cache_root,
        spec.project_root,
    ):
        _absolute(path, must_exist=True)
    if spec.controller_root.parent != spec.namespace_root:
        raise TasteMainControllerError("controller root must be one direct fresh namespace child")
    if spec.controller_root.exists():
        raise TasteMainControllerError("Taste main controller root must be fresh")
    for path in (
        spec.gine_controller_root,
        spec.gine_output_root,
        spec.gine_training_state_root,
    ):
        if path.exists():
            raise TasteMainControllerError("Taste GINE output/control/state roots must be fresh")
    if (
        type(spec.reservation_gb) is not int
        or type(spec.minimum_free_after_reservations_gb) is not int
        or spec.reservation_gb <= 0
        or spec.minimum_free_after_reservations_gb != 100
    ):
        raise TasteMainControllerError("Taste storage reservation contract is invalid")
    free_gb = shutil.disk_usage(spec.runtime_root).free // (1024**3)
    if free_gb - spec.reservation_gb < spec.minimum_free_after_reservations_gb:
        raise TasteMainControllerError("TASTE_BLOCKED_STORAGE")

    policy = load_tastemolnet_research_policy(spec.policy_path)
    policy.require_main_route()
    authority = validate_tastemolnet_local_authority(
        policy,
        prepared_root=spec.prepared_root,
        graph_cache_root=spec.graph_cache_root,
    )
    receipt = validate_tastemolnet_policy_receipt(
        spec.policy_receipt,
        policy=policy,
        authority=authority,
        require_active=True,
        require_policy_version=2,
    )
    old = _validate_old_block(
        source_manifest=spec.old_source_manifest,
        task_root=spec.old_task_root,
    )

    spec.namespace_root.mkdir(mode=0o700, parents=False, exist_ok=False)
    adoption = {
        "schema_version": ADOPTION_SCHEMA,
        "created_at": _utc(),
        "old_controller_id": "four_methods_four_datasets_continuation_v1",
        "old_manifest_path": old["task_manifest"]["path"],
        "old_manifest_sha256": old["task_manifest"]["sha256"],
        "old_state": OLD_STATE,
        "old_blocker_code": OLD_BLOCKER,
        "old_authority": old,
        "new_policy_path": str(policy.path),
        "new_policy_sha256": policy.file_sha256,
        "new_policy_version": 2,
        "new_policy_receipt_path": str(receipt.path),
        "new_policy_receipt_sha256": receipt.sha256,
        "research_compute_allowed": True,
        "paper_result_reporting_allowed": True,
        "data_redistribution_allowed": False,
        "upstream_license_status": "NOT_EXPLICITLY_STATED",
        "upstream_license_claimed_resolved": False,
        "old_state_superseded": True,
        "superseded_state": SUPERSEDED_STATE,
        "current_route_state": CURRENT_ROUTE_STATE,
        "old_science_adopted": False,
        "reason": "no_heavy_science_was_started",
        "license_pass_claimed": False,
        "terminal_markers": [POLICY_MARKER, NO_REDISTRIBUTION_MARKER],
    }
    _atomic_json(spec.namespace_root / "policy_adoption.json", adoption, replace=False)

    spec.controller_root.mkdir(mode=0o700, exist_ok=False)
    controller_root_identity = _directory_identity(os.stat(spec.controller_root))
    controller_spec = {
        "schema_version": MAIN_SCHEMA,
        "controller_id": spec.controller_id,
        "controller_root": str(spec.controller_root),
        "controller_root_identity": controller_root_identity,
        "project_root": str(spec.project_root),
        "policy_path": str(spec.policy_path),
        "policy_sha256": sha256_file(spec.policy_path),
        "policy_receipt_path": str(spec.policy_receipt),
        "policy_receipt_sha256": sha256_file(spec.policy_receipt),
        "policy_sha256": policy.file_sha256,
        "policy_receipt_path": str(spec.policy_receipt),
        "policy_receipt_sha256": receipt.sha256,
        "prepared_root": str(spec.prepared_root),
        "graph_cache_root": str(spec.graph_cache_root),
        "gine_controller_root": str(spec.gine_controller_root),
        "gine_output_root": str(spec.gine_output_root),
        "gine_training_state_root": str(spec.gine_training_state_root),
        "gine_gpu_index": 1,
        "gpu0_gpu3_protected": True,
        "gnn_ablation_started": False,
        "created_at": _utc(),
    }
    _atomic_json(spec.controller_root / "controller_spec.json", controller_spec, replace=False)
    _write_text_new(spec.controller_root / POLICY_MARKER, POLICY_MARKER)
    _write_text_new(spec.controller_root / SUPERSEDED_MARKER, SUPERSEDED_MARKER)
    _write_text_new(
        spec.controller_root / NO_REDISTRIBUTION_MARKER,
        NO_REDISTRIBUTION_MARKER,
    )
    stages_root = spec.controller_root / "stages"
    stages_root.mkdir(mode=0o700, exist_ok=False)
    data_evidence = authority.evidence()
    stage_statuses = {
        STAGES[0]: "PASS",
        STAGES[1]: "PASS",
        STAGES[2]: "READY",
        STAGES[5]: "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT",
    }
    stage_evidence = {
        STAGES[0]: {"policy_adoption_sha256": stable_json_sha256(adoption)},
        STAGES[1]: data_evidence,
        STAGES[2]: {
            "dataset": "tastemolnet",
            "backbone": "gine",
            "num_classes": 3,
            "source_label": 1,
            "gpu_index": 1,
            "output_root": str(spec.gine_output_root),
        },
        STAGES[5]: {
            "gpu_index": 2,
            "science_started": False,
            "allowed_splits": ["train"],
            "initializer_data_split_used": "none",
            "taste_split_access_max": "train_only",
            "t5_release_enabled": False,
            "t5_release_state": "RELEASE_DISABLED_PENDING_FINAL_T3_T4_SOURCE_EXECUTION_RECEIPT",
            "test_loaded": False,
            "classifier_dependent": False,
        },
    }
    for stage in STAGES:
        _write_stage(
            stages_root / stage,
            _stage_payloads(
                stage=stage,
                controller_id=spec.controller_id,
                status=stage_statuses.get(stage, "WAITING_DEPENDENCY"),
                evidence=stage_evidence.get(
                    stage, {"implementation_status": "NOT_YET_RELEASED"}
                ),
            ),
        )
    queue = _queue(spec, t2_status="READY")
    state = {
        "schema_version": STATE_SCHEMA,
        "controller_id": spec.controller_id,
        "phase": "READY_FOR_MAIN_ROUTE",
        "current_stage": STAGES[2],
        "controller_pid": None,
        "created_at": _utc(),
        "updated_at": _utc(),
        "gine_controller_root": str(spec.gine_controller_root),
        "gine_output_root": str(spec.gine_output_root),
        "gine_training_state_root": str(spec.gine_training_state_root),
        "policy_adoption_path": str(spec.namespace_root / "policy_adoption.json"),
        "taste_completed_cells": 0,
        "taste_total_cells": 4,
        "main_matrix_mutated": False,
        "main_matrix_evidence_at_start": _matrix_evidence(),
        "gine_run_id": None,
        "gine_gpu_index": 1,
        "gine_gpu_uuid": None,
        "gpu2_stage": "READY_CLASSIFIER_INDEPENDENT_PRECOMPUTE",
        "gpu2_science_started": False,
        "boot_id": _boot_id(),
        "hostname": socket.gethostname(),
    }
    storage = {
        "schema_version": "tastemolnet_main_storage_reservation_evidence_v1",
        "controller_id": spec.controller_id,
        "reservation_scope": "controller_local_fail_closed_planning_evidence",
        "requested_gb": spec.reservation_gb,
        "free_gb_observed": free_gb,
        "minimum_free_after_reservations_gb": spec.minimum_free_after_reservations_gb,
        "free_after_reservation_gb": free_gb - spec.reservation_gb,
        "global_storage_manager_claimed": False,
        "status": "PASS",
    }
    _atomic_json(spec.controller_root / "queue.json", queue, replace=False)
    _atomic_json(spec.controller_root / "state.json", state, replace=False)
    _atomic_json(spec.controller_root / "storage_reservations.json", storage, replace=False)
    _append_jsonl(
        spec.controller_root / "runs.jsonl",
        {
            "event": "MAIN_CONTROLLER_PREPARED",
            "controller_id": spec.controller_id,
            "pid": os.getpid(),
            "stage": STAGES[2],
            "state": "READY",
            "at": _utc(),
        },
    )
    _append_jsonl(
        spec.controller_root / "status_updates.jsonl",
        {
            "event": "POLICY_V2_AND_DATA_READY",
            "controller_id": spec.controller_id,
            "old_state": SUPERSEDED_STATE,
            "current_route_state": CURRENT_ROUTE_STATE,
            "at": _utc(),
        },
    )
    return {
        "policy_adoption": adoption,
        "data_authority": data_evidence,
        "queue": queue,
        "state": state,
        "storage": storage,
    }


def _validate_controller_spec(spec: TasteMainSpec) -> None:
    if spec.controller_root.parent != spec.namespace_root:
        raise TasteMainControllerError(
            "controller root must remain one direct namespace child"
        )
    payload, _, _ = _read_json(
        spec.controller_root / "controller_spec.json", label="Taste main controller spec"
    )
    expected = {
        "schema_version": MAIN_SCHEMA,
        "controller_id": spec.controller_id,
        "controller_root": str(spec.controller_root),
        "controller_root_identity": _directory_identity(os.stat(spec.controller_root)),
        "project_root": str(spec.project_root),
        "policy_path": str(spec.policy_path),
        "prepared_root": str(spec.prepared_root),
        "graph_cache_root": str(spec.graph_cache_root),
        "gine_controller_root": str(spec.gine_controller_root),
        "gine_output_root": str(spec.gine_output_root),
        "gine_training_state_root": str(spec.gine_training_state_root),
        "gine_gpu_index": 1,
        "gpu0_gpu3_protected": True,
        "gnn_ablation_started": False,
    }
    for key, value in expected.items():
        if type(payload.get(key)) is not type(value) or payload.get(key) != value:
            raise TasteMainControllerError(f"Taste main controller spec changed: {key}")
    if (
        not isinstance(payload.get("created_at"), str)
        or not payload["created_at"]
    ):
        raise TasteMainControllerError("Taste main controller spec authority is malformed")


def _heartbeat_loop(root: Path, controller_id: str, stop: threading.Event) -> None:
    while not stop.is_set():
        payload = {
            "schema_version": HEARTBEAT_SCHEMA,
            "controller_id": controller_id,
            "pid": os.getpid(),
            "heartbeat_at": _utc(),
            "phase": "T2_GINE_FULL_RUNNING",
            "boot_id": _boot_id(),
            "hostname": socket.gethostname(),
        }
        _atomic_json(root / "heartbeat.json", payload, replace=(root / "heartbeat.json").exists())
        stop.wait(10.0)


def run_tastemolnet_main(spec: TasteMainSpec, *, resume: bool = False) -> int:
    _validate_runtime_environment()
    if resume:
        if not spec.controller_root.is_dir():
            raise TasteMainControllerError("resume requested without main controller root")
        adoption, _, _ = _read_json(
            spec.namespace_root / "policy_adoption.json", label="Taste policy adoption"
        )
        if (
            adoption.get("schema_version") != ADOPTION_SCHEMA
            or adoption.get("new_policy_path") != str(spec.policy_path)
            or adoption.get("old_state_superseded") is not True
            or adoption.get("old_science_adopted") is not False
        ):
            raise TasteMainControllerError("Taste policy adoption changed before resume")
        policy = load_tastemolnet_research_policy(spec.policy_path)
        policy.require_main_route()
        authority = validate_tastemolnet_local_authority(
            policy,
            prepared_root=spec.prepared_root,
            graph_cache_root=spec.graph_cache_root,
        )
        receipt = validate_tastemolnet_policy_receipt(
            spec.policy_receipt,
            policy=policy,
            authority=authority,
            require_active=True,
            require_policy_version=2,
        )
        old = _validate_old_block(
            source_manifest=spec.old_source_manifest,
            task_root=spec.old_task_root,
        )
        if (
            adoption.get("new_policy_sha256") != policy.file_sha256
            or adoption.get("new_policy_receipt_sha256") != receipt.sha256
            or adoption.get("old_manifest_sha256")
            != old["task_manifest"]["sha256"]
        ):
            raise TasteMainControllerError("Taste adoption authority changed before resume")
    else:
        prepare_tastemolnet_main(spec)

    try:
        main_lock_fd = _acquire_main_lock(spec.namespace_root)
    except BlockingIOError as exc:
        raise TasteMainControllerError("Taste main controller already owns its namespace") from exc

    _validate_controller_spec(spec)

    state_path = spec.controller_root / "state.json"
    state, _, _ = _read_json(state_path, label="Taste main state")
    state.update(
        {
            "phase": "RUNNING",
            "current_stage": STAGES[2],
            "controller_pid": os.getpid(),
            "updated_at": _utc(),
        }
    )
    _atomic_json(state_path, state, replace=True)
    queue = _queue(spec, t2_status="RUNNING")
    _atomic_json(spec.controller_root / "queue.json", queue, replace=True)
    _replace_stage(
        spec.controller_root / "stages" / STAGES[2],
        _stage_payloads(
            stage=STAGES[2],
            controller_id=spec.controller_id,
            status="RUNNING",
            evidence={
                "dataset": "tastemolnet",
                "backbone": "gine",
                "num_classes": 3,
                "source_label": 1,
                "gpu_index": 1,
                "output_root": str(spec.gine_output_root),
                "controller_pid": os.getpid(),
            },
        ),
    )
    _append_jsonl(
        spec.controller_root / "runs.jsonl",
        {
            "event": "TASTE_GINE_CONTROLLER_STARTING",
            "controller_id": spec.controller_id,
            "pid": os.getpid(),
            "stage": STAGES[2],
            "state": "RUNNING",
            "gpu_index": 1,
            "at": _utc(),
        },
    )
    _append_jsonl(
        spec.controller_root / "status_updates.jsonl",
        {
            "event": "T2_GINE_FULL_RUNNING",
            "controller_id": spec.controller_id,
            "pid": os.getpid(),
            "at": _utc(),
        },
    )

    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(spec.controller_root, spec.controller_id, stop),
        daemon=True,
    )
    heartbeat.start()
    try:
        wrapper = spec.project_root / "scripts/autodl/run_tastemolnet_gnn_full.sh"
        gine_spec = TasteGINEControllerSpec.build(
            cid=os.environ["TASTEMOLNET_GINE_CONTROLLER_CID"],
            controller_root=spec.gine_controller_root,
            project_root=spec.project_root,
            output_dir=spec.gine_output_root,
            training_state_root=spec.gine_training_state_root,
            worker_argv=("bash", str(wrapper)),
            poll_seconds=float(os.environ.get("TASTEMOLNET_GINE_CONTROLLER_POLL_SECONDS", "30")),
            terminal_stability_seconds=float(
                os.environ.get("TASTEMOLNET_GINE_TERMINAL_STABILITY_SECONDS", "2")
            ),
        )
        rc = run_tastemolnet_gine_controller(
            gine_spec,
            resume=resume and spec.gine_controller_root.exists(),
        )
        state["phase"] = "T2_GINE_FULL_PASS" if rc == 0 else "FAILED"
        state["current_stage"] = STAGES[3] if rc == 0 else STAGES[2]
        state["updated_at"] = _utc()
        state["gine_controller_status"] = inspect_tastemolnet_gine_controller(
            spec.gine_controller_root
        )
        _atomic_json(state_path, state, replace=True)
        _atomic_json(
            spec.controller_root / "queue.json",
            _queue(spec, t2_status="PASS" if rc == 0 else "FAILED"),
            replace=True,
        )
        _replace_stage(
            spec.controller_root / "stages" / STAGES[2],
            _stage_payloads(
                stage=STAGES[2],
                controller_id=spec.controller_id,
                status="PASS" if rc == 0 else "FAILED",
                evidence={
                    "dataset": "tastemolnet",
                    "backbone": "gine",
                    "num_classes": 3,
                    "source_label": 1,
                    "gpu_index": 1,
                    "output_root": str(spec.gine_output_root),
                    "exit_code": int(rc),
                },
            ),
        )
        _append_jsonl(
            spec.controller_root / "runs.jsonl",
            {
                "event": "TASTE_GINE_CONTROLLER_TERMINAL",
                "controller_id": spec.controller_id,
                "pid": os.getpid(),
                "stage": STAGES[2],
                "state": "PASS" if rc == 0 else "FAILED",
                "exit_code": int(rc),
                "at": _utc(),
            },
        )
        return int(rc)
    except Exception as exc:
        state["phase"] = "FAILED"
        state["current_stage"] = STAGES[2]
        state["failure"] = f"{type(exc).__name__}: {exc}"
        state["updated_at"] = _utc()
        _atomic_json(state_path, state, replace=True)
        _atomic_json(
            spec.controller_root / "queue.json",
            _queue(spec, t2_status="FAILED"),
            replace=True,
        )
        _replace_stage(
            spec.controller_root / "stages" / STAGES[2],
            _stage_payloads(
                stage=STAGES[2],
                controller_id=spec.controller_id,
                status="FAILED",
                evidence={
                    "dataset": "tastemolnet",
                    "backbone": "gine",
                    "num_classes": 3,
                    "source_label": 1,
                    "gpu_index": 1,
                    "output_root": str(spec.gine_output_root),
                    "error_type": type(exc).__name__,
                },
            ),
        )
        _append_jsonl(
            spec.controller_root / "status_updates.jsonl",
            {
                "event": "T2_GINE_FULL_FAILED",
                "controller_id": spec.controller_id,
                "pid": os.getpid(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "at": _utc(),
            },
        )
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        os.close(main_lock_fd)


def inspect_tastemolnet_main(controller_root: str | Path) -> dict[str, Any]:
    root = _absolute(controller_root, must_exist=True)
    state, state_data, _ = _read_json(root / "state.json", label="Taste main state")
    queue, queue_data, _ = _read_json(root / "queue.json", label="Taste main queue")
    heartbeat_path = root / "heartbeat.json"
    heartbeat = None
    if heartbeat_path.exists():
        heartbeat, _, _ = _read_json(heartbeat_path, label="Taste main heartbeat")
    gine = None
    gine_root = Path(str(state.get("gine_controller_root") or ""))
    if gine_root.is_absolute() and gine_root.exists():
        gine = inspect_tastemolnet_gine_controller(gine_root)
    return {
        "schema_version": MAIN_SCHEMA,
        "controller_root": str(root),
        "state": state,
        "state_sha256": _sha256(state_data),
        "queue": queue,
        "queue_sha256": _sha256(queue_data),
        "heartbeat": heartbeat,
        "gine_controller": gine,
        "markers": {
            name: (root / name).is_file()
            for name in (POLICY_MARKER, SUPERSEDED_MARKER, NO_REDISTRIBUTION_MARKER)
        },
    }


__all__ = [
    "ADOPTION_SCHEMA",
    "CURRENT_ROUTE_STATE",
    "NAMESPACE_NAME",
    "STAGES",
    "SUPERSEDED_STATE",
    "SUPERSEDED_MARKER",
    "TasteMainControllerError",
    "TasteMainSpec",
    "inspect_tastemolnet_main",
    "prepare_tastemolnet_main",
    "run_tastemolnet_main",
]
