#!/usr/bin/env python3
"""Continue a frozen COMRECGC generation into one standardized paper cell.

The completed generation is adopted read-only.  Every downstream stage writes
below a fresh output root and the PASS marker is published last.  This entry
point intentionally does not regenerate random walks or modify the recovery
root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import stat as stat_module
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_comrecgc_checkout import verify_checkout  # noqa: E402
from src.baselines.comrecgc.contracts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    METHOD,
    UPSTREAM_COMMIT,
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
    write_json,
)
from src.baselines.comrecgc.external_memory_dbscan import (  # noqa: E402
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ALL_CORE_ONE_COMPONENT_SHORTCUT,
    SKLEARN_FLOAT64_EXACT_MULTI_COMPONENT,
    _validate_component_recovery_closure,
    _validate_shortcut_proof_closure,
)
from src.baselines.comrecgc.external_component_summary import (  # noqa: E402
    validate_proven_all_core_component_summary,
)
from src.baselines.comrecgc.external_memory_recourse import (  # noqa: E402
    PAIR_STORE_SCHEMA,
    validate_adopted_pair_store_read_only,
    validate_proven_one_cluster_summary,
)
from src.baselines.comrecgc.external_pair_chunk_cache import (  # noqa: E402
    validate_cartesian_chunk_vector_cache,
)
from src.baselines.comrecgc.close_pair_view import (  # noqa: E402
    validate_theta_close_pair_view,
)
from src.utils.autodl_exec_startup_barrier import (  # noqa: E402
    ArmedExecStartupBarrier,
    StartupBarrierRecord,
    arm_exec_startup_barrier,
    reconcile_interrupted_startup_barrier_publication,
    validate_reopenable_unreleased_barrier,
    validate_startup_barrier_record,
)


DATASET_CONTRACTS: dict[str, dict[str, int]] = {
    "aids": {"generation_parent_limit": 1283, "evaluation_parent_count": 1283},
    "mutagenicity": {
        "generation_parent_limit": 1448,
        "evaluation_parent_count": 217,
    },
}

_PROC_ROOT = Path("/proc")
_CRITICAL_SOURCE_MANIFESTS = (
    "run_manifest.json",
    "_RUN_COMPLETE.json",
    "freeze_only_recovery.json",
    "frozen_payload_closure_audit.json",
    "adoption_manifest.json",
)
_STARTUP_BARRIER_BINDING_SCHEMA = "comrecgc_continuation_exec_startup_binding_v1"
_STARTUP_BARRIER_MAX_GENERATIONS = 32
_PARTIAL_STAGE_ARCHIVE_MAX_BYTES = 1024**3


class _StageTerminationRequested(RuntimeError):
    """Stop the outer continuation after forwarding an operator termination."""


@dataclass(frozen=True)
class ContinuationInputs:
    dataset: str
    source_generation_root: Path
    upstream_root: Path
    dataset_dir: Path
    source_csv: Path | None
    distance_checkpoint: Path
    dataset_csv: Path
    teacher_path: Path
    molclr_root: Path
    molclr_checkpoint: Path
    thresholds_path: Path
    output_root: Path
    device: str
    theta_star: float | None
    cost_cap: float | None
    common_recourse_engine: str = "legacy_in_memory"
    external_max_rss_gb: float = 96.0
    external_query_block_size: int = 8
    external_checkpoint_interval_blocks: int = 1
    external_dbscan_shortcut_mode: str = "disabled"
    external_shortcut_seed_count: int = 3
    external_shortcut_failure_cap: int = 4_096
    external_shortcut_query_block_size: int = 65_536
    external_exact_fallback_max_samples: int = 100_000
    external_summary_block_size: int = 65_536
    external_pair_store_source_manifest: Path | None = None
    external_pair_store_source_checkpoint: Path | None = None
    external_pair_store_source_owner_root: Path | None = None
    external_close_pair_view_manifest: Path | None = None
    external_vector_cache_root: Path | None = None
    external_vector_cache_lock: Path | None = None
    external_vector_cache_route_lock: Path | None = None
    external_vector_cache_min_free_gb: float = 3.0
    external_vector_cache_proc_root: Path = Path("/proc")
    expected_sklearn_version: str = "1.7.2"
    common_recourse_resume: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(resolved)
    return resolved


def _require_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _git_head(project_root: Path = PROJECT_ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
        timeout=30,
    ).strip()


def _stat_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _snapshot_frozen_file(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    logical = Path(os.path.abspath(os.fspath(path.expanduser())))
    logical_stat = logical.lstat()
    resolved = logical.resolve(strict=True)
    resolved_stat = resolved.stat()
    if not stat_module.S_ISREG(resolved_stat.st_mode) or resolved_stat.st_size <= 0:
        raise ValueError(f"Frozen source file is not a nonempty regular file: {logical}")
    snapshot: dict[str, Any] = {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "logical_lstat": _stat_identity(logical_stat),
        "resolved_stat": _stat_identity(resolved_stat),
    }
    if include_sha256:
        snapshot["sha256"] = sha256_file(resolved)
    return snapshot


def _snapshot_critical_manifests(source: Path) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for name in _CRITICAL_SOURCE_MANIFESTS:
        logical = source / name
        resolved = _require_file(logical)
        try:
            resolved.relative_to(source)
        except ValueError as exc:
            raise ValueError(
                f"Frozen closure manifest resolves outside source root: {logical}"
            ) from exc
        snapshots[name] = _snapshot_frozen_file(logical, include_sha256=True)
    return snapshots


def _assert_snapshots_equal(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if dict(expected) == dict(observed):
        return
    changed = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    raise ValueError(f"SOURCE_CLOSURE_CHANGED:{label}:changed={changed}")


def _scan_live_source_writers(
    source: Path,
    *,
    protected_snapshots: Sequence[Mapping[str, Any]],
    proc_root: Path = _PROC_ROOT,
) -> dict[str, Any]:
    """Fail closed when procfs exposes a writable FD for the frozen source."""

    proc = proc_root.expanduser()
    if not proc.is_dir():
        raise ValueError(
            "LIVE_WRITER_AUDIT_UNAVAILABLE: Linux procfs is required for adoption"
        )
    protected_inodes = {
        (
            int(snapshot["resolved_stat"]["device"]),
            int(snapshot["resolved_stat"]["inode"]),
        )
        for snapshot in protected_snapshots
    }
    writers: list[dict[str, Any]] = []
    scanned_processes = 0
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        fd_dir = pid_dir / "fd"
        try:
            descriptors = list(fd_dir.iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError(
                f"LIVE_WRITER_AUDIT_UNAVAILABLE: cannot inspect pid={pid_dir.name}"
            ) from exc
        scanned_processes += 1
        for descriptor in descriptors:
            if not descriptor.name.isdigit():
                continue
            try:
                descriptor_stat = descriptor.stat()
                target = descriptor.resolve(strict=True)
            except FileNotFoundError:
                continue
            except PermissionError as exc:
                raise ValueError(
                    "LIVE_WRITER_AUDIT_UNAVAILABLE: "
                    f"cannot inspect pid={pid_dir.name} fd={descriptor.name}"
                ) from exc
            inode = (int(descriptor_stat.st_dev), int(descriptor_stat.st_ino))
            try:
                target.relative_to(source)
                inside_source = True
            except ValueError:
                inside_source = False
            if not inside_source and inode not in protected_inodes:
                continue
            fdinfo = pid_dir / "fdinfo" / descriptor.name
            try:
                lines = fdinfo.read_text(
                    encoding="utf-8", errors="strict"
                ).splitlines()
            except FileNotFoundError:
                continue
            except (PermissionError, UnicodeError, OSError) as exc:
                raise ValueError(
                    "LIVE_WRITER_AUDIT_UNAVAILABLE: "
                    f"cannot read pid={pid_dir.name} fd={descriptor.name} flags"
                ) from exc
            flags_text = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in lines
                    if line.startswith("flags:")
                ),
                None,
            )
            try:
                flags = int(flags_text, 8) if flags_text is not None else None
            except ValueError as exc:
                raise ValueError(
                    "LIVE_WRITER_AUDIT_UNAVAILABLE: "
                    f"invalid pid={pid_dir.name} fd={descriptor.name} flags"
                ) from exc
            if flags is None:
                raise ValueError(
                    "LIVE_WRITER_AUDIT_UNAVAILABLE: "
                    f"missing pid={pid_dir.name} fd={descriptor.name} flags"
                )
            if (flags & os.O_ACCMODE) in {os.O_WRONLY, os.O_RDWR}:
                writers.append(
                    {
                        "pid": int(pid_dir.name),
                        "fd": int(descriptor.name),
                        "path": str(target),
                        "flags_octal": flags_text,
                    }
                )
    if writers:
        raise ValueError(f"LIVE_WRITER_DETECTED:{writers[:8]}")
    return {
        "procfs_verified": True,
        "proc_root": str(proc.resolve(strict=True)),
        "scanned_process_count": scanned_processes,
        "writable_fd_count": 0,
        "writers": [],
    }


def _verify_adopted_generation_integrity(adoption: Mapping[str, Any]) -> dict[str, Any]:
    integrity = adoption.get("source_integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("Missing source_integrity evidence from adoption entry gate")
    source = _require_directory(Path(str(adoption["source_generation_root"])))
    expected_manifests = integrity.get("critical_manifests_after_payload_hash")
    expected_payload = integrity.get("payload_after_sha256")
    if not isinstance(expected_manifests, Mapping) or not isinstance(
        expected_payload, Mapping
    ):
        raise ValueError("Incomplete source_integrity evidence from adoption entry gate")
    payload_path = Path(str(adoption["counterfactuals_path"]))
    payload_now = _snapshot_frozen_file(payload_path, include_sha256=False)
    protected = [*expected_manifests.values(), expected_payload]
    writer_before = _scan_live_source_writers(
        source,
        protected_snapshots=protected,
        proc_root=_PROC_ROOT,
    )
    manifests_now = _snapshot_critical_manifests(source)
    writer_after = _scan_live_source_writers(
        source,
        protected_snapshots=[*manifests_now.values(), payload_now],
        proc_root=_PROC_ROOT,
    )
    _assert_snapshots_equal(
        expected_manifests,
        manifests_now,
        label="critical_manifests_after_continuation",
    )
    _assert_snapshots_equal(
        expected_payload,
        payload_now,
        label="payload_stat_after_continuation",
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "payload_sha256_recomputed": False,
        "payload_stat_unchanged": True,
        "critical_manifest_stat_and_hash_unchanged": True,
        "critical_manifests": manifests_now,
        "payload": payload_now,
        "live_writer_audit_before_snapshot": writer_before,
        "live_writer_audit_after_snapshot": writer_after,
        "verified_at": _utc_now(),
    }


def validate_adopted_generation(inputs: ContinuationInputs) -> dict[str, Any]:
    """Validate the frozen closure and hash its large payload exactly once."""

    if inputs.dataset not in DATASET_CONTRACTS:
        raise ValueError(f"Unsupported dataset: {inputs.dataset}")
    source = _require_directory(inputs.source_generation_root)
    contract = DATASET_CONTRACTS[inputs.dataset]
    closure_before = _snapshot_critical_manifests(source)
    manifest_path = _require_file(source / "run_manifest.json")
    complete_path = _require_file(source / "_RUN_COMPLETE.json")
    recovery_path = _require_file(source / "freeze_only_recovery.json")
    closure_path = _require_file(source / "frozen_payload_closure_audit.json")
    original_adoption_path = _require_file(source / "adoption_manifest.json")
    manifest = _load_object(manifest_path)
    complete = _load_object(complete_path)
    recovery = _load_object(recovery_path)
    closure = _load_object(closure_path)
    original_adoption = _load_object(original_adoption_path)

    failures: list[str] = []
    expected = {
        "dataset": inputs.dataset,
        "mode": "full",
        "parent_limit": contract["generation_parent_limit"],
        "run_complete": True,
        "freeze_only_recovery": True,
        "algorithm_rerun": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "generation_mode": "adopted_read_only_cache",
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            failures.append(
                f"run_manifest.{field}:actual={manifest.get(field)!r}:"
                f"expected={expected_value!r}"
            )
    if complete.get("run_complete") is not True:
        failures.append("_RUN_COMPLETE.run_complete")
    if complete.get("freeze_only_recovery") is not True:
        failures.append("_RUN_COMPLETE.freeze_only_recovery")
    if recovery.get("recovery_completed") is not True:
        failures.append("freeze_only_recovery.recovery_completed")
    if int(recovery.get("completed_steps", -1)) != 50_000:
        failures.append("freeze_only_recovery.completed_steps")
    if recovery.get("algorithm_rerun") is not False:
        failures.append("freeze_only_recovery.algorithm_rerun")
    if closure.get("closure_complete") is not True:
        failures.append("frozen_payload_closure.closure_complete")
    if closure.get("post_write_reload_verified") is not True:
        failures.append("frozen_payload_closure.post_write_reload_verified")
    if original_adoption.get("generation_mode") != "adopted_read_only_cache":
        failures.append("adoption_manifest.generation_mode")

    payload = Path(str(manifest.get("counterfactuals_path") or "")).expanduser()
    try:
        payload = payload.resolve(strict=True)
        payload.relative_to(source)
    except (FileNotFoundError, ValueError):
        failures.append("counterfactuals_path_not_inside_frozen_source")
    claimed_payload_sha = str(manifest.get("counterfactuals_sha256") or "")
    if len(claimed_payload_sha) != 64:
        failures.append("counterfactuals_sha256")
    if complete.get("counterfactuals_sha256") != claimed_payload_sha:
        failures.append("counterfactuals_sha256_gate_disagreement")
    if recovery.get("counterfactuals_sha256") != claimed_payload_sha:
        failures.append("counterfactuals_sha256_recovery_disagreement")
    candidate_count = int(manifest.get("counterfactual_candidate_count", -1))
    if candidate_count <= 0:
        failures.append("counterfactual_candidate_count")
    if failures:
        raise ValueError("Frozen generation adoption failed: " + "; ".join(failures))

    payload_before = _snapshot_frozen_file(payload, include_sha256=False)
    protected_before = [*closure_before.values(), payload_before]
    writer_before = _scan_live_source_writers(
        source,
        protected_snapshots=protected_before,
        proc_root=_PROC_ROOT,
    )
    actual_payload_sha = sha256_file(payload)
    payload_after = _snapshot_frozen_file(payload, include_sha256=False)
    closure_after = _snapshot_critical_manifests(source)
    writer_after = _scan_live_source_writers(
        source,
        protected_snapshots=[*closure_after.values(), payload_after],
        proc_root=_PROC_ROOT,
    )
    _assert_snapshots_equal(
        closure_before,
        closure_after,
        label="critical_manifests_around_payload_hash",
    )
    _assert_snapshots_equal(
        payload_before,
        payload_after,
        label="payload_stat_around_sha256",
    )
    if actual_payload_sha != claimed_payload_sha:
        raise ValueError(
            "COUNTERFACTUALS_PAYLOAD_SHA256_MISMATCH:"
            f"actual={actual_payload_sha}:claimed={claimed_payload_sha}"
        )

    return {
        "schema_version": 1,
        "status": "PASS",
        "dataset": inputs.dataset,
        "generation_adopted": True,
        "generation_mode": "adopted_read_only_cache",
        "generation_rerun": False,
        "source_generation_root": str(source),
        "source_run_manifest_sha256": closure_after["run_manifest.json"]["sha256"],
        "source_complete_sha256": closure_after["_RUN_COMPLETE.json"]["sha256"],
        "source_recovery_sha256": closure_after["freeze_only_recovery.json"][
            "sha256"
        ],
        "source_closure_sha256": closure_after[
            "frozen_payload_closure_audit.json"
        ]["sha256"],
        "source_adoption_manifest_sha256": closure_after["adoption_manifest.json"][
            "sha256"
        ],
        "counterfactuals_path": str(payload),
        "counterfactuals_sha256_claimed": claimed_payload_sha,
        "counterfactuals_sha256_actual": actual_payload_sha,
        "counterfactuals_sha256_verified": True,
        "counterfactuals_sha256_computation_count": 1,
        "counterfactual_candidate_count": candidate_count,
        "source_project_commit": manifest.get("project_commit"),
        "upstream_commit": manifest.get("upstream_commit"),
        "serialization_rerun": False,
        "lineage_resolution_rerun": False,
        "downstream_common_recourse_rerun": True,
        "downstream_chemistry_rerun": True,
        "downstream_unified_evaluation_rerun": True,
        "source_checksums": original_adoption.get("source_checksums"),
        "source_integrity": {
            "schema_version": 1,
            "critical_manifests_before_payload_hash": closure_before,
            "critical_manifests_after_payload_hash": closure_after,
            "payload_before_sha256": payload_before,
            "payload_after_sha256": payload_after,
            "live_writer_audit_before_payload_hash": writer_before,
            "live_writer_audit_after_payload_hash": writer_after,
        },
        "validated_at": _utc_now(),
    }


def build_stage_commands(
    inputs: ContinuationInputs,
    *,
    project_commit: str,
    candidate_count: int,
    teacher_sha256: str,
    execution_project_root: Path = PROJECT_ROOT,
) -> list[tuple[str, list[str], Path, str]]:
    """Return ordered stage commands and their required completion markers."""

    execution_project_root = execution_project_root.expanduser().resolve(strict=True)
    contract = DATASET_CONTRACTS[inputs.dataset]
    python = sys.executable
    source_args: list[str] = []
    if inputs.source_csv is not None:
        source_args = ["--source-csv", str(inputs.source_csv)]
    common = inputs.output_root / "common_recourse"
    chemistry = inputs.output_root / "chemistry"
    evaluation = inputs.output_root / "unified_eval"
    gate = inputs.output_root / "full_gate"
    standardized = inputs.output_root / "standardized"
    trace = inputs.source_generation_root / "trace"
    counterfactuals_sha = _load_object(
        inputs.source_generation_root / "run_manifest.json"
    )["counterfactuals_sha256"]

    common_argv = [
        python,
        str(
            execution_project_root
            / "scripts/baselines/comrecgc/run_common_recourse.py"
        ),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        inputs.dataset,
        "--mode",
        "full",
        "--upstream-root",
        str(inputs.upstream_root),
        "--dataset-dir",
        str(inputs.dataset_dir),
        *source_args,
        "--generation-dir",
        str(inputs.source_generation_root),
        "--distance-checkpoint",
        str(inputs.distance_checkpoint),
        "--output-dir",
        str(common),
        "--parent-limit",
        str(contract["generation_parent_limit"]),
        "--device",
        inputs.device,
        "--engine",
        inputs.common_recourse_engine,
    ]
    if inputs.common_recourse_engine == "external_memory_exact_v1":
        common_argv.extend(
            [
                "--external-max-rss-gb",
                format(float(inputs.external_max_rss_gb), ".17g"),
                "--external-query-block-size",
                str(int(inputs.external_query_block_size)),
                "--external-checkpoint-interval-blocks",
                str(int(inputs.external_checkpoint_interval_blocks)),
                "--external-dbscan-shortcut-mode",
                inputs.external_dbscan_shortcut_mode,
                "--external-shortcut-seed-count",
                str(int(inputs.external_shortcut_seed_count)),
                "--external-shortcut-failure-cap",
                str(int(inputs.external_shortcut_failure_cap)),
                "--external-shortcut-query-block-size",
                str(int(inputs.external_shortcut_query_block_size)),
                "--external-exact-fallback-max-samples",
                str(int(inputs.external_exact_fallback_max_samples)),
                "--external-summary-block-size",
                str(int(inputs.external_summary_block_size)),
                "--expected-sklearn-version",
                inputs.expected_sklearn_version,
            ]
        )
        if inputs.external_pair_store_source_manifest is not None:
            common_argv.extend(
                [
                    "--external-pair-store-source-manifest",
                    str(inputs.external_pair_store_source_manifest),
                    "--external-pair-store-source-owner-root",
                    str(inputs.external_pair_store_source_owner_root),
                ]
            )
            if inputs.external_close_pair_view_manifest is not None:
                common_argv.extend(
                    [
                        "--external-close-pair-view-manifest",
                        str(inputs.external_close_pair_view_manifest),
                    ]
                )
        if inputs.external_pair_store_source_checkpoint is not None:
            common_argv.extend(
                [
                    "--external-pair-store-source-checkpoint",
                    str(inputs.external_pair_store_source_checkpoint),
                    "--external-pair-store-source-owner-root",
                    str(inputs.external_pair_store_source_owner_root),
                    "--external-close-pair-view-manifest",
                    str(inputs.external_close_pair_view_manifest),
                    "--external-vector-cache-root",
                    str(inputs.external_vector_cache_root),
                    "--external-vector-cache-lock",
                    str(inputs.external_vector_cache_lock),
                    "--external-vector-cache-route-lock",
                    str(inputs.external_vector_cache_route_lock),
                    "--external-vector-cache-min-free-gb",
                    format(float(inputs.external_vector_cache_min_free_gb), ".17g"),
                    "--external-vector-cache-proc-root",
                    str(inputs.external_vector_cache_proc_root),
                ]
            )
    if inputs.common_recourse_resume:
        common_argv.append("--resume")
    chemistry_argv = [
        python,
        str(
            execution_project_root
            / "scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"
        ),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(execution_project_root),
        "--dataset",
        inputs.dataset,
        "--dataset-dir",
        str(inputs.dataset_dir),
        *source_args,
        "--generation-dir",
        str(inputs.source_generation_root),
        "--trace-lineage-path",
        str(trace / "candidate_action_lineage.json"),
        "--trace-evidence-path",
        str(trace / "trace_summary.json"),
        "--common-recourse-dir",
        str(common),
        "--output-dir",
        str(chemistry),
        "--preregistration-path",
        str(inputs.output_root / "preregistration/deterministic_chem_repair.json"),
        "--parent-limit",
        str(contract["generation_parent_limit"]),
        "--expected-candidate-count",
        str(candidate_count),
        "--expected-counterfactuals-sha256",
        str(counterfactuals_sha),
    ]
    evaluation_argv = [
        python,
        str(
            execution_project_root
            / "scripts/baselines/comrecgc/run_slot_unified_eval.py"
        ),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        inputs.dataset,
        "--mode",
        "full",
        "--chemistry-dir",
        str(chemistry),
        "--dataset-csv",
        str(inputs.dataset_csv),
        "--teacher-path",
        str(inputs.teacher_path),
        "--molclr-root",
        str(inputs.molclr_root),
        "--molclr-checkpoint",
        str(inputs.molclr_checkpoint),
        "--thresholds-json",
        str(inputs.thresholds_path),
        "--output-dir",
        str(evaluation),
        "--expected-parent-count",
        str(contract["evaluation_parent_count"]),
        "--max-k",
        "20",
        "--device",
        inputs.device,
    ]
    if inputs.theta_star is not None:
        evaluation_argv.extend(["--theta-star", format(inputs.theta_star, ".17g")])
    if inputs.cost_cap is not None:
        evaluation_argv.extend(["--cost-cap", format(inputs.cost_cap, ".17g")])
    gate_argv = [
        python,
        str(execution_project_root / "scripts/baselines/comrecgc/gate_recovery.py"),
        "--stage",
        "project-full",
        "--dataset",
        inputs.dataset,
        "--expected-parent-count",
        str(contract["evaluation_parent_count"]),
        "--expected-teacher-sha256",
        teacher_sha256,
        "--expected-project-commit",
        project_commit,
        "--input-dir",
        str(evaluation),
        "--output-dir",
        str(gate),
    ]
    freeze_argv = [
        python,
        str(
            execution_project_root
            / "scripts/baselines/comrecgc/freeze_recovery_result.py"
        ),
        "--dataset",
        inputs.dataset,
        "--source-dir",
        str(evaluation),
        "--gate-dir",
        str(gate),
        "--output-dir",
        str(standardized),
    ]
    return [
        ("common_recourse", common_argv, common / "_RUN_COMPLETE.json", "run_complete"),
        ("chemistry", chemistry_argv, chemistry / "_RUN_COMPLETE.json", "run_complete"),
        ("unified_eval", evaluation_argv, evaluation / "_RUN_COMPLETE.json", "run_complete"),
        ("full_gate", gate_argv, gate / "gate_result.json", "audit_passed"),
        ("freeze", freeze_argv, standardized / "_FINALIZED.json", "finalized"),
    ]


def _stage_startup_barrier_paths(
    output_root: Path, *, stage: str, generation: int
) -> tuple[Path, Path]:
    if (
        stage not in {"common_recourse", "chemistry", "unified_eval", "full_gate", "freeze"}
        or not 0 <= generation < _STARTUP_BARRIER_MAX_GENERATIONS
    ):
        raise ValueError("CONTINUATION_STARTUP_BARRIER_GENERATION_INVALID")
    checkpoint_root = output_root / "stage_checkpoints"
    checkpoint_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    return (
        checkpoint_root / f".{stage}.exec-startup.lock",
        checkpoint_root / f".{stage}.exec-startup.{generation:02d}.json",
    )


def _validate_stage_startup_binding(
    *,
    output_root: Path,
    stage: str,
    argv: Sequence[str],
    binding: Any,
    allowed_phases: set[str],
) -> StartupBarrierRecord | None:
    if not isinstance(binding, Mapping):
        raise ValueError(f"CONTINUATION_STARTUP_BARRIER_BINDING_MISSING:{stage}")
    phase = binding.get("phase")
    common = {
        "schema_version",
        "stage",
        "generation",
        "phase",
        "record_path",
        "lock_path",
        "target_argv_sha256",
    }
    armed = common | {"record_sha256", "launcher_argv_sha256"}
    expected_fields = common if phase == "PRE_ARM" else armed
    if (
        phase not in allowed_phases
        or set(binding) != expected_fields
        or binding.get("schema_version") != _STARTUP_BARRIER_BINDING_SCHEMA
        or binding.get("stage") != stage
        or isinstance(binding.get("generation"), bool)
        or not isinstance(binding.get("generation"), int)
        or binding.get("target_argv_sha256") != stable_json_sha256(list(argv))
    ):
        raise ValueError(f"CONTINUATION_STARTUP_BARRIER_BINDING_CHANGED:{stage}")
    lock_path, record_path = _stage_startup_barrier_paths(
        output_root, stage=stage, generation=int(binding["generation"])
    )
    if (
        binding.get("lock_path") != str(lock_path)
        or binding.get("record_path") != str(record_path)
    ):
        raise ValueError(f"CONTINUATION_STARTUP_BARRIER_PATH_CHANGED:{stage}")
    if phase == "PRE_ARM":
        try:
            reconcile_interrupted_startup_barrier_publication(
                lock_path=lock_path,
                record_path=record_path,
                timeout_seconds=30.0,
            )
        except Exception as exc:
            raise ValueError(
                f"CONTINUATION_STARTUP_BARRIER_PREARM_RECONCILIATION_FAILED:{stage}"
            ) from exc
    if phase == "PRE_ARM" and not record_path.exists():
        return None
    try:
        record = validate_startup_barrier_record(
            record_path,
            expected_target_argv=list(argv),
            validate_lock_path=True,
        )
    except Exception as exc:
        raise ValueError(
            f"CONTINUATION_STARTUP_BARRIER_RECORD_INVALID:{stage}"
        ) from exc
    if (
        record.lock_path != str(lock_path)
        or (phase != "PRE_ARM" and binding.get("record_sha256") != sha256_file(record_path))
        or (
            phase != "PRE_ARM"
            and binding.get("launcher_argv_sha256")
            != stable_json_sha256(record.launcher_argv)
        )
    ):
        raise ValueError(f"CONTINUATION_STARTUP_BARRIER_RECORD_CHANGED:{stage}")
    return record


def _next_stage_startup_generation(
    *,
    output_root: Path,
    stage: str,
    argv: Sequence[str],
    checkpoint_path: Path | None,
) -> int:
    state_path = checkpoint_path or (output_root / "stage_state.json")
    if not state_path.exists():
        return 0
    checkpoint = _load_object(_require_file(state_path))
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("stage") != stage
        or checkpoint.get("argv_sha256") != stable_json_sha256(list(argv))
        or checkpoint.get("status") not in {"RUNNING", "FAILED"}
    ):
        raise ValueError(f"CONTINUATION_PREVIOUS_STAGE_STATE_INVALID:{stage}")
    binding = checkpoint.get("startup_barrier")
    if binding is None:
        # Compatibility is limited to the already-safe private-session v1
        # checkpoints. They still need a fully quiescent process group.
        if checkpoint.get("process_group_contract") != "dedicated_child_session_v1":
            raise ValueError(f"CONTINUATION_STARTUP_BARRIER_BINDING_MISSING:{stage}")
        process_group_id = int(checkpoint.get("process_group_id") or -1)
        _wait_for_process_group_quiescence(
            process_group_id, proc_root=_PROC_ROOT, timeout_seconds=30.0
        )
        return 0
    record = _validate_stage_startup_binding(
        output_root=output_root,
        stage=stage,
        argv=argv,
        binding=binding,
        allowed_phases={"PRE_ARM", "ARMED", "BOUND", "QUIESCENT"},
    )
    phase = binding["phase"]
    if phase == "BOUND":
        process_group_id = int(checkpoint.get("process_group_id") or -1)
        _wait_for_process_group_quiescence(
            process_group_id, proc_root=_PROC_ROOT, timeout_seconds=30.0
        )
    elif record is not None:
        validate_reopenable_unreleased_barrier(
            record.record_path,
            expected_target_argv=list(argv),
            timeout_seconds=30.0,
        )
    generation = int(binding["generation"]) + 1
    if generation >= _STARTUP_BARRIER_MAX_GENERATIONS:
        raise ValueError(f"CONTINUATION_STARTUP_BARRIER_BUDGET_EXHAUSTED:{stage}")
    return generation


def _run_stage(
    *,
    stage: str,
    argv: Sequence[str],
    marker: Path,
    required_field: str,
    environment: Mapping[str, str],
    output_root: Path,
    checkpoint_path: Path | None = None,
) -> None:
    argv_sha256 = stable_json_sha256(list(argv))
    generation = _next_stage_startup_generation(
        output_root=output_root,
        stage=stage,
        argv=argv,
        checkpoint_path=checkpoint_path,
    )
    lock_path, record_path = _stage_startup_barrier_paths(
        output_root, stage=stage, generation=generation
    )
    startup_binding: dict[str, Any] = {
        "schema_version": _STARTUP_BARRIER_BINDING_SCHEMA,
        "stage": stage,
        "generation": generation,
        "phase": "PRE_ARM",
        "record_path": str(record_path),
        "lock_path": str(lock_path),
        "target_argv_sha256": argv_sha256,
    }
    running = {
        "schema_version": 2,
        "status": "RUNNING",
        "stage": stage,
        "runner_pid": os.getpid(),
        "child_pid": None,
        "child_start_ticks": None,
        "process_group_id": None,
        "process_group_contract": "durable_barrier_dedicated_child_session_v2",
        "startup_barrier": startup_binding,
        "argv_sha256": argv_sha256,
        "marker": str(marker),
        "required_field": required_field,
        "started_at": _utc_now(),
    }
    write_json(output_root / "stage_state.json", running)
    if checkpoint_path is not None:
        write_json(checkpoint_path, running)
    process: subprocess.Popen[bytes] | None = None
    barrier: ArmedExecStartupBarrier | None = None
    science_released = False
    pending_signals: list[int] = []
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        pending_signals.append(signum)
        # Before the durable BOUND checkpoint has been fsynced and the parent
        # deliberately releases the startup barrier, the child is only the
        # inert launcher.  Closing the capability pipe here makes a stop
        # request fail closed even when it arrives between launch and bind.
        if barrier is not None and not science_released:
            barrier.abort()
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def reject_pending_stop(*, phase: str) -> None:
        if not pending_signals:
            return
        if barrier is not None and not science_released:
            barrier.abort()
        raise _StageTerminationRequested(
            "STAGE_TERMINATED_BEFORE_SCIENCE_RELEASE:"
            f"stage={stage}:phase={phase}:signals={pending_signals}"
        )

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
        barrier = arm_exec_startup_barrier(
            lock_path=lock_path,
            record_path=record_path,
            target_argv=list(argv),
            python_executable=sys.executable,
            record_policy="fresh",
        )
        reject_pending_stop(phase="AFTER_ARM")
        startup_binding = {
            **startup_binding,
            "phase": "ARMED",
            "record_sha256": sha256_file(record_path),
            "launcher_argv_sha256": stable_json_sha256(barrier.launcher_argv),
        }
        running = {**running, "startup_barrier": startup_binding}
        write_json(output_root / "stage_state.json", running)
        if checkpoint_path is not None:
            write_json(checkpoint_path, running)
        reject_pending_stop(phase="BEFORE_LAUNCH")
        process = barrier.launch(
            cwd=PROJECT_ROOT,
            env=dict(environment),
            start_new_session=True,
        )
        reject_pending_stop(phase="AFTER_LAUNCH")
        child_start_ticks = _read_proc_start_ticks(process.pid, proc_root=_PROC_ROOT)
        observed_launcher = _proc_argv(process.pid, proc_root=_PROC_ROOT)
        if (
            child_start_ticks is None
            or observed_launcher is None
            or stable_json_sha256(observed_launcher)
            != startup_binding["launcher_argv_sha256"]
        ):
            raise ValueError(f"STAGE_STARTUP_WRAPPER_IDENTITY_UNBOUND:{stage}")
        startup_binding = {**startup_binding, "phase": "BOUND"}
        running = {
            **running,
            "child_pid": process.pid,
            "child_start_ticks": child_start_ticks,
            "process_group_id": process.pid,
            "child_started_at": _utc_now(),
            "startup_barrier": startup_binding,
        }
        write_json(output_root / "stage_state.json", running)
        if checkpoint_path is not None:
            write_json(checkpoint_path, running)
        reject_pending_stop(phase="BEFORE_RELEASE")
        barrier.release()
        science_released = True
        for signum in pending_signals:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        return_code = process.wait()
        _wait_for_process_group_quiescence(
            process.pid, proc_root=_PROC_ROOT, timeout_seconds=None
        )
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(argv))
        payload = _load_object(_require_file(marker))
        if payload.get(required_field) is not True:
            raise ValueError(
                f"Stage {stage} completion field {required_field!r} is not true: {marker}"
            )
        if stage == "common_recourse":
            try:
                engine_index = list(argv).index("--engine")
                expected_engine = str(argv[engine_index + 1])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    "COMMON_RECOURSE_ENGINE_MISSING_FROM_STAGE_ARGV"
                ) from exc
            if expected_engine == "external_memory_exact_v1":
                _validate_common_recourse_completion(marker=marker, terminal=payload)
        if pending_signals:
            # A child may checkpoint, finish its marker, and exit zero after a
            # forwarded SIGTERM.  That completion is recoverable on the next
            # explicit resume, but the current wrapper must not swallow the
            # stop request and proceed to a later scientific stage.
            raise _StageTerminationRequested(
                "STAGE_TERMINATED_AFTER_CHILD_COMPLETION:"
                f"stage={stage}:signals={pending_signals}"
            )
    except Exception as exc:
        if barrier is not None and not science_released:
            barrier.abort()
        if process is not None and process.poll() is None:
            if science_released:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Retain wrapper ownership and the invocation lock until every
                # non-zombie member is gone; never start a concurrent retry.
                pass
            _wait_for_process_group_quiescence(
                process.pid, proc_root=_PROC_ROOT, timeout_seconds=None
            )
        failed = {
            **running,
            "status": "FAILED",
            "error_class": type(exc).__name__,
            "message": str(exc),
            "failed_at": _utc_now(),
        }
        write_json(output_root / "stage_state.json", failed)
        if checkpoint_path is not None:
            write_json(checkpoint_path, failed)
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    passed = {
        **running,
        "status": "PASS",
        "marker_sha256": sha256_file(marker),
        "completed_at": _utc_now(),
    }
    write_json(output_root / "stage_state.json", passed)
    if checkpoint_path is not None:
        write_json(checkpoint_path, passed)


def _resume_contract(
    *,
    inputs: ContinuationInputs,
    adoption: Mapping[str, Any],
    checkout: Mapping[str, Any],
    project_commit: str,
    teacher_sha256: str,
    commands: Sequence[tuple[str, list[str], Path, str]],
) -> dict[str, Any]:
    """Freeze everything that may influence a resumed stage trajectory."""

    adoption_path = inputs.output_root / "generation_adoption_manifest.json"
    checkout_path = inputs.output_root / "upstream_checkout_audit.json"
    dataset_files = (
        ("graphs.pt", "dataset_summary.json")
        if inputs.dataset == "aids"
        else ("generation_source_graphs.pt", "dataset_summary.json")
    )
    scientific_files: dict[str, Path] = {
        "distance_checkpoint": inputs.distance_checkpoint,
        "dataset_csv": inputs.dataset_csv,
        "teacher_path": inputs.teacher_path,
        "molclr_checkpoint": inputs.molclr_checkpoint,
        "thresholds_path": inputs.thresholds_path,
        **{
            f"dataset_dir/{name}": _require_file(inputs.dataset_dir / name)
            for name in dataset_files
        },
    }
    if inputs.source_csv is not None:
        scientific_files["source_csv"] = inputs.source_csv
    if inputs.external_pair_store_source_manifest is not None:
        scientific_files["external_pair_store_source_manifest"] = (
            inputs.external_pair_store_source_manifest
        )
    if inputs.external_pair_store_source_checkpoint is not None:
        scientific_files["external_pair_store_source_checkpoint"] = (
            inputs.external_pair_store_source_checkpoint
        )
    if inputs.external_close_pair_view_manifest is not None:
        scientific_files["external_close_pair_view_manifest"] = (
            inputs.external_close_pair_view_manifest
        )
    scientific_input_files = {
        key: {
            "path": str(path.resolve(strict=True)),
            "sha256": sha256_file(path),
        }
        for key, path in sorted(scientific_files.items())
    }
    return {
        "schema_version": "comrecgc_standardized_stage_resume_v1",
        "dataset": inputs.dataset,
        "output_root": str(inputs.output_root),
        "project_commit": project_commit,
        "upstream_commit": checkout.get("actual_commit"),
        "generation_adoption_manifest_sha256": sha256_file(adoption_path),
        "upstream_checkout_audit_sha256": sha256_file(checkout_path),
        "source_generation_manifest_sha256": adoption[
            "source_run_manifest_sha256"
        ],
        "source_payload_sha256": adoption["counterfactuals_sha256_actual"],
        "teacher_sha256": teacher_sha256,
        "scientific_input_files": scientific_input_files,
        "common_recourse_engine": inputs.common_recourse_engine,
        "external_max_rss_gb": float(inputs.external_max_rss_gb),
        "external_query_block_size": int(inputs.external_query_block_size),
        "external_checkpoint_interval_blocks": int(
            inputs.external_checkpoint_interval_blocks
        ),
        "external_dbscan_shortcut_mode": inputs.external_dbscan_shortcut_mode,
        "external_shortcut_seed_count": int(inputs.external_shortcut_seed_count),
        "external_shortcut_failure_cap": int(inputs.external_shortcut_failure_cap),
        "external_shortcut_query_block_size": int(
            inputs.external_shortcut_query_block_size
        ),
        "external_exact_fallback_max_samples": int(
            inputs.external_exact_fallback_max_samples
        ),
        "external_summary_block_size": int(inputs.external_summary_block_size),
        "external_pair_store_source_manifest": (
            None
            if inputs.external_pair_store_source_manifest is None
            else str(inputs.external_pair_store_source_manifest.resolve(strict=True))
        ),
        "external_pair_store_source_checkpoint": (
            None
            if inputs.external_pair_store_source_checkpoint is None
            else str(
                inputs.external_pair_store_source_checkpoint.resolve(strict=True)
            )
        ),
        "external_pair_store_source_owner_root": (
            None
            if inputs.external_pair_store_source_owner_root is None
            else str(inputs.external_pair_store_source_owner_root.resolve(strict=True))
        ),
        "external_close_pair_view_manifest": (
            None
            if inputs.external_close_pair_view_manifest is None
            else str(inputs.external_close_pair_view_manifest.resolve(strict=True))
        ),
        "external_vector_cache_root": (
            None
            if inputs.external_vector_cache_root is None
            else str(inputs.external_vector_cache_root.resolve(strict=False))
        ),
        "external_vector_cache_lock": (
            None
            if inputs.external_vector_cache_lock is None
            else str(inputs.external_vector_cache_lock.resolve(strict=False))
        ),
        "external_vector_cache_route_lock": (
            None
            if inputs.external_vector_cache_route_lock is None
            else str(inputs.external_vector_cache_route_lock.resolve(strict=False))
        ),
        "external_vector_cache_min_free_gb": float(
            inputs.external_vector_cache_min_free_gb
        ),
        "external_vector_cache_proc_root": (
            None
            if inputs.external_pair_store_source_checkpoint is None
            else str(inputs.external_vector_cache_proc_root.resolve(strict=True))
        ),
        "expected_sklearn_version": inputs.expected_sklearn_version,
        "stages": [
            {
                "stage": stage,
                "argv_sha256": stable_json_sha256(argv),
                "marker": str(marker),
                "required_field": required_field,
            }
            for stage, argv, marker, required_field in commands
        ],
    }


def _validate_completed_stage(
    *,
    stage: str,
    argv: Sequence[str],
    marker: Path,
    required_field: str,
    checkpoint_path: Path,
) -> bool:
    checkpoint = _load_object(_require_file(checkpoint_path))
    payload = _load_object(_require_file(marker))
    if stage == "common_recourse":
        _validate_common_recourse_completion(marker=marker, terminal=payload)
    failures: list[str] = []
    if checkpoint.get("schema_version") != 2:
        failures.append("schema_version")
    status = checkpoint.get("status")
    if status not in {"RUNNING", "FAILED", "PASS"} or checkpoint.get("stage") != stage:
        failures.append("status_or_stage")
    if checkpoint.get("argv_sha256") != stable_json_sha256(list(argv)):
        failures.append("argv_sha256")
    if checkpoint.get("marker") != str(marker):
        failures.append("marker_path")
    if checkpoint.get("required_field") != required_field:
        failures.append("required_field")
    if payload.get(required_field) is not True:
        failures.append("marker_completion")
    marker_sha256 = sha256_file(marker)
    if status == "PASS" and checkpoint.get("marker_sha256") != marker_sha256:
        failures.append("marker_sha256")
    if failures:
        raise ValueError(
            f"RESUME_STAGE_CHECKPOINT_MISMATCH:{stage}:fields={failures}"
        )
    if status != "PASS":
        write_json(
            checkpoint_path,
            {
                **checkpoint,
                "status": "PASS",
                "marker_sha256": marker_sha256,
                "reconciled_after_child_completion": True,
                "completed_at": _utc_now(),
            },
        )
        return True
    return False


def _require_completed_stage_writer_quiescence(
    *,
    output_root: Path,
    stage: str,
    argv: Sequence[str],
    checkpoint_path: Path,
) -> None:
    checkpoint = _load_object(_require_file(checkpoint_path))
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("stage") != stage
        or checkpoint.get("argv_sha256") != stable_json_sha256(list(argv))
    ):
        raise ValueError(f"COMPLETED_STAGE_CHECKPOINT_IDENTITY_CHANGED:{stage}")
    contract = checkpoint.get("process_group_contract")
    if contract == "durable_barrier_dedicated_child_session_v2":
        binding = checkpoint.get("startup_barrier")
        _validate_stage_startup_binding(
            output_root=output_root,
            stage=stage,
            argv=argv,
            binding=binding,
            allowed_phases={"BOUND", "QUIESCENT"},
        )
    elif contract != "dedicated_child_session_v1":
        raise ValueError(f"COMPLETED_STAGE_PROCESS_GROUP_CONTRACT_CHANGED:{stage}")
    try:
        process_group_id = int(checkpoint["process_group_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"COMPLETED_STAGE_PROCESS_GROUP_MISSING:{stage}") from exc
    _wait_for_process_group_quiescence(
        process_group_id, proc_root=_PROC_ROOT, timeout_seconds=30.0
    )


def _validate_common_recourse_completion(
    *, marker: Path, terminal: Mapping[str, Any]
) -> None:
    """Close every large external-memory artifact before stage reconciliation."""

    root = marker.parent.resolve(strict=True)
    if (
        terminal.get("schema_version")
        != "comrecgc_common_recourse_terminal_v2"
        or terminal.get("run_complete") is not True
        or terminal.get("common_recourse_engine") != "external_memory_exact_v1"
    ):
        raise ValueError("RESUME_COMMON_TERMINAL_CONTRACT_MISMATCH")
    closure = terminal.get("artifact_sha256")
    if not isinstance(closure, Mapping):
        raise ValueError("RESUME_COMMON_TERMINAL_CLOSURE_MISSING")
    required = {
        "run_manifest.json",
        "selected_common_recourses.json",
        "selected_common_recourses.csv",
        "representative_counterfactuals.pt",
    }
    if not required.issubset(closure):
        raise ValueError("RESUME_COMMON_TERMINAL_CLOSURE_INCOMPLETE")
    for relative, expected_sha256 in closure.items():
        logical = root / str(relative)
        resolved = _require_file(logical)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("RESUME_COMMON_TERMINAL_PATH_ESCAPE") from exc
        if sha256_file(resolved) != str(expected_sha256):
            raise ValueError(
                f"RESUME_COMMON_TERMINAL_HASH_MISMATCH:{relative}"
            )
    manifest = _load_object(root / "run_manifest.json")
    external = manifest.get("external_memory_artifacts")
    if (
        manifest.get("run_complete") is not True
        or manifest.get("common_recourse_engine") != "external_memory_exact_v1"
        or not isinstance(external, Mapping)
    ):
        raise ValueError("RESUME_COMMON_RUN_MANIFEST_MISMATCH")
    adopted_pair_store = external.get("pair_store_adopted_read_only") is True
    adopted_pair_chunks = external.get("pair_chunks_adopted_read_only") is True
    if adopted_pair_store and adopted_pair_chunks:
        raise ValueError("RESUME_COMMON_MULTIPLE_PAIR_ADOPTION_ROUTES")
    pair_closure_relative = (
        "external_memory/chunk_vector_cache/run_manifest.json"
        if adopted_pair_chunks
        else (
            "external_memory/pair_store_adoption/run_manifest.json"
            if adopted_pair_store
            else "external_memory/pair_store/run_manifest.json"
        )
    )
    if pair_closure_relative not in closure:
        raise ValueError("RESUME_COMMON_PAIR_CLOSURE_MISSING")
    physical_vectors_path: Path
    physical_vectors_sha256: str
    physical_pairs_sha256: str
    if adopted_pair_chunks:
        chunk_manifest_path = _require_file(root / pair_closure_relative)
        if (
            str(external.get("chunk_vector_cache_manifest"))
            != str(chunk_manifest_path)
            or str(external.get("chunk_vector_cache_manifest_sha256"))
            != sha256_file(chunk_manifest_path)
            or external.get("pair_indices_materialized") is not False
            or external.get("local_vector_cache_is_scientific_authority") is not False
        ):
            raise ValueError("RESUME_COMMON_CHUNK_ADOPTION_BINDING_MISMATCH")
        try:
            chunk_result = validate_cartesian_chunk_vector_cache(
                chunk_manifest_path,
                require_cache=True,
            )
        except Exception as exc:
            raise ValueError("RESUME_COMMON_CHUNK_ADOPTION_CLOSURE_MISMATCH") from exc
        pair_manifest_path = chunk_result.manifest_path
        pair_identity = _load_object(pair_manifest_path).get("scientific_identity")
        if (
            not isinstance(pair_identity, Mapping)
            or pair_identity.get("source_pair_scientific_identity_sha256")
            != external.get("pair_store_scientific_identity_sha256")
        ):
            raise ValueError("RESUME_COMMON_CHUNK_ADOPTION_SCIENTIFIC_MISMATCH")
        physical_vectors_path = chunk_result.vectors_path
        physical_vectors_sha256 = chunk_result.vectors_sha256
        physical_pairs_sha256 = chunk_result.pairs.logical_npy_sha256
    elif adopted_pair_store:
        adoption_manifest_path = _require_file(root / pair_closure_relative)
        if (
            str(external.get("pair_store_adoption_manifest"))
            != str(adoption_manifest_path)
            or str(external.get("pair_store_adoption_manifest_sha256"))
            != sha256_file(adoption_manifest_path)
        ):
            raise ValueError("RESUME_COMMON_PAIR_ADOPTION_BINDING_MISMATCH")
        try:
            adopted = validate_adopted_pair_store_read_only(adoption_manifest_path)
        except Exception as exc:
            raise ValueError("RESUME_COMMON_PAIR_ADOPTION_CLOSURE_MISMATCH") from exc
        pair_manifest_path = adopted.pair_store.manifest_path
        physical_vectors_path = adopted.pair_store.vectors_path
        physical_vectors_sha256 = adopted.pair_store.vectors_sha256
        physical_pairs_sha256 = adopted.pair_store.pairs_sha256
    else:
        if (
            external.get("pair_store_adoption_manifest") is not None
            or external.get("pair_store_adoption_manifest_sha256") is not None
        ):
            raise ValueError("RESUME_COMMON_UNEXPECTED_PAIR_ADOPTION")
        pair_manifest_path = _require_file(root / pair_closure_relative)
    if (
        str(external.get("pair_store_manifest")) != str(pair_manifest_path)
        or str(external.get("pair_store_manifest_sha256"))
        != sha256_file(pair_manifest_path)
    ):
        raise ValueError("RESUME_COMMON_PAIR_MANIFEST_BINDING_MISMATCH")
    if not adopted_pair_chunks:
        pair_manifest = _load_object(pair_manifest_path)
        pair_identity = pair_manifest.get("scientific_identity")
        pair_identity_sha = pair_manifest.get("scientific_identity_sha256")
        if (
            pair_manifest.get("schema_version") != PAIR_STORE_SCHEMA
            or pair_manifest.get("run_complete") is not True
            or not isinstance(pair_identity, Mapping)
            or pair_identity_sha != stable_json_sha256(pair_identity)
            or external.get("pair_store_scientific_identity_sha256")
            != pair_identity_sha
        ):
            raise ValueError("RESUME_COMMON_PAIR_MANIFEST_INCOMPLETE")
        for path_field, hash_field in (
            ("pairs_path", "pairs_sha256"),
            ("vectors_path", "vectors_sha256"),
        ):
            artifact = _require_file(Path(str(pair_manifest.get(path_field) or "")))
            if (
                artifact.parent != pair_manifest_path.parent
                or sha256_file(artifact) != pair_manifest.get(hash_field)
            ):
                raise ValueError(
                    f"RESUME_COMMON_PAIR_ARTIFACT_MISMATCH:{path_field}"
                )
        physical_vectors_path = Path(str(pair_manifest["vectors_path"]))
        physical_vectors_sha256 = str(pair_manifest["vectors_sha256"])
        physical_pairs_sha256 = str(pair_manifest["pairs_sha256"])
    close_view = None
    close_manifest_raw = external.get("close_pair_view_manifest")
    close_manifest_sha = external.get("close_pair_view_manifest_sha256")
    if close_manifest_raw is None:
        if close_manifest_sha is not None:
            raise ValueError("RESUME_COMMON_CLOSE_VIEW_BINDING_MISMATCH")
        if (
            (adopted_pair_store or adopted_pair_chunks)
            and external.get("physical_pair_count") is not None
            and int(external["physical_pair_count"])
            == int(manifest.get("distance_pair_count", -2))
        ):
            raise ValueError("UNPROVEN_CARTESIAN_DBSCAN_INPUT")
    else:
        close_manifest_path = _require_file(Path(str(close_manifest_raw)))
        if sha256_file(close_manifest_path) != str(close_manifest_sha):
            raise ValueError("RESUME_COMMON_CLOSE_VIEW_BINDING_MISMATCH")
        try:
            close_view = validate_theta_close_pair_view(
                close_manifest_path,
                expected_physical_vectors_path=physical_vectors_path,
                expected_physical_vectors_sha256=physical_vectors_sha256,
                require_dbscan_eligible=True,
                require_pair_semantics_authority=True,
            )
        except Exception as exc:
            raise ValueError("RESUME_COMMON_CLOSE_VIEW_CLOSURE_MISMATCH") from exc
        if (
            close_view.logical_close_rows
            != int(external.get("logical_close_pair_count", -1))
            or close_view.logical_close_rows
            != int(external.get("dbscan_input_count", -1))
            or close_view.pairs_sha256 != external.get("pair_indices_sha256")
            or close_view.vectors_sha256 != external.get("recourse_vectors_sha256")
            or (
                close_view.all_pairs_close
                and close_view.pairs_sha256 != physical_pairs_sha256
            )
        ):
            raise ValueError("RESUME_COMMON_CLOSE_VIEW_SCIENTIFIC_MISMATCH")
    dbscan_manifest_raw = external.get("dbscan_manifest")
    if dbscan_manifest_raw is None:
        if int(manifest.get("theta_eligible_pair_count", -1)) != 0:
            raise ValueError("RESUME_COMMON_DBSCAN_MANIFEST_MISSING")
        return
    dbscan_manifest_path = _require_file(
        root / "external_memory/dbscan/run_manifest.json"
    )
    if (
        str(dbscan_manifest_raw) != str(dbscan_manifest_path)
        or str(external.get("dbscan_manifest_sha256"))
        != sha256_file(dbscan_manifest_path)
    ):
        raise ValueError("RESUME_COMMON_DBSCAN_MANIFEST_BINDING_MISMATCH")
    dbscan_manifest = _load_object(dbscan_manifest_path)
    if dbscan_manifest.get("run_complete") is not True:
        raise ValueError("RESUME_COMMON_DBSCAN_MANIFEST_INCOMPLETE")
    shortcut = dbscan_manifest.get("clustering_path") in {
        ALL_CORE_ONE_COMPONENT_SHORTCUT,
        ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    }
    component_recovery = (
        dbscan_manifest.get("clustering_path")
        == ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
    )
    required_dbscan_artifacts = [
        ("core_mask_path", "core_mask_sha256"),
        ("labels_path", "labels_sha256"),
    ]
    if shortcut:
        if dbscan_manifest.get("neighbor_counts_available") is not False:
            raise ValueError("RESUME_COMMON_DBSCAN_SHORTCUT_COUNT_CONTRACT_MISMATCH")
        try:
            _validate_shortcut_proof_closure(
                manifest=dbscan_manifest, root=dbscan_manifest_path.parent
            )
        except Exception as exc:
            raise ValueError("RESUME_COMMON_DBSCAN_SHORTCUT_CLOSURE_MISMATCH") from exc
    elif component_recovery:
        if dbscan_manifest.get("neighbor_counts_available") is not False:
            raise ValueError(
                "RESUME_COMMON_COMPONENT_RECOVERY_COUNT_CONTRACT_MISMATCH"
            )
        try:
            _validate_component_recovery_closure(
                manifest=dbscan_manifest, root=dbscan_manifest_path.parent
            )
        except Exception as exc:
            raise ValueError(
                "RESUME_COMMON_COMPONENT_RECOVERY_CLOSURE_MISMATCH"
            ) from exc
    else:
        required_dbscan_artifacts.insert(
            0, ("neighbor_counts_path", "neighbor_counts_sha256")
        )
    for path_field, hash_field in required_dbscan_artifacts:
        artifact = _require_file(Path(str(dbscan_manifest.get(path_field) or "")))
        if (
            artifact.parent != dbscan_manifest_path.parent
            or sha256_file(artifact) != dbscan_manifest.get(hash_field)
        ):
            raise ValueError(
                f"RESUME_COMMON_DBSCAN_ARTIFACT_MISMATCH:{path_field}"
            )
    summary_manifest_raw = external.get("one_cluster_summary_manifest")
    summary_manifest_sha256 = external.get("one_cluster_summary_manifest_sha256")
    if shortcut:
        if "external_memory/one_cluster_summary/run_manifest.json" not in closure:
            raise ValueError("RESUME_COMMON_ONE_CLUSTER_CLOSURE_MISSING")
        summary_manifest_path = _require_file(
            root / "external_memory/one_cluster_summary/run_manifest.json"
        )
        if (
            str(summary_manifest_raw) != str(summary_manifest_path)
            or str(summary_manifest_sha256) != sha256_file(summary_manifest_path)
        ):
            raise ValueError("RESUME_COMMON_ONE_CLUSTER_MANIFEST_BINDING_MISMATCH")
        try:
            exact_summary = validate_proven_one_cluster_summary(
                summary_manifest_path
            )
        except Exception as exc:
            raise ValueError("RESUME_COMMON_ONE_CLUSTER_CLOSURE_MISMATCH") from exc
        summary_manifest = _load_object(summary_manifest_path)
        identity = summary_manifest.get("scientific_identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("dbscan_manifest_sha256")
            != external.get("dbscan_manifest_sha256")
            or identity.get("pairs_sha256") != external.get("pair_indices_sha256")
            or [list(value) for value in exact_summary.official_result]
            != manifest.get("official_coverage_summary_result")
            or manifest.get("official_coverage_summary_invoked") is not False
            or manifest.get(
                "official_coverage_semantics_derived_for_single_label_zero"
            )
            is not True
        ):
            raise ValueError("RESUME_COMMON_ONE_CLUSTER_SCIENTIFIC_MISMATCH")
    elif summary_manifest_raw is not None or summary_manifest_sha256 is not None:
        raise ValueError("RESUME_COMMON_UNEXPECTED_ONE_CLUSTER_MANIFEST")
    component_summary_raw = external.get(
        "all_core_component_summary_manifest"
    )
    component_summary_sha = external.get(
        "all_core_component_summary_manifest_sha256"
    )
    if component_recovery:
        relative = "external_memory/all_core_component_summary/run_manifest.json"
        if relative not in closure:
            raise ValueError("RESUME_COMMON_COMPONENT_SUMMARY_CLOSURE_MISSING")
        component_summary_path = _require_file(root / relative)
        if (
            str(component_summary_raw) != str(component_summary_path)
            or str(component_summary_sha) != sha256_file(component_summary_path)
        ):
            raise ValueError(
                "RESUME_COMMON_COMPONENT_SUMMARY_MANIFEST_BINDING_MISMATCH"
            )
        try:
            component_summary = validate_proven_all_core_component_summary(
                component_summary_path,
                pair_indices=None,
                full_replay=True,
            )
        except Exception as exc:
            raise ValueError(
                "RESUME_COMMON_COMPONENT_SUMMARY_CLOSURE_MISMATCH"
            ) from exc
        component_identity = _load_object(component_summary_path).get(
            "scientific_identity"
        )
        selected_file = json.loads(
            (root / "selected_common_recourses.json").read_text(
                encoding="utf-8"
            )
        )
        selected_scientific_projection = []
        if isinstance(selected_file, list):
            for expected_row, published_row in zip(
                component_summary.selected, selected_file, strict=False
            ):
                if not isinstance(published_row, Mapping):
                    selected_scientific_projection.append(None)
                    continue
                selected_scientific_projection.append(
                    {
                        key: published_row.get(key)
                        for key in expected_row
                    }
                )
        if (
            not isinstance(component_identity, Mapping)
            or component_identity.get("dbscan_manifest_sha256")
            != external.get("dbscan_manifest_sha256")
            or component_identity.get("pairs_sha256")
            != external.get("pair_indices_sha256")
            or [list(value) for value in component_summary.official_result]
            != manifest.get("official_coverage_summary_result")
            or not isinstance(selected_file, list)
            or len(component_summary.selected) != len(selected_file)
            or component_summary.selected != selected_scientific_projection
        ):
            raise ValueError(
                "RESUME_COMMON_COMPONENT_SUMMARY_SCIENTIFIC_MISMATCH"
            )
    elif component_summary_raw is not None or component_summary_sha is not None:
        raise ValueError("RESUME_COMMON_UNEXPECTED_COMPONENT_SUMMARY_MANIFEST")


def _archive_previous_failure(output_root: Path) -> None:
    failure = output_root / "FAILED.json"
    if not failure.exists():
        return
    history = output_root / "failure_history"
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"FAILED.{os.stat(failure).st_mtime_ns}.json"
    if destination.exists():
        raise FileExistsError(f"Failure history collision: {destination}")
    os.replace(failure, destination)
    directory_fd = os.open(history, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _partial_directory_identity(path: Path) -> dict[str, int]:
    if path.is_symlink():
        raise ValueError(f"PARTIAL_STAGE_DIRECTORY_IS_SYMLINK:{path}")
    value = path.stat()
    if not stat_module.S_ISDIR(value.st_mode):
        raise ValueError(f"PARTIAL_STAGE_PATH_IS_NOT_DIRECTORY:{path}")
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "nlink": int(value.st_nlink),
    }


def _partial_directory_usage(path: Path) -> dict[str, int]:
    """Return a conservative logical-byte count for a physical partial tree."""

    root = path.resolve(strict=True)
    total = 0
    regular_files = 0
    directories = 0
    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        directories += 1
        for name in directory_names:
            entry = current / name
            value = entry.lstat()
            if entry.is_symlink() or not stat_module.S_ISDIR(value.st_mode):
                raise ValueError(f"PARTIAL_STAGE_TREE_ENTRY_NOT_PHYSICAL:{entry}")
        for name in file_names:
            entry = current / name
            value = entry.lstat()
            if entry.is_symlink() or not stat_module.S_ISREG(value.st_mode):
                raise ValueError(f"PARTIAL_STAGE_TREE_ENTRY_NOT_REGULAR:{entry}")
            total += int(value.st_size)
            regular_files += 1
            if total > _PARTIAL_STAGE_ARCHIVE_MAX_BYTES:
                raise ValueError(
                    "PARTIAL_STAGE_ARCHIVE_BYTE_LIMIT_EXCEEDED:"
                    f"bytes={total}:limit={_PARTIAL_STAGE_ARCHIVE_MAX_BYTES}"
                )
    return {
        "logical_bytes": total,
        "regular_file_count": regular_files,
        "directory_count": directories,
        "limit_bytes": _PARTIAL_STAGE_ARCHIVE_MAX_BYTES,
    }


def _read_proc_start_ticks(
    pid: int, *, proc_root: Path = _PROC_ROOT
) -> int | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        return int(raw[close + 2 :].split()[19])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def _proc_argv(pid: int, *, proc_root: Path = _PROC_ROOT) -> list[str] | None:
    try:
        stat_path = proc_root / str(pid) / "stat"
        before = stat_path.stat()
        raw = (proc_root / str(pid) / "cmdline").read_bytes()
        after = stat_path.stat()
    except (FileNotFoundError, PermissionError):
        return None
    if (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("STAGE_PROCESS_CHANGED_AROUND_CMDLINE_READ")
    return [part.decode("utf-8", errors="strict") for part in raw.split(b"\0") if part]


def _process_group_member_pids(
    process_group_id: int, *, proc_root: Path = _PROC_ROOT
) -> tuple[int, ...]:
    proc = proc_root.expanduser()
    if process_group_id <= 0 or not proc.is_dir():
        raise ValueError("PARTIAL_STAGE_PROCESS_GROUP_AUDIT_UNAVAILABLE")
    members: list[int] = []
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            raw = (pid_dir / "stat").read_text(encoding="utf-8")
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            process_state = fields[0]
            observed_group = int(fields[2])
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, UnicodeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"PARTIAL_STAGE_PROCESS_GROUP_AUDIT_UNAVAILABLE:pid={pid_dir.name}"
            ) from exc
        if observed_group == process_group_id and process_state not in {"Z", "X"}:
            members.append(int(pid_dir.name))
    return tuple(sorted(members))


def _wait_for_process_group_quiescence(
    process_group_id: int,
    *,
    proc_root: Path = _PROC_ROOT,
    timeout_seconds: float | None = 30.0,
    poll_seconds: float = 0.05,
) -> None:
    deadline: float | None = None
    while True:
        members = _process_group_member_pids(
            process_group_id, proc_root=proc_root
        )
        if not members:
            return
        now = time.monotonic()
        if deadline is None and timeout_seconds is not None:
            deadline = now + timeout_seconds
        if deadline is not None and now >= deadline:
            raise ValueError(
                "STAGE_CHILD_PROCESS_GROUP_NOT_QUIESCENT:"
                f"pgid={process_group_id}:members={list(members)[:16]}"
            )
        time.sleep(poll_seconds)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish an immutable JSON receipt without exposing a partial final file.

    The deterministic temporary inode is protected by ``flock``.  A prior
    writer killed during the write/link/fsync sequence can therefore be
    reconciled in place, while a live concurrent writer cannot be mistaken for
    a crash.  The final name is installed with hard-link no-replace semantics.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    ).encode("utf-8")
    # One fixed private temp per final path prevents changing audit timestamps
    # from accumulating a new orphan after every interrupted retry.
    temporary = path.parent / f".{path.name}.partial"
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow
    created = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        created = True
    except FileExistsError:
        descriptor = os.open(temporary, os.O_RDWR | nofollow)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        before = os.fstat(descriptor)
        current = os.lstat(temporary)
        if (
            not stat_module.S_ISREG(before.st_mode)
            or (before.st_mode & 0o777) != 0o600
            or before.st_uid != os.getuid()
            or before.st_dev != current.st_dev
            or before.st_ino != current.st_ino
            or before.st_nlink not in {1, 2}
        ):
            raise ValueError(f"IMMUTABLE_JSON_TEMP_IDENTITY_CHANGED:{temporary}")
        # A crash after link but before temporary cleanup leaves two names for
        # the same fully-written inode.  Validate and finish that publication.
        if before.st_nlink == 2:
            final_stat = os.lstat(path)
            if (
                final_stat.st_dev != before.st_dev
                or final_stat.st_ino != before.st_ino
                or path.read_bytes() != encoded
            ):
                raise ValueError(f"IMMUTABLE_JSON_LINK_RECONCILIATION_FAILED:{path}")
            temporary.unlink()
            _fsync_directory(path.parent)
            return
        current_bytes = os.pread(descriptor, before.st_size, 0)
        if created or current_bytes != encoded:
            os.ftruncate(descriptor, 0)
            written = 0
            while written < len(encoded):
                count = os.write(descriptor, encoded[written:])
                if count <= 0:
                    raise OSError("immutable JSON temporary write made no progress")
                written += count
            os.ftruncate(descriptor, len(encoded))
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            final_stat = os.lstat(path)
            if (
                not stat_module.S_ISREG(final_stat.st_mode)
                or path.read_bytes() != encoded
            ):
                raise ValueError(f"IMMUTABLE_JSON_FINAL_ALREADY_CHANGED:{path}")
            temporary.unlink()
            _fsync_directory(path.parent)
            return
        _fsync_directory(path.parent)
        final_stat = os.lstat(path)
        temporary_stat = os.fstat(descriptor)
        if (
            final_stat.st_dev != temporary_stat.st_dev
            or final_stat.st_ino != temporary_stat.st_ino
            or path.read_bytes() != encoded
        ):
            raise ValueError(f"IMMUTABLE_JSON_FINAL_IDENTITY_CHANGED:{path}")
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        os.close(descriptor)


def _archive_noncheckpointed_partial_stage(
    *,
    stage: str,
    argv: Sequence[str],
    marker: Path,
    required_field: str,
    checkpoint_path: Path,
    output_root: Path,
) -> Path:
    """Preserve one interrupted deterministic child before a fresh retry.

    The RUNNING/FAILED checkpoint was durably written before the child was
    spawned.  A recovery-only archive authorization is published with
    ``O_EXCL`` before the directory rename.  Therefore either side of a crash
    around the rename can be audited without deleting or overwriting the
    interrupted evidence.
    """

    if stage == "common_recourse":
        raise ValueError("COMMON_RECOURSE_MUST_USE_NATIVE_RESUME")
    partial = marker.parent
    if marker.exists():
        raise ValueError(f"PARTIAL_STAGE_ALREADY_HAS_TERMINAL:{stage}")
    root = output_root.resolve(strict=True)
    if partial.parent != root or partial.name != {
        "chemistry": "chemistry",
        "unified_eval": "unified_eval",
        "full_gate": "full_gate",
        "freeze": "standardized",
    }.get(stage):
        raise ValueError(f"PARTIAL_STAGE_PATH_CONTRACT_MISMATCH:{stage}:{partial}")
    checkpoint = _load_object(_require_file(checkpoint_path))
    if (
        checkpoint.get("schema_version") != 2
        or checkpoint.get("status") not in {"RUNNING", "FAILED"}
        or checkpoint.get("stage") != stage
        or checkpoint.get("argv_sha256") != stable_json_sha256(list(argv))
        or checkpoint.get("marker") != str(marker)
        or checkpoint.get("required_field") != required_field
        or checkpoint.get("process_group_contract")
        not in {
            "dedicated_child_session_v1",
            "durable_barrier_dedicated_child_session_v2",
        }
    ):
        raise ValueError(f"PARTIAL_STAGE_CHECKPOINT_MISMATCH:{stage}")
    process_group_id: int | None
    if (
        checkpoint.get("process_group_contract")
        == "durable_barrier_dedicated_child_session_v2"
    ):
        binding = checkpoint.get("startup_barrier")
        record = _validate_stage_startup_binding(
            output_root=output_root,
            stage=stage,
            argv=argv,
            binding=binding,
            allowed_phases={"PRE_ARM", "ARMED", "BOUND", "QUIESCENT"},
        )
        if binding["phase"] in {"PRE_ARM", "ARMED"}:
            if record is not None:
                validate_reopenable_unreleased_barrier(
                    record.record_path,
                    expected_target_argv=list(argv),
                    timeout_seconds=30.0,
                )
            process_group_id = None
        else:
            try:
                process_group_id = int(checkpoint["process_group_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"PARTIAL_STAGE_PROCESS_GROUP_MISSING:{stage}"
                ) from exc
    else:
        try:
            process_group_id = int(checkpoint["process_group_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"PARTIAL_STAGE_PROCESS_GROUP_MISSING:{stage}") from exc
    group_members_before = (
        ()
        if process_group_id is None
        else _process_group_member_pids(process_group_id, proc_root=_PROC_ROOT)
    )
    if group_members_before:
        raise ValueError(
            f"PARTIAL_STAGE_PROCESS_GROUP_STILL_LIVE:{stage}:"
            f"members={list(group_members_before)[:16]}"
        )
    writer_audit_before = _scan_live_source_writers(
        partial.resolve(strict=True),
        protected_snapshots=[],
        proc_root=_PROC_ROOT,
    )
    source_identity = _partial_directory_identity(partial)
    source_usage = _partial_directory_usage(partial)
    token = stable_json_sha256(
        {
            "stage": stage,
            "source": str(partial),
            "source_identity": source_identity,
            "argv_sha256": checkpoint["argv_sha256"],
        }
    )
    archive_root = root / "partial_stage_history"
    archive_root.mkdir(mode=0o755, exist_ok=True)
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise ValueError("PARTIAL_STAGE_ARCHIVE_ROOT_IS_NOT_PHYSICAL")
    destination = archive_root / f"{stage}.{token}"
    authorization_path = archive_root / f"{stage}.{token}.archive.json"
    prior_authorities = [
        candidate
        for candidate in archive_root.glob(f"{stage}.*.archive.json")
        if candidate != authorization_path
    ]
    if prior_authorities:
        raise ValueError(
            f"PARTIAL_STAGE_RETRY_LIMIT_REACHED:{stage}:"
            f"prior={sorted(str(value) for value in prior_authorities)}"
        )
    move_identity = {
        "schema_version": "comrecgc_partial_stage_archive_identity_v2",
        "status": "MOVE_AUTHORIZED",
        "stage": stage,
        "source": str(partial),
        "destination": str(destination),
        "source_identity": source_identity,
        "source_usage": source_usage,
        "argv_sha256": checkpoint["argv_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "process_group_id": process_group_id,
        "marker": str(marker),
        "marker_absent": True,
    }
    authorization = {
        "schema_version": "comrecgc_partial_stage_archive_v2",
        "status": "MOVE_AUTHORIZED",
        "move_identity": move_identity,
        "move_identity_sha256": stable_json_sha256(move_identity),
        # This snapshot is audit evidence from the publication instant, not a
        # replay identity.  /proc population and timestamps may legitimately
        # differ after a crash before the authorized rename.
        "publication_audit": {
            "process_group_members_before": [],
            "live_writer_audit_before": writer_audit_before,
            "observed_at": _utc_now(),
        },
    }
    if authorization_path.exists():
        if authorization_path.is_symlink():
            raise ValueError(f"PARTIAL_STAGE_ARCHIVE_AUTHORITY_CHANGED:{stage}")
        existing_authorization = _load_object(authorization_path)
        # Idempotent publication also reconciles the hardlink-after-publish
        # crash window (final + deterministic temporary name, nlink == 2).
        _write_new_json(authorization_path, existing_authorization)
        if (
            existing_authorization.get("schema_version")
            != "comrecgc_partial_stage_archive_v2"
            or existing_authorization.get("status") != "MOVE_AUTHORIZED"
            or existing_authorization.get("move_identity") != move_identity
            or existing_authorization.get("move_identity_sha256")
            != stable_json_sha256(move_identity)
            or not isinstance(existing_authorization.get("publication_audit"), Mapping)
        ):
            raise ValueError(f"PARTIAL_STAGE_ARCHIVE_AUTHORITY_CHANGED:{stage}")
    else:
        _write_new_json(authorization_path, authorization)
    group_members_after = (
        ()
        if process_group_id is None
        else _process_group_member_pids(process_group_id, proc_root=_PROC_ROOT)
    )
    if group_members_after:
        raise ValueError(
            f"PARTIAL_STAGE_PROCESS_GROUP_REAPPEARED:{stage}:"
            f"members={list(group_members_after)[:16]}"
        )
    _scan_live_source_writers(
        partial.resolve(strict=True),
        protected_snapshots=[],
        proc_root=_PROC_ROOT,
    )
    if _partial_directory_usage(partial) != source_usage:
        raise ValueError(f"PARTIAL_STAGE_ARCHIVE_USAGE_CHANGED:{stage}")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"PARTIAL_STAGE_ARCHIVE_DESTINATION_EXISTS:{stage}")
    os.rename(partial, destination)
    _fsync_directory(root)
    _fsync_directory(archive_root)
    if _partial_directory_identity(destination) != source_identity:
        raise ValueError(f"PARTIAL_STAGE_ARCHIVE_IDENTITY_CHANGED:{stage}")
    return destination


def _validate_route_inputs(inputs: ContinuationInputs) -> None:
    """Validate the physical-source route before bootstrap or continuation."""

    chunk_source_values = (
        inputs.external_pair_store_source_checkpoint,
        inputs.external_vector_cache_root,
        inputs.external_vector_cache_lock,
        inputs.external_vector_cache_route_lock,
    )
    if any(value is not None for value in chunk_source_values) and not all(
        value is not None for value in chunk_source_values
    ):
        raise ValueError("CHUNK_PAIR_STORE_ROUTE_ARGUMENTS_INCOMPLETE")
    if (
        inputs.external_pair_store_source_checkpoint is not None
        and inputs.external_close_pair_view_manifest is None
    ):
        raise ValueError("UNPROVEN_CARTESIAN_DBSCAN_INPUT")
    if (
        inputs.external_pair_store_source_manifest is not None
        and inputs.external_pair_store_source_checkpoint is not None
    ):
        raise ValueError("PAIR_STORE_ADOPTION_SOURCES_ARE_MUTUALLY_EXCLUSIVE")
    source_requested = (
        inputs.external_pair_store_source_manifest is not None
        or inputs.external_pair_store_source_checkpoint is not None
    )
    if inputs.external_close_pair_view_manifest is not None and not source_requested:
        raise ValueError("CLOSE_PAIR_VIEW_WITHOUT_PHYSICAL_SOURCE")
    if source_requested and inputs.external_pair_store_source_owner_root is None:
        raise ValueError("PAIR_STORE_ADOPTION_OWNER_ROOT_MISSING")
    if not source_requested and inputs.external_pair_store_source_owner_root is not None:
        raise ValueError("PAIR_STORE_ADOPTION_OWNER_ROOT_WITHOUT_SOURCE")
    if source_requested:
        route_matches = bool(
            inputs.device == "cpu"
            and inputs.common_recourse_engine == "external_memory_exact_v1"
            and (
                (
                    inputs.dataset == "aids"
                    and inputs.external_dbscan_shortcut_mode
                    == ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
                )
                or (
                    inputs.dataset == "mutagenicity"
                    and inputs.external_dbscan_shortcut_mode
                    == SKLEARN_FLOAT64_EXACT_MULTI_COMPONENT
                    and inputs.external_pair_store_source_manifest is not None
                    and inputs.external_pair_store_source_checkpoint is None
                )
            )
        )
        if not route_matches:
            raise ValueError("PAIR_STORE_ADOPTION_ROUTE_CONTRACT_MISMATCH")


def bootstrap_external_common_recovery_continuation(
    inputs: ContinuationInputs,
    *,
    execution_project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Create/reopen only the standardized continuation's immutable prelude.

    The caller must hold its own invocation-wide recovery writer lock.  This
    helper never starts common recourse or any downstream subprocess; it only
    freezes the same adoption, checkout, command and resume contract that
    :func:`run_continuation` will reopen after an independently proven exact
    DBSCAN/summary has been placed in the native external-memory directories.
    """

    _validate_route_inputs(inputs)
    if (
        inputs.dataset != "aids"
        or inputs.device != "cpu"
        or inputs.common_recourse_engine != "external_memory_exact_v1"
        or inputs.external_dbscan_shortcut_mode
        != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        or inputs.external_pair_store_source_manifest is None
        or inputs.external_pair_store_source_checkpoint is not None
        or inputs.external_close_pair_view_manifest is None
        or inputs.external_pair_store_source_owner_root is None
        or inputs.common_recourse_resume is not True
    ):
        raise ValueError("EXTERNAL_COMMON_RECOVERY_BOOTSTRAP_CONTRACT_MISMATCH")
    root = inputs.output_root
    if root.is_symlink():
        raise FileExistsError(f"Recovery OUTPUT_ROOT is a symlink: {root}")
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o755)
    if not root.is_dir() or (root / "PASS").exists():
        raise FileExistsError(f"Recovery OUTPUT_ROOT is not bootstrap-safe: {root}")
    adoption_path = root / "generation_adoption_manifest.json"
    checkout_path = root / "upstream_checkout_audit.json"
    resume_contract_path = root / "continuation_resume_contract.json"
    bootstrap_path = root / "exact_recovery_continuation_bootstrap.json"

    if adoption_path.exists():
        adoption = _load_object(_require_file(adoption_path))
        _verify_adopted_generation_integrity(adoption)
    else:
        adoption = validate_adopted_generation(inputs)
        write_json(adoption_path, adoption)
    checkout = verify_checkout(
        inputs.upstream_root,
        expected_commit=UPSTREAM_COMMIT,
        validate_imports=True,
    )
    if checkout_path.exists():
        frozen_checkout = _load_object(_require_file(checkout_path))
        if checkout.get("actual_commit") != frozen_checkout.get("actual_commit"):
            raise ValueError("BOOTSTRAP_UPSTREAM_COMMIT_MISMATCH")
        checkout = frozen_checkout
    else:
        write_json(checkout_path, checkout)
    execution_project_root = execution_project_root.expanduser().resolve(strict=True)
    project_commit = _git_head(execution_project_root)
    teacher_sha256 = sha256_file(inputs.teacher_path)
    commands = build_stage_commands(
        inputs,
        project_commit=project_commit,
        candidate_count=int(adoption["counterfactual_candidate_count"]),
        teacher_sha256=teacher_sha256,
        execution_project_root=execution_project_root,
    )
    contract = _resume_contract(
        inputs=inputs,
        adoption=adoption,
        checkout=checkout,
        project_commit=project_commit,
        teacher_sha256=teacher_sha256,
        commands=commands,
    )
    if resume_contract_path.exists():
        if _load_object(_require_file(resume_contract_path)) != contract:
            raise ValueError("BOOTSTRAP_SCIENTIFIC_CONTRACT_MISMATCH")
    else:
        write_json(resume_contract_path, contract)
    result = {
        "schema_version": "comrecgc_external_common_recovery_bootstrap_v1",
        "status": "READY_FOR_EXTERNAL_COMMON_RECOVERY",
        "output_root": str(root),
        "generation_adoption_manifest_sha256": sha256_file(adoption_path),
        "upstream_checkout_audit_sha256": sha256_file(checkout_path),
        "continuation_resume_contract_sha256": sha256_file(resume_contract_path),
        "common_recourse_started": False,
        "downstream_started": False,
    }
    if bootstrap_path.exists():
        if _load_object(_require_file(bootstrap_path)) != result:
            raise ValueError("BOOTSTRAP_RECEIPT_MISMATCH")
    else:
        write_json(bootstrap_path, result)
    return result


def run_continuation(inputs: ContinuationInputs) -> dict[str, Any]:
    _validate_route_inputs(inputs)
    resuming = inputs.output_root.exists()
    if resuming:
        if (
            not inputs.common_recourse_resume
            or inputs.common_recourse_engine != "external_memory_exact_v1"
            or inputs.dataset != "aids"
            or inputs.output_root.is_symlink()
            or not inputs.output_root.is_dir()
            or (inputs.output_root / "PASS").exists()
        ):
            raise FileExistsError(
                f"Fresh OUTPUT_ROOT already exists and is not resumable: {inputs.output_root}"
            )
    else:
        inputs.output_root.parent.mkdir(parents=True, exist_ok=True)
        inputs.output_root.mkdir(mode=0o755)
    try:
        adoption_path = inputs.output_root / "generation_adoption_manifest.json"
        checkout_path = inputs.output_root / "upstream_checkout_audit.json"
        resume_contract_path = inputs.output_root / "continuation_resume_contract.json"
        if resuming:
            adoption = _load_object(_require_file(adoption_path))
            _verify_adopted_generation_integrity(adoption)
        else:
            adoption = validate_adopted_generation(inputs)
        checkout = verify_checkout(
            inputs.upstream_root,
            expected_commit=UPSTREAM_COMMIT,
            validate_imports=True,
        )
        if not resuming:
            write_json(adoption_path, adoption)
            write_json(checkout_path, checkout)
        else:
            # Bind the live checkout verification to the immutable first-run
            # audit without rewriting it during recovery.
            frozen_checkout = _load_object(_require_file(checkout_path))
            if checkout.get("actual_commit") != frozen_checkout.get("actual_commit"):
                raise ValueError("RESUME_UPSTREAM_COMMIT_MISMATCH")
        project_commit = _git_head()
        # The frozen teacher is hashed exactly once here.  The shared evaluator
        # performs its own scientific input check; this driver reuses the same
        # identity for the downstream gate and final provenance instead of
        # rescanning a potentially large model repeatedly.
        teacher_sha256 = sha256_file(inputs.teacher_path)
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        commands = build_stage_commands(
            inputs,
            project_commit=project_commit,
            candidate_count=int(adoption["counterfactual_candidate_count"]),
            teacher_sha256=teacher_sha256,
        )
        contract = _resume_contract(
            inputs=inputs,
            adoption=adoption,
            checkout=_load_object(checkout_path),
            project_commit=project_commit,
            teacher_sha256=teacher_sha256,
            commands=commands,
        )
        if resuming:
            frozen_contract = _load_object(_require_file(resume_contract_path))
            if frozen_contract != contract:
                changed = sorted(
                    key
                    for key in set(frozen_contract) | set(contract)
                    if frozen_contract.get(key) != contract.get(key)
                )
                raise ValueError(
                    f"RESUME_SCIENTIFIC_CONTRACT_MISMATCH:fields={changed}"
                )
            _archive_previous_failure(inputs.output_root)
        else:
            write_json(resume_contract_path, contract)
        checkpoints = inputs.output_root / "stage_checkpoints"
        for stage, argv, marker, field in commands:
            checkpoint_path = checkpoints / f"{stage}.json"
            if resuming and marker.exists():
                _require_completed_stage_writer_quiescence(
                    output_root=inputs.output_root,
                    stage=stage,
                    argv=argv,
                    checkpoint_path=checkpoint_path,
                )
                _validate_completed_stage(
                    stage=stage,
                    argv=argv,
                    marker=marker,
                    required_field=field,
                    checkpoint_path=checkpoint_path,
                )
                continue
            if resuming and marker.parent.exists() and stage != "common_recourse":
                _archive_noncheckpointed_partial_stage(
                    stage=stage,
                    argv=argv,
                    marker=marker,
                    required_field=field,
                    checkpoint_path=checkpoint_path,
                    output_root=inputs.output_root,
                )
            _run_stage(
                stage=stage,
                argv=argv,
                marker=marker,
                required_field=field,
                environment=environment,
                output_root=inputs.output_root,
                checkpoint_path=checkpoint_path,
            )

        standardized = inputs.output_root / "standardized"
        source_manifest = _load_object(standardized / "run_manifest.json")
        freeze_manifest = _load_object(standardized / "freeze_manifest.json")
        if source_manifest.get("dataset_key") != inputs.dataset:
            raise ValueError("Standardized dataset identity mismatch")
        if source_manifest.get("cf_mode") != CF_MODE:
            raise ValueError("Standardized counterfactual mode mismatch")
        if source_manifest.get("distance_line") != DISTANCE_LINE:
            raise ValueError("Standardized distance line mismatch")
        if source_manifest.get("teacher_sha256") != teacher_sha256:
            raise ValueError("Standardized frozen teacher identity mismatch")
        if freeze_manifest.get("dataset_key") != inputs.dataset:
            raise ValueError("Freeze dataset identity mismatch")

        source_integrity_final = _verify_adopted_generation_integrity(adoption)
        source_integrity_final_path = inputs.output_root / "source_integrity_final.json"
        write_json(source_integrity_final_path, source_integrity_final)

        final = {
            "schema_version": 1,
            "status": "PASS",
            "dataset": inputs.dataset,
            "method": METHOD,
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "cf_mode": CF_MODE,
            "distance_line": DISTANCE_LINE,
            "generation_adopted": True,
            "generation_rerun": False,
            "ordering_adopted": False,
            "evaluation_adopted": False,
            "source_generation_root": str(inputs.source_generation_root),
            "standardized_output_root": str(standardized),
            "project_commit": project_commit,
            "source_generation_manifest_sha256": adoption[
                "source_run_manifest_sha256"
            ],
            "source_payload_sha256": adoption["counterfactuals_sha256_actual"],
            "source_payload_sha256_verified_once": True,
            "source_integrity_final_sha256": sha256_file(
                source_integrity_final_path
            ),
            "standardized_run_manifest_sha256": sha256_file(
                standardized / "run_manifest.json"
            ),
            "freeze_manifest_sha256": sha256_file(
                standardized / "freeze_manifest.json"
            ),
            "teacher_sha256": source_manifest.get("teacher_sha256"),
            "molclr_checkpoint_sha256": source_manifest.get(
                "molclr_checkpoint_sha256"
            ),
            "dataset_csv_sha256": source_manifest.get("dataset_csv_sha256"),
            "completed_at": _utc_now(),
        }
        write_json(inputs.output_root / "run_manifest.json", final)
        write_json(inputs.output_root / "final_gate.json", final)
        write_json(inputs.output_root / "_RUN_COMPLETE.json", {**final, "run_complete": True})
        atomic_write_bytes(inputs.output_root / "PASS", b"PASS\n")
        print(f"[COMRECGC_STANDARDIZED_CONTINUATION_PASS] dataset={inputs.dataset}")
        return final
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "FAILED",
            "dataset": inputs.dataset,
            "error_class": type(exc).__name__,
            "message": str(exc),
            "output_root": str(inputs.output_root),
            "failed_at": _utc_now(),
        }
        write_json(inputs.output_root / "FAILED.json", failure)
        raise


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=tuple(DATASET_CONTRACTS), required=True)
    parser.add_argument("--source-generation-root", type=_absolute, required=True)
    parser.add_argument("--upstream-root", type=_absolute, required=True)
    parser.add_argument("--dataset-dir", type=_absolute, required=True)
    parser.add_argument("--source-csv", type=_absolute)
    parser.add_argument("--distance-checkpoint", type=_absolute, required=True)
    parser.add_argument("--dataset-csv", type=_absolute, required=True)
    parser.add_argument("--teacher-path", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--thresholds-path", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--theta-star", type=float)
    parser.add_argument("--cost-cap", type=float)
    parser.add_argument(
        "--common-recourse-engine",
        choices=("legacy_in_memory", "external_memory_exact_v1"),
        default="legacy_in_memory",
    )
    parser.add_argument("--external-max-rss-gb", type=float, default=96.0)
    parser.add_argument("--external-query-block-size", type=int, default=8)
    parser.add_argument(
        "--external-checkpoint-interval-blocks", type=int, default=1
    )
    parser.add_argument(
        "--external-dbscan-shortcut-mode",
        choices=(
            "disabled",
            "all_core_one_component_adaptive_anchor_v1",
            "sklearn_float64_exact_multi_component_v1",
        ),
        default="disabled",
    )
    parser.add_argument("--external-shortcut-seed-count", type=int, default=3)
    parser.add_argument("--external-shortcut-failure-cap", type=int, default=4096)
    parser.add_argument(
        "--external-shortcut-query-block-size", type=int, default=65536
    )
    parser.add_argument(
        "--external-exact-fallback-max-samples", type=int, default=100000
    )
    parser.add_argument("--external-summary-block-size", type=int, default=65536)
    parser.add_argument(
        "--external-pair-store-source-manifest",
        type=_absolute,
        help="Completed pair-store manifest adopted read-only into this fresh run.",
    )
    parser.add_argument("--external-pair-store-source-checkpoint", type=_absolute)
    parser.add_argument("--external-pair-store-source-owner-root", type=_absolute)
    parser.add_argument("--external-close-pair-view-manifest", type=_absolute)
    parser.add_argument("--external-vector-cache-root", type=_absolute)
    parser.add_argument("--external-vector-cache-lock", type=_absolute)
    parser.add_argument("--external-vector-cache-route-lock", type=_absolute)
    parser.add_argument(
        "--external-vector-cache-min-free-gb", type=float, default=3.0
    )
    parser.add_argument(
        "--external-vector-cache-proc-root", type=_absolute, default=Path("/proc")
    )
    parser.add_argument("--expected-sklearn-version", default="1.7.2")
    parser.add_argument("--common-recourse-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "aids" and args.source_csv is None:
        raise SystemExit("AIDS requires --source-csv")
    if args.dataset == "mutagenicity" and args.source_csv is not None:
        raise SystemExit("Mutagenicity does not accept --source-csv")
    values = ContinuationInputs(
        dataset=args.dataset,
        source_generation_root=args.source_generation_root,
        upstream_root=_require_directory(args.upstream_root),
        dataset_dir=_require_directory(args.dataset_dir),
        source_csv=_require_file(args.source_csv) if args.source_csv else None,
        distance_checkpoint=_require_file(args.distance_checkpoint),
        dataset_csv=_require_file(args.dataset_csv),
        teacher_path=_require_file(args.teacher_path),
        molclr_root=_require_directory(args.molclr_root),
        molclr_checkpoint=_require_file(args.molclr_checkpoint),
        thresholds_path=_require_file(args.thresholds_path),
        output_root=args.output_root,
        device=str(args.device),
        theta_star=args.theta_star,
        cost_cap=args.cost_cap,
        common_recourse_engine=args.common_recourse_engine,
        external_max_rss_gb=args.external_max_rss_gb,
        external_query_block_size=args.external_query_block_size,
        external_checkpoint_interval_blocks=args.external_checkpoint_interval_blocks,
        external_dbscan_shortcut_mode=args.external_dbscan_shortcut_mode,
        external_shortcut_seed_count=args.external_shortcut_seed_count,
        external_shortcut_failure_cap=args.external_shortcut_failure_cap,
        external_shortcut_query_block_size=args.external_shortcut_query_block_size,
        external_exact_fallback_max_samples=(
            args.external_exact_fallback_max_samples
        ),
        external_summary_block_size=args.external_summary_block_size,
        external_pair_store_source_manifest=(
            _require_file(args.external_pair_store_source_manifest)
            if args.external_pair_store_source_manifest
            else None
        ),
        external_pair_store_source_checkpoint=(
            _require_file(args.external_pair_store_source_checkpoint)
            if args.external_pair_store_source_checkpoint
            else None
        ),
        external_pair_store_source_owner_root=(
            _require_directory(args.external_pair_store_source_owner_root)
            if args.external_pair_store_source_owner_root
            else None
        ),
        external_close_pair_view_manifest=(
            _require_file(args.external_close_pair_view_manifest)
            if args.external_close_pair_view_manifest
            else None
        ),
        external_vector_cache_root=args.external_vector_cache_root,
        external_vector_cache_lock=args.external_vector_cache_lock,
        external_vector_cache_route_lock=args.external_vector_cache_route_lock,
        external_vector_cache_min_free_gb=args.external_vector_cache_min_free_gb,
        external_vector_cache_proc_root=_require_directory(
            args.external_vector_cache_proc_root
        ),
        expected_sklearn_version=args.expected_sklearn_version,
        common_recourse_resume=args.common_recourse_resume,
    )
    run_continuation(values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
