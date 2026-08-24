#!/usr/bin/env python3
"""Continue a frozen COMRECGC generation into one standardized paper cell.

The completed generation is adopted read-only.  Every downstream stage writes
below a fresh output root and the PASS marker is published last.  This entry
point intentionally does not regenerate random walks or modify the recovery
root.
"""

from __future__ import annotations

import argparse
import json
import os
import stat as stat_module
import subprocess
import sys
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
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ALL_CORE_ONE_COMPONENT_SHORTCUT,
    _validate_shortcut_proof_closure,
)
from src.baselines.comrecgc.external_memory_recourse import (  # noqa: E402
    PAIR_STORE_SCHEMA,
    validate_adopted_pair_store_read_only,
    validate_proven_one_cluster_summary,
)
from src.baselines.comrecgc.external_pair_chunk_cache import (  # noqa: E402
    validate_cartesian_chunk_vector_cache,
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
    external_vector_cache_root: Path | None = None
    external_vector_cache_lock: Path | None = None
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
) -> list[tuple[str, list[str], Path, str]]:
    """Return ordered stage commands and their required completion markers."""

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
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_common_recourse.py"),
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
        if inputs.external_pair_store_source_checkpoint is not None:
            common_argv.extend(
                [
                    "--external-pair-store-source-checkpoint",
                    str(inputs.external_pair_store_source_checkpoint),
                    "--external-pair-store-source-owner-root",
                    str(inputs.external_pair_store_source_owner_root),
                    "--external-vector-cache-root",
                    str(inputs.external_vector_cache_root),
                    "--external-vector-cache-lock",
                    str(inputs.external_vector_cache_lock),
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
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(PROJECT_ROOT),
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
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/run_slot_unified_eval.py"),
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
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/gate_recovery.py"),
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
        str(PROJECT_ROOT / "scripts/baselines/comrecgc/freeze_recovery_result.py"),
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
    running = {
        "schema_version": 2,
        "status": "RUNNING",
        "stage": stage,
        "argv_sha256": argv_sha256,
        "marker": str(marker),
        "required_field": required_field,
        "started_at": _utc_now(),
    }
    write_json(output_root / "stage_state.json", running)
    if checkpoint_path is not None:
        write_json(checkpoint_path, running)
    try:
        subprocess.run(
            list(argv),
            cwd=PROJECT_ROOT,
            env=dict(environment),
            check=True,
        )
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
    except Exception as exc:
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
                "schema_version": 2,
                "status": "PASS",
                "stage": stage,
                "argv_sha256": stable_json_sha256(list(argv)),
                "marker": str(marker),
                "required_field": required_field,
                "marker_sha256": marker_sha256,
                "reconciled_after_child_completion": True,
                "completed_at": _utc_now(),
            },
        )
        return True
    return False


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
            or chunk_result.pairs.logical_npy_sha256
            != external.get("pair_indices_sha256")
            or chunk_result.vectors_sha256
            != external.get("recourse_vectors_sha256")
        ):
            raise ValueError("RESUME_COMMON_CHUNK_ADOPTION_SCIENTIFIC_MISMATCH")
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


def run_continuation(inputs: ContinuationInputs) -> dict[str, Any]:
    chunk_source_values = (
        inputs.external_pair_store_source_checkpoint,
        inputs.external_vector_cache_root,
        inputs.external_vector_cache_lock,
    )
    if any(value is not None for value in chunk_source_values) and not all(
        value is not None for value in chunk_source_values
    ):
        raise ValueError("CHUNK_PAIR_STORE_ROUTE_ARGUMENTS_INCOMPLETE")
    if (
        inputs.external_pair_store_source_manifest is not None
        and inputs.external_pair_store_source_checkpoint is not None
    ):
        raise ValueError("PAIR_STORE_ADOPTION_SOURCES_ARE_MUTUALLY_EXCLUSIVE")
    source_requested = (
        inputs.external_pair_store_source_manifest is not None
        or inputs.external_pair_store_source_checkpoint is not None
    )
    if source_requested and inputs.external_pair_store_source_owner_root is None:
        raise ValueError("PAIR_STORE_ADOPTION_OWNER_ROOT_MISSING")
    if not source_requested and inputs.external_pair_store_source_owner_root is not None:
        raise ValueError("PAIR_STORE_ADOPTION_OWNER_ROOT_WITHOUT_SOURCE")
    if (
        inputs.external_pair_store_source_manifest is not None
        or inputs.external_pair_store_source_checkpoint is not None
    ) and (
        inputs.dataset != "aids"
        or inputs.device != "cpu"
        or inputs.common_recourse_engine != "external_memory_exact_v1"
        or inputs.external_dbscan_shortcut_mode
        != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
    ):
        raise ValueError("PAIR_STORE_ADOPTION_ROUTE_CONTRACT_MISMATCH")
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
                _validate_completed_stage(
                    stage=stage,
                    argv=argv,
                    marker=marker,
                    required_field=field,
                    checkpoint_path=checkpoint_path,
                )
                continue
            if resuming and marker.parent.exists() and stage != "common_recourse":
                raise ValueError(
                    f"RESUME_NONCHECKPOINTED_PARTIAL_STAGE:{stage}:{marker.parent}"
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
        choices=("disabled", "all_core_one_component_adaptive_anchor_v1"),
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
    parser.add_argument("--external-vector-cache-root", type=_absolute)
    parser.add_argument("--external-vector-cache-lock", type=_absolute)
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
        external_vector_cache_root=args.external_vector_cache_root,
        external_vector_cache_lock=args.external_vector_cache_lock,
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
