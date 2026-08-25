"""Executable stages for the fresh AIDS disconnected-exact recovery route.

The exact DBSCAN and streamed component summary are deliberately placed below
the final continuation's ``common_recourse/external_memory`` directory.  The
existing continuation can then resume the common-recourse stage, reopen those
terminal artifacts, and complete chemistry, WNode evaluation, export, and
freeze without copying the 25 GB source or replaying the adopted seed/failure
scan.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterator, Mapping

import numpy as np

from src.baselines.comrecgc.close_pair_view import validate_theta_close_pair_view
from src.baselines.comrecgc.external_component_summary import (
    summarize_proven_all_core_components_external,
    validate_proven_all_core_component_summary,
)
from src.baselines.comrecgc.external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY,
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ExternalDBSCANContract,
    _load_checkpoint,
    _validate_component_recovery_closure,
)
from src.baselines.comrecgc.failed_selection_recovery import (
    FailedSelectionRecoverySource,
    fit_promoted_failed_selection_component_recovery,
    promote_failed_adaptive_selection_for_component_recovery,
)
from src.baselines.comrecgc.production_subset_audit import (
    ProductionSubsetAuditContract,
    run_production_subset_equivalence_audit,
)
from src.baselines.comrecgc.upstream import imported_upstream
from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (
    ADOPTION_STAGE,
    DOWNSTREAM_STAGE,
    EXACT_STAGE,
    EXACT_STAGE_RECEIPT_SCHEMA,
    FINAL_STAGE_RECEIPT_SCHEMA,
    FINAL_STAGE,
    SUBSET_STAGE_RECEIPT_SCHEMA,
    SUBSET_MAX_ATTEMPTS,
    SUBSET_STAGE,
    RecoveryControllerError,
    load_bound_controller_manifest,
    open_typed_recovery_gate,
    sha256_file,
    stable_json_sha256,
    validate_stage_terminal,
    validate_typed_adoption_receipt,
)


SOURCE_EVIDENCE_RECEIPT_SCHEMA = "aids_c766_failed_tree_small_evidence_copy_v1"
EXPECTED_FAILED_TREE_FILES = 14
EXPECTED_PROMOTED_EVIDENCE_FILES = 13


class RecoveryStageError(RuntimeError):
    """A typed recovery stage failed closed."""


def _require_cpu_stage_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Reject direct CLI execution outside the controller's frozen CPU env."""

    threads = str(manifest["resources"]["thread_count"])
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "DEVICE": "cpu",
        "GPU_REQUIRED": "0",
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
    }
    observed = {field: os.environ.get(field) for field in expected}
    if observed != expected:
        raise RecoveryStageError(
            f"CPU-only stage environment changed: expected={expected}:observed={observed}"
        )
    return expected


def _require_controller_process_group() -> dict[str, int]:
    pid = os.getpid()
    process_group_id = os.getpgrp()
    if pid != process_group_id:
        raise RecoveryStageError(
            "final recovery stage requires controller start_new_session ownership"
        )
    return {"runner_pid": pid, "process_group_id": process_group_id}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise RecoveryStageError(f"JSON authority may not be a symlink: {source}")
    resolved = source.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryStageError(f"JSON authority is not an object: {resolved}")
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.publish.tmp"
    if (path.exists() or path.is_symlink()) and not (
        temporary.exists() or temporary.is_symlink()
    ):
        raise RecoveryStageError(f"immutable stage output exists: {path}")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except FileExistsError:
        descriptor = os.open(
            temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        current = temporary.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink not in {1, 2}
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RecoveryStageError("immutable stage temp identity changed")
        if opened.st_nlink == 2:
            final = path.lstat()
            if (
                (final.st_dev, final.st_ino) != (opened.st_dev, opened.st_ino)
                or path.read_bytes() != encoded
            ):
                raise RecoveryStageError("linked stage publication changed")
            temporary.unlink()
            _fsync_directory(path.parent)
            return
        if path.exists() or path.is_symlink():
            raise RecoveryStageError(f"immutable stage output exists: {path}")
        os.ftruncate(descriptor, 0)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RecoveryStageError("immutable stage write made no progress")
            offset += written
        os.ftruncate(descriptor, len(encoded))
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoveryStageError(f"immutable stage output exists: {path}") from exc
        _fsync_directory(path.parent)
        final = path.lstat()
        linked = os.fstat(descriptor)
        if (
            (final.st_dev, final.st_ino) != (linked.st_dev, linked.st_ino)
            or path.read_bytes() != encoded
        ):
            raise RecoveryStageError("immutable stage output identity changed")
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        os.close(descriptor)


def _reconcile_immutable_stage_publication(path: Path) -> bool:
    temporary = path.parent / f".{path.name}.publish.tmp"
    if not temporary.exists() and not temporary.is_symlink():
        return False
    if not path.exists() or path.is_symlink():
        return False
    descriptor = os.open(
        temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        final = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 2
            or (opened.st_dev, opened.st_ino) != (final.st_dev, final.st_ino)
        ):
            raise RecoveryStageError("stage publication temp cannot be reconciled")
        temporary.unlink()
        _fsync_directory(path.parent)
        return True
    finally:
        os.close(descriptor)


def _copy_new_file(source: Path, target: Path, *, expected_sha256: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise RecoveryStageError(f"source evidence is not physical: {source}")
    if sha256_file(source) != expected_sha256:
        raise RecoveryStageError(f"source evidence SHA256 changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.copy.tmp"
    if target.exists() and temporary.exists():
        descriptor = os.open(
            temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            final = target.lstat()
            if (
                opened.st_nlink != 2
                or (opened.st_dev, opened.st_ino) != (final.st_dev, final.st_ino)
            ):
                raise RecoveryStageError(f"evidence copy temp changed: {target}")
            temporary.unlink()
            _fsync_directory(target.parent)
        finally:
            os.close(descriptor)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or sha256_file(target) != expected_sha256:
            raise RecoveryStageError(f"promoted evidence changed: {target}")
        return
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
    except FileExistsError:
        descriptor = os.open(
            temporary, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened = os.fstat(descriptor)
        current = temporary.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RecoveryStageError(f"evidence copy temp changed: {target}")
        os.ftruncate(descriptor, 0)
        with source.open("rb") as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                offset = 0
                while offset < len(chunk):
                    written = os.write(descriptor, chunk[offset:])
                    if written <= 0:
                        raise RecoveryStageError("evidence copy made no progress")
                    offset += written
        os.fsync(descriptor)
        if sha256_file(temporary) != expected_sha256:
            raise RecoveryStageError(f"evidence copy changed bytes: {source}")
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError as exc:
            raise RecoveryStageError(f"evidence copy raced: {target}") from exc
        _fsync_directory(target.parent)
        temporary.unlink()
        _fsync_directory(target.parent)
    finally:
        os.close(descriptor)


class _HeldStageLock:
    def __init__(self, root: Path, descriptor: int, root_stat: os.stat_result) -> None:
        self.root = root
        self.path = root / ".recovery-stage.lock"
        self.descriptor = descriptor
        self.root_device = int(root_stat.st_dev)
        self.root_inode = int(root_stat.st_ino)
        opened = os.fstat(descriptor)
        self.lock_device = int(opened.st_dev)
        self.lock_inode = int(opened.st_ino)

    def verify(self) -> None:
        root_stat = self.root.stat()
        lock_stat = self.path.lstat()
        opened = os.fstat(self.descriptor)
        if (
            self.root.is_symlink()
            or self.path.is_symlink()
            or (int(root_stat.st_dev), int(root_stat.st_ino))
            != (self.root_device, self.root_inode)
            or (int(opened.st_dev), int(opened.st_ino))
            != (self.lock_device, self.lock_inode)
            or (int(lock_stat.st_dev), int(lock_stat.st_ino))
            != (self.lock_device, self.lock_inode)
            or not stat.S_ISREG(opened.st_mode)
        ):
            raise RecoveryStageError("recovery stage writer/root identity changed")


@contextmanager
def _stage_writer(root: Path, *, resume: bool) -> Iterator[_HeldStageLock]:
    requested = root.expanduser().resolve(strict=False)
    if requested.exists() or requested.is_symlink():
        if not resume or requested.is_symlink() or not requested.is_dir():
            raise RecoveryStageError("stage root exists but resume was not authorized")
    else:
        if resume:
            raise RecoveryStageError("resume stage root is absent")
        requested.parent.mkdir(parents=True, exist_ok=True)
        requested.mkdir(mode=0o755)
    lock_path = requested / ".recovery-stage.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RecoveryStageError("another recovery stage writer is active") from exc
        held = _HeldStageLock(requested, descriptor, requested.stat())
        held.verify()
        yield held
        held.verify()
    finally:
        os.close(descriptor)


def _require_stage_paths(
    manifest: Mapping[str, Any],
    *,
    stage_id: str,
    output_dir: str | Path,
    bindings: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    stage = next(row for row in manifest["stages"] if row["stage_id"] == stage_id)
    if str(Path(output_dir).resolve(strict=False)) != stage["output_dir"]:
        raise RecoveryStageError(f"{stage_id} output path changed")
    for role, value in bindings.items():
        expected = stage["argv_bindings"][role]["value"]
        if str(Path(value).resolve(strict=False)) != expected:
            raise RecoveryStageError(f"{stage_id} typed binding changed: {role}")
    return stage


def _archive_failed_tree(
    *, receipt: Mapping[str, Any], target_root: Path
) -> dict[str, Any]:
    inventory = receipt.get("failed_tree_inventory")
    final_task = receipt.get("final_task")
    if (
        not isinstance(inventory, list)
        or len(inventory) != EXPECTED_FAILED_TREE_FILES
        or not isinstance(final_task, Mapping)
    ):
        raise RecoveryStageError("typed adoption failed-tree inventory is incomplete")
    source_root = Path(str(final_task.get("expected_output") or "")).resolve(strict=True)
    relative_names = [str(row.get("relative_path") or "") for row in inventory]
    if relative_names.count("FAILED.json") != 1:
        raise RecoveryStageError("failed-tree inventory lacks its failed marker")
    copied: list[dict[str, Any]] = []
    for row in inventory:
        relative = str(row.get("relative_path") or "")
        digest = str(row.get("sha256") or "")
        if relative == "FAILED.json":
            continue
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise RecoveryStageError("failed-tree relative path escaped")
        source = (source_root / rel).resolve(strict=True)
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise RecoveryStageError("failed-tree source path escaped") from exc
        target = target_root / rel
        _copy_new_file(source, target, expected_sha256=digest)
        copied.append(
            {"relative_path": relative, "path": str(target), "sha256": digest}
        )
    if len(copied) != EXPECTED_PROMOTED_EVIDENCE_FILES:
        raise RecoveryStageError("unexpected promoted failed-tree evidence count")
    if (target_root / "FAILED.json").exists():
        raise RecoveryStageError("FAILED marker was copied into fresh science")
    payload = {
        "schema_version": SOURCE_EVIDENCE_RECEIPT_SCHEMA,
        "status": "RECOVERY_ONLY_EVIDENCE_COPIED",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_inventory_count": len(inventory),
        "promoted_artifact_count": len(copied),
        "failed_marker_excluded": True,
        "source_large_arrays_copied": False,
        "artifacts": copied,
    }
    payload["receipt_sha256"] = stable_json_sha256(payload)
    receipt_path = target_root / "source_evidence_receipt.json"
    _reconcile_immutable_stage_publication(receipt_path)
    if receipt_path.exists():
        if _read_json(receipt_path) != payload:
            raise RecoveryStageError("source evidence receipt changed")
    else:
        _write_new_json(receipt_path, payload)
    return payload


def run_subset_stage(
    *,
    controller_manifest: str | Path,
    adoption_gate: str | Path,
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    manifest = load_bound_controller_manifest(controller_manifest)
    observed_environment = _require_cpu_stage_environment(manifest)
    _require_stage_paths(
        manifest,
        stage_id=SUBSET_STAGE,
        output_dir=output_dir,
        bindings={
            "controller_manifest": controller_manifest,
            "adoption_gate": adoption_gate,
        },
    )
    gate = open_typed_recovery_gate(manifest, ADOPTION_STAGE)
    if str(Path(adoption_gate).resolve(strict=True)) != str(
        Path(manifest["controller_root"]) / "gates/01_failed_selection_adoption.json"
    ) or gate["gate_sha256"] != _read_json(adoption_gate)["gate_sha256"]:
        raise RecoveryStageError("subset adoption gate changed")
    source = manifest["source_authority"]
    runtime = manifest["runtime_inputs"]
    root = Path(output_dir).resolve(strict=False)
    terminal_path = Path(
        next(
            row["terminal_path"]
            for row in manifest["stages"]
            if row["stage_id"] == SUBSET_STAGE
        )
    )
    with _stage_writer(root, resume=resume) as held:
        _reconcile_immutable_stage_publication(terminal_path)
        if terminal_path.exists():
            if not resume:
                raise RecoveryStageError(
                    "completed subset stage requires explicit resume adoption"
                )
            validate_stage_terminal(manifest, stage_id=SUBSET_STAGE)
            held.verify()
            return _read_json(terminal_path)
        attempts: list[int] = []
        for entry in root.iterdir():
            if entry.name.startswith("attempt-"):
                if entry.is_symlink() or not entry.is_dir():
                    raise RecoveryStageError("subset attempt authority is not physical")
                suffix = entry.name[len("attempt-") :]
                if not suffix.isdigit():
                    raise RecoveryStageError("subset attempt name is malformed")
                attempts.append(int(suffix))
        completed: tuple[int, Path] | None = None
        for attempt in sorted(attempts):
            candidate = root / f"attempt-{attempt}/production_subset_equivalence.json"
            marker = candidate.parent / "PASS"
            if candidate.is_file() and marker.is_file() and marker.read_bytes() == b"PASS\n":
                if completed is not None:
                    raise RecoveryStageError("multiple subset attempts claim terminal PASS")
                completed = (attempt, candidate)
        if completed is None:
            attempt = max(attempts, default=-1) + 1
            if attempt >= SUBSET_MAX_ATTEMPTS:
                raise RecoveryStageError("subset recovery attempt limit exceeded")
            attempt_root = root / f"attempt-{attempt}"
            run_production_subset_equivalence_audit(
                close_pair_contract_path=source["close_pair_manifest_path"],
                expected_close_pair_contract_sha256=source[
                    "close_pair_manifest_sha256"
                ],
                physical_pairs_path=source["physical_pairs_path"],
                expected_physical_pairs_sha256=source["physical_pairs_sha256"],
                output_dir=attempt_root,
                contract=ProductionSubsetAuditContract(
                    eps=0.02,
                    min_samples=3,
                    radius=0.02,
                    recourse_size=100,
                    subset_size=int(manifest["resources"]["subset_size"]),
                    seed=0,
                    scan_block_size=int(manifest["resources"]["block_size"]),
                    query_block_size=64,
                    max_rss_bytes=8 * 1024**3,
                    expected_sklearn_version=str(
                        runtime["expected_sklearn_version"]
                    ),
                ),
            )
            completed = (
                attempt,
                attempt_root / "production_subset_equivalence.json",
            )
        attempt, subset_manifest_path = completed
        payload = {
            "schema_version": SUBSET_STAGE_RECEIPT_SCHEMA,
            "status": "PASS",
            "run_complete": True,
            "recovery_only": True,
            "ordinary_pass_dependency_eligible": False,
            "controller_manifest_path": manifest["manifest_path"],
            "controller_manifest_sha256": manifest["manifest_sha256"],
            "adoption_gate_sha256": gate["gate_sha256"],
            "attempt": attempt,
            "subset_manifest_path": str(subset_manifest_path),
            "subset_manifest_sha256": sha256_file(subset_manifest_path),
            "full_production_dbscan_equivalence_claimed": False,
            "observed_environment": observed_environment,
            "completed_at": _utc_now(),
        }
        held.verify()
        _write_new_json(terminal_path, payload)
        held.verify()
        return payload


def run_exact_stage(
    *,
    controller_manifest: str | Path,
    adoption_gate: str | Path,
    subset_gate: str | Path,
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    manifest = load_bound_controller_manifest(controller_manifest)
    observed_environment = _require_cpu_stage_environment(manifest)
    stage = _require_stage_paths(
        manifest,
        stage_id=EXACT_STAGE,
        output_dir=output_dir,
        bindings={
            "controller_manifest": controller_manifest,
            "adoption_gate": adoption_gate,
            "subset_gate": subset_gate,
        },
    )
    open_typed_recovery_gate(manifest, ADOPTION_STAGE)
    open_typed_recovery_gate(manifest, SUBSET_STAGE)
    typed = validate_typed_adoption_receipt(manifest=manifest)
    root = Path(output_dir).resolve(strict=False)
    terminal_path = Path(stage["terminal_path"])
    with _stage_writer(root, resume=resume) as held:
        _reconcile_immutable_stage_publication(terminal_path)
        existing_receipt: dict[str, Any] | None = None
        if terminal_path.exists():
            if not resume:
                raise RecoveryStageError(
                    "completed exact stage requires explicit resume adoption"
                )
            existing_receipt = _read_json(terminal_path)
        final_output = Path(
            next(
                row["output_dir"]
                for row in manifest["stages"]
                if row["stage_id"] == FINAL_STAGE
            )
        )
        from scripts.autodl.run_comrecgc_standardized_continuation import (
            bootstrap_external_common_recovery_continuation,
        )

        bootstrap = bootstrap_external_common_recovery_continuation(
            _continuation_inputs(manifest, final_output)
        )
        held.verify()
        evidence = _archive_failed_tree(
            receipt=typed["receipt"], target_root=root / "source_evidence"
        )
        source = manifest["source_authority"]
        checkpoint = _load_checkpoint(Path(source["failed_checkpoint_path"]))
        contract = ExternalDBSCANContract(**checkpoint["identity"]["contract"])
        if (
            contract.shortcut_mode != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
            or contract.eps != 0.02
            or contract.min_samples != 3
            or contract.max_rss_bytes != manifest["resources"]["max_rss_bytes"]
        ):
            raise RecoveryStageError("adopted DBSCAN contract changed")
        recovery_source = FailedSelectionRecoverySource(
            checkpoint_path=Path(source["failed_checkpoint_path"]),
            checkpoint_sha256=source["failed_checkpoint_sha256"],
            selection_manifest_path=Path(source["adaptive_selection_path"]),
            selection_manifest_sha256=source["adaptive_selection_sha256"],
            failure_artifact_path=Path(source["failed_shortcut_artifact_path"]),
            failure_artifact_sha256=source["failed_shortcut_artifact_sha256"],
            failure_indices_sha256=source["failure_indices_sha256"],
            anchor_indices_sha256=source["anchor_indices_sha256"],
            anchor_rows_sha256=source["anchor_rows_sha256"],
        )
        dbscan_root = root / "dbscan"
        promotion = promote_failed_adaptive_selection_for_component_recovery(
            vectors_path=source["source_vectors_path"],
            work_dir=dbscan_root,
            source=recovery_source,
            contract=contract,
            expected_vectors_sha256=source["source_vectors_sha256"],
            adoption_receipt_path=typed["receipt_path"],
            adoption_receipt_sha256=typed["receipt_sha256"],
            source_authority_sha256=stable_json_sha256(source),
            resume=resume,
        )
        result = fit_promoted_failed_selection_component_recovery(
            vectors_path=source["source_vectors_path"],
            work_dir=dbscan_root,
            contract=contract,
            expected_vectors_sha256=source["source_vectors_sha256"],
        )
        dbscan = _read_json(result.manifest_path)
        if (
            dbscan.get("clustering_path") != ADAPTIVE_ALL_CORE_COMPONENT_RECOVERY
            or dbscan.get("approximation_used") is not False
        ):
            raise RecoveryStageError("exact component DBSCAN did not close")
        _validate_component_recovery_closure(manifest=dbscan, root=dbscan_root)
        payload = {
            "schema_version": EXACT_STAGE_RECEIPT_SCHEMA,
            "status": "PASS",
            "run_complete": True,
            "recovery_only": True,
            "ordinary_pass_dependency_eligible": False,
            "dbscan_partition_proven": True,
            "dbscan_manifest_path": str(result.manifest_path),
            "dbscan_manifest_sha256": result.manifest_sha256,
            "promotion_manifest_path": str(promotion.promotion_manifest_path),
            "promotion_manifest_sha256": promotion.promotion_manifest_sha256,
            "source_evidence_receipt_path": str(
                root / "source_evidence/source_evidence_receipt.json"
            ),
            "source_evidence_receipt_sha256": sha256_file(
                root / "source_evidence/source_evidence_receipt.json"
            ),
            "promoted_source_artifact_count": evidence["promoted_artifact_count"],
            "continuation_bootstrap_path": str(
                final_output / "exact_recovery_continuation_bootstrap.json"
            ),
            "continuation_bootstrap_sha256": sha256_file(
                final_output / "exact_recovery_continuation_bootstrap.json"
            ),
            "continuation_bootstrap": bootstrap,
            "recovery_source_authority": {
                "adoption_receipt_sha256": typed["receipt_sha256"],
                "task_state_projection_sha256": manifest["adoption_contract"][
                    "expected_task_state_projection_sha256"
                ],
                "source_authority_sha256": stable_json_sha256(source),
                "seed_failure_scan_reexecuted": False,
                "source_seed_failure_ledgers_adopted_read_only": True,
                "fresh_component_ledger_derived": True,
                "failed_dbscan_terminal_adopted_as_pass": False,
                "source_access": "read_only",
                "source_vectors_zero_copy": True,
            },
            "gpu_used": False,
            "observed_environment": observed_environment,
            "completed_at": (
                existing_receipt.get("completed_at")
                if existing_receipt is not None
                else _utc_now()
            ),
        }
        if (
            not isinstance(payload["completed_at"], str)
            or not payload["completed_at"]
        ):
            raise RecoveryStageError("exact stage completion timestamp changed")
        held.verify()
        if existing_receipt is not None:
            if existing_receipt != payload:
                raise RecoveryStageError("exact stage terminal identity changed")
        else:
            _write_new_json(terminal_path, payload)
        held.verify()
        return payload


def run_downstream_stage(
    *,
    controller_manifest: str | Path,
    exact_gate: str | Path,
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    manifest = load_bound_controller_manifest(controller_manifest)
    observed_environment = _require_cpu_stage_environment(manifest)
    _require_stage_paths(
        manifest,
        stage_id=DOWNSTREAM_STAGE,
        output_dir=output_dir,
        bindings={
            "controller_manifest": controller_manifest,
            "exact_gate": exact_gate,
        },
    )
    gate = open_typed_recovery_gate(manifest, EXACT_STAGE)
    if str(Path(exact_gate).resolve(strict=True)) != str(
        Path(manifest["controller_root"]) / "gates/03_exact_component_recovery.json"
    ) or gate["gate_sha256"] != _read_json(exact_gate)["gate_sha256"]:
        raise RecoveryStageError("downstream exact gate changed")
    exact_receipt = _read_json(gate["artifact"]["path"])
    dbscan_path = Path(exact_receipt["dbscan_manifest_path"]).resolve(strict=True)
    dbscan_sha = str(exact_receipt["dbscan_manifest_sha256"])
    dbscan = _read_json(dbscan_path)
    _validate_component_recovery_closure(manifest=dbscan, root=dbscan_path.parent)
    source = manifest["source_authority"]
    close_view = validate_theta_close_pair_view(
        source["close_pair_manifest_path"],
        require_dbscan_eligible=True,
        require_pair_semantics_authority=True,
    )
    vectors = close_view.open_vectors()
    pairs = close_view.open_pairs()
    labels = np.load(dbscan["labels_path"], mmap_mode="r", allow_pickle=False)
    import torch

    with imported_upstream(manifest["runtime_inputs"]["upstream_root"]) as modules:
        result = summarize_proven_all_core_components_external(
            work_dir=output_dir,
            dbscan_manifest_path=dbscan_path,
            dbscan_manifest_sha256=dbscan_sha,
            labels=labels,
            recourse_vectors=vectors,
            pair_indices=pairs,
            pairs_sha256=close_view.pairs_sha256,
            pair_authority_manifest_path=close_view.manifest_path,
            pair_authority_manifest_sha256=close_view.manifest_sha256,
            radius=0.02,
            theta=0.1,
            recourse_size=100,
            official_greedy=modules[
                "common_recourse"
            ].greedy_counterfactual_summary_from_covering_sets,
            torch_module=torch,
            max_rss_bytes=int(manifest["resources"]["max_rss_bytes"]),
            block_size=int(manifest["resources"]["block_size"]),
            resume=resume,
        )
    # The stage returns only after a full terminal replay, which is the
    # multi-component replacement for the historical one-cluster radius A/B.
    validate_proven_all_core_component_summary(
        result.manifest_path, pair_indices=None, full_replay=True
    )
    return _read_json(result.manifest_path)


def _continuation_inputs(manifest: Mapping[str, Any], output: Path) -> Any:
    from scripts.autodl.run_comrecgc_standardized_continuation import (
        ContinuationInputs,
    )

    runtime = manifest["runtime_inputs"]
    source = manifest["source_authority"]
    checkpoint = _load_checkpoint(Path(source["failed_checkpoint_path"]))
    contract = ExternalDBSCANContract(**checkpoint["identity"]["contract"])
    return ContinuationInputs(
        dataset="aids",
        source_generation_root=Path(runtime["source_generation_root"]),
        upstream_root=Path(runtime["upstream_root"]),
        dataset_dir=Path(runtime["dataset_dir"]),
        source_csv=Path(runtime["source_csv"]),
        distance_checkpoint=Path(runtime["distance_checkpoint"]),
        dataset_csv=Path(runtime["dataset_csv"]),
        teacher_path=Path(runtime["teacher_path"]),
        molclr_root=Path(runtime["molclr_root"]),
        molclr_checkpoint=Path(runtime["molclr_checkpoint"]),
        thresholds_path=Path(runtime["thresholds_path"]),
        output_root=output,
        device="cpu",
        theta_star=runtime.get("theta_star"),
        cost_cap=runtime.get("cost_cap"),
        common_recourse_engine="external_memory_exact_v1",
        external_max_rss_gb=float(contract.max_rss_bytes) / 1024**3,
        external_query_block_size=contract.query_block_size,
        external_checkpoint_interval_blocks=contract.checkpoint_interval_blocks,
        external_dbscan_shortcut_mode=contract.shortcut_mode,
        external_shortcut_seed_count=contract.shortcut_seed_count,
        external_shortcut_failure_cap=contract.shortcut_failure_cap,
        external_shortcut_query_block_size=contract.shortcut_query_block_size,
        external_exact_fallback_max_samples=contract.exact_fallback_max_samples,
        external_summary_block_size=int(manifest["resources"]["block_size"]),
        external_pair_store_source_manifest=Path(source["pair_store_manifest_path"]),
        external_pair_store_source_checkpoint=None,
        external_pair_store_source_owner_root=Path(runtime["pair_store_owner_root"]),
        external_close_pair_view_manifest=Path(source["close_pair_manifest_path"]),
        expected_sklearn_version=contract.expected_sklearn_version,
        common_recourse_resume=True,
    )


def run_final_stage(
    *,
    controller_manifest: str | Path,
    adoption_gate: str | Path,
    subset_gate: str | Path,
    exact_gate: str | Path,
    downstream_gate: str | Path,
    output_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    manifest = load_bound_controller_manifest(controller_manifest)
    observed_environment = _require_cpu_stage_environment(manifest)
    observed_process_group = _require_controller_process_group()
    stage = _require_stage_paths(
        manifest,
        stage_id=FINAL_STAGE,
        output_dir=output_dir,
        bindings={
            "controller_manifest": controller_manifest,
            "adoption_gate": adoption_gate,
            "subset_gate": subset_gate,
            "exact_gate": exact_gate,
            "downstream_gate": downstream_gate,
        },
    )
    output = Path(output_dir).resolve(strict=True)
    owner_root = output / ".exact-recovery-final-owner"
    if owner_root.exists() and not resume:
        raise RecoveryStageError(
            "partial/completed final stage requires explicit resume authorization"
        )
    with _stage_writer(owner_root, resume=owner_root.exists()) as held:
        gates = [
            open_typed_recovery_gate(manifest, stage_id)
            for stage_id in (
                ADOPTION_STAGE,
                SUBSET_STAGE,
                EXACT_STAGE,
                DOWNSTREAM_STAGE,
            )
        ]
        terminal_path = Path(stage["terminal_path"])
        _reconcile_immutable_stage_publication(terminal_path)
        existing_binding: dict[str, Any] | None = None
        if terminal_path.exists():
            if not resume:
                raise RecoveryStageError(
                    "completed final stage requires explicit resume adoption"
                )
            existing_binding = _read_json(terminal_path)
        published_process_group = observed_process_group
        if existing_binding is not None:
            frozen_group = existing_binding.get("observed_process_group")
            if (
                not isinstance(frozen_group, Mapping)
                or int(frozen_group.get("runner_pid", -1)) <= 0
                or frozen_group.get("runner_pid")
                != frozen_group.get("process_group_id")
            ):
                raise RecoveryStageError(
                    "completed final stage process-group receipt changed"
                )
            published_process_group = dict(frozen_group)
        continuation_terminal = output / "_RUN_COMPLETE.json"
        if not continuation_terminal.exists():
            from scripts.autodl.run_comrecgc_standardized_continuation import (
                run_continuation,
            )

            run_continuation(_continuation_inputs(manifest, output))
        held.verify()
        continuation = _read_json(continuation_terminal)
        common_terminal = output / "common_recourse/_RUN_COMPLETE.json"
        from scripts.autodl.run_comrecgc_standardized_continuation import (
            _validate_common_recourse_completion,
        )

        _validate_common_recourse_completion(
            marker=common_terminal, terminal=_read_json(common_terminal)
        )
        freeze_path = output / "standardized/freeze_manifest.json"
        if (
            continuation.get("status") != "PASS"
            or continuation.get("run_complete") is not True
            or not (output / "PASS").is_file()
            or (output / "PASS").read_bytes() != b"PASS\n"
            or not freeze_path.is_file()
        ):
            raise RecoveryStageError("standardized continuation is not terminal")
        binding = {
            "schema_version": FINAL_STAGE_RECEIPT_SCHEMA,
            "status": "PASS",
            "run_complete": True,
            "dataset": "aids",
            "method": "COMRECGC",
            "controller_manifest_path": manifest["manifest_path"],
            "controller_manifest_sha256": manifest["manifest_sha256"],
            "typed_dependency_gate_sha256": [
                gate["gate_sha256"] for gate in gates
            ],
            "continuation_terminal_path": str(continuation_terminal),
            "continuation_terminal_sha256": sha256_file(continuation_terminal),
            "common_terminal_path": str(common_terminal),
            "common_terminal_sha256": sha256_file(common_terminal),
            "freeze_manifest_path": str(freeze_path),
            "freeze_manifest_sha256": sha256_file(freeze_path),
            "failed_evidence_adopted_as_pass": False,
            "seed_failure_scan_reexecuted": False,
            "component_downstream_full_replay_pass": True,
            "gpu_used": False,
            "observed_environment": observed_environment,
            "observed_process_group": published_process_group,
            "completed_at": (
                existing_binding.get("completed_at")
                if existing_binding is not None
                else _utc_now()
            ),
        }
        if (
            not isinstance(binding["completed_at"], str)
            or not binding["completed_at"]
        ):
            raise RecoveryStageError("final stage completion timestamp changed")
        held.verify()
        if existing_binding is not None:
            if existing_binding != binding:
                raise RecoveryStageError("final stage terminal identity changed")
        else:
            _write_new_json(terminal_path, binding)
        held.verify()
        return binding


__all__ = [
    "FINAL_STAGE_RECEIPT_SCHEMA",
    "RecoveryStageError",
    "run_downstream_stage",
    "run_exact_stage",
    "run_final_stage",
    "run_subset_stage",
]
