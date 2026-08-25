"""Promote a failed adaptive selection into a fresh exact-recovery root.

The c766 DBSCAN root stopped before the ordinary anchor lower-bound scan
because its exact anchor graph was disconnected.  Its completed adaptive seed
and failure ledgers remain useful scientific evidence, but the failed root is
never resumed or written.  This module validates those ledgers against the
immutable vector source, copies only the three small selection arrays, rewrites
their root-local paths, and creates a new authenticated checkpoint from which
``fit_external_memory_dbscan`` can enter the exact component-recovery proof.

This is a project recovery adapter, not a generic way to turn a failed DBSCAN
run into PASS.  The resulting root still has to produce and reopen all exact
component certificates before it is scientifically usable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from .external_memory_dbscan import (
    ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT,
    ADAPTIVE_SELECTION_SCHEMA_VERSION,
    ExternalDBSCANContract,
    ExternalMemoryDBSCANError,
    ExternalDBSCANResult,
    SCHEMA_VERSION as DBSCAN_SCHEMA_VERSION,
    _checkpoint,
    _fit_all_core_one_component_shortcut,
    _load_checkpoint,
    _load_object,
    _load_progress_ledgers,
    _open_npy_memmap,
    _progress_checkpoint_extra,
    _progress_ledger_sha256,
    _sample_indices_sha256,
    _sha256_file,
    _stable_hash,
    _validate_adaptive_selection_manifest,
    _verify_source_identity,
    fit_external_memory_dbscan,
)


PROMOTION_SCHEMA_VERSION = "comrecgc_failed_selection_fresh_promotion_v1"
PROMOTION_MANIFEST_NAME = "failed_selection_fresh_promotion.json"
PROMOTION_CLAIM_NAME = "failed_selection_fresh_promotion_claim.json"
PROMOTION_CLAIM_SCHEMA_VERSION = "comrecgc_failed_selection_fresh_promotion_claim_v1"
RECOVERY_CHECKPOINT_PHASE = "shortcut_anchor_scan"
_SEED_PHASE = "adaptive_seed_scan"
_FAILURE_PHASE = "adaptive_failure_scan"
_ANCHOR_PHASE = "shortcut_anchor_scan"


class FailedSelectionRecoveryError(ExternalMemoryDBSCANError):
    """The failed selection cannot safely seed a fresh recovery root."""


@dataclass(frozen=True)
class FailedSelectionRecoverySource:
    checkpoint_path: Path
    checkpoint_sha256: str
    selection_manifest_path: Path
    selection_manifest_sha256: str
    failure_artifact_path: Path
    failure_artifact_sha256: str
    failure_indices_sha256: str
    anchor_indices_sha256: str
    anchor_rows_sha256: str


@dataclass(frozen=True)
class FailedSelectionPromotionResult:
    work_dir: Path
    checkpoint_path: Path
    selection_manifest_path: Path
    promotion_manifest_path: Path
    promotion_manifest_sha256: str
    source_checkpoint_sha256: str
    source_selection_manifest_sha256: str
    selection_manifest_sha256: str
    seed_failure_scan_reexecuted: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _physical_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise FailedSelectionRecoveryError(f"{label} may not be a symlink")
    resolved = path.expanduser().resolve(strict=True)
    value = resolved.stat()
    if not stat.S_ISREG(value.st_mode) or value.st_size <= 0:
        raise FailedSelectionRecoveryError(f"{label} is not a physical file")
    return resolved


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FailedSelectionRecoveryError(
                f"immutable promotion output already exists: {path}"
            ) from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_npy(path: Path, values: np.ndarray) -> str:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FailedSelectionRecoveryError(
                f"immutable promotion array already exists: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def _ensure_exact_npy(
    path: Path, values: np.ndarray, *, expected_sha256: str, label: str
) -> str:
    if path.exists() or path.is_symlink():
        physical = _physical_file(path, label=label)
        _validate_hash(physical, expected_sha256, label=label)
        actual = np.load(physical, allow_pickle=False)
        expected = np.asarray(values)
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise FailedSelectionRecoveryError(f"{label} content changed")
        return expected_sha256
    observed = _write_new_npy(path, values)
    if observed != expected_sha256:
        raise FailedSelectionRecoveryError(f"{label} changed during promotion")
    return observed


def _ensure_exact_json(path: Path, payload: Mapping[str, Any], *, label: str) -> str:
    if path.exists() or path.is_symlink():
        physical = _physical_file(path, label=label)
        if _load_object(physical) != dict(payload):
            raise FailedSelectionRecoveryError(f"{label} content changed")
    else:
        _write_new_json(path, payload)
    return _sha256_file(path)


@contextmanager
def _promotion_writer(root: Path) -> Iterator[None]:
    parent = root.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise FailedSelectionRecoveryError("fresh promotion parent is invalid")
    parent_before = parent.stat()
    lock_path = parent / f".{root.name}.failed-selection-promotion.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise FailedSelectionRecoveryError(
                "another failed-selection promotion writer is active"
            ) from exc
        opened = os.fstat(descriptor)

        def verify() -> None:
            current_parent = parent.stat()
            current_lock = lock_path.lstat()
            if (
                parent.is_symlink()
                or lock_path.is_symlink()
                or (current_parent.st_dev, current_parent.st_ino)
                != (parent_before.st_dev, parent_before.st_ino)
                or (opened.st_dev, opened.st_ino)
                != (current_lock.st_dev, current_lock.st_ino)
                or (opened.st_dev, opened.st_ino)
                != (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino)
            ):
                raise FailedSelectionRecoveryError(
                    "failed-selection promotion writer identity changed"
                )

        verify()
        yield
        verify()
    finally:
        os.close(descriptor)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _validate_hash(path: Path, expected: str, *, label: str) -> None:
    if not _is_sha256(expected) or _sha256_file(path) != expected:
        raise FailedSelectionRecoveryError(f"{label} SHA256 mismatch")


def _frozen_neighbor_backend(
    *, identity: Mapping[str, Any], contract: ExternalDBSCANContract
) -> tuple[str, str]:
    """Reopen the authenticated backend identity without fitting all N rows."""

    try:
        import sklearn
    except Exception as exc:  # pragma: no cover - dependency gate
        raise FailedSelectionRecoveryError(
            "failed-selection recovery requires scikit-learn"
        ) from exc
    version = str(sklearn.__version__)
    fit_method = str(identity.get("nearest_neighbors_fit_method") or "")
    if (
        version != contract.expected_sklearn_version
        or version != identity.get("sklearn_version")
        or fit_method != "brute"
    ):
        raise FailedSelectionRecoveryError(
            "failed checkpoint sklearn identity changed"
        )
    return version, fit_method


def _source_root(source: FailedSelectionRecoverySource) -> Path:
    paths = (
        source.checkpoint_path,
        source.selection_manifest_path,
        source.failure_artifact_path,
    )
    resolved = [_physical_file(Path(path), label="failed-selection source") for path in paths]
    parents = {path.parent for path in resolved}
    if len(parents) != 1:
        raise FailedSelectionRecoveryError("failed-selection evidence spans source roots")
    return resolved[0].parent


def _validate_failed_checkpoint(
    *,
    vectors_path: Path,
    source: FailedSelectionRecoverySource,
    contract: ExternalDBSCANContract,
    expected_vectors_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], np.ndarray]:
    contract.validate()
    if contract.shortcut_mode != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT:
        raise FailedSelectionRecoveryError("recovery requires the adaptive shortcut")
    root = _source_root(source)
    checkpoint_path = _physical_file(source.checkpoint_path, label="source checkpoint")
    selection_path = _physical_file(
        source.selection_manifest_path, label="source selection"
    )
    failure_path = _physical_file(source.failure_artifact_path, label="source failure")
    _validate_hash(checkpoint_path, source.checkpoint_sha256, label="source checkpoint")
    _validate_hash(
        selection_path, source.selection_manifest_sha256, label="source selection"
    )
    _validate_hash(failure_path, source.failure_artifact_sha256, label="source failure")
    checkpoint = _load_checkpoint(checkpoint_path)
    identity = checkpoint.get("identity")
    if (
        checkpoint.get("schema_version") != DBSCAN_SCHEMA_VERSION
        or checkpoint.get("phase") != "shortcut_blocked"
        or int(checkpoint.get("next_offset", -1)) != 0
        or checkpoint.get("identity_sha256") != _stable_hash(identity)
        or not isinstance(identity, Mapping)
        or identity.get("contract") != asdict(contract)
        or identity.get("vectors_path") != str(vectors_path)
        or identity.get("vectors_sha256") != expected_vectors_sha256
        or identity.get("shortcut_contract", {}).get("mode")
        != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT
        or checkpoint.get("adaptive_selection_manifest_path") != str(selection_path)
        or checkpoint.get("adaptive_selection_manifest_sha256")
        != source.selection_manifest_sha256
        or checkpoint.get("shortcut_failure_path") != str(failure_path)
        or checkpoint.get("shortcut_failure_sha256") != source.failure_artifact_sha256
        or checkpoint.get("shortcut_approximation_used") is not False
    ):
        raise FailedSelectionRecoveryError("failed checkpoint contract changed")
    vector_stat = identity.get("vectors_stat_identity")
    if not isinstance(vector_stat, Mapping):
        raise FailedSelectionRecoveryError("failed checkpoint vector stat is absent")
    _verify_source_identity(
        vectors_path,
        expected_sha256=expected_vectors_sha256,
        expected_stat=vector_stat,
        phase="failed-selection promotion",
    )
    vectors = _open_npy_memmap(vectors_path, mode="r")
    if list(vectors.shape) != identity.get("vectors_shape") or str(vectors.dtype) != identity.get(
        "vectors_dtype"
    ):
        raise FailedSelectionRecoveryError("failed checkpoint vector schema changed")
    _frozen_neighbor_backend(identity=identity, contract=contract)
    failure = _load_object(failure_path)
    if (
        failure.get("status") != "INCONCLUSIVE"
        or failure.get("reason") != "anchor_epsilon_graph_disconnected"
        or failure.get("fallback_allowed") is not False
        or failure.get("approximation_used") is not False
        or failure.get("scientific_identity_sha256") != _stable_hash(identity)
    ):
        raise FailedSelectionRecoveryError("source failure is not disconnected exact evidence")
    n_samples = int(vectors.shape[0])
    ledgers = _load_progress_ledgers(
        checkpoint, identity=identity, num_samples=n_samples
    )
    if set(ledgers) != {_SEED_PHASE, _FAILURE_PHASE, _ANCHOR_PHASE}:
        raise FailedSelectionRecoveryError("failed checkpoint ledger phase set changed")
    for phase in (_SEED_PHASE, _FAILURE_PHASE):
        if (
            ledgers[phase].get("complete") is not True
            or int(ledgers[phase].get("committed_offset", -1)) != n_samples
        ):
            raise FailedSelectionRecoveryError(f"{phase} ledger is incomplete")
    anchor_ledger = ledgers[_ANCHOR_PHASE]
    if (
        anchor_ledger.get("complete") is not False
        or int(anchor_ledger.get("committed_offset", -1)) != 0
        or anchor_ledger.get("entries") != []
        or anchor_ledger.get("result") is not None
    ):
        raise FailedSelectionRecoveryError("failed anchor ledger was unexpectedly advanced")
    anchors, selection = _validate_adaptive_selection_manifest(
        path=selection_path,
        expected_sha256=source.selection_manifest_sha256,
        root=root,
        identity=identity,
        progress_ledgers=ledgers,
    )
    selected = selection["selection_identity"]
    for field, expected in (
        ("failure_indices_sha256", source.failure_indices_sha256),
        ("anchor_indices_sha256", source.anchor_indices_sha256),
        ("anchor_rows_sha256", source.anchor_rows_sha256),
    ):
        if selected.get(field) != expected:
            raise FailedSelectionRecoveryError(f"source selection changed: {field}")
    return dict(identity), ledgers, dict(selection), np.asarray(anchors, dtype=np.intp)


def _validate_promoted_selection(
    *,
    root: Path,
    identity: Mapping[str, Any],
    ledgers: Mapping[str, Mapping[str, Any]],
    expected_manifest_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _validate_adaptive_selection_manifest(
        path=root / "adaptive_anchor_selection.json",
        expected_sha256=expected_manifest_sha256,
        root=root,
        identity=identity,
        progress_ledgers=ledgers,
    )


def _promote_failed_adaptive_selection_for_component_recovery_locked(
    *,
    vectors_path: str | Path,
    work_dir: str | Path,
    source: FailedSelectionRecoverySource,
    contract: ExternalDBSCANContract,
    expected_vectors_sha256: str,
    adoption_receipt_path: str | Path,
    adoption_receipt_sha256: str,
    source_authority_sha256: str,
    resume: bool = False,
) -> FailedSelectionPromotionResult:
    """Create or reopen a fresh checkpoint without rerunning seed/failure scans."""

    vector_path = _physical_file(Path(vectors_path), label="recovery vectors")
    receipt_path = _physical_file(
        Path(adoption_receipt_path), label="typed adoption receipt"
    )
    _validate_hash(receipt_path, adoption_receipt_sha256, label="adoption receipt")
    if not _is_sha256(source_authority_sha256):
        raise FailedSelectionRecoveryError("source authority SHA256 is invalid")
    root = Path(work_dir).expanduser().resolve(strict=False)
    manifest_path = root / PROMOTION_MANIFEST_NAME
    claim_path = root / PROMOTION_CLAIM_NAME
    checkpoint_path = root / "checkpoint.json"
    selection_path = root / "adaptive_anchor_selection.json"
    identity, ledgers, selection, anchors = _validate_failed_checkpoint(
        vectors_path=vector_path,
        source=source,
        contract=contract,
        expected_vectors_sha256=expected_vectors_sha256,
    )
    selected = dict(selection["selection_identity"])
    source_checkpoint_path = _physical_file(
        source.checkpoint_path, label="source checkpoint"
    )
    source_selection_path = _physical_file(
        source.selection_manifest_path, label="source selection"
    )
    source_failure_path = _physical_file(
        source.failure_artifact_path, label="source failure"
    )
    source_failure_indices = _physical_file(
        Path(str(selected["failure_indices_path"])), label="failure indices"
    )
    source_anchor_indices = _physical_file(
        Path(str(selected["anchor_indices_path"])), label="anchor indices"
    )
    source_anchor_rows = _physical_file(
        Path(str(selected["anchor_rows_path"])), label="anchor rows"
    )
    failure_indices = np.load(source_failure_indices, allow_pickle=False)
    anchor_indices = np.load(source_anchor_indices, allow_pickle=False)
    anchor_rows = np.load(source_anchor_rows, allow_pickle=False)
    if not np.array_equal(anchor_indices, anchors):
        raise FailedSelectionRecoveryError("validated anchor array changed before copy")
    claim_identity = {
        "schema_version": PROMOTION_CLAIM_SCHEMA_VERSION,
        "work_dir": str(root),
        "vectors_path": str(vector_path),
        "vectors_sha256": expected_vectors_sha256,
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "source_selection_manifest_path": str(source_selection_path),
        "source_selection_manifest_sha256": source.selection_manifest_sha256,
        "source_failure_artifact_path": str(source_failure_path),
        "source_failure_artifact_sha256": source.failure_artifact_sha256,
        "adoption_receipt_path": str(receipt_path),
        "adoption_receipt_sha256": adoption_receipt_sha256,
        "source_authority_sha256": source_authority_sha256,
        "contract": asdict(contract),
    }
    if root.exists() or root.is_symlink():
        if not resume:
            raise FailedSelectionRecoveryError(
                "fresh promotion work directory already exists"
            )
        if root.is_symlink() or not root.is_dir():
            raise FailedSelectionRecoveryError("promotion root is not physical")
    else:
        root.mkdir(mode=0o755)
    if claim_path.exists() or claim_path.is_symlink():
        claim = _load_object(_physical_file(claim_path, label="promotion claim"))
        claim_comparable = dict(claim)
        created_at = claim_comparable.pop("created_at", None)
        if (
            claim_comparable != claim_identity
            or not isinstance(created_at, str)
            or not created_at
        ):
            raise FailedSelectionRecoveryError("promotion claim identity changed")
    else:
        if any(root.iterdir()):
            raise FailedSelectionRecoveryError(
                "partial promotion root lacks its immutable claim"
            )
        created_at = _utc_now()
        claim = {**claim_identity, "created_at": created_at}
        _write_new_json(claim_path, claim)
    claim_sha = _sha256_file(claim_path)
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = _load_object(
            _physical_file(manifest_path, label="promotion manifest")
        )
        static_expected = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "status": "READY_FOR_EXACT_COMPONENT_RECOVERY",
            "work_dir": str(root),
            "vectors_path": str(vector_path),
            "vectors_sha256": expected_vectors_sha256,
            "source_checkpoint_path": str(source_checkpoint_path),
            "source_checkpoint_sha256": source.checkpoint_sha256,
            "source_selection_manifest_path": str(source_selection_path),
            "source_selection_manifest_sha256": source.selection_manifest_sha256,
            "source_failure_artifact_path": str(source_failure_path),
            "source_failure_artifact_sha256": source.failure_artifact_sha256,
            "adoption_receipt_path": str(receipt_path),
            "adoption_receipt_sha256": adoption_receipt_sha256,
            "source_authority_sha256": source_authority_sha256,
            "promotion_claim_path": str(claim_path),
            "promotion_claim_sha256": claim_sha,
            "selection_manifest_path": str(selection_path),
            "seed_progress_ledger_sha256": _progress_ledger_sha256(
                ledgers[_SEED_PHASE]
            ),
            "failure_progress_ledger_sha256": _progress_ledger_sha256(
                ledgers[_FAILURE_PHASE]
            ),
            "seed_failure_scan_reexecuted": False,
            "source_root_written": False,
            "source_large_arrays_copied": False,
            "fresh_checkpoint_rebuilt": True,
            "failed_terminal_adopted_as_pass": False,
            "recovery_only": True,
            "approximation_used": False,
            "created_at": created_at,
        }
        selection_sha = manifest.get("selection_manifest_sha256")
        if (
            not _is_sha256(selection_sha)
            or {
                key: manifest.get(key)
                for key in static_expected
            }
            != static_expected
            or set(manifest) != {*static_expected, "selection_manifest_sha256"}
        ):
            raise FailedSelectionRecoveryError("promotion terminal identity changed")
        state = _load_checkpoint(
            _physical_file(checkpoint_path, label="fresh checkpoint")
        )
        current_identity = state.get("identity")
        if current_identity != identity:
            raise FailedSelectionRecoveryError("fresh recovery identity changed")
        current_ledgers = _load_progress_ledgers(
            state,
            identity=identity,
            num_samples=int(identity["vectors_shape"][0]),
        )
        _validate_promoted_selection(
            root=root,
            identity=identity,
            ledgers=current_ledgers,
            expected_manifest_sha256=str(selection_sha),
        )
        return FailedSelectionPromotionResult(
            work_dir=root,
            checkpoint_path=checkpoint_path,
            selection_manifest_path=selection_path,
            promotion_manifest_path=manifest_path,
            promotion_manifest_sha256=_sha256_file(manifest_path),
            source_checkpoint_sha256=source.checkpoint_sha256,
            source_selection_manifest_sha256=source.selection_manifest_sha256,
            selection_manifest_sha256=str(selection_sha),
            seed_failure_scan_reexecuted=False,
        )
    target_failure = root / "adaptive_first_pass_failure_indices.npy"
    target_anchor_indices = root / "shortcut_anchor_indices.npy"
    target_anchor_rows = root / "adaptive_selected_anchor_rows.npy"
    copied = {
        "failure_indices_sha256": _ensure_exact_npy(
            target_failure,
            failure_indices,
            expected_sha256=source.failure_indices_sha256,
            label="promoted failure indices",
        ),
        "anchor_indices_sha256": _ensure_exact_npy(
            target_anchor_indices,
            anchor_indices,
            expected_sha256=source.anchor_indices_sha256,
            label="promoted anchor indices",
        ),
        "anchor_rows_sha256": _ensure_exact_npy(
            target_anchor_rows,
            anchor_rows,
            expected_sha256=source.anchor_rows_sha256,
            label="promoted anchor rows",
        ),
    }
    for field, expected in (
        ("failure_indices_sha256", source.failure_indices_sha256),
        ("anchor_indices_sha256", source.anchor_indices_sha256),
        ("anchor_rows_sha256", source.anchor_rows_sha256),
    ):
        if copied[field] != expected:
            raise FailedSelectionRecoveryError(f"promoted array changed: {field}")
    selected.update(
        {
            "failure_indices_path": str(target_failure),
            "anchor_indices_path": str(target_anchor_indices),
            "anchor_rows_path": str(target_anchor_rows),
        }
    )
    promoted_selection = {
        "schema_version": ADAPTIVE_SELECTION_SCHEMA_VERSION,
        "run_complete": True,
        "selection_identity": selected,
        "selection_identity_sha256": _stable_hash(selected),
        "completed_at": created_at,
    }
    selection_sha = _ensure_exact_json(
        selection_path, promoted_selection, label="promoted selection manifest"
    )
    _validate_promoted_selection(
        root=root,
        identity=identity,
        ledgers=ledgers,
        expected_manifest_sha256=selection_sha,
    )
    extra = _progress_checkpoint_extra(ledgers, identity=identity)
    extra.update(
        {
            "adaptive_selection_manifest_path": str(selection_path),
            "adaptive_selection_manifest_sha256": selection_sha,
            "selected_anchor_indices_sha256": selected[
                "selected_anchor_indices_sha256"
            ],
            "failed_selection_promotion_source_sha256": _stable_hash(
                {
                    "source_checkpoint_sha256": source.checkpoint_sha256,
                    "source_selection_manifest_sha256": source.selection_manifest_sha256,
                    "adoption_receipt_sha256": adoption_receipt_sha256,
                    "source_authority_sha256": source_authority_sha256,
                }
            ),
        }
    )
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        current = _load_checkpoint(
            _physical_file(checkpoint_path, label="fresh checkpoint")
        )
        if (
            current.get("phase") != RECOVERY_CHECKPOINT_PHASE
            or int(current.get("next_offset", -1)) != 0
            or current.get("identity") != identity
            or current.get("identity_sha256") != _stable_hash(identity)
            or any(current.get(key) != value for key, value in extra.items())
        ):
            raise FailedSelectionRecoveryError(
                "partial fresh checkpoint identity changed"
            )
    else:
        _checkpoint(
            checkpoint_path,
            identity=identity,
            phase=RECOVERY_CHECKPOINT_PHASE,
            # Seed/failure ledgers each cover N rows, while this fresh primary
            # component ledger is deliberately empty. Keep checkpoint offset
            # equal to that active ledger's committed offset.
            next_offset=0,
            peak_rss_bytes=_rss_bytes(),
            extra=extra,
        )
    promotion = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "READY_FOR_EXACT_COMPONENT_RECOVERY",
        "work_dir": str(root),
        "vectors_path": str(vector_path),
        "vectors_sha256": expected_vectors_sha256,
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "source_selection_manifest_path": str(source_selection_path),
        "source_selection_manifest_sha256": source.selection_manifest_sha256,
        "source_failure_artifact_path": str(source_failure_path),
        "source_failure_artifact_sha256": source.failure_artifact_sha256,
        "adoption_receipt_path": str(receipt_path),
        "adoption_receipt_sha256": adoption_receipt_sha256,
        "source_authority_sha256": source_authority_sha256,
        "promotion_claim_path": str(claim_path),
        "promotion_claim_sha256": claim_sha,
        "selection_manifest_path": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "seed_progress_ledger_sha256": _progress_ledger_sha256(ledgers[_SEED_PHASE]),
        "failure_progress_ledger_sha256": _progress_ledger_sha256(ledgers[_FAILURE_PHASE]),
        "seed_failure_scan_reexecuted": False,
        "source_root_written": False,
        "source_large_arrays_copied": False,
        "fresh_checkpoint_rebuilt": True,
        "failed_terminal_adopted_as_pass": False,
        "recovery_only": True,
        "approximation_used": False,
        "created_at": created_at,
    }
    _ensure_exact_json(manifest_path, promotion, label="promotion manifest")
    # A final reopen proves the fresh selection paths and source vectors still
    # match after all small artifacts have been published.
    current = _load_checkpoint(checkpoint_path)
    reopened_ledgers = _load_progress_ledgers(
        current, identity=identity, num_samples=int(identity["vectors_shape"][0])
    )
    _validate_promoted_selection(
        root=root,
        identity=identity,
        ledgers=reopened_ledgers,
        expected_manifest_sha256=selection_sha,
    )
    _verify_source_identity(
        vector_path,
        expected_sha256=expected_vectors_sha256,
        expected_stat=identity["vectors_stat_identity"],
        phase="fresh promotion terminal reopen",
    )
    return FailedSelectionPromotionResult(
        work_dir=root,
        checkpoint_path=checkpoint_path,
        selection_manifest_path=selection_path,
        promotion_manifest_path=manifest_path,
        promotion_manifest_sha256=_sha256_file(manifest_path),
        source_checkpoint_sha256=source.checkpoint_sha256,
        source_selection_manifest_sha256=source.selection_manifest_sha256,
        selection_manifest_sha256=selection_sha,
        seed_failure_scan_reexecuted=False,
    )


def promote_failed_adaptive_selection_for_component_recovery(
    *,
    vectors_path: str | Path,
    work_dir: str | Path,
    source: FailedSelectionRecoverySource,
    contract: ExternalDBSCANContract,
    expected_vectors_sha256: str,
    adoption_receipt_path: str | Path,
    adoption_receipt_sha256: str,
    source_authority_sha256: str,
    resume: bool = False,
) -> FailedSelectionPromotionResult:
    """Single-writer public wrapper for restartable small-evidence promotion."""

    root = Path(work_dir).expanduser().resolve(strict=False)
    with _promotion_writer(root):
        return _promote_failed_adaptive_selection_for_component_recovery_locked(
            vectors_path=vectors_path,
            work_dir=root,
            source=source,
            contract=contract,
            expected_vectors_sha256=expected_vectors_sha256,
            adoption_receipt_path=adoption_receipt_path,
            adoption_receipt_sha256=adoption_receipt_sha256,
            source_authority_sha256=source_authority_sha256,
            resume=resume,
        )


def fit_promoted_failed_selection_component_recovery(
    *,
    vectors_path: str | Path,
    work_dir: str | Path,
    contract: ExternalDBSCANContract,
    expected_vectors_sha256: str,
) -> ExternalDBSCANResult:
    """Continue only the fresh component proof, never the seed/failure scan.

    Calling :func:`fit_external_memory_dbscan` on a promoted nonterminal
    checkpoint would intentionally replay every old committed seed/failure
    block.  That is correct for an ordinary same-root resume, but violates this
    recovery route's no-rescan contract.  This public adapter therefore
    validates the promoted selection and calls the already-reviewed shortcut
    continuation directly.  Component-recovery prefixes created in this fresh
    root retain their normal deterministic replay on restart.
    """

    contract.validate()
    if contract.shortcut_mode != ADAPTIVE_ALL_CORE_ONE_COMPONENT_SHORTCUT:
        raise FailedSelectionRecoveryError("promoted recovery requires adaptive mode")
    source = _physical_file(Path(vectors_path), label="recovery vectors")
    root = Path(work_dir).expanduser().resolve(strict=True)
    promotion_path = _physical_file(
        root / PROMOTION_MANIFEST_NAME, label="promotion manifest"
    )
    promotion = _load_object(promotion_path)
    state_path = _physical_file(root / "checkpoint.json", label="fresh checkpoint")
    final_path = root / "run_manifest.json"
    if final_path.exists() or final_path.is_symlink():
        # Terminal reopen is safe: the generic entrypoint validates the final
        # certificates before it considers any mutable checkpoint.
        return fit_external_memory_dbscan(
            vectors_path=source,
            work_dir=root,
            contract=contract,
            expected_vectors_sha256=expected_vectors_sha256,
            resume=True,
        )
    state = _load_checkpoint(state_path)
    identity = state.get("identity")
    if (
        promotion.get("schema_version") != PROMOTION_SCHEMA_VERSION
        or promotion.get("status") != "READY_FOR_EXACT_COMPONENT_RECOVERY"
        or promotion.get("seed_failure_scan_reexecuted") is not False
        or promotion.get("failed_terminal_adopted_as_pass") is not False
        or promotion.get("selection_manifest_path")
        != str(root / "adaptive_anchor_selection.json")
        or promotion.get("vectors_path") != str(source)
        or promotion.get("vectors_sha256") != expected_vectors_sha256
        or not isinstance(identity, Mapping)
        or identity.get("vectors_path") != str(source)
        or identity.get("vectors_sha256") != expected_vectors_sha256
        or identity.get("contract") != asdict(contract)
        or state.get("identity_sha256") != _stable_hash(identity)
    ):
        raise FailedSelectionRecoveryError("fresh promotion identity changed")
    _verify_source_identity(
        source,
        expected_sha256=expected_vectors_sha256,
        expected_stat=identity["vectors_stat_identity"],
        phase="promoted component recovery",
    )
    vectors = _open_npy_memmap(source, mode="r")
    ledgers = _load_progress_ledgers(
        state, identity=identity, num_samples=int(vectors.shape[0])
    )
    for phase, expected_field in (
        (_SEED_PHASE, "seed_progress_ledger_sha256"),
        (_FAILURE_PHASE, "failure_progress_ledger_sha256"),
    ):
        if (
            ledgers.get(phase, {}).get("complete") is not True
            or int(ledgers[phase].get("committed_offset", -1))
            != int(vectors.shape[0])
            or _progress_ledger_sha256(ledgers[phase])
            != promotion.get(expected_field)
        ):
            raise FailedSelectionRecoveryError(f"promoted {phase} ledger changed")
    anchors, selection = _validate_promoted_selection(
        root=root,
        identity=identity,
        ledgers=ledgers,
        expected_manifest_sha256=str(promotion["selection_manifest_sha256"]),
    )
    sklearn_version, fit_method = _frozen_neighbor_backend(
        identity=identity, contract=contract
    )
    result = _fit_all_core_one_component_shortcut(
        vectors=vectors,
        root=root,
        state_path=state_path,
        final_manifest_path=final_path,
        identity=identity,
        contract=contract,
        full_fit_method=fit_method,
        sklearn_version=sklearn_version,
        peak_rss_bytes=max(int(state.get("peak_rss_bytes", 0)), _rss_bytes()),
        anchor_indices_override=anchors,
        adaptive_selection_manifest=selection,
        # Never replay the adopted seed/failure prefix.  Fresh component
        # ledgers are independently replayed inside the component proof.
        resume_replay_required=False,
    )
    if result is None:
        raise FailedSelectionRecoveryError(
            "promoted component recovery fell through to a forbidden general route"
        )
    return result


__all__ = [
    "FailedSelectionPromotionResult",
    "FailedSelectionRecoveryError",
    "FailedSelectionRecoverySource",
    "PROMOTION_MANIFEST_NAME",
    "PROMOTION_SCHEMA_VERSION",
    "fit_promoted_failed_selection_component_recovery",
    "promote_failed_adaptive_selection_for_component_recovery",
]
