"""One locked pointer for sequential fast16 matrix authority publication.

The pointer is not a controller.  It only serializes the already existing
Taste and non-Taste append CLIs so that both always consume the same immutable
predecessor and cannot publish divergent authority chains.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from scripts.autodl.append_bace_gcf_matrix_authority import _verify_authority
from src.eval.four_by_four_registry import DATASETS, METHODS, PASS_STATUSES


POINTER_SCHEMA = "fast16_matrix_authority_pointer_v1"
DEFAULT_LOCK_PATH = Path(
    "/autodl-fs/data/counterfactual-subgraph-runtime/control/"
    "fast16_matrix_authority/publish.lock"
)
DEFAULT_STATE_PATH = Path(
    "/autodl-fs/data/counterfactual-subgraph-runtime/control/"
    "fast16_matrix_authority/state.json"
)


class MatrixAuthorityPointerError(RuntimeError):
    """The shared matrix pointer is malformed or an append escaped its chain."""


def _passing_cells(authority: Mapping[str, Any]) -> list[str]:
    values = {status.value for status in PASS_STATUSES}
    return [
        f"{dataset}/{method}"
        for dataset in DATASETS
        for method in METHODS
        if str(authority["rows"][(dataset, method)].get("status") or "") in values
    ]


def _read_state(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MatrixAuthorityPointerError(f"Authority pointer is not physical: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixAuthorityPointerError(f"Invalid authority pointer: {path}") from exc
    if not isinstance(value, dict):
        raise MatrixAuthorityPointerError("Authority pointer must be one JSON object")
    return dict(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_state(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise MatrixAuthorityPointerError(f"Authority pointer may not be a symlink: {path}")
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _state_for(authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": POINTER_SCHEMA,
        "latest_authority_root": str(authority["root"]),
        "latest_count": int(authority["complete"]),
        "latest_matrix_status_sha256": str(authority["matrix_sha256"]),
        "latest_combined_audit_sha256": str(authority["combined_sha256"]),
        "applied_cells": _passing_cells(authority),
    }


def _validate_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    if state.get("schema_version") != POINTER_SCHEMA:
        raise MatrixAuthorityPointerError("Authority pointer schema changed")
    root_raw = str(state.get("latest_authority_root") or "")
    if not root_raw or not Path(root_raw).expanduser().is_absolute():
        raise MatrixAuthorityPointerError("Authority pointer root is not absolute")
    authority = _verify_authority(root_raw)
    expected = _state_for(authority)
    if dict(state) != expected:
        changed = sorted(
            key
            for key in set(state) | set(expected)
            if state.get(key) != expected.get(key)
        )
        raise MatrixAuthorityPointerError(
            "Authority pointer does not match its hash-closed root: " + ", ".join(changed)
        )
    return authority


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise MatrixAuthorityPointerError(f"Authority lock may not be a symlink: {path}")
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    opened = os.fstat(descriptor)
    named = path.lstat()
    if (
        not stat.S_ISREG(opened.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        os.close(descriptor)
        raise MatrixAuthorityPointerError("Authority lock identity changed")
    return descriptor


def append_under_authority_pointer(
    *,
    state_path: str | Path,
    lock_path: str | Path,
    initial_authority_root: str | Path | None,
    requested_cells: Sequence[str],
    append: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run one append while holding the shared pointer lock, then advance it."""

    state = Path(state_path).expanduser()
    lock = Path(lock_path).expanduser()
    if not state.is_absolute() or not lock.is_absolute() or state == lock:
        raise MatrixAuthorityPointerError("Authority state/lock must be distinct absolute paths")
    if state.parent != lock.parent:
        raise MatrixAuthorityPointerError("Authority state and lock must share one directory")
    requested = list(requested_cells)
    valid_cells = {f"{dataset}/{method}" for dataset in DATASETS for method in METHODS}
    if not requested or len(requested) != len(set(requested)) or any(
        cell not in valid_cells for cell in requested
    ):
        raise MatrixAuthorityPointerError("Requested matrix cells are invalid or duplicated")
    state.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _open_lock(lock)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if state.exists() or state.is_symlink():
            before_state = _read_state(state)
            prior = _validate_state(state, before_state)
        else:
            if initial_authority_root is None:
                raise MatrixAuthorityPointerError(
                    "Missing pointer requires --initial-authority-root exactly once"
                )
            prior = _verify_authority(initial_authority_root)
            before_state = _state_for(prior)
            _atomic_state(state, before_state)
        prior_cells = set(_passing_cells(prior))
        overlap = prior_cells.intersection(requested)
        if overlap:
            raise MatrixAuthorityPointerError(
                "Pointer already contains requested cells: " + ", ".join(sorted(overlap))
            )
        raw_result = dict(append(Path(prior["root"])))
        output_raw = str(raw_result.get("output_root") or "")
        if not output_raw:
            raise MatrixAuthorityPointerError("Append result omitted output_root")
        current = _verify_authority(
            output_raw, expected_complete=int(prior["complete"]) + len(requested)
        )
        current_cells = set(_passing_cells(current))
        if current_cells != prior_cells.union(requested):
            raise MatrixAuthorityPointerError(
                "Append output changed cells outside the requested pointer transition"
            )
        after_state = _state_for(current)
        _atomic_state(state, after_state)
        reopened = _validate_state(state, _read_state(state))
        if reopened["root"] != current["root"]:
            raise MatrixAuthorityPointerError("Authority pointer changed on atomic reopen")
        return {
            **raw_result,
            "authority_state_path": str(state),
            "authority_lock_path": str(lock),
            "authority_pointer_before": before_state,
            "authority_pointer_after": after_state,
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def read_authority_pointer(
    *,
    state_path: str | Path,
    lock_path: str | Path,
    initial_authority_root: str | Path | None,
) -> dict[str, Any]:
    """Initialize or reopen the shared pointer while holding its exact lock."""

    state = Path(state_path).expanduser()
    lock = Path(lock_path).expanduser()
    if not state.is_absolute() or not lock.is_absolute() or state == lock:
        raise MatrixAuthorityPointerError("Authority state/lock must be distinct absolute paths")
    if state.parent != lock.parent:
        raise MatrixAuthorityPointerError("Authority state and lock must share one directory")
    state.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _open_lock(lock)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if state.exists() or state.is_symlink():
            payload = _read_state(state)
            _validate_state(state, payload)
            return payload
        if initial_authority_root is None:
            raise MatrixAuthorityPointerError(
                "Missing pointer requires --initial-authority-root exactly once"
            )
        authority = _verify_authority(initial_authority_root)
        payload = _state_for(authority)
        _atomic_state(state, payload)
        _validate_state(state, _read_state(state))
        return payload
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = [
    "DEFAULT_LOCK_PATH",
    "DEFAULT_STATE_PATH",
    "MatrixAuthorityPointerError",
    "POINTER_SCHEMA",
    "append_under_authority_pointer",
    "read_authority_pointer",
]
