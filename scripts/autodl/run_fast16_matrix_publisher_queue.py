#!/usr/bin/env python3
"""Persistently append known full-cell terminals through one fast16 pointer.

This process owns no scientific task and no GPU.  It waits for PASS-last roots
or a future root locator, then invokes the existing strict dataset-specific
publisher for one cell at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.fast16_matrix_authority_pointer import (  # noqa: E402
    DEFAULT_LOCK_PATH,
    DEFAULT_STATE_PATH,
    append_under_authority_pointer,
    read_authority_pointer,
)
from src.eval.non_taste_matrix_append import TARGETS, append_non_taste_matrix_cell  # noqa: E402
from src.eval.tastemolnet_matrix_append import (  # noqa: E402
    METHOD_CONTRACTS,
    append_tastemolnet_cells,
)


QUEUE_SCHEMA = "fast16_matrix_publisher_queue_v1"
LOCATOR_SCHEMA = "fast16_matrix_cell_root_locator_v1"
HEARTBEAT_SCHEMA = "fast16_matrix_publisher_heartbeat_v1"


class Fast16PublisherQueueError(RuntimeError):
    """The fixed publisher queue is malformed."""


def _absolute(value: Any, *, label: str, must_exist: bool = False) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise Fast16PublisherQueueError(f"{label} must be an absolute non-symlink path")
    try:
        return path.resolve(strict=must_exist)
    except OSError as exc:
        raise Fast16PublisherQueueError(f"{label} is absent: {path}") from exc


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise Fast16PublisherQueueError(f"{label} is not a physical nonempty file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Fast16PublisherQueueError(f"Invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise Fast16PublisherQueueError(f"{label} must be one JSON object")
    return dict(value)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise Fast16PublisherQueueError(f"Heartbeat may not be a symlink: {path}")
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
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _load_queue(path: Path) -> dict[str, Any]:
    payload = _json(path, label="publisher queue manifest")
    if payload.get("schema_version") != QUEUE_SCHEMA:
        raise Fast16PublisherQueueError("Publisher queue schema changed")
    initial = _absolute(
        payload.get("initial_authority_root"), label="initial_authority_root", must_exist=True
    )
    state = _absolute(
        payload.get("authority_state_path", DEFAULT_STATE_PATH),
        label="authority_state_path",
    )
    lock = _absolute(
        payload.get("authority_lock_path", DEFAULT_LOCK_PATH),
        label="authority_lock_path",
    )
    try:
        poll = int(payload.get("poll_seconds", 60))
    except (TypeError, ValueError) as exc:
        raise Fast16PublisherQueueError("poll_seconds is invalid") from exc
    if not 5 <= poll <= 3600:
        raise Fast16PublisherQueueError("poll_seconds must be in [5, 3600]")
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise Fast16PublisherQueueError("Publisher queue has no cells")
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping):
            raise Fast16PublisherQueueError(f"cells[{index}] is not an object")
        dataset = str(raw.get("dataset") or "")
        method = str(raw.get("method") or "")
        cell_id = f"{dataset}/{method}"
        supported = dataset == "TasteMolNet" and method in METHOD_CONTRACTS
        supported = supported or (dataset, method) in TARGETS
        if not supported or cell_id in seen:
            raise Fast16PublisherQueueError(f"Unsupported/duplicate queue cell: {cell_id}")
        seen.add(cell_id)
        fixed = raw.get("terminal_root")
        locator = raw.get("terminal_root_locator")
        if (fixed is None) == (locator is None):
            raise Fast16PublisherQueueError(
                f"{cell_id} requires exactly one terminal_root or terminal_root_locator"
            )
        cell = {
            "dataset": dataset,
            "method": method,
            "cell_id": cell_id,
            "terminal_root": (
                str(_absolute(fixed, label=f"{cell_id}.terminal_root"))
                if fixed is not None
                else None
            ),
            "terminal_root_locator": (
                str(_absolute(locator, label=f"{cell_id}.terminal_root_locator"))
                if locator is not None
                else None
            ),
            "output_root": str(
                _absolute(raw.get("output_root"), label=f"{cell_id}.output_root")
            ),
            "aids_controller_manifest": (
                str(
                    _absolute(
                        raw.get("aids_controller_manifest"),
                        label=f"{cell_id}.aids_controller_manifest",
                    )
                )
                if raw.get("aids_controller_manifest") is not None
                else None
            ),
        }
        if dataset == "AIDS" and cell["aids_controller_manifest"] is None:
            raise Fast16PublisherQueueError("AIDS queue cell lacks controller manifest")
        if dataset != "AIDS" and cell["aids_controller_manifest"] is not None:
            raise Fast16PublisherQueueError("AIDS controller manifest is AIDS-only")
        cells.append(cell)
    taste = payload.get("taste")
    if any(cell["dataset"] == "TasteMolNet" for cell in cells):
        if not isinstance(taste, Mapping):
            raise Fast16PublisherQueueError("Taste queue cells require one taste binding")
        taste_binding = {
            key: str(_absolute(taste.get(key), label=f"taste.{key}"))
            for key in (
                "t3_root",
                "policy_path",
                "policy_receipt",
                "prepared_root",
                "graph_cache_root",
            )
        }
    else:
        taste_binding = {}
    return {
        "manifest_path": str(path),
        "manifest_sha256": _sha(path),
        "initial_authority_root": str(initial),
        "authority_state_path": str(state),
        "authority_lock_path": str(lock),
        "poll_seconds": poll,
        "cells": cells,
        "taste": taste_binding,
    }


def _locator_root(cell: Mapping[str, Any]) -> tuple[Path | None, str]:
    fixed = cell.get("terminal_root")
    if fixed:
        root = Path(str(fixed))
        return (root.resolve(strict=True), "FIXED") if root.is_dir() else (None, "ROOT_ABSENT")
    locator = Path(str(cell["terminal_root_locator"]))
    if not locator.exists():
        return None, "LOCATOR_ABSENT"
    payload = _json(locator, label=f"{cell['cell_id']} root locator")
    if (
        payload.get("schema_version") != LOCATOR_SCHEMA
        or payload.get("status") != "READY"
        or payload.get("dataset") != cell["dataset"]
        or payload.get("method") != cell["method"]
    ):
        raise Fast16PublisherQueueError(f"{cell['cell_id']} root locator changed")
    root = _absolute(
        payload.get("terminal_root"), label=f"{cell['cell_id']} locator terminal", must_exist=True
    )
    if not root.is_dir():
        raise Fast16PublisherQueueError(f"{cell['cell_id']} locator root is not a directory")
    return root, f"LOCATOR:{_sha(locator)}"


def _terminal_fingerprint(cell: Mapping[str, Any], root: Path) -> tuple[str, str] | None:
    contract = METHOD_CONTRACTS.get(str(cell["method"])) if cell["dataset"] == "TasteMolNet" else None
    expected = contract.pass_payload if contract is not None else b"PASS\n"
    marker = root / "PASS"
    if marker.is_symlink() or not marker.is_file():
        return None
    try:
        if marker.read_bytes() != expected:
            return None
        stats = []
        for name in ("PASS", "run_manifest.json", "final_artifact_audit.json"):
            path = root / name
            if path.is_file() and not path.is_symlink():
                value = path.stat()
                stats.append((name, value.st_size, value.st_mtime_ns, value.st_ctime_ns))
        return str(root), hashlib.sha256(repr(stats).encode("utf-8")).hexdigest()
    except OSError:
        return None


def _append_cell(queue: Mapping[str, Any], cell: Mapping[str, Any], root: Path) -> dict[str, Any]:
    state = queue["authority_state_path"]
    lock = queue["authority_lock_path"]
    cell_id = str(cell["cell_id"])

    if cell["dataset"] == "TasteMolNet":
        taste = queue["taste"]

        def _append(prior: Path) -> Mapping[str, Any]:
            return append_tastemolnet_cells(
                prior_authority_root=prior,
                taste_cells={str(cell["method"]): root},
                output_root=cell["output_root"],
                t3_root=taste["t3_root"],
                policy_path=taste["policy_path"],
                policy_receipt=taste["policy_receipt"],
                prepared_root=taste["prepared_root"],
                graph_cache_root=taste["graph_cache_root"],
            )

    else:

        def _append(prior: Path) -> Mapping[str, Any]:
            return append_non_taste_matrix_cell(
                prior_authority_root=prior,
                dataset=str(cell["dataset"]),
                method=str(cell["method"]),
                cell_terminal_root=root,
                output_root=cell["output_root"],
                aids_controller_manifest=cell.get("aids_controller_manifest"),
            )

    return append_under_authority_pointer(
        state_path=state,
        lock_path=lock,
        initial_authority_root=queue["initial_authority_root"],
        requested_cells=(cell_id,),
        append=_append,
    )


def _heartbeat(
    *,
    queue: Mapping[str, Any],
    heartbeat_path: Path,
    state: str,
    pointer: Mapping[str, Any],
    cell_states: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": HEARTBEAT_SCHEMA,
        "pid": os.getpid(),
        "state": state,
        "queue_manifest_path": queue["manifest_path"],
        "queue_manifest_sha256": queue["manifest_sha256"],
        "authority_state_path": queue["authority_state_path"],
        "authority_lock_path": queue["authority_lock_path"],
        "latest_authority_root": pointer["latest_authority_root"],
        "latest_count": pointer["latest_count"],
        "applied_cells": pointer["applied_cells"],
        "cells": dict(cell_states),
        "updated_epoch": time.time(),
    }
    _atomic_json(heartbeat_path, payload)
    return payload


def run_queue(
    *, queue_manifest: str | Path, heartbeat_path: str | Path, once: bool = False
) -> dict[str, Any]:
    manifest_path = _absolute(queue_manifest, label="queue_manifest", must_exist=True)
    heartbeat = _absolute(heartbeat_path, label="heartbeat_path")
    queue = _load_queue(manifest_path)
    stopped = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    previous = {signum: signal.signal(signum, _stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    failures: dict[str, tuple[tuple[str, str], str]] = {}
    cell_states: dict[str, Any] = {}
    try:
        while True:
            pointer = read_authority_pointer(
                state_path=queue["authority_state_path"],
                lock_path=queue["authority_lock_path"],
                initial_authority_root=queue["initial_authority_root"],
            )
            applied = set(pointer["applied_cells"])
            for cell in queue["cells"]:
                cell_id = str(cell["cell_id"])
                if cell_id in applied:
                    cell_states[cell_id] = {"state": "APPLIED"}
                    continue
                fingerprint: tuple[str, str] | None = None
                try:
                    root, locator_state = _locator_root(cell)
                    if root is None:
                        cell_states[cell_id] = {"state": "WAITING", "reason": locator_state}
                        continue
                    fingerprint = _terminal_fingerprint(cell, root)
                    if fingerprint is None:
                        cell_states[cell_id] = {
                            "state": "WAITING",
                            "reason": "EXACT_FULL_PASS_NOT_VISIBLE",
                            "terminal_root": str(root),
                        }
                        continue
                    prior_failure = failures.get(cell_id)
                    if prior_failure is not None and prior_failure[0] == fingerprint:
                        cell_states[cell_id] = {
                            "state": "BLOCKED_TERMINAL_VALIDATION",
                            "terminal_root": str(root),
                            "error": prior_failure[1],
                            "retry_policy": "retry_only_after_terminal_fingerprint_changes",
                        }
                        continue
                    result = _append_cell(queue, cell, root)
                    failures.pop(cell_id, None)
                    cell_states[cell_id] = {
                        "state": "APPLIED",
                        "terminal_root": str(root),
                        "authority_root": result["output_root"],
                        "matrix_complete_cells": result["matrix_complete_cells"],
                    }
                    pointer = result["authority_pointer_after"]
                    applied = set(pointer["applied_cells"])
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if fingerprint is not None:
                        failures[cell_id] = (fingerprint, message)
                    cell_states[cell_id] = {
                        "state": "BLOCKED_TERMINAL_VALIDATION",
                        "error": message,
                    }
            configured = {str(cell["cell_id"]) for cell in queue["cells"]}
            complete = configured.issubset(set(pointer["applied_cells"]))
            state = "PASS" if complete else "WAITING"
            result = _heartbeat(
                queue=queue,
                heartbeat_path=heartbeat,
                state=state,
                pointer=pointer,
                cell_states=cell_states,
            )
            if complete or once or stopped:
                return result
            time.sleep(int(queue["poll_seconds"]))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--queue-manifest", required=True)
    parser.add_argument("--heartbeat-path", required=True)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("unsupported --set override")
    result = run_queue(
        queue_manifest=args.queue_manifest,
        heartbeat_path=args.heartbeat_path,
        once=args.once,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
