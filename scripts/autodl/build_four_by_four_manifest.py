#!/usr/bin/env python3
"""Compose validated task fragments into one fresh four-by-four manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


from scripts.autodl.run_four_gpu_recovery_controller import load_controller_manifest


DEFAULT_RUNTIME = {
    "max_gpus": 4,
    "stable_idle_seconds": 60,
    "sample_interval_seconds": 5,
    "poll_seconds": 60,
    "min_free_memory_mb": 16000,
    "idle_util_threshold": 10,
    "worker_launcher": "auto",
    "max_cpu_tasks": 4,
    "launch_grace_seconds": 180,
    "max_transient_retries": 1,
    "keep_alive_when_blocked": True,
}
DEFAULT_RESOURCE_GATES = {
    "min_available_ram_gb": 32,
    "min_free_disk_gb": 20,
    "max_cpu_load_fraction": 0.9,
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _fragment_tasks(path: Path) -> list[dict[str, Any]]:
    payload = _read(path)
    if isinstance(payload, list):
        tasks = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        tasks = payload["tasks"]
    else:
        raise ValueError(f"Task fragment must be a list or contain tasks: {path}")
    if not all(isinstance(task, dict) for task in tasks):
        raise ValueError(f"Task fragment contains a non-object task: {path}")
    return [dict(task) for task in tasks]


def compose_manifest(
    *,
    controller_id: str,
    fragments: Sequence[Path],
    output: Path,
    base_manifest: Path | None = None,
) -> dict[str, Any]:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Manifest output must be fresh: {destination}")
    runtime = dict(DEFAULT_RUNTIME)
    resource_gates = dict(DEFAULT_RESOURCE_GATES)
    continuation: dict[str, Any] | None = None
    if base_manifest is not None:
        base = _read(base_manifest.expanduser().resolve(strict=True))
        if not isinstance(base, dict):
            raise ValueError("Base manifest must be one JSON object")
        if isinstance(base.get("runtime"), dict):
            runtime.update(base["runtime"])
        if isinstance(base.get("resource_gates"), dict):
            resource_gates.update(base["resource_gates"])
        raw_continuation = base.get("continuation")
        if raw_continuation is not None:
            if not isinstance(raw_continuation, dict):
                raise ValueError("Base manifest continuation must be one object")
            continuation = dict(raw_continuation)
    # The continuation never exits merely because the remaining work is
    # license/code blocked.  It keeps heartbeat and queue state without a dummy
    # process or GPU allocation.
    runtime["keep_alive_when_blocked"] = True

    tasks: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fragment in fragments:
        resolved = fragment.expanduser().resolve(strict=True)
        rows = _fragment_tasks(resolved)
        for task in rows:
            task_id = str(task.get("id") or "")
            if not task_id:
                raise ValueError(f"Task without id in {resolved}")
            if task_id in seen:
                raise ValueError(f"Duplicate task id across fragments: {task_id}")
            seen.add(task_id)
            if "__CONFIGURE_" in json.dumps(task, sort_keys=True):
                raise ValueError(f"Unresolved configuration placeholder: {task_id}")
            tasks.append(task)
        provenance.append(
            {
                "path": str(resolved),
                "sha256": _sha(resolved),
                "task_count": len(rows),
            }
        )
    if not tasks:
        raise ValueError("At least one task fragment is required")

    payload = {
        "schema_version": 1,
        "controller_id": controller_id,
        "paper_frozen": True,
        "runtime": runtime,
        "resource_gates": resource_gates,
        "task_fragment_provenance": provenance,
        "tasks": tasks,
    }
    if continuation is not None:
        # Preserve the BACE predecessor quiescence policy when its continuation
        # manifest is used as the base of the wider four-by-four controller.
        # The production loader revalidates this object against the new, fresh
        # controller ID before the composed manifest is published.
        payload["continuation"] = continuation
    _atomic_json(destination, payload)
    try:
        manifest = load_controller_manifest(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {
        "status": "PASS",
        "controller_id": manifest.controller_id,
        "manifest": str(destination),
        "manifest_sha256": manifest.sha256,
        "task_count": len(manifest.tasks),
        "task_ids": [task.task_id for task in manifest.tasks],
        "fragment_provenance": provenance,
        "keep_alive_when_blocked": runtime["keep_alive_when_blocked"],
        "paper_frozen": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--task-fragment", action="append", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compose_manifest(
        controller_id=args.controller_id,
        fragments=args.task_fragment,
        output=args.output,
        base_manifest=args.base_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[FOUR_BY_FOUR_MANIFEST_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
