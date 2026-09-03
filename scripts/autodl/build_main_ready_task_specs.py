#!/usr/bin/env python3
"""Seal resolved JSON descriptors as immutable main-ready task specs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.main_ready_task_specs import (  # noqa: E402
    atomic_json,
    file_sha256,
    manifest_for_specs,
    materialize_task_spec_path,
    seal_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--descriptor", type=_absolute, action="append", required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("task-spec output root must be fresh")
    parent = args.output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.", suffix=".tmp", dir=parent
        )
    )
    published_paths: list[Path] = []
    staged_paths: list[Path] = []
    task_ids: set[str] = set()
    published = False
    try:
        for descriptor in args.descriptor:
            if (
                not descriptor.is_file()
                or descriptor.is_symlink()
                or descriptor.resolve(strict=True) != descriptor
            ):
                raise ValueError("task descriptor must be an absolute physical file")
            raw = json.loads(descriptor.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("task descriptor must be a JSON object")
            task_id = raw.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                raise ValueError("task descriptors must have unique non-empty task IDs")
            task_ids.add(task_id)
            final_path = args.output_root / f"{task_id}.json"
            raw = materialize_task_spec_path(raw, final_path)
            for field in (
                "output_root",
                "expected_heartbeat_path",
                "expected_pid_file",
                "expected_terminal_path",
            ):
                candidate = raw.get(field)
                if candidate is None:
                    continue
                path = Path(str(candidate))
                if path.exists() or path.is_symlink():
                    raise FileExistsError(
                        f"fresh descriptor path already exists: {field}={path}"
                    )
            raw.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            raw["config_sha256"] = file_sha256(Path(str(raw["config_path"])))
            raw["manifest_sha256"] = file_sha256(Path(str(raw["manifest_path"])))
            spec = seal_spec(raw)
            staged_path = staging / final_path.name
            atomic_json(staged_path, spec)
            staged_paths.append(staged_path)
            published_paths.append(final_path)
        manifest = manifest_for_specs(
            staged_paths, published_paths=published_paths
        )
        atomic_json(staging / "task_specs_manifest.json", manifest)
        directory_fd = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(staging, args.output_root)
        published = True
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    paths = published_paths
    manifest_path = args.output_root / "task_specs_manifest.json"
    print(json.dumps({
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "specs": [str(path) for path in paths],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
