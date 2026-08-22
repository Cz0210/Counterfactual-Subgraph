#!/usr/bin/env python3
"""Write the immutable license-blocked TasteMolNet baseline task fragment.

This entrypoint performs no training, candidate generation, classifier
inference, or remote work.  It refuses a PASS license gate because releasing
heavy work requires a separate fresh fragment bound to a frozen three-class
GINE and exact native-input manifests.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from src.baselines.tastemolnet_multiclass_tasks import build_blocked_fragment


def _absolute_fresh(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _absolute_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    if path.is_symlink():
        raise argparse.ArgumentTypeError(
            f"physical license gate file required: {path}"
        )
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise argparse.ArgumentTypeError(f"path does not exist: {value}") from exc
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(
            f"physical license gate file required: {resolved}"
        )
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Task fragment output must be fresh: {path}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=_absolute_fresh, required=True)
    parser.add_argument("--license-gate", type=_absolute_file)
    parser.add_argument(
        "--license-task-id", default="tastemolnet_license_audit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate = _read_object(args.license_gate) if args.license_gate is not None else None
    payload = build_blocked_fragment(
        license_gate=gate,
        license_gate_path=args.license_gate,
        license_task_id=args.license_task_id,
    )
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "task_count": len(payload["tasks"]),
                "task_ids": [task["id"] for task in payload["tasks"]],
                "heavy_route_authorized": False,
                "release_contract_hash": payload["release_contract_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("[TASTEMOLNET_MULTICLASS_BASELINES_BLOCKED_LICENSE_REVIEW]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
