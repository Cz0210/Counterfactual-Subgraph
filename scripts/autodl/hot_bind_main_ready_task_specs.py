#!/usr/bin/env python3
"""Atomically publish main-ready specs and detect live-sidecar acknowledgement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.main_ready_task_specs import (  # noqa: E402
    atomic_json,
    file_sha256,
    load_manifest,
    load_pointer,
    manifest_for_specs,
    process_identity,
    seal_pointer,
    stable_sha256,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _heartbeat(path: Path) -> dict:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _heartbeat_epoch(value: dict) -> float | None:
    raw = value.get("written_at", value.get("written_at_unix"))
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _validate_live_sidecar(
    *,
    root: Path,
    expected_pid: int,
    expected_start_ticks: int,
    expected_command_sha256: str,
    heartbeat_max_age_seconds: float,
    proc_root: Path,
) -> tuple[dict, dict]:
    heartbeat_path = root / "heartbeat.json"
    heartbeat = _heartbeat(heartbeat_path)
    if not heartbeat:
        raise RuntimeError("sidecar heartbeat is absent or unreadable")
    if heartbeat.get("controller_pid") != expected_pid:
        raise RuntimeError("sidecar heartbeat PID changed")
    written_epoch = _heartbeat_epoch(heartbeat)
    if written_epoch is None:
        raise RuntimeError("sidecar heartbeat timestamp is absent")
    age = time.time() - written_epoch
    if age < -30 or age > heartbeat_max_age_seconds:
        raise RuntimeError(f"sidecar heartbeat is stale: {age:.1f}s")
    identity = process_identity(expected_pid, proc_root=proc_root)
    if identity is None or not identity["alive"]:
        raise RuntimeError("sidecar process is not live")
    if identity["start_ticks"] != expected_start_ticks:
        raise RuntimeError("sidecar PID generation changed")
    if identity["command_sha256"] != expected_command_sha256:
        raise RuntimeError("sidecar command SHA256 changed")
    return heartbeat, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--sidecar-control-root", type=_absolute, required=True)
    parser.add_argument("--task-spec", type=_absolute, action="append", required=True)
    parser.add_argument("--pointer", type=_absolute)
    parser.add_argument("--sidecar-pid", type=int, required=True)
    parser.add_argument("--sidecar-start-ticks", type=int, required=True)
    parser.add_argument("--sidecar-command-sha256", required=True)
    parser.add_argument("--sidecar-heartbeat-max-age-seconds", type=float, default=120.0)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--poll-count", type=int, default=2)
    args = parser.parse_args(argv)
    if args.sidecar_pid <= 0 or args.sidecar_start_ticks <= 0:
        raise ValueError("sidecar PID and start ticks must be positive")
    if args.sidecar_heartbeat_max_age_seconds <= 0:
        raise ValueError("sidecar heartbeat maximum age must be positive")
    supplied_root = args.sidecar_control_root
    root = supplied_root.resolve(strict=True)
    if supplied_root.is_symlink() or root != supplied_root or not root.is_dir():
        raise ValueError("sidecar control root must be one physical directory")
    before, before_identity = _validate_live_sidecar(
        root=root,
        expected_pid=args.sidecar_pid,
        expected_start_ticks=args.sidecar_start_ticks,
        expected_command_sha256=args.sidecar_command_sha256,
        heartbeat_max_age_seconds=args.sidecar_heartbeat_max_age_seconds,
        proc_root=args.proc_root,
    )
    expected_manifest = manifest_for_specs(args.task_spec)
    manifest_path = args.task_spec[0].parent / "task_specs_manifest.json"
    if any(path.parent != args.task_spec[0].parent for path in args.task_spec):
        raise ValueError("task specs must belong to one immutable bundle")
    manifest = load_manifest(manifest_path, spec_paths=args.task_spec)
    if manifest != expected_manifest:
        raise ValueError("existing immutable task-spec manifest differs")
    expected_pointer_path = root.parent / "main_ready_spec_pointer.json"
    pointer_path = args.pointer or expected_pointer_path
    if pointer_path != expected_pointer_path:
        raise ValueError("pointer path must be the sidecar's single fixed pointer")
    if pointer_path.is_symlink():
        raise ValueError("task-spec pointer cannot replace a symlink")
    published_at = datetime.now(timezone.utc).isoformat()
    published_epoch = datetime.fromisoformat(published_at).timestamp()
    pointer = seal_pointer({
        "manifest_path": str(manifest_path.resolve(strict=True)),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "sidecar_control_root": str(root),
        "sidecar_pid": args.sidecar_pid,
        "sidecar_start_ticks": args.sidecar_start_ticks,
        "sidecar_command_sha256": args.sidecar_command_sha256,
        "published_at": published_at,
    })
    atomic_json(pointer_path, pointer)
    load_pointer(pointer_path)
    before_digest = stable_sha256(before)
    acknowledged = False
    acknowledged_heartbeat: dict = {}
    for _ in range(max(0, args.poll_count)):
        time.sleep(max(0.0, args.poll_seconds))
        heartbeat, current_identity = _validate_live_sidecar(
            root=root,
            expected_pid=args.sidecar_pid,
            expected_start_ticks=args.sidecar_start_ticks,
            expected_command_sha256=args.sidecar_command_sha256,
            heartbeat_max_age_seconds=args.sidecar_heartbeat_max_age_seconds,
            proc_root=args.proc_root,
        )
        written_epoch = _heartbeat_epoch(heartbeat)
        if (
            current_identity["start_ticks"] == before_identity["start_ticks"]
            and stable_sha256(heartbeat) != before_digest
            and written_epoch is not None
            and written_epoch >= published_epoch
            and (
                heartbeat.get("main_ready_spec_manifest_sha256")
                == manifest["manifest_sha256"]
                or heartbeat.get("main_ready_spec_pointer_sha256")
                == pointer["pointer_sha256"]
            )
        ):
            acknowledged = True
            acknowledged_heartbeat = heartbeat
            break
    result = {
        "status": "HOT_BIND_ACKNOWLEDGED" if acknowledged else "HOT_BIND_UNSUPPORTED_FALLBACK_REQUIRED",
        "pointer": str(pointer_path),
        "pointer_sha256": pointer["pointer_sha256"],
        "manifest": str(manifest_path),
        "sidecar_pid": args.sidecar_pid,
        "sidecar_start_ticks": args.sidecar_start_ticks,
        "sidecar_command_sha256": args.sidecar_command_sha256,
        "acknowledged_heartbeat_sha256": (
            stable_sha256(acknowledged_heartbeat) if acknowledged else None
        ),
        "sidecar_restarted": False,
    }
    # The spec bundle is immutable after its directory-level publish.  Runtime
    # acknowledgement belongs beside the mutable sidecar pointer, not inside
    # the sealed bundle.
    atomic_json(root.parent / "main_ready_hot_bind_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if acknowledged else 75


if __name__ == "__main__":
    raise SystemExit(main())
