#!/usr/bin/env python3
"""Validate and print the final16 successors controller heartbeat."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.final16_owner_registry_v1 import process_start_ticks  # noqa: E402
from src.utils.final16_successors_v1 import HEARTBEAT_SCHEMA  # noqa: E402
from src.utils.final_four_cells_observer import stable_sha256  # noqa: E402


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def read_status(
    *, state_root: Path, max_age_seconds: float, proc_root: Path = Path("/proc")
) -> tuple[dict[str, Any], int]:
    if state_root.is_symlink() or not state_root.is_dir():
        raise ValueError("state root is absent or indirect")
    heartbeat_path = state_root / "heartbeat.json"
    if heartbeat_path.is_symlink() or not heartbeat_path.is_file():
        raise ValueError("controller heartbeat is absent or indirect")
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    if not isinstance(heartbeat, dict) or heartbeat.get("schema_version") != HEARTBEAT_SCHEMA:
        raise ValueError("controller heartbeat schema changed")
    claimed = heartbeat.pop("heartbeat_sha256", None)
    if claimed != stable_sha256(heartbeat):
        raise ValueError("controller heartbeat self hash changed")
    heartbeat["heartbeat_sha256"] = claimed
    written = datetime.fromisoformat(str(heartbeat["written_at"]).replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - written).total_seconds()
    pid = heartbeat.get("controller_pid")
    ticks = heartbeat.get("controller_start_ticks")
    live = (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and isinstance(ticks, int)
        and not isinstance(ticks, bool)
        and process_start_ticks(proc_root, pid) == ticks
    )
    fresh = -30.0 <= age <= max_age_seconds
    status = "RUNNING" if live and fresh else "STALE"
    value = {
        "status": status,
        "heartbeat_age_seconds": age,
        "controller_process_live": live,
        **heartbeat,
    }
    return value, 0 if status == "RUNNING" else 75


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=_absolute, required=True)
    parser.add_argument("--max-age-seconds", type=float, default=180.0)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml", str(PROJECT_ROOT / "configs/hpc.yaml")):
        raise ValueError("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise ValueError("unsupported --set override")
    if not 10.0 <= args.max_age_seconds <= 86_400.0:
        raise ValueError("heartbeat age limit must be in [10, 86400] seconds")
    value, return_code = read_status(
        state_root=args.state_root,
        max_age_seconds=args.max_age_seconds,
        proc_root=args.proc_root,
    )
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
