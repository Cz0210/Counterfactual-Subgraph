#!/usr/bin/env python3
"""Validate the final-four observer heartbeat and process generation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.final_four_cells_observer import stable_sha256  # noqa: E402


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=_absolute, required=True)
    parser.add_argument("--max-age-seconds", type=float, default=180.0)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    heartbeat_path = args.state_root / "heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    claimed = heartbeat.pop("heartbeat_sha256")
    if stable_sha256(heartbeat) != claimed:
        raise ValueError("heartbeat self-hash mismatch")
    heartbeat["heartbeat_sha256"] = claimed
    written = datetime.fromisoformat(heartbeat["written_at"].replace("Z", "+00:00"))
    age = time.time() - written.timestamp()
    pid = int(heartbeat["controller_pid"])
    stat = (args.proc_root / str(pid) / "stat").read_text(encoding="ascii").split()
    if int(stat[21]) != int(heartbeat["controller_start_ticks"]):
        raise ValueError("observer PID generation changed")
    status = "RUNNING" if -30.0 <= age <= args.max_age_seconds else "STALE"
    print(json.dumps({"status": status, "age_seconds": age, **heartbeat}, sort_keys=True))
    return 0 if status == "RUNNING" else 75


if __name__ == "__main__":
    raise SystemExit(main())
