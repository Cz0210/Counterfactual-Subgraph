#!/usr/bin/env python3
"""Print one fast-v2 heartbeat and reject a stale controller."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--max-age-seconds", type=float, default=180.0)
    args = parser.parse_args()
    value = json.loads((args.state_root / "heartbeat.json").read_text("utf-8"))
    written = datetime.fromisoformat(value["written_at"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - written).total_seconds()
    value["heartbeat_age_seconds"] = age
    value["status"] = "RUNNING" if age <= args.max_age_seconds else "STALE"
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "RUNNING" else 75


if __name__ == "__main__":
    raise SystemExit(main())
