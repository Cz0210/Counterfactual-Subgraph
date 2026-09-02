#!/usr/bin/env python3
"""Print the durable main-and-ablations heartbeat and staleness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--max-age-seconds", type=float, default=120.0)
    args = parser.parse_args()
    heartbeat = args.state_root.absolute() / "heartbeat.json"
    value = json.loads(heartbeat.read_text(encoding="utf-8"))
    if value.get("schema_version") != "main_and_ablations_heartbeat_v1":
        raise SystemExit("heartbeat schema changed")
    written = datetime.fromisoformat(str(value["written_at"]).replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - written).total_seconds()
    value["heartbeat_age_seconds"] = age
    value["status"] = "RUNNING" if age <= args.max_age_seconds else "STALE"
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "RUNNING" else 75


if __name__ == "__main__":
    raise SystemExit(main())
