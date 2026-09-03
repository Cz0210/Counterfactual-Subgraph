#!/usr/bin/env python3
"""Read immutable main-ready specs and current owner evidence without mutation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.main_ready_task_specs import (  # noqa: E402
    load_spec,
    probe_owner,
    probe_terminal,
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
    parser.add_argument("--task-spec", type=_absolute, action="append", required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    rows = []
    for path in args.task_spec:
        spec = load_spec(path)
        owner = probe_owner(spec, proc_root=args.proc_root)
        terminal = probe_terminal(spec)
        rows.append({
            "task_id": spec["task_id"],
            "task_kind": spec["task_kind"],
            "spec_sha256": spec["spec_sha256"],
            "owner": owner,
            "terminal": terminal,
        })
    def terminal_pass(row: dict) -> bool:
        terminal = row["terminal"] or {}
        status = str(terminal.get("status", ""))
        return (
            terminal.get("state") == "TERMINAL"
            and "PASS" in status
            and "FAILED" not in status
            and "BLOCKED" not in status
        )

    if all(terminal_pass(row) for row in rows):
        aggregate = "PASS"
    elif all(
        row["owner"].get("state") == "OWNER_CONFIRMED" or terminal_pass(row)
        for row in rows
    ):
        aggregate = "RUNNING"
    else:
        aggregate = "BLOCKED"
    print(json.dumps({"status": aggregate, "tasks": rows}, sort_keys=True))
    return 0 if aggregate in {"PASS", "RUNNING"} else 75


if __name__ == "__main__":
    raise SystemExit(main())
