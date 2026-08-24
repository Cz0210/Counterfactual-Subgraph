#!/usr/bin/env python3
"""Verify the sole allowed old read-only common-recourse process identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence


def _start_ticks(stat_path: Path) -> int:
    raw = stat_path.read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError("malformed proc stat")
    remainder = raw[closing + 2 :].split()
    # Remainder starts at field 3 (state); starttime is field 22.
    return int(remainder[19])


def verify_process_set(
    *,
    proc_root: str | Path,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: str | Path,
    allowed_project_root: str | Path,
) -> dict[str, Any]:
    if int(allowed_pid) <= 0 or int(allowed_start_ticks) <= 0:
        raise RuntimeError("INVALID_ALLOWED_OLD_PROCESS_GENERATION")
    if re.fullmatch(r"[0-9a-f]{64}", allowed_cmdline_sha256) is None:
        raise RuntimeError("INVALID_ALLOWED_OLD_PROCESS_COMMAND_SHA256")
    proc = Path(proc_root).expanduser().resolve(strict=True)
    output = str(Path(allowed_output_root).expanduser().resolve(strict=True))
    project = str(Path(allowed_project_root).expanduser().resolve(strict=True))
    active: list[dict[str, Any]] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        if "run_common_recourse.py" not in command:
            continue
        pid = int(entry.name)
        try:
            ticks = _start_ticks(entry / "stat")
            cwd = str((entry / "cwd").resolve(strict=True))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            raise RuntimeError(f"COMMON_RECOURSE_PROCESS_IDENTITY_UNREADABLE:{pid}")
        tokens = [
            token.decode("utf-8", errors="surrogateescape")
            for token in raw.rstrip(b"\0").split(b"\0")
        ]
        active.append(
            {
                "pid": pid,
                "start_ticks": ticks,
                "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                "command": command,
                "command_tokens": tokens,
                "cwd": cwd,
            }
        )
    active.sort(key=lambda row: int(row["pid"]))
    if len(active) > 1:
        raise RuntimeError(
            "UNEXPECTED_COMMON_RECOURSE_PROCESS_SET:" + json.dumps(active, sort_keys=True)
        )
    if active:
        row = active[0]
        if (
            int(row["pid"]) != int(allowed_pid)
            or int(row["start_ticks"]) != int(allowed_start_ticks)
            or row["cmdline_sha256"] != allowed_cmdline_sha256
            or output not in row["command_tokens"]
            or row["cwd"] != project
        ):
            raise RuntimeError(
                "OLD_COMMON_RECOURSE_PROCESS_IDENTITY_MISMATCH:"
                + json.dumps(row, sort_keys=True)
            )
        status = "ALLOWED_OLD_READ_ONLY_PROCESS_PRESENT"
    else:
        status = "ALLOWED_OLD_PROCESS_NATURALLY_EXITED"
    return {
        "status": "PASS",
        "process_set_status": status,
        "active_common_recourse_count": len(active),
        "allowed_pid": int(allowed_pid),
        "allowed_start_ticks": int(allowed_start_ticks),
        "allowed_cmdline_sha256": allowed_cmdline_sha256,
        "allowed_output_root": output,
        "allowed_project_root": project,
        "active": active,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--proc-root", type=Path, required=True)
    parser.add_argument("--allowed-pid", type=int, required=True)
    parser.add_argument("--allowed-start-ticks", type=int, required=True)
    parser.add_argument("--allowed-cmdline-sha256", required=True)
    parser.add_argument("--allowed-output-root", type=Path, required=True)
    parser.add_argument("--allowed-project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_process_set(
        proc_root=args.proc_root,
        allowed_pid=args.allowed_pid,
        allowed_start_ticks=args.allowed_start_ticks,
        allowed_cmdline_sha256=args.allowed_cmdline_sha256,
        allowed_output_root=args.allowed_output_root,
        allowed_project_root=args.allowed_project_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[AIDS_COMRECGC_EXACT_ROUTE_V5_PROCESS_SET_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
