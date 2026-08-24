#!/usr/bin/env python3
"""Verify old/read-only and current-route common-recourse process identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence


_PYTHON_INTERPRETER_RE = re.compile(r"python(?:\d+(?:\.\d+)*)?$")
_PYTHON_ENTRYPOINT_FLAGS = frozenset(
    {
        "-B",
        "-E",
        "-I",
        "-O",
        "-OO",
        "-P",
        "-R",
        "-S",
        "-b",
        "-bb",
        "-d",
        "-i",
        "-q",
        "-s",
        "-u",
        "-v",
        "-x",
    }
)


def _decode_cmdline(raw: bytes) -> list[str]:
    """Decode one procfs NUL argv without treating argument text as a command."""

    if not raw:
        return []
    return [
        token.decode("utf-8", errors="surrogateescape")
        for token in raw.rstrip(b"\0").split(b"\0")
    ]


def _common_recourse_script_index(tokens: Sequence[str]) -> int | None:
    """Return the script argv index for a plausible direct/Python launch."""

    if not tokens:
        return None
    executable = Path(tokens[0]).expanduser()
    if executable.name == "run_common_recourse.py":
        return 0
    if _PYTHON_INTERPRETER_RE.fullmatch(executable.name) is None:
        return None
    script_index = 1
    while script_index < len(tokens):
        token = tokens[script_index]
        if token in _PYTHON_ENTRYPOINT_FLAGS:
            script_index += 1
            continue
        if token in {"-W", "-X", "--check-hash-based-pycs"}:
            script_index += 2
            continue
        if (
            (token.startswith("-W") and token != "-W")
            or (token.startswith("-X") and token != "-X")
            or token.startswith("--check-hash-based-pycs=")
        ):
            script_index += 1
            continue
        if token == "--":
            script_index += 1
        break
    if script_index >= len(tokens):
        return None
    # -c/-m and all unrecognized interpreter flags are not physical script
    # entrypoints.  They must not turn a later diagnostic literal into one.
    if tokens[script_index].startswith("-"):
        return None
    return script_index


def _common_recourse_entrypoint(
    tokens: Sequence[str], *, process_cwd: Path
) -> str | None:
    """Return the physical entrypoint only for a real Python script argv.

    A shell, grep, monitoring command, or ``bash -c`` payload may contain the
    literal script name in an arbitrary argument.  Such text is evidence about
    that diagnostic command, not an executing common-recourse worker.  The
    production worker contract is deliberately narrower: argv[0] is either the
    exact executable script or a Python interpreter followed by only inert
    interpreter flags and the exact physical ``run_common_recourse.py``
    entrypoint.  Actual workers with a different output/cwd/generation remain
    visible here and are rejected later as rogue processes.
    """

    script_index = _common_recourse_script_index(tokens)
    if script_index is None:
        return None
    logical = Path(tokens[script_index]).expanduser()
    if logical.name != "run_common_recourse.py":
        return None
    if not logical.is_absolute():
        logical = process_cwd / logical
    try:
        if logical.is_symlink():
            raise ValueError("script entrypoint is a symlink")
        resolved = logical.resolve(strict=True)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise ValueError("script entrypoint is unavailable") from exc
    if not resolved.is_file() or resolved.name != "run_common_recourse.py":
        raise ValueError("script entrypoint is not a regular physical file")
    return str(resolved)


def _proc_stat(stat_path: Path) -> tuple[str, int, int]:
    raw = stat_path.read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError("malformed proc stat")
    remainder = raw[closing + 2 :].split()
    # Remainder starts at field 3 (state); starttime is field 22.
    if len(remainder) <= 19:
        raise ValueError("truncated proc stat")
    return remainder[0], int(remainder[1]), int(remainder[19])


def _is_descendant_of_generation(
    *, proc_root: Path, pid: int, root_pid: int, root_start_ticks: int
) -> bool:
    current = int(pid)
    visited: set[int] = set()
    for _depth in range(128):
        if current <= 0 or current in visited:
            return False
        visited.add(current)
        try:
            _state, parent, ticks = _proc_stat(proc_root / str(current) / "stat")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            return False
        if current == int(root_pid):
            return int(ticks) == int(root_start_ticks)
        current = int(parent)
    return False


def verify_process_set(
    *,
    proc_root: str | Path,
    allowed_pid: int,
    allowed_start_ticks: int,
    allowed_cmdline_sha256: str,
    allowed_output_root: str | Path,
    allowed_project_root: str | Path,
    allowed_route_root_pid: int | None = None,
    allowed_route_root_start_ticks: int | None = None,
    allowed_route_output_root: str | Path | None = None,
    allowed_route_project_root: str | Path | None = None,
) -> dict[str, Any]:
    if int(allowed_pid) <= 0 or int(allowed_start_ticks) <= 0:
        raise RuntimeError("INVALID_ALLOWED_OLD_PROCESS_GENERATION")
    if re.fullmatch(r"[0-9a-f]{64}", allowed_cmdline_sha256) is None:
        raise RuntimeError("INVALID_ALLOWED_OLD_PROCESS_COMMAND_SHA256")
    proc = Path(proc_root).expanduser().resolve(strict=True)
    output = str(Path(allowed_output_root).expanduser().resolve(strict=True))
    project = str(Path(allowed_project_root).expanduser().resolve(strict=True))
    route_values = (
        allowed_route_root_pid,
        allowed_route_root_start_ticks,
        allowed_route_output_root,
        allowed_route_project_root,
    )
    route_enabled = any(value is not None for value in route_values)
    if route_enabled and not all(value is not None for value in route_values):
        raise RuntimeError("INCOMPLETE_ALLOWED_ROUTE_PROCESS_CONTRACT")
    route_output = None
    route_project = None
    route_script = None
    if route_enabled:
        if int(allowed_route_root_pid or 0) <= 0 or int(
            allowed_route_root_start_ticks or 0
        ) <= 0:
            raise RuntimeError("INVALID_ALLOWED_ROUTE_PROCESS_GENERATION")
        route_output_logical = Path(str(allowed_route_output_root)).expanduser()
        if not route_output_logical.is_absolute() or route_output_logical.is_symlink():
            raise RuntimeError("INVALID_ALLOWED_ROUTE_OUTPUT_ROOT")
        # The stage directory may not exist during the first supervisor poll;
        # its exact fresh absolute spelling is nevertheless frozen here.
        route_output = str(route_output_logical.resolve(strict=False))
        route_project = str(
            Path(str(allowed_route_project_root)).expanduser().resolve(strict=True)
        )
        route_script = str(
            (
                Path(route_project)
                / "scripts/baselines/comrecgc/run_common_recourse.py"
            ).resolve(strict=True)
        )
    active: list[dict[str, Any]] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        pid = int(entry.name)
        # The frozen old PID is inspected independently below.  In particular,
        # a reused PID whose new command is not run_common_recourse.py must not
        # disappear through this command-name filter and be mistaken for a
        # naturally exited old generation.
        if pid == int(allowed_pid):
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        tokens = _decode_cmdline(raw)
        if _common_recourse_script_index(tokens) is None:
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
        try:
            _state, parent_pid, ticks = _proc_stat(entry / "stat")
            cwd_path = (entry / "cwd").resolve(strict=True)
            cwd = str(cwd_path)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            raise RuntimeError(f"COMMON_RECOURSE_PROCESS_IDENTITY_UNREADABLE:{pid}")
        try:
            entrypoint = _common_recourse_entrypoint(tokens, process_cwd=cwd_path)
        except ValueError as exc:
            raise RuntimeError(
                f"COMMON_RECOURSE_ENTRYPOINT_IDENTITY_UNREADABLE:{pid}"
            ) from exc
        if entrypoint is None:
            continue
        active.append(
            {
                "pid": pid,
                "start_ticks": ticks,
                "parent_pid": parent_pid,
                "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                "command": command,
                "command_tokens": tokens,
                "entrypoint": entrypoint,
                "cwd": cwd,
            }
        )

    old_entry = proc / str(int(allowed_pid))
    try:
        _old_state, old_parent_pid, old_ticks = _proc_stat(old_entry / "stat")
    except FileNotFoundError:
        old_ticks = None
    except (PermissionError, ProcessLookupError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"ALLOWED_OLD_PROCESS_GENERATION_UNREADABLE:{allowed_pid}"
        ) from exc
    if old_ticks is not None:
        if int(old_ticks) != int(allowed_start_ticks):
            raise RuntimeError(
                "ALLOWED_OLD_PROCESS_PID_REUSED:"
                f"{allowed_pid}:{old_ticks}!={allowed_start_ticks}"
            )
        try:
            old_raw = (old_entry / "cmdline").read_bytes()
            old_cwd = str((old_entry / "cwd").resolve(strict=True))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as exc:
            # A same-generation process may disappear between the stat and
            # identity reads.  Treat only a now-absent /proc generation as a
            # natural exit; any still-present but unreadable identity is a
            # fail-closed production error.
            try:
                _proc_stat(old_entry / "stat")
            except FileNotFoundError:
                old_ticks = None
            except (PermissionError, ProcessLookupError, OSError, ValueError):
                raise RuntimeError(
                    f"ALLOWED_OLD_PROCESS_IDENTITY_UNREADABLE:{allowed_pid}"
                ) from exc
            else:
                raise RuntimeError(
                    f"ALLOWED_OLD_PROCESS_IDENTITY_UNREADABLE:{allowed_pid}"
                ) from exc
        if old_ticks is not None:
            old_tokens = _decode_cmdline(old_raw)
            active.append(
                {
                    "pid": int(allowed_pid),
                    "start_ticks": int(old_ticks),
                    "parent_pid": int(old_parent_pid),
                    "cmdline_sha256": hashlib.sha256(old_raw).hexdigest(),
                    "command": old_raw.replace(b"\0", b" ").decode(
                        "utf-8", errors="replace"
                    ),
                    "command_tokens": old_tokens,
                    "cwd": old_cwd,
                }
            )
    active.sort(key=lambda row: int(row["pid"]))
    old_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for row in active:
        if int(row["pid"]) != int(allowed_pid):
            is_route = bool(
                route_enabled
                and route_output in row["command_tokens"]
                and row.get("entrypoint") == route_script
                and row["cwd"] == route_project
                and _is_descendant_of_generation(
                    proc_root=proc,
                    pid=int(row["pid"]),
                    root_pid=int(allowed_route_root_pid or 0),
                    root_start_ticks=int(allowed_route_root_start_ticks or 0),
                )
            )
            if is_route:
                route_rows.append(row)
            else:
                unexpected.append(row)
            continue
        if (
            int(row["start_ticks"]) == int(allowed_start_ticks)
            and row["cmdline_sha256"] == allowed_cmdline_sha256
            and output in row["command_tokens"]
            and row["cwd"] == project
        ):
            old_rows.append(row)
        else:
            unexpected.append(row)
    if unexpected or len(old_rows) > 1 or len(route_rows) > 1:
        raise RuntimeError(
            "UNEXPECTED_COMMON_RECOURSE_PROCESS_SET:"
            + json.dumps(
                {
                    "unexpected": unexpected,
                    "old_rows": old_rows,
                    "route_rows": route_rows,
                },
                sort_keys=True,
            )
        )
    if old_rows:
        status = "ALLOWED_OLD_READ_ONLY_PROCESS_PRESENT"
    else:
        status = "ALLOWED_OLD_PROCESS_NATURALLY_EXITED"
    return {
        "status": "PASS",
        "process_set_status": status,
        "active_common_recourse_count": len(active),
        "allowed_old_process_count": len(old_rows),
        "allowed_route_process_count": len(route_rows),
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
    parser.add_argument("--allowed-route-root-pid", type=int)
    parser.add_argument("--allowed-route-root-start-ticks", type=int)
    parser.add_argument("--allowed-route-output-root", type=Path)
    parser.add_argument("--allowed-route-project-root", type=Path)
    parser.add_argument("--quiet", action="store_true")
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
        allowed_route_root_pid=args.allowed_route_root_pid,
        allowed_route_root_start_ticks=args.allowed_route_root_start_ticks,
        allowed_route_output_root=args.allowed_route_output_root,
        allowed_route_project_root=args.allowed_route_project_root,
    )
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
        print("[AIDS_COMRECGC_EXACT_ROUTE_V5_PROCESS_SET_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
