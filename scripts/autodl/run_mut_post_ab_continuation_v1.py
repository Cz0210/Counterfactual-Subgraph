#!/usr/bin/env python3
"""Persist the next Mut action after the bounded same-contract A/B exits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_first_divergence_v1 import atomic_json  # noqa: E402
from src.utils.autodl_mut_post_ab_continuation_v1 import (  # noqa: E402
    select_post_ab_action,
    validate_ab_owner_terminal,
)
from src.utils.autodl_mut_same_contract_ab_v1 import (  # noqa: E402
    validate_same_contract_ab_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected one JSON object: {path}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--ab-task-spec", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if args.poll_seconds < 30:
        raise ValueError("post-A/B continuation poll interval must be >=30 seconds")
    spec = validate_same_contract_ab_spec(_json(args.ab_task_spec), check_files=True)
    output = args.output_root
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"post-A/B continuation output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    pid = os.getpid()
    atomic_json(
        output / "owner_pid.json",
        {
            "schema_version": "mut_post_ab_continuation_owner_pid_v1",
            "pid": pid,
            "ab_task_spec": str(args.ab_task_spec),
            "started_at": _now(),
        },
    )
    terminal_path = Path(spec["control_root"]) / "terminal.json"
    gate_path = Path(spec["output_dir"]) / "trace_on_off_500_step_equivalence.json"
    sequence = 0
    try:
        while not terminal_path.is_file():
            sequence += 1
            atomic_json(
                output / "heartbeat.json",
                {
                    "schema_version": "mut_post_ab_continuation_heartbeat_v1",
                    "pid": pid,
                    "state": "WAITING_FOR_BOUNDED_A_B_TERMINAL",
                    "sequence": sequence,
                    "ab_control_root": spec["control_root"],
                    "fresh_50k_started": False,
                    "written_at": _now(),
                },
            )
            time.sleep(args.poll_seconds)
        if not gate_path.is_file():
            raise RuntimeError("A/B owner exited without a structured equivalence gate")
        terminal = validate_ab_owner_terminal(
            _json(terminal_path), task_id=str(spec["task_id"]), gate_path=gate_path
        )
        decision = select_post_ab_action(
            terminal=terminal,
            gate=_json(gate_path),
            ab_spec_path=args.ab_task_spec,
            gate_path=gate_path,
        )
        atomic_json(output / "next_action.json", decision)
        atomic_json(
            output / "terminal.json",
            {
                "schema_version": "mut_post_ab_continuation_terminal_v1",
                "status": "NEXT_ACTION_PERSISTED",
                "branch": decision["branch"],
                "classification": decision["classification"],
                "next_action": str(output / "next_action.json"),
                "fresh_50k_started": False,
                "completed_at": _now(),
            },
        )
        print(json.dumps(decision, sort_keys=True), flush=True)
        print("[MUT_POST_AB_NEXT_ACTION_PERSISTED]", flush=True)
        return 0
    except BaseException as exc:
        atomic_json(
            output / "terminal.json",
            {
                "schema_version": "mut_post_ab_continuation_terminal_v1",
                "status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
                "fresh_50k_started": False,
                "failed_at": _now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
