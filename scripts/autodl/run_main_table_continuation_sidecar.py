#!/usr/bin/env python3
"""Run, tick, or inspect the bounded AutoDL main-table continuation sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_main_table_continuation_sidecar import (  # noqa: E402
    ContinuationSidecar,
    ContinuationSidecarError,
    load_continuation_spec,
    read_sidecar_status,
    run_child_with_terminal_receipt,
)
from src.eval.bace_comrecgc_convergence import (  # noqa: E402
    audit_bace_comrecgc_convergence,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("absolute path required")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "once"):
        command = commands.add_parser(name)
        command.add_argument("--spec", type=_absolute, required=True)
    status = commands.add_parser("status")
    status.add_argument("--state-root", type=_absolute, required=True)
    child = commands.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--task", choices=("T9", "NEUROSED"), required=True)
    child.add_argument("--terminal-receipt", type=_absolute, required=True)
    child.add_argument("child_argv", nargs=argparse.REMAINDER)
    return parser


def _convergence_auditor(spec_path: Path):
    spec = load_continuation_spec(spec_path)
    authority = spec["observers"]["bace_comrecgc"].get("convergence_audit")
    if authority is None:
        return None

    def run(hook_input):
        evaluation_step = int(hook_input["evaluation_step"])
        audit_root = Path(authority["audit_parent"]) / (
            f"step-{evaluation_step:05d}-{uuid.uuid4()}"
        )
        return audit_bace_comrecgc_convergence(
            resolved_config_path=authority["resolved_config_path"],
            trace_chunks_dir=authority["trace_chunks_dir"],
            local_checkpoint_root=authority["local_checkpoint_root"],
            mirror_checkpoint_root=authority["mirror_checkpoint_root"],
            audit_root=audit_root,
            evaluation_step=evaluation_step,
            expected_config_sha256=authority["expected_config_sha256"],
            expected_parent_ids_sha256=authority[
                "expected_parent_ids_sha256"
            ],
        )

    return run


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.config.is_file():
            raise ContinuationSidecarError(f"config does not exist: {args.config}")
        if args.command == "status":
            print(json.dumps(read_sidecar_status(args.state_root), indent=2, sort_keys=True))
            return 0
        if args.command == "_child":
            return run_child_with_terminal_receipt(
                task=args.task,
                terminal_receipt=args.terminal_receipt,
                command=args.child_argv,
            )
        with ContinuationSidecar(
            args.spec,
            convergence_auditor=_convergence_auditor(args.spec),
        ) as sidecar:
            return sidecar.run(once=args.command == "once")
    except (ContinuationSidecarError, OSError, ValueError) as exc:
        print(
            f"[MAIN_TABLE_CONTINUATION_SIDECAR_BLOCKED] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
