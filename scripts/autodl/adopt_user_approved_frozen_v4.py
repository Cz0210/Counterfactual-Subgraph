#!/usr/bin/env python3
"""Adopt the checksum-pinned AIDS/Mutagenicity v4 tables into fresh roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.user_approved_frozen_v4 import (  # noqa: E402
    adopt_user_approved_frozen_v4,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy exact rows from the checksum-pinned, user-approved AIDS/Mut "
            "v4 CSV bundle into six fresh standardized cell roots."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Runtime config recorded for CLI parity; scientific values are never read from it.",
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--no-proc-writer-audit",
        action="store_true",
        help="Development-only escape hatch; formal AutoDL adoption must not use it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Path(args.config).expanduser().resolve(strict=True)
    if args.no_proc_writer_audit and Path("/proc").is_dir():
        raise ValueError("formal Linux/AutoDL adoption may not disable procfs writer audit")
    result = adopt_user_approved_frozen_v4(
        source_root=args.source_root,
        runtime_root=args.runtime_root,
        output_root=args.output_root,
        require_proc_writer_audit=not args.no_proc_writer_audit,
    )
    explicit_args = [
        f"--explicit-cell={cell}={root}"
        for cell, root in sorted(result.cell_roots.items())
    ]
    print(
        json.dumps(
            {
                "approval_policy_sha256": result.approval_policy_sha256,
                "cell_count": len(result.cell_roots),
                "cell_roots": result.cell_roots,
                "config": str(config),
                "explicit_cell_args": explicit_args,
                "output_root": str(result.output_root),
                "source_file_count": len(result.source_inventory),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("[USER_APPROVED_FROZEN_V4_ADOPTION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
