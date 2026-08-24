#!/usr/bin/env python3
"""Create a fresh physical snapshot of the promoted AIDS ComRecGC pair store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.aids_comrecgc_v5_snapshot import (  # noqa: E402
    EXPECTED_CANDIDATE_COUNT,
    EXPECTED_PARENT_COUNT,
    EXPECTED_ROWS,
    EXPECTED_VECTOR_DIM,
    MIN_FREE_AFTER_BYTES,
    create_promoted_pair_store_snapshot,
    validate_promoted_pair_store_snapshot,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    # Preserve the logical spelling so the core can reject a symlink instead
    # of silently resolving it before the physical-source/output gate.
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=_absolute, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--allowed-pid", type=int, required=True)
    parser.add_argument("--allowed-start-ticks", type=int, required=True)
    parser.add_argument("--allowed-cmdline-sha256", required=True)
    parser.add_argument("--allowed-output-root", type=_absolute, required=True)
    parser.add_argument("--allowed-project-root", type=_absolute, required=True)
    parser.add_argument(
        "--min-free-after-bytes", type=int, default=MIN_FREE_AFTER_BYTES
    )
    parser.add_argument("--expected-row-count", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--expected-vector-dim", type=int, default=EXPECTED_VECTOR_DIM
    )
    parser.add_argument(
        "--expected-parent-count", type=int, default=EXPECTED_PARENT_COUNT
    )
    parser.add_argument(
        "--expected-candidate-count", type=int, default=EXPECTED_CANDIDATE_COUNT
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = dict(
        source_root=args.source_root,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        output_dir=args.output_dir,
        proc_root=args.proc_root,
        allowed_pid=args.allowed_pid,
        allowed_start_ticks=args.allowed_start_ticks,
        allowed_cmdline_sha256=args.allowed_cmdline_sha256,
        allowed_output_root=args.allowed_output_root,
        allowed_project_root=args.allowed_project_root,
        min_free_after_bytes=args.min_free_after_bytes,
        expected_row_count=args.expected_row_count,
        expected_vector_dim=args.expected_vector_dim,
        expected_parent_count=args.expected_parent_count,
        expected_candidate_count=args.expected_candidate_count,
    )
    if args.validate_only:
        result = validate_promoted_pair_store_snapshot(**common, require_pass=True)
        marker = "[AIDS_COMRECGC_PAIR_STORE_PHYSICAL_SNAPSHOT_VALIDATE_PASS]"
    else:
        result = create_promoted_pair_store_snapshot(
            **common,
            min_free_after_bytes=args.min_free_after_bytes,
            resume=args.resume,
        )
        marker = "[AIDS_COMRECGC_PAIR_STORE_PHYSICAL_SNAPSHOT_PASS]"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
