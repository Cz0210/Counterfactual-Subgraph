#!/usr/bin/env python3
"""Append one strict AIDS/BACE terminal through the shared fast16 pointer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.fast16_matrix_authority_pointer import (  # noqa: E402
    DEFAULT_LOCK_PATH,
    DEFAULT_STATE_PATH,
    append_under_authority_pointer,
)
from src.eval.non_taste_matrix_append import (  # noqa: E402
    append_non_taste_matrix_cell,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=("AIDS", "Mutagenicity", "BACE"), required=True)
    parser.add_argument("--method", choices=("GlobalGCE", "ComRecGC"), required=True)
    parser.add_argument("--cell-terminal-root", type=_absolute, required=True)
    parser.add_argument("--aids-controller-manifest", type=_absolute)
    parser.add_argument(
        "--prior-authority-root",
        type=_absolute,
        help="Required only to initialize a missing shared pointer; otherwise omit it.",
    )
    parser.add_argument(
        "--authority-state-path", type=_absolute, default=DEFAULT_STATE_PATH
    )
    parser.add_argument(
        "--authority-lock-path", type=_absolute, default=DEFAULT_LOCK_PATH
    )
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit(
            "--set may only be inference.fallback_to_heuristic=false"
        )

    def _append(prior: Path) -> dict[str, object]:
        return append_non_taste_matrix_cell(
            prior_authority_root=prior,
            dataset=args.dataset,
            method=args.method,
            cell_terminal_root=args.cell_terminal_root,
            aids_controller_manifest=args.aids_controller_manifest,
            output_root=args.output_root,
            proc_root=args.proc_root,
        )

    result = append_under_authority_pointer(
        state_path=args.authority_state_path,
        lock_path=args.authority_lock_path,
        initial_authority_root=args.prior_authority_root,
        requested_cells=(f"{args.dataset}/{args.method}",),
        append=_append,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(result["marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
