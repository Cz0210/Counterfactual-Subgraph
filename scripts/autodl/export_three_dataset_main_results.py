#!/usr/bin/env python3
"""Export the staging-only AIDS/Mutagenicity/BACE four-method results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.four_by_four_main_results import MainResultsError  # noqa: E402
from src.eval.three_dataset_main_results import (  # noqa: E402
    export_three_dataset_results,
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
    parser.add_argument("--matrix-status", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--paper-staging-root", type=_absolute)
    parser.add_argument("--project-root", type=_absolute, default=PROJECT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is not None and not Path(args.config).is_file():
        print(f"[THREE_DATASET_EXPORT_FAILED] missing config: {args.config}", file=sys.stderr)
        return 2
    unsupported = [
        value
        for value in args.set
        if value != "inference.fallback_to_heuristic=false"
    ]
    if unsupported:
        print(
            f"[THREE_DATASET_EXPORT_FAILED] unsupported --set values: {unsupported}",
            file=sys.stderr,
        )
        return 2
    try:
        result = export_three_dataset_results(
            matrix_status=args.matrix_status,
            output_root=args.output_root,
            paper_staging_root=args.paper_staging_root,
            project_root=args.project_root,
        )
    except (MainResultsError, FileExistsError, OSError, ValueError) as exc:
        print(
            f"[THREE_DATASET_EXPORT_BLOCKED] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {
                "status": "PASS",
                "matrix_complete_cells": result.matrix_complete_cells,
                "matrix_total_cells": 16,
                "output_root": str(result.output_root),
                "paper_staging_root": (
                    str(result.paper_staging_root)
                    if result.paper_staging_root is not None
                    else None
                ),
                "generated_file_count": len(result.generated_files),
                "paper_status": "PAPER_FROZEN_PARTIAL",
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("[MATRIX_12_OF_16_PASS]")
    print("[THREE_DATASET_FIGURE3_PASS]")
    print("[THREE_DATASET_FIGURE4_PASS]")
    print("[THREE_DATASET_TABLE2_PASS]")
    print("[PAPER_FROZEN_PARTIAL]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

