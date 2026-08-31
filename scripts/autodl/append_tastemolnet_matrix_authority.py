#!/usr/bin/env python3
"""Strictly append one or more completed TasteMolNet full cells to the matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_matrix_append import (  # noqa: E402
    METHOD_CONTRACTS,
    append_tastemolnet_cells,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _cells(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for declaration in values:
        if "=" not in declaration:
            raise argparse.ArgumentTypeError("--taste-cell must use METHOD=ROOT")
        method, root = declaration.split("=", 1)
        method = method.strip()
        if method not in METHOD_CONTRACTS or method in result or not root.strip():
            raise argparse.ArgumentTypeError(
                "--taste-cell requires one unique Ours/GCFExplainer/GlobalGCE/ComRecGC"
            )
        result[method] = _absolute(root.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--prior-authority-root", type=_absolute, required=True)
    parser.add_argument(
        "--taste-cell",
        action="append",
        required=True,
        metavar="METHOD=ROOT",
        help="Completed full-cell root; repeat to append multiple simultaneous cells.",
    )
    parser.add_argument("--t3-root", type=_absolute, required=True)
    parser.add_argument(
        "--taste-policy",
        type=_absolute,
        default=PROJECT_ROOT
        / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml",
    )
    parser.add_argument("--taste-policy-receipt", type=_absolute, required=True)
    parser.add_argument("--prepared-root", type=_absolute, required=True)
    parser.add_argument("--graph-cache-root", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = append_tastemolnet_cells(
        prior_authority_root=args.prior_authority_root,
        taste_cells=_cells(args.taste_cell),
        output_root=args.output_root,
        t3_root=args.t3_root,
        policy_path=args.taste_policy,
        policy_receipt=args.taste_policy_receipt,
        prepared_root=args.prepared_root,
        graph_cache_root=args.graph_cache_root,
        proc_root=args.proc_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(result["marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
