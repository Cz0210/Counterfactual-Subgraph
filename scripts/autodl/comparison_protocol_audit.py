#!/usr/bin/env python3
"""Run the frozen, parent-unit comparison audit after the matrix reaches 16/16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.comparison_protocol_audit import (  # noqa: E402
    ComparisonAuditError,
    run_comparison_audit,
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
    parser.add_argument("--final-export-root", type=_absolute, required=True)
    parser.add_argument(
        "--frozen-contract",
        type=_absolute,
        default=PROJECT_ROOT / "configs/autodl/final_paper_evaluation_v1.json",
    )
    parser.add_argument("--output-root", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_comparison_audit(
            matrix_status=args.matrix_status,
            final_export_root=args.final_export_root,
            frozen_contract=args.frozen_contract,
            output_root=args.output_root,
        )
    except (ComparisonAuditError, FileExistsError, OSError, ValueError) as exc:
        print(
            f"[COMPARISON_PROTOCOL_AUDIT_FAILED] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[FINAL_COMBINED_AUDIT_PASS]")
    print(f"[{result['claim_status']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
