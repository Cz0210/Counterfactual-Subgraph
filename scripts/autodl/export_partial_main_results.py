#!/usr/bin/env python3
"""Render PARTIAL AIDS/BACE tables and figures without writing matrix or paper."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.partial_main_results import export_partial_results  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--matrix-authority-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.config and not Path(args.config).is_file():
        parser.error("config does not exist")
    if any(v != "inference.fallback_to_heuristic=false" for v in args.set):
        parser.error("unsupported scientific override")
    try:
        result = export_partial_results(matrix_authority_state=args.matrix_authority_state, output_root=args.output_root, project_root=PROJECT_ROOT)
    except (OSError, ValueError) as exc:
        print(f"[PARTIAL_EXPORT_BLOCKED] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ("status", "matrix_complete_cells", "rendered_cells", "matrix_status_sha256")}, indent=2))
    print("[AIDS_BACE_PARTIAL_PRESENTATION_EXPORTED]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
