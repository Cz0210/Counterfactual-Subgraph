#!/usr/bin/env python3
"""Create immutable preregistration records for COMRECGC recovery stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.preregistration import (  # noqa: E402
    write_aids_density_preregistration,
    write_aids_native_full_preregistration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    density = subparsers.add_parser("aids-density")
    density.add_argument("--existing-audit-path", required=True)
    density.add_argument("--output-path", required=True)
    native = subparsers.add_parser("aids-native-full")
    native.add_argument("--project-root", default=str(PROJECT_ROOT))
    native.add_argument("--upstream-root", default="external/COMRECGC")
    native.add_argument("--output-path", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "aids-density":
        result = write_aids_density_preregistration(
            existing_audit_path=args.existing_audit_path,
            output_path=args.output_path,
        )
    else:
        upstream = Path(args.upstream_root)
        if not upstream.is_absolute():
            upstream = Path(args.project_root) / upstream
        result = write_aids_native_full_preregistration(
            project_root=args.project_root,
            upstream_root=upstream,
            output_path=args.output_path,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
