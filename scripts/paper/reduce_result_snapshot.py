#!/usr/bin/env python3
"""Reduce collected frozen prefix records into PARTIAL paper CSVs and numeric audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.paper_snapshot_reduction import export_snapshot  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-root', type=Path, required=True,
                        help='read-only root containing canonical/ and lineage/ snapshot JSON')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='fresh external artifact directory; cannot overlap snapshot or repository')
    parser.add_argument('--config', type=Path, help='optional existing runtime config; no science is configured')
    parser.add_argument('--set', action='append', default=[], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.config and not args.config.is_file():
        parser.error('config does not exist')
    if any(value != 'inference.fallback_to_heuristic=false' for value in args.set):
        parser.error('scientific overrides are not supported')
    try:
        result = export_snapshot(snapshot_root=args.snapshot_root, output_root=args.output_root,
                                 project_root=PROJECT_ROOT)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'[SNAPSHOT_REDUCTION_BLOCKED] {exc}', file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ('status', 'main_registered_cells',
                     'pending_cells', 'legacy_differing_metric_points')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
