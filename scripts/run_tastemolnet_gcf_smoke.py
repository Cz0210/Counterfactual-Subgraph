#!/usr/bin/env python3
"""Thin AutoDL CLI for the release-gated TasteMolNet T7 GCF smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_gcf_smoke import (  # noqa: E402
    STAGE,
    TasteGCFSmokeReleaseDisabled,
    load_tastemolnet_gcf_verified_gate,
    run_tastemolnet_gcf_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(STAGE,), default=STAGE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        result = load_tastemolnet_gcf_verified_gate(args.output_dir)
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    else:
        if args.set != ["inference.fallback_to_heuristic=false"]:
            raise ValueError(
                "Taste T7 requires exactly --set "
                "inference.fallback_to_heuristic=false"
            )
        try:
            result = run_tastemolnet_gcf_smoke(
                output_dir=args.output_dir,
                config_path=args.config,
            )
        except TasteGCFSmokeReleaseDisabled as exc:
            print(str(exc), file=sys.stderr)
            return 78
        # This is a worker SEALED receipt, never terminal PASS.
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
