#!/usr/bin/env python3
"""Resume only the AIDS exact stage from an independently receipted checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (  # noqa: E402
    ADOPTION_STAGE,
    EXACT_STAGE,
    SUBSET_STAGE,
    _gate_path,
    _stage,
    load_bound_controller_manifest,
    validate_exact_checkpoint_adoption_receipt,
)
from src.utils.autodl_aids_comrecgc_exact_recovery_stages_v1 import (  # noqa: E402
    run_exact_stage,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=_absolute, required=True)
    parser.add_argument("--expected-progress-rows", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_progress_rows <= 0:
        raise ValueError("expected-progress-rows must be positive")
    # The independent receipt is reopened immediately before the scientific
    # stage.  run_exact_stage then reopens the same release-bound controller
    # manifest and typed predecessor gates; it cannot regenerate the pair
    # store or take the fresh path.
    adoption = validate_exact_checkpoint_adoption_receipt(
        args.manifest,
        expected_progress_rows=args.expected_progress_rows,
    )
    manifest = load_bound_controller_manifest(args.manifest)
    stage = _stage(manifest, EXACT_STAGE)
    result = run_exact_stage(
        controller_manifest=args.manifest,
        output_dir=Path(str(stage["output_dir"])),
        adoption_gate=_gate_path(manifest, ADOPTION_STAGE),
        subset_gate=_gate_path(manifest, SUBSET_STAGE),
        resume=True,
    )
    print(
        json.dumps(
            {
                "checkpoint_adoption": adoption,
                "exact_stage": result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print("[AIDS_EXACT_ADOPTED_CHECKPOINT_STAGE_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
