#!/usr/bin/env python3
"""Independently verify exact TasteMolNet T12 10k/20k generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_gcf_full_resume import (  # noqa: E402
    TasteGCFFullResumeError,
)
from src.baselines.tastemolnet_gcf_full_verify import (  # noqa: E402
    GENERATION_PASS_MARKER,
    verify_t12_generation,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or Path(path.absolute()) != path:
        raise argparse.ArgumentTypeError("path must be normalized and absolute")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--production-root", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--formal-checkpoint-cadence", action="store_true")
    args = parser.parse_args(argv)
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml").resolve(
        strict=True
    ):
        raise TasteGCFFullResumeError("T12 verifier requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGCFFullResumeError(
            "T12 verifier requires fail-closed inference override"
        )
    checkpoint_cursors = None
    if args.formal_checkpoint_cadence:
        from src.utils.tastemolnet_t12_formal_profile_v1 import (
            FORMAL_PRODUCTION_CHECKPOINT_CURSORS,
            configure_t12_formal_production_profile,
        )

        configure_t12_formal_production_profile()
        checkpoint_cursors = FORMAL_PRODUCTION_CHECKPOINT_CURSORS
    result = verify_t12_generation(
        production_root=args.production_root,
        verification_root=args.output_root,
        checkpoint_cursors=checkpoint_cursors,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print(GENERATION_PASS_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
