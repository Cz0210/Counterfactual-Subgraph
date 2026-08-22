#!/usr/bin/env python3
"""Publish the frozen Mutagenicity GCF result in the common cell schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.mut_gcf_legacy_standardization import (  # noqa: E402
    MutGcfStandardizationError,
    PASS_MARKER,
    standardize_mut_gcf_legacy_cell,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--heldout-root", type=_absolute, required=True)
    parser.add_argument("--frozen-root", type=_absolute, required=True)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    args = parser.parse_args(argv)
    try:
        result = standardize_mut_gcf_legacy_cell(
            heldout_root=args.heldout_root,
            frozen_root=args.frozen_root,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
    except (MutGcfStandardizationError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(
            f"[MUT_GCF_LEGACY_STANDARDIZATION_FAILED] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    print(PASS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
