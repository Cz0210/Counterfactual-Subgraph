#!/usr/bin/env python3
"""Build a fresh content-addressed policy-path overlay for T11 publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_t11_policy_path_relocation import (  # noqa: E402
    T11PolicyPathRelocationError,
    build_t11_policy_path_relocation,
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
    parser.add_argument("--current-policy", type=_absolute, required=True)
    parser.add_argument("--policy-receipt", type=_absolute, required=True)
    parser.add_argument("--prepared-root", type=_absolute, required=True)
    parser.add_argument("--graph-cache-root", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_t11_policy_path_relocation(
            current_policy_path=args.current_policy,
            policy_receipt=args.policy_receipt,
            prepared_root=args.prepared_root,
            graph_cache_root=args.graph_cache_root,
            output_root=args.output_root,
        )
    except (OSError, ValueError, T11PolicyPathRelocationError) as exc:
        print(f"TASTE_T11_POLICY_PATH_RELOCATION_FAILED: {exc}", flush=True)
        return 65
    payload = result.publication_evidence()
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print("[TASTE_T11_POLICY_PATH_RELOCATION_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
