#!/usr/bin/env python3
"""Build, independently verify, or validate TasteGCFReleasePinsV1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tastemolnet_t7_typed_release_v1 import (  # noqa: E402
    build_t7_release_candidate,
    validate_t7_release_root,
    verify_and_publish_t7_release,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--managed-neurosed-root", type=_absolute, required=True)
    build.add_argument("--t3-root", type=_absolute, required=True)
    build.add_argument("--official-gcf-root", type=_absolute, required=True)
    build.add_argument("--neurosed-distance-threshold", type=float, required=True)
    build.add_argument("--candidate-root", type=_absolute, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--candidate-root", type=_absolute, required=True)
    verify.add_argument("--release-root", type=_absolute, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--release-root", type=_absolute, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        raise ValueError("config is unavailable")
    if args.mode == "build":
        result = build_t7_release_candidate(
            managed_neurosed_root=args.managed_neurosed_root,
            t3_root=args.t3_root,
            official_gcf_root=args.official_gcf_root,
            neurosed_distance_threshold=args.neurosed_distance_threshold,
            output_root=args.candidate_root,
        )
    elif args.mode == "verify":
        result = verify_and_publish_t7_release(
            candidate_root=args.candidate_root,
            output_root=args.release_root,
        )
    else:
        result = validate_t7_release_root(args.release_root, reopen_sources=True)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    if args.mode in {"verify", "validate"}:
        print(result["marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
