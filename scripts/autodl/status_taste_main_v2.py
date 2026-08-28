#!/usr/bin/env python3
"""Read release-v3 foundation status from compatibility main-v2 paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_tastemolnet_main_v2 import (  # noqa: E402
    DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    controller_status,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--controller-root", type=_absolute, required=True)
    parser.add_argument(
        "--max-heartbeat-age-seconds",
        type=float,
        default=DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config.resolve(strict=True) != PROJECT_ROOT / "configs/hpc.yaml":
        raise SystemExit("--config must be this checkout's configs/hpc.yaml")
    result = controller_status(
        controller_root=args.controller_root,
        max_age_seconds=args.max_heartbeat_age_seconds,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "RUNNING" else 75


if __name__ == "__main__":
    raise SystemExit(main())
