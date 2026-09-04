#!/usr/bin/env python3
"""Read-only status for the predeployed T8 HPC import/T13 chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.t8_hpc_t13_successor_v1 import (  # noqa: E402
    read_json,
    validate_spec_set,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--spec-root", type=_absolute, required=True)
    parser.add_argument("--heartbeat", type=_absolute, required=True)
    parser.add_argument("--release", type=_absolute, required=True)
    args = parser.parse_args(argv)
    specs = validate_spec_set(args.spec_root, check_files=False)
    payload = {
        "spec_set_sha256": specs["manifest"]["spec_set_sha256"],
        "import_status": specs["import"]["status"],
        "t13_status": specs["t13"]["status"],
        "publisher_status": specs["publisher"]["status"],
        "heartbeat": (
            read_json(args.heartbeat, label="owner heartbeat")
            if args.heartbeat.is_file()
            else None
        ),
        "release": (
            read_json(args.release, label="T13 release")
            if args.release.is_file()
            else None
        ),
        "import_pass": (
            Path(specs["import"]["output_root"]) / "HPC_IMPORT_PASS"
        ).is_file(),
        "t13_pass": (
            Path(specs["t13"]["output_root"]) / "PASS"
        ).is_file(),
        "locator_present": Path(specs["publisher"]["terminal_root_locator"]).is_file(),
        "matrix_write_enabled": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
