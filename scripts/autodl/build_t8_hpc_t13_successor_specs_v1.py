#!/usr/bin/env python3
"""Predeploy the T8 HPC import, T13 science, and unique publisher specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.t8_hpc_t13_successor_v1 import build_spec_set  # noqa: E402


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=None, help=argparse.SUPPRESS)
    result.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    for name in (
        "repo-root",
        "output-root",
        "relay-import-parent",
        "import-output-root",
        "t13-output-root",
        "t13-locator",
        "matrix-authority-root",
        "publisher-lease-path",
        "gpu-lease-path",
        "gnn-checkpoint",
        "train-csv",
        "calibration-csv",
        "test-csv",
        "official-root",
        "molclr-root",
        "molclr-checkpoint",
        "threshold-contract",
        "wnode-cache-db",
        "node-embedding-cache-dir",
    ):
        result.add_argument(f"--{name}", type=_absolute, required=True)
    result.add_argument("--python", required=True)
    result.add_argument("--expected-hpc-execution-commit", required=True)
    result.add_argument("--expected-scientific-input-sha256", required=True)
    result.add_argument("--expected-partition-manifest-sha256", required=True)
    result.add_argument("--gpu-index", type=int, default=1)
    result.add_argument("--gpu-uuid")
    result.add_argument("--import-attempt-id")
    result.add_argument("--t13-attempt-id")
    result.add_argument(
        "--verified-import-adoption", action="store_true",
        help="Reuse a deep-PASS matching HPC import in a fresh T13 spec; do not re-import.",
    )
    result.add_argument(
        "--publisher-id", default="taste-globalgce-final16-canonical"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.config not in (None, "configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml when supplied")
    if args.set not in ([], ["inference.fallback_to_heuristic=false"]):
        raise SystemExit("unsupported --set override")
    values = vars(args)
    values.pop("config")
    values.pop("set")
    result = build_spec_set(**values)
    print(json.dumps(result["manifest"], sort_keys=True), flush=True)
    print("[T8_HPC_IMPORT_T13_SUCCESSORS_PREDEPLOYED]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
