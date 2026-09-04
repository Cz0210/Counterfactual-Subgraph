#!/usr/bin/env python3
"""Stream-verify a storage-safe exact T8 bundle without extracting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_hpc_exact import validate_hpc_cli_contract  # noqa: E402
from src.baselines.globalgce_hpc_storage_safe import (  # noqa: E402
    stream_verify_storage_safe_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-packaging-commit", required=True)
    parser.add_argument("--expected-scientific-input-sha256", required=True)
    parser.add_argument(
        "--expected-partition-manifest-file-sha256",
        required=True,
        help="SHA-256 of the exact partition_manifest.json file bytes",
    )
    parser.add_argument(
        "--expected-partition-manifest-self-sha256",
        required=True,
        help="Canonical self-hash stored in partition_manifest.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_hpc_cli_contract(args.config, args.set)
    report = stream_verify_storage_safe_bundle(
        args.archive,
        receipt_path=args.receipt,
    )
    expected = {
        "archive_sha256": args.expected_archive_sha256,
        "packaging_commit": args.expected_packaging_commit,
        "scientific_input_sha256": args.expected_scientific_input_sha256,
        "partition_manifest_file_sha256": (
            args.expected_partition_manifest_file_sha256
        ),
        "partition_manifest_sha256": (
            args.expected_partition_manifest_self_sha256
        ),
    }
    mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise SystemExit(
            "storage-safe bundle expected-identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
