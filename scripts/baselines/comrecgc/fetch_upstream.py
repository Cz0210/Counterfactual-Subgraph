#!/usr/bin/env python3
"""Fetch the license-separated COMRECGC checkout at its pinned commit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_COMMIT_DATE,
    UPSTREAM_URL,
    write_json,
)
from src.baselines.comrecgc.upstream import validate_upstream_checkout  # noqa: E402


def _run(argv: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(argv, cwd=cwd, check=True, timeout=600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default="external/COMRECGC")
    parser.add_argument(
        "--provenance-output",
        default="outputs/hpc/baselines/comrecgc/provenance/upstream.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    checkout = (PROJECT_ROOT / args.checkout).resolve()
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", UPSTREAM_URL, str(checkout)])
    if not (checkout / ".git").exists():
        raise RuntimeError(f"Refusing to replace non-Git path: {checkout}")
    _run(["git", "fetch", "origin", UPSTREAM_COMMIT], cwd=checkout)
    _run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=checkout)
    validate_upstream_checkout(checkout)
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = {
        "repository": UPSTREAM_URL,
        "commit": UPSTREAM_COMMIT,
        "commit_date": UPSTREAM_COMMIT_DATE,
        "dirty": False,
        "runtime_data_untracked": bool(result.stdout.strip()),
        "license": "no_clear_redistribution_license_detected",
        "source_committed_to_project": False,
        "checkout": str(checkout),
        "fetch_time": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
    }
    output = (PROJECT_ROOT / args.provenance_output).resolve()
    write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
