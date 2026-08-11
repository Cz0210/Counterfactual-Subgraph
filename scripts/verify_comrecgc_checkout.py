#!/usr/bin/env python3
"""Fail-closed verification for an offline pinned COMRECGC checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import UPSTREAM_COMMIT  # noqa: E402
from src.baselines.comrecgc.upstream import (  # noqa: E402
    imported_upstream,
    validate_upstream_checkout,
)


REQUIRED_FILES = ("comrecgc.py", "common_recourse.py", "data.py", "gnn.py")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def verify_checkout(
    root: str | Path,
    *,
    expected_commit: str = UPSTREAM_COMMIT,
    validate_imports: bool = False,
) -> dict[str, Any]:
    source = Path(root).expanduser().resolve()
    if str(expected_commit) != UPSTREAM_COMMIT:
        raise ValueError(
            "Requested COMRECGC commit differs from the frozen project contract: "
            f"actual={expected_commit}, expected={UPSTREAM_COMMIT}."
        )
    validated = validate_upstream_checkout(source)
    commit = _git(validated, "rev-parse", "HEAD")
    identities = {
        name: _sha(validated / name)
        for name in REQUIRED_FILES
        if (validated / name).is_file()
    }
    if set(identities) != set(REQUIRED_FILES):
        raise FileNotFoundError("Pinned COMRECGC checkout is incomplete.")
    manifest_path = validated / "vendor_manifest.json"
    manifest_match: bool | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_match = bool(
            str(manifest.get("commit") or "") == commit
            and manifest.get("key_file_sha256") == identities
            and manifest.get("read_only_usage") is True
        )
        if not manifest_match:
            raise ValueError("COMRECGC vendor manifest integrity check failed.")
    imported = False
    if validate_imports:
        with imported_upstream(validated) as modules:
            imported = all(name in modules for name in ("comrecgc", "common_recourse"))
        if not imported:
            raise ImportError("Pinned COMRECGC imports are incomplete.")
    return {
        "root": str(validated),
        "expected_commit": str(expected_commit),
        "actual_commit": commit,
        "commit_match": commit == str(expected_commit),
        "required_files": identities,
        "vendor_manifest_present": manifest_path.is_file(),
        "vendor_manifest_match": manifest_match,
        "import_pass": imported if validate_imports else None,
        "network_required": False,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-commit", default=UPSTREAM_COMMIT)
    parser.add_argument("--validate-imports", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = verify_checkout(
        args.root,
        expected_commit=args.expected_commit,
        validate_imports=bool(args.validate_imports),
    )
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[COMRECGC_CHECKOUT_GATE_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
