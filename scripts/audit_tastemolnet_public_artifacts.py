#!/usr/bin/env python3
"""Audit a fresh, aggregate-only TasteMolNet public report bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from src.utils.tastemolnet_public_artifacts import (
    AUDIT_MARKER,
    TastePublicArtifactError,
    audit_tastemolnet_public_artifacts,
)
from src.utils.tastemolnet_research_policy import TasteResearchPolicyError


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _write_fresh_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"audit output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--public-root", required=True, type=_absolute)
    parser.add_argument("--policy", required=True, type=_absolute)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--prepared-root", required=True, type=_absolute)
    parser.add_argument("--graph-cache-root", required=True, type=_absolute)
    parser.add_argument("--output", required=True, type=_absolute)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output == args.public_root or args.public_root in args.output.parents:
            raise TastePublicArtifactError(
                "audit output must remain outside the manifest-closed public root"
            )
        result = audit_tastemolnet_public_artifacts(
            public_root=args.public_root,
            policy_path=args.policy,
            expected_policy_sha256=args.expected_policy_sha256,
            prepared_root=args.prepared_root,
            graph_cache_root=args.graph_cache_root,
        )
        _write_fresh_json(args.output, result)
    except (
        FileExistsError,
        OSError,
        TastePublicArtifactError,
        TasteResearchPolicyError,
        ValueError,
    ) as exc:
        print(f"TASTEMOLNET_PUBLIC_ARTIFACT_AUDIT_FAILED: {exc}", flush=True)
        return 65
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"[{AUDIT_MARKER}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
