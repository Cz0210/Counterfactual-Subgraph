#!/usr/bin/env python3
"""Build the aggregate-only Taste T4 candidate; never publish terminal state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_t4_oracle_smoke_v2 import (  # noqa: E402
    TasteT4OracleSmokeError,
    build_t4_candidate,
)
from src.utils.process_identity_v2 import require_auto_termination_disabled  # noqa: E402


def _config(path: str) -> Path:
    selected = Path(path)
    expected = PROJECT_ROOT / "configs/hpc.yaml"
    if selected.resolve(strict=True) != expected:
        raise argparse.ArgumentTypeError("--config must be this checkout's configs/hpc.yaml")
    info = os.lstat(selected)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise argparse.ArgumentTypeError("--config must be one physical single-link file")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_config, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--t3-root", type=Path, required=True)
    parser.add_argument("--graph-cache-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--artifact-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        require_auto_termination_disabled()
        if args.set != ["inference.fallback_to_heuristic=false"]:
            raise TasteT4OracleSmokeError("fail-closed inference override is required")
        artifact_root = args.artifact_root or os.environ.get("MANAGED_ARTIFACT_ROOT")
        attempt_id = os.environ.get("MANAGED_ATTEMPT_ID")
        generation_token = os.environ.get("MANAGED_GENERATION_TOKEN")
        if not artifact_root or not attempt_id or not generation_token:
            raise TasteT4OracleSmokeError("managed-v2 worker environment is incomplete")
        result = build_t4_candidate(
            t3_root=args.t3_root,
            graph_cache_root=args.graph_cache_root,
            artifact_root=artifact_root,
            attempt_id=attempt_id,
            generation_token=generation_token,
            gpu_uuid=args.gpu_uuid,
            batch_size=args.batch_size,
        )
        result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except (TasteT4OracleSmokeError, OSError, ValueError) as exc:
        print(f"T4_ORACLE_SMOKE_WORKER_BLOCKED: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
