#!/usr/bin/env python3
"""Repeat, verify, and atomically publish a SEALED Taste T4 candidate."""

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
    PASS_MARKER,
    TasteT4OracleSmokeError,
    verify_and_publish_t4,
)
from src.utils.process_identity_v2 import require_auto_termination_disabled  # noqa: E402
from src.utils.autodl_tastemolnet_main_v2 import inspect_clean_git  # noqa: E402


def _config(path: str) -> Path:
    selected = Path(path)
    expected = PROJECT_ROOT / "configs/hpc.yaml"
    if selected.resolve(strict=True) != expected:
        raise argparse.ArgumentTypeError("--config must be this checkout's configs/hpc.yaml")
    info = os.lstat(selected)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise argparse.ArgumentTypeError("--config must be one physical single-link file")
    return selected


def _clean_git() -> tuple[str, str]:
    return inspect_clean_git(PROJECT_ROOT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_config, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--final-path", type=Path, required=True)
    parser.add_argument("--t3-root", type=Path, required=True)
    parser.add_argument("--graph-cache-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--expected-generation-token", required=True)
    parser.add_argument("--expected-controller-id", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-git-tree", required=True)
    parser.add_argument("--controller-launcher-receipt", type=Path, required=True)
    parser.add_argument("--controller-receipt", type=Path, required=True)
    parser.add_argument("--controller-anchor-heartbeat", type=Path, required=True)
    parser.add_argument(
        "--expected-controller-launcher-receipt-sha256", required=True
    )
    parser.add_argument("--expected-controller-receipt-sha256", required=True)
    parser.add_argument(
        "--expected-controller-anchor-heartbeat-sha256", required=True
    )
    parser.add_argument("--expected-gpu-lease-uuid", required=True)
    parser.add_argument("--expected-gpu-lease-sha256", required=True)
    parser.add_argument(
        "--controller-max-heartbeat-age-seconds", type=float, default=35
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        require_auto_termination_disabled()
        if args.set != ["inference.fallback_to_heuristic=false"]:
            raise TasteT4OracleSmokeError("fail-closed inference override is required")
        commit, tree = _clean_git()
        if commit != args.expected_git_commit or tree != args.expected_git_tree:
            raise TasteT4OracleSmokeError("verifier checkout differs from authority")
        publication, verification = verify_and_publish_t4(
            sealed_path=args.sealed,
            final_path=args.final_path,
            t3_root=args.t3_root,
            graph_cache_root=args.graph_cache_root,
            gpu_uuid=args.gpu_uuid,
            expected_attempt_id=args.expected_attempt_id,
            expected_generation_token=args.expected_generation_token,
            expected_controller_id=args.expected_controller_id,
            expected_git_commit=args.expected_git_commit,
            expected_git_tree=args.expected_git_tree,
            expected_config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            controller_launcher_receipt_path=args.controller_launcher_receipt,
            controller_receipt_path=args.controller_receipt,
            controller_anchor_heartbeat_path=args.controller_anchor_heartbeat,
            expected_controller_launcher_receipt_sha256=(
                args.expected_controller_launcher_receipt_sha256
            ),
            expected_controller_receipt_sha256=(
                args.expected_controller_receipt_sha256
            ),
            expected_controller_anchor_heartbeat_sha256=(
                args.expected_controller_anchor_heartbeat_sha256
            ),
            expected_gpu_lease_uuid=args.expected_gpu_lease_uuid,
            expected_gpu_lease_sha256=args.expected_gpu_lease_sha256,
            controller_max_heartbeat_age_seconds=(
                args.controller_max_heartbeat_age_seconds
            ),
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "final_path": str(publication.final_path),
                    "attempt_id": publication.attempt_id,
                    "generation_token": publication.generation_token,
                    "verification_sha256": publication.verification_sha256,
                    "gate_sha256": publication.gate_sha256,
                    "publish_mode": publication.publish_mode,
                    **verification,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        print(PASS_MARKER, flush=True)
        return 0
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(
            f"T4_ORACLE_SMOKE_VERIFIER_BLOCKED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
