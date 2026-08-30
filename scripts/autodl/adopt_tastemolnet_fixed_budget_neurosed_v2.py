#!/usr/bin/env python3
"""Losslessly adopt a verified fixed-budget Taste NeuroSED root into managed-v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.tastemolnet_neurosed_fixed_budget_adoption import (  # noqa: E402
    copy_fixed_budget_neurosed_pass,
    inspect_fixed_budget_neurosed_pass,
    verify_fixed_budget_managed_adoption,
)
from src.utils.process_identity_v2 import require_auto_termination_disabled  # noqa: E402
from src.utils.terminal_publisher_v2 import (  # noqa: E402
    open_sealed_worker_artifact,
    verify_and_publish_sealed_attempt,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vendored-gcf-root", type=Path, required=True)
    parser.add_argument("--expected-source-inventory-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    _common(inspect_parser)
    copy_parser = commands.add_parser("copy")
    _common(copy_parser)
    copy_parser.add_argument("--artifact-root", type=Path)
    verify_parser = commands.add_parser("verify-and-publish")
    _common(verify_parser)
    verify_parser.add_argument("--sealed", type=Path, required=True)
    verify_parser.add_argument("--final-path", type=Path, required=True)
    verify_parser.add_argument("--expected-attempt-id", required=True)
    verify_parser.add_argument("--expected-generation-token")
    verify_parser.add_argument("--force-cross-filesystem", action="store_true")
    return parser


def _require_config(path: Path) -> None:
    expected = (PROJECT_ROOT / "configs/hpc.yaml").resolve(strict=True)
    if path.resolve(strict=True) != expected:
        raise ValueError("fixed-budget adoption requires configs/hpc.yaml")


def run(args: argparse.Namespace) -> int:
    _require_config(args.config)
    if args.command == "inspect":
        result = inspect_fixed_budget_neurosed_pass(
            args.source_root, vendored_gcf_root=args.vendored_gcf_root
        )
        if result["inventory_sha256"] != args.expected_source_inventory_sha256:
            raise ValueError("fixed-budget source inventory pin changed")
        print(
            json.dumps(
                {
                    "state": "VERIFIED_FIXED_BUDGET_SOURCE",
                    "source_root": result["root"],
                    "source_inventory_sha256": result["inventory_sha256"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "pass_sha256": result["pass_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    require_auto_termination_disabled()
    if args.command == "copy":
        artifact_root = args.artifact_root
        if artifact_root is None:
            raw = os.environ.get("MANAGED_ARTIFACT_ROOT")
            if not raw:
                raise ValueError("MANAGED_ARTIFACT_ROOT is required")
            artifact_root = Path(raw)
        result = copy_fixed_budget_neurosed_pass(
            source_root=args.source_root,
            artifact_root=artifact_root,
            expected_source_inventory_sha256=(
                args.expected_source_inventory_sha256
            ),
            vendored_gcf_root=args.vendored_gcf_root,
        )
        print(
            json.dumps(
                {
                    "state": "COPIED_PENDING_INDEPENDENT_MANAGED_VERIFICATION",
                    "artifact_root": str(artifact_root),
                    "source_inventory_sha256": result["inventory_sha256"],
                    "scientific_artifact_modified": False,
                },
                sort_keys=True,
            )
        )
        return 0
    with open_sealed_worker_artifact(
        args.sealed,
        expected_attempt_id=args.expected_attempt_id,
        expected_generation_token=args.expected_generation_token,
    ) as held:
        verification = verify_fixed_budget_managed_adoption(
            held,
            source_root=args.source_root,
            expected_source_inventory_sha256=(
                args.expected_source_inventory_sha256
            ),
            vendored_gcf_root=args.vendored_gcf_root,
        )
        publication = verify_and_publish_sealed_attempt(
            held,
            final_path=args.final_path,
            verification=verification,
            force_cross_filesystem=args.force_cross_filesystem,
        )
    print(
        json.dumps(
            {
                "state": "PASS",
                "final_path": str(publication.final_path),
                "attempt_id": publication.attempt_id,
                "generation_token": publication.generation_token,
                "verification_sha256": publication.verification_sha256,
                "gate_sha256": publication.gate_sha256,
                "publish_mode": publication.publish_mode,
            },
            sort_keys=True,
        )
    )
    print("[TASTE_NEUROSED_FIXED_BUDGET_MANAGED_V2_PUBLISHED]")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(
            json.dumps(
                {
                    "state": "QUARANTINED",
                    "quarantine_reason": f"{type(exc).__name__}: {exc}",
                    "science_adopted": False,
                    "downstream_released": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
