#!/usr/bin/env python3
"""Validate or build the typed AIDS disconnected-exact recovery controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_exact_recovery_controller_v1 import (  # noqa: E402
    build_controller_manifest,
    build_controller_payload,
    validate_controller_manifest,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="action", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--spec", type=_absolute, required=True)
    build = commands.add_parser("build")
    build.add_argument("--spec", type=_absolute, required=True)
    build.add_argument("--output", type=_absolute, required=True)
    generate = commands.add_parser(
        "generate-production",
        help="derive a production spec from typed adoption evidence",
    )
    generate.add_argument("--adoption-output", type=_absolute, required=True)
    generate.add_argument("--controller-parent", type=_absolute, required=True)
    generate.add_argument("--python", type=_absolute, required=True)
    generate.add_argument("--project-root", type=_absolute, default=PROJECT_ROOT)
    generate.add_argument("--controller-manifest", type=_absolute, required=True)
    generate.add_argument("--timestamp")
    generate.add_argument(
        "--thread-count",
        type=int,
        choices=range(8, 13),
        default=8,
        help="CPU worker/thread budget; the fast route starts with 8 (max 12)",
    )
    generate.add_argument("--adoption-commit")
    generate.add_argument("--controller-commit")
    generate.add_argument("--exact-runner-commit")
    generate.add_argument("--subset-runner-commit")
    generate.add_argument("--downstream-runner-commit")
    generate.add_argument("--standardization-runner-commit")
    generate.add_argument(
        "--authorize-production-deployment",
        action="store_true",
        help=(
            "explicit trusted-operator authorization; requires every release "
            "pin and controller_commit equal to the execution HEAD"
        ),
    )
    generate.add_argument("--output", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "generate-production":
        from src.utils.autodl_aids_comrecgc_exact_recovery_spec_v1 import (
            write_production_spec,
        )

        release_pins = {
            name: value
            for name, value in {
                "adoption_commit": args.adoption_commit,
                "controller_commit": args.controller_commit,
                "exact_runner_commit": args.exact_runner_commit,
                "subset_runner_commit": args.subset_runner_commit,
                "downstream_runner_commit": args.downstream_runner_commit,
                "standardization_runner_commit": (
                    args.standardization_runner_commit
                ),
            }.items()
            if value is not None
        }
        payload = write_production_spec(
            output=args.output,
            adoption_output=args.adoption_output,
            controller_parent=args.controller_parent,
            python=args.python,
            project_root=args.project_root,
            controller_manifest_path=args.controller_manifest,
            timestamp=args.timestamp,
            thread_count=args.thread_count,
            release_pins=release_pins,
            production_deployment_authorized=(
                args.authorize_production_deployment
            ),
        )
        result = {
            "status": "PASS",
            "spec_path": str(args.output),
            "cid": payload["cid"],
            "controller_root": payload["controller_root"],
            "controller_manifest_path": payload["controller_manifest_path"],
            "production_deployment_authorized": payload[
                "production_deployment_authorized"
            ],
            "missing_release_pins": [
                name
                for name, value in payload["release_pins"].items()
                if value is None
            ],
            "thread_count": int(payload["resources"]["thread_count"]),
        }
        marker = "[AIDS_EXACT_RECOVERY_PRODUCTION_SPEC_GENERATED]"
    elif args.action == "validate":
        payload = build_controller_payload(args.spec)
        result = validate_controller_manifest(payload)
        marker = "[AIDS_EXACT_RECOVERY_CONTROLLER_VALIDATE_PASS]"
    else:
        payload = build_controller_manifest(spec_path=args.spec, output_path=args.output)
        result = {
            "status": "PASS",
            "manifest_path": str(args.output),
            "release_ready": payload["release_ready"],
            "missing_release_pins": payload["missing_release_pins"],
        }
        marker = "[AIDS_EXACT_RECOVERY_CONTROLLER_BUILD_PASS]"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
