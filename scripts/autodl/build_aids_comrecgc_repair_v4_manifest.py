#!/usr/bin/env python3
"""Build or validate the exact external-memory AIDS repair-v4 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_aids_comrecgc_repair_v4 import (  # noqa: E402
    GENERATION_SOURCE_KEY,
    THRESHOLD_SOURCE_KEY,
    build_manifest,
    build_payload,
    publish_source_gate,
    validate_payload,
    verify_same_root_resume_failure,
    verify_source,
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
    validate.add_argument("--proc-root", type=_absolute)
    build = commands.add_parser("build")
    build.add_argument("--spec", type=_absolute, required=True)
    build.add_argument("--output", type=_absolute, required=True)
    build.add_argument("--proc-root", type=_absolute)
    source = commands.add_parser("verify-source")
    source.add_argument(
        "--source-key",
        choices=(GENERATION_SOURCE_KEY, THRESHOLD_SOURCE_KEY),
        required=True,
    )
    source.add_argument("--source-manifest", type=_absolute, required=True)
    source.add_argument("--source-controller-root", type=_absolute, required=True)
    source.add_argument("--control-root", type=_absolute, required=True)
    source.add_argument("--expected-output-root", type=_absolute, required=True)
    source.add_argument("--project-root", type=_absolute, required=True)
    source.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    source.add_argument("--output-dir", type=_absolute, required=True)
    resume = commands.add_parser("verify-resume-failure")
    resume.add_argument("--output-root", type=_absolute, required=True)
    resume.add_argument("--exit-code", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "validate":
        payload, summary = build_payload(
            spec_path=args.spec, proc_root_override=args.proc_root
        )
        result = {**summary, **validate_payload(payload)}
        marker = "[AIDS_COMRECGC_REPAIR_V4_VALIDATE_PASS]"
    elif args.action == "build":
        result = build_manifest(
            spec_path=args.spec,
            output_path=args.output,
            proc_root_override=args.proc_root,
        )
        marker = "[AIDS_COMRECGC_REPAIR_V4_BUILD_PASS]"
    elif args.action == "verify-source":
        evidence = verify_source(
            source_key=args.source_key,
            source_manifest=args.source_manifest,
            source_controller_root=args.source_controller_root,
            control_root=args.control_root,
            expected_output_root=args.expected_output_root,
            project_root=args.project_root,
            proc_root=args.proc_root,
        )
        result = publish_source_gate(
            source_key=args.source_key,
            evidence=evidence,
            output_dir=args.output_dir,
        )
        marker = f"[AIDS_COMRECGC_REPAIR_V4_SOURCE_PASS] source={args.source_key}"
    else:
        result = verify_same_root_resume_failure(
            output_root=args.output_root,
            exit_code=args.exit_code,
        )
        marker = "[AIDS_COMRECGC_REPAIR_V4_SAME_ROOT_RESUME_GATE_PASS]"
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
