#!/usr/bin/env python3
"""Build or validate the bounded four-by-four AutoDL repair continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_four_by_four_repair import (  # noqa: E402
    build_repair_manifest,
    build_repair_payload,
    publish_source_adoption,
    validate_repair_payload,
    verify_comrecgc_generation_terminal,
    verify_controller_terminal,
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

    controller = commands.add_parser("verify-controller-terminal")
    controller.add_argument("--source-name", required=True)
    controller.add_argument("--source-manifest", type=_absolute, required=True)
    controller.add_argument("--source-controller-root", type=_absolute, required=True)
    controller.add_argument("--task-id", required=True)
    controller.add_argument("--expected-output-root", type=_absolute, required=True)
    controller.add_argument("--required-file", action="append", required=True)
    controller.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    controller.add_argument("--output-dir", type=_absolute, required=True)

    generation = commands.add_parser("verify-generation-terminal")
    generation.add_argument("--source-name", required=True)
    generation.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    generation.add_argument("--expected-output-root", type=_absolute, required=True)
    generation.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    generation.add_argument("--output-dir", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "validate":
        payload, build_summary = build_repair_payload(
            spec_path=args.spec, proc_root_override=args.proc_root
        )
        result = {**build_summary, **validate_repair_payload(payload)}
        marker = "[FOUR_BY_FOUR_REPAIR_MANIFEST_VALIDATE_PASS]"
    elif args.action == "build":
        result = build_repair_manifest(
            spec_path=args.spec,
            output_path=args.output,
            proc_root_override=args.proc_root,
        )
        marker = "[FOUR_BY_FOUR_REPAIR_MANIFEST_BUILD_PASS]"
    elif args.action == "verify-controller-terminal":
        evidence = verify_controller_terminal(
            source_manifest=args.source_manifest,
            source_controller_root=args.source_controller_root,
            task_id=args.task_id,
            expected_output_root=args.expected_output_root,
            required_files=args.required_file,
            proc_root=args.proc_root,
        )
        result = publish_source_adoption(
            name=args.source_name, evidence=evidence, output_dir=args.output_dir
        )
        marker = (
            "[FOUR_BY_FOUR_REPAIR_SOURCE_ADOPTION_PASS] "
            f"source={args.source_name}"
        )
    else:
        evidence = verify_comrecgc_generation_terminal(
            dataset=args.dataset,
            expected_output_root=args.expected_output_root,
            proc_root=args.proc_root,
        )
        result = publish_source_adoption(
            name=args.source_name, evidence=evidence, output_dir=args.output_dir
        )
        marker = (
            "[FOUR_BY_FOUR_REPAIR_SOURCE_ADOPTION_PASS] "
            f"source={args.source_name}"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
