#!/usr/bin/env python3
"""Build and operate the fail-closed Mut trace-off parity controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_mut_traceoff_parity_v1 import (  # noqa: E402
    assert_mut_trace_parity,
    build_manifest,
    build_payload,
    prepare_generation_resume,
    publish_common_adoption_gate,
    publish_traceoff_reference_gate,
    publish_threshold_source_gate,
    publish_traced_source_gate,
    validate_instrumentation_equivalence_gate,
    validate_payload,
    wait_for_aids_pass,
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

    source = commands.add_parser("verify-traced-source")
    source.add_argument("--source-root", type=_absolute, required=True)
    source.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    source.add_argument("--output-dir", type=_absolute, required=True)

    threshold = commands.add_parser("verify-threshold")
    threshold.add_argument("--source-manifest", type=_absolute, required=True)
    threshold.add_argument("--source-controller-root", type=_absolute, required=True)
    threshold.add_argument("--control-root", type=_absolute, required=True)
    threshold.add_argument("--expected-output-root", type=_absolute, required=True)
    threshold.add_argument("--project-root", type=_absolute, required=True)
    threshold.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    threshold.add_argument("--output-dir", type=_absolute, required=True)

    wait = commands.add_parser("wait-aids")
    wait.add_argument("--expected-controller-id", required=True)
    wait.add_argument("--expected-task-id", required=True)
    wait.add_argument("--expected-wrapper", required=True)
    wait.add_argument("--expected-manifest-sha256", required=True)
    wait.add_argument("--source-manifest", type=_absolute, required=True)
    wait.add_argument("--source-controller-root", type=_absolute, required=True)
    wait.add_argument("--control-root", type=_absolute, required=True)
    wait.add_argument("--expected-output-root", type=_absolute, required=True)
    wait.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    wait.add_argument("--poll-seconds", type=int, default=60)
    wait.add_argument("--output-dir", type=_absolute, required=True)

    parity = commands.add_parser("assert-parity")
    parity.add_argument("--reference-root", type=_absolute, required=True)
    parity.add_argument("--traced-source-root", type=_absolute, required=True)
    parity.add_argument("--expected-project-commit", required=True)
    parity.add_argument("--expected-scientific-command-sha256", required=True)
    parity.add_argument("--checkpoint-root", type=_absolute, required=True)
    parity.add_argument("--mirror-root", type=_absolute, required=True)
    parity.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parity.add_argument("--output-dir", type=_absolute, required=True)

    reference = commands.add_parser("verify-traceoff-reference")
    reference.add_argument("--reference-root", type=_absolute, required=True)
    reference.add_argument("--traced-source-root", type=_absolute, required=True)
    reference.add_argument("--expected-project-commit", required=True)
    reference.add_argument("--expected-scientific-command-sha256", required=True)
    reference.add_argument("--checkpoint-root", type=_absolute, required=True)
    reference.add_argument("--mirror-root", type=_absolute, required=True)
    reference.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    reference.add_argument("--output-dir", type=_absolute, required=True)

    common = commands.add_parser("adopt-common")
    common.add_argument("--repair-v2-output", type=_absolute, required=True)
    common.add_argument("--parity-gate", type=_absolute, required=True)
    common.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    common.add_argument("--output-dir", type=_absolute, required=True)

    resume = commands.add_parser("prepare-resume")
    resume.add_argument("--output-root", type=_absolute, required=True)
    resume.add_argument("--checkpoint-root", type=_absolute, required=True)
    resume.add_argument("--mirror-root", type=_absolute, required=True)

    instrumentation = commands.add_parser("verify-instrumentation-equivalence")
    instrumentation.add_argument("--gate", type=_absolute, required=True)
    instrumentation.add_argument(
        "--expected-legacy-inventory-sha256", required=True
    )
    instrumentation.add_argument(
        "--expected-instrumentation-inventory-sha256", required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "validate":
        payload, summary = build_payload(spec_path=args.spec)
        result = {**summary, **validate_payload(payload)}
        marker = "[MUT_TRACEOFF_CONTROLLER_VALIDATE_PASS]"
    elif args.action == "build":
        result = build_manifest(spec_path=args.spec, output_path=args.output)
        marker = "[MUT_TRACEOFF_CONTROLLER_BUILD_PASS]"
    elif args.action == "verify-traced-source":
        result = publish_traced_source_gate(
            source_root=args.source_root,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
        marker = "[MUT_TRACE_ON_SOURCE_PASS]"
    elif args.action == "verify-threshold":
        result = publish_threshold_source_gate(
            source_manifest=args.source_manifest,
            source_controller_root=args.source_controller_root,
            control_root=args.control_root,
            expected_output_root=args.expected_output_root,
            project_root=args.project_root,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
        marker = "[MUT_THRESHOLD_SOURCE_PASS]"
    elif args.action == "wait-aids":
        result = wait_for_aids_pass(
            source_manifest=args.source_manifest,
            source_controller_root=args.source_controller_root,
            control_root=args.control_root,
            expected_output_root=args.expected_output_root,
            expected_controller_id=args.expected_controller_id,
            expected_task_id=args.expected_task_id,
            expected_wrapper=args.expected_wrapper,
            expected_manifest_sha256=args.expected_manifest_sha256,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
            poll_seconds=args.poll_seconds,
        )
        marker = "[MUT_AIDS_DEPENDENCY_PASS]"
    elif args.action == "assert-parity":
        result = assert_mut_trace_parity(
            reference_root=args.reference_root,
            traced_source_root=args.traced_source_root,
            expected_project_commit=args.expected_project_commit,
            expected_scientific_command_sha256=(
                args.expected_scientific_command_sha256
            ),
            checkpoint_root=args.checkpoint_root,
            mirror_root=args.mirror_root,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
        marker = "[MUT_TRACE_PARITY_PASS]"
    elif args.action == "verify-traceoff-reference":
        result = publish_traceoff_reference_gate(
            reference_root=args.reference_root,
            traced_source_root=args.traced_source_root,
            expected_project_commit=args.expected_project_commit,
            expected_scientific_command_sha256=(
                args.expected_scientific_command_sha256
            ),
            checkpoint_root=args.checkpoint_root,
            mirror_root=args.mirror_root,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
        marker = "[MUT_TRACEOFF_REFERENCE_PASS]"
    elif args.action == "adopt-common":
        result = publish_common_adoption_gate(
            repair_v2_output=args.repair_v2_output,
            parity_gate=args.parity_gate,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
        marker = "[MUT_COMMON_ADOPTION_PASS]"
    elif args.action == "verify-instrumentation-equivalence":
        result = validate_instrumentation_equivalence_gate(
            gate_path=args.gate,
            expected_legacy_inventory_sha256=(
                args.expected_legacy_inventory_sha256
            ),
            expected_instrumentation_inventory_sha256=(
                args.expected_instrumentation_inventory_sha256
            ),
        )
        marker = "[MUT_CHECKPOINT_INSTRUMENTATION_GATE_REVALIDATED]"
    else:
        result = prepare_generation_resume(
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
            mirror_root=args.mirror_root,
        )
        marker = "[MUT_TRACEOFF_RESUME_PREPARED]"
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
