#!/usr/bin/env python3
"""Recover and queue the completed BACE GlobalGCE affine-edge terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bace_globalgce_terminal_recovery import (  # noqa: E402
    PASS_MARKER,
    build_recovery_controller_fragment,
    recover_failed_bace_globalgce_terminal,
)
from src.baselines.bace_gnn_baseline_generic_adapter import (  # noqa: E402
    atomic_write_generic_fragment,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _recovery_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--failed-controller-root", type=_absolute, required=True)
    parser.add_argument("--source-round-root", type=_absolute, required=True)
    parser.add_argument("--source-manifest", type=_absolute, required=True)
    parser.add_argument("--native-train-csv", type=_absolute, required=True)
    parser.add_argument("--official-root", type=_absolute, required=True)
    parser.add_argument("--gnn-checkpoint", type=_absolute, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    recover = commands.add_parser("recover")
    _recovery_arguments(recover)
    recover.add_argument("--output-dir", type=_absolute, required=True)
    recover.add_argument("--proc-root", type=_absolute, default=Path("/proc"))

    fragment = commands.add_parser("build-fragment")
    _recovery_arguments(fragment)
    fragment.add_argument("--python", type=_absolute, required=True)
    fragment.add_argument("--project-root", type=_absolute, required=True)
    fragment.add_argument("--output-root", type=_absolute, required=True)
    fragment.add_argument("--dataset-dir", type=_absolute, required=True)
    fragment.add_argument("--calibration-split", type=_absolute, required=True)
    fragment.add_argument("--test-split", type=_absolute, required=True)
    fragment.add_argument("--molclr-root", type=_absolute, required=True)
    fragment.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    fragment.add_argument("--neurosed-checkpoint", type=_absolute, required=True)
    fragment.add_argument("--fragment-output", type=_absolute, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "recover":
        result = recover_failed_bace_globalgce_terminal(
            failed_controller_root=args.failed_controller_root,
            source_round_root=args.source_round_root,
            source_manifest=args.source_manifest,
            native_train_csv=args.native_train_csv,
            official_root=args.official_root,
            gnn_checkpoint=args.gnn_checkpoint,
            output_dir=args.output_dir,
            proc_root=args.proc_root,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        print(PASS_MARKER, flush=True)
        return 0

    fragment = build_recovery_controller_fragment(
        python=args.python,
        project_root=args.project_root,
        output_root=args.output_root,
        failed_controller_root=args.failed_controller_root,
        source_round_root=args.source_round_root,
        source_manifest=args.source_manifest,
        native_train_csv=args.native_train_csv,
        official_root=args.official_root,
        gnn_checkpoint=args.gnn_checkpoint,
        dataset_dir=args.dataset_dir,
        calibration_split=args.calibration_split,
        test_split=args.test_split,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        neurosed_checkpoint=args.neurosed_checkpoint,
    )
    destination = atomic_write_generic_fragment(args.fragment_output, fragment)
    result = {
        "status": "PASS",
        "fragment": str(destination),
        "task_ids": [task["id"] for task in fragment["tasks"]],
        "root_task_ids": fragment["root_task_ids"],
        "terminal_task_ids": fragment["terminal_task_ids"],
        "retraining_forbidden": True,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GLOBALGCE_TERMINAL_RECOVERY_FRAGMENT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
