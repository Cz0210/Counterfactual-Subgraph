#!/usr/bin/env python3
"""Export or schedule the frozen four-method by four-dataset main results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.four_by_four_export_tasks import (  # noqa: E402
    EXPORT_LOG_MARKER,
    atomic_write_fragment,
    build_export_task_fragment,
)
from src.eval.four_by_four_main_results import (  # noqa: E402
    MainResultsError,
    export_main_results,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    export = subparsers.add_parser("export", help="Audit 16 cells and render final outputs")
    export.add_argument("--config", default=None, help=argparse.SUPPRESS)
    export.add_argument("--matrix-status", type=_absolute, required=True)
    export.add_argument("--output-root", type=_absolute, required=True)
    export.add_argument("--project-root", type=_absolute, default=PROJECT_ROOT)
    export.add_argument("--require-complete", action="store_true")

    fragment = subparsers.add_parser(
        "task-fragment", help="Build the generic persistent-controller task fragment"
    )
    fragment.add_argument("--config", default=None, help=argparse.SUPPRESS)
    fragment.add_argument("--controller-id", required=True)
    fragment.add_argument("--dependency-contract", type=_absolute, required=True)
    fragment.add_argument("--output-root", type=_absolute, required=True)
    fragment.add_argument("--fragment-output", type=_absolute, required=True)
    fragment.add_argument("--priority", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "task-fragment":
            payload = build_export_task_fragment(
                controller_id=args.controller_id,
                dependency_contract=args.dependency_contract,
                output_root=args.output_root,
                priority=args.priority,
            )
            destination = atomic_write_fragment(args.fragment_output, payload)
            task = payload["tasks"][0]
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "fragment": str(destination),
                        "task_id": task["id"],
                        "dependency_count": len(task["depends_on"]),
                        "required_log_marker": task["required_log_marker"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            print("[FOUR_BY_FOUR_EXPORT_TASK_FRAGMENT_PASS]")
            return 0

        result = export_main_results(
            matrix_status=args.matrix_status,
            output_root=args.output_root,
            project_root=args.project_root,
        )
        print(
            json.dumps(
                {
                    "status": "PASS" if result.complete else "BLOCKED_INCOMPLETE_MATRIX",
                    "output_root": str(result.output_root),
                    "matrix_complete_cells": result.matrix_complete_cells,
                    "generated_file_count": len(result.generated_files),
                    "blocked_reasons": list(result.blocked_reasons),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if result.complete:
            print(EXPORT_LOG_MARKER)
            return 0
        print("[FOUR_BY_FOUR_EXPORT_BLOCKED_INCOMPLETE]")
        return 3 if args.require_complete else 0
    except (MainResultsError, FileExistsError, OSError, ValueError) as exc:
        print(f"[FOUR_BY_FOUR_EXPORT_FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
