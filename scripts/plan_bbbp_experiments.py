#!/usr/bin/env python3
"""Validate and render BBBP experiment plans without submitting jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.bbbp_framework import (  # noqa: E402
    ALL_PLAN_NAMES,
    load_plans,
    plans_json,
    render_future_shell,
    render_plan_text,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        choices=(*ALL_PLAN_NAMES, "all"),
        default="all",
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--emit-json",
        nargs="?",
        const="-",
        metavar="PATH",
        help="Print JSON, or write it to PATH. This never creates experiment outputs.",
    )
    parser.add_argument(
        "--emit-shell",
        nargs="?",
        const="-",
        metavar="PATH",
        help="Print future exp_sbatch commands, or write them to PATH; never execute them.",
    )
    return parser


def _emit(value: str, destination: str | None) -> None:
    if destination is None:
        return
    if destination == "-":
        print(value, end="")
        return
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = load_plans(args.plan, project_root=args.project_root)
    if args.emit_json is not None:
        _emit(json.dumps(plans_json(plans), indent=2, sort_keys=True) + "\n", args.emit_json)
    if args.emit_shell is not None:
        _emit(render_future_shell(plans), args.emit_shell)
    if args.emit_json is None and args.emit_shell is None:
        print(render_plan_text(plans), end="")
    print(
        "[BBBP_EXPERIMENT_PLAN_VALID] "
        "submission_performed=false registry_written=false formal_output_written=false",
        file=sys.stderr if args.emit_json == "-" or args.emit_shell == "-" else sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
