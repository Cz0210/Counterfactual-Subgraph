#!/usr/bin/env python3
"""Audit a strict train-only CLEAR Mutagenicity smoke candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.clear_mutagenicity_train_pool import (  # noqa: E402
    GenerationProfile,
    audit_train_pool,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--generation-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--expected-model-train-rows", type=int, default=2885)
    parser.add_argument("--expected-model-val-rows", type=int, default=355)
    parser.add_argument(
        "--expected-generation-parent-rows", type=int, default=1448
    )
    parser.add_argument("--expected-selected-parents", type=int, default=64)
    parser.add_argument(
        "--expected-generation-profile",
        choices=[profile.value for profile in GenerationProfile],
        default=GenerationProfile.SMOKE.value,
    )
    parser.add_argument(
        "--require-generation-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-target-label-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-unique-universe",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.require_target_label_zero:
        raise ValueError("Strict audit requires target label zero.")
    if not args.require_unique_universe:
        raise ValueError("Strict audit requires a unique candidate universe.")
    if not args.forbid_calibration_test:
        raise ValueError("Strict audit must forbid calibration/test.")
    teacher = TeacherSemanticScorer(args.teacher_path, device="cpu")
    if not teacher.available:
        raise RuntimeError(
            f"Mutagenicity RF teacher unavailable: {teacher.availability_reason}"
        )
    result = audit_train_pool(
        run_dir=args.run_dir,
        generation_csv=args.generation_csv,
        expected_model_train_rows=int(args.expected_model_train_rows),
        expected_model_val_rows=int(args.expected_model_val_rows),
        expected_generation_parent_rows=int(
            args.expected_generation_parent_rows
        ),
        expected_selected_parents=int(args.expected_selected_parents),
        expected_generation_profile=str(args.expected_generation_profile),
        require_generation_only=bool(args.require_generation_only),
        require_complete=bool(args.require_complete),
        teacher=teacher,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[MUTAGENICITY_CLEAR_TRAIN_POOL_AUDIT_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
