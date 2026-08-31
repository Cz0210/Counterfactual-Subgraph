#!/usr/bin/env python3
"""Adopt completed Mut exact science and run only downstream postprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.run_comrecgc_standardized_continuation import (  # noqa: E402
    ContinuationInputs,
    _require_directory,
    _require_file,
)
from src.utils.autodl_mut_comrecgc_exact_postprocess_v1 import (  # noqa: E402
    PASS_MARKER,
    run_mut_exact_postprocess,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def _absolute_output(value: str) -> Path:
    """Keep the requested final component visible for the no-symlink gate."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--source-generation-root", type=_absolute, required=True)
    parser.add_argument("--upstream-root", type=_absolute, required=True)
    parser.add_argument("--dataset-dir", type=_absolute, required=True)
    parser.add_argument("--distance-checkpoint", type=_absolute, required=True)
    parser.add_argument("--dataset-csv", type=_absolute, required=True)
    parser.add_argument("--teacher-path", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--thresholds-path", type=_absolute, required=True)
    parser.add_argument("--exact-adoption-receipt", type=_absolute, required=True)
    parser.add_argument("--common-root", type=_absolute, required=True)
    parser.add_argument("--trace-parity", type=_absolute, required=True)
    parser.add_argument("--prior-matrix-root", type=_absolute, required=True)
    parser.add_argument("--matrix-output-root", type=_absolute_output, required=True)
    parser.add_argument("--output-root", type=_absolute_output, required=True)
    parser.add_argument("--proc-root", type=_absolute, default=Path("/proc"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if Path(args.config) != Path("configs/hpc.yaml"):
        raise SystemExit("--config must be configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise SystemExit(
            "Mut exact postprocess requires exactly "
            "--set inference.fallback_to_heuristic=false"
        )
    inputs = ContinuationInputs(
        dataset="mutagenicity",
        source_generation_root=_require_directory(args.source_generation_root),
        upstream_root=_require_directory(args.upstream_root),
        dataset_dir=_require_directory(args.dataset_dir),
        source_csv=None,
        distance_checkpoint=_require_file(args.distance_checkpoint),
        dataset_csv=_require_file(args.dataset_csv),
        teacher_path=_require_file(args.teacher_path),
        molclr_root=_require_directory(args.molclr_root),
        molclr_checkpoint=_require_file(args.molclr_checkpoint),
        thresholds_path=_require_file(args.thresholds_path),
        output_root=args.output_root,
        device="cpu",
        theta_star=None,
        cost_cap=None,
        common_recourse_engine="adopted_exact_read_only_v1",
    )
    result = run_mut_exact_postprocess(
        inputs=inputs,
        exact_adoption_receipt=args.exact_adoption_receipt,
        common_root=args.common_root,
        trace_parity_path=args.trace_parity,
        prior_matrix_root=args.prior_matrix_root,
        matrix_output_root=args.matrix_output_root,
        resume=args.resume,
        proc_root=args.proc_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(PASS_MARKER, flush=True)
    print(result["matrix_complete_cells"], "/ 16", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
