#!/usr/bin/env python3
"""Select Mutagenicity WNode-aware nested prefixes on calibration only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.mutagenicity_wnode_selector import (  # noqa: E402
    DEFAULT_COST_CAP_QUANTILE,
    DEFAULT_PREFIX_WEIGHTS,
    DEFAULT_THETA_STAR_QUANTILE,
    DEFAULT_THRESHOLD_QUANTILES,
    DEFAULT_THRESHOLD_WEIGHTS,
    run_mutagenicity_wnode_selector,
)


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(
            float(token.strip())
            for token in str(value).split(",")
            if token.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid comma-separated floats: {value}") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return parsed


def _render(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--table-k", type=int, default=10)
    parser.add_argument(
        "--threshold-quantiles",
        type=_csv_floats,
        default=DEFAULT_THRESHOLD_QUANTILES,
    )
    parser.add_argument(
        "--threshold-weights",
        type=_csv_floats,
        default=DEFAULT_THRESHOLD_WEIGHTS,
    )
    parser.add_argument(
        "--theta-star-quantile",
        type=float,
        default=DEFAULT_THETA_STAR_QUANTILE,
    )
    parser.add_argument(
        "--cost-cap-quantile",
        type=float,
        default=DEFAULT_COST_CAP_QUANTILE,
    )
    parser.add_argument(
        "--prefix-weights",
        type=_csv_floats,
        default=DEFAULT_PREFIX_WEIGHTS,
    )
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--local-swap-passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--forbid-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {output_dir}")
    print("===== MUTAGENICITY WNODE PREFIX SELECTOR =====")
    print(f"matrix_run_dir={Path(args.matrix_run_dir).expanduser().resolve()}")
    print(f"output_dir={output_dir}")
    print(f"top_k={args.top_k}")
    print(f"table_k={args.table_k}")
    print(f"threshold_quantiles={_render(args.threshold_quantiles)}")
    print(f"threshold_weights={_render(args.threshold_weights)}")
    print(f"theta_star_quantile={args.theta_star_quantile}")
    print(f"cost_cap_quantile={args.cost_cap_quantile}")
    print(f"prefix_weights={_render(args.prefix_weights)}")
    print(f"parent_limit={args.parent_limit}")
    print(f"candidate_limit={args.candidate_limit}")
    print(f"local_swap_passes={args.local_swap_passes}")
    print(f"seed={args.seed}")
    print(f"forbid_test={args.forbid_test}")
    summary = run_mutagenicity_wnode_selector(
        matrix_run_dir=args.matrix_run_dir,
        output_dir=output_dir,
        top_k=int(args.top_k),
        table_k=int(args.table_k),
        threshold_quantiles=args.threshold_quantiles,
        threshold_weights=args.threshold_weights,
        theta_star_quantile=float(args.theta_star_quantile),
        cost_cap_quantile=float(args.cost_cap_quantile),
        prefix_weights=args.prefix_weights,
        parent_limit=int(args.parent_limit),
        candidate_limit=int(args.candidate_limit),
        local_swap_passes=int(args.local_swap_passes),
        seed=int(args.seed),
        forbid_test=bool(args.forbid_test),
    )
    print(f"selected_variant={summary['selected_variant']}")
    print(f"theta_star={summary['theta_star']}")
    print(f"cost_cap={summary['cost_cap']}")
    print("[MUTAGENICITY_WNODE_PREFIX_SELECTOR_BUILD_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
