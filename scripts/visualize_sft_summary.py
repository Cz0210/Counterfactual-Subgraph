#!/usr/bin/env python3
"""Create presentation-ready visualizations for the SFT-stage project summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.sft_visualization import create_sft_summary_figures


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "hpc" / "figures" / "sft_summary"
DEFAULT_SMILES = "CC1=CC=C(C=C1)S(=O)(=O)N*"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the presentation figures will be saved.",
    )
    parser.add_argument(
        "--smiles",
        default=DEFAULT_SMILES,
        help="Example capped fragment SMILES to render with RDKit.",
    )
    parser.add_argument(
        "--base-validity",
        type=float,
        default=70.0,
        help="Base-model validity rate in percent.",
    )
    parser.add_argument(
        "--base-capping",
        type=float,
        default=0.0,
        help="Base-model capping rate in percent. Default is 0.0 as a configurable presentation baseline.",
    )
    parser.add_argument(
        "--sft-validity",
        type=float,
        default=90.0,
        help="SFT-model validity rate in percent.",
    )
    parser.add_argument(
        "--sft-capping",
        type=float,
        default=100.0,
        help="SFT-model capping rate in percent.",
    )
    parser.add_argument(
        "--final-accuracy",
        type=float,
        default=87.8,
        help="Final token accuracy. Values <= 1.0 are interpreted as fractions; values > 1.0 are treated as percentages.",
    )
    parser.add_argument(
        "--training-epochs",
        type=float,
        default=1.78,
        help="Approximate number of epochs used for the simulated convergence plot.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    figures = create_sft_summary_figures(
        output_dir=output_dir,
        smiles=args.smiles,
        final_accuracy=args.final_accuracy,
        training_epochs=args.training_epochs,
        base_validity=args.base_validity,
        base_capping=args.base_capping,
        sft_validity=args.sft_validity,
        sft_capping=args.sft_capping,
    )

    print("SFT summary figures generated successfully.")
    print(f"Output directory: {output_dir}")
    for name, path in figures.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
