#!/usr/bin/env python3
"""Evaluate an existing Mutagenicity RF teacher on all four fixed splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.mutagenicity_rf_teacher import (
    FeatureConfig,
    evaluate_model,
    load_all_splits,
    load_bundle,
    split_manifest,
    write_evaluation_artifacts,
    write_json,
)


DEFAULT_DATA_DIR = Path("outputs/hpc/datasets/final/mutagenicity_v1_processed")
DEFAULT_MODEL_PATH = Path("outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    return parser


def evaluate_teacher(args: argparse.Namespace) -> Path:
    bundle = load_bundle(args.model_path)
    feature_config = FeatureConfig(
        radius=int(bundle["fingerprint_radius"]),
        n_bits=int(bundle["fingerprint_bits"]),
    )
    datasets = load_all_splits(
        args.data_dir,
        feature_config=feature_config,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
    )
    metrics, predictions = evaluate_model(bundle["model"], datasets)
    manifest = split_manifest(datasets)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "feature_config.json", feature_config.to_dict())
    write_json(
        output_dir / "evaluation_config.json",
        {
            "mode": "evaluate_existing_mutagenicity_rf_teacher",
            "data_dir": str(args.data_dir),
            "model_path": str(args.model_path),
            "output_dir": str(output_dir),
            "smiles_col": args.smiles_col,
            "label_col": args.label_col,
            "source_label": 1,
            "target_label": 0,
        },
    )
    write_evaluation_artifacts(
        output_dir,
        metrics=metrics,
        predictions=predictions,
        manifest=manifest,
        model_path=args.model_path,
        selection=bundle.get("selection"),
    )
    print("[MUTAGENICITY_RF_TEACHER_EVAL_OK]", flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_teacher(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

