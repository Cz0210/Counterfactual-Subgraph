#!/usr/bin/env python3
"""Train and validate-select the Mutagenicity Morgan-RF teacher."""

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
    atomic_pickle,
    build_bundle,
    evaluate_model,
    load_all_splits,
    parse_int_grid,
    parse_optional_int_grid,
    select_random_forest,
    split_manifest,
    write_evaluation_artifacts,
    write_json,
)


DEFAULT_DATA_DIR = Path("outputs/hpc/datasets/final/mutagenicity_v1_processed")
DEFAULT_OUTPUT_DIR = Path("outputs/hpc/oracle/mutagenicity_rf_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--n-estimators-grid", default="300,600")
    parser.add_argument("--max-depth-grid", default="none,20,40")
    parser.add_argument("--min-samples-leaf-grid", default="1,2")
    parser.add_argument(
        "--selection-metric",
        choices=("balanced_accuracy", "auroc", "average_precision"),
        default="balanced_accuracy",
    )
    parser.add_argument("--class-weight", default="balanced_subsample")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def train_teacher(args: argparse.Namespace) -> Path:
    feature_config = FeatureConfig(radius=int(args.radius), n_bits=int(args.n_bits))
    datasets = load_all_splits(
        args.data_dir,
        feature_config=feature_config,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
    )
    n_estimators_grid = parse_int_grid(args.n_estimators_grid)
    max_depth_grid = parse_optional_int_grid(args.max_depth_grid)
    min_samples_leaf_grid = parse_int_grid(args.min_samples_leaf_grid)
    class_weight = None if str(args.class_weight).lower() in {"none", "null"} else args.class_weight
    model, selection, search_results = select_random_forest(
        datasets["train"],
        datasets["val"],
        n_estimators_grid=n_estimators_grid,
        max_depth_grid=max_depth_grid,
        min_samples_leaf_grid=min_samples_leaf_grid,
        random_seed=int(args.random_seed),
        n_jobs=int(args.n_jobs),
        class_weight=class_weight,
        selection_metric=args.selection_metric,
    )
    metrics, predictions = evaluate_model(model, datasets)
    manifest = split_manifest(datasets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mutagenicity_rf_model.pkl"
    bundle = build_bundle(
        model,
        feature_config=feature_config,
        selection=selection,
        metrics=metrics,
        manifest=manifest,
    )
    atomic_pickle(model_path, bundle)
    config = {
        "dataset": "Mutagenicity",
        "dataset_version": "v1",
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "smiles_col": args.smiles_col,
        "label_col": args.label_col,
        "label_semantics": {"0": "non_mutagenic", "1": "mutagenic"},
        "source_label": 1,
        "target_label": 0,
        "fit_split": "train",
        "selection_split": "val",
        "calibration_role": "reserved_for_future_probability_or_threshold_calibration",
        "test_role": "final_teacher_quality_evaluation",
        "random_seed": int(args.random_seed),
        "n_jobs": int(args.n_jobs),
        "class_weight": class_weight,
        "selection_metric": args.selection_metric,
        "hyperparameter_grid": {
            "n_estimators": n_estimators_grid,
            "max_depth": max_depth_grid,
            "min_samples_leaf": min_samples_leaf_grid,
        },
        "selection": selection,
        "search_results": search_results,
        "model_path": str(model_path),
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "feature_config.json", feature_config.to_dict())
    write_evaluation_artifacts(
        output_dir,
        metrics=metrics,
        predictions=predictions,
        manifest=manifest,
        model_path=model_path,
        selection=selection,
    )
    print("[MUTAGENICITY_RF_TEACHER_TRAIN_OK]", flush=True)
    print(f"model_path={model_path}", flush=True)
    print(f"selected_parameters={json.dumps(selection['selected_parameters'], sort_keys=True)}")
    for split in ("train", "val", "calibration", "test"):
        values = metrics[split]
        print(
            f"{split}: accuracy={values['accuracy']:.6f} "
            f"balanced_accuracy={values['balanced_accuracy']:.6f} "
            f"auroc={values['auroc']:.6f} average_precision={values['average_precision']:.6f} "
            f"brier_score={values['brier_score']:.6f}",
            flush=True,
        )
    return model_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_teacher(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

