#!/usr/bin/env python3
"""Fit scalar temperature on validation logits in a frozen GNN bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.oracles.gnn_oracle import (  # noqa: E402
    fit_temperature_scaling,
    sha256_file,
    update_checkpoint_sha256sums,
    verify_checkpoint_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--split", choices=("validation", "val"), default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iter", type=int, default=100)
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"Validation CSV is empty: {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    del args.device  # Temperature is a single CPU scalar optimization.
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    validation_csv = Path(args.validation_csv).expanduser().resolve()
    if not validation_csv.is_file():
        raise FileNotFoundError(validation_csv)
    verify_checkpoint_bundle(checkpoint)
    prediction_path = checkpoint / "validation_predictions.csv"
    predictions = _read_rows(prediction_path)
    validation_rows = _read_rows(validation_csv)
    id_column = "molecule_id" if "molecule_id" in validation_rows[0] else "id"
    expected = {
        str(row[id_column]): int(str(row["label"]).strip()) for row in validation_rows
    }
    observed = {str(row["molecule_id"]): int(row["label"]) for row in predictions}
    if observed != expected:
        raise ValueError(
            "Frozen validation predictions do not match the supplied validation split."
        )
    bad_splits = sorted(
        {
            str(row.get("split") or "").strip().lower()
            for row in predictions
            if str(row.get("split") or "").strip().lower()
            not in {"val", "validation", "valid"}
        }
    )
    if bad_splits:
        raise ValueError(f"Temperature scaling received non-validation rows: {bad_splits}")
    logits = np.asarray([json.loads(row["logits"]) for row in predictions], dtype=np.float64)
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.int64)
    result = fit_temperature_scaling(logits, labels, max_iter=args.max_iter)
    result.update(
        {
            "validation_csv": str(validation_csv),
            "validation_csv_sha256": sha256_file(validation_csv),
            "validation_predictions_sha256": sha256_file(prediction_path),
            "split_argument": args.split,
        }
    )
    _atomic_json(checkpoint / "temperature_scaling.json", result)
    model_card_path = checkpoint / "model_card.json"
    model_card = json.loads(model_card_path.read_text(encoding="utf-8"))
    model_card["temperature_calibration"] = {
        "status": "fit",
        "split": "validation",
        "test_used_for_fit": False,
        "temperature": result["temperature"],
    }
    _atomic_json(model_card_path, model_card)
    update_checkpoint_sha256sums(checkpoint)
    verify_checkpoint_bundle(checkpoint)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[GNN_TEMPERATURE_CALIBRATION_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
