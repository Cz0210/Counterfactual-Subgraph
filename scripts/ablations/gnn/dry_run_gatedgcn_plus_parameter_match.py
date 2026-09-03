#!/usr/bin/env python3
"""Verify the GatedGCN+ width using CPU-only actual parameter counts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.molecular_graph_featurizer import (  # noqa: E402
    default_molecular_feature_schema,
)
from src.models.gatedgcn_plus_backbone import (  # noqa: E402
    GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS,
    GATEDGCN_PLUS_OFFICIAL_COMMIT,
    build_gatedgcn_plus_molecular_gnn,
    estimate_gatedgcn_plus_parameter_count,
    match_gatedgcn_plus_hidden_dim,
)


DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "configs/ablations/gnn/bace_gine_reference_parameter_receipt_v1.json"
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(reference_path: Path) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference.get("status") != "PASS"
        or reference.get("source") != "ACTUAL_LOADED_WEIGHTS"
        or reference.get("validation_metrics_loaded_for_parameter_count") is not False
        or reference.get("test_metrics_loaded_for_parameter_count") is not False
    ):
        raise ValueError("GINE reference parameter authority is not closed")
    reference_count = int(reference["total_parameters"])
    match = match_gatedgcn_plus_hidden_dim(reference_count)
    schema = default_molecular_feature_schema()
    candidates: list[dict[str, object]] = []
    for candidate in match.candidates:
        model = build_gatedgcn_plus_molecular_gnn(
            num_classes=2,
            node_feature_schema=schema,
            edge_feature_schema=schema,
            hidden_dim=candidate.hidden_dim,
        )
        actual = sum(parameter.numel() for parameter in model.parameters())
        estimated = estimate_gatedgcn_plus_parameter_count(candidate.hidden_dim)
        candidates.append(
            {
                **candidate.to_dict(),
                "actual_loaded_parameter_count": actual,
                "formula_parameter_count": estimated,
                "actual_matches_formula": actual == estimated,
            }
        )
    if not all(bool(row["actual_matches_formula"]) for row in candidates):
        raise RuntimeError("GatedGCN+ analytical and actual parameter counts differ")
    return {
        "schema_version": "gatedgcn_plus_parameter_dry_run_v1",
        "status": "PASS",
        "dataset": "bace",
        "method": "ours",
        "reference_parameter_count": reference_count,
        "allowed_hidden_dims": list(GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS),
        "selected_hidden_dim": match.selected_hidden_dim,
        "selected_parameter_count": match.selected_parameter_count,
        "selected_relative_difference": match.selected_relative_difference,
        "official_commit": GATEDGCN_PLUS_OFFICIAL_COMMIT,
        "candidates": candidates,
        "selection_uses_validation": False,
        "selection_uses_test": False,
        "science_started": False,
        "gpu_lock_acquired": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml", help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.reference.expanduser().resolve(strict=True))
    if arguments.output is not None:
        _atomic_json(arguments.output.expanduser().resolve(strict=False), result)
    print(json.dumps(result, sort_keys=True))
    print("[GATEDGCN_PLUS_PARAMETER_DRY_RUN_PASS]")
