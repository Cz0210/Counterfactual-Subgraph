#!/usr/bin/env python3
"""Validate the BACE GINE-to-GraphGPS parameter match without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.molecular_graph_featurizer import default_molecular_feature_schema  # noqa: E402
from src.models.graphgps_backbone import (  # noqa: E402
    GRAPHGPS_ALLOWED_HIDDEN_DIMS,
    build_graphgps_molecular_gnn,
    graphgps_runtime_capabilities,
    match_graphgps_hidden_dim,
)


DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "configs/ablations/gnn/bace_gine_reference_parameter_receipt_v1.json"
)
DEFAULT_MATCH = (
    PROJECT_ROOT / "configs/ablations/gnn/bace_graphgps_parameter_match_v1.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON receipt must contain an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_counts() -> dict[str, int]:
    capabilities = graphgps_runtime_capabilities()
    if not capabilities["gpsconv_available"]:
        raise RuntimeError("--verify-runtime requires PyG GPSConv")
    schema = default_molecular_feature_schema()
    counts: dict[str, int] = {}
    for hidden_dim in GRAPHGPS_ALLOWED_HIDDEN_DIMS:
        model = build_graphgps_molecular_gnn(
            num_classes=2,
            node_feature_schema=schema,
            edge_feature_schema=schema,
            hidden_dim=hidden_dim,
            backend="pyg_gpsconv",
        )
        counts[str(hidden_dim)] = sum(
            int(parameter.numel()) for parameter in model.parameters()
        )
        del model
    return counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Required only to preserve the project's CLI/HPC contract.  This dry-run
    # neither loads dataset rows nor uses any value from the runtime config.
    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = (PROJECT_ROOT / config).resolve(strict=False)
    if not config.is_file():
        raise ValueError(f"runtime config does not exist: {config}")
    reference_path = Path(args.reference_receipt).expanduser().resolve(strict=True)
    expected_path = Path(args.expected_receipt).expanduser().resolve(strict=True)
    reference = _load_json(reference_path)
    expected = _load_json(expected_path)
    if (
        reference.get("status") != "PASS"
        or reference.get("source") != "ACTUAL_LOADED_WEIGHTS"
        or type(reference.get("total_parameters")) is not int
    ):
        raise ValueError("reference receipt is not an actual-loaded PASS")
    match = match_graphgps_hidden_dim(reference["total_parameters"])
    core = match.to_dict()
    expected_candidates = expected.get("candidates")
    if (
        expected.get("status") != "PASS"
        or expected.get("reference_parameter_count")
        != core["reference_parameter_count"]
        or expected.get("selected_hidden_dim") != core["selected_hidden_dim"]
        or expected.get("selected_parameter_count") != core["selected_parameter_count"]
        or expected.get("selected_relative_difference")
        != core["selected_relative_difference"]
        or expected_candidates != core["candidates"]
    ):
        raise ValueError("checked-in match receipt differs from recomputation")
    runtime_counts = _runtime_counts() if args.verify_runtime else None
    formula_counts = {
        str(item["hidden_dim"]): int(item["parameter_count"])
        for item in core["candidates"]
    }
    if runtime_counts is not None and runtime_counts != formula_counts:
        raise ValueError("PyG runtime parameter counts differ from the frozen formula")
    payload = {
        **core,
        "status": "PASS",
        "reference_receipt": str(reference_path),
        "reference_receipt_sha256": _sha256_file(reference_path),
        "expected_receipt": str(expected_path),
        "expected_receipt_sha256": _sha256_file(expected_path),
        "runtime_config": str(config),
        "runtime_verification_requested": bool(args.verify_runtime),
        "runtime_capabilities": graphgps_runtime_capabilities(),
        "runtime_parameter_counts": runtime_counts,
        "formula_parameter_counts": formula_counts,
        "science_started": False,
        "gpu_lock_acquired": False,
        "cuda_requested": False,
    }
    if args.output is not None:
        _atomic_json(Path(args.output).expanduser(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--reference-receipt", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--expected-receipt", type=Path, default=DEFAULT_MATCH)
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
