#!/usr/bin/env python3
"""Adopt one shared AIDS/Mut WNode threshold contract before held-out access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


DISTANCE_LINE = "MolCLR-Node-Wasserstein"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text(path: Path, value: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def verify(*, dataset: str, source: Path, output: Path) -> dict[str, Any]:
    normalized = dataset.strip().lower()
    if normalized not in {"aids", "mutagenicity"}:
        raise ValueError("Threshold adoption supports only AIDS/Mutagenicity")
    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Fresh output already exists: {output}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Threshold contract must be one JSON object")
    expected_dataset = "AIDS" if normalized == "aids" else "Mutagenicity"
    failures: list[str] = []
    if payload.get("status") != "PASS":
        failures.append("status")
    if payload.get("dataset") != expected_dataset:
        failures.append("dataset")
    if payload.get("cf_mode") != "strict_flip":
        failures.append("cf_mode")
    if payload.get("distance_line") != DISTANCE_LINE:
        failures.append("distance_line")
    if payload.get("threshold_source_split") not in {
        "existing_frozen_protocol",
        "frozen_protocol",
        "legacy_frozen_protocol",
    }:
        failures.append("threshold_source_split")
    if payload.get("test_used_for_selection") is not False:
        failures.append("test_used_for_selection")
    values = payload.get("thresholds")
    try:
        thresholds = [float(value) for value in values]
    except (TypeError, ValueError):
        thresholds = []
    if (
        len(thresholds) != 601
        or any(not math.isfinite(value) or value < 0 for value in thresholds)
        or thresholds != sorted(set(thresholds))
        or not math.isclose(thresholds[0], 0.0, abs_tol=1e-15)
        or not math.isclose(thresholds[-1], 0.0535, abs_tol=1e-15)
    ):
        failures.append("threshold_grid")
    try:
        theta_star = float(payload.get("theta_star"))
        cost_cap = float(payload.get("cost_cap"))
    except (TypeError, ValueError):
        theta_star = cost_cap = math.nan
    if not math.isclose(theta_star, 0.05, abs_tol=1e-15):
        failures.append("theta_star")
    if not math.isclose(cost_cap, 0.0535, abs_tol=1e-15):
        failures.append("cost_cap")
    config_hash = str(payload.get("threshold_config_hash") or "")
    if len(config_hash) != 64 or any(ch not in "0123456789abcdef" for ch in config_hash):
        failures.append("threshold_config_hash")
    if failures:
        raise ValueError("Frozen threshold adoption failed: " + ",".join(failures))
    output.mkdir(parents=True)
    frozen = dict(payload)
    frozen.update(
        {
            "threshold_fitted_on_test": False,
            "selection_used_test": False,
            "shared_across_methods": True,
            "source_contract": str(source),
            "source_contract_sha256": _sha(source),
        }
    )
    audit = {
        "schema_version": "frozen_threshold_adoption_audit_v1",
        "status": "PASS",
        "dataset": expected_dataset,
        "source_contract": str(source),
        "source_contract_sha256": _sha(source),
        "threshold_count": len(thresholds),
        "theta_star": theta_star,
        "cost_cap": cost_cap,
        "test_used_for_selection": False,
        "shared_across_methods": True,
        "failures": [],
    }
    _write(output / "frozen_threshold_contract.json", frozen)
    _write(output / "threshold_adoption_audit.json", audit)
    _write_text(output / "PASS", "PASS\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify(
        dataset=args.dataset,
        source=Path(args.source),
        output=Path(args.output),
    )
    print(json.dumps(result, sort_keys=True))
    print(f"[FROZEN_THRESHOLD_CONTRACT_PASS] dataset={args.dataset.lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
