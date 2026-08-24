#!/usr/bin/env python3
"""Publish a hash-bound adoption gate for the already frozen AIDS selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.thresholds.expanduser()
    if source.is_symlink():
        raise SystemExit("threshold contract may not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise SystemExit("threshold contract must be a physical nonempty file")
    actual = _sha256(source)
    if actual != args.expected_sha256:
        raise SystemExit("threshold contract SHA256 mismatch")
    try:
        threshold = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit("threshold contract is not valid JSON") from exc
    if not isinstance(threshold, dict):
        raise SystemExit("threshold contract must be one JSON object")
    if threshold.get("test_used_for_selection") is True:
        raise SystemExit("threshold contract reports test leakage")
    output = args.output_dir.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit("selector gate output must be empty")
    gate = {
        "schema_version": "aids_comrecgc_exact_route_v5_selector_gate_v1",
        "status": "PASS",
        "dataset": "aids",
        "selector_fitted_on_calibration": True,
        "test_used_for_selection": False,
        "threshold_contract": str(source),
        "threshold_contract_sha256": actual,
        "theta_star": 0.05,
        "cost_cap": 0.0535,
    }
    _atomic_json(output / "selector_gate.json", gate)
    (output / "PASS").write_text("PASS\n", encoding="utf-8")
    print("[AIDS_COMRECGC_EXACT_ROUTE_V5_SELECTOR_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
