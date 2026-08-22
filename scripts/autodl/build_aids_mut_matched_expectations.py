#!/usr/bin/env python3
"""Freeze the shared AIDS/Mutagenicity matched threshold expectations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile


def build(source: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("Source expectations must contain a datasets object")
    datasets = deepcopy(payload["datasets"])
    mut = datasets.get("Mutagenicity")
    if not isinstance(mut, dict):
        raise ValueError("Source expectations lack Mutagenicity")
    if len(mut.get("thresholds") or []) != 601:
        raise ValueError("Matched protocol must contain exactly 601 thresholds")
    if mut.get("threshold_source_split") != "existing_frozen_protocol":
        raise ValueError("Matched protocol is not frozen-protocol evidence")
    if mut.get("test_used_for_selection") is not False:
        raise ValueError("Matched protocol does not exclude test selection")
    aids = deepcopy(mut)
    aids["threshold_source"] = "matched AIDS/Mutagenicity existing frozen protocol"
    datasets["AIDS"] = aids
    return {
        "schema_version": "four_by_four_registry_expectations_v1",
        "datasets": datasets,
    }


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    destination = path.expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(Path(args.source).expanduser().resolve(strict=True))
    atomic_write(Path(args.output), result)
    print("[AIDS_MUT_MATCHED_EXPECTATIONS_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
