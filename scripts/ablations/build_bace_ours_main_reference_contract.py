#!/usr/bin/env python3
"""Freeze the BACE/Ours main-table provenance for later ablations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablations.main_reference import (  # noqa: E402
    BaceOursReferenceInputs,
    build_bace_ours_main_reference,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    for name in (
        "matrix_authority_state",
        "final_root",
        "oracle_root",
        "ppo_root",
        "train_parent_prep_manifest",
        "base_pool_manifest",
        "high_temperature_pool_manifest",
        "merged_pool_manifest",
        "verification_manifest",
        "selector_manifest",
        "molclr_checkpoint",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def main() -> int:
    args = _arguments()
    inputs = BaceOursReferenceInputs(
        **{
            field: getattr(args, field)
            for field in BaceOursReferenceInputs.__dataclass_fields__
        }
    )
    payload = build_bace_ours_main_reference(inputs)
    _atomic_write(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output), "sha256": payload["reference_contract_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
