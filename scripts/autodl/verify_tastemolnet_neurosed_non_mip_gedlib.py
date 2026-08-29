#!/usr/bin/env python3
"""Independently reopen and verify one non-MIP GEDLIB selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.tastemolnet_neurosed_non_mip import (  # noqa: E402
    validate_non_mip_selection_manifest,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.receipt.exists():
        raise RuntimeError("verifier receipt already exists")
    raw = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if type(raw) is not dict:
        raise RuntimeError("selection manifest is not one JSON object")
    selection = validate_non_mip_selection_manifest(raw, reopen_artifacts=True)
    receipt = {
        "schema_version": "tastemolnet_neurosed_non_mip_gedlib_verifier_v1",
        "status": "PASS",
        "marker": "[TASTE_NON_MIP_GEDLIB_BACKEND_VERIFIED]",
        "independent_process_reopened_all_candidate_artifacts": True,
        "selection_manifest_path": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": hashlib.sha256(
            args.selection_manifest.read_bytes()
        ).hexdigest(),
        "selection_sha256": selection["selection_sha256"],
        "selected_ged_backend": selection["selected_ged_backend"],
        "selected_ged_backend_config": selection["backend_config"],
        "GED_LABEL_BACKEND_VARIANT": selection["GED_LABEL_BACKEND_VARIANT"],
        "F2_BLP_USED": selection["F2_BLP_USED"],
        "GUROBI_USED": selection["GUROBI_USED"],
        "selected_neurosed_train_pair_budget": selection[
            "selected_neurosed_train_pair_budget"
        ],
        "selected_neurosed_validation_pair_budget": selection[
            "selected_neurosed_validation_pair_budget"
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _atomic_json(args.receipt, receipt)
    print(receipt["marker"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
