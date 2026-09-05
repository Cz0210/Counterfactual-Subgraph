#!/usr/bin/env python3
"""Prepare portable L0 inputs, or run the corrected-GNN CPU-only successor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.ablations.llm.bace_l0_hpc import build_inputs, run_l0
from src.eval.bace_frozen_gnn_contracts import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build-inputs")
    for name in ("task-spec", "task-spec-sha256", "gnn-input-bundle", "output-root"):
        build.add_argument("--" + name, required=True)
    run = commands.add_parser("run")
    for name in ("portable-input-bundle", "gnn-input-bundle", "corrected-gnn-archive", "registry-root", "output-root"):
        run.add_argument("--" + name, required=True)
    identity = run.add_mutually_exclusive_group(required=True)
    identity.add_argument("--corrected-gnn-sha256")
    identity.add_argument("--corrected-package-receipt", help="afterok predecessor's package receipt; path and SHA are rechecked")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cpu-threads", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=64)
    args = vars(parser.parse_args(argv))
    if args.pop("config") != "configs/hpc.yaml" or set(args.pop("set")) - {"inference.fallback_to_heuristic=false"}:
        parser.error("Use configs/hpc.yaml without scientific overrides")
    action = args.pop("action")
    if action == "run":
        receipt_path = args.pop("corrected_package_receipt")
        if receipt_path:
            receipt = json.loads(Path(receipt_path).read_text())
            archive = Path(args["corrected_gnn_archive"]).resolve(strict=True)
            if Path(receipt["path"]).resolve(strict=True) != archive or receipt["sha256"] != sha256_file(archive):
                raise ValueError("L0_CORRECTIVE_PACKAGE_RECEIPT_MISMATCH")
            args["corrected_gnn_sha256"] = receipt["sha256"]
        if not 1 <= args["cpu_threads"] <= 8 or args["batch_size"] < 1:
            parser.error("Use 1..8 CPU threads and positive batch size")
    result = build_inputs(**args) if action == "build-inputs" else run_l0(**args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if action == "build-inputs" or result["state"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
