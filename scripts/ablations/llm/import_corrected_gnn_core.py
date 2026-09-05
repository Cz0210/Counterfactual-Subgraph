#!/usr/bin/env python3
"""Fresh audit overlay for one corrected GNN package; never train or publish main."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.ablations.llm.bace_native_runtime import verified_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    for name in ("archive-path", "expected-sha256", "output-root", "expected-driver-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    if args.config != "configs/hpc.yaml": parser.error("Use configs/hpc.yaml")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != args.expected_driver_commit:
        raise ValueError("Corrective importer execution commit differs")
    archive = verified_file({"path": args.archive_path, "sha256": args.expected_sha256})
    output = Path(args.output_root).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Corrective import audit requires a fresh physical root")
    from src.ablations.gnn.temperature_repair import verify_corrective_package
    audit = verify_corrective_package(archive, output_root=output)
    if audit.get("state") != "GNN_CORE_SEED7_CORRECTED_PASS":
        raise ValueError("Independent corrective core did not pass")
    print(json.dumps({"state": audit["state"], "archive_sha256": args.expected_sha256,
        "audit_root": str(output), "repair_driver_commit": commit, "main_matrix_write": False,
        "training_performed": False, "raw_ot_recomputed_count": audit["raw_ot_recomputed_count"],
        "archive_preserved": True, "old_blocked_receipt_modified": False}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
