#!/usr/bin/env python3
"""Audit BACE source codec and the completed VRRW candidate pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_bace_runtime import (  # noqa: E402
    audit_bace_source_roundtrip,
    audit_bace_vrrw_candidate_sufficiency,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--vrrw-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--parent-limit", type=int, required=True)
    parser.add_argument("--target-k", type=int, default=20)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--scan-all", action="store_true")
    parser.add_argument("--require-connected", action="store_true")
    parser.add_argument("--calibration-source-csv")
    parser.add_argument("--test-source-csv")
    parser.add_argument("--external-sample-limit", type=int, default=16)
    args = parser.parse_args(argv)

    teacher = TeacherSemanticScorer(args.teacher_path, device="cpu")
    if not teacher.available:
        raise RuntimeError(f"BACE RF teacher unavailable: {teacher.availability_reason}")
    candidate_audit = audit_bace_vrrw_candidate_sufficiency(
        dataset_dir=args.dataset_dir,
        vrrw_dir=args.vrrw_dir,
        teacher=teacher,
        teacher_path=args.teacher_path,
        output_dir=args.output_dir,
        profile=args.profile,
        parent_limit=args.parent_limit,
        target_k=args.target_k,
        scan_limit=args.scan_limit,
        scan_all=bool(args.scan_all),
        require_connected=bool(args.require_connected),
    )
    roundtrip = audit_bace_source_roundtrip(
        dataset_dir=args.dataset_dir,
        teacher=teacher,
        output_dir=args.output_dir,
        calibration_source_csv=args.calibration_source_csv,
        test_source_csv=args.test_source_csv,
        external_sample_limit=args.external_sample_limit,
    )
    result = {
        "candidate_sufficiency": candidate_audit,
        "source_roundtrip": roundtrip,
        "mapping_roundtrip_pass": roundtrip["round_trip_passed"],
        "num_retained": candidate_audit["candidate_attrition"]["num_retained"],
        "native_order_preserved": candidate_audit["candidate_attrition"][
            "native_order_preserved"
        ],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GCFEXPLAINER_NATIVE_CANDIDATE_AUDIT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
