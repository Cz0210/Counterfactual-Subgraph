#!/usr/bin/env python3
"""Audit and deterministically repair the frozen COMRECGC Mutagenicity smoke."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import write_json  # noqa: E402
from src.baselines.comrecgc.mutagenicity_chemistry_audit import (  # noqa: E402
    run_mutagenicity_chemistry_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--dataset", choices=("aids", "mutagenicity", "bace"), default="mutagenicity"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-csv")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--trace-lineage-path", required=True)
    parser.add_argument("--trace-parity-path")
    parser.add_argument("--trace-evidence-path")
    parser.add_argument("--common-recourse-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preregistration-path", required=True)
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--expected-candidate-count", type=int)
    parser.add_argument("--expected-medoid-count", type=int)
    parser.add_argument("--expected-counterfactuals-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    try:
        trace_evidence_path = args.trace_evidence_path or args.trace_parity_path
        if not trace_evidence_path:
            raise ValueError("A trace parity or trace integrity evidence path is required.")
        audit = run_mutagenicity_chemistry_audit(
            project_root=args.project_root,
            dataset_dir=args.dataset_dir,
            generation_dir=args.generation_dir,
            trace_lineage_path=args.trace_lineage_path,
            trace_parity_path=trace_evidence_path,
            common_recourse_dir=args.common_recourse_dir,
            output_dir=output,
            preregistration_path=args.preregistration_path,
            parent_limit=args.parent_limit,
            expected_candidate_count=args.expected_candidate_count,
            expected_medoid_count=args.expected_medoid_count,
            expected_counterfactuals_sha256=args.expected_counterfactuals_sha256,
            dataset=args.dataset,
            source_csv=args.source_csv,
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": f"{args.dataset}_project_chemistry_audit",
            "error_class": type(exc).__name__,
            "message": str(exc),
            "calibration_loaded": False,
            "test_loaded": False,
            "run_complete": False,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "failure_summary.json", failure)
        write_json(output / "_RUN_FAILED.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 3
    print(json.dumps(audit, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
