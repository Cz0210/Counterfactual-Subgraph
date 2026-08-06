#!/usr/bin/env python3
"""Validate and adopt an existing COMRECGC Mutagenicity action trace."""

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
from src.baselines.comrecgc.trace_recovery import (  # noqa: E402
    recover_mutagenicity_trace_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-failed-generation-dir", required=True)
    parser.add_argument("--reference-counterfactuals-path", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-candidate-count", type=int, default=164)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    try:
        result = recover_mutagenicity_trace_run(
            source_failed_generation_dir=args.source_failed_generation_dir,
            reference_counterfactuals_path=args.reference_counterfactuals_path,
            output_dir=output,
            expected_reference_sha256=args.expected_reference_sha256,
            expected_candidate_count=args.expected_candidate_count,
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": "mut_trace_adopt",
            "error_class": type(exc).__name__,
            "message": str(exc),
            "algorithm_rerun": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "failure_summary.json", failure)
        write_json(output / "_RUN_FAILED.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 3
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
