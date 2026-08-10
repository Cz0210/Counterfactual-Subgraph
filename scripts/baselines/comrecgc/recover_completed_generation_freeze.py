#!/usr/bin/env python3
"""Audit or recover a completed COMRECGC walk without rerunning it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import write_json  # noqa: E402
from src.baselines.comrecgc.freeze_recovery import (  # noqa: E402
    recover_completed_generation_freeze,
    validate_completed_generation_freeze,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-generation-dir", required=True)
    value.add_argument("--output-dir")
    value.add_argument("--audit-output", required=True)
    value.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    value.add_argument("--dataset-dir", required=True)
    value.add_argument("--source-csv")
    value.add_argument("--expected-steps", type=int, default=50_000)
    value.add_argument("--expected-project-commit")
    value.add_argument("--validate-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.validate_only:
        audit, _payload = validate_completed_generation_freeze(
            source_generation_dir=args.source_generation_dir,
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
            source_csv=args.source_csv,
            expected_steps=args.expected_steps,
            expected_project_commit=args.expected_project_commit,
        )
        write_json(args.audit_output, audit)
        print(json.dumps(audit, sort_keys=True, default=str))
        return 0 if audit["FREEZE_ONLY_RECOVERY_SAFE"] is True else 3
    if not args.output_dir:
        raise ValueError("--output-dir is required unless --validate-only is used.")
    result = recover_completed_generation_freeze(
        source_generation_dir=args.source_generation_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        source_csv=args.source_csv,
        expected_steps=args.expected_steps,
        expected_project_commit=args.expected_project_commit,
    )
    write_json(args.audit_output, result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
