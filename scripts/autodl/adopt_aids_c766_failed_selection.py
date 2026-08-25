#!/usr/bin/env python3
"""Publish or reopen the frozen c766 failed-selection recovery receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.baselines.comrecgc.failed_selection_adoption import (  # noqa: E402
    PRODUCTION_CONTROL_ROOT,
    PRODUCTION_OUTPUT_PARENT,
    PRODUCTION_PROC_ROOT,
    create_or_validate_aids_c766_failed_selection_adoption,
    verify_aids_c766_failed_selection_recovery_evidence,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only adoption of the failed c766 adaptive selection for a "
            "separate recovery route. This never produces a scientific PASS."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Repository runtime config (validated for workflow parity; not mutated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Fresh direct child of "
            f"{PRODUCTION_OUTPUT_PARENT}; ordinary task-attempt roots are rejected."
        ),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=PRODUCTION_CONTROL_ROOT,
        help="Frozen AutoDL control root; alternate values are rejected.",
    )
    parser.add_argument(
        "--proc-root",
        type=Path,
        default=PRODUCTION_PROC_ROOT,
        help="Frozen procfs root; alternate values are rejected.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Require and fully reopen an already-terminal receipt.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = args.config.expanduser().resolve(strict=True)
    if not config.is_file():
        raise ValueError(f"config is not a regular file: {config}")
    operation = (
        verify_aids_c766_failed_selection_recovery_evidence
        if args.validate_only
        else create_or_validate_aids_c766_failed_selection_adoption
    )
    receipt = operation(
        output_dir=args.output_dir,
        control_root=args.control_root,
        proc_root=args.proc_root,
    )
    failed = receipt["failed_selection"]
    print(
        json.dumps(
            {
                "receipt": str(args.output_dir / "failed_selection_adoption_receipt.json"),
                "status": receipt["status"],
                "artifact_kind": receipt["artifact_kind"],
                "source_final_status": receipt["source_final_status"],
                "recovery_only": receipt[
                    "failed_evidence_adopted_for_recovery_only"
                ],
                "selection_manifest_sha256": failed[
                    "selection_manifest_sha256"
                ],
                "anchor_count": failed["anchor_count"],
                "dbscan_partition_proven": failed["dbscan_partition_proven"],
            },
            sort_keys=True,
        )
    )
    print("[AIDS_C766_FAILED_SELECTION_RECOVERY_EVIDENCE_READY]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
