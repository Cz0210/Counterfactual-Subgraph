#!/usr/bin/env python3
"""Run or strictly validate the managed TasteMolNet T9 COMRECGC smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_comrecgc_smoke import (  # noqa: E402
    PASS_MARKER,
    STAGE,
    validate_tastemolnet_comrecgc_output,
)
from src.utils.tastemolnet_t9_comrecgc_release import (  # noqa: E402
    TasteComRecGCReleaseDisabled,
    assert_t9_execution_released,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(STAGE,), default=STAGE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--t2-adoption-root", type=Path, required=True)
    parser.add_argument("--t2-adoption-gate-sha256", required=True)
    parser.add_argument("--t2-adoption-receipt-sha256", required=True)
    parser.add_argument("--t2-source-evidence-sha256", required=True)
    parser.add_argument("--t3-output-root", type=Path, required=True)
    parser.add_argument("--t4-output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        evidence = validate_tastemolnet_comrecgc_output(args.output_dir)
        print(json.dumps(evidence, sort_keys=True, ensure_ascii=True))
        print(PASS_MARKER)
        return 0
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError(
            "Taste T9 requires exactly --set "
            "inference.fallback_to_heuristic=false"
        )
    try:
        assert_t9_execution_released()
    except TasteComRecGCReleaseDisabled as exc:
        print(str(exc), file=sys.stderr)
        return 78

    # Imported only after the static release gate.  The implementation is
    # added together with the reviewed managed ACTIVE integration; keeping the
    # checked-in release false guarantees this candidate cannot reach it.
    from src.baselines.tastemolnet_comrecgc_smoke import (
        run_tastemolnet_comrecgc_smoke,
    )

    run_tastemolnet_comrecgc_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        t2_adoption_root=args.t2_adoption_root,
        t2_adoption_gate_sha256=args.t2_adoption_gate_sha256,
        t2_adoption_receipt_sha256=args.t2_adoption_receipt_sha256,
        t2_source_evidence_sha256=args.t2_source_evidence_sha256,
        t3_output_root=args.t3_output_root,
        t4_output_root=args.t4_output_root,
        checkpoint_dir=args.checkpoint_dir,
        train_csv=args.train_csv,
        official_root=args.official_root,
    )
    # Runtime success is deliberately stdout-quiet after PASS. The managed
    # parent performs the external strict reopen and publishes COMPLETION.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
