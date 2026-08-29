#!/usr/bin/env python3
"""Independently verify and atomically publish a SEALED Taste T9 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_comrecgc_smoke import PASS_MARKER  # noqa: E402
from src.utils.tastemolnet_t9_managed_v2 import (  # noqa: E402
    hold_t9_inputs,
    open_t9_sealed,
    verify_and_publish_t9_sealed,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--sealed", type=_absolute, required=True)
    parser.add_argument("--final-path", type=_absolute, required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--expected-generation-token", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--t2-adoption-root", type=_absolute, required=True)
    parser.add_argument("--t2-adoption-gate-sha256", required=True)
    parser.add_argument("--t2-adoption-receipt-sha256", required=True)
    parser.add_argument("--t2-source-evidence-sha256", required=True)
    parser.add_argument("--t3-output-root", type=_absolute, required=True)
    parser.add_argument("--t4-output-root", type=_absolute, required=True)
    parser.add_argument("--checkpoint-dir", type=_absolute, required=True)
    parser.add_argument("--train-csv", type=_absolute, required=True)
    parser.add_argument("--official-root", type=_absolute, required=True)
    parser.add_argument("--force-cross-filesystem", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError(
            "Taste T9 requires exactly --set "
            "inference.fallback_to_heuristic=false"
        )
    with hold_t9_inputs(
        config_path=args.config,
        run_id=args.run_id,
        gpu_uuid=args.gpu_uuid,
        t2_adoption_root=args.t2_adoption_root,
        t2_adoption_gate_sha256=args.t2_adoption_gate_sha256,
        t2_adoption_receipt_sha256=args.t2_adoption_receipt_sha256,
        t2_source_evidence_sha256=args.t2_source_evidence_sha256,
        t3_output_root=args.t3_output_root,
        t4_output_root=args.t4_output_root,
        checkpoint_dir=args.checkpoint_dir,
        train_csv=args.train_csv,
        official_root=args.official_root,
    ) as inputs, open_t9_sealed(
        args.sealed,
        expected_attempt_id=args.expected_attempt_id,
        expected_generation_token=args.expected_generation_token,
    ) as sealed:
        authority = inputs.revalidate()
        publication, verification = verify_and_publish_t9_sealed(
            sealed,
            final_path=args.final_path,
            expected_authority=authority,
            revalidate_inputs=inputs.revalidate,
            force_cross_filesystem=args.force_cross_filesystem,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_path": str(publication.final_path),
                "attempt_id": publication.attempt_id,
                "generation_token": publication.generation_token,
                "verification_sha256": publication.verification_sha256,
                "gate_sha256": publication.gate_sha256,
                "publish_mode": publication.publish_mode,
                "verification": verification,
            },
            sort_keys=True,
            ensure_ascii=True,
        ),
        flush=True,
    )
    print(PASS_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
