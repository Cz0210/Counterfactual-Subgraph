#!/usr/bin/env python3
"""Run or independently verify the fixed-budget TasteMolNet T14 route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_comrecgc_full import (  # noqa: E402
    run_t14_full,
    validate_t14_full_output,
)
from src.utils.tastemolnet_t9_managed_v2 import (  # noqa: E402
    hold_t9_inputs,
    require_gpu_runtime,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument(
        "--physical-gpu-index", type=int, choices=range(4), required=True
    )
    parser.add_argument("--t2-adoption-root", type=_absolute, required=True)
    parser.add_argument("--t2-adoption-gate-sha256", required=True)
    parser.add_argument("--t2-adoption-receipt-sha256", required=True)
    parser.add_argument("--t2-source-evidence-sha256", required=True)
    parser.add_argument("--t3-output-root", type=_absolute, required=True)
    parser.add_argument("--t4-output-root", type=_absolute, required=True)
    parser.add_argument("--checkpoint-dir", type=_absolute, required=True)
    parser.add_argument("--train-csv", type=_absolute, required=True)
    parser.add_argument("--official-root", type=_absolute, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("Taste T14 requires the fail-closed inference override")
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
        physical_gpu_index=args.physical_gpu_index,
    ) as inputs:
        require_gpu_runtime(
            args.gpu_uuid,
            physical_gpu_index=args.physical_gpu_index,
        )
        if args.validate_only:
            result = validate_t14_full_output(args.output_dir)
        else:
            result = run_t14_full(
                inputs=inputs,
                output_root=args.output_dir,
                resume=args.resume,
            )
            inputs.revalidate()
            result = {
                "science": result,
                "verification": validate_t14_full_output(args.output_dir),
            }
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
