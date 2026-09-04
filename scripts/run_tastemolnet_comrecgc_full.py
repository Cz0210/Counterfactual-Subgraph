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
    parser.add_argument("--resume-spec", type=_absolute)
    parser.add_argument("--route-c-spec", type=_absolute)
    parser.add_argument(
        "--route-c-storage", choices=("reference", "lowmemory")
    )
    parser.add_argument(
        "--checkpoint-only-step",
        type=int,
        choices=(250, 500, 510, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000),
    )
    parser.add_argument("--convergence-receipt", type=_absolute)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("Taste T14 requires the fail-closed inference override")
    if args.resume_spec and args.route_c_spec:
        raise ValueError("Taste T14 accepts only one resume authority")
    if bool(args.route_c_storage) != bool(args.route_c_spec):
        raise ValueError(
            "Taste T14 Route C spec and storage mode must be supplied together"
        )
    if args.resume and not (args.resume_spec or args.route_c_spec):
        raise ValueError("Taste T14 resume requires a bound resume authority")
    if args.resume_spec and not args.resume:
        raise ValueError("Taste T14 --resume-spec requires --resume")
    if args.checkpoint_only_step is not None and args.route_c_spec is None:
        raise ValueError(
            "Taste T14 checkpoint-only mode requires one fresh Route C spec"
        )
    if args.convergence_receipt and (
        not args.resume or args.route_c_spec is None or args.checkpoint_only_step is not None
    ):
        raise ValueError(
            "Taste T14 convergence receipt requires Route C finalization resume"
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
                resume_spec=args.resume_spec,
                route_c_spec=args.route_c_spec,
                route_c_storage=args.route_c_storage,
                checkpoint_only_step=args.checkpoint_only_step,
                convergence_receipt=args.convergence_receipt,
            )
            inputs.revalidate()
            if result.get("status") in {
                "CHECKPOINT_BOUNDARY_REACHED",
                "REPLAY_BOUNDARY_REACHED",
            }:
                result = {
                    "science": result,
                    "verification": {
                        "status": (
                            "REPLAY_BOUNDARY_COMPLETE"
                            if result.get("status") == "REPLAY_BOUNDARY_REACHED"
                            else "PENDING_INDEPENDENT_RELOAD"
                        ),
                        "completed_step": args.checkpoint_only_step,
                        "generation_pass": False,
                    },
                }
            else:
                result = {
                    "science": result,
                    "verification": validate_t14_full_output(args.output_dir),
                }
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
