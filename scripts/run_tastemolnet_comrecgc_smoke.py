#!/usr/bin/env python3
"""Run the trusted-root TasteMolNet T9 worker or validate its managed final."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_comrecgc_smoke import (  # noqa: E402
    STAGE,
    run_tastemolnet_comrecgc_smoke,
)
from src.utils.tastemolnet_t9_managed_v2 import (  # noqa: E402
    TasteT9ManagedV2Error,
    load_t9_verified_gate,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--stage", choices=(STAGE,), default=STAGE)
    parser.add_argument("--stage-root", type=_absolute)
    parser.add_argument("--output-dir", type=_absolute, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--t2-adoption-root", type=_absolute)
    parser.add_argument("--t2-adoption-gate-sha256")
    parser.add_argument("--t2-adoption-receipt-sha256")
    parser.add_argument("--t2-source-evidence-sha256")
    parser.add_argument("--t3-output-root", type=_absolute)
    parser.add_argument("--t4-output-root", type=_absolute)
    parser.add_argument("--checkpoint-dir", type=_absolute)
    parser.add_argument("--train-csv", type=_absolute)
    parser.add_argument("--official-root", type=_absolute)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser


def _required_worker_args(args: argparse.Namespace) -> None:
    names = (
        "stage_root",
        "run_id",
        "gpu_uuid",
        "t2_adoption_root",
        "t2_adoption_gate_sha256",
        "t2_adoption_receipt_sha256",
        "t2_source_evidence_sha256",
        "t3_output_root",
        "t4_output_root",
        "checkpoint_dir",
        "train_csv",
        "official_root",
    )
    missing = [name for name in names if getattr(args, name) in (None, "")]
    if missing:
        raise TasteT9ManagedV2Error(
            "T9 worker arguments are absent: " + ",".join(missing)
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        gate = load_t9_verified_gate(args.output_dir)
        print(json.dumps(gate, sort_keys=True, ensure_ascii=True))
        return 0
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteT9ManagedV2Error(
            "Taste T9 requires exactly --set "
            "inference.fallback_to_heuristic=false"
        )
    _required_worker_args(args)
    result = run_tastemolnet_comrecgc_smoke(
        config_path=args.config,
        stage_root=args.stage_root,
        output_dir=args.output_dir,
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
    )
    # This is a worker SEALED receipt. It is never a terminal PASS.
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
