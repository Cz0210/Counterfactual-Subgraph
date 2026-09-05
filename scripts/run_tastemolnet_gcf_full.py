#!/usr/bin/env python3
"""Run one exact TasteMolNet T12 GCF generation segment (fresh 10k/resume 20k)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_gcf_full import run_t12_generation_segment  # noqa: E402
from src.baselines.tastemolnet_gcf_full_resume import (  # noqa: E402
    TasteGCFFullResumeError,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or Path(path.absolute()) != path:
        raise argparse.ArgumentTypeError("path must be normalized and absolute")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--checkpoint-manifest", type=_absolute)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--generation-token", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--managed-neurosed-root", type=_absolute, required=True)
    parser.add_argument("--t3-root", type=_absolute, required=True)
    parser.add_argument("--official-root", type=_absolute, required=True)
    parser.add_argument(
        "--neurosed-threshold-authority", type=_absolute, required=True
    )
    parser.add_argument("--exact-replay-gate", type=_absolute, required=True)
    parser.add_argument(
        "--formal-checkpoint-cadence",
        action="store_true",
        help="use the final-16 100..20k recovery-only checkpoint schedule",
    )
    parser.add_argument(
        "--disposable-index-root",
        type=_absolute,
        help="node-local root for non-authoritative history indexes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml").resolve(
        strict=True
    ):
        raise TasteGCFFullResumeError("T12 production requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGCFFullResumeError(
            "T12 production requires fail-closed inference override"
        )
    if (args.mode == "resume") is not (args.checkpoint_manifest is not None):
        raise TasteGCFFullResumeError(
            "T12 resume requires one 10k manifest; fresh forbids it"
        )
    if args.formal_checkpoint_cadence:
        if args.disposable_index_root is None:
            raise TasteGCFFullResumeError(
                "formal T12 production requires a node-local disposable index root"
            )
        from src.utils.tastemolnet_t12_formal_profile_v1 import (
            configure_t12_formal_production_profile,
        )

        configure_t12_formal_production_profile()
    elif args.disposable_index_root is not None:
        raise TasteGCFFullResumeError(
            "a disposable T12 index root requires the explicit formal cadence"
        )
    result = run_t12_generation_segment(
        mode=args.mode,
        output_root=args.output_root,
        checkpoint_manifest=args.checkpoint_manifest,
        attempt_id=args.attempt_id,
        generation_token=args.generation_token,
        gpu_uuid=args.gpu_uuid,
        managed_neurosed_root=args.managed_neurosed_root,
        t3_root=args.t3_root,
        official_root=args.official_root,
        threshold_authority_path=args.neurosed_threshold_authority,
        replay_gate_path=args.exact_replay_gate,
        disposable_index_root=args.disposable_index_root,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
