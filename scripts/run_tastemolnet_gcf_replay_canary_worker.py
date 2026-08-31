#!/usr/bin/env python3
"""Run one real-A800 process role of the bounded Taste T12 GCF replay canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_gcf_full_resume import (  # noqa: E402
    TasteGCFFullResumeError,
)
from src.baselines.tastemolnet_gcf_replay_canary import (  # noqa: E402
    run_replay_canary_phase,
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
    parser.add_argument(
        "--mode", choices=("uninterrupted", "checkpoint", "resume"), required=True
    )
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--observation", type=_absolute)
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
    return parser


def _validate_phase_arguments(args: argparse.Namespace) -> None:
    terminal = args.mode in {"uninterrupted", "resume"}
    if terminal is not (args.observation is not None):
        raise TasteGCFFullResumeError(
            "T12 uninterrupted/resume requires one observation; checkpoint forbids it"
        )
    if (args.mode == "resume") is not (args.checkpoint_manifest is not None):
        raise TasteGCFFullResumeError(
            "T12 resume requires one manifest; other process roles forbid it"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config.resolve(strict=True) != (REPO_ROOT / "configs/hpc.yaml").resolve(
        strict=True
    ):
        raise TasteGCFFullResumeError("T12 canary requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGCFFullResumeError(
            "T12 canary requires fail-closed inference override"
        )
    _validate_phase_arguments(args)
    result = run_replay_canary_phase(
        mode=args.mode,
        output_root=args.output_root,
        observation_path=args.observation,
        checkpoint_manifest=args.checkpoint_manifest,
        attempt_id=args.attempt_id,
        generation_token=args.generation_token,
        gpu_uuid=args.gpu_uuid,
        managed_neurosed_root=args.managed_neurosed_root,
        t3_root=args.t3_root,
        official_root=args.official_root,
        threshold_authority_path=args.neurosed_threshold_authority,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
