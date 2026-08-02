#!/usr/bin/env python3
"""Run official VRRW on the strict Mutagenicity train-source cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.gcfexplainer_mutagenicity_adapter import (  # noqa: E402
    EXPECTED_GENERATION_SOURCE_ROWS,
    GCFExplainerVRRWConfigError,
    get_vrrw_profile_contract,
    validate_vrrw_profile,
    write_failure_artifacts,
    write_json,
)
from src.baselines.gcfexplainer_mutagenicity_runtime import run_official_vrrw  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--gnn-checkpoint", required=True)
    parser.add_argument("--neurosed-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--parent-limit", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--teleport", type=float, default=0.1)
    parser.add_argument("--candidate-capacity", type=int, default=100000)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device1", default="cuda:0")
    parser.add_argument("--device2", default="cuda:0")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _resolve_profile_defaults(args: argparse.Namespace) -> None:
    contract = get_vrrw_profile_contract(args.profile)
    if args.parent_limit is None:
        args.parent_limit = contract.parent_limit
    if args.m is None:
        args.m = contract.default_m
    if args.alpha is None:
        args.alpha = contract.alpha
    if args.theta is None:
        args.theta = contract.theta
    if args.seed is None:
        args.seed = contract.seed


def _print_config(args: argparse.Namespace) -> None:
    print("[MUTAGENICITY_GCFEXPLAINER_VRRW_CONFIG]", flush=True)
    print(f"profile={args.profile}", flush=True)
    print(f"parent_limit={args.parent_limit}", flush=True)
    print(f"M={args.m}", flush=True)
    print(f"alpha={args.alpha}", flush=True)
    print(f"theta={args.theta}", flush=True)
    print(f"seed={args.seed}", flush=True)
    print(f"dataset_dir={Path(args.dataset_dir).expanduser().resolve()}", flush=True)
    print(
        f"gnn_checkpoint={Path(args.gnn_checkpoint).expanduser().resolve()}",
        flush=True,
    )
    print(
        f"generation_source_rows={EXPECTED_GENERATION_SOURCE_ROWS}",
        flush=True,
    )
    print("calibration_loaded=false", flush=True)
    print("test_loaded=false", flush=True)


def _write_config_failure(
    args: argparse.Namespace,
    error: GCFExplainerVRRWConfigError,
) -> None:
    destination = Path(args.output_dir).expanduser().resolve()
    if (destination / "_FINALIZED.json").exists() or (
        destination / "_RUN_COMPLETE.json"
    ).exists():
        return
    if destination.exists() and any(destination.iterdir()):
        marker = destination / "_RUN_FAILED.json"
        previous = (
            json.loads(marker.read_text(encoding="utf-8"))
            if marker.is_file()
            else {}
        )
        if previous.get("stage") != "vrrw_config":
            raise FileExistsError(
                "Refusing to replace a non-config VRRW failure with a config "
                f"failure: {destination}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        **error.details,
        "run_complete": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "gnn_checkpoint": str(
            Path(args.gnn_checkpoint).expanduser().resolve()
        ),
        "model_training_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
    }
    write_json(destination / "failure_summary.json", payload)
    write_json(destination / "_RUN_FAILED.json", payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_profile_defaults(args)
    _print_config(args)
    if not args.forbid_calibration_test:
        raise ValueError("VRRW requires --forbid-calibration-test.")
    try:
        validate_vrrw_profile(
            args.profile,
            parent_limit=args.parent_limit,
            m=args.m,
            alpha=args.alpha,
            theta=args.theta,
            seed=args.seed,
        )
    except GCFExplainerVRRWConfigError as exc:
        _write_config_failure(args, exc)
        print("[MUTAGENICITY_GCFEXPLAINER_CONFIG_ERROR]", file=sys.stderr)
        for key in (
            "profile",
            "parent_limit",
            "M",
            "expected_parent_limit",
            "expected_M",
            "alpha",
            "theta",
            "seed",
        ):
            value = exc.details[key]
            if key == "expected_M":
                value = "_or_".join(str(item) for item in value)
            print(f"{key}={value}", file=sys.stderr)
        return 2
    try:
        summary = run_official_vrrw(
            dataset_dir=args.dataset_dir,
            official_root=args.official_root,
            gnn_checkpoint=args.gnn_checkpoint,
            neurosed_checkpoint=args.neurosed_checkpoint,
            output_dir=args.output_dir,
            profile=args.profile,
            parent_limit=args.parent_limit,
            m=args.m,
            alpha=args.alpha,
            theta=args.theta,
            teleport=args.teleport,
            candidate_capacity=args.candidate_capacity,
            sample=args.sample,
            sample_size=args.sample_size,
            seed=args.seed,
            device1=args.device1,
            device2=args.device2,
            resume=args.resume,
        )
    except Exception as exc:
        write_failure_artifacts(
            args.output_dir,
            error=exc,
            resolved_config=vars(args),
            extra={
                "stage": "vrrw_runtime",
                "profile": args.profile,
                "parent_limit": args.parent_limit,
                "M": args.m,
                "model_training_performed": False,
            },
        )
        raise
    marker = (
        "[MUTAGENICITY_GCFEXPLAINER_VRRW_SMOKE_OK]"
        if args.profile == "smoke"
        else "[MUTAGENICITY_GCFEXPLAINER_VRRW_FULL_OK]"
    )
    print(marker, flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
