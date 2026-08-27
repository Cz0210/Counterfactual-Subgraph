#!/usr/bin/env python3
"""Inspect or build the policy-v2 TasteMolNet T5 clean initializer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.tastemolnet_clean_policy_init import (  # noqa: E402
    TasteCleanPolicyError,
    TasteCleanPolicyReleaseDisabled,
    build_clean_policy_initializer,
    inspect_generic_chemllm_base,
    validate_clean_policy_output,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/autodl/tastemolnet_clean_policy_initializer_v1.yaml"
DEFAULT_POLICY = REPO_ROOT / "configs/data_usage/tastemolnet_research_reporting_no_redistribution.yaml"


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    inspect = subparsers.add_parser("inspect-source")
    inspect.add_argument("--model-path", type=_absolute, required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--release-authority", type=_absolute, required=True)
    build.add_argument("--expected-release-authority-sha256", required=True)
    build.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    build.add_argument("--policy-receipt", type=_absolute, required=True)
    build.add_argument("--model-path", type=_absolute, required=True)
    build.add_argument("--output-root", type=_absolute, required=True)
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--lora-rank", type=int, default=8)
    build.add_argument("--lora-alpha", type=int, default=16)
    build.add_argument("--lora-dropout", type=float, default=0.05)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-root", type=_absolute, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "inspect-source":
            result = inspect_generic_chemllm_base(args.model_path)
        elif args.action == "validate":
            result = validate_clean_policy_output(args.output_root)
        else:
            result = build_clean_policy_initializer(
                config_path=args.config,
                release_authority_path=args.release_authority,
                expected_release_authority_sha256=args.expected_release_authority_sha256,
                policy_path=args.policy,
                policy_receipt_path=args.policy_receipt,
                source_model_path=args.model_path,
                project_root=REPO_ROOT,
                output_root=args.output_root,
                seed=args.seed,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
    except TasteCleanPolicyReleaseDisabled as exc:
        print(f"TASTE_CLEAN_POLICY_RELEASE_DISABLED: {exc}", file=sys.stderr, flush=True)
        return 78
    except (FileExistsError, OSError, TasteCleanPolicyError, ValueError) as exc:
        print(f"TASTE_CLEAN_POLICY_INITIALIZER_FAILED: {exc}", file=sys.stderr, flush=True)
        return 65
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.action == "build":
        print("[TASTE_CLEAN_POLICY_INITIALIZER_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
