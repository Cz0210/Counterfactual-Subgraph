#!/usr/bin/env python3
"""Train NeuroSED from frozen fixed-budget Taste GED interval labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.tastemolnet_neurosed_fixed_budget import (  # noqa: E402
    NEUROSED_PASS_MARKER,
    TRAINER_READY_MARKER,
    train_fixed_budget_neurosed,
    verify_fixed_budget_neurosed,
)


def normalize_path(
    value: str | Path,
    *,
    relative_to: Path | None = None,
    must_exist: bool = True,
) -> Path:
    """Normalize one CLI path without ever dispatching ``resolve`` on a str.

    Relative configuration paths are anchored to the immutable repository
    checkout, not to the caller's working directory.  Scientific inputs fail
    closed when their target (including a symlink target) is absent; only the
    fresh output root is allowed not to exist yet.
    """

    if not isinstance(value, (str, Path)):
        raise TypeError("NeuroSED paths must be str or pathlib.Path values")
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    try:
        return path.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise ValueError(f"NeuroSED path does not exist: {path}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--ged-label-root", type=Path, required=True)
    parser.add_argument("--train-pair-root", type=Path, required=True)
    parser.add_argument("--validation-pair-root", type=Path, required=True)
    parser.add_argument("--feature-schema-json", type=Path, required=True)
    parser.add_argument("--non-mip-selection-manifest", type=Path, required=True)
    parser.add_argument("--non-mip-verifier-receipt", type=Path, required=True)
    parser.add_argument("--vendored-gcf-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--execution-git-tree", required=True)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-existing-root",
        action="store_true",
        help="independent second process reopens the worker root and writes PASS last",
    )
    mode.add_argument(
        "--train-and-verify",
        action="store_true",
        help=(
            "train, then launch this CLI in an independent verifier process; "
            "success requires the verifier's PASS-last file"
        ),
    )
    return parser.parse_args(argv)


def _independent_verifier_argv(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--config",
        str(args.config),
    ]
    for setting in args.set:
        command.extend(("--set", setting))
    for flag, value in (
        ("--ged-label-root", args.ged_label_root),
        ("--train-pair-root", args.train_pair_root),
        ("--validation-pair-root", args.validation_pair_root),
        ("--feature-schema-json", args.feature_schema_json),
        ("--non-mip-selection-manifest", args.non_mip_selection_manifest),
        ("--non-mip-verifier-receipt", args.non_mip_verifier_receipt),
        ("--vendored-gcf-root", args.vendored_gcf_root),
        ("--output-root", args.output_root),
        ("--execution-git-commit", args.execution_git_commit),
        ("--execution-git-tree", args.execution_git_tree),
        ("--device", args.device),
    ):
        command.extend((flag, str(value)))
    command.append("--verify-existing-root")
    return command


def _run_independent_verifier(args: argparse.Namespace) -> Path:
    subprocess.run(_independent_verifier_argv(args), check=True)
    pass_path = args.output_root.absolute() / "PASS"
    if not pass_path.is_file() or pass_path.read_text(encoding="utf-8") != (
        NEUROSED_PASS_MARKER + "\n"
    ):
        raise RuntimeError("independent NeuroSED verifier did not publish PASS last")
    return pass_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected_config = normalize_path(REPO_ROOT / "configs" / "hpc.yaml")
    config = normalize_path(args.config, relative_to=REPO_ROOT)
    if config != expected_config:
        raise ValueError("fixed-budget NeuroSED requires configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError(
            "fixed-budget NeuroSED requires exactly --set "
            "inference.fallback_to_heuristic=false"
        )
    common = {
        "ged_label_root": normalize_path(args.ged_label_root),
        "train_pair_root": normalize_path(args.train_pair_root),
        "validation_pair_root": normalize_path(args.validation_pair_root),
        "feature_schema_path": normalize_path(args.feature_schema_json),
        "non_mip_selection_manifest_path": normalize_path(
            args.non_mip_selection_manifest
        ),
        "non_mip_verifier_receipt_path": normalize_path(
            args.non_mip_verifier_receipt
        ),
        "vendored_gcf_root": normalize_path(args.vendored_gcf_root),
        "output_root": normalize_path(args.output_root, must_exist=False),
        "execution_git_commit": args.execution_git_commit,
        "execution_git_tree": args.execution_git_tree,
        "device": args.device,
    }
    if args.verify_existing_root:
        result = verify_fixed_budget_neurosed(**common)
        marker = NEUROSED_PASS_MARKER
    else:
        result = train_fixed_budget_neurosed(
            **common,
            source_execution_config_sha256=hashlib.sha256(
                config.read_bytes()
            ).hexdigest(),
        )
        if args.train_and_verify:
            pass_path = _run_independent_verifier(args)
            result = {
                **result,
                "state": "PASS",
                "marker": NEUROSED_PASS_MARKER,
                "success_marker_path": str(pass_path),
                "independent_verifier_process": True,
            }
            marker = NEUROSED_PASS_MARKER
        else:
            marker = TRAINER_READY_MARKER
    print(json.dumps(result, sort_keys=True), flush=True)
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
