#!/usr/bin/env python3
"""Preflight, publish, or validate the frozen TasteMolNet T2 PASS adoption."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tastemolnet_gine_pass_adoption_v1 import (  # noqa: E402
    T2PassAdoptionError,
    T2PassAdoptionReleaseDisabled,
    T2PassAdoptionSources,
    adoption_output_root,
    preflight_t2_gine_pass_adoption,
    publish_t2_gine_pass_adoption,
    reviewed_release_candidate,
    validate_t2_gine_pass_adoption,
)


def _physical_config(path: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    normalized = Path(os.path.normpath(str(raw)))
    expected = PROJECT_ROOT / "configs/hpc.yaml"
    if normalized != expected:
        raise ValueError("--config must be the checkout's exact configs/hpc.yaml")
    descriptor = os.open(normalized, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        named = os.stat(normalized, follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError("--config must be one physical regular file")
    finally:
        os.close(descriptor)
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "action", choices=("preflight", "status", "publish"), help="No action launches science"
    )
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--scientific-output-root", type=Path, required=True)
    parser.add_argument("--training-state-root", type=Path, required=True)
    parser.add_argument("--execution-project-root", type=Path, required=True)
    parser.add_argument("--identity-fix-project-root", type=Path, required=True)
    parser.add_argument(
        "--assert-adoption-root",
        type=Path,
        help="Optional exact-path assertion; never selects an arbitrary destination",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if sys.flags.isolated != 1 or not sys.dont_write_bytecode:
            raise T2PassAdoptionError(
                "T2 adoption CLI requires Python -I -B"
            )
        _physical_config(args.config)
        sources = T2PassAdoptionSources.build(
            control_root=args.control_root,
            controller_root=args.controller_root,
            output_root=args.scientific_output_root,
            training_state_root=args.training_state_root,
            execution_project_root=args.execution_project_root,
            identity_fix_project_root=args.identity_fix_project_root,
        )
        expected = adoption_output_root(args.control_root)
        if args.assert_adoption_root is not None:
            asserted = Path(os.path.abspath(args.assert_adoption_root.expanduser()))
            if asserted != expected:
                raise T2PassAdoptionError(
                    "--assert-adoption-root differs from the exact derived formula"
                )
        if args.action == "preflight":
            evidence = preflight_t2_gine_pass_adoption(sources)
            result = {
                "status": "PREFLIGHT_PASS",
                "adoption_root": str(expected),
                "source_evidence_sha256": evidence["source_evidence_sha256"],
                "unreviewed_release_candidate": reviewed_release_candidate(evidence),
                "release_authorized": False,
                "writes_performed": False,
            }
            marker = "[T2_GINE_FULL_PASS_ADOPTION_PREFLIGHT_PASS]"
        elif args.action == "status":
            result = validate_t2_gine_pass_adoption(sources)
            marker = "[T2_GINE_FULL_PASS_ADOPTION_STATUS_PASS]"
        else:
            result = publish_t2_gine_pass_adoption(sources)
            marker = "[T2_GINE_FULL_PASS_ADOPTED]"
        print(json.dumps(result, indent=2, sort_keys=True))
        print(marker)
        return 0
    except T2PassAdoptionReleaseDisabled as exc:
        print(f"T2_GINE_PASS_ADOPTION_RELEASE_DISABLED: {exc}", file=sys.stderr)
        return 78
    except (T2PassAdoptionError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"T2_GINE_PASS_ADOPTION_REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
