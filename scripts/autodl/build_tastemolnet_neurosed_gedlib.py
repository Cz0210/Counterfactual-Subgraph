#!/usr/bin/env python3
"""Build and smoke-test pinned pyged/GEDLIB in a fresh offline directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.tastemolnet_neurosed_gedlib_build import (  # noqa: E402
    PINNED_GEDLIB_COMMIT,
    TasteGEDLIBBuildError,
    atomic_write_json,
    blocked_build_manifest,
    isolated_build_smoke,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hpc.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--greed-root", type=Path, required=True)
    parser.add_argument("--greed-expts-root", type=Path, required=True)
    parser.add_argument("--gedlib-root", type=Path, required=True)
    parser.add_argument(
        "--expected-gedlib-commit", default=PINNED_GEDLIB_COMMIT
    )
    parser.add_argument("--pybind11-cmake-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest or args.output_root.parent / (
        args.output_root.name + "-build-manifest.json"
    )
    if manifest_path.exists():
        print("BLOCKED_GEDLIB_BUILD", file=sys.stderr)
        print("build manifest already exists", file=sys.stderr)
        return 78
    try:
        result = isolated_build_smoke(
            greed_root=args.greed_root,
            greed_expts_root=args.greed_expts_root,
            gedlib_root=args.gedlib_root,
            expected_gedlib_commit=args.expected_gedlib_commit,
            pybind11_cmake_dir=args.pybind11_cmake_dir,
            output_root=args.output_root,
            cmake_executable=args.cmake,
            cxx_executable=args.cxx,
            python_executable=args.python,
        )
    except (TasteGEDLIBBuildError, OSError, subprocess.SubprocessError) as exc:
        result = blocked_build_manifest(
            error=exc,
            greed_root=args.greed_root,
            greed_expts_root=args.greed_expts_root,
            gedlib_root=args.gedlib_root,
            output_root=args.output_root,
        )
        atomic_write_json(manifest_path, result)
        print("BLOCKED_GEDLIB_BUILD", file=sys.stderr)
        print(result["error"], file=sys.stderr)
        return 78
    atomic_write_json(manifest_path, result)
    print(result["marker"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
