#!/usr/bin/env python3
"""Run the T7 worker and independent verifier under one held GPU0 lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_gcf_smoke import PASS_MARKER  # noqa: E402
from src.utils.tastemolnet_t7_typed_release_v1 import (  # noqa: E402
    hold_verified_t7_release,
)
from src.utils.tastemolnet_t7_typed_runtime_v1 import (  # noqa: E402
    load_t7_verified_gate,
    open_t7_sealed,
    run_t7_worker,
    verify_and_publish_t7_sealed,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "worker", "verifier", "validate"), required=True)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--release-root", type=_absolute, required=True)
    parser.add_argument("--stage-root", type=_absolute)
    parser.add_argument("--final-path", type=_absolute, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--sealed", type=_absolute)
    parser.add_argument("--expected-attempt-id")
    parser.add_argument("--expected-generation-token")
    parser.add_argument("--force-cross-filesystem", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args(argv)


def _require_common(args: argparse.Namespace) -> None:
    expected_config = PROJECT_ROOT / "configs/hpc.yaml"
    if args.config.resolve(strict=True) != expected_config:
        raise ValueError("T7 requires the checked-in configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError(
            "T7 requires exactly --set inference.fallback_to_heuristic=false"
        )
    if args.mode != "validate" and (not args.run_id or not args.gpu_uuid):
        raise ValueError("T7 run ID and GPU UUID are required")


def _common(args: argparse.Namespace) -> list[str]:
    return [
        "--config", str(args.config),
        "--release-root", str(args.release_root),
        "--final-path", str(args.final_path),
        "--run-id", args.run_id,
        "--gpu-uuid", args.gpu_uuid,
        "--set", "inference.fallback_to_heuristic=false",
    ]


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if type(value) is dict:
            return value
    raise RuntimeError("T7 worker emitted no SEALED JSON receipt")


def _run_or_worker(args: argparse.Namespace) -> int:
    if args.stage_root is None:
        raise ValueError("T7 stage root is required")
    if args.mode == "worker":
        with hold_verified_t7_release(args.release_root) as release:
            result = run_t7_worker(
                stage_root=args.stage_root,
                final_path=args.final_path,
                release=release,
                run_id=args.run_id,
                gpu_uuid=args.gpu_uuid,
            )
        print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
        return 0

    command = [
        sys.executable, "-I", "-B", str(Path(__file__).resolve()),
        "--mode", "worker", *_common(args),
        "--stage-root", str(args.stage_root),
    ]
    worker = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={**os.environ, "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0"},
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        check=False,
    )
    if worker.stdout:
        print(worker.stdout, end="", flush=True)
    if worker.returncode != 0:
        return int(worker.returncode)
    receipt = _last_json(worker.stdout)
    if receipt.get("status") != "SEALED_PENDING_INDEPENDENT_VERIFICATION":
        raise RuntimeError("T7 worker did not produce one SEALED receipt")
    verifier = subprocess.run(
        [
            sys.executable, "-I", "-B", str(Path(__file__).resolve()),
            "--mode", "verifier", *_common(args),
            "--sealed", str(receipt["staging_path"]),
            "--expected-attempt-id", str(receipt["attempt_id"]),
            "--expected-generation-token", str(receipt["generation_token"]),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0"},
        check=False,
    )
    return int(verifier.returncode)


def _verify(args: argparse.Namespace) -> int:
    if not args.sealed or not args.expected_attempt_id or not args.expected_generation_token:
        raise ValueError("T7 verifier requires exact SEALED identity")
    with hold_verified_t7_release(args.release_root) as release, open_t7_sealed(
        args.sealed,
        expected_attempt_id=args.expected_attempt_id,
        expected_generation_token=args.expected_generation_token,
    ) as sealed:
        release.revalidate()
        publication, verification = verify_and_publish_t7_sealed(
            sealed,
            final_path=args.final_path,
            release=release,
            run_id=args.run_id,
            gpu_uuid=args.gpu_uuid,
            force_cross_filesystem=args.force_cross_filesystem,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_path": str(publication.final_path),
                "attempt_id": publication.attempt_id,
                "generation_token": publication.generation_token,
                "verification_sha256": publication.verification_sha256,
                "gate_sha256": publication.gate_sha256,
                "publish_mode": publication.publish_mode,
                "verification": verification,
            },
            sort_keys=True,
            ensure_ascii=True,
        ),
        flush=True,
    )
    print(PASS_MARKER, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_common(args)
    if args.mode in {"run", "worker"}:
        return _run_or_worker(args)
    if args.mode == "verifier":
        return _verify(args)
    print(
        json.dumps(load_t7_verified_gate(args.final_path), sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
