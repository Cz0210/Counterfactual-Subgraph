#!/usr/bin/env python3
"""Run the T9 worker and independent verifier under one held GPU1 lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--stage-root", type=_absolute, required=True)
    parser.add_argument("--final-path", type=_absolute, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--t2-adoption-root", type=_absolute, required=True)
    parser.add_argument("--t2-adoption-gate-sha256", required=True)
    parser.add_argument("--t2-adoption-receipt-sha256", required=True)
    parser.add_argument("--t2-source-evidence-sha256", required=True)
    parser.add_argument("--t3-output-root", type=_absolute, required=True)
    parser.add_argument("--t4-output-root", type=_absolute, required=True)
    parser.add_argument("--checkpoint-dir", type=_absolute, required=True)
    parser.add_argument("--train-csv", type=_absolute, required=True)
    parser.add_argument("--official-root", type=_absolute, required=True)
    parser.add_argument("--set", action="append", default=[])
    return parser.parse_args(argv)


def _common(args: argparse.Namespace) -> list[str]:
    return [
        "--config", str(args.config),
        "--run-id", args.run_id,
        "--gpu-uuid", args.gpu_uuid,
        "--t2-adoption-root", str(args.t2_adoption_root),
        "--t2-adoption-gate-sha256", args.t2_adoption_gate_sha256,
        "--t2-adoption-receipt-sha256", args.t2_adoption_receipt_sha256,
        "--t2-source-evidence-sha256", args.t2_source_evidence_sha256,
        "--t3-output-root", str(args.t3_output_root),
        "--t4-output-root", str(args.t4_output_root),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--train-csv", str(args.train_csv),
        "--official-root", str(args.official_root),
        "--set", "inference.fallback_to_heuristic=false",
    ]


def _last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if type(value) is dict:
            return value
    raise RuntimeError("T9 worker emitted no SEALED JSON receipt")


def run(args: argparse.Namespace) -> int:
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("fail-closed inference override is required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("T9 runner must remain inside the GPU1 lock wrapper")
    if os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != args.gpu_uuid:
        raise RuntimeError("T9 runner GPU UUID differs from its lock")
    worker = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(PROJECT_ROOT / "scripts/run_tastemolnet_comrecgc_smoke.py"),
            *_common(args),
            "--stage-root", str(args.stage_root),
            "--output-dir", str(args.final_path),
        ],
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
        raise RuntimeError("T9 worker did not produce a SEALED receipt")
    verifier = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(
                PROJECT_ROOT
                / "scripts/autodl/tastemolnet_t9_comrecgc_verifier_v2.py"
            ),
            *_common(args),
            "--sealed", str(receipt["staging_path"]),
            "--final-path", str(args.final_path),
            "--expected-attempt-id", str(receipt["attempt_id"]),
            "--expected-generation-token", str(receipt["generation_token"]),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "AUTO_TERMINATE_UNCONTROLLED_CHILDREN": "0"},
        check=False,
    )
    return int(verifier.returncode)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
