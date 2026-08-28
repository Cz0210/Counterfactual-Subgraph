#!/usr/bin/env python3
"""Hold Taste main-v2/T2/T3 authorities across NeuroSED worker publication."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.process_identity_v2 import require_auto_termination_disabled  # noqa: E402
from src.utils.tastemolnet_neurosed_authority import (  # noqa: E402
    hold_tastemolnet_neurosed_data_authority,
)
from src.utils.retained_readonly_file import hold_readonly_file  # noqa: E402


def _hold_controller_authority(*args: object, **kwargs: object) -> object:
    try:
        from src.utils.autodl_tastemolnet_main_v2 import (
            hold_taste_main_v2_controller_authority,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Taste main-v2 controller authority integration is required"
        ) from exc
    return hold_taste_main_v2_controller_authority(*args, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--neurosed-config", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--t2-receipt-root", type=Path, required=True)
    parser.add_argument("--t2-source-bundle-root", type=Path, required=True)
    parser.add_argument("--t3-final-root", type=Path, required=True)
    parser.add_argument("--controller-receipt", type=Path, required=True)
    parser.add_argument("--controller-heartbeat", type=Path, required=True)
    parser.add_argument("--expected-controller-id", required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--execution-git-tree", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-cuda-tolerance", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    require_auto_termination_disabled()
    python = str(args.python)
    attempt_id = str(uuid.uuid4())
    with hold_readonly_file(args.neurosed_config) as config_file, _hold_controller_authority(
        args.controller_receipt,
        args.controller_heartbeat,
        expected_controller_id=args.expected_controller_id,
        expected_git_commit=args.execution_git_commit,
        expected_git_tree=args.execution_git_tree,
    ) as controller, hold_tastemolnet_neurosed_data_authority(
        t2_receipt_root=args.t2_receipt_root,
        t2_source_bundle_root=args.t2_source_bundle_root,
        t3_final_root=args.t3_final_root,
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
    ) as data_authority:
        controller_initial = dict(controller.evidence)
        input_hashes = {
            "controller_receipt": controller_initial["receipt_sha256"],
            "worker_initial_heartbeat": controller_initial["heartbeat_sha256"],
            **data_authority.input_hashes,
        }
        worker_command = [
            python,
            "-B",
            str(PROJECT_ROOT / "scripts/autodl/managed_worker_v2.py"),
            "--stage-root",
            str(args.stage_root),
            "--controller-id",
            args.expected_controller_id,
            "--task-id",
            "TASTE_GCF_NEUROSED",
            "--git-commit",
            args.execution_git_commit,
            "--config-hash",
            config_file.sha256,
        ]
        for name, digest in sorted(input_hashes.items()):
            worker_command.extend(("--input-hash", f"{name}={digest}"))
        worker_command.extend(
            [
                "--attempt-id",
                attempt_id,
                "--cwd",
                str(PROJECT_ROOT),
                "--config",
                str(args.config),
                "--",
                python,
                "-B",
                str(PROJECT_ROOT / "scripts/autodl/train_tastemolnet_neurosed.py"),
                "--config",
                str(args.config),
                "--neurosed-config",
                str(args.neurosed_config),
                "--train-csv",
                str(args.train_csv),
                "--validation-csv",
                str(args.validation_csv),
                "--t2-receipt-root",
                str(args.t2_receipt_root),
                "--t2-source-bundle-root",
                str(args.t2_source_bundle_root),
                "--t3-final-root",
                str(args.t3_final_root),
                "--controller-receipt",
                str(args.controller_receipt),
                "--controller-heartbeat",
                str(args.controller_heartbeat),
                "--expected-controller-id",
                args.expected_controller_id,
                "--expected-controller-receipt-sha256",
                controller_initial["receipt_sha256"],
                "--expected-controller-heartbeat-sha256",
                controller_initial["heartbeat_sha256"],
                "--expected-controller-heartbeat-sequence",
                str(controller_initial["sequence"]),
                "--expected-controller-heartbeat-uuid",
                controller_initial["heartbeat_uuid"],
                "--expected-neurosed-config-sha256",
                config_file.sha256,
                "--execution-git-commit",
                args.execution_git_commit,
                "--execution-git-tree",
                args.execution_git_tree,
                "--device",
                args.device,
            ]
        )
        subprocess.run(worker_command, cwd=PROJECT_ROOT, check=True)
        data_authority.revalidate()
        controller.revalidate()
        config_file.revalidate()
        sealed = list(
            (args.stage_root / "attempts" / attempt_id).glob(
                "worker_staging/*/SEALED.json"
            )
        )
        if len(sealed) != 1:
            raise RuntimeError("managed NeuroSED worker lacks one SEALED generation")
        verifier_command = [
            python,
            "-B",
            str(PROJECT_ROOT / "scripts/autodl/verify_tastemolnet_neurosed.py"),
            "--config",
            str(args.config),
            "--set",
            "inference.fallback_to_heuristic=false",
            "--sealed",
            str(sealed[0]),
            "--final-path",
            str(args.final_root),
            "--expected-attempt-id",
            attempt_id,
            "--train-csv",
            str(args.train_csv),
            "--validation-csv",
            str(args.validation_csv),
            "--t2-receipt-root",
            str(args.t2_receipt_root),
            "--t2-source-bundle-root",
            str(args.t2_source_bundle_root),
            "--t3-final-root",
            str(args.t3_final_root),
            "--controller-receipt",
            str(args.controller_receipt),
            "--controller-heartbeat",
            str(args.controller_heartbeat),
            "--expected-controller-id",
            args.expected_controller_id,
            "--expected-controller-receipt-sha256",
            controller_initial["receipt_sha256"],
            "--execution-git-commit",
            args.execution_git_commit,
            "--execution-git-tree",
            args.execution_git_tree,
        ]
        if args.require_cuda_tolerance:
            verifier_command.append("--require-cuda-tolerance")
        subprocess.run(verifier_command, cwd=PROJECT_ROOT, check=True)
        data_authority.revalidate()
        controller.revalidate()
        config_file.revalidate()
    print(f"neurosed_attempt_id={attempt_id}", flush=True)
    print(f"neurosed_root={args.final_root}", flush=True)
    print(f"neurosed_checkpoint={args.final_root / 'artifacts/best.pt'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f"Taste NeuroSED managed launch quarantined: {exc}", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
