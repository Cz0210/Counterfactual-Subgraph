#!/usr/bin/env python3
"""Adopt one verified T8 25-epoch deadline PASS into managed-v2.

``run`` deliberately launches a worker process that can only produce SEALED
evidence and a later verifier process that alone may publish PASS.  The source
deadline science and private state remain retained in place and are never
copied into the managed terminal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_globalgce_full import validate_t8_pass  # noqa: E402
from src.baselines.tastemolnet_globalgce_smoke import (  # noqa: E402
    TasteGlobalGCESmokeError,
)
from src.utils.tastemolnet_t8_deadline_managed_v2 import (  # noqa: E402
    ADOPTION_MARKER,
    DeadlineRecoveryInputs,
    create_deadline_adoption_sealed,
    inspect_clean_execution,
    verify_and_publish_deadline_adoption,
)
from src.utils.terminal_publisher_v2 import (  # noqa: E402
    open_sealed_worker_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("run", "worker", "verifier", "validate"),
        default="run",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--deadline-output-root", type=Path)
    parser.add_argument("--deadline-state-root", type=Path)
    parser.add_argument("--deadline-attempt-id")
    parser.add_argument("--recovery-source-attempt-id")
    parser.add_argument("--t3-output", type=Path)
    parser.add_argument("--t4-output", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--final-path", type=Path, required=True)
    parser.add_argument("--managed-attempt-id")
    parser.add_argument("--run-id")
    parser.add_argument("--sealed-root", type=Path)
    parser.add_argument("--expected-generation-token")
    parser.add_argument("--force-cross-filesystem", action="store_true")
    return parser


def _require_args(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption is missing: " + ", ".join(missing)
        )


def _inputs(args: argparse.Namespace) -> DeadlineRecoveryInputs:
    names = (
        "deadline_output_root",
        "deadline_state_root",
        "deadline_attempt_id",
        "recovery_source_attempt_id",
        "t3_output",
        "t4_output",
        "gnn_checkpoint",
        "train_csv",
        "official_root",
    )
    _require_args(args, names)
    return DeadlineRecoveryInputs(
        config=args.config,
        deadline_output_root=args.deadline_output_root,
        deadline_state_root=args.deadline_state_root,
        deadline_attempt_id=args.deadline_attempt_id,
        recovery_source_attempt_id=args.recovery_source_attempt_id,
        t3_output=args.t3_output,
        t4_output=args.t4_output,
        gnn_checkpoint=args.gnn_checkpoint,
        train_csv=args.train_csv,
        official_root=args.official_root,
    )


def _common_child_args(args: argparse.Namespace) -> list[str]:
    inputs = _inputs(args)
    return [
        "--config",
        str(inputs.config),
        "--set",
        "inference.fallback_to_heuristic=false",
        "--deadline-output-root",
        str(inputs.deadline_output_root),
        "--deadline-state-root",
        str(inputs.deadline_state_root),
        "--deadline-attempt-id",
        inputs.deadline_attempt_id,
        "--recovery-source-attempt-id",
        inputs.recovery_source_attempt_id,
        "--t3-output",
        str(inputs.t3_output),
        "--t4-output",
        str(inputs.t4_output),
        "--gnn-checkpoint",
        str(inputs.gnn_checkpoint),
        "--train-csv",
        str(inputs.train_csv),
        "--official-root",
        str(inputs.official_root),
        "--final-path",
        str(args.final_path),
        "--run-id",
        str(args.run_id),
    ]


def _parse_last_json(stdout: str, *, label: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise TasteGlobalGCESmokeError(f"T8 deadline adoption {label} was silent")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption {label} did not emit terminal JSON"
        ) from exc
    if type(value) is not dict:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption {label} terminal JSON is not an object"
        )
    return value


def _run_child(arguments: list[str], *, label: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["AUTO_TERMINATE_UNCONTROLLED_CHILDREN"] = "0"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise TasteGlobalGCESmokeError(
            f"T8 deadline adoption {label} failed ({completed.returncode}): "
            + completed.stderr[-4000:]
        )
    return _parse_last_json(completed.stdout, label=label)


def _run(args: argparse.Namespace) -> int:
    _require_args(args, ("stage_root", "managed_attempt_id", "run_id"))
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption forbids heuristic fallback"
        )
    common = _common_child_args(args)
    worker = _run_child(
        [
            "--mode",
            "worker",
            *common,
            "--stage-root",
            str(args.stage_root),
            "--managed-attempt-id",
            args.managed_attempt_id,
        ],
        label="worker",
    )
    if (
        worker.get("status") != "SEALED_PENDING_INDEPENDENT_VERIFICATION"
        or worker.get("attempt_id") != args.managed_attempt_id
    ):
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption worker did not produce the expected SEALED"
        )
    verifier_args = [
        "--mode",
        "verifier",
        *common,
        "--sealed-root",
        str(worker["staging_path"]),
        "--managed-attempt-id",
        args.managed_attempt_id,
        "--expected-generation-token",
        str(worker["generation_token"]),
    ]
    if args.force_cross_filesystem:
        verifier_args.append("--force-cross-filesystem")
    verified = _run_child(verifier_args, label="verifier")
    if verified.get("status") != "PASS" or verified.get("marker") != ADOPTION_MARKER:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption verifier did not publish typed PASS"
        )
    print(ADOPTION_MARKER)
    print(json.dumps(verified, sort_keys=True))
    return 0


def _worker(args: argparse.Namespace) -> int:
    _require_args(args, ("stage_root", "managed_attempt_id", "run_id"))
    execution_commit = inspect_clean_execution(REPO_ROOT)
    result = create_deadline_adoption_sealed(
        inputs=_inputs(args),
        stage_root=args.stage_root,
        final_path=args.final_path,
        managed_attempt_id=args.managed_attempt_id,
        run_id=args.run_id,
        execution_commit=execution_commit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _verifier(args: argparse.Namespace) -> int:
    _require_args(
        args,
        (
            "sealed_root",
            "managed_attempt_id",
            "expected_generation_token",
            "run_id",
        ),
    )
    execution_commit = inspect_clean_execution(REPO_ROOT)
    with open_sealed_worker_artifact(
        args.sealed_root,
        expected_attempt_id=args.managed_attempt_id,
        expected_generation_token=args.expected_generation_token,
    ) as held:
        publication, verification = verify_and_publish_deadline_adoption(
            held,
            inputs=_inputs(args),
            final_path=args.final_path,
            run_id=args.run_id,
            execution_commit=execution_commit,
            force_cross_filesystem=args.force_cross_filesystem,
        )
    validated_path, adoption = validate_t8_pass(args.final_path)
    result = {
        "status": "PASS",
        "marker": ADOPTION_MARKER,
        "final_path": str(validated_path),
        "publication": {
            **asdict(publication),
            "final_path": str(publication.final_path),
        },
        "typed_verification": verification,
        "t13_adoption_sha256": adoption["adoption_sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _validate(args: argparse.Namespace) -> int:
    final, adoption = validate_t8_pass(args.final_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "marker": ADOPTION_MARKER,
                "final_path": str(final),
                "t13_adoption_sha256": adoption["adoption_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode != "validate" and args.set != [
        "inference.fallback_to_heuristic=false"
    ]:
        raise TasteGlobalGCESmokeError(
            "T8 deadline adoption requires exactly one heuristic-disable override"
        )
    if args.mode == "run":
        return _run(args)
    if args.mode == "worker":
        return _worker(args)
    if args.mode == "verifier":
        return _verifier(args)
    return _validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
