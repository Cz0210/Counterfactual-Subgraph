#!/usr/bin/env python3
"""Seal one fresh Taste T14 Route C task spec for the one-shot owner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.tastemolnet_t14_route_c_fresh import (  # noqa: E402
    audit_no_live_t14_science_owner,
    audit_route_c_matrix_cell_absent,
    build_spec,
    file_sha256,
    validate_fresh_retry_retirement_receipt,
    write_spec,
)


SCIENCE_ENVIRONMENT = (
    "RUN_TASTEMOLNET",
    "TASTE_RESEARCH_COMPUTE_ALLOWED",
    "TASTE_PAPER_RESULTS_ALLOWED",
    "TASTE_DATA_REDISTRIBUTION_ALLOWED",
    "RUN_GNN_ABLATION",
    "RUN_LLM_ABLATION",
    "TASTEMOLNET_T2_ADOPTION_ROOT",
    "TASTEMOLNET_T2_ADOPTION_GATE_SHA256",
    "TASTEMOLNET_T2_ADOPTION_RECEIPT_SHA256",
    "TASTEMOLNET_T2_SOURCE_EVIDENCE_SHA256",
    "TASTEMOLNET_T3_OUTPUT_ROOT",
    "TASTEMOLNET_T4_OUTPUT_ROOT",
    "TASTEMOLNET_TRAIN_CSV",
    "COMRECGC_OFFICIAL_ROOT",
    "AUTODL_DATA_ROOT",
    "AUTODL_PYTHON",
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--attempt-uuid", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--python", type=_absolute, required=True)
    parser.add_argument("--science-wrapper", type=_absolute, required=True)
    parser.add_argument("--owner-entrypoint", type=_absolute, required=True)
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--owner-root", type=_absolute, required=True)
    parser.add_argument("--forbidden-legacy-root", type=_absolute, required=True)
    parser.add_argument("--cgroup-limit-file", type=_absolute, required=True)
    parser.add_argument("--cgroup-current-file", type=_absolute, required=True)
    parser.add_argument("--cgroup-failcnt-file", type=_absolute, required=True)
    parser.add_argument("--max-process-rss-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--launch-headroom-bytes", type=int, default=384 * 1024**3)
    parser.add_argument("--runtime-headroom-bytes", type=int, default=96 * 1024**3)
    parser.add_argument("--sample-seconds", type=float, default=30.0)
    parser.add_argument("--launch-samples-required", type=int, default=3)
    parser.add_argument("--runtime-low-headroom-samples", type=int, default=3)
    parser.add_argument("--fresh-retry-receipt", type=_absolute)
    parser.add_argument("--spec-out", type=_absolute, required=True)
    parser.add_argument("--no-live-owner-receipt", type=_absolute, required=True)
    parser.add_argument("--matrix-authority-state", type=_absolute, required=True)
    parser.add_argument("--matrix-authority-lock", type=_absolute, required=True)
    parser.add_argument("--matrix-cell-absent-receipt", type=_absolute, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file() or args.config.is_symlink():
        raise ValueError("T14 Route C requires one physical config")
    environment = {name: os.environ.get(name, "") for name in SCIENCE_ENVIRONMENT}
    missing = sorted(name for name, value in environment.items() if not value)
    if missing:
        raise ValueError(f"T14 Route C science environment is incomplete: {missing}")
    audit_no_live_t14_science_owner(
        control_root=Path(os.environ["AUTODL_CONTROL_ROOT"]),
        receipt_path=args.no_live_owner_receipt,
    )
    audit_route_c_matrix_cell_absent(
        state_path=args.matrix_authority_state,
        lock_path=args.matrix_authority_lock,
        receipt_path=args.matrix_cell_absent_receipt,
    )
    fresh_retry = None
    if args.fresh_retry_receipt is not None:
        authorization = {
            "ALLOW_T14_ROUTE_C_FRESH_RETRY_AFTER_RESOURCE_WATCHDOG": "1",
            "T14_ROUTE_C_FRESH_RETRY_MAX_ATTEMPTS": "1",
            "PRESERVE_FAILED_ROUTE_C_ATTEMPT": "1",
            "REUSE_PARTIAL_STEP161": "0",
        }
        for name, expected in authorization.items():
            if os.environ.get(name) != expected:
                raise ValueError(f"T14 Route C retry authorization changed: {name}")
        retirement = validate_fresh_retry_retirement_receipt(
            args.fresh_retry_receipt
        )
        train_csv = Path(environment["TASTEMOLNET_TRAIN_CSV"])
        t3_checkpoint = (
            Path(environment["TASTEMOLNET_T3_OUTPUT_ROOT"])
            / "artifacts"
            / "checkpoint"
        )
        model_path = t3_checkpoint / "model.pt"
        split_path = t3_checkpoint / "split_manifest.json"
        for label, path in (
            ("train split", train_csv),
            ("T3 GINE", model_path),
            ("T3 split manifest", split_path),
        ):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"T14 Route C retry {label} is absent or indirect")
        old_cohort_manifest = (
            Path(retirement["old_output_root"]) / "cohort_manifest.json"
        )
        if not old_cohort_manifest.is_file() or old_cohort_manifest.is_symlink():
            raise ValueError("T14 Route C failed cohort manifest is absent")
        old_cohort = json.loads(old_cohort_manifest.read_text(encoding="utf-8"))
        cohort_sha = str(old_cohort.get("cohort_jsonl_sha256") or "")
        if len(cohort_sha) != 64 or any(
            character not in "0123456789abcdef" for character in cohort_sha
        ):
            raise ValueError("T14 Route C failed cohort SHA is invalid")
        if (
            args.launch_headroom_bytes != 384 * 1024**3
            or args.runtime_headroom_bytes != 96 * 1024**3
            or args.launch_samples_required != 3
            or args.runtime_low_headroom_samples != 3
            or args.sample_seconds != 30.0
        ):
            raise ValueError("T14 Route C retry memory policy was weakened")
        fresh_retry = {
            "schema_version": "tastemolnet_t14_route_c_fresh_retry_task_v1",
            "retry_index": 1,
            "max_retries": 1,
            "reuse_partial_step161": False,
            "preserve_failed_attempt": True,
            "fresh_uuid": args.attempt_uuid,
            "fresh_output_root": str(args.output_root),
            "previous_attempt_uuid": retirement["old_attempt_uuid"],
            "previous_output_root": retirement["old_output_root"],
            "retirement_receipt": str(args.fresh_retry_receipt),
            "retirement_receipt_sha256": file_sha256(args.fresh_retry_receipt),
            "dataset_sha256": file_sha256(train_csv),
            "train_split_sha256": file_sha256(split_path),
            "cohort_sha256": cohort_sha,
            "t3_gine_sha256": file_sha256(model_path),
            "seed": 7,
            "config_sha256": file_sha256(args.config),
            "candidate_capacity": 50_000,
            "m_configured_max": 20_000,
            "m_fallback_max": 25_000,
            "min_valid_unique": 10,
            "gpu_index": 2,
            "memory_policy": {
                "start_headroom_bytes": 384 * 1024**3,
                "runtime_reserve_bytes": 96 * 1024**3,
                "launch_samples_required": 3,
                "runtime_low_headroom_samples": 3,
                "sample_seconds": 30.0,
            },
            "checkpoint_policy": {
                "early_steps": [50, 100, 250, 500],
                "production_steps": [
                    2_500,
                    5_000,
                    7_500,
                    10_000,
                    12_500,
                    15_000,
                    17_500,
                    20_000,
                ],
                "fresh_process_reload_each_checkpoint": True,
                "route_c_500_promoted_to_full_without_replay": True,
            },
            "matrix_authority_root": str(args.matrix_authority_state.parent),
            "matrix_authority_state": str(args.matrix_authority_state),
            "matrix_authority_lock": str(args.matrix_authority_lock),
        }
    spec = build_spec(
        attempt_uuid=args.attempt_uuid,
        execution_commit=args.execution_commit,
        python=args.python,
        science_wrapper=args.science_wrapper,
        owner_entrypoint=args.owner_entrypoint,
        output_root=args.output_root,
        owner_root=args.owner_root,
        cgroup_limit_path=args.cgroup_limit_file,
        cgroup_current_path=args.cgroup_current_file,
        cgroup_failcnt_path=args.cgroup_failcnt_file,
        forbidden_legacy_root=args.forbidden_legacy_root,
        science_environment=environment,
        storage_mode="lowmemory",
        canary_role="PROMOTABLE_LOW_MEMORY",
        max_process_rss_bytes=args.max_process_rss_bytes,
        launch_headroom_bytes=args.launch_headroom_bytes,
        runtime_headroom_bytes=args.runtime_headroom_bytes,
        sample_seconds=args.sample_seconds,
        launch_samples_required=args.launch_samples_required,
        runtime_low_headroom_samples=args.runtime_low_headroom_samples,
        fresh_retry=fresh_retry,
    )
    write_spec(args.spec_out, spec)
    print(
        json.dumps(
            {
                "status": "PASS",
                "task_kind": "T14_ROUTE_C_TASK_SPEC",
                "spec_path": str(args.spec_out),
                "spec_sha256": spec["spec_sha256"],
                "output_root": spec["output_root"],
                "legacy_checkpoint_loaded": False,
                "fresh_retry": fresh_retry is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
