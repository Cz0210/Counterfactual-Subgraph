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
    parser.add_argument("--launch-headroom-bytes", type=int, default=96 * 1024**3)
    parser.add_argument("--runtime-headroom-bytes", type=int, default=48 * 1024**3)
    parser.add_argument("--sample-seconds", type=float, default=10.0)
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
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
