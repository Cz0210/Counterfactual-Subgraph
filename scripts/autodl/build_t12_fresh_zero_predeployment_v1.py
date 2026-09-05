#!/usr/bin/env python3
"""Build non-dispatchable fresh T12 10k/20k and publisher handoff specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tastemolnet_t12_fresh_zero_plan_v1 import (  # noqa: E402
    build_fresh_zero_plan,
    write_plan_bundle,
)


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path.is_symlink():
        raise argparse.ArgumentTypeError("absolute non-symlink path required")
    return path.resolve(strict=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-root", type=_absolute, required=True)
    parser.add_argument("--repo-root", type=_absolute, required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--python", type=_absolute, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--diagnostic-terminal", type=_absolute, required=True)
    parser.add_argument("--future-parity-receipt", type=_absolute, required=True)
    parser.add_argument("--managed-neurosed-root", type=_absolute, required=True)
    parser.add_argument("--t3-root", type=_absolute, required=True)
    parser.add_argument("--official-root", type=_absolute, required=True)
    parser.add_argument("--threshold-authority", type=_absolute, required=True)
    parser.add_argument("--replay-gate", type=_absolute, required=True)
    parser.add_argument("--production-root", type=_absolute, required=True)
    parser.add_argument("--postprocess-root", type=_absolute, required=True)
    parser.add_argument("--train-csv", type=_absolute, required=True)
    parser.add_argument("--calibration-csv", type=_absolute, required=True)
    parser.add_argument("--test-csv", type=_absolute, required=True)
    parser.add_argument("--gnn-checkpoint", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--threshold-contract", type=_absolute, required=True)
    parser.add_argument("--wnode-cache-db", type=_absolute, required=True)
    parser.add_argument("--node-embedding-cache-dir", type=_absolute, required=True)
    parser.add_argument("--verification-root", type=_absolute, required=True)
    parser.add_argument("--publisher-id", required=True)
    parser.add_argument("--publisher-locator", type=_absolute, required=True)
    parser.add_argument("--owner-registry", type=_absolute, required=True)
    parser.add_argument("--expected-owner-registry-sha256", required=True)
    parser.add_argument("--expected-owner-registry-file-sha256", required=True)
    parser.add_argument("--matrix-authority-root", type=_absolute, required=True)
    parser.add_argument("--diagnostic-bridge-history-root", type=_absolute, required=True)
    parser.add_argument("--nvme-disposable-index-root", type=_absolute, required=True)
    args = parser.parse_args(argv)
    if args.config != (args.repo_root / "configs/hpc.yaml").resolve(strict=False):
        raise ValueError("T12 predeployment requires execution-worktree configs/hpc.yaml")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T12 predeployment requires fail-closed inference")
    plan = build_fresh_zero_plan(
        repo_root=args.repo_root,
        python=args.python,
        config=args.config,
        execution_commit=args.execution_commit,
        attempt_id=str(uuid.uuid4()),
        generation_token=secrets.token_hex(32),
        gpu_index=args.gpu_index,
        gpu_uuid=args.gpu_uuid,
        diagnostic_terminal=args.diagnostic_terminal,
        required_parity_receipt=args.future_parity_receipt,
        managed_neurosed_root=args.managed_neurosed_root,
        t3_root=args.t3_root,
        official_root=args.official_root,
        threshold_authority=args.threshold_authority,
        replay_gate=args.replay_gate,
        production_root=args.production_root,
        postprocess_root=args.postprocess_root,
        train_csv=args.train_csv,
        calibration_csv=args.calibration_csv,
        test_csv=args.test_csv,
        gnn_checkpoint=args.gnn_checkpoint,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        threshold_contract=args.threshold_contract,
        wnode_cache_db=args.wnode_cache_db,
        node_embedding_cache_dir=args.node_embedding_cache_dir,
        verification_root=args.verification_root,
        publisher_id=args.publisher_id,
        publisher_locator=args.publisher_locator,
        owner_registry=args.owner_registry,
        expected_owner_registry_sha256=args.expected_owner_registry_sha256,
        expected_owner_registry_file_sha256=args.expected_owner_registry_file_sha256,
        matrix_authority_root=args.matrix_authority_root,
        diagnostic_bridge_history_root=args.diagnostic_bridge_history_root,
        nvme_disposable_index_root=args.nvme_disposable_index_root,
    )
    write_plan_bundle(args.output_root, plan)
    print(json.dumps(plan, sort_keys=True), flush=True)
    print("[T12_FRESH_ZERO_PREDEPLOYMENT_READY_BLOCKED_ON_PARITY]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
