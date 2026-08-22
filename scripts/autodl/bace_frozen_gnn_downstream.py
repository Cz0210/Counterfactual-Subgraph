#!/usr/bin/env python3
"""Foreground AutoDL CLI for the provenance-clean BACE B7-prep and B8--B14 route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_marker, utc_now

from src.eval.bace_frozen_gnn_pool import merge_pool_shards, run_pool_shard
from src.eval.bace_frozen_gnn_prep import (
    PREP_ACTIONS,
    run_b7_parallel_prep,
    run_postfreeze_test_shard_manifest,
)
from src.eval.bace_frozen_gnn_selection import (
    run_b12_selector,
    run_b14_manifest_freeze,
)
from src.eval.bace_frozen_gnn_verification import (
    merge_verification_shards,
    run_verification_shard,
)


def _shard_index(value: str) -> int:
    normalized = str(value).strip().lower()
    if normalized.startswith("shard-"):
        normalized = normalized.removeprefix("shard-")
    try:
        result = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shard index: {value!r}") from exc
    if not 0 <= result < 4:
        raise argparse.ArgumentTypeError("shard index must be in [0, 3]")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    commands = parser.add_subparsers(dest="action", required=True)

    prep = commands.add_parser("prep", help="Run one B6-released B7-parallel prep action")
    prep.add_argument("--prep-action", choices=PREP_ACTIONS, required=True)
    prep.add_argument("--b6-output", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--calibration-split")
    prep.add_argument("--train-split")
    prep.add_argument("--gnn-checkpoint")
    prep.add_argument("--molclr-root")
    prep.add_argument("--molclr-checkpoint")
    prep.add_argument("--node-embedding-cache-dir")
    prep.add_argument("--planned-output-root", action="append", default=[])
    prep.add_argument("--device", default="cuda:0")
    prep.add_argument("--batch-size", type=int, default=256)

    pool = commands.add_parser("pool-shard", help="Run one fixed B8/B9 train-parent shard")
    pool.add_argument("--stage", choices=("B8_POOL_BASE", "B9_POOL_HIGHTEMP"), required=True)
    pool.add_argument("--train-split", required=True)
    pool.add_argument("--b7-output", required=True)
    pool.add_argument("--policy-checkpoint", required=True)
    pool.add_argument("--base-model-path", required=True)
    pool.add_argument("--gnn-checkpoint", required=True)
    pool.add_argument("--output-dir", required=True)
    pool.add_argument("--shard-index", type=_shard_index, required=True)
    pool.add_argument("--parent-shard-manifest")
    pool.add_argument("--device", default="cuda:0")
    pool.add_argument("--batch-size", type=int, default=1)
    pool.add_argument("--oracle-batch-size", type=int, default=256)
    pool.add_argument("--resume", action="store_true")

    merge_pool = commands.add_parser("merge-pools", help="Merge four B8 plus four B9 shards")
    merge_pool.add_argument("--shard-dir", action="append", required=True)
    merge_pool.add_argument("--output-dir", required=True)

    verify = commands.add_parser("verify-shard", help="Run one fixed B11/B13 parent shard")
    verify.add_argument(
        "--stage",
        choices=("B11_CROSS_PARENT_VERIFIED", "B13_FINAL_EVAL"),
        required=True,
    )
    verify.add_argument("--split-path", required=True)
    verify.add_argument("--predecessor-output", required=True)
    verify.add_argument("--gnn-checkpoint", required=True)
    verify.add_argument("--molclr-root", required=True)
    verify.add_argument("--molclr-checkpoint", required=True)
    verify.add_argument("--output-dir", required=True)
    verify.add_argument("--shard-index", type=_shard_index, required=True)
    verify.add_argument("--parent-shard-manifest")
    verify.add_argument("--wnode-cache-db", required=True)
    verify.add_argument("--node-embedding-cache-dir", required=True)
    verify.add_argument("--frozen-selection-manifest")
    verify.add_argument("--parent-before-cache")
    verify.add_argument("--device", default="cuda:0")
    verify.add_argument("--oracle-batch-size", type=int, default=256)

    merge_verification = commands.add_parser(
        "merge-verification", help="Merge four complete B11/B13 fixed shards"
    )
    merge_verification.add_argument(
        "--stage",
        choices=("B11_CROSS_PARENT_VERIFIED", "B13_FINAL_EVAL"),
        required=True,
    )
    merge_verification.add_argument("--shard-dir", action="append", required=True)
    merge_verification.add_argument("--predecessor-output", required=True)
    merge_verification.add_argument("--output-dir", required=True)

    select = commands.add_parser("select", help="Fit/freeze B12 on calibration only")
    select.add_argument("--matrix-output", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--seed", type=int, default=13)

    test_shards = commands.add_parser(
        "prepare-test-shards",
        help="Freeze test parent IDs only after validating the B12 selector",
    )
    test_shards.add_argument("--b12-output", required=True)
    test_shards.add_argument("--test-split", required=True)
    test_shards.add_argument("--output-dir", required=True)

    freeze = commands.add_parser("freeze", help="Run manifest-only B14 final gate")
    freeze.add_argument("--b12-output", required=True)
    freeze.add_argument("--b13-output", required=True)
    freeze.add_argument("--output-dir", required=True)
    return parser


def _execute(args: argparse.Namespace) -> tuple[dict[str, object], str]:
    if args.action == "prep":
        result = run_b7_parallel_prep(
            action=args.prep_action,
            b6_output=args.b6_output,
            output_dir=args.output_dir,
            calibration_split=args.calibration_split,
            train_split=args.train_split,
            gnn_checkpoint=args.gnn_checkpoint,
            molclr_root=args.molclr_root,
            molclr_checkpoint=args.molclr_checkpoint,
            node_embedding_cache_dir=args.node_embedding_cache_dir,
            planned_output_roots=args.planned_output_root,
            device=args.device,
            batch_size=args.batch_size,
        )
        marker = "BACE_B7_PARALLEL_PREP_PASS"
    elif args.action == "pool-shard":
        result = run_pool_shard(
            stage=args.stage,
            train_split=args.train_split,
            b7_output=args.b7_output,
            policy_checkpoint=args.policy_checkpoint,
            base_model_path=args.base_model_path,
            gnn_checkpoint=args.gnn_checkpoint,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            device=args.device,
            batch_size=args.batch_size,
            oracle_batch_size=args.oracle_batch_size,
            resume=args.resume,
            parent_shard_manifest=args.parent_shard_manifest,
        )
        marker = f"BACE_{args.stage}_SHARD_PASS"
    elif args.action == "merge-pools":
        result = merge_pool_shards(
            shard_dirs=args.shard_dir,
            output_dir=args.output_dir,
        )
        marker = "BACE_B10_PASS"
    elif args.action == "verify-shard":
        result = run_verification_shard(
            stage=args.stage,
            split_path=args.split_path,
            predecessor_output=args.predecessor_output,
            gnn_checkpoint=args.gnn_checkpoint,
            molclr_root=args.molclr_root,
            molclr_checkpoint=args.molclr_checkpoint,
            output_dir=args.output_dir,
            shard_index=args.shard_index,
            wnode_cache_db=args.wnode_cache_db,
            node_embedding_cache_dir=args.node_embedding_cache_dir,
            frozen_selection_manifest=args.frozen_selection_manifest,
            parent_before_cache=args.parent_before_cache,
            parent_shard_manifest=args.parent_shard_manifest,
            device=args.device,
            oracle_batch_size=args.oracle_batch_size,
        )
        marker = f"BACE_{args.stage}_SHARD_PASS"
    elif args.action == "merge-verification":
        result = merge_verification_shards(
            stage=args.stage,
            shard_dirs=args.shard_dir,
            predecessor_output=args.predecessor_output,
            output_dir=args.output_dir,
        )
        marker = "BACE_B11_PASS" if args.stage.startswith("B11") else "BACE_B13_PASS"
    elif args.action == "select":
        result = run_b12_selector(
            matrix_output=args.matrix_output,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        marker = "BACE_B12_SELECTOR_FROZEN"
    elif args.action == "prepare-test-shards":
        result = run_postfreeze_test_shard_manifest(
            b12_output=args.b12_output,
            test_split=args.test_split,
            output_dir=args.output_dir,
        )
        marker = "BACE_B13_TEST_PARENT_MANIFEST_PASS"
    elif args.action == "freeze":
        result = run_b14_manifest_freeze(
            b12_output=args.b12_output,
            b13_output=args.b13_output,
            output_dir=args.output_dir,
        )
        marker = "BACE_B14_FINAL_PASS"
    else:  # pragma: no cover
        raise ValueError(f"Unsupported action: {args.action}")
    return result, marker


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve(strict=False)
    output_existed = output.exists()
    try:
        result, marker = _execute(args)
    except Exception as exc:
        # Preserve a machine-readable failure only when this invocation created
        # its fresh root; never write into a pre-existing scientific output.
        if not output_existed and not output.exists():
            output.mkdir(parents=True, exist_ok=False)
        owned_partial = (
            output_existed
            and output.is_dir()
            and (output / "IN_PROGRESS.json").is_file()
            and not (output / "PASS").exists()
        )
        if (not output_existed and output.is_dir()) or owned_partial:
            atomic_json(
                output / "FAIL.json",
                {
                    "status": "FAILED",
                    "action": args.action,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at": utc_now(),
                },
            )
            atomic_marker(output / "FAILED", "FAILED")
        raise
    print(json.dumps(result, sort_keys=True), flush=True)
    print(f"[{marker}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
