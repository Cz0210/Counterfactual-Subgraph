#!/usr/bin/env python3
"""Seal the deferred T14 Route C postprocess/matrix continuation descriptor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_t14_route_c_continuation import (  # noqa: E402
    build_continuation_spec,
    write_continuation_spec,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _config(value: str) -> Path:
    if value == "configs/hpc.yaml":
        return PROJECT_ROOT / value
    return _absolute(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_config, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--route-c-spec", type=_absolute, required=True)
    parser.add_argument("--science-root", type=_absolute, required=True)
    parser.add_argument("--final-root", type=_absolute, required=True)
    parser.add_argument("--locator-path", type=_absolute, required=True)
    parser.add_argument("--calibration-csv", type=_absolute, required=True)
    parser.add_argument("--test-csv", type=_absolute, required=True)
    parser.add_argument("--t3-output-root", type=_absolute, required=True)
    parser.add_argument("--molclr-root", type=_absolute, required=True)
    parser.add_argument("--molclr-checkpoint", type=_absolute, required=True)
    parser.add_argument("--threshold-contract", type=_absolute, required=True)
    parser.add_argument("--wnode-cache-db", type=_absolute, required=True)
    parser.add_argument("--node-embedding-cache-dir", type=_absolute, required=True)
    parser.add_argument("--autodl-data-root", type=_absolute, required=True)
    parser.add_argument("--autodl-runtime-root", type=_absolute, required=True)
    parser.add_argument("--autodl-control-root", type=_absolute, required=True)
    parser.add_argument("--publisher-queue-manifest", type=_absolute, required=True)
    parser.add_argument("--publisher-heartbeat", type=_absolute, required=True)
    parser.add_argument("--publisher-pid-file", type=_absolute, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--spec-out", type=_absolute, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise ValueError("T14 continuation requires fail-closed inference")
    route_root = Path(args.route_c_spec).parent
    repo_root = Path(
        json.loads(args.route_c_spec.read_text(encoding="utf-8"))["owner_entrypoint"]
    ).parents[2]
    environment = {
        "RUN_TASTEMOLNET": "1",
        "TASTE_RESEARCH_COMPUTE_ALLOWED": "1",
        "TASTE_PAPER_RESULTS_ALLOWED": "1",
        "TASTE_DATA_REDISTRIBUTION_ALLOWED": "0",
        "RUN_GNN_ABLATION": "0",
        "RUN_LLM_ABLATION": "0",
        "TASTEMOLNET_CALIBRATION_CSV": str(args.calibration_csv),
        "TASTEMOLNET_TEST_CSV": str(args.test_csv),
        "TASTEMOLNET_T3_OUTPUT_ROOT": str(args.t3_output_root),
        "MOLCLR_ROOT": str(args.molclr_root),
        "MOLCLR_CHECKPOINT": str(args.molclr_checkpoint),
        "TASTEMOLNET_WNODE_THRESHOLD_JSON": str(args.threshold_contract),
        "WNODE_CACHE_DB": str(args.wnode_cache_db),
        "NODE_EMBEDDING_CACHE_DIR": str(args.node_embedding_cache_dir),
        "AUTODL_DATA_ROOT": str(args.autodl_data_root),
        "AUTODL_RUNTIME_ROOT": str(args.autodl_runtime_root),
        "AUTODL_CONTROL_ROOT": str(args.autodl_control_root),
        "AUTODL_PYTHON": str(
            json.loads(args.route_c_spec.read_text(encoding="utf-8"))["python"]
        ),
    }
    spec = build_continuation_spec(
        descriptor_path=args.spec_out,
        route_c_spec_path=args.route_c_spec,
        config_path=args.config,
        continuation_entrypoint=repo_root
        / "scripts/autodl/run_t14_route_c_continuation.py",
        postprocess_wrapper=repo_root
        / "scripts/autodl/run_tastemolnet_t14_comrecgc_postprocess.sh",
        postprocess_science_root=args.science_root,
        postprocess_final_root=args.final_root,
        locator_path=args.locator_path,
        publisher_queue_manifest=args.publisher_queue_manifest,
        publisher_heartbeat=args.publisher_heartbeat,
        publisher_pid_file=args.publisher_pid_file,
        postprocess_environment=environment,
        poll_seconds=args.poll_seconds,
    )
    if route_root != args.spec_out.parent:
        raise ValueError("continuation spec must share the Route C owner root")
    write_continuation_spec(args.spec_out, spec)
    print(json.dumps(spec, sort_keys=True), flush=True)
    print("[T14_ROUTE_C_CONTINUATION_SPEC_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
