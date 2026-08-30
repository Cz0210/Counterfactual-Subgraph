#!/usr/bin/env python3
"""Cluster COMRECGC graph recourses and export real graph medoids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import RecourseParameters  # noqa: E402
from src.baselines.comrecgc.recourse import run_common_recourse  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--dataset", choices=("aids", "mutagenicity", "bace"), required=True
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--upstream-root", default="external/COMRECGC")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-csv")
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--distance-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--engine",
        choices=("legacy_in_memory", "external_memory_exact_v1"),
        default="legacy_in_memory",
    )
    parser.add_argument("--external-max-rss-gb", type=float, default=96.0)
    parser.add_argument("--external-query-block-size", type=int, default=8)
    parser.add_argument(
        "--external-checkpoint-interval-blocks", type=int, default=1
    )
    parser.add_argument(
        "--external-dbscan-shortcut-mode",
        choices=(
            "disabled",
            "all_core_one_component_adaptive_anchor_v1",
            "sklearn_float64_exact_multi_component_v1",
        ),
        default="disabled",
    )
    parser.add_argument("--external-shortcut-seed-count", type=int, default=3)
    parser.add_argument("--external-shortcut-failure-cap", type=int, default=4096)
    parser.add_argument(
        "--external-shortcut-query-block-size", type=int, default=65536
    )
    parser.add_argument(
        "--external-exact-fallback-max-samples", type=int, default=100000
    )
    parser.add_argument("--external-summary-block-size", type=int, default=65536)
    parser.add_argument(
        "--external-pair-store-source-manifest",
        help=(
            "Completed external pair-store manifest to adopt by validated "
            "physical read-only reference into this fresh output root."
        ),
    )
    parser.add_argument(
        "--external-pair-store-source-checkpoint",
        help=(
            "Immutable chunk-phase checkpoint whose hash-closed chunks form a "
            "complete Cartesian pair source. Mutually exclusive with the "
            "terminal source manifest."
        ),
    )
    parser.add_argument(
        "--external-pair-store-source-owner-root",
        help="Old read-only run root whose owner process must be absent.",
    )
    parser.add_argument(
        "--external-close-pair-view-manifest",
        help=(
            "Hash-closed logical theta-close view required when adopting a "
            "physical Cartesian chunk snapshot."
        ),
    )
    parser.add_argument(
        "--external-vector-cache-root",
        help="Local-XFS root for the reconstructible contiguous vector cache.",
    )
    parser.add_argument(
        "--external-vector-cache-lock",
        help="Exclusive lock used while constructing the local vector cache.",
    )
    parser.add_argument(
        "--external-vector-cache-route-lock",
        help=(
            "Independent outer route lock required only to recover a malformed "
            "pre-allocation cache artifact."
        ),
    )
    parser.add_argument(
        "--external-vector-cache-min-free-gb", type=float, default=3.0
    )
    parser.add_argument("--external-vector-cache-proc-root", default="/proc")
    parser.add_argument("--expected-sklearn-version", default="1.7.2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    upstream = Path(args.upstream_root)
    if not upstream.is_absolute():
        upstream = PROJECT_ROOT / upstream
    manifest = run_common_recourse(
        upstream_root=upstream,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        source_csv=args.source_csv,
        generation_dir=args.generation_dir,
        distance_checkpoint=args.distance_checkpoint,
        output_dir=args.output_dir,
        mode=args.mode,
        parent_limit=args.parent_limit,
        parameters=RecourseParameters.for_mode(args.mode),
        device=args.device,
        batch_size=args.batch_size,
        resume=args.resume,
        engine=args.engine,
        external_max_rss_bytes=int(float(args.external_max_rss_gb) * 1024**3),
        external_query_block_size=args.external_query_block_size,
        external_checkpoint_interval_blocks=args.external_checkpoint_interval_blocks,
        external_dbscan_shortcut_mode=args.external_dbscan_shortcut_mode,
        external_shortcut_seed_count=args.external_shortcut_seed_count,
        external_shortcut_failure_cap=args.external_shortcut_failure_cap,
        external_shortcut_query_block_size=args.external_shortcut_query_block_size,
        external_exact_fallback_max_samples=(
            args.external_exact_fallback_max_samples
        ),
        external_summary_block_size=args.external_summary_block_size,
        external_pair_store_source_manifest=(
            args.external_pair_store_source_manifest
        ),
        external_pair_store_source_checkpoint=(
            args.external_pair_store_source_checkpoint
        ),
        external_pair_store_source_owner_root=(
            args.external_pair_store_source_owner_root
        ),
        external_close_pair_view_manifest=args.external_close_pair_view_manifest,
        external_vector_cache_root=args.external_vector_cache_root,
        external_vector_cache_lock=args.external_vector_cache_lock,
        external_vector_cache_route_lock=args.external_vector_cache_route_lock,
        external_vector_cache_min_free_bytes=int(
            float(args.external_vector_cache_min_free_gb) * 1024**3
        ),
        external_vector_cache_proc_root=args.external_vector_cache_proc_root,
        expected_sklearn_version=args.expected_sklearn_version,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
