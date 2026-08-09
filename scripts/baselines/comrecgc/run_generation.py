#!/usr/bin/env python3
"""Run pinned native or project-adapted COMRECGC generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import GenerationParameters  # noqa: E402
from src.baselines.comrecgc.runtime import run_native_smoke, run_project_generation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("native", "project"), required=True)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--upstream-root", default="external/COMRECGC")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--source-csv")
    parser.add_argument("--gnn-checkpoint")
    parser.add_argument("--distance-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trace-output-dir",
        help="Optional project-owned action trace directory; does not modify upstream output.",
    )
    parser.add_argument(
        "--parity-reference",
        help="Trace-disabled counterfactuals.pt used for normalized trace parity.",
    )
    parser.add_argument("--trusted-dataset-payload")
    parser.add_argument("--expected-cache-inventory-sha256")
    parser.add_argument(
        "--graph-state-dir",
        help="Project-owned authoritative graph-state store for full random walks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parameters = GenerationParameters.for_mode(args.mode)
    upstream = Path(args.upstream_root)
    if not upstream.is_absolute():
        upstream = Path(args.project_root) / upstream
    if args.route == "native":
        manifest = run_native_smoke(
            project_root=args.project_root,
            upstream_root=upstream,
            dataset=args.dataset,
            output_dir=args.output_dir,
            parameters=parameters,
            parent_limit=args.parent_limit or (32 if args.mode == "smoke" else 0),
            device=args.device,
            mode=args.mode,
            trusted_dataset_payload=args.trusted_dataset_payload,
            expected_cache_inventory_sha256=args.expected_cache_inventory_sha256,
        )
    else:
        required = {
            "dataset_dir": args.dataset_dir,
            "gnn_checkpoint": args.gnn_checkpoint,
            "distance_checkpoint": args.distance_checkpoint,
            "parent_limit": args.parent_limit,
        }
        missing = [name for name, value in required.items() if value in (None, "")]
        if missing:
            raise ValueError(f"Project generation missing required arguments: {missing}")
        manifest = run_project_generation(
            project_root=args.project_root,
            upstream_root=upstream,
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
            source_csv=args.source_csv,
            gnn_checkpoint=args.gnn_checkpoint,
            distance_checkpoint=args.distance_checkpoint,
            output_dir=args.output_dir,
            mode=args.mode,
            parent_limit=int(args.parent_limit),
            parameters=parameters,
            device=args.device,
            batch_size=args.batch_size,
            resume=args.resume,
            trace_output_dir=args.trace_output_dir,
            parity_reference_path=args.parity_reference,
            graph_state_dir=args.graph_state_dir,
        )
    print(json.dumps(manifest, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
