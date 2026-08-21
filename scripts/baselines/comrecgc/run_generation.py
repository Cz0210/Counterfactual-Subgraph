#!/usr/bin/env python3
"""Run pinned native or project-adapted COMRECGC generation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import GenerationParameters  # noqa: E402
from src.baselines.comrecgc.generation_checkpoint import (  # noqa: E402
    scientific_command_sha256,
)
from src.baselines.comrecgc.runtime import run_native_smoke, run_project_generation  # noqa: E402


_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:password|passwd|secret|token|api[_-]?key|authorization|credential|private[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)


def _redact_cli_value(key: str, value: object) -> object:
    if _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, list):
        redacted: list[object] = []
        for item in value:
            if isinstance(item, str):
                if "=" in item:
                    item_key, _item_value = item.split("=", 1)
                    if _SECRET_KEY.search(item_key):
                        redacted.append(f"{item_key}=<redacted>")
                    else:
                        redacted.append(item)
                elif _SECRET_KEY.search(item):
                    redacted.append("<redacted>")
                else:
                    redacted.append(item)
            else:
                redacted.append(item)
        return redacted
    return value


def canonical_scientific_argv(args: argparse.Namespace) -> tuple[str, ...]:
    """Canonicalize all parsed CLI values, excluding only ``--resume``.

    Defaults are included so equivalent explicit/default invocations have one
    identity. Sensitive assignments are redacted before persistence.
    """

    canonical = ["scripts/baselines/comrecgc/run_generation.py"]
    for key, value in sorted(vars(args).items()):
        if key == "resume":
            continue
        redacted = _redact_cli_value(key, value)
        encoded = json.dumps(redacted, sort_keys=True, separators=(",", ":"))
        canonical.append(f"--{key.replace('_', '-')}={encoded}")
    return tuple(canonical)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--route", choices=("native", "project"), required=True)
    parser.add_argument(
        "--dataset", choices=("aids", "mutagenicity", "bace"), required=True
    )
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
        "--checkpoint-root",
        help="Durable completed-step checkpoints (required with --resume).",
    )
    parser.add_argument(
        "--checkpoint-mirror-root",
        help="Independent persistent mirror; mandatory for BACE full generation.",
    )
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--progress-interval-steps", type=int, default=25)
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
    parser.add_argument(
        "--storage-guard-root",
        help="Persistent scratch root monitored during full random walks.",
    )
    parser.add_argument("--storage-check-every-steps", type=int, default=500)
    parser.add_argument("--storage-min-free-gib", type=float, default=20.0)
    parser.add_argument("--storage-min-free-ratio", type=float, default=0.05)
    parser.add_argument("--storage-min-free-inodes", type=int, default=100_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scientific_argv = canonical_scientific_argv(args)
    command_sha256 = scientific_command_sha256(scientific_argv)
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
            storage_guard_root=args.storage_guard_root,
            storage_check_every_steps=args.storage_check_every_steps,
            storage_min_free_bytes=int(args.storage_min_free_gib * 1024**3),
            storage_min_free_ratio=args.storage_min_free_ratio,
            storage_min_free_inodes=args.storage_min_free_inodes,
            checkpoint_root=args.checkpoint_root,
            checkpoint_mirror_root=args.checkpoint_mirror_root,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            checkpoint_keep_last=args.checkpoint_keep_last,
            progress_interval_steps=args.progress_interval_steps,
            scientific_argv=scientific_argv,
            command_sha256=command_sha256,
        )
    print(json.dumps(manifest, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
