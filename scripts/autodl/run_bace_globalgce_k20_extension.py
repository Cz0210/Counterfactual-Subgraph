#!/usr/bin/env python3
"""Run the reviewed BACE GlobalGCE K20 controller or one internal raw round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve(strict=True).parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bace_globalgce_k20_extension import (  # noqa: E402
    BACEGlobalGCEK20Error,
    RAW_SHORTFALL_EXIT_CODE,
    run_extension,
    run_raw_round,
    unblock_deferred_signals_for_science_child,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=_absolute, required=True)
    parser.add_argument("--set", action="append", default=[])
    commands = parser.add_subparsers(dest="command", required=True)

    controller = commands.add_parser("controller")
    controller.add_argument("--controller-id", required=True)
    controller.add_argument("--output-root", type=_absolute, required=True)
    controller.add_argument("--source-manifest", type=_absolute, required=True)
    controller.add_argument("--native-train-csv", type=_absolute, required=True)
    controller.add_argument("--official-root", type=_absolute, required=True)
    controller.add_argument("--gnn-checkpoint", type=_absolute, required=True)
    controller.add_argument(
        "--protected-gpu0-process",
        required=True,
        help="Exact GPU0 BACE GCFExplainer PID:START_TICKS.",
    )
    controller.add_argument(
        "--protected-gpu3-process",
        required=True,
        help="Exact GPU3 BACE ComRecGC PID:START_TICKS.",
    )

    raw = commands.add_parser("raw-round")
    raw.add_argument("--gnn-checkpoint", type=_absolute, required=True)
    raw.add_argument("--source-manifest", type=_absolute, required=True)
    raw.add_argument("--native-train-csv", type=_absolute, required=True)
    raw.add_argument("--official-root", type=_absolute, required=True)
    raw.add_argument("--output-dir", type=_absolute, required=True)
    raw.add_argument("--expected-parent-count", type=int, required=True)
    raw.add_argument("--seed", type=int, required=True)
    raw.add_argument("--min-freq", type=int, required=True)
    raw.add_argument("--epochs", type=int, required=True)
    raw.add_argument("--top-k-native", type=int, required=True)
    raw.add_argument("--device", required=True)
    raw.add_argument("--no-resume", action="store_true", required=True)
    raw.add_argument(
        "--gspan-exact-top-k-pruning", action="store_true", required=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        expected_config = (PROJECT_ROOT / "configs/hpc.yaml").resolve(strict=True)
        if args.config.resolve(strict=True) != expected_config:
            raise BACEGlobalGCEK20Error(
                "K20 requires the immutable checkout's configs/hpc.yaml"
            )
        if args.set != ["inference.fallback_to_heuristic=false"]:
            raise BACEGlobalGCEK20Error(
                "K20 requires the exact fail-closed inference override"
            )
        if args.command == "controller":
            result = run_extension(
                controller_id=args.controller_id,
                project_root=PROJECT_ROOT,
                python=Path(sys.executable),
                config=args.config,
                output_root=args.output_root,
                source_manifest=args.source_manifest,
                native_train_csv=args.native_train_csv,
                official_root=args.official_root,
                gnn_checkpoint=args.gnn_checkpoint,
                protected_gpu0_process=args.protected_gpu0_process,
                protected_gpu3_process=args.protected_gpu3_process,
            )
        else:
            unblock_deferred_signals_for_science_child()
            result = run_raw_round(
                source_manifest=args.source_manifest,
                native_train_csv=args.native_train_csv,
                official_root=args.official_root,
                gnn_checkpoint=args.gnn_checkpoint,
                output_dir=args.output_dir,
                expected_parent_count=args.expected_parent_count,
                seed=args.seed,
                min_freq=args.min_freq,
                epochs=args.epochs,
                top_k_native=args.top_k_native,
                device=args.device,
                resume=not args.no_resume,
                gspan_exact_top_k_pruning=args.gspan_exact_top_k_pruning,
            )
    except (BACEGlobalGCEK20Error, OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "signal_sent": False,
                    "auto_terminate_uncontrolled_children": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 75
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "raw-round" and result.get("status") == "EXPECTED_SHORTFALL":
        return RAW_SHORTFALL_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
