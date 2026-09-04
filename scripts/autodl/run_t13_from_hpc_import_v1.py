#!/usr/bin/env python3
"""Run AutoDL-only T13 from a deeply verified HPC train-side mining import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tastemolnet_globalgce_full import (  # noqa: E402
    PASS_MARKER,
    TasteGlobalGCEFullConfig,
    TasteGlobalGCEFullError,
    load_input_authority,
    run_t13_full,
)
from src.utils.t8_hpc_t13_successor_v1 import (  # noqa: E402
    publish_verified_t13_locator,
    validate_spec_set,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hpc-import-root", type=Path)
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--molclr-root", type=Path)
    parser.add_argument("--molclr-checkpoint", type=Path)
    parser.add_argument("--wnode-cache-db", type=Path)
    parser.add_argument("--node-embedding-cache-dir", type=Path)
    parser.add_argument("--threshold-contract", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top-k-native", type=int, default=20)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--generation-chunk-size", type=int, default=32)
    parser.add_argument("--oracle-batch-size", type=int, default=256)
    parser.add_argument("--gspan-flush-every", type=int, default=256)
    parser.add_argument("--gspan-max-in-memory-candidates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name.replace("_", "-") for name in names if getattr(args, name) is None]
    if missing:
        raise TasteGlobalGCEFullError(
            "T13 HPC successor lacks required arguments: " + ", ".join(missing)
        )


def run(args: argparse.Namespace) -> int:
    if not args.config.is_file():
        raise TasteGlobalGCEFullError(f"config does not exist: {args.config}")
    if args.set != ["inference.fallback_to_heuristic=false"]:
        raise TasteGlobalGCEFullError(
            "T13 requires exactly --set inference.fallback_to_heuristic=false"
        )
    if args.verify_only:
        if args.spec_root is None:
            raise TasteGlobalGCEFullError("--verify-only requires --spec-root")
        locator = publish_verified_t13_locator(
            spec_root=args.spec_root, terminal_root=args.output_dir
        )
        print(json.dumps(locator, sort_keys=True), flush=True)
        print(PASS_MARKER, flush=True)
        return 0
    if args.spec_root is None:
        raise TasteGlobalGCEFullError("T13 HPC science requires --spec-root")
    _require(
        args,
        (
            "hpc_import_root",
            "gnn_checkpoint",
            "train_csv",
            "calibration_csv",
            "test_csv",
            "official_root",
            "molclr_root",
            "molclr_checkpoint",
            "wnode_cache_db",
            "node_embedding_cache_dir",
            "threshold_contract",
        ),
    )
    specs = validate_spec_set(args.spec_root, check_files=True)
    t13_spec = specs["t13"]
    expected_paths = dict(t13_spec["input_paths"])
    observed_paths = {
        name: str(Path(getattr(args, name)).expanduser().resolve(strict=False))
        for name in expected_paths
        if getattr(args, name) is not None
    }
    if (
        str(args.output_dir.expanduser().resolve(strict=False))
        != t13_spec.get("output_root")
        or str(args.hpc_import_root.expanduser().resolve(strict=False))
        != t13_spec.get("required_import_root")
        or observed_paths != expected_paths
        or args.device != "cuda:0"
        or args.epochs != 100
        or args.top_k_native != 20
        or args.min_freq != 2
        or args.learning_rate != 0.1
        or args.dropout != 0.5
        or args.generation_chunk_size != 32
        or args.oracle_batch_size != 256
        or args.gspan_flush_every != 256
        or args.gspan_max_in_memory_candidates != 256
        or args.seed != 7
    ):
        raise TasteGlobalGCEFullError("T13 CLI differs from sealed successor spec")
    config = TasteGlobalGCEFullConfig(
        epochs=args.epochs,
        top_k_native=args.top_k_native,
        min_freq=args.min_freq,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        generation_chunk_size=args.generation_chunk_size,
        oracle_batch_size=args.oracle_batch_size,
        gspan_flush_every=args.gspan_flush_every,
        gspan_max_in_memory_candidates=args.gspan_max_in_memory_candidates,
        seed=args.seed,
    )
    authority = load_input_authority(
        train_csv=args.train_csv,
        calibration_csv=args.calibration_csv,
        test_csv=args.test_csv,
        gnn_checkpoint=args.gnn_checkpoint,
        official_root=args.official_root,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        t8_pass_root=None,
        hpc_import_root=args.hpc_import_root,
        threshold_contract=args.threshold_contract,
    )
    manifest = run_t13_full(
        authority=authority,
        output_dir=args.output_dir,
        config=config,
        resume=args.resume,
        device=args.device,
        wnode_cache_db=args.wnode_cache_db,
        node_embedding_cache_dir=args.node_embedding_cache_dir,
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)
    print("[TASTE_T13_FROM_HPC_GLOBALGCE_FULL_SEALED]", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
