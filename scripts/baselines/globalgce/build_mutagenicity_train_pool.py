#!/usr/bin/env python3
"""Build a train-only native GlobalGCE pool for strict Mutagenicity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.globalgce_mutagenicity_adapter import (  # noqa: E402
    DEFAULT_EXPECTED_PARENT_COUNT,
    OfficialGlobalGCEMutagenicityGenerator,
    PoolBuildConfig,
    build_mutagenicity_train_pool,
    load_strict_train_parents,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top-k-native", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-chunk-size", type=int, default=32)
    parser.add_argument("--generation-num-workers", type=int, default=0)
    parser.add_argument("--memory-log-every-chunks", type=int, default=1)
    parser.add_argument(
        "--codec-probe-only",
        action="store_true",
        help="Run only the deterministic source graph round-trip gate.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--forbid-calibration-test",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--native-train-csv",
        default=None,
        help=(
            "Current processed train.csv containing both labels, used only to "
            "train GlobalGCE's required native GNN."
        ),
    )
    parser.add_argument(
        "--expected-parent-count",
        type=int,
        default=DEFAULT_EXPECTED_PARENT_COUNT,
    )
    parser.add_argument("--config", default=None, help="HPC compatibility config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Ignored compatibility override; generation has an explicit CLI.",
    )
    return parser


def _required_file(path_like: str, description: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_csv = _required_file(args.train_csv, "strict train CSV")
    official_root = Path(args.official_root).expanduser().resolve()
    if not official_root.is_dir():
        raise FileNotFoundError(f"GlobalGCE official root is missing: {official_root}")
    native_train_csv = (
        _required_file(args.native_train_csv, "native GNN train CSV")
        if args.native_train_csv
        else None
    )

    generator = OfficialGlobalGCEMutagenicityGenerator(
        official_root,
        native_train_csv=native_train_csv,
    )
    if args.codec_probe_only:
        _all_parents, selected_parents = load_strict_train_parents(
            train_csv,
            parent_limit=int(args.parent_limit),
            expected_parent_count=int(args.expected_parent_count),
            forbid_calibration_test=bool(args.forbid_calibration_test),
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = generator.probe_codec(
            selected_parents,
            seed=int(args.seed),
            output_path=output_dir / "codec_probe_summary.json",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("[MUTAGENICITY_GLOBALGCE_CODEC_PROBE_OK]")
        return 0

    teacher_path = _required_file(args.teacher_path, "Mutagenicity RF teacher")
    teacher = TeacherSemanticScorer(teacher_path)
    if not teacher.available:
        raise RuntimeError(
            "Mutagenicity RF teacher is unavailable: "
            f"{teacher.availability_reason}"
        )
    summary = build_mutagenicity_train_pool(
        train_csv=train_csv,
        teacher_path=teacher_path,
        official_root=official_root,
        output_dir=args.output_dir,
        teacher=teacher,
        generator=generator,
        config=PoolBuildConfig(
            parent_limit=int(args.parent_limit),
            expected_parent_count=int(args.expected_parent_count),
            seed=int(args.seed),
            epochs=int(args.epochs),
            top_k_native=int(args.top_k_native),
            learning_rate=float(args.learning_rate),
            dropout=float(args.dropout),
            device=str(args.device),
            resume=bool(args.resume),
            forbid_calibration_test=bool(args.forbid_calibration_test),
            generation_chunk_size=int(args.generation_chunk_size),
            generation_num_workers=int(args.generation_num_workers),
            memory_log_every_chunks=int(args.memory_log_every_chunks),
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[MUTAGENICITY_GLOBALGCE_TRAIN_POOL_BUILD_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
