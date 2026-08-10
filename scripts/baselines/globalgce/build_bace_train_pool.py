#!/usr/bin/env python3
"""Build BACE GlobalGCE candidates with the established train-pool runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.globalgce_bace_adapter import (  # noqa: E402
    EXPECTED_TRAIN_SOURCE_COUNT,
    OfficialGlobalGCEBACEGenerator,
    build_bace_train_pool,
)
from src.baselines.globalgce_mutagenicity_adapter import PoolBuildConfig  # noqa: E402
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--native-train-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--expected-parent-count", type=int, default=EXPECTED_TRAIN_SOURCE_COUNT)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--top-k-native", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-chunk-size", type=int, default=32)
    parser.add_argument("--generation-num-workers", type=int, default=0)
    parser.add_argument("--memory-log-every-chunks", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _file(path_like: str, description: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_csv = _file(args.train_csv, "BACE strict train-source CSV")
    native_train_csv = _file(args.native_train_csv, "BACE two-class train CSV")
    teacher_path = _file(args.teacher_path, "BACE RF teacher")
    official_root = Path(args.official_root).expanduser().resolve()
    if not official_root.is_dir():
        raise FileNotFoundError(official_root)
    teacher = TeacherSemanticScorer(teacher_path)
    if not teacher.available:
        raise RuntimeError(f"BACE RF teacher is unavailable: {teacher.availability_reason}")
    generator = OfficialGlobalGCEBACEGenerator(
        official_root,
        native_train_csv=native_train_csv,
    )
    summary = build_bace_train_pool(
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
            forbid_calibration_test=True,
            generation_chunk_size=int(args.generation_chunk_size),
            generation_num_workers=int(args.generation_num_workers),
            memory_log_every_chunks=int(args.memory_log_every_chunks),
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("[BACE_GLOBALGCE_TRAIN_POOL_BUILD_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
