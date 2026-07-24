#!/usr/bin/env python3
"""Build independent strict-first Mutagenicity SFT/PPO v2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity_sft_ppo import (  # noqa: E402
    DEFAULT_EXPECTED_COUNTS,
    MutagenicitySFTPPOConfig,
)
from src.data.mutagenicity_sft_v2 import (  # noqa: E402
    build_mutagenicity_sft_ppo_v2,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hpc.yaml"))
    parser.add_argument(
        "--teacher-consistent-root",
        type=Path,
        default=Path("outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"),
    )
    parser.add_argument(
        "--teacher-path",
        type=Path,
        default=Path(
            "outputs/hpc/oracle/final/mutagenicity_rf_v1/"
            "mutagenicity_rf_model.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hpc/mutagenicity/sft_ppo_data_v2"),
    )
    parser.add_argument("--min-atom-ratio", type=float, default=0.10)
    parser.add_argument("--max-atom-ratio", type=float, default=0.55)
    parser.add_argument("--min-frag-atoms", type=int, default=3)
    parser.add_argument("--max-frag-atoms", type=int, default=30)
    parser.add_argument("--max-candidates-per-parent", type=int, default=160)
    parser.add_argument("--max-targets-per-parent", type=int, default=3)
    parser.add_argument("--max-completion-frequency", type=int, default=0)
    parser.add_argument("--expected-train", type=int, default=1448)
    parser.add_argument("--expected-val", type=int, default=260)
    parser.add_argument("--expected-calibration", type=int, default=235)
    parser.add_argument("--expected-test", type=int, default=217)
    return parser


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _resolve(args.teacher_consistent_root)
    output_dir = _resolve(args.output_dir)
    teacher_path = _resolve(args.teacher_path)
    expected = {
        "train": int(args.expected_train),
        "val": int(args.expected_val),
        "calibration": int(args.expected_calibration),
        "test": int(args.expected_test),
    }
    config = MutagenicitySFTPPOConfig(
        min_atom_ratio=float(args.min_atom_ratio),
        max_atom_ratio=float(args.max_atom_ratio),
        min_frag_atoms=int(args.min_frag_atoms),
        max_frag_atoms=int(args.max_frag_atoms),
        max_candidates_per_parent=int(args.max_candidates_per_parent),
        use_teacher_ranking=True,
    )
    print("[MUTAGENICITY_SFT_STRICT_V2_CONFIG]")
    print(f"teacher_consistent_root={root}")
    print(f"teacher_path={teacher_path}")
    print(f"output_dir={output_dir}")
    print(f"expected_counts={expected}")
    print(f"config={config}")
    summary = build_mutagenicity_sft_ppo_v2(
        train_input=root / "train_source_label1_teacher_correct.csv",
        val_input=root / "val_source_label1_teacher_correct.csv",
        calibration_exclusion_input=(
            root / "calibration_source_label1_teacher_correct.csv"
        ),
        test_exclusion_input=root / "test_source_label1_teacher_correct.csv",
        teacher_path=teacher_path,
        output_dir=output_dir,
        config=config,
        expected_counts=expected,
        max_targets_per_parent=int(args.max_targets_per_parent),
        max_completion_frequency=(
            int(args.max_completion_frequency)
            if int(args.max_completion_frequency) > 0
            else None
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
