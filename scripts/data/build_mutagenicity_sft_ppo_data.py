#!/usr/bin/env python3
"""Build Mutagenicity SFT v3 targets and stable-PPO parent prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mutagenicity_sft_ppo import (  # noqa: E402
    DEFAULT_EXPECTED_COUNTS,
    MutagenicitySFTPPOConfig,
    build_mutagenicity_sft_ppo_data,
)
from src.utils.env import load_and_merge_config_files  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "mutagenicity_sft_ppo.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="YAML config path. May be repeated; later files override earlier files.",
    )
    parser.add_argument("--teacher-consistent-root", default=None)
    parser.add_argument("--train-input", default=None)
    parser.add_argument("--val-input", default=None)
    parser.add_argument("--calibration-exclusion-input", default=None)
    parser.add_argument("--test-exclusion-input", default=None)
    parser.add_argument("--teacher-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-parents", type=int, default=None)
    parser.add_argument("--max-val-parents", type=int, default=None)
    parser.add_argument("--expected-train-parents", type=int, default=None)
    parser.add_argument("--expected-val-parents", type=int, default=None)
    parser.add_argument("--expected-calibration-parents", type=int, default=None)
    parser.add_argument("--expected-test-parents", type=int, default=None)
    parser.add_argument("--min-atom-ratio", type=float, default=None)
    parser.add_argument("--max-atom-ratio", type=float, default=None)
    parser.add_argument("--min-frag-atoms", type=int, default=None)
    parser.add_argument("--max-frag-atoms", type=int, default=None)
    parser.add_argument("--max-candidates-per-parent", type=int, default=None)
    parser.add_argument(
        "--use-teacher-ranking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the fixed Mutagenicity RF teacher to rank filtered SFT v3 candidates.",
    )
    return parser


def _nested(config: Mapping[str, Any], section: str, key: str, default: Any) -> Any:
    section_value = config.get(section)
    if not isinstance(section_value, Mapping):
        return default
    value = section_value.get(key)
    return default if value is None else value


def _choice(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _optional_cap(value: Any) -> int | None:
    numeric = int(value or 0)
    return numeric if numeric > 0 else None


def _optional_expected(value: Any) -> int | None:
    numeric = int(value or 0)
    return numeric if numeric > 0 else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_paths = [Path(path).expanduser().resolve() for path in args.config]
    if not config_paths:
        config_paths = [DEFAULT_CONFIG.resolve()]
    config_payload = load_and_merge_config_files(config_paths)

    root = Path(
        _choice(
            args.teacher_consistent_root,
            _nested(config_payload, "inputs", "teacher_consistent_root", None),
            "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent",
        )
    )
    train_input = Path(
        _choice(args.train_input, None, root / "train_source_label1_teacher_correct.csv")
    )
    val_input = Path(
        _choice(args.val_input, None, root / "val_source_label1_teacher_correct.csv")
    )
    calibration_input = Path(
        _choice(
            args.calibration_exclusion_input,
            None,
            root / "calibration_source_label1_teacher_correct.csv",
        )
    )
    test_input = Path(
        _choice(args.test_exclusion_input, None, root / "test_source_label1_teacher_correct.csv")
    )
    teacher_path = Path(
        _choice(
            args.teacher_path,
            _nested(config_payload, "inputs", "teacher_path", None),
            "outputs/hpc/oracle/final/mutagenicity_rf_v1/mutagenicity_rf_model.pkl",
        )
    )
    output_dir = Path(
        _choice(
            args.output_dir,
            _nested(config_payload, "outputs", "root", None),
            "outputs/hpc/mutagenicity/sft_ppo_data_v1",
        )
    )

    build_config = MutagenicitySFTPPOConfig(
        seed=int(_choice(args.seed, config_payload.get("seed"), 42)),
        max_train_parents=_optional_cap(
            _choice(
                args.max_train_parents,
                _nested(config_payload, "sampling", "max_train_parents", 0),
                0,
            )
        ),
        max_val_parents=_optional_cap(
            _choice(
                args.max_val_parents,
                _nested(config_payload, "sampling", "max_val_parents", 0),
                0,
            )
        ),
        min_atom_ratio=float(
            _choice(
                args.min_atom_ratio,
                _nested(config_payload, "sft_target", "min_atom_ratio", None),
                0.10,
            )
        ),
        max_atom_ratio=float(
            _choice(
                args.max_atom_ratio,
                _nested(config_payload, "sft_target", "max_atom_ratio", None),
                0.55,
            )
        ),
        min_frag_atoms=int(
            _choice(
                args.min_frag_atoms,
                _nested(config_payload, "sft_target", "min_frag_atoms", None),
                3,
            )
        ),
        max_frag_atoms=int(
            _choice(
                args.max_frag_atoms,
                _nested(config_payload, "sft_target", "max_frag_atoms", None),
                30,
            )
        ),
        max_candidates_per_parent=int(
            _choice(
                args.max_candidates_per_parent,
                _nested(config_payload, "sft_target", "max_candidates_per_parent", None),
                160,
            )
        ),
        use_teacher_ranking=bool(
            _choice(
                args.use_teacher_ranking,
                _nested(config_payload, "sft_target", "use_teacher_ranking", None),
                True,
            )
        ),
    )
    expected_counts = {
        "train": _optional_expected(
            _choice(
                args.expected_train_parents,
                _nested(config_payload, "expected_parent_counts", "train", None),
                DEFAULT_EXPECTED_COUNTS["train"],
            )
        ),
        "val": _optional_expected(
            _choice(
                args.expected_val_parents,
                _nested(config_payload, "expected_parent_counts", "val", None),
                DEFAULT_EXPECTED_COUNTS["val"],
            )
        ),
        "calibration": _optional_expected(
            _choice(
                args.expected_calibration_parents,
                _nested(config_payload, "expected_parent_counts", "calibration", None),
                DEFAULT_EXPECTED_COUNTS["calibration"],
            )
        ),
        "test": _optional_expected(
            _choice(
                args.expected_test_parents,
                _nested(config_payload, "expected_parent_counts", "test", None),
                DEFAULT_EXPECTED_COUNTS["test"],
            )
        ),
    }

    print("[MUTAGENICITY_SFT_PPO_CONFIG]")
    print(f"config={[str(path) for path in config_paths]}")
    print(f"train_input={train_input}")
    print(f"val_input={val_input}")
    print(f"calibration_exclusion_input={calibration_input}")
    print(f"test_exclusion_input={test_input}")
    print(f"teacher_path={teacher_path}")
    print(f"output_dir={output_dir}")
    print("source_label=1")
    print("target_label=0")
    print(f"expected_counts={expected_counts}")
    print(f"build_config={build_config}")

    summary = build_mutagenicity_sft_ppo_data(
        train_input=train_input,
        val_input=val_input,
        calibration_exclusion_input=calibration_input,
        test_exclusion_input=test_input,
        teacher_path=teacher_path,
        output_dir=output_dir,
        config=build_config,
        expected_counts=expected_counts,
    )
    print(json.dumps(summary["splits"], indent=2, ensure_ascii=False, sort_keys=True))
    print("[MUTAGENICITY_SFT_PPO_BUILD_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
