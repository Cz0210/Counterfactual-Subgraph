#!/usr/bin/env python3
"""Evaluate the frozen Mutagenicity WNode selector once on the final test cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.molclr_node_embeddings import DEFAULT_NODE_EMB_CACHE_DIR  # noqa: E402
from src.eval.mutagenicity_wnode_frozen_test import (  # noqa: E402
    EXPECTED_TEST_PARENT_COUNT,
    FrozenTestConfig,
    build_frozen_test_run,
)
from src.eval.node_wasserstein_distance import (  # noqa: E402
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selector-root", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-root", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wnode-cache-db", required=True)
    parser.add_argument("--id-col", default="molecule_id")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--cohort-name", default="test")
    parser.add_argument("--wnode-size-penalty-beta", type=float, default=0.0)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _require_path(path_like: str, description: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def main() -> int:
    args = build_parser().parse_args()
    frozen_root = _require_path(args.frozen_selector_root, "frozen selector root")
    test_csv = _require_path(args.test_csv, "test CSV")
    teacher_path = _require_path(args.teacher_path, "Mutagenicity RF teacher")
    molclr_root = _require_path(args.molclr_root, "MolCLR root")
    molclr_checkpoint = _require_path(
        args.molclr_checkpoint,
        "MolCLR checkpoint",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_db = Path(args.wnode_cache_db).expanduser().resolve()
    cache_db.parent.mkdir(parents=True, exist_ok=True)

    teacher = TeacherSemanticScorer(teacher_path)
    if not teacher.available:
        raise RuntimeError(
            f"Mutagenicity teacher is unavailable: {teacher.availability_reason}"
        )
    provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=molclr_root,
            molclr_ckpt=molclr_checkpoint,
            cache_db=cache_db,
            node_emb_cache_dir=DEFAULT_NODE_EMB_CACHE_DIR,
            feature_cost="cosine",
            node_mass="uniform",
            size_penalty_beta=float(args.wnode_size_penalty_beta),
            device="cuda",
            encoder_type="gin",
        )
    )
    try:
        summary = build_frozen_test_run(
            frozen_selector_root=frozen_root,
            test_csv=test_csv,
            teacher_path=teacher_path,
            molclr_root=molclr_root,
            molclr_checkpoint=molclr_checkpoint,
            output_dir=output_dir,
            wnode_cache_db=cache_db,
            teacher=teacher,
            distance_provider=provider,
            config=FrozenTestConfig(
                id_col=str(args.id_col),
                smiles_col=str(args.smiles_col),
                label_col=str(args.label_col),
                cohort_name=str(args.cohort_name),
                expected_parent_count=EXPECTED_TEST_PARENT_COUNT,
                wnode_size_penalty_beta=float(args.wnode_size_penalty_beta),
                flush_every=int(args.flush_every),
                resume=bool(args.resume),
                local_files_only=bool(args.local_files_only),
            ),
        )
    finally:
        provider.close()

    print(f"output_dir={output_dir}")
    print(f"selected_variant={summary['selected_variant']}")
    print(f"test_parent_count={summary['test_parent_count']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"actual_pair_rows={summary['actual_pair_rows']}")
    print(f"theta_star={summary['theta_star']}")
    print(f"cost_cap={summary['cost_cap']}")
    print("[MUTAGENICITY_WNODE_FROZEN_TEST_BUILD_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
