#!/usr/bin/env python3
"""Precompute a calibration-only parent x action WNode matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.molclr_node_embeddings import DEFAULT_NODE_EMB_CACHE_DIR  # noqa: E402
from src.chem.hard_deletion import (  # noqa: E402
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
    CONNECTED_WNODE_CACHE_NAMESPACE,
)
from src.eval.node_wasserstein_distance import (  # noqa: E402
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.eval.wnode_action_matrix import (  # noqa: E402
    DELETION_IMPLEMENTATION_VERSION,
    LEGACY_MATCH_SELECTION_POLICY,
    MatrixBuildConfig,
    build_bace_action_matrix,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=("BACE",), required=True)
    parser.add_argument("--split", choices=("calibration",), required=True)
    parser.add_argument("--parent-csv", required=True)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-root", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--wnode-cache-db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--expected-pool-rows", type=int, default=0)
    parser.add_argument("--expected-source-parent-count", type=int, default=0)
    parser.add_argument("--expected-source-eligible-rows", type=int, default=0)
    parser.add_argument("--expected-unique-candidates", type=int, default=0)
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--cf-mode", choices=("strict_flip",), default="strict_flip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--action-semantics-version",
        choices=(DELETION_IMPLEMENTATION_VERSION, CONNECTED_ACTION_SEMANTICS),
        default=DELETION_IMPLEMENTATION_VERSION,
    )
    parser.add_argument(
        "--match-selection-policy",
        choices=(LEGACY_MATCH_SELECTION_POLICY, CONNECTED_MATCH_SELECTION_POLICY),
        default=LEGACY_MATCH_SELECTION_POLICY,
    )
    return parser


def _existing(path_like: str, label: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split != "calibration":
        raise ValueError("Action selection may only precompute the calibration split.")
    if args.action_semantics_version == CONNECTED_ACTION_SEMANTICS and (
        args.match_selection_policy != CONNECTED_MATCH_SELECTION_POLICY
    ):
        raise ValueError("Connected residual semantics require the connected match policy.")
    parent_csv = _existing(args.parent_csv, "parent CSV")
    if "test" in {part.lower() for part in parent_csv.parts}:
        raise ValueError(f"Test parent input is forbidden: {parent_csv}")
    candidate_pool = _existing(args.candidate_pool, "candidate pool")
    teacher_path = _existing(args.teacher_path, "teacher")
    molclr_root = _existing(args.molclr_root, "MolCLR root")
    molclr_checkpoint = _existing(args.molclr_checkpoint, "MolCLR checkpoint")
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_db = Path(args.wnode_cache_db).expanduser().resolve()
    cache_db.parent.mkdir(parents=True, exist_ok=True)

    teacher = TeacherSemanticScorer(teacher_path)
    if not teacher.available:
        raise RuntimeError(f"BACE teacher unavailable: {teacher.availability_reason}")
    provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=molclr_root,
            molclr_ckpt=molclr_checkpoint,
            cache_db=cache_db,
            node_emb_cache_dir=DEFAULT_NODE_EMB_CACHE_DIR,
            feature_cost="cosine",
            node_mass="uniform",
            size_penalty_beta=0.0,
            device=str(args.device),
            encoder_type="gin",
            distance_namespace=(
                CONNECTED_WNODE_CACHE_NAMESPACE
                if args.action_semantics_version == CONNECTED_ACTION_SEMANTICS
                else "molclr_node_wasserstein_v1"
            ),
        )
    )
    try:
        summary = build_bace_action_matrix(
            candidate_pool=candidate_pool,
            calibration_csv=parent_csv,
            output_dir=output_dir,
            teacher_path=teacher_path,
            molclr_root=molclr_root,
            molclr_checkpoint=molclr_checkpoint,
            wnode_cache_db=cache_db,
            teacher=teacher,
            distance_provider=provider,
            config=MatrixBuildConfig(
                id_col="molecule_id",
                smiles_col="smiles",
                label_col="label",
                cohort_name="calibration",
                parent_limit=int(args.parent_limit),
                candidate_limit=int(args.candidate_limit),
                expected_parent_count=int(args.expected_parent_count),
                expected_pool_rows=int(args.expected_pool_rows),
                expected_source_parent_count=int(args.expected_source_parent_count),
                expected_source_eligible_rows=int(args.expected_source_eligible_rows),
                expected_unique_candidates=int(args.expected_unique_candidates),
                flush_every=int(args.flush_every),
                resume=bool(args.resume),
                local_files_only=True,
                wnode_size_penalty_beta=0.0,
                action_semantics_version=str(args.action_semantics_version),
                match_selection_policy=str(args.match_selection_policy),
            ),
        )
    finally:
        provider.close()
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BACE_WNODE_ACTION_MATRIX_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
