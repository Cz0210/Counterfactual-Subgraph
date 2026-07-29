#!/usr/bin/env python3
"""Build the Mutagenicity calibration parent-candidate WNode action matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.molclr_node_embeddings import DEFAULT_NODE_EMB_CACHE_DIR  # noqa: E402
from src.eval.mutagenicity_wnode_matrix import (  # noqa: E402
    CANDIDATE_ORDER_SOURCE_SUPPORT,
    DEFAULT_WNODE_SIZE_PENALTY_BETA,
    MatrixBuildConfig,
    build_calibration_matrix_run,
)
from src.eval.node_wasserstein_distance import (  # noqa: E402
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-root", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wnode-cache-db", required=True)
    parser.add_argument(
        "--wnode-size-penalty-beta",
        type=float,
        default=DEFAULT_WNODE_SIZE_PENALTY_BETA,
    )
    parser.add_argument("--id-col", default="molecule_id")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--cohort-name", default="calibration")
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--expected-parent-count", type=int, default=235)
    parser.add_argument(
        "--candidate-order",
        choices=(CANDIDATE_ORDER_SOURCE_SUPPORT,),
        default=CANDIDATE_ORDER_SOURCE_SUPPORT,
    )
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


def _require_local_path(path_like: str, description: str) -> Path:
    path = Path(path_like).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def main() -> int:
    args = build_parser().parse_args()
    candidate_pool = _require_local_path(args.candidate_pool, "candidate pool")
    calibration_csv = _require_local_path(args.calibration_csv, "calibration CSV")
    teacher_path = _require_local_path(args.teacher_path, "teacher")
    molclr_root = _require_local_path(args.molclr_root, "MolCLR root")
    molclr_checkpoint = _require_local_path(
        args.molclr_checkpoint,
        "MolCLR checkpoint",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_db = Path(args.wnode_cache_db).expanduser().resolve()

    existing = list(output_dir.iterdir()) if output_dir.is_dir() else []
    if existing and not bool(args.resume):
        raise FileExistsError(
            f"Output directory is non-empty and --no-resume was requested: {output_dir}"
        )
    if existing and bool(args.resume):
        for required in ("run_manifest.json", "resume_checkpoint.json"):
            if not (output_dir / required).is_file():
                raise ValueError(
                    f"Resume output is missing {required}: {output_dir}"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
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
        summary = build_calibration_matrix_run(
            candidate_pool=candidate_pool,
            calibration_csv=calibration_csv,
            output_dir=output_dir,
            teacher_path=teacher_path,
            molclr_root=molclr_root,
            molclr_checkpoint=molclr_checkpoint,
            wnode_cache_db=cache_db,
            teacher=teacher,
            distance_provider=provider,
            config=MatrixBuildConfig(
                id_col=str(args.id_col),
                smiles_col=str(args.smiles_col),
                label_col=str(args.label_col),
                cohort_name=str(args.cohort_name),
                parent_limit=int(args.parent_limit),
                candidate_limit=int(args.candidate_limit),
                expected_parent_count=int(args.expected_parent_count),
                candidate_order=str(args.candidate_order),
                flush_every=int(args.flush_every),
                resume=bool(args.resume),
                local_files_only=bool(args.local_files_only),
                wnode_size_penalty_beta=float(args.wnode_size_penalty_beta),
            ),
        )
    finally:
        provider.close()

    print(f"output_dir={output_dir}")
    print(f"parent_count={summary['parent_count']}")
    print(f"selected_candidate_count={summary['selected_candidate_count']}")
    print(f"actual_pair_rows={summary['actual_pair_rows']}")
    print(f"strict_flip_pair_count={summary['strict_flip_pair_count']}")
    print(f"wnode_size_penalty_beta={summary['wnode_size_penalty_beta']}")
    print("[MUTAGENICITY_WNODE_CALIBRATION_MATRIX_BUILD_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
