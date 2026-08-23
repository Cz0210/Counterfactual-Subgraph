#!/usr/bin/env python3
"""Build a fresh GCF-only BACE equivalence sidecar controller manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.autodl_bace_equivalence_sidecar import (  # noqa: E402
    ProtectedRun,
    build_sidecar_manifest,
)


DEFAULT_PROTECTED_RUNS = (
    ProtectedRun(
        "four_methods_four_datasets_continuation_v1-"
        "bace_gcfexplainer_train_vrrw-main-a0",
        "legacy_bace_gcf_full",
    ),
    ProtectedRun(
        "bace_globalgce_frozen_gine_v5_cc941fc-"
        "bace_globalgce_train_candidates-main-a0",
        "bace_globalgce_v5_full",
    ),
    ProtectedRun(
        "bace-comrecgc-accel-m500-a261937-20260822T185900Z",
        "existing_bace_comrecgc_m500_pair",
    ),
    ProtectedRun(
        "four_methods_four_datasets_repair_v1-"
        "bace_comrecgc_train_generation-main-a0",
        "legacy_bace_comrecgc_full",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--controller-id", required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--build-audit", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--gcf-official-root", type=Path, required=True)
    parser.add_argument("--gine-checkpoint", type=Path, required=True)
    parser.add_argument("--neurosed-checkpoint", type=Path, required=True)
    parser.add_argument("--neurosed-manifest", type=Path, required=True)
    parser.add_argument(
        "--comrec-run-id",
        default="bace-comrecgc-accel-m500-a261937-20260822T185900Z",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_sidecar_manifest(
        controller_id=args.controller_id,
        project_root=args.project_root,
        runtime_root=args.runtime_root,
        python=args.python,
        output_root=args.output_root,
        output_manifest=args.output_manifest,
        build_audit=args.build_audit,
        protected_runs=DEFAULT_PROTECTED_RUNS,
        comrec_run_id=args.comrec_run_id,
        dataset_dir=args.dataset_dir,
        gcf_official_root=args.gcf_official_root,
        gine_checkpoint=args.gine_checkpoint,
        neurosed_checkpoint=args.neurosed_checkpoint,
        neurosed_manifest=args.neurosed_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print("[BACE_EQUIVALENCE_SIDECAR_MANIFEST_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
