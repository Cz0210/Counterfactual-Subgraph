#!/usr/bin/env python3
"""Compare official TU identities with project AIDS/HIV and Mutagenicity."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.audit import official_dataset_audit  # noqa: E402
from src.baselines.comrecgc.contracts import sha256_file, write_json  # noqa: E402
from src.baselines.comrecgc.project_dataset import (  # noqa: E402
    load_aids_generation_bundle,
    load_mutagenicity_generation_bundle,
    verify_evaluation_parent_ids,
)
from src.baselines.comrecgc.upstream import imported_upstream  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--upstream-root", default="external/COMRECGC")
    parser.add_argument("--aids-dataset-dir", required=True)
    parser.add_argument("--aids-source-csv", required=True)
    parser.add_argument("--aids-eval-parent-ids", required=True)
    parser.add_argument("--mut-dataset-dir", required=True)
    parser.add_argument("--mut-eval-parent-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project = Path(args.project_root).expanduser().resolve()
    upstream = Path(args.upstream_root)
    if not upstream.is_absolute():
        upstream = project / upstream
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(upstream)
        with imported_upstream(upstream) as modules:
            official_aids = modules["data"].load_dataset("aids")
            official_mut = modules["data"].load_dataset("mutagenicity")
            official = {
                "aids": official_dataset_audit("AIDS", list(official_aids)),
                "mutagenicity": official_dataset_audit(
                    "Mutagenicity", list(official_mut)
                ),
            }
    finally:
        os.chdir(old_cwd)
    project_aids = load_aids_generation_bundle(
        dataset_dir=args.aids_dataset_dir,
        source_csv=args.aids_source_csv,
        parent_limit=1283,
    )
    project_mut = load_mutagenicity_generation_bundle(
        dataset_dir=args.mut_dataset_dir,
        parent_limit=1448,
    )
    aids_eval = verify_evaluation_parent_ids(
        args.aids_eval_parent_ids,
        expected_count=1283,
        id_field="parent_id",
    )
    mut_eval = verify_evaluation_parent_ids(
        args.mut_eval_parent_csv,
        expected_count=217,
        id_field="molecule_id",
    )
    try:
        import torch
        import torch_geometric
        import numpy
        import networkx

        runtime = {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_geometric_version": torch_geometric.__version__,
            "numpy_version": numpy.__version__,
            "networkx_version": networkx.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover
        runtime = {"environment_probe_error": str(exc)}
    payload = {
        "schema_version": 1,
        "audit_time": datetime.now(timezone.utc).isoformat(),
        "official_aids_source": official["aids"],
        "project_aids_source": project_aids.audit(),
        "official_mutagenicity_source": official["mutagenicity"],
        "project_mutagenicity_source": project_mut.audit(),
        "evaluation_parent_universes": {
            "aids": aids_eval,
            "mutagenicity": mut_eval,
        },
        "runtime": runtime,
        "conclusions": {
            "official_tu_aids_is_project_aids": False,
            "official_tu_mutagenicity_is_project_mutagenicity": False,
            "native_runs_eligible_for_project_figures": False,
            "project_adapters_required": True,
            "calibration_loaded": False,
            "test_loaded_for_generation": False,
        },
        "run_complete": True,
    }
    destination = output / "dataset_identity_audit.json"
    write_json(destination, payload)
    text = [
        "COMRECGC dataset identity audit",
        "",
        f"official AIDS: TU/AIDs graphs={official['aids']['num_graphs']}",
        f"project AIDS: HIV.csv generation graphs={len(project_aids.graphs)}",
        f"official Mutagenicity: TU graphs={official['mutagenicity']['num_graphs']}",
        f"project Mutagenicity: strict train-source graphs={len(project_mut.graphs)}",
        "native TU results are not eligible for project figures: true",
        "calibration/test loaded for generation: false",
        f"audit_sha256={sha256_file(destination)}",
        "[COMRECGC_DATASET_IDENTITY_PASS]",
    ]
    (output / "dataset_identity_audit.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
