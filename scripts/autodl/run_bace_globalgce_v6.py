#!/usr/bin/env python3
"""Build and execute the fail-closed BACE GlobalGCE v6 control-plane stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.autodl.build_four_by_four_manifest import compose_manifest  # noqa: E402
from src.baselines.bace_gnn_baseline_generic_adapter import (  # noqa: E402
    atomic_write_generic_fragment,
    build_bace_baseline_generic_controller_fragment,
)
from src.baselines.globalgce_mining_adoption import (  # noqa: E402
    EXPECTED_OFFICIAL_COMMIT,
    GlobalGCEMiningAdoptionError,
    build_globalgce_gspan_adoption,
)
from src.eval.bace_frozen_gnn_contracts import (  # noqa: E402
    atomic_json,
    atomic_marker,
    fresh_output_dir,
    utc_now,
)


DECISION_SCHEMA = "bace_globalgce_v6_mining_decision_v1"
DECISION_TASK_ID = "bace_globalgce_mining_decision"


def _adoption_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--source-task-state", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sqlite", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--native-train-csv", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--gine-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-official-commit", default=EXPECTED_OFFICIAL_COMMIT
    )
    parser.add_argument("--expected-pattern-count", type=int, default=5_441_858)
    parser.add_argument("--expected-root-count", type=int, default=19)
    parser.add_argument("--min-freq", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="stage", required=True)

    decision = sub.add_parser("mining-decision")
    _adoption_arguments(decision)
    decision.add_argument("--output-dir", type=Path, required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--controller-id", required=True)
    build.add_argument("--python", type=Path, required=True)
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--runtime-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--fragment-output", type=Path, required=True)
    build.add_argument("--manifest-output", type=Path, required=True)
    build.add_argument("--dataset-dir", type=Path, required=True)
    build.add_argument("--calibration-split", type=Path, required=True)
    build.add_argument("--test-split", type=Path, required=True)
    build.add_argument("--molclr-root", type=Path, required=True)
    build.add_argument("--molclr-checkpoint", type=Path, required=True)
    build.add_argument("--neurosed-checkpoint", type=Path, required=True)
    _adoption_arguments(build)
    return parser


def _mining_decision(args: argparse.Namespace) -> dict[str, Any]:
    output = fresh_output_dir(args.output_dir)
    adoption_root = output / "adoption"
    try:
        identity = build_globalgce_gspan_adoption(
            source_run_manifest=args.source_run_manifest,
            source_task_state=args.source_task_state,
            source_checkpoint=args.source_checkpoint,
            source_sqlite=args.source_sqlite,
            official_root=args.official_root,
            native_train_csv=args.native_train_csv,
            source_manifest=args.source_manifest,
            gine_checkpoint=args.gine_checkpoint,
            output_dir=adoption_root,
            expected_official_commit=args.expected_official_commit,
            expected_pattern_count=args.expected_pattern_count,
            expected_root_count=args.expected_root_count,
            min_freq=args.min_freq,
            top_k=args.top_k,
            seed=args.seed,
        )
        route = "adopt_v5_exhaustive"
        reason = None
        adoption_proof = str((adoption_root / "adoption_proof.json").resolve())
    except (
        GlobalGCEMiningAdoptionError,
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as exc:
        identity = None
        route = "fresh_exact_top_k_v2"
        reason = f"{type(exc).__name__}:{exc}"
        adoption_proof = None
        atomic_marker(output / "FRESH_REMINE_REQUIRED", reason)
    decision = {
        "schema_version": DECISION_SCHEMA,
        "status": "PASS",
        "route": route,
        "adoption_proof": adoption_proof,
        "adoption_identity": identity,
        "fresh_remine_fallback": route == "fresh_exact_top_k_v2",
        "fallback_reason": reason,
        "fallback_semantics": (
            "fresh_exact_stable_topk_v2_no_v5_bytes_consumed"
            if route == "fresh_exact_top_k_v2"
            else None
        ),
        "calibration_loaded": False,
        "test_loaded": False,
        "created_at": utc_now(),
    }
    atomic_json(output / "decision.json", decision)
    atomic_marker(output / "DECISION_COMPLETE", route)
    atomic_marker(output / "PASS", "PASS")
    return decision


def _decision_task(args: argparse.Namespace, script: str) -> dict[str, Any]:
    output = f"{args.output_root.resolve(strict=False)}/mining_decision/attempt-{{attempt}}"
    command = [
        str(args.python.resolve(strict=False)),
        script,
        "mining-decision",
        "--source-run-manifest",
        str(args.source_run_manifest.resolve(strict=False)),
        "--source-task-state",
        str(args.source_task_state.resolve(strict=False)),
        "--source-checkpoint",
        str(args.source_checkpoint.resolve(strict=False)),
        "--source-sqlite",
        str(args.source_sqlite.resolve(strict=False)),
        "--official-root",
        str(args.official_root.resolve(strict=False)),
        "--native-train-csv",
        str(args.native_train_csv.resolve(strict=False)),
        "--source-manifest",
        str(args.source_manifest.resolve(strict=False)),
        "--gine-checkpoint",
        str(args.gine_checkpoint.resolve(strict=False)),
        "--output-dir",
        "{task_output}",
        "--expected-official-commit",
        str(args.expected_official_commit),
        "--expected-pattern-count",
        str(args.expected_pattern_count),
        "--expected-root-count",
        str(args.expected_root_count),
        "--min-freq",
        str(args.min_freq),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
    ]
    return {
        "id": DECISION_TASK_ID,
        "dataset": "bace",
        "stage": "BACE_GLOBALGCE_GSPAN_MINING_DECISION",
        "runner_dataset": "bace-baseline-globalgce",
        "runner_stage": "BACE_GLOBALGCE_GSPAN_MINING_DECISION",
        "depends_on": [],
        "resource": "cpu",
        "priority": 60,
        "enabled": True,
        "data_splits": ["train"],
        "manifest_only": False,
        "command": command,
        "input_manifest": str(args.source_run_manifest.resolve(strict=False)),
        "expected_output": output,
        "required_output_files": ["decision.json", "DECISION_COMPLETE", "PASS"],
        "required_log_marker": "[BACE_GLOBALGCE_V6_MINING_DECISION_PASS]",
        "environment": {
            "PYTHONPATH": "{project_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUN_TASTEMOLNET": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }


def _build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve(strict=False)
    fragment = build_bace_baseline_generic_controller_fragment(
        method="GlobalGCE",
        python=args.python,
        project_root=project,
        output_root=args.output_root,
        gnn_checkpoint=args.gine_checkpoint,
        dataset_dir=args.dataset_dir,
        calibration_split=args.calibration_split,
        test_split=args.test_split,
        molclr_root=args.molclr_root,
        molclr_checkpoint=args.molclr_checkpoint,
        neurosed_checkpoint=args.neurosed_checkpoint,
        official_root=args.official_root,
        globalgce_source_manifest=args.source_manifest,
        globalgce_native_train_csv=args.native_train_csv,
    )
    tasks = list(fragment["tasks"])
    tasks.insert(
        0,
        _decision_task(
            args,
            str((project / "scripts/autodl/run_bace_globalgce_v6.py").resolve(strict=False)),
        ),
    )
    for task in tasks:
        if task["id"] == "bace_globalgce_bridge_smoke":
            task["resource"] = "cpu"
            task["priority"] = 61
            task["environment"]["CUDA_VISIBLE_DEVICES"] = ""
            device = task["command"].index("--device") + 1
            task["command"][device] = "cpu"
        elif task["id"] == "bace_globalgce_train_candidates":
            task["depends_on"] = [
                "bace_globalgce_bridge_smoke",
                DECISION_TASK_ID,
            ]
            task["command"].extend(
                [
                    "--gspan-mining-decision",
                    "{dep_bace_globalgce_mining_decision_output}/decision.json",
                ]
            )
            task["input_manifest"] = (
                "{dep_bace_globalgce_mining_decision_output}/decision.json"
            )
    fragment["tasks"] = tasks
    fragment["root_task_ids"] = [DECISION_TASK_ID, "bace_globalgce_preflight"]
    fragment["v6_contract"] = {
        "bridge_cpu_only": True,
        "formal_train_exclusive_gpu": True,
        "adoption_or_explicit_fresh_remine": True,
        "v5_root_writable": False,
    }
    fragment_path = atomic_write_generic_fragment(args.fragment_output, fragment)
    result = compose_manifest(
        controller_id=args.controller_id,
        fragments=[fragment_path],
        output=args.manifest_output,
    )
    result["fragment"] = str(fragment_path)
    result["v6_contract"] = fragment["v6_contract"]
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "mining-decision":
        result = _mining_decision(args)
        marker = "[BACE_GLOBALGCE_V6_MINING_DECISION_PASS]"
    else:
        result = _build_manifest(args)
        marker = "[BACE_GLOBALGCE_V6_MANIFEST_PASS]"
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(marker, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
