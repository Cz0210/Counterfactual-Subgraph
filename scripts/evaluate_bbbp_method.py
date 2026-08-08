#!/usr/bin/env python3
"""Run shared WNode evaluation and freeze one BBBP method's paper artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bbbp_paper_artifacts import (  # noqa: E402
    METHODS,
    QUANTILES,
    export_bbbp_method_artifacts,
    freeze_bbbp_thresholds,
    load_bbbp_thresholds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--teacher-path", default="outputs/hpc/oracle/bbbp/bbbp_teacher.pkl")
    parser.add_argument("--molclr-root", default="pretrained_models/MolCLR")
    parser.add_argument(
        "--molclr-checkpoint",
        default="pretrained_models/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth",
    )
    parser.add_argument("--calibration-csv")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-test-parents", type=int, required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--split-leakage-audit", required=True)
    parser.add_argument("--candidate-lineage-audit", required=True)
    parser.add_argument("--calibrate-thresholds", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _complete(path: Path) -> bool:
    return (path / "_RUN_COMPLETE.json").is_file()


def _evaluator_argv(
    *,
    args: argparse.Namespace,
    dataset_csv: str | Path,
    output_dir: Path,
    thresholds: str,
) -> list[str]:
    spec = METHODS[args.method]
    argv = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/evaluate_ccrcov_with_molclr_node_wasserstein.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset-csv",
        str(Path(dataset_csv).expanduser().resolve()),
        "--teacher-path",
        str(Path(args.teacher_path).expanduser().resolve()),
        "--molclr-root",
        str(Path(args.molclr_root).expanduser().resolve()),
        "--molclr-checkpoint",
        str(Path(args.molclr_checkpoint).expanduser().resolve()),
        "--label",
        "1",
        "--smiles-col",
        "smiles",
        "--label-col",
        "label",
        "--cf-mode",
        "strict_flip",
        "--output-dir",
        str(output_dir),
        "--max-parents",
        "0",
        "--max-candidates",
        "20",
        "--wnode-thresholds",
        thresholds,
        "--wnode-quantiles",
        ",".join(format(value, ".17g") for value in QUANTILES),
        "--feature-cost",
        "cosine",
        "--node-mass",
        "uniform",
        "--size-penalty-beta",
        "0.0",
        "--device",
        "cuda",
        "--preselected-topk",
        "20",
        "--require-preselected-topk",
        "1",
        "--selection-method",
        str(spec["selection_method"]),
        "--skip-redundancy",
        "1",
        "--resume",
        "1" if args.resume else "0",
    ]
    if spec["candidate_kind"] == "ours":
        argv.extend(
            [
                "--run-ours",
                "1",
                "--run-fullgraph",
                "0",
                "--ours-selected-path",
                str(Path(args.candidate_path).expanduser().resolve()),
            ]
        )
    else:
        argv.extend(
            [
                "--run-ours",
                "0",
                "--run-fullgraph",
                "1",
                "--fullgraph-candidates-path",
                str(Path(args.candidate_path).expanduser().resolve()),
                "--fullgraph-method-name",
                str(spec["display"]),
            ]
        )
    return argv


def _run(argv: list[str], output_dir: Path, *, resume: bool) -> None:
    if _complete(output_dir):
        if not resume:
            raise FileExistsError(f"BBBP evaluation already completed: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"BBBP evaluation work directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(argv, cwd=PROJECT_ROOT, check=True)
    if not _complete(output_dir):
        raise RuntimeError(f"BBBP evaluator did not write completion marker: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required_inputs = (
        args.candidate_path,
        args.teacher_path,
        args.molclr_checkpoint,
        args.test_csv,
        args.thresholds_json,
        args.protocol_manifest,
        args.split_manifest,
        args.split_leakage_audit,
        args.candidate_lineage_audit,
    )
    missing = [str(Path(value).expanduser()) for value in required_inputs if not Path(value).expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"BBBP evaluation inputs are missing: {missing}")
    if args.validate_only or args.dry_run:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_NOT_RUN",
                    "dataset": "BBBP",
                    "method": args.method,
                    "cf_mode": "strict_flip",
                    "distance_line": "MolCLR-Node-Wasserstein",
                    "selection_performed_in_eval": False,
                    "threshold_fitted_on_test": False,
                    "planned_output_dir": str(Path(args.output_dir).expanduser()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    work = Path(args.work_dir).expanduser().resolve()
    test_run = work / "test"
    threshold_path = Path(args.thresholds_json).expanduser().resolve()
    if args.calibrate_thresholds:
        if args.method != "ours":
            raise ValueError("Only Ours may freeze the shared BBBP calibration thresholds.")
        if not args.calibration_csv:
            raise ValueError("--calibrate-thresholds requires --calibration-csv.")
        calibration_run = work / "calibration"
        calibration_argv = _evaluator_argv(
            args=args,
            dataset_csv=args.calibration_csv,
            output_dir=calibration_run,
            thresholds="auto_quantile",
        )
        _run(calibration_argv, calibration_run, resume=args.resume)
        freeze_bbbp_thresholds(
            calibration_run_dir=calibration_run,
            output_path=threshold_path,
            calibration_parent_csv=args.calibration_csv,
        )
    contract = load_bbbp_thresholds(threshold_path)
    explicit = ",".join(format(float(value), ".17g") for value in contract["thresholds"])
    test_argv = _evaluator_argv(
        args=args,
        dataset_csv=args.test_csv,
        output_dir=test_run,
        thresholds=explicit,
    )
    _run(test_argv, test_run, resume=args.resume)
    summary = export_bbbp_method_artifacts(
        method=args.method,
        test_run_dir=test_run,
        thresholds_json=threshold_path,
        output_dir=args.output_dir,
        expected_parent_count=args.expected_test_parents,
        expected_top_k=20,
        protocol_manifest=args.protocol_manifest,
        split_manifest=args.split_manifest,
        split_leakage_audit=args.split_leakage_audit,
        candidate_lineage_audit=args.candidate_lineage_audit,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("[BBBP_METHOD_PAPER_ARTIFACTS_OK]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
