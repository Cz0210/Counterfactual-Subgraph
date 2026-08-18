#!/usr/bin/env python3
"""Freeze the calibration-selected BACE GlobalGCE fullgraph sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=None, help=argparse.SUPPRESS)
    value.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    value.add_argument("--calibration-manifest", required=True)
    value.add_argument("--candidate-root", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--git-commit", required=True)
    value.add_argument("--validate-only", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    calibration_path = Path(args.calibration_manifest).expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("selection_split") != "calibration" or calibration.get("test_loaded") is not False:
        raise ValueError("GlobalGCE min_freq selection must be calibration-only.")
    value = int(calibration["selected_min_freq"])
    selected = (
        Path(args.candidate_root).expanduser().resolve()
        / "candidates"
        / f"min_freq_{value}"
        / "fullgraph_candidates.csv"
    )
    if not selected.is_file() or selected.stat().st_size <= 0:
        raise FileNotFoundError(selected)
    payload = {
        "schema_version": "bace_globalgce_frozen_selection_v7",
        "dataset": "BACE",
        "method": "GlobalGCE",
        "selection_frozen": True,
        "selection_split": "train",
        "min_freq_selection_split": "calibration",
        "selected_min_freq": value,
        "selected_sequence_sha256": _sha(selected),
        "test_used": False,
        "gcf_result_used": False,
        "selection_performed_in_eval": False,
        "threshold_fitted_on_test": False,
        "action_semantics_version": "connected_sanitized_residual_v1",
        "match_selection_policy": "existential_min_wnode_among_valid_connected_strict_flips_v1",
        "native_output_type": "full_counterfactual_graph",
        "action_adapter": "connected_sanitized_fullgraph_counterfactual_v1",
        "calibration_manifest": str(calibration_path),
        "calibration_manifest_sha256": _sha(calibration_path),
        "git_commit": str(args.git_commit),
    }
    if args.validate_only:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copy2(selected, output / "selected_top20_for_eval.csv")
    (output / "frozen_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "_RUN_COMPLETE.json").write_text(
        json.dumps({"run_complete": True, **payload}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[BACE_GLOBALGCE_FROZEN_SUMMARY_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
