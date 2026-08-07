from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.audit_bace_paper_artifacts import audit_bace_artifacts
from src.eval.bace_paper_artifacts import (
    FIGURE3_FIELDS,
    FIGURE4_FIELDS,
    QUANTILES,
    TABLE2_FIELDS,
    export_bace_method_artifacts,
    freeze_bace_thresholds,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _threshold_contract(tmp_path: Path) -> Path:
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    _write_csv(
        calibration / "distance_quantiles.csv",
        ["threshold_source", "quantile", "threshold"],
        [
            {"threshold_source": "auto_quantile", "quantile": q, "threshold": t}
            for q, t in zip(QUANTILES, thresholds, strict=True)
        ],
    )
    (calibration / "run_config.json").write_text(
        json.dumps({"threshold_source": "auto_quantile"}), encoding="utf-8"
    )
    parents = tmp_path / "calibration_parents.csv"
    _write_csv(parents, ["parent_id", "smiles", "label"], [{"parent_id": "p", "smiles": "CC", "label": 1}])
    target = tmp_path / "thresholds.json"
    freeze_bace_thresholds(
        calibration_run_dir=calibration,
        output_path=target,
        calibration_parent_csv=parents,
    )
    return target


def _fake_run(tmp_path: Path, *, display: str, parents: int = 2) -> Path:
    root = tmp_path / f"run_{display.lower()}"
    ours = display == "Ours"
    candidate_root = tmp_path / f"{display.lower()}_candidates"
    candidate_path = (
        candidate_root / "selected_subgraphs.csv"
        if ours
        else tmp_path / f"{display.lower()}_candidates.csv"
    )
    candidate_rows = [
        {
            "rank": rank,
            "candidate_id": f"c{rank}",
            ("final_fragment" if ours else "candidate_smiles"): "C" * rank,
        }
        for rank in range(1, 21)
    ]
    _write_csv(
        candidate_path,
        ["rank", "candidate_id", "final_fragment" if ours else "candidate_smiles"],
        candidate_rows,
    )
    config = {
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "preselected_topk": 20,
        "cf_mode": "strict_flip",
        "main_ccrcov_uses": "teacher_strict_flip",
        "selection_method": "frozen_external_order",
        "teacher_path": "bace_teacher.pkl",
        "molclr_checkpoint": "model.pth",
    }
    if ours:
        config["ours_selected_path"] = str(candidate_root)
    else:
        config["fullgraph_candidates_path"] = str(candidate_path)
    root.mkdir()
    (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "cache_stats.json").write_text("{}", encoding="utf-8")
    _write_csv(
        root / "combined/combined_threshold_summary.csv",
        ["method", "num_candidates"],
        [{"method": "ours_selected_subgraphs" if ours else display, "num_candidates": 20}],
    )
    details: list[dict[str, object]] = []
    for parent in range(parents):
        for rank in range(1, 21):
            details.append(
                {
                    "method": "ours_selected_subgraphs" if ours else display,
                    "parent_id": f"p{parent}",
                    "candidate_id": f"c{rank}",
                    "candidate_smiles": "C" * rank,
                    "fragment_smiles": "C" * rank if ours else "",
                    "distance": 0.005 * rank + 0.001 * parent,
                    "label": 1,
                    "pred_before": 1,
                    "pred_after": 0,
                    "cf_flip": True,
                    "cf_drop": 0.5,
                }
            )
    _write_csv(
        root / "details/pair_details.csv",
        [
            "method",
            "parent_id",
            "candidate_id",
            "candidate_smiles",
            "fragment_smiles",
            "distance",
            "label",
            "pred_before",
            "pred_after",
            "cf_flip",
            "cf_drop",
        ],
        details,
    )
    return root


def test_export_bace_method_uses_fixed_schema_and_prefix(tmp_path: Path) -> None:
    thresholds = _threshold_contract(tmp_path)
    run = _fake_run(tmp_path, display="Ours")
    output = tmp_path / "ours"
    summary = export_bace_method_artifacts(
        method="ours",
        test_run_dir=run,
        thresholds_json=thresholds,
        output_dir=output,
        expected_parent_count=2,
    )
    with (output / "figure3_coverage_vs_k.csv").open() as handle:
        figure3 = csv.DictReader(handle)
        rows3 = list(figure3)
        assert tuple(figure3.fieldnames or ()) == FIGURE3_FIELDS
    with (output / "figure4_coverage_vs_threshold.csv").open() as handle:
        figure4 = csv.DictReader(handle)
        rows4 = list(figure4)
        assert tuple(figure4.fieldnames or ()) == FIGURE4_FIELDS
    with (output / "table2_ours_k10.csv").open() as handle:
        table = csv.DictReader(handle)
        assert tuple(table.fieldnames or ()) == TABLE2_FIELDS
        assert len(list(table)) == 1
    assert [int(row["k"]) for row in rows3] == list(range(1, 21))
    assert len(rows4) == 7
    assert summary["selection_performed_in_eval"] is False


def test_single_ours_audit_accepts_direct_paper_root(tmp_path: Path) -> None:
    thresholds = _threshold_contract(tmp_path)
    paper_root = tmp_path / "bace_ours_wnode"
    run = _fake_run(tmp_path, display="Ours")
    export_bace_method_artifacts(
        method="ours",
        test_run_dir=run,
        thresholds_json=thresholds,
        output_dir=paper_root,
        expected_parent_count=2,
    )
    audit = audit_bace_artifacts(
        paper_root,
        methods=(("ours", "Ours"),),
        thresholds_path=thresholds,
    )
    assert audit["passed"] is True
    assert audit["methods"] == ["Ours"]
    assert audit["test_parent_count"] == 2
    assert (paper_root / "table2_bace_k10.csv").is_file()
