from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/paper/plot_aids_mut_gcf_style.py"
WRAPPER_PATH = ROOT / "scripts/slurm/export_and_plot_aids_mut_wnode_gpu.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_aids_mut_gcf_style", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plotter = _load_module()


METHOD_SLUGS = {
    "Ours": "ours",
    "GlobalGCE": "globalgce",
    "CLEAR": "clear",
    "GCFExplainer": "gcfexplainer",
}


def _write_root(base: Path, *, dataset: str, method: str) -> Path:
    root = base / dataset.lower() / METHOD_SLUGS[method]
    root.mkdir(parents=True)
    parent_count = 1283 if dataset == "AIDS" else 217
    theta = 0.014630082696799 if dataset == "AIDS" else 0.038576244576299636
    figure4_thresholds = (
        np.linspace(0.0, 0.0391548051165848, 102)
        if dataset == "AIDS"
        else np.asarray(
            [
                0.014088122444763422,
                0.02289075857275116,
                0.03237569932491265,
                0.038576244576299636,
                0.04961842688391724,
                0.06406452526754104,
                0.09832242115448658,
            ]
        )
    )
    offset = 0.01 * list(METHOD_SLUGS).index(method)
    figure3 = pd.DataFrame(
        {
            "k": range(1, 21),
            "coverage": np.linspace(0.05 + offset, 0.55 + offset, 20),
            "conditional_median_cost": np.linspace(0.08 + offset, 0.03 + offset, 20),
            "fixed_capped_median_cost": 0.9,
            "distance_line": "MolCLR-Node-Wasserstein",
            "distance_type": "node_wasserstein",
            "cf_mode": "strict_flip",
            "num_parents": parent_count,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
        }
    )
    figure3.to_csv(root / "figure3_coverage_vs_k.csv", index=False)
    figure4 = pd.DataFrame(
        {
            "k": 10,
            "threshold": figure4_thresholds,
            "coverage": np.linspace(0.0, 0.7 + offset, len(figure4_thresholds)),
            "distance_line": "MolCLR-Node-Wasserstein",
            "distance_type": "node_wasserstein",
            "cf_mode": "strict_flip",
            "num_parents": parent_count,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
        }
    )
    figure4.to_csv(root / "figure4_coverage_vs_threshold.csv", index=False)
    table = pd.DataFrame(
        {
            "k": [10],
            "theta": [theta],
            "coverage": [0.4 + offset],
            "conditional_median_cost": [0.04 + offset],
            "fixed_capped_median_cost": [0.8],
            "distance_line": ["MolCLR-Node-Wasserstein"],
            "distance_type": ["node_wasserstein"],
            "cf_mode": ["strict_flip"],
            "num_parents": [parent_count],
            "candidate_set_preselected": [True],
            "selection_performed_in_eval": [False],
        }
    )
    table.to_csv(root / f"table2_{METHOD_SLUGS[method]}_k10.csv", index=False)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "method": method,
                "distance_line": "MolCLR-Node-Wasserstein",
                "distance_type": "node_wasserstein",
                "cf_mode": "strict_flip",
                "test_parent_count": parent_count,
                "candidate_count": 20,
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "run_complete": True,
            }
        ),
        encoding="utf-8",
    )
    return root


def _all_roots(base: Path) -> dict[tuple[str, str], Path]:
    return {
        (dataset, method): _write_root(base, dataset=dataset, method=method)
        for dataset in ("AIDS", "Mutagenicity")
        for method in METHOD_SLUGS
    }


def _write_raw_aids_gcf_run(base: Path) -> Path:
    root = base / "ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_gcf"
    (root / "combined").mkdir(parents=True)
    (root / "details").mkdir()
    candidates = pd.DataFrame(
        {
            "rank": range(1, 21),
            "candidate_id": [f"gcf-{rank:02d}" for rank in range(1, 21)],
            "canonical_smiles": ["C" * rank for rank in range(1, 21)],
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
        }
    )
    candidate_path = root / "selected_top20.csv"
    candidates.to_csv(candidate_path, index=False)
    thresholds = np.asarray(
        [
            0.0036989375029972,
            0.0048443801866389,
            0.0070071265985866,
            0.0092885245733581,
            0.014630082696799,
            0.0218176142567855,
            0.0391548051165848,
        ]
    )
    best_distances = np.asarray(
        [0.001 + 0.000001 * parent_index for parent_index in range(1, 1284)]
    )
    method = "gcfexplainer_top20_normalized"
    summary = pd.DataFrame(
        {
            "method": method,
            "distance_type": "node_wasserstein",
            "distance_line": "MolCLR-Node-Wasserstein",
            "threshold": thresholds,
            "cf_mode": "strict_flip",
            "num_parents": 1283,
            "num_candidates": 20,
            "close_cf_coverage": [
                float(np.count_nonzero(best_distances <= threshold) / 1283)
                for threshold in thresholds
            ],
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
        }
    )
    summary.to_csv(root / "combined/combined_threshold_summary.csv", index=False)
    details = [
        {
            "method": method,
            "parent_id": f"parent-{parent_index:04d}",
            "candidate_id": f"gcf-{rank:02d}",
            "candidate_smiles": "C" * rank,
            "distance": 0.001 * rank + 0.000001 * parent_index,
            "label": 1,
            "pred_before": 1,
            "pred_after": 0,
            "teacher_strict_flip": True,
            "cf_drop": 0.5,
        }
        for parent_index in range(1, 1284)
        for rank in range(1, 21)
    ]
    pd.DataFrame(details).to_csv(root / "details/pair_details.csv", index=False)
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "distance_line": "MolCLR-Node-Wasserstein",
                "distance_type": "node_wasserstein",
                "cf_mode": "strict_flip",
                "main_ccrcov_uses": "teacher_strict_flip",
                "fullgraph_candidates_path": str(candidate_path),
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "preselected_topk": 20,
                "selection_method": "normalized_top20",
                "thresholds": thresholds.tolist(),
            }
        ),
        encoding="utf-8",
    )
    (root / "cache_stats.json").write_text("{}\n", encoding="utf-8")
    (root / "_RUN_COMPLETE.json").write_text(
        json.dumps({"complete": True}),
        encoding="utf-8",
    )
    return root


def _argv(base: Path, roots: dict[tuple[str, str], Path], output: Path) -> list[str]:
    return [
        "--project-root",
        str(base),
        "--aids-ours-root",
        str(roots[("AIDS", "Ours")]),
        "--aids-globalgce-root",
        str(roots[("AIDS", "GlobalGCE")]),
        "--aids-clear-root",
        str(roots[("AIDS", "CLEAR")]),
        "--aids-gcf-root",
        str(roots[("AIDS", "GCFExplainer")]),
        "--mut-ours-root",
        str(roots[("Mutagenicity", "Ours")]),
        "--mut-globalgce-root",
        str(roots[("Mutagenicity", "GlobalGCE")]),
        "--mut-clear-root",
        str(roots[("Mutagenicity", "CLEAR")]),
        "--mut-gcf-root",
        str(roots[("Mutagenicity", "GCFExplainer")]),
        "--output-dir",
        str(output),
    ]


def test_figure3_and_table_prefer_conditional_cost(tmp_path: Path) -> None:
    root = _write_root(tmp_path, dataset="Mutagenicity", method="Ours")
    figure3, source = plotter.read_figure3(root, "Mutagenicity", "Ours")
    table, table_source = plotter.read_table2(root, "Mutagenicity", "Ours")
    assert source["columns"]["cost"] == "conditional_median_cost"
    assert table_source["columns"]["cost"] == "conditional_median_cost"
    assert figure3.loc[0, "ConditionalMedianCost"] != 0.9
    assert table["Cost"] != 0.8


def test_combined_plot_writes_wasserstein_only_manifest(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path)
    output = tmp_path / "combined"
    assert plotter.main(_argv(tmp_path, roots, output)) == 0
    manifest = json.loads((output / "combined_manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((output / "_RUN_COMPLETE.json").read_text(encoding="utf-8"))
    assert manifest["distance_label"] == "MolCLR-Node-Wasserstein"
    assert manifest["distance_type"] == "node_wasserstein"
    assert manifest["cf_mode"] == "strict_flip"
    assert manifest["distance_recomputed"] is False
    assert manifest["candidate_order_changed"] is False
    assert complete["run_complete"] is True
    assert manifest["dataset_audit"]["AIDS"]["figure4_threshold_count"] == 102
    assert manifest["dataset_audit"]["Mutagenicity"]["figure4_threshold_count"] == 7
    assert "fgw_lambda" not in json.dumps(manifest).lower()


def test_node_fgw_provenance_is_rejected(tmp_path: Path) -> None:
    root = _write_root(tmp_path, dataset="AIDS", method="Ours")
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["distance_line"] = "MolCLR-Node-FGW"
    manifest["distance_type"] = "node_fgw"
    manifest["fgw_lambda"] = 0.5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = [
        root / "figure3_coverage_vs_k.csv",
        root / "figure4_coverage_vs_threshold.csv",
        root / "table2_ours_k10.csv",
    ]
    with pytest.raises(ValueError, match="Forbidden non-WNode provenance"):
        plotter._metadata_evidence(root=root, project_root=tmp_path, csv_paths=paths)


def test_unreferenced_sibling_manifest_does_not_contaminate_root(tmp_path: Path) -> None:
    root = _write_root(tmp_path, dataset="AIDS", method="Ours")
    (root.parent / "unrelated_manifest.json").write_text(
        json.dumps(
            {
                "output_root": str(root.parent / "different_run"),
                "distance_line": "MolCLR-Node-FGW",
                "distance_type": "node_fgw",
                "fgw_lambda": 0.5,
            }
        ),
        encoding="utf-8",
    )
    paths = [
        root / "figure3_coverage_vs_k.csv",
        root / "figure4_coverage_vs_threshold.csv",
        root / "table2_ours_k10.csv",
    ]
    evidence = plotter._metadata_evidence(
        root=root,
        project_root=tmp_path,
        csv_paths=paths,
    )
    provenance_paths = {Path(item["path"]).name for item in evidence["provenance_files"]}
    assert "run_manifest.json" in provenance_paths
    assert "unrelated_manifest.json" not in provenance_paths


def test_raw_aids_gcf_run_is_standardized_from_frozen_pairs(tmp_path: Path) -> None:
    root = _write_raw_aids_gcf_run(tmp_path)
    before = {
        path.relative_to(root).as_posix(): plotter.sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    (
        figure3,
        figure4,
        table,
        figure3_source,
        figure4_source,
        table_source,
        evidence,
    ) = plotter.standardize_raw_aids_gcf_run(
        root,
        project_root=tmp_path,
        theta_star=0.014630082696799,
        figure4_thresholds=np.sort(
            np.append(
                np.linspace(0.0, 0.0391548051165848, 101),
                0.014630082696799,
            )
        ),
        figure4_threshold_source={
            "path": str(tmp_path / "aids/ours/figure4_coverage_vs_threshold.csv"),
            "sha256": "0" * 64,
        },
    )
    after = {
        path.relative_to(root).as_posix(): plotter.sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert figure3["K"].tolist() == list(range(1, 21))
    assert len(figure4) == 102
    assert figure4["K"].unique().tolist() == [10]
    assert table["K"] == 10
    assert table["Theta"] == pytest.approx(0.014630082696799, abs=1e-12)
    assert figure3_source["distance_recomputed"] is False
    assert figure4_source["teacher_recomputed"] is False
    assert table_source["candidate_order_changed"] is False
    assert evidence["top20_frozen_ranking_verified"] is True
    assert evidence["complete_cartesian_verified"] is True
    assert figure4_source["official_summary_threshold_count"] == 7
    assert len(figure4_source["official_summary_reconstruction"]) == 7
    assert not (root / "figure3_coverage_vs_k.csv").exists()
    assert not (root / "figure4_coverage_vs_threshold.csv").exists()
    assert not (root / "table2_gcfexplainer_k10.csv").exists()


def test_all_four_methods_are_required(tmp_path: Path) -> None:
    roots = _all_roots(tmp_path)
    missing = roots[("AIDS", "GCFExplainer")]
    for path in missing.iterdir():
        path.unlink()
    missing.rmdir()
    with pytest.raises(FileNotFoundError, match="Missing frozen plotting root"):
        plotter.main(_argv(tmp_path, roots, tmp_path / "combined"))


def test_wrapper_uses_correct_fixed_roots_and_resources() -> None:
    text = WRAPPER_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert (
        "ccrcov_molclr_node_wasserstein_full_fixed_oursref1283_"
        "gcfexplainer_top20_normalized_final" in text
    )
    assert "aids_common3_standardized_v2/gcfexplainer" not in text
    assert "combined/combined_threshold_summary.csv" in text
    assert "details/pair_details.csv" in text
    assert "gcfexplainer_native5000_top20_wnode_test_v1" in text
    assert "distance_label=MolCLR-Node-Wasserstein" in text
    assert "distance_type=node_wasserstein" in text
    assert not re.search(r"(^|\s)unset\s+(http|https|all)_proxy", text, re.IGNORECASE)
    assert "fgw_lambda" not in text.lower()
    assignments = [
        line for line in text.splitlines()
        if "ROOT=" in line and not line.lstrip().startswith("#")
    ]
    assert all("ccrcov_molclr_node_fgw_" not in line for line in assignments)
    assert all("lam05" not in line.lower() for line in assignments)
