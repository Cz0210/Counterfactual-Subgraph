from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/paper/plot_aids_mut_gcf_style.py"
WRAPPER_PATH = ROOT / "scripts/slurm/export_and_plot_aids_mut_wnode_gpu.sh"
OLD_RENDERER_PATH = (
    ROOT
    / "outputs/hpc/eval/paper/Wasserstein_0720_gcfStyle/render_gcf_style_results.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_aids_mut_gcf_style", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plotter = _load_module()


def _write_aids_figure3(path: Path, *, theta: float = 0.05) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_index, method in enumerate(plotter.METHODS):
        for k in range(1, 21):
            rows.append(
                {
                    "method": method,
                    "distance_label": "MolCLR-Node-Wasserstein",
                    "k": k,
                    "theta": theta,
                    "coverage": 0.1 + 0.01 * method_index + 0.02 * k,
                    "conditional_median_cost": 0.1 - 0.002 * k + 0.001 * method_index,
                    "plotted_cost": 0.1 - 0.002 * k + 0.001 * method_index,
                    "plotted_cost_metric": "conditional_median_cost",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_aids_figure4(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.0, 0.0535, 601)
    rows = []
    for method_index, method in enumerate(plotter.METHODS):
        for index, threshold in enumerate(thresholds):
            rows.append(
                {
                    "method": method,
                    "distance_label": "MolCLR-Node-Wasserstein",
                    "k": 10,
                    "threshold": threshold,
                    "coverage": min(0.79, index / 1000 + 0.01 * method_index),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_mut_root(
    base: Path,
    method: str,
    *,
    metadata_in_csv: bool = True,
) -> Path:
    slug = {
        "Ours": "ours",
        "GlobalGCE": "globalgce",
        "CLEAR": "clear",
        "GCFExplainer": "gcfexplainer",
    }[method]
    root = base / slug
    root.mkdir(parents=True)
    metadata = {
        "distance_line": "MolCLR-Node-Wasserstein",
        "distance_type": "node_wasserstein",
        "cf_mode": "strict_flip",
        "num_parents": 217,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
    }
    row_metadata = metadata if metadata_in_csv else {}
    theta = 0.038576244576299636
    pd.DataFrame(
        [
            {
                **row_metadata,
                "k": k,
                "theta": theta,
                "coverage": 0.1 + 0.02 * k,
                "conditional_median_cost": 0.08 - 0.001 * k,
            }
            for k in range(1, 21)
        ]
    ).to_csv(root / "figure3_coverage_vs_k.csv", index=False)
    thresholds = np.asarray(
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
    pd.DataFrame(
        [
            {
                **row_metadata,
                "k": 10,
                "threshold": threshold,
                "coverage": 0.1 + 0.1 * index,
            }
            for index, threshold in enumerate(thresholds)
        ]
    ).to_csv(root / "figure4_coverage_vs_threshold.csv", index=False)
    pd.DataFrame(
        [
            {
                **row_metadata,
                "k": 10,
                "theta": theta,
                "coverage": 0.4,
                "conditional_median_cost": 0.04,
                "num_test_parents": 217,
            }
        ]
    ).to_csv(root / f"table2_{slug}_k10.csv", index=False)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "strict_flip_definition": "pred_before == 1 and pred_after == 0",
                "run_complete": True,
            }
        ),
        encoding="utf-8",
    )

    candidate_ids = [f"{slug}-candidate-{rank:02d}" for rank in range(1, 21)]
    order_rows = [
        {
            "rank": rank,
            "candidate_id": candidate_id,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
        }
        for rank, candidate_id in enumerate(candidate_ids, start=1)
    ]
    if method in {"Ours", "GCFExplainer"}:
        order_name = "selected_sequence.jsonl"
        (root / order_name).write_text(
            "".join(json.dumps(row) + "\n" for row in order_rows), encoding="utf-8"
        )
    else:
        order_name = "selected_top20.csv" if method == "GlobalGCE" else "selected_candidates.csv"
        pd.DataFrame(order_rows).to_csv(root / order_name, index=False)

    pair_rows = []
    for parent_index in range(217):
        for rank, candidate_id in enumerate(candidate_ids, start=1):
            strict = (parent_index + rank) % 4 != 0
            distance = 0.001 * rank + 0.0001 * parent_index
            if method == "Ours":
                pair_rows.append(
                    {
                        "parent_id": f"parent-{parent_index:03d}",
                        "candidate_id": candidate_id,
                        "pair_strict_flip": strict,
                        "wnode_distance": distance if strict else None,
                    }
                )
            else:
                pair_rows.append(
                    {
                        "parent_id": f"parent-{parent_index:03d}",
                        "candidate_id": candidate_id,
                        "teacher_strict_flip": strict,
                        "distance": distance,
                        "distance_line": "MolCLR-Node-Wasserstein",
                        "distance_type": "node_wasserstein",
                    }
                )
    if method == "Ours":
        (root / "pair_matrix.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in pair_rows), encoding="utf-8"
        )
    elif method == "CLEAR":
        pair_path = root / "test/k20_pair_details.csv"
        pair_path.parent.mkdir(parents=True)
        pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    else:
        pd.DataFrame(pair_rows).to_csv(root / "test_pair_details.csv", index=False)
    return root


def test_old_correct_renderer_is_preserved_as_style_source() -> None:
    assert OLD_RENDERER_PATH.is_file()
    old = OLD_RENDERER_PATH.read_text(encoding="utf-8")
    assert 'figsize=(16.0, 6.3)' in old
    assert 'figsize=(16.0, 3.8)' in old
    assert 'figsize=(15.8, 3.3)' in old


def test_aids_frozen_source_paths_are_exact() -> None:
    assert plotter.AIDS_FIGURE3_RELATIVE_PATH.as_posix() == (
        "outputs/hpc/eval/paper/molclr_node_wasserstein_figure3_theta005_raw/"
        "wnode_fig3_theta005_figure3_wnode_coverage_cost_vs_k.csv"
    )
    assert plotter.AIDS_FIGURE4_RELATIVE_PATH.as_posix() == (
        "outputs/hpc/eval/paper/molclr_node_wasserstein_figure4_redline_k10/"
        "wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv"
    )


def test_aids_figure3_gate_reads_80_rows_at_theta_005(tmp_path: Path) -> None:
    path = _write_aids_figure3(tmp_path / plotter.AIDS_FIGURE3_RELATIVE_PATH)
    frame, source = plotter.load_aids_figure3(path)
    assert len(frame) == 80
    assert source["row_count"] == 80
    assert source["theta"] == 0.05
    assert source["cost_source_column"] == "conditional_median_cost"
    assert set(frame["Method"]) == set(plotter.METHODS)
    for method in plotter.METHODS:
        rows = frame.loc[frame["Method"] == method]
        assert rows["K"].tolist() == list(range(1, 21))
        assert np.allclose(rows["Theta"], 0.05, rtol=0.0, atol=1e-12)


def test_aids_figure3_rejects_a_different_theta(tmp_path: Path) -> None:
    path = _write_aids_figure3(tmp_path / "figure3.csv", theta=0.049)
    with pytest.raises(ValueError, match="theta must be 0.05"):
        plotter.load_aids_figure3(path)


def test_aids_figure4_gate_reads_dense_601_point_curves(tmp_path: Path) -> None:
    path = _write_aids_figure4(tmp_path / plotter.AIDS_FIGURE4_RELATIVE_PATH)
    frame, source = plotter.load_aids_figure4(path)
    assert len(frame) == 2404
    assert source["points_per_method"] == 601
    assert source["threshold_min"] == pytest.approx(0.0, abs=1e-15)
    assert source["threshold_max"] == pytest.approx(0.0535, abs=1e-15)
    assert source["interpolation_performed"] is False
    for method in plotter.METHODS:
        rows = frame.loc[frame["Method"] == method]
        assert len(rows) == 601
        assert rows["K"].unique().tolist() == [10]


def test_aids_figure4_rejects_different_method_grid(tmp_path: Path) -> None:
    path = _write_aids_figure4(tmp_path / "figure4.csv")
    frame = pd.read_csv(path)
    mask = (frame["method"] == "CLEAR") & (frame.index == 2 * 601 + 300)
    frame.loc[mask, "threshold"] += 1e-6
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="threshold grids differ|strictly increasing"):
        plotter.load_aids_figure4(path)


def test_aids_table2_exact_values_and_ranking_styles(monkeypatch, tmp_path: Path) -> None:
    applicable_counts = {
        "Ours": (998, 961),
        "GlobalGCE": (1097, 576),
        "CLEAR": (1097, 278),
        "GCFExplainer": (1097, 824),
    }
    roots = {}
    for method in plotter.METHODS:
        root = tmp_path / method
        root.mkdir()
        roots[method] = root

    def fake_parent_best(root: Path, *, k: int):
        method = root.name
        expected_coverage, expected_cost = plotter.AIDS_TABLE2_EXPECTED[method]
        count, covered = applicable_counts[method]
        below = min(expected_cost, 0.04)
        values = np.full(count, expected_cost, dtype=float)
        values[:covered] = below
        # Preserve the exact audited median while retaining the exact covered count.
        if expected_cost <= plotter.AIDS_TABLE2_THETA:
            values[:covered] = expected_cost
            values[covered:] = 0.1
        assert np.count_nonzero(values <= plotter.AIDS_TABLE2_THETA) == round(
            expected_coverage * 1283
        )
        assert np.median(values) == pytest.approx(expected_cost, abs=1e-15)
        return pd.Series(values), {"candidate_order_changed": False, "k": k}

    monkeypatch.setattr(plotter, "load_parent_best_distances", fake_parent_best)
    table, _ = plotter.load_aids_table2(roots)
    assert table["Method"].tolist() == list(plotter.METHODS)
    for method, (coverage, cost) in plotter.AIDS_TABLE2_EXPECTED.items():
        row = table.loc[table["Method"] == method].iloc[0]
        assert row["Coverage"] == pytest.approx(coverage, abs=1e-12)
        assert row["Cost"] == pytest.approx(cost, abs=1e-12)
        assert row["Theta"] == 0.05
        assert row["K"] == 10
        assert row["NumParents"] == 1283
    styles = plotter.table_cell_styles(table)
    assert styles[("AIDS", "Ours", "Coverage")] == "best"
    assert styles[("AIDS", "Ours", "Cost")] == "best"
    assert styles[("AIDS", "GCFExplainer", "Coverage")] == "second"
    assert styles[("AIDS", "GCFExplainer", "Cost")] == "second"


def test_mutagenicity_loader_keeps_its_own_frozen_theta_and_grid(tmp_path: Path) -> None:
    root = _write_mut_root(tmp_path, "Ours")
    figure3, figure4, table, source = plotter.load_mut_native_method(root, "Ours")
    assert figure3["K"].tolist() == list(range(1, 21))
    assert len(figure4) == 7
    assert table["Theta"] == pytest.approx(0.038576244576299636, abs=1e-12)
    assert source["distance_recomputed"] is False
    assert source["candidate_order_changed"] is False


def test_mutagenicity_provenance_may_come_from_run_manifest(tmp_path: Path) -> None:
    root = _write_mut_root(tmp_path, "Ours", metadata_in_csv=False)
    figure3, figure4, table, _source = plotter.load_mut_native_method(root, "Ours")
    assert len(figure3) == 20
    assert len(figure4) == 7
    assert table["NumParents"] == 217


def test_mutagenicity_match_aids_profile_uses_saved_pairs_on_dense_grid(tmp_path: Path) -> None:
    root = _write_mut_root(tmp_path, "GCFExplainer")
    figure3, figure4, table, source = plotter.load_mut_method(root, "GCFExplainer")
    assert figure3["K"].tolist() == list(range(1, 21))
    assert set(figure3["Theta"]) == {0.05}
    assert len(figure4) == 601
    assert figure4["Threshold"].iloc[0] == pytest.approx(0.0, abs=1e-15)
    assert figure4["Threshold"].iloc[-1] == pytest.approx(0.0535, abs=1e-15)
    assert table["Theta"] == 0.05
    assert source["threshold_mode"] == "match-aids"
    assert source["pair_count"] == 217 * 20
    assert source["complete_cartesian"] is True
    assert source["distance_recomputed"] is False
    assert source["candidate_order_changed"] is False


def test_layout_and_method_style_match_reference() -> None:
    assert plotter.DATASET_ORDER == ("AIDS", "NCI1", "Mutagenicity", "Proteins")
    assert plotter.ACTIVE_DATASET_COLUMNS == {"AIDS": 0, "Mutagenicity": 2}
    assert plotter.METHODS == ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
    assert plotter.METHOD_STYLES == {
        "Ours": {"color": "black", "marker": "s", "label": "Ours"},
        "GlobalGCE": {"color": "#E53935", "marker": "x", "label": "GlobalGCE"},
        "CLEAR": {"color": "#2E7D32", "marker": "*", "label": "CLEAR"},
        "GCFExplainer": {"color": "#B02BC7", "marker": "^", "label": "GCFExplainer"},
    }
    assert plotter.FIGURE3_MARKER_K == (1, 3, 5, 10, 15, 20)
    assert plotter.FIGURE3_MARKER_INDICES == (0, 2, 4, 9, 14, 19)
    assert plotter.FIGURE4_AIDS_MARKEVERY == 100


def test_reference_canvas_aspect_ratios_are_retained() -> None:
    expected = {
        plotter.FIGURE3_FIGSIZE: 2048 / 826,
        plotter.FIGURE4_FIGSIZE: 2048 / 514,
        plotter.TABLE2_FIGSIZE: 2048 / 458,
    }
    for size, target_ratio in expected.items():
        assert size[0] / size[1] == pytest.approx(target_ratio, rel=0.08)


def test_plotter_does_not_recompute_distance_or_candidate_rank() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "standardize_raw_aids_gcf_run" not in source
    assert "compute_k_curve" not in source
    assert "compute_prefix_metrics" not in source
    assert "MolCLR" in source
    assert '"distance_recomputed": False' in source
    assert '"candidate_order_changed": False' in source
    assert "q30" not in source.lower()
    forbidden_threshold = "0.0545" + "395671276376"
    assert forbidden_threshold not in source


def test_exact_cli_paths_are_enforced(tmp_path: Path) -> None:
    namespace = argparse.Namespace(
        aids_figure3_csv=str(plotter.AIDS_FIGURE3_RELATIVE_PATH),
        aids_figure4_csv=str(plotter.AIDS_FIGURE4_RELATIVE_PATH),
        aids_ours_root=str(plotter.AIDS_TABLE_ROOTS["Ours"]),
        aids_globalgce_root=str(plotter.AIDS_TABLE_ROOTS["GlobalGCE"]),
        aids_clear_root=str(plotter.AIDS_TABLE_ROOTS["CLEAR"]),
        aids_gcf_root=str(plotter.AIDS_TABLE_ROOTS["GCFExplainer"]),
        mut_ours_root="mut/ours",
        mut_globalgce_root="mut/globalgce",
        mut_clear_root="mut/clear",
        mut_gcf_root="mut/gcf",
    )
    paths = plotter._args_paths(namespace, tmp_path)
    assert paths["aids_figure3"] == (tmp_path / plotter.AIDS_FIGURE3_RELATIVE_PATH).resolve()
    namespace.aids_figure3_csv = "outputs/hpc/eval/paper/wrong.csv"
    with pytest.raises(ValueError, match="Expected frozen source"):
        plotter._args_paths(namespace, tmp_path)


def test_wrapper_supports_native_and_match_aids_outputs() -> None:
    text = WRAPPER_PATH.read_text(encoding="utf-8")
    assert plotter.AIDS_FIGURE3_RELATIVE_PATH.as_posix() in text
    assert plotter.AIDS_FIGURE4_RELATIVE_PATH.as_posix() in text
    for root in plotter.AIDS_TABLE_ROOTS.values():
        assert root.as_posix() in text
    assert '--aids-figure3-csv "$AIDS_FIGURE3_CSV"' in text
    assert '--aids-figure4-csv "$AIDS_FIGURE4_CSV"' in text
    assert '--mut-threshold-mode "$MUT_THRESHOLD_MODE"' in text
    assert 'MUT_THRESHOLD_MODE="${MUT_THRESHOLD_MODE:-match-aids}"' in text
    assert "aids_mutagenicity_wnode_gcf_style_v2" in text
    assert "aids_mutagenicity_wnode_gcf_style_matched_aids_v1" in text
    assert "[AIDS_MUT_WNODE_GCF_STYLE_V2_SUCCESS]" in text
    assert "[AIDS_MUT_WNODE_GCF_STYLE_MATCHED_AIDS_SUCCESS]" in text
    assert "figure3_gcf_style_aids_mut.png" in text
    assert "figure4_gcf_style_aids_mut.png" in text
    assert "table2_gcf_style_aids_mut.png" in text
    assert "aids_common3_standardized_v2" not in text
    assert "combined/combined_threshold_summary.csv" not in text


def test_wrapper_keeps_verified_resources_and_no_proxy_mutation() -> None:
    text = WRAPPER_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --partition=A800" in text
    assert "#SBATCH --gres=gpu:a800:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "conda activate smiles_pip118" in text
    assert "export MPLBACKEND=Agg" in text
    assert not re.search(r"(^|\s)unset\s+(http|https|all)_proxy", text, re.IGNORECASE)
    assignments = [
        line
        for line in text.splitlines()
        if ("ROOT=" in line or "AIDS_FIGURE" in line) and not line.lstrip().startswith("#")
    ]
    assert all("ccrcov_molclr_node_fgw_" not in line for line in assignments)
    assert all("lam05" not in line.lower() for line in assignments)


def test_native_and_match_aids_profiles_coexist_with_wnode_provenance() -> None:
    assert plotter.DISTANCE_LABEL == "MolCLR-Node-Wasserstein"
    assert plotter.DISTANCE_TYPE == "node_wasserstein"
    assert plotter.AIDS_FIGURE3_THETA == 0.05
    assert plotter.AIDS_TABLE2_THETA == 0.05
    source = inspect.getsource(plotter)
    assert 'choices=("native", "match-aids")' in source
    assert '"threshold_mode": "native"' in source
    assert '"threshold_mode": "match-aids"' in source
