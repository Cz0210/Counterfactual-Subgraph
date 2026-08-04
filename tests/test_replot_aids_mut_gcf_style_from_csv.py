from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/paper/replot_aids_mut_gcf_style_from_csv.py"
WRAPPER = ROOT / "scripts/slurm/replot_aids_mut_gcf_style_from_csv.sh"
SPEC = importlib.util.spec_from_file_location("replot_aids_mut", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replot)


def _write_valid_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    figure3_rows = []
    for dataset in replot.EXPECTED_DATASETS:
        for method_index, method in enumerate(replot.renderer.METHODS):
            for k in range(1, 21):
                figure3_rows.append(
                    {
                        "Dataset": dataset,
                        "Method": method,
                        "K": k,
                        "Theta": 0.05,
                        "Coverage": min(0.95, 0.01 * k + 0.02 * method_index),
                        "Cost": 0.1 - 0.002 * k + 0.001 * method_index,
                    }
                )
    figure3 = pd.DataFrame(figure3_rows)

    grid = np.linspace(0.0, 0.0535, 601)
    figure4_rows = []
    for dataset in replot.EXPECTED_DATASETS:
        for method_index, method in enumerate(replot.renderer.METHODS):
            for threshold in grid:
                figure4_rows.append(
                    {
                        "Dataset": dataset,
                        "Method": method,
                        "K": 10,
                        "Threshold": threshold,
                        "Coverage": min(1.0, threshold * 10 + 0.01 * method_index),
                    }
                )
    figure4 = pd.DataFrame(figure4_rows)

    table_rows = []
    for method in replot.renderer.METHODS:
        row = {"Method": method}
        for dataset in replot.renderer.DATASET_ORDER:
            if dataset in replot.EXPECTED_DATASETS:
                k10 = figure3.loc[
                    (figure3["Dataset"] == dataset)
                    & (figure3["Method"] == method)
                    & (figure3["K"] == 10)
                ].iloc[0]
                row[f"{dataset} Coverage"] = k10["Coverage"]
                row[f"{dataset} Cost"] = k10["Cost"]
            else:
                row[f"{dataset} Coverage"] = np.nan
                row[f"{dataset} Cost"] = np.nan
        table_rows.append(row)
    table = pd.DataFrame(table_rows)

    root.mkdir(parents=True)
    figure3.to_csv(root / replot.FIGURE3_FILENAME, index=False)
    figure4.to_csv(root / replot.FIGURE4_FILENAME, index=False)
    table.to_csv(root / replot.TABLE2_FILENAME, index=False)
    return figure3, figure4, table


def test_frozen_csv_loader_validates_complete_combined_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected_figure3, expected_figure4, _ = _write_valid_inputs(source)
    figure3, figure4, table = replot.load_frozen_csvs(source)
    pd.testing.assert_frame_equal(figure3, expected_figure3, check_dtype=False)
    pd.testing.assert_frame_equal(figure4, expected_figure4, check_dtype=False)
    assert len(table) == 8
    assert set(table["Dataset"]) == {"AIDS", "Mutagenicity"}


def test_loader_rejects_table_values_that_differ_from_figure3_k10(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _, _, table = _write_valid_inputs(source)
    table.loc[0, "Mutagenicity Coverage"] += 0.01
    table.to_csv(source / replot.TABLE2_FILENAME, index=False)
    with pytest.raises(ValueError, match="differs from Figure 3 K=10"):
        replot.load_frozen_csvs(source)


def test_loader_rejects_incomplete_dense_threshold_grid(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _, figure4, _ = _write_valid_inputs(source)
    figure4.iloc[:-1].to_csv(source / replot.FIGURE4_FILENAME, index=False)
    with pytest.raises(ValueError, match="expected 4808 rows"):
        replot.load_frozen_csvs(source)


def test_wrapper_uses_v2_copy_as_read_only_source_and_new_v3_output() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "aids_mutagenicity_wnode_gcf_style_matched_aids_v2_copy" in text
    assert "aids_mutagenicity_wnode_gcf_style_matched_aids_v3" in text
    assert 'FIGURE3_COVERAGE_YMAX="${FIGURE3_COVERAGE_YMAX:-90}"' in text
    assert "--figure3-coverage-ymax" in text
    assert "[AIDS_MUT_WNODE_GCF_STYLE_V3_SUCCESS]" in text
    assert "pair_details" not in text
    assert "teacher" not in text.lower()
    assert "distance_recomputed" not in text
    assert not re.search(r"(^|\s)unset\s+(http|https|all)_proxy", text, re.IGNORECASE)


def test_renderer_default_stays_80_and_v3_contract_is_90() -> None:
    assert replot.renderer.render_figure3.__kwdefaults__["coverage_ymax"] == 80.0
    parser = replot._build_parser()
    args = parser.parse_args(["--input-dir", "source", "--output-dir", "v3"])
    assert args.figure3_coverage_ymax == 90.0


def test_main_replays_exact_csv_bytes_and_records_current_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_valid_inputs(source)
    (source / "combined_manifest.json").write_text(
        json.dumps(
            {
                "outputs": {
                    replot.FIGURE3_FILENAME: {"sha256": "0" * 64},
                    replot.FIGURE4_FILENAME: {
                        "sha256": replot.renderer.sha256(source / replot.FIGURE4_FILENAME)
                    },
                    replot.TABLE2_FILENAME: {
                        "sha256": replot.renderer.sha256(source / replot.TABLE2_FILENAME)
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    before = {name: (source / name).read_bytes() for name in replot.SOURCE_FILENAMES}
    output = tmp_path / "v3"

    assert (
        replot.main(
            [
                "--project-root",
                str(tmp_path),
                "--input-dir",
                "source",
                "--output-dir",
                "v3",
                "--figure3-coverage-ymax",
                "90",
            ]
        )
        == 0
    )

    for name, content in before.items():
        assert (source / name).read_bytes() == content
        assert (output / name).read_bytes() == content
    manifest = json.loads((output / "combined_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_only"] is True
    assert manifest["figure3_coverage_ymax"] == 90.0
    assert manifest["distance_recomputed"] is False
    assert manifest["source_csv_inventory"][replot.FIGURE3_FILENAME][
        "source_manifest_sha256_matches"
    ] is False
    complete = json.loads((output / "_RUN_COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["run_complete"] is True
