from __future__ import annotations

import argparse

import pytest

from scripts.autodl.build_four_by_four_final_tasks import DATASETS, METHODS, build


def _args(tmp_path):
    expectations = tmp_path / "expectations.json"
    expectations.write_text("{}\n")
    scan = tmp_path / "scan"
    scan.mkdir()
    bace_standardized = {
        "Ours": "bace_ours_standardized",
        "GCFExplainer": "bace_gcfexplainer_standardized",
        "GlobalGCE": "bace_globalgce_standardized",
        "ComRecGC": "bace_comrecgc_standardized",
    }
    cells = [
        f"{dataset}/{method}="
        + (
            bace_standardized[method]
            if dataset == "BACE" and method in bace_standardized
            else f"cell_{dataset.lower()}_{method.lower()}"
        )
        for dataset in DATASETS
        for method in METHODS
    ]
    cells = [
        "Mutagenicity/GCFExplainer=mut_gcf_legacy_standardized"
        if value.startswith("Mutagenicity/GCFExplainer=")
        else value
        for value in cells
    ]
    return argparse.Namespace(
        controller_id="four_methods_four_datasets_continuation_v1",
        cell_task=cells,
        taste_license_task_id="tastemolnet_license_audit",
        expectations_json=str(expectations),
        scan_root=[str(scan)],
        output_root=str(tmp_path / "results"),
    )


def test_final_matrix_audit_binds_all_cells_and_export_contract(tmp_path):
    fragment, contract = build(_args(tmp_path))
    task = fragment["tasks"][0]
    assert len(contract["cells"]) == 16
    assert len(set(contract["cells"].values())) == 16
    assert task["id"] == contract["matrix_task_id"] == "final_matrix_audit"
    assert "--explicit-cell" in task["command"]
    assert task["data_splits"] == []
    assert "--require-complete" not in task["command"]


def test_final_matrix_audit_rejects_duplicate_terminal_task(tmp_path):
    args = _args(tmp_path)
    args.cell_task[-1] = args.cell_task[-1].split("=", 1)[0] + "=" + args.cell_task[0].split("=", 1)[1]
    with pytest.raises(ValueError, match="distinct"):
        build(args)


@pytest.mark.parametrize(
    ("method", "science_terminal"),
    [
        ("Ours", "bace_b14_frozen"),
        ("GCFExplainer", "bace_gcfexplainer_final_freeze"),
        ("GlobalGCE", "bace_globalgce_final_freeze"),
        ("ComRecGC", "bace_comrecgc_final_freeze"),
    ],
)
def test_final_matrix_audit_rejects_raw_bace_science_terminal(
    tmp_path, method, science_terminal
):
    args = _args(tmp_path)
    prefix = f"BACE/{method}="
    args.cell_task = [
        prefix + science_terminal if value.startswith(prefix) else value
        for value in args.cell_task
    ]
    with pytest.raises(ValueError, match="must bind standardized task"):
        build(args)


def test_final_matrix_audit_rejects_raw_mut_gcf_heldout_root(tmp_path):
    args = _args(tmp_path)
    args.cell_task = [
        "Mutagenicity/GCFExplainer=mut_gcf_legacy_heldout"
        if value.startswith("Mutagenicity/GCFExplainer=")
        else value
        for value in args.cell_task
    ]
    with pytest.raises(ValueError, match="standardized closure"):
        build(args)
