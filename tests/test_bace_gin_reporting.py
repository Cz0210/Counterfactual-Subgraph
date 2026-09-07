import csv
import math
from pathlib import Path

import pytest

from src.experiments.bace_gin_reporting import (
    INPUTS, METHODS, STYLES, derive, read_csv, reduce_parent_prefix, write_csv,
)


def fixture(root, *, zero=False):
    parents, curves, points = [], [], []
    for k in range(1, 21):
        rows = [dict(method="Ours", parent_id=p, pred_before=pred,
                     K_requested=k, K_effective=min(k, 15), best_valid_distance=distance)
                for p, pred, distance in (("p0", 0, None), ("p1", 1, None if zero else .01))]
        parents += rows
        metric = {"method": "Ours", **reduce_parent_prefix(rows, theta=.02, cap=.03)}
        curves.append(metric)
        if k in (10, 20):
            for x in (0., .01, .02, .03):
                count = int(not zero and x >= .01)
                points.append(dict(method="Ours", K_requested=k, K_effective=min(k, 15),
                                   threshold=x, covered_count=count, coverage=count/2))
    table = [{**curves[9], "state": "EVALUATED"}]
    table += [dict(method=m, state="BLOCKED_MATERIALIZATION" if m=="GlobalGCE" else "READY")
              for m in METHODS[1:]]
    root.mkdir()
    for name, rows in zip(INPUTS, (table, curves, points, parents)):
        write_csv(root/name, rows)
    return root


def change(path, modifier):
    rows = read_csv(path)
    modifier(rows)
    write_csv(path, rows)


def test_one_reducer_k10_percent_and_original_cost(tmp_path):
    data = derive(fixture(tmp_path/"inputs"), expected_parents=2)
    table = data["table2"][0]
    assert table == {**data["figure3"][9], "state": "EVALUATED"}
    assert table["coverage"] == .5 and table["coverage_percent"] == 50
    assert table["fixed_capped_mean"] == .02
    assert table["conditional_median"] == .01
    theta = next(r for r in data["figure4"][10] if r["threshold"] == .02)
    assert theta["covered_count"] == table["covered_count"]
    assert [r["K_effective"] for r in data["figure3"]][15:] == [15]*5
    assert data["table2"][1]["state"] == "BLOCKED"
    assert "coverage" not in data["table2"][1]


def test_real_zero_na_and_pending_distinct(tmp_path):
    data = derive(fixture(tmp_path/"inputs", zero=True), expected_parents=2)
    row = data["table2"][0]
    assert row["coverage"] == 0 and row["conditional_median"] is None
    assert row["fixed_capped_mean"] == .03
    assert data["table2"][2]["state"] == "PENDING"
    assert not any(r["method"] != "Ours" for r in data["figure4"][20])


@pytest.mark.parametrize("index,key,value", [(0,"coverage",".75"), (1,"fixed_capped_mean",".019"),
                                             (2,"covered_count","9")])
def test_any_panel_conflict_refuses_export(tmp_path,index,key,value):
    root = fixture(tmp_path/"inputs")
    change(root/INPUTS[index], lambda rows: rows[0].update({key:value}))
    with pytest.raises(ValueError, match="SOURCE_REDUCER_CONFLICT"):
        derive(root, expected_parents=2)


def test_missing_parent_no_denominator_shrink(tmp_path):
    root = fixture(tmp_path/"inputs")
    change(root/INPUTS[3], lambda rows: rows.pop())
    with pytest.raises(ValueError, match="INCOMPLETE_FIXED_PARENT"):
        derive(root, expected_parents=2)


def test_prefix_cannot_lose_reachability(tmp_path):
    root = fixture(tmp_path/"inputs")
    change(root/INPUTS[3], lambda rows: rows[3].update(best_valid_distance=""))
    with pytest.raises(ValueError, match="NOT_NESTED"):
        derive(root, expected_parents=2)


def test_native_pred0_cannot_have_finite_strictflip():
    rows=[dict(parent_id="a", pred_before=0, K_requested=1, K_effective=1, best_valid_distance=.01)]
    with pytest.raises(ValueError, match="INVALID_SAVED_STRICT_FLIP"):
        reduce_parent_prefix(rows, theta=.02, cap=.03)


def test_coverage_is_not_interpolated(tmp_path):
    data=derive(fixture(tmp_path/"inputs"),expected_parents=2)
    assert {r["coverage"] for r in data["figure4"][10]} == {0, .5}
    assert next(r for r in data["figure4"][10] if r["threshold"] == .01)["coverage"] == .5


def test_required_style_and_thin_cpu_entry():
    assert STYLES["Ours"][1] == "s" and STYLES["GlobalGCE"][1] == "x"
    assert STYLES["GCFExplainer"][1] == "^" and STYLES["ComRecGC"][1] == "*"
    repo=Path(__file__).parents[1]
    assert "--version-label" in (repo/"scripts/experiments/replot_bace_gin.py").read_text()
    assert "#SBATCH --gres" not in (repo/"scripts/slurm/replot_bace_gin.sh").read_text()
