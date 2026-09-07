import copy
import math
import numpy as np
import pytest

from src.eval.bace_reach_selector import ReachMasks
from src.experiments.bace_gin_reach_selector import POLICY, objective, select
from src.experiments.bace_gin_reach_v2 import require_freeze, seal, verified
from src.experiments.bace_gin_fixed_pool import prefix_metrics


def masks(d):
    return ReachMasks.from_distances([f"r{i:02}" for i in range(d.shape[1])], d,
        dict(theta_star=.01, cost_cap=.03, merged_thresholds=[
            dict(threshold=.01, weight=1), dict(threshold=.03, weight=3)]))


def test_objective_is_predeclared_not_old_lexicographic():
    m = masks(np.array([[.005, math.inf], [math.inf, .02], [math.inf, 1.]]))
    assert objective(m, [0]) == pytest.approx((.25 + .75) / 3)
    assert objective(m, [1]) == pytest.approx((.25 * 2 + .75 * .75) / 3)


def test_floor_and_nested_prefixes():
    d = np.full((22, 25), math.inf)
    for i in range(22):
        d[i, i] = .005
    m = masks(d)
    s = select(m, list(m.candidate_ids[:20]))
    assert len(s["ordered_rule_ids"]) == 20
    assert len(set(s["ordered_rule_ids"])) == 20
    assert s["new_S10_theta_count"] >= s["old_S10_theta_count"]
    assert s["policy"] == POLICY and s["test_loaded"] is False
    assert not s["global_optimality_claimed"]


def test_fewer_rules_never_pad():
    m = masks(np.array([[.005, math.inf], [math.inf, .02]]))
    result = select(m, list(m.candidate_ids))
    assert len(result["ordered_rule_ids"]) == 2


def test_freeze_gate_never_uses_old_test_or_main_matrix():
    from src.eval.bace_frozen_gnn_contracts import stable_sha256
    spec = {"experiment_id": "test"}
    good = dict(state="CALIBRATION_SELECTOR_FROZEN", test_loaded=False,
        spec_sha256=stable_sha256(spec), policy=POLICY,
        controls=dict(old66_old_selector=[], old66_new_selector=[], adopted2607_new_selector=[]))
    require_freeze(spec, good)
    bad = dict(good, test_loaded=True)
    with pytest.raises(ValueError):
        require_freeze(spec, bad)


def test_immutable_overlay_preserves_old_failures(tmp_path):
    path = tmp_path / "receipt.json"
    seal(path, {"state": "FAIL"})
    with pytest.raises(ValueError):
        seal(path, {"state": "PASS"})
    assert verified(path)["state"] == "FAIL"


def test_fixed_cost_denominator_and_k_semantics():
    parents = ["p0", "p1"]
    rows = [dict(parent_id=p, candidate_id="r", pred_before=1 if p=="p0" else 0,
        pred_after=0 if p=="p0" else None, pair_strict_flip=p=="p0",
        wnode_distance=.005 if p=="p0" else None) for p in parents]
    result = prefix_metrics(parents, ["r"], rows, theta=.01, cap=.03, endpoints=[.01,.03])
    fixed = [r for r in result["prefix_rows"] if r["cohort"]=="fixed141"]
    assert all(r["denominator"]==2 and r["coverage"]==.5 and r["K_effective"]==1 for r in fixed)
    assert all(r["fixed_capped_mean"]==pytest.approx(.0175) for r in fixed)


def test_mask_original66_adapter_unchanged():
    from src.chem.bace_reach_search import deletion_outcomes
    from src.chem.hard_deletion import enumerate_connected_hard_deletions
    candidate = dict(candidate_id="r", canonical_fragment="C")
    a = deletion_outcomes("CCC", candidate, "p")
    b = enumerate_connected_hard_deletions("CCC", "C", parent_id="p", candidate_id="r")
    assert [x.as_dict() for x in a] == [x.as_dict() for x in b]
