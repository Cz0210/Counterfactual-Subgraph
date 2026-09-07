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


@pytest.mark.parametrize("flip,expected", [(True,"NO_TRAIN_REACH_GAP_REQUIRING_SUPPLEMENT"),
    (False,"SUPPLEMENTAL_TRAIN_SEARCH_REQUIRED")])
def test_train_gate_uses_complete_train_not_calibration_or_test(tmp_path, monkeypatch, flip, expected):
    from src.experiments import bace_gin_reach_v2 as driver
    from src.eval.bace_frozen_gnn_contracts import stable_sha256
    spec = {"output_root": str(tmp_path)}
    sha = stable_sha256(spec)
    monkeypatch.setattr(driver, "plan", lambda s: {"adopted_candidate_count": 2})
    seal(tmp_path / "adopted2607/train/terminal.json", dict(parent_count=386, spec_sha256=sha))
    def units(s, group, split):
        assert (group, split) == ("adopted2607", "train")
        for i in range(386):
            yield dict(parent_id=str(i), pair_rows=[dict(pred_before=int(i<318),
                pair_strict_flip=flip and i<318) for _ in range(2)])
    monkeypatch.setattr(driver, "_iter_units", units)
    receipt = driver.train_gate(spec)
    assert receipt["state"] == expected
    assert receipt["eligible_count"] == 318
    assert receipt["calibration_used"] is False and receipt["test_used"] is False
    assert len(receipt["uncovered_parent_ids"]) == (0 if flip else 318)


def test_train_gate_refuses_probe(tmp_path, monkeypatch):
    from src.experiments import bace_gin_reach_v2 as driver
    from src.eval.bace_frozen_gnn_contracts import stable_sha256
    spec = {"output_root": str(tmp_path)}
    monkeypatch.setattr(driver, "plan", lambda s: {"adopted_candidate_count": 2})
    seal(tmp_path / "adopted2607/train/terminal.json", dict(parent_count=2, spec_sha256=stable_sha256(spec)))
    with pytest.raises(ValueError, match="FULL_386"):
        driver.train_gate(spec)


def test_same_source_supplement_retains_pool_and_rejects_test_guidance(tmp_path):
    from src.experiments import bace_gin_reach_v2 as driver
    from src.eval.bace_frozen_gnn_contracts import atomic_json, atomic_jsonl, sha256_file, stable_sha256
    old = [dict(candidate_id="a"), dict(candidate_id="b")]
    source = dict(gin_files={"model.pt":"frozen"})
    source_path = tmp_path / "source.json"
    atomic_json(source_path, source)
    pool = tmp_path / "pool.jsonl"
    atomic_jsonl(pool, old)
    data = dict(state="NO_SUPPLEMENT_REQUIRED", source_spec_sha256=stable_sha256(source),
        old2607_content_unchanged=True, candidate_count=2, candidate_universe_sha256=sha256_file(pool),
        test_loaded=False, calibration_loaded=False)
    path = tmp_path / "receipt.json"
    seal(path, data)
    spec = dict(gin_files=source["gin_files"],
        adopted_source_spec=dict(path=str(source_path),sha256=sha256_file(source_path)),
        supplement=dict(receipt=dict(path=str(path),sha256=sha256_file(path)), candidate_file=str(pool)))
    assert driver.supplement_binding(spec, old)[0] == old
    with pytest.raises(ValueError, match="SOURCE_ROWS"):
        driver.supplement_binding(spec, list(reversed(old)))
    bad = tmp_path / "bad.json"
    seal(bad, dict(data, test_loaded=True))
    spec["supplement"]["receipt"] = dict(path=str(bad),sha256=sha256_file(bad))
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        driver.supplement_binding(spec, old)


def test_final_extended_pool_control_not_mislabeled_2607():
    from src.eval.bace_frozen_gnn_contracts import stable_sha256
    spec = {"new_control_name":"expanded_pool_new_selector"}
    freeze = dict(state="CALIBRATION_SELECTOR_FROZEN",test_loaded=False,
        spec_sha256=stable_sha256(spec),policy=POLICY,
        controls=dict(old66_old_selector=[],old66_new_selector=[],expanded_pool_new_selector=[]))
    require_freeze(spec, freeze)
    freeze["controls"]["adopted2607_new_selector"] = freeze["controls"].pop("expanded_pool_new_selector")
    with pytest.raises(ValueError):
        require_freeze(spec, freeze)
