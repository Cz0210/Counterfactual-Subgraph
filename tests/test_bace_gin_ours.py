from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.experiments import bace_gin_ours as module
from src.eval.bace_frozen_gnn_contracts import BACEParent


def test_test_blocked_before_opening_any_bundle(monkeypatch):
    monkeypatch.setattr(module, "original_bundle", lambda spec: pytest.fail("opened bundle"))
    with pytest.raises(ValueError, match="TEST_REQUIRES"):
        module.fixed_source_parents({}, "test")


def test_fixed_calibration_keeps_all_true_source_parents(monkeypatch, tmp_path):
    from src.ablations.gnn import cpu_training
    parents = [SimpleNamespace(parent_id=str(i), label=1) for i in range(66)]
    monkeypatch.setattr(module, "original_bundle", lambda spec: (tmp_path, {"splits": {"calibration": "cal.csv"}}))
    monkeypatch.setattr(cpu_training, "bundle_file", lambda *args: tmp_path / "cal.csv")
    monkeypatch.setattr(module, "load_bace_parents", lambda *a, **k: parents)
    assert module.fixed_source_parents({}, "calibration") is parents
    parents.pop()
    with pytest.raises(ValueError, match="FIXED_BASE_COHORT_CHANGED"):
        module.fixed_source_parents({}, "calibration")


def test_original_pool_rejects_reach_or_other_source(monkeypatch, tmp_path):
    manifest = {"files": {"pool": {"sha256": "bad"}}, "candidate_universe_path": "pool"}
    monkeypatch.setattr(module, "original_bundle", lambda spec: (tmp_path, manifest))
    with pytest.raises(ValueError, match="OURS_REQUIRES_ORIGINAL66"):
        module.load_original_pool({"expected_original_universe_sha256": "bad"})


def test_evaluate_parent_does_not_drop_gin_predicted_zero(monkeypatch):
    from src.eval import bace_frozen_gnn_verification as original
    oracle = SimpleNamespace(backbone="gin", source_label=1, checkpoint_id="new-gin", temperature=1.1)
    parent = SimpleNamespace(parent_id="base1", label=1, smiles="CC")
    seen = {}
    def evaluate(parents, candidates, **kwargs):
        seen.update(kwargs)
        assert parents == [parent]
        return [{"pred_before": 0, "pair_strict_flip": False}], []
    monkeypatch.setattr(original, "_evaluate_rows", evaluate)
    pairs, _ = module.evaluate_parent(parent, [{"candidate_id": "c"}], oracle, None, None,
        "calibration", parent_prediction={"predicted_label": 0, "probabilities": [.9, .1],
            "checkpoint_id": "new-gin", "backbone": "gin", "temperature": 1.1})
    assert len(pairs) == 1 and pairs[0]["pred_before"] == 0
    assert seen["oracle_checkpoint_id"] == "new-gin"
    assert seen["parent_prediction_cache"]["base1"]["pred_before"] == 0


def test_rejects_gine_parent_prediction_cache_before_evaluation():
    with pytest.raises(ValueError, match="NOT_BOUND_TO_CURRENT_GIN"):
        module.evaluate_parent(SimpleNamespace(parent_id="p", label=1, smiles="CC"), [],
            SimpleNamespace(backbone="gin", source_label=1, checkpoint_id="gin", temperature=1.1),
            None, None, "calibration", parent_prediction={"checkpoint_id": "gine", "backbone": "gine"})


def test_distance_failure_cannot_be_zero_result(monkeypatch):
    from src.eval import bace_frozen_gnn_verification as original
    monkeypatch.setattr(original, "_evaluate_rows", lambda *a, **k: ([{}], [{"cf_flip": True, "distance_ok": False}]))
    with pytest.raises(ValueError, match="STRICT_FLIP_RAW_DISTANCE_FAILURE"):
        module.evaluate_parent(SimpleNamespace(label=1), [dict(candidate_id="c")],
            SimpleNamespace(backbone="gin", source_label=1, checkpoint_id="gin"), None, None, "train")


def test_original_selector_rejects_fixed_old_winner():
    matrix = SimpleNamespace(manifest={"split": "calibration", "test_loaded": False})
    with pytest.raises(ValueError, match="ALL_FOUR_VARIANTS_NOT_OLD_WINNER"):
        module.select_calibration(matrix, {"variant": "A4"})


def test_original_selector_rejects_test_matrix():
    matrix = SimpleNamespace(manifest={"split": "test", "test_loaded": True})
    with pytest.raises(ValueError, match="CALIBRATION_ONLY"):
        module.select_calibration(matrix, {})


def test_adoption_actual_fit_contract(monkeypatch, tmp_path):
    root = tmp_path / "gin"
    root.mkdir()
    sha = {"model.pt": "model", "temperature_scaling.json": "temperature", "feature_schema.json": "schema"}
    spec = {"gin_root": str(root), "gin_files": sha,
        "classifier_adoption": {"root": str(tmp_path),
            "acceptance": {"relative_path": "acceptance.json", "sha256": "acceptance"},
            "independent_science_replay": {"relative_path": "replay.json", "sha256": "replay"}}}
    fit = dict(status="fit", fit_split="validation", num_examples=187, test_used=False,
        calibration_used=False, model_weight_sha="model", fitted_temperature=1.0,
        optimizer_parameter_count=1, model_optimizer_parameters=0, sample_ids=list(range(187)))
    docs = {"acceptance.json": dict(state="GNN_CORE_SEED7_CORRECTED_PASS", all_weights_unchanged=True,
                independent_science_replay_sha256="replay"),
        "replay.json": dict(state="PASS", all_five_validation_temperatures_fitted_and_input_bound=True,
                models={"gin": dict(root=str(root), model_sha256="model", temperature_sha256="temperature")}),
        "temperature_scaling.json": dict(status="fit", temperature=1.0, num_examples=187,
                selection_split="validation", test_used_for_fit=False),
        "feature_schema.json": {}, "temperature_fit_receipt.json": fit}
    monkeypatch.setattr(module, "_bound_json", lambda p, s: docs[p.name])
    monkeypatch.setattr(module, "read_json", lambda p: dict(fit_receipt_sha256="fit",
        weights_changed=False, weight_sha256="model", temperature_sha256="temperature"))
    assert module.validate_gin_adoption(spec)["temperature"] == 1.0
    fit["status"] = "not_fit"
    with pytest.raises(ValueError, match="ACTUAL_VALIDATION187"):
        module.validate_gin_adoption(spec)


def test_train_timing_has_no_calibration_or_test_loader():
    import inspect
    source = inspect.getsource(module.train_only_timing)
    assert 'fixed_source_parents(spec, "train")[:2]' in source
    assert "select_calibration(" not in source
    assert "fit_temperature" not in source


def test_slurm_cpu_and_no_gpu():
    script = (Path(__file__).resolve().parents[1] / "scripts/slurm/time_bace_gin_ours.sh").read_text()
    assert "--partition=intel" in script and "--gres=" not in script
    assert "export CUDA_VISIBLE_DEVICES=" in script
    assert "--config configs/hpc.yaml" in script
    assert script.index("source ~/.bashrc") < script.index("set -u")


def test_native_graph_pair_reuses_raw_cost_without_fake_delete_key(monkeypatch, tmp_path):
    from src.ablations.gnn import reach_raw_distance_reuse as raw
    def init(self, delegate, *, index, **kwargs):
        self.delegate, self.index, self.used, self.fresh, self.local = delegate, index, [], 0, {}
    monkeypatch.setattr(raw.VerifiedRawGraphDistance, "__init__", init)
    monkeypatch.setattr(raw, "graph_key", lambda p, c, h: (p + ":" + c, p, c))
    delegate = SimpleNamespace(distance=lambda p, c: {"ok": True, "distance": .7, "cache_hit": False})
    index = {"self_sha256": "index", "raw_contract_sha256": "contract",
        "graph_costs": {"CC:CO": {"distance": .4, "source_records": ["actual-native-or-deletion-pair"]}}}
    provider = module.with_native_graph_distance(delegate, index=index, current_raw_contract={}, repo=tmp_path)
    assert provider.distance("CC", "CO")["distance"] == .4
    assert "current_action_context" not in provider.used[0]
    assert provider.distance("CC", "CN")["distance"] == .7
    assert provider.fresh == 1
    provider.distance("CC", "CN")
    assert provider.fresh == 1


def test_four_variant_replay_matches_original_b12_per_variant(tmp_path):
    import numpy as np
    from src.ablations.gnn.cpu_evaluation import matrix_from_pairs
    from src.eval.mutagenicity_wnode_selector import (
        derive_thresholds, preregistered_variant_configs, build_candidate_chemistry,
        build_coverage_redundancy_matrix, _run_one_variant, choose_variant,
    )
    rules = [{"candidate_id": f"c{i:02}", "canonical_fragment": "C"*(i%3+1)} for i in range(22)]
    pairs = [dict(parent_id=p, candidate_id=c["candidate_id"], applicable=True,
        pair_strict_flip=True, wnode_distance=.1+(i%9)*.03+(j%2)*.07, cf_drop=.4)
        for j,p in enumerate(["p0","p1"]) for i,c in enumerate(rules)]
    matrix = matrix_from_pairs(["p0","p1"], rules, pairs, root=tmp_path, split="calibration")
    thresholds = derive_thresholds(np.array([p["wnode_distance"] for p in pairs]))
    variants = preregistered_variant_configs()
    selector = dict(variants={v.name:v for v in variants}, thresholds=thresholds,
        prefix_weights=tuple([1.]*10+[.5]*10), local_swap_passes=2, input_sha256="original-config")
    sequence, receipt = module.select_calibration(matrix, selector)
    chemistry = build_candidate_chemistry(rules, size_normalization_rows=rules)
    redundancy = build_coverage_redundancy_matrix(matrix.distances, thresholds.levels)
    expected_rows = []
    for variant in variants:
        comparison, expected = _run_one_variant(variant, matrix=matrix, chemistry=chemistry,
            thresholds=thresholds, prefix_weights=selector["prefix_weights"], top_k=20,
            table_k=10, local_swap_passes=2, coverage_redundancy_matrix=redundancy,
            output_dir=tmp_path / "original")
        assert receipt["variant_sequences"][variant.name] == expected
        expected_rows.append(comparison)
    assert receipt["variant_comparison"] == expected_rows
    assert receipt["selected_metrics"] == choose_variant(expected_rows)
    assert sequence == receipt["variant_sequences"][receipt["selected_variant"]]
