from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from src.ablations.gnn import cpu_evaluation as route
from src.eval.bace_frozen_gnn_contracts import BACEParent, atomic_json, atomic_jsonl, sha256_file
from src.eval.mutagenicity_wnode_selector import derive_thresholds, VariantConfig


def candidates(count=20):
    return [{"candidate_id": f"c{i:02d}", "canonical_fragment": "C"} for i in range(count)]


def pairs(ids, rules, *, flip=False):
    return [{"parent_id": p, "candidate_id": c["candidate_id"], "canonical_fragment": "C",
             "applicable": True, "pair_strict_flip": flip, "wnode_distance": 0.1 if flip else None,
             "cf_drop": 0.4 if flip else None} for p in ids for c in rules]


def selector():
    return {"thresholds": derive_thresholds(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
            "variant": VariantConfig("fixture", False, 0.2, 0, 0, 0, False, False),
            "prefix_weights": tuple([1.0] * 20), "local_swap_passes": 0,
            "input_sha256": "a" * 64}


def test_native_and_five_way_common_cohort():
    parents = [BACEParent("p0", "CC", 0, 0), BACEParent("p1", "CCC", 1, 1),
               BACEParent("p2", "CCCC", 1, 2)]
    prediction = {name: [{"predicted_label": 1}, {"predicted_label": 1}, {"predicted_label": 1}]
                  for name in route.BACKBONES}
    prediction["gcn"][2] = {"predicted_label": 0}
    cohort = route.cohort_ids(parents, prediction)
    assert cohort["native"]["gine"] == ["p1", "p2"]
    assert cohort["common"] == ["p1"]
    with pytest.raises(ValueError, match="all five"):
        route.cohort_ids(parents, {"gine": prediction["gine"]})


def test_complete_pair_cartesian_product(tmp_path):
    rules = candidates(2)
    data = pairs(["a", "b"], rules)
    matrix = route.matrix_from_pairs(["a", "b"], rules, data, root=tmp_path, split="calibration")
    assert matrix.distances.shape == (2, 2)
    with pytest.raises(ValueError, match="Incomplete"):
        route.matrix_from_pairs(["a", "b"], rules, data[:-1], root=tmp_path, split="calibration")
    with pytest.raises(ValueError, match="duplicated"):
        route.matrix_from_pairs(["a", "b"], rules, data + data[:1], root=tmp_path, split="calibration")


def test_valid_zero_and_undefined_cost(tmp_path):
    rules = candidates()
    matrix = route.matrix_from_pairs(["p"], rules, pairs(["p"], rules), root=tmp_path, split="test")
    result = route.explanation_metrics(matrix, list(range(20)), selector()["thresholds"])
    assert result["CCRCov@10"] == result["CCRCov@20"] == 0
    assert result["conditional_median_WNode"] is None
    assert result["strict_flip_rate"] == 0
    assert result["applicable_rate"] == 1


def test_selector_rejects_test_and_empty_calibration(tmp_path):
    rules = candidates()
    matrix = route.matrix_from_pairs(["p"], rules, pairs(["p"], rules), root=tmp_path, split="test")
    with pytest.raises(ValueError, match="calibration-only"):
        route.select_calibration(matrix, selector())
    empty = route.matrix_from_pairs([], rules, [], root=tmp_path, split="calibration")
    with pytest.raises(ValueError, match="EMPTY_CALIBRATION"):
        route.select_calibration(empty, selector())


def test_zero_calibration_uses_same_frozen_selector_deterministically(tmp_path):
    rules = candidates(22)
    matrix = route.matrix_from_pairs(["p"], rules, pairs(["p"], rules), root=tmp_path, split="calibration")
    a, trace_a = route.select_calibration(matrix, selector())
    b, trace_b = route.select_calibration(matrix, selector())
    assert a == b == list(range(20))
    assert trace_a == trace_b


def test_structural_universe_not_model_filtered(tmp_path):
    rows = candidates(66)
    universe = tmp_path / "universe.jsonl"
    pool = tmp_path / "pool.jsonl"
    atomic_jsonl(universe, rows)
    atomic_jsonl(pool, [{"predicted_label": i % 2} for i in range(1412)])
    manifest = {"candidate_universe_path": universe.name, "candidate_pool_path": pool.name,
        "files": {p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in (universe, pool)}}
    assert route._candidates(tmp_path, manifest) == rows
    atomic_jsonl(universe, rows[:-1])
    manifest["files"][universe.name] = {"size": universe.stat().st_size, "sha256": sha256_file(universe)}
    with pytest.raises(ValueError, match="66 candidates"):
        route._candidates(tmp_path, manifest)


def test_parent_checkpoint_resume_and_tamper(tmp_path, monkeypatch):
    import src.eval.bace_frozen_gnn_verification as core
    calls = []
    monkeypatch.setattr(core, "_evaluate_rows", lambda parents, rules, **kw:
        (calls.append(parents[0].parent_id) or pairs([parents[0].parent_id], rules), []))
    parents = [BACEParent("p", "CCC", 1, 0)]
    args = dict(oracle=SimpleNamespace(checkpoint_id="b" * 64), featurizer=None,
        distance=None, split="calibration", output=tmp_path / "parents", binding="a" * 64,
        batch_size=2, predictions={"p": {"predicted_label": 1, "probabilities": [0.2, 0.8]}})
    first = route._pairs(parents, candidates(2), **args)
    assert route._pairs(parents, candidates(2), **args) == first
    assert calls == ["p"]
    path = next(p for p in args["output"].glob("*.json") if p.name != "progress.json")
    value = json.loads(path.read_text())
    value["science"]["pair_rows"][0]["applicable"] = False
    atomic_json(path, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        route._pairs(parents, candidates(2), **args)


def test_full_phases_freeze_all_orders_before_test(tmp_path, monkeypatch):
    root, output = tmp_path / "bundle", tmp_path / "result"
    root.mkdir()
    output.mkdir()
    for split in ("calibration", "test"):
        (root / f"{split}.csv").write_text("fixture\n")
    manifest = {"splits": {x: f"{x}.csv" for x in ("calibration", "test")},
                "files": {p.name: {"size": p.stat().st_size,"sha256": sha256_file(p)} for p in root.glob("*.csv")}}
    observed = []
    parent_rows = [BACEParent("p0", "CC", 0, 0), BACEParent("p1", "CCC", 1, 1)]
    def read_parents(path):
        observed.append(path.stem)
        if path.stem == "test":
            frozen = json.loads((output / "CALIBRATION_FREEZE.json").read_text())
            assert frozen["all_five_calibration_orders_frozen"] is True
            assert len(frozen["source_files"]) == 10
        return [BACEParent(f"{path.stem}_{p.parent_id}", p.smiles, p.label, p.source_row_index)
                for p in parent_rows]
    monkeypatch.setattr(route, "_all_parents", read_parents)
    monkeypatch.setattr(route, "_predict", lambda parents, oracle, featurizer, split, batch_size:
        [{"predicted_label": p.label, "probabilities": [0.8,0.2] if p.label == 0 else [0.2,0.8]} for p in parents])
    monkeypatch.setattr(route, "_pairs", lambda parents, rules, **kwargs: pairs([p.parent_id for p in parents], rules))
    models = {name: root / name for name in route.BACKBONES}
    for model_root in models.values():
        model_root.mkdir()
        (model_root / "model.pt").write_bytes(b"tiny-test-checkpoint")
        (model_root / "sha256sums.txt").write_text(sha256_file(model_root / "model.pt") + "  model.pt\n")
    atomic_json(output / "run_manifest.json", {"binding_sha256": "b" * 64,
        "main_matrix_write": False, "execution_commit": "a" * 40,
        "model_roots": {name: str(path) for name, path in models.items()}})
    (output / "writer.lock").touch()
    oracles = {name: SimpleNamespace(model=SimpleNamespace(parameters=lambda: [])) for name in route.BACKBONES}
    result = route._run_phases(root, manifest, output, candidates(), selector(), oracles, models,
        None, None, "b" * 64, 32, lambda labels, probs, num_classes: {"roc_auc": 1.0})
    assert observed == ["calibration", "test"]
    assert result["main_matrix_write"] is False
    assert result["test_loaded_after_calibration_freeze"] is True
    assert (output / "GNN_CORE_SEED7_PASS").exists()
    data = json.loads((output / "gine" / "common" / "explanation_metrics.json").read_text())
    assert data["CCRCov@20"] == 0
    overlap = json.loads((output / "gin" / "common" / "rule_overlap_with_gine.json").read_text())
    assert overlap["exact_rule_jaccard"] == 1
    assert overlap["covered_parent_jaccard"] is None
    assert route.verify_evaluation(output)["state"] == "PASS"
    environment = tmp_path / "env.json"
    atomic_json(environment, {"fixture": True})
    packaged = route.package_evaluation(evaluation_root=output, output_root=tmp_path / "package",
        environment_manifest=environment, execution_commit="a" * 40)
    assert packaged["state"] == "PASS"
    assert packaged["main_matrix_write"] is False
    data["CCRCov@20"] = 1
    atomic_json(output / "gine" / "common" / "explanation_metrics.json", data)
    with pytest.raises(ValueError, match="artifact drift"):
        route.verify_evaluation(output)


def test_package_rejects_nonterminal_or_corrupted_results(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    atomic_json(root / "gnn_seed7_final_audit.json", {"state": "PASS"})
    with pytest.raises(ValueError, match="complete five-backbone"):
        route.package_evaluation(evaluation_root=root, output_root=tmp_path / "package",
            environment_manifest=tmp_path / "env.json", execution_commit="a" * 40)


@pytest.mark.parametrize("pair_seconds,wnode,eligible", [(0.1, [0.2], True), (10000, [1], False), (0.1, [], False)])
def test_cpu_eval_admission_uses_only_train_timing_and_metadata(tmp_path, monkeypatch, pair_seconds, wnode, eligible):
    root = tmp_path / "bundle"
    root.mkdir()
    # Deliberately no calibration/test files: admission may only use counts.
    atomic_json(root / "bundle_manifest.json", {"fixture": True})
    manifest = {"split_row_counts": {"train": 20, "calibration": 10, "test": 10}}
    monkeypatch.setattr(route, "load_bundle", lambda path: (root, manifest))
    models = {}
    for name in route.BACKBONES:
        path = tmp_path / name
        path.mkdir()
        (path / "model.pt").write_bytes(name.encode())
        models[name] = str(path)
    def probe(bundle, checkpoint, output, **kwargs):
        output.mkdir()
        report = {"status": "PASS", "test_loaded": False, "selector_timing_available": True,
            "selector_seconds": 0.01, "parent_count": 2, "seconds_per_pair": pair_seconds,
            "wnode_nonself_diagnostic_seconds": wnode,
            "bundle_manifest_sha256": sha256_file(root / "bundle_manifest.json"),
            "checkpoint_sha256": sha256_file(Path(checkpoint) / "model.pt")}
        atomic_json(output / "verification_timing.json", report)
        return report
    monkeypatch.setattr(route, "benchmark_verification", probe)
    calls = []
    monkeypatch.setattr(route, "run_evaluation", lambda **kw: calls.append(kw) or {"state": "TEST_EVALUATION_CALLED"})
    result = route.evaluate_with_cpu_admission(bundle_root=root, model_roots=models,
        output_root=tmp_path / "output", cpu_threads=1)
    assert bool(calls) == eligible
    if not eligible:
        assert result["state"] == "READY_GNN_GPU_FALLBACK"
        assert result["test_loaded"] is False
        assert result["core_pass"] is False
