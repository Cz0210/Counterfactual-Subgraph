from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ablations.gnn import cpu_evaluation as core
from src.ablations.gnn import scientific_verification as audit
from src.ablations.contracts import canonical_json_sha256
from src.eval.bace_frozen_gnn_contracts import BACEParent, atomic_csv, atomic_json, atomic_jsonl, sha256_file, stable_sha256
from src.eval.mutagenicity_wnode_selector import derive_thresholds, VariantConfig
from src.chem.hard_deletion import CONNECTED_ACTION_SEMANTICS, CONNECTED_MATCH_SELECTION_POLICY


def test_temperature_identity_placeholder_is_not_fitted_evidence():
    kwargs = dict(model_card={}, backbone="gin", validation_sha256="a" * 64,
                  validation_predictions_sha256="b" * 64, validation_count=187)
    fitted = {"temperature": 1.0, "status": "fit", "selection_split": "validation",
              "test_used_for_fit": False, "argmax_invariant": True, "num_classes": 2,
              "num_examples": 187, "validation_csv_sha256": "a" * 64,
              "validation_predictions_sha256": "b" * 64, "nll_before": .4, "nll_after": .4}
    # T=1 alone is not a failure if actually fitted and input-bound.
    audit.require_validation_fitted_temperature(fitted, **kwargs)
    for change in ({"status": "not_fit"}, {"validation_csv_sha256": "c" * 64},
                   {"num_examples": 0}, {"test_used_for_fit": True}, {"argmax_invariant": False}):
        with pytest.raises(ValueError, match="GNN_TEMPERATURE_NOT_VALIDATION_FITTED:gin"):
            audit.require_validation_fitted_temperature({**fitted, **change}, **kwargs)
    with pytest.raises(ValueError, match="model_card_explicitly"):
        audit.require_validation_fitted_temperature(fitted, **{**kwargs,
            "model_card": {"temperature_calibration_fit_on_validation": False}})


def match_fixture():
    before = {"predicted_label": 1, "probabilities": [0.2, 0.8]}
    matches = [{"match_index": i, "match_atom_indices": [i], "oracle_checkpoint_hash": "a" * 64,
        "p_before": [0.2, 0.8], "pred_before": 1, "delete_valid": True, "residual_smiles": "CC",
        "p_after": [0.7, 0.3], "pred_after": 0, "cf_drop": 0.5, "cf_flip": True,
        "teacher_strict_flip": True, "distance_ok": True, "wnode_distance": distance}
        for i, distance in enumerate([0.2, 0.1])]
    for item in matches:
        item.update({"action_semantics_version": CONNECTED_ACTION_SEMANTICS, "residual_connected": True,
            "sanitize_ok": True, "contains_dot": False, "residual_num_components": 1})
    pair = {"oracle_checkpoint_hash": "a" * 64, "source_label": 1, "cf_mode": "strict_flip", "rf_oracle_used": False,
        "pred_before": 1, "p1_before": 0.8, "num_matches": 2, "num_valid_residuals": 2,
        "num_strict_flip_matches": 2, "applicable": True, "pair_strict_flip": True,
        "best_match_index": 1, "best_match_atom_indices": [1], "residual_smiles": "CC",
        "pred_after": 0, "wnode_distance": 0.1, "cf_drop": 0.5,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS, "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY}
    return before, matches, pair


def test_match_minimum_and_per_backbone_flip_replay():
    before, matches, pair = match_fixture()
    audit.verify_matches(pair, matches, checkpoint="a" * 64, before=before)
    pair["best_match_index"] = 0
    with pytest.raises(ValueError, match="replay differs"):
        audit.verify_matches(pair, matches, checkpoint="a" * 64, before=before)
    pair["best_match_index"] = 1
    matches[0]["oracle_checkpoint_hash"] = "b" * 64
    with pytest.raises(ValueError, match="Cross|cross"):
        audit.verify_matches(pair, matches, checkpoint="a" * 64, before=before)


def test_matched_mask_cannot_be_adopted_from_another_classifier():
    before, matches, pair = match_fixture()
    matches[0]["pred_after"] = 1
    matches[0]["p_after"] = [0.1, 0.9]
    matches[0]["cf_drop"] = -0.1
    with pytest.raises(ValueError, match="Strict flip mask"):
        audit.verify_matches(pair, matches, checkpoint="a" * 64, before=before)


def test_prediction_completeness_and_argmax(tmp_path):
    path = tmp_path / "predictions.csv"
    parents = [BACEParent("p", "CC", 1, 0)]
    atomic_csv(path, [{"parent_id": "p", "label": 1, "predicted_label": 1, "probabilities": [0.1, 0.9]}])
    assert audit.predictions(path, parents)[0]["predicted_label"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        audit.predictions(path, parents + [BACEParent("q", "CCC", 1, 1)])
    atomic_csv(path, [{"parent_id": "p", "label": 1, "predicted_label": 0, "probabilities": [0.1, 0.9]}])
    with pytest.raises(ValueError, match="argmax"):
        audit.predictions(path, parents)


def test_explicit_reference_cohort_drift_is_blocked(tmp_path):
    atomic_json(tmp_path / "bundle_manifest.json", {})
    contract = {"source_label": 1, "mode": "proposal_fixed", "source_cohort_definition": "predicted_source_only"}
    contract["contract_sha256"] = canonical_json_sha256(contract)
    path = tmp_path / "reference.json"
    atomic_json(path, contract)
    manifest = {"reference_contract_path": "reference.json", "files": {"reference.json": {
        "size": path.stat().st_size, "sha256": sha256_file(path)}}}
    with pytest.raises(ValueError, match="different source cohort"):
        audit.cohort_contract(tmp_path, manifest)


def full_fixture(tmp_path, monkeypatch, *, empty_test=False):
    root, source = tmp_path / "bundle", tmp_path / "evaluation"
    root.mkdir()
    source.mkdir()
    (source / "writer.lock").touch()
    for split in ("validation", "calibration", "test"):
        (root / f"{split}.csv").write_text("fixture\n")
    reference = {"source_label": 1, "mode": "proposal_fixed"}
    reference["contract_sha256"] = canonical_json_sha256(reference)
    atomic_json(root / "reference.json", reference)
    (root / "molclr.pth").write_bytes(b"fixture-not-a-loaded-model")
    manifest = {"reference_contract_path": "reference.json", "execution_commit": "a" * 40,
        "gine_reference_root": "gine", "molclr_checkpoint_path": "molclr.pth", "wnode_config": {"solver": "exact_emd2"},
        "splits": {x: f"{x}.csv" for x in ("validation", "calibration", "test")},
        "split_row_counts": {"validation": 2},
        "files": {p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in root.iterdir() if p.is_file()}}
    atomic_json(root / "bundle_manifest.json", manifest)
    def parents(path):
        return [BACEParent(f"{path.stem}_{i}", "CCC", i, i) for i in (0, 1)]
    def prediction(parents, *_args):
        if empty_test and parents and parents[0].parent_id.startswith("test_"):
            return [{"predicted_label": 0, "probabilities": [0.8, 0.2]} for p in parents]
        return [{"predicted_label": p.label, "probabilities": [0.8, 0.2] if p.label == 0 else [0.2, 0.8]} for p in parents]
    rules = [{"candidate_id": f"c{i:02d}", "canonical_fragment": "N"} for i in range(66)]
    selector = {"thresholds": derive_thresholds(np.array([0.1, 0.2, 0.3, 0.4, 0.5])),
        "variant": VariantConfig("fixture", False, 0.2, 0, 0, 0, False, False),
        "prefix_weights": tuple([1.0] * 20), "local_swap_passes": 0, "input_sha256": "d" * 64}
    monkeypatch.setattr(core, "_all_parents", parents)
    monkeypatch.setattr(core, "_predict", prediction)
    monkeypatch.setattr(core, "_candidates", lambda *_: rules)
    monkeypatch.setattr(core, "frozen_selector", lambda *_: selector)
    monkeypatch.setattr(audit, "load_bundle", lambda *_: (root, manifest))
    models = {n: root / n for n in core.BACKBONES}
    for name, model in models.items():
        model.mkdir()
        (model / "model.pt").write_bytes(name.encode())
        atomic_json(model / "model_card.json", {"backbone": name, "source_label": 1, "num_classes": 2, "dataset": "bace",
            "selection_split": "validation", "calibration_used_for_model_fit_or_selection": False,
            "test_used_for_model_fit_or_selection": False, "test_loaded_during_training": False, "test_evaluated_during_training": False})
        atomic_csv(model / "validation_predictions.csv", [{"parent_id": "fixture0", "label": 0, "logits": [1, 0]},
                                                           {"parent_id": "fixture1", "label": 1, "logits": [0, 1]}])
        atomic_json(model / "temperature_scaling.json", {"temperature": 1.0, "selection_split": "validation", "test_used_for_fit": False,
            "status": "fit", "argmax_invariant": True, "num_classes": 2, "num_examples": 2,
            "validation_csv_sha256": manifest["files"]["validation.csv"]["sha256"],
            "validation_predictions_sha256": sha256_file(model / "validation_predictions.csv"),
            "nll_before": .2, "nll_after": .2})
        atomic_json(model / "feature_schema.json", {"fixture": True})
        (model / "sha256sums.txt").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in model.iterdir() if p.is_file()))
    import src.oracles.gnn_oracle as oracle_module
    monkeypatch.setattr(oracle_module, "verify_checkpoint_bundle", lambda path: {"model_card": json.loads((path / "model_card.json").read_text())})
    run = {"schema": core.SCHEMA, "execution_commit": "a" * 40, "bundle_sha256": sha256_file(root / "bundle_manifest.json"),
        "oracle_batch_size": 256, "cpu_threads": 1, "checkpoints": {n: sha256_file(p / "model.pt") for n, p in models.items()},
        "temperatures": {n: 1.0 for n in core.BACKBONES}, "selector_sha256": selector["input_sha256"],
        "candidate_universe_sha256": stable_sha256(rules)}
    run["binding_sha256"] = stable_sha256(run)
    run.update({"main_matrix_write": False, "model_roots": {n: str(p) for n, p in models.items()}})
    atomic_json(source / "run_manifest.json", run)
    def compute_pairs(parents, rules, **kwargs):
        output = kwargs["output"]
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for parent in parents:
            current = [{"parent_id": parent.parent_id, "parent_smiles": parent.smiles, "candidate_id": c["candidate_id"],
                "canonical_fragment": c["canonical_fragment"], "applicable": False, "pair_strict_flip": False,
                "wnode_distance": None, "cf_drop": None, "oracle_checkpoint_hash": kwargs["oracle"].checkpoint_id,
                "source_label": 1, "cf_mode": "strict_flip", "rf_oracle_used": False, "pred_before": 1,
                "p1_before": 0.8, "num_matches": 0, "num_valid_residuals": 0, "num_strict_flip_matches": 0,
                "pred_after": None, "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY} for c in rules]
            key = stable_sha256({"parent": asdict(parent), "candidates": rules, "split": kwargs["split"],
                "binding": kwargs["binding"], "checkpoint": kwargs["oracle"].checkpoint_id})
            science = {"pair_rows": current, "match_rows": []}
            atomic_json(output / f"{key}.json", {"binding": key, "science": science, "science_sha256": stable_sha256(science)})
            rows.extend(current)
        return rows
    monkeypatch.setattr(core, "_pairs", compute_pairs)
    oracles = {n: SimpleNamespace(checkpoint_id=run["checkpoints"][n], model=SimpleNamespace(parameters=lambda: [])) for n in core.BACKBONES}
    core._run_phases(root, manifest, source, rules, selector, oracles, models, None, None, run["binding_sha256"],
                     256, oracle_module.classification_metrics)
    for name in core.BACKBONES:
        cal = parents(root / "calibration.csv")
        atomic_csv(source / name / "calibration_classifier_predictions.csv", [
            {"parent_id": p.parent_id, "label": p.label, **r} for p, r in zip(cal, prediction(cal))])
    return root, source


def test_full_offline_replay_and_fresh_overlay(tmp_path, monkeypatch):
    root, source = full_fixture(tmp_path, monkeypatch)
    checked = audit.verify_science(bundle_root=root, evaluation_root=source)
    assert checked["state"] == "PASS"
    assert checked["ot_recomputed"] is False
    assert checked["all_five_validation_temperatures_fitted_and_input_bound"] is True
    assert checked["cohort_contract"]["original_reference_missing_explicit_cohort_definition"] is True
    original = {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    environment = tmp_path / "environment.json"
    atomic_json(environment, {"fixture": True})
    output = tmp_path / "publication"
    result = audit.publish_overlay(bundle_root=root, evaluation_root=source, output_root=output,
        environment_manifest=environment, driver_commit="b" * 40)
    assert result["scientific_engine_commit"] == "a" * 40
    assert result["publication_driver_commit"] == "b" * 40
    assert (output / "gnn_seed7_explanation_native.csv").is_file()
    assert (output / "gnn_seed7_explanation_common.csv").is_file()
    assert (output / "gnn_seed7_table.tex").is_file()
    assert {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()} == original
    archived = result["package"]
    # Import must work even when none of the historical absolute roots exists.
    source.rename(tmp_path / "old-evaluation-moved")
    root.rename(tmp_path / "old-bundle-moved")
    imported = audit.import_verified_bundle(archive_path=archived["bundle"], expected_sha256=archived["sha256"],
                                           output_root=tmp_path / "fresh-import")
    assert imported["historical_hpc_paths_opened"] is False
    assert imported["main_matrix_write"] is False
    assert imported["model_roots"]["gine"].endswith("fresh-import/classifiers/gine")
    assert json.loads((tmp_path / "fresh-import/evaluation/run_manifest.json").read_text())["model_roots"]["gine"] == str(root / "gine")


def test_missing_unreported_output_fails_independent_inventory(tmp_path, monkeypatch):
    root, source = full_fixture(tmp_path, monkeypatch)
    path = source / "gnn_seed7_final_audit.json"
    value = json.loads(path.read_text())
    value["files"].pop("gnn_seed7_classifier_table.csv")
    atomic_json(path, value)
    (source / "GNN_CORE_SEED7_PASS").write_text(sha256_file(path))
    with pytest.raises(ValueError, match="required output inventory"):
        audit.verify_science(bundle_root=root, evaluation_root=source)


def test_missing_all_parent_calibration_prediction_is_not_guessed(tmp_path, monkeypatch):
    root, source = full_fixture(tmp_path, monkeypatch)
    (source / "gine/calibration_classifier_predictions.csv").unlink()
    # Simulate an older producer that never declared this necessary evidence.
    path = source / "gnn_seed7_final_audit.json"
    value = json.loads(path.read_text())
    value["files"].pop("gine/calibration_classifier_predictions.csv", None)
    atomic_json(path, value)
    (source / "GNN_CORE_SEED7_PASS").write_text(sha256_file(path))
    with pytest.raises(ValueError, match="MISSING_CALIBRATION_PREDICTIONS"):
        audit.verify_science(bundle_root=root, evaluation_root=source)


def test_resealed_incorrect_metric_still_fails_scientific_replay(tmp_path, monkeypatch):
    root, source = full_fixture(tmp_path, monkeypatch)
    metric = source / "gine/common/explanation_metrics.json"
    value = json.loads(metric.read_text())
    value["CCRCov@10"] = 0.5
    atomic_json(metric, value)
    final = source / "gnn_seed7_final_audit.json"
    payload = json.loads(final.read_text())
    payload["files"]["gine/common/explanation_metrics.json"] = sha256_file(metric)
    atomic_json(final, payload)
    (source / "GNN_CORE_SEED7_PASS").write_text(sha256_file(final))
    with pytest.raises(ValueError, match="numerical replay"):
        audit.verify_science(bundle_root=root, evaluation_root=source)


def test_empty_test_cohort_is_na_not_fake_zero(tmp_path, monkeypatch):
    root, source = full_fixture(tmp_path, monkeypatch, empty_test=True)
    result = audit.verify_science(bundle_root=root, evaluation_root=source)
    assert result["state"] == "PASS"
    report = json.loads((source / "gine/common/explanation_metrics.json").read_text())
    assert report["state"] == "VALID_EMPTY_COHORT"
    assert report["CCRCov@10"] is None


def test_import_rejects_wrong_transport_hash_before_creating_output(tmp_path):
    archive = tmp_path / "wrong.tar.gz"
    archive.write_bytes(b"not a valid package")
    with pytest.raises(ValueError, match="Transport archive SHA"):
        audit.import_verified_bundle(archive_path=archive, expected_sha256="f" * 64, output_root=tmp_path / "import")
    assert not (tmp_path / "import").exists()
