"""Offline, row-replaying acceptance of completed BACE GNN ablations.

This module never invokes an oracle, trainer, MolCLR encoder, or OT solver.
It reopens frozen predictions, match records, selections and metrics and writes
only a fresh publication overlay. A source producer's PASS is not sufficient.
"""
from __future__ import annotations

import ast
import csv
from dataclasses import asdict
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np

from src.ablations.gnn import cpu_evaluation as core
from src.ablations.contracts import canonical_json_sha256
from src.ablations.gnn.cpu_training import bundle_file, load_bundle
from src.eval.bace_frozen_gnn_contracts import (
    atomic_csv, atomic_json, read_json, read_jsonl, sha256_file, stable_sha256,
)

ELIGIBILITY = "true_label == source_label and pred_before == source_label"
MODES = ("native", "common")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equal(actual: Any, expected: Any, where: str = "value") -> None:
    """Compare deterministic replays, tolerating only floating-point roundoff."""
    if isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping) and set(actual) == set(expected), f"{where}: keys differ")
        for key, value in expected.items():
            equal(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, (tuple, list)):
        _require(isinstance(actual, (tuple, list)) and len(actual) == len(expected), f"{where}: length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, f"{where}[{index}]")
    elif isinstance(expected, float):
        _require(isinstance(actual, (int, float)) and math.isfinite(actual)
                 and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
                 f"{where}: numerical replay differs")
    else:
        _require(actual == expected, f"{where}: replay differs")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _probabilities(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    values = [float(x) for x in value]
    _require(len(values) == 2 and all(math.isfinite(x) and 0 <= x <= 1 for x in values)
             and math.isclose(sum(values), 1, abs_tol=1e-6), "Invalid sealed prediction probabilities")
    return values


def predictions(path: Path, parents: Sequence[Any]) -> list[dict[str, Any]]:
    _require(path.is_file(), f"BLOCKED_MISSING_CALIBRATION_PREDICTIONS: {path}")
    rows = _csv(path)
    by_id = {row["parent_id"]: row for row in rows}
    _require(len(by_id) == len(rows) and set(by_id) == {p.parent_id for p in parents},
             "Prediction source cohort is duplicated, incomplete, or extraneous")
    result = []
    for parent in parents:
        row = by_id[parent.parent_id]
        probs = _probabilities(row["probabilities"])
        pred = int(row["predicted_label"])
        _require(int(row["label"]) == parent.label and pred == int(np.argmax(probs)),
                 "Prediction label/argmax differs from exact split")
        result.append({"predicted_label": pred, "probabilities": probs})
    return result


def cohort_contract(bundle_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    reference_path = bundle_file(bundle_root, manifest, manifest["reference_contract_path"])
    reference = read_json(reference_path)
    _require(reference.get("contract_sha256") == canonical_json_sha256({
        k: v for k, v in reference.items() if k != "contract_sha256"}), "Reference contract self hash differs")
    _require(reference.get("source_label") == 1 and reference.get("mode") == "proposal_fixed",
             "Reference source label/method differs")
    existing = reference.get("source_cohort_definition")
    _require(existing in (None, ELIGIBILITY), "Reference explicitly requires a different source cohort")
    framework = Path(__file__).with_name("framework.py")
    _require(f'"native_eligibility": "{ELIGIBILITY}"' in framework.read_text(),
             "Original ablation framework eligibility evidence differs")
    return {"source_label": 1, "native_eligibility": ELIGIBILITY,
        "common_definition": "ordered_intersection_of_all_native_parent_ids_per_split",
        "reference_contract_sha256": sha256_file(reference_path),
        "bundle_manifest_sha256": sha256_file(bundle_root / "bundle_manifest.json"),
        "original_reference_missing_explicit_cohort_definition": existing is None,
        "definition_preserved_from_original_framework": True,
        "framework_source_sha256": sha256_file(framework), "scientific_definition_changed": False}


def verify_matches(pair: Mapping[str, Any], matches: Sequence[Mapping[str, Any]],
                   *, checkpoint: str, before: Mapping[str, Any]) -> None:
    """Replay strict flip and existential best-match from already-computed rows."""
    _require(pair.get("oracle_checkpoint_hash") == checkpoint, "Cross-backbone pair checkpoint")
    _require(pair.get("source_label") == 1 and pair.get("cf_mode") == "strict_flip"
             and pair.get("rf_oracle_used") is False, "Pair scientific semantics differ")
    _require(pair.get("pred_before") == before["predicted_label"], "Pair parent prediction differs")
    equal(float(pair["p1_before"]), before["probabilities"][1], "pair.p1_before")
    seen = set()
    finite = []
    from src.chem.hard_deletion import CONNECTED_ACTION_SEMANTICS, CONNECTED_MATCH_SELECTION_POLICY
    _require(pair.get("action_semantics_version") == CONNECTED_ACTION_SEMANTICS
             and pair.get("match_selection_policy") == CONNECTED_MATCH_SELECTION_POLICY,
             "Pair deletion/minimum-match semantics differ")
    for match in matches:
        key = int(match["match_index"])
        _require(key not in seen and match["oracle_checkpoint_hash"] == checkpoint,
                 "Duplicate match or cross-backbone match checkpoint")
        seen.add(key)
        equal(match["p_before"], before["probabilities"], "match.p_before")
        _require(match["pred_before"] == before["predicted_label"], "Match parent prediction differs")
        valid = bool(match["delete_valid"] and match.get("residual_smiles"))
        _require(match.get("action_semantics_version") == CONNECTED_ACTION_SEMANTICS,
                 "Match deletion semantics differ")
        if valid:
            _require(match.get("residual_connected") is True and match.get("sanitize_ok") is True
                     and match.get("contains_dot") is False and match.get("residual_num_components") == 1,
                     "Valid deletion does not preserve connected chemical residual")
        flip = False
        if valid:
            after = _probabilities(match["p_after"])
            _require(match["pred_after"] == int(np.argmax(after)), "Match post-deletion argmax differs")
            flip = match["pred_before"] == 1 and match["pred_after"] != 1
            equal(float(match["cf_drop"]), before["probabilities"][1] - after[1], "match.cf_drop")
        _require(match["cf_flip"] is flip and match["teacher_strict_flip"] is flip,
                 "Strict flip mask differs from this backbone's predictions")
        if flip:
            _require(match.get("distance_ok") is True and math.isfinite(float(match["wnode_distance"]))
                     and float(match["wnode_distance"]) >= 0, "Strict flip lacks finite exact WNode")
            finite.append(match)
    _require(pair["num_matches"] == len(matches)
             and pair["num_valid_residuals"] == sum(bool(x["delete_valid"]) for x in matches)
             and pair["num_strict_flip_matches"] == len(finite), "Match counters differ")
    _require(pair["applicable"] is bool(matches) and pair["pair_strict_flip"] is bool(finite),
             "Pair applicability/flip aggregation differs")
    best = min(finite, key=lambda r: (float(r["wnode_distance"]), -float(r["cf_drop"]),
                                    tuple(r["match_atom_indices"]))) if finite else None
    if best:
        for field, source in (("best_match_index", "match_index"), ("best_match_atom_indices", "match_atom_indices"),
                              ("residual_smiles", "residual_smiles"), ("pred_after", "pred_after"),
                              ("wnode_distance", "wnode_distance"), ("cf_drop", "cf_drop")):
            equal(pair[field], best[source], f"pair.{field}")
    else:
        _require(pair.get("wnode_distance") is None and pair.get("pred_after") is None,
                 "Non-flip pair contains a selected match")


def pair_checkpoints(source: Path, name: str, split: str, parents: Sequence[Any],
        candidates: list[dict[str, Any]], before: Mapping[str, Any], run: Mapping[str, Any]) -> tuple[list, dict]:
    pairs = read_jsonl(source / name / split / "pair_matrix.jsonl")
    expected = {}
    evidence = {}
    for parent in parents:
        key = stable_sha256({"parent": asdict(parent), "candidates": candidates, "split": split,
            "binding": run["binding_sha256"], "checkpoint": run["checkpoints"][name]})
        path = source / name / split / "parents" / f"{key}.json"
        state = read_json(path)
        _require(state.get("binding") == key and state.get("science_sha256") == stable_sha256(state["science"]),
                 "Parent checkpoint scientific hash/binding differs")
        science = state["science"]
        fragments = {row["candidate_id"]: row["canonical_fragment"] for row in candidates}
        match_groups = {str(row["candidate_id"]): [] for row in candidates}
        for match in science["match_rows"]:
            _require(match["parent_id"] == parent.parent_id and match["candidate_id"] in match_groups
                     and match.get("parent_smiles") == parent.smiles
                     and match.get("canonical_fragment") == fragments[match["candidate_id"]],
                     "Match outside frozen parent/candidate space")
            match_groups[match["candidate_id"]].append(match)
        rows = science["pair_rows"]
        core.matrix_from_pairs([parent.parent_id], candidates, rows, root=source, split=split)
        for row in rows:
            _require(row["parent_smiles"] == parent.smiles
                     and row["canonical_fragment"] == fragments[row["candidate_id"]], "Pair graph identity differs")
            verify_matches(row, match_groups[row["candidate_id"]], checkpoint=run["checkpoints"][name],
                           before=before[parent.parent_id])
            expected[(row["parent_id"], row["candidate_id"])] = row
        evidence[str(path.relative_to(source))] = sha256_file(path)
    actual = {(row["parent_id"], row["candidate_id"]): row for row in pairs}
    _require(len(actual) == len(pairs) and set(actual) == set(expected), "Pair matrix is not the complete checkpoint union")
    for key, row in actual.items():
        equal(row, expected[key], "pair_matrix")
    return pairs, evidence


def required_files() -> set[str]:
    files = {"run_manifest.json", "CALIBRATION_FREEZE.json", "calibration_cohorts.json", "test_cohorts.json",
        "gnn_seed7_classifier_table.csv", "gnn_seed7_explanation_table.csv", "gnn_seed7_rule_stability.csv",
        "gnn_seed7_common_cohort.csv", "gnn_seed7_classifier_table.tex", "gnn_seed7_explanation_table.tex"}
    for name in core.BACKBONES:
        files.update(f"{name}/{item}" for item in ("classifier_metrics.json", "test_classifier_predictions.csv",
            "verification_manifest.json", "calibration/pair_matrix.jsonl", "test/pair_matrix.jsonl"))
        for mode in MODES:
            files.update(f"{name}/{mode}/{item}" for item in ("selected_rules.json", "explanation_metrics.json",
                "cohort_manifest.json", "rule_overlap_with_gine.json"))
    return files


def compare_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Use the same scalar CSV representation; numeric results are not rounded."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    buffer.seek(0)
    equal(_csv(path), list(csv.DictReader(buffer)), f"published CSV {path.name}")


def verify_science(*, bundle_root: str | Path, evaluation_root: str | Path,
                   calibration_prediction_root: str | Path | None = None) -> dict[str, Any]:
    """Read only; classifier predictions/OT are never recomputed by this verifier."""
    root, manifest = load_bundle(bundle_root)
    source = Path(evaluation_root).resolve(strict=True)
    audit = core.verify_evaluation(source)
    _require(required_files() <= set(audit["files"]), "Independent audit: required output inventory incomplete")
    run = read_json(source / "run_manifest.json")
    freeze = read_json(source / "CALIBRATION_FREEZE.json")
    binding_fields = ("schema", "execution_commit", "bundle_sha256", "oracle_batch_size", "cpu_threads",
                      "checkpoints", "temperatures", "selector_sha256", "candidate_universe_sha256")
    _require(run["binding_sha256"] == stable_sha256({key: run[key] for key in binding_fields}),
             "Scientific run binding does not hash its actual inputs")
    _require(freeze.get("binding_sha256") == run["binding_sha256"], "Freeze/run binding differs")
    _require(run["bundle_sha256"] == sha256_file(root / "bundle_manifest.json"), "Wrong original input bundle")
    _require(run["execution_commit"] == manifest["execution_commit"], "Wrong scientific engine commit")
    candidates = core._candidates(root, manifest)
    selector = core.frozen_selector(root, manifest)
    _require(run["candidate_universe_sha256"] == stable_sha256(candidates)
             and run["selector_sha256"] == selector["input_sha256"], "Candidate/selector binding differs")
    contract = cohort_contract(root, manifest)
    from src.oracles.gnn_oracle import verify_checkpoint_bundle
    models = {}
    _require(set(run["model_roots"]) == set(run["checkpoints"]) == set(run["temperatures"]) == set(core.BACKBONES),
             "Model inventory is not the complete five-backbone set")
    for name in core.BACKBONES:
        model_root = Path(run["model_roots"][name]).resolve(strict=True)
        checked = verify_checkpoint_bundle(model_root)
        card = checked["model_card"]
        _require(card["backbone"] == name and card["source_label"] == 1 and card["num_classes"] == 2
                 and card["dataset"] == "bace" and sha256_file(model_root / "model.pt") == run["checkpoints"][name],
                 "Model bundle/checkpoint differs from evaluated model")
        temperature = read_json(model_root / "temperature_scaling.json")
        _require(card.get("selection_split") == "validation"
                 and all(card.get(field) is False for field in ("calibration_used_for_model_fit_or_selection",
                    "test_used_for_model_fit_or_selection", "test_loaded_during_training", "test_evaluated_during_training"))
                 and temperature.get("selection_split") == "validation" and temperature.get("test_used_for_fit") is False,
                 "Classifier or temperature split-isolation provenance differs")
        equal(float(temperature["temperature"]), float(run["temperatures"][name]), "classifier temperature")
        models[name] = {"root": str(model_root), "model_sha256": run["checkpoints"][name],
            "temperature_sha256": sha256_file(model_root / "temperature_scaling.json"),
            "model_card_sha256": sha256_file(model_root / "model_card.json")}
    _require(Path(models["gine"]["root"]) == (root / manifest["gine_reference_root"]).resolve(strict=True),
             "GINE is not the exact frozen reference bundle")
    evidence, results, selections = {}, {}, {}
    prediction_source = source if calibration_prediction_root is None else Path(calibration_prediction_root).resolve(strict=True)
    # Fully reopen/check the ten frozen selections before parsing held-out rows.
    for name in core.BACKBONES:
        selections[name] = {}
        for mode in MODES:
            rules = read_json(source / name / mode / "selected_rules.json")
            _require(rules.get("binding_sha256") == run["binding_sha256"] and rules.get("split") == "calibration"
                     and rules.get("backbone") == name and rules.get("cohort") == mode, "Frozen selection contract differs")
            equal(rules["selector"], asdict(selector["variant"]), "frozen selector")
            equal(rules["thresholds"], selector["thresholds"].to_dict(), "frozen thresholds")
            by_id = {row["candidate_id"]: row for row in candidates}
            _require(set(rules["candidate_ids"]) <= set(by_id), "Selected rule outside fixed candidate universe")
            expected = [by_id[value] for value in rules["candidate_ids"]]
            equal(rules["candidates"], expected, "frozen candidate identity")
            equal(rules["selected_rules"], [{**row, "rule_id": row["candidate_id"], "fragment": row["canonical_fragment"]}
                                           for row in expected], "frozen rule identity")
            selections[name][mode] = rules
    cal = core._all_parents(bundle_file(root, manifest, manifest["splits"]["calibration"]))
    cal_predictions = {n: predictions(prediction_source / n / "calibration_classifier_predictions.csv", cal) for n in core.BACKBONES}
    cohorts = core.cohort_ids(cal, cal_predictions)
    equal(read_json(source / "calibration_cohorts.json"), cohorts, "calibration cohort")
    for name in core.BACKBONES:
        native = [p for p in cal if p.parent_id in set(cohorts["native"][name])]
        before = dict(zip((p.parent_id for p in cal), cal_predictions[name], strict=True))
        rows, checked = pair_checkpoints(source, name, "calibration", native, candidates, before, run)
        evidence.update(checked)
        for mode in MODES:
            ids = cohorts["native"][name] if mode == "native" else cohorts["common"]
            matrix = core.matrix_from_pairs(ids, candidates, [r for r in rows if r["parent_id"] in set(ids)], root=source, split="calibration")
            order, trace = core.select_calibration(matrix, selector)
            equal(selections[name][mode]["candidate_ids"], [candidates[i]["candidate_id"] for i in order], "global selector replay")
            equal(selections[name][mode]["trace"], trace, "calibration selector trace")
    # Selection replay uses calibration only; test cannot influence any order.
    test = core._all_parents(bundle_file(root, manifest, manifest["splits"]["test"]))
    _require(not ({p.parent_id for p in test} & {p.parent_id for p in cal}), "Calibration/test overlap")
    test_predictions = {n: predictions(source / n / "test_classifier_predictions.csv", test) for n in core.BACKBONES}
    test_cohorts = core.cohort_ids(test, test_predictions)
    equal(read_json(source / "test_cohorts.json"), test_cohorts, "test cohort")
    from src.oracles.gnn_oracle import classification_metrics
    classifier_table, explanation_table = [], []
    for name in core.BACKBONES:
        probs = np.asarray([r["probabilities"] for r in test_predictions[name]])
        labels = np.asarray([p.label for p in test])
        replay = classification_metrics(labels, probs, num_classes=2)
        replay["NLL"] = float(-np.mean(np.log(np.clip(probs[np.arange(len(test)), labels], 1e-12, 1))))
        report = read_json(source / name / "classifier_metrics.json")
        for key, value in replay.items():
            equal(report[key], value, f"{name}.classifier.{key}")
        classifier_table.append({k: v for k, v in report.items() if not isinstance(v, (list, dict))})
        chosen = set().union(*(set(selections[name][m]["candidate_ids"]) for m in MODES))
        selected = [c for c in candidates if c["candidate_id"] in chosen]
        native = [p for p in test if p.parent_id in set(test_cohorts["native"][name])]
        before = dict(zip((p.parent_id for p in test), test_predictions[name], strict=True))
        rows, checked = pair_checkpoints(source, name, "test", native, selected, before, run)
        evidence.update(checked)
        verification = read_json(source / name / "verification_manifest.json")
        _require(verification.get("binding_sha256") == run["binding_sha256"]
                 and verification.get("backbone") == name and verification.get("test_loaded_after_freeze") is True
                 and verification.get("calibration_freeze_sha256") == sha256_file(source / "CALIBRATION_FREEZE.json")
                 and verification.get("calibration_pair_matrix_sha256") == sha256_file(source / name / "calibration" / "pair_matrix.jsonl")
                 and verification.get("test_pair_matrix_sha256") == sha256_file(source / name / "test" / "pair_matrix.jsonl"),
                 "Verification matrix/freeze provenance differs")
        results[name] = {}
        for mode in MODES:
            ids = test_cohorts["native"][name] if mode == "native" else test_cohorts["common"]
            order = selections[name][mode]["candidate_ids"]
            selected = [c for c in candidates if c["candidate_id"] in set(order)]
            matrix = core.matrix_from_pairs(ids, selected, [r for r in rows if r["parent_id"] in set(ids)
                and r["candidate_id"] in set(order)], root=source, split="test")
            replay = core.explanation_metrics(matrix, [matrix.candidate_index[x] for x in order], selector["thresholds"])
            equal(read_json(source / name / mode / "explanation_metrics.json"), replay, "explanation metric replay")
            cohort = read_json(source / name / mode / "cohort_manifest.json")
            equal(cohort["test_parent_ids"], ids, "reported test cohort")
            equal(cohort["calibration_parent_ids"], cohorts["native"][name] if mode == "native" else cohorts["common"], "reported calibration cohort")
            _require(cohort["freeze_sha256"] == sha256_file(source / "CALIBRATION_FREEZE.json")
                     and cohort["split_sha256"] == sha256_file(bundle_file(root, manifest, manifest["splits"]["test"])),
                     "Cohort split/freeze hashes differ")
            if replay["prefix_rows"]:
                compare_csv(source / name / mode / "figure3_coverage_vs_k.csv", replay["prefix_rows"])
                compare_csv(source / name / mode / "figure4_coverage_vs_threshold.csv", replay["threshold_rows"])
                compare_csv(source / name / mode / "table2_k10.csv", [replay["prefix_rows"][9]])
            explanation_table.append({"backbone": name, "cohort": mode,
                **{k: v for k, v in replay.items() if not isinstance(v, (dict, list))}})
            results[name][mode] = replay
    from src.eval.rule_stability import compare_frozen_rule_selections
    stability_table = []
    for name in core.BACKBONES:
        for mode in MODES:
            replay = compare_frozen_rule_selections(source / "gine" / mode / "selected_rules.json",
                                                   source / name / mode / "selected_rules.json")
            left, right = (set(results[n][mode]["covered_parent_ids"]) for n in ("gine", name))
            replay["covered_parent_jaccard"] = len(left & right) / len(left | right) if left | right else None
            equal(read_json(source / name / mode / "rule_overlap_with_gine.json"), replay, "rule stability replay")
            stability_table.append({"backbone": name, "cohort": mode, "exact_rule_jaccard": replay["exact_rule_jaccard"],
                                    "covered_parent_jaccard": replay["covered_parent_jaccard"]})
    compare_csv(source / "gnn_seed7_classifier_table.csv", classifier_table)
    compare_csv(source / "gnn_seed7_explanation_table.csv", explanation_table)
    compare_csv(source / "gnn_seed7_rule_stability.csv", stability_table)
    return {"schema_version": "bace_gnn_independent_scientific_audit_v1", "state": "PASS",
        "scientific_engine_commit": run["execution_commit"], "source_final_audit_sha256": sha256_file(source / "gnn_seed7_final_audit.json"),
        "source_evaluation_root": str(source), "source_bundle_root": str(root), "cohort_contract": contract,
        "models": models, "distance_contract": manifest["wnode_config"],
        "molclr_checkpoint_sha256": sha256_file(bundle_file(root, manifest, manifest["molclr_checkpoint_path"])),
        "parent_scientific_checkpoint_sha256s": evidence,
        "calibration_prediction_sha256s": {n: sha256_file(prediction_source / n / "calibration_classifier_predictions.csv") for n in core.BACKBONES},
        "global_calibration_selector_replayed": True, "test_used_for_selection": False,
        "classifier_metrics_replayed": True, "native_common_metrics_replayed": True,
        "per_match_flip_and_best_match_replayed": True, "ot_recomputed": False,
        "classifier_inference_rerun": False, "seed": 7, "cross_seed_standard_deviation_claimed": False,
        "proposal_fixed": True, "main_matrix_write": False, "verifier_source_sha256": sha256_file(Path(__file__))}


def publish_overlay(*, bundle_root: str | Path, evaluation_root: str | Path, output_root: str | Path,
                    environment_manifest: str | Path, driver_commit: str,
                    calibration_prediction_root: str | Path | None = None) -> dict[str, Any]:
    """Fresh additive publication; original result and frozen classifiers remain unchanged."""
    _require(len(driver_commit) == 40 and all(x in "0123456789abcdef" for x in driver_commit), "Exact driver commit required")
    source = Path(evaluation_root).resolve(strict=True)
    target = Path(output_root).resolve()
    _require(all(target != p and target not in p.parents and p not in target.parents
                 for p in (source, Path(bundle_root).resolve())), "Publication root overlaps source")
    _require(not target.exists(), "Publication requires a fresh root")
    with (source / "writer.lock").open() as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        audit = verify_science(bundle_root=bundle_root, evaluation_root=source,
                              calibration_prediction_root=calibration_prediction_root)
        target.mkdir(parents=True)
        atomic_json(target / "independent_core_audit.json", {**audit, "publication_driver_commit": driver_commit})
        atomic_json(target / "cohort_contract_overlay.json", audit["cohort_contract"])
        shutil.copyfile(Path(__file__).with_name("framework.py"), target / "cohort_framework_source.py")
        atomic_json(target / "environment_manifest.json", read_json(environment_manifest))
        root = Path(bundle_root).resolve(strict=True)
        bundle_manifest = read_json(root / "bundle_manifest.json")
        # Original manifest identities are byte hashes, not merely JSON hashes.
        shutil.copyfile(root / "bundle_manifest.json", target / "input_bundle_manifest.json")
        shutil.copyfile(root / bundle_manifest["reference_contract_path"], target / "reference_contract.json")
        tables = {}
        for mode in MODES:
            rows = []
            for name in core.BACKBONES:
                report = read_json(source / name / mode / "explanation_metrics.json")
                rows.append({"backbone": name, "cohort": mode, "seed": 7,
                    **{k: v for k, v in report.items() if not isinstance(v, (list, dict))}})
            tables[mode] = rows
            atomic_csv(target / f"gnn_seed7_explanation_{mode}.csv", rows)
        core._latex(target / "gnn_seed7_table.tex", tables["common"] + tables["native"],
                    ("backbone", "cohort", "seed", "cohort_size", "CCRCov@10", "CCRCov@20", "conditional_median_WNode"))
        files = {p.name: sha256_file(p) for p in target.iterdir() if p.is_file()}
        publication = {"state": "PASS", "files": files, "source_evaluation_root": str(source),
            "source_final_audit_sha256": audit["source_final_audit_sha256"],
            "scientific_engine_commit": audit["scientific_engine_commit"], "publication_driver_commit": driver_commit,
            "main_matrix_write": False, "original_results_modified": False}
        atomic_json(target / "publication_manifest.json", publication)
        package = package_verified_overlay(source=source, overlay=target, audit=audit,
            calibration_prediction_root=calibration_prediction_root)
        return {**publication, "package": package}


def verify_package_archive(path: str | Path) -> dict[str, Any]:
    """Stream-verify a portable result package without extraction or model loading."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        _require(len(names) == len(set(names)) and "package_manifest.json" in names,
                 "Duplicate package member or missing manifest")
        _require(all(m.isfile() and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members),
                 "Unsafe package member")
        manifest = json.load(archive.extractfile("package_manifest.json"))
        _require(manifest.get("manifest_sha256") == canonical_json_sha256({
            k: v for k, v in manifest.items() if k != "manifest_sha256"}), "Portable package manifest hash differs")
        _require(set(names) == {*manifest["files"], "package_manifest.json"}, "Package inventory incomplete")
        required = {"publication/independent_core_audit.json", "publication/cohort_contract_overlay.json",
            "publication/cohort_framework_source.py",
            "publication/reference_contract.json", "publication/input_bundle_manifest.json",
            "publication/environment_manifest.json", "publication/gnn_seed7_explanation_native.csv",
            "publication/gnn_seed7_explanation_common.csv", "publication/gnn_seed7_table.tex"}
        for name in core.BACKBONES:
            required.update(f"classifiers/{name}/{leaf}" for leaf in ("model.pt", "model_card.json",
                "feature_schema.json", "temperature_scaling.json", "sha256sums.txt"))
            required.add(f"evaluation/{name}/calibration_classifier_predictions.csv")
        _require(required <= set(names), "Portable scientific package lacks required provenance")
        for member in members:
            if member.name == "package_manifest.json":
                continue
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            expected = manifest["files"][member.name]
            _require(member.size == expected["size"] and digest.hexdigest() == expected["sha256"],
                     "Portable package byte identity differs")
        evidence = json.load(archive.extractfile("publication/independent_core_audit.json"))
        _require(evidence.get("state") == "PASS" and evidence.get("main_matrix_write") is False
                 and evidence.get("global_calibration_selector_replayed") is True
                 and evidence.get("native_common_metrics_replayed") is True
                 and evidence.get("per_match_flip_and_best_match_replayed") is True,
                 "Portable package lacks independent scientific PASS")
        run = json.load(archive.extractfile("evaluation/run_manifest.json"))
        producer = json.load(archive.extractfile("evaluation/gnn_seed7_final_audit.json"))
        _require(evidence["source_final_audit_sha256"] == manifest["files"]["evaluation/gnn_seed7_final_audit.json"]["sha256"]
                 and evidence["scientific_engine_commit"] == run["execution_commit"] == manifest["scientific_engine_commit"]
                 and evidence["publication_driver_commit"] == manifest["publication_driver_commit"],
                 "Portable original-science/driver/audit binding differs")
        for rel, digest in producer["files"].items():
            _require(manifest["files"].get(f"evaluation/{rel}", {}).get("sha256") == digest,
                     "Portable producer audit inventory differs")
        _require(required_files() <= set(producer["files"]), "Portable producer inventory incomplete")
        for rel, digest in evidence["parent_scientific_checkpoint_sha256s"].items():
            _require(manifest["files"].get(f"evaluation/{rel}", {}).get("sha256") == digest,
                     "Portable parent scientific checkpoint differs")
        for name in core.BACKBONES:
            _require(manifest["files"][f"classifiers/{name}/model.pt"]["sha256"]
                     == evidence["models"][name]["model_sha256"] == run["checkpoints"][name]
                     and manifest["files"][f"classifiers/{name}/temperature_scaling.json"]["sha256"]
                     == evidence["models"][name]["temperature_sha256"]
                     and manifest["files"][f"evaluation/{name}/calibration_classifier_predictions.csv"]["sha256"]
                     == evidence["calibration_prediction_sha256s"][name], "Portable classifier/prediction binding differs")
        _require(manifest["files"]["publication/input_bundle_manifest.json"]["sha256"] == run["bundle_sha256"]
                 and manifest["files"]["publication/reference_contract.json"]["sha256"]
                 == evidence["cohort_contract"]["reference_contract_sha256"]
                 and manifest["files"]["publication/cohort_framework_source.py"]["sha256"]
                 == evidence["cohort_contract"]["framework_source_sha256"], "Portable reference inputs differ")
        return {"state": "PASS", "sha256": sha256_file(path), "file_count": len(names),
            "scientific_engine_commit": manifest["scientific_engine_commit"],
            "publication_driver_commit": manifest["publication_driver_commit"], "main_matrix_write": False}


def package_verified_overlay(*, source: Path, overlay: Path, audit: Mapping[str, Any],
                            calibration_prediction_root: str | Path | None = None) -> dict[str, Any]:
    """Package only verified inputs/results, never training checkpoints or caches."""
    producer = read_json(source / "gnn_seed7_final_audit.json")
    publication = read_json(overlay / "publication_manifest.json")
    paths = {f"evaluation/{rel}": source / rel for rel in producer["files"]}
    for rel in ("gnn_seed7_final_audit.json", "GNN_CORE_SEED7_PASS", *audit["parent_scientific_checkpoint_sha256s"]):
        paths[f"evaluation/{rel}"] = source / rel
    prediction_root = source if calibration_prediction_root is None else Path(calibration_prediction_root)
    for name in core.BACKBONES:
        paths[f"evaluation/{name}/calibration_classifier_predictions.csv"] = prediction_root / name / "calibration_classifier_predictions.csv"
        model = Path(audit["models"][name]["root"])
        for line in (model / "sha256sums.txt").read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            _require(Path(rel).name == rel and sha256_file(model / rel) == digest, "Classifier changed before packaging")
            paths[f"classifiers/{name}/{rel}"] = model / rel
        paths[f"classifiers/{name}/sha256sums.txt"] = model / "sha256sums.txt"
    for leaf in (*publication["files"], "publication_manifest.json"):
        paths[f"publication/{leaf}"] = overlay / leaf
    metadata = {"schema_version": "bace_gnn_independently_verified_package_v1", "state": "PASS",
        "files": {name: {"sha256": sha256_file(path), "size": path.stat().st_size} for name, path in paths.items()},
        "scientific_engine_commit": publication["scientific_engine_commit"],
        "publication_driver_commit": publication["publication_driver_commit"], "main_matrix_write": False}
    metadata["manifest_sha256"] = canonical_json_sha256(metadata)
    atomic_json(overlay / "package_manifest.json", metadata)
    partial = overlay / "bace_gnn_seed7_verified.tar.gz.partial"
    with partial.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, path in sorted(paths.items()):
                _require(path.is_file() and not path.is_symlink(), "Non-regular package source")
                archive.add(path, arcname=name, recursive=False)
            archive.add(overlay / "package_manifest.json", arcname="package_manifest.json", recursive=False)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = verify_package_archive(partial)
    destination = overlay / "bace_gnn_seed7_verified.tar.gz"
    os.replace(partial, destination)
    receipt.update({"bundle": str(destination), "bytes": destination.stat().st_size})
    atomic_json(overlay / "package_receipt.json", receipt)
    atomic_json(overlay / "result_package.json", {**receipt, "path": str(destination)})
    return receipt


def import_verified_bundle(*, archive_path: str | Path, expected_sha256: str,
                           output_root: str | Path) -> dict[str, Any]:
    """Fresh portable import; historical HPC absolute paths are never reopened."""
    archive_path = Path(archive_path).resolve(strict=True)
    _require(sha256_file(archive_path) == expected_sha256, "Transport archive SHA differs")
    verified = verify_package_archive(archive_path)
    destination = Path(output_root).absolute()
    _require(not destination.exists() and destination not in archive_path.parents,
             "Import requires a fresh disjoint root")
    _require(not any(parent.is_symlink() for parent in (destination, *destination.parents)),
             "Import destination contains a symlink")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            # verify_package_archive already rejected links and traversal.
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(block)
                sink.flush()
                os.fsync(sink.fileno())
    metadata = read_json(destination / "package_manifest.json")
    for rel, identity in metadata["files"].items():
        target = destination / rel
        _require(target.stat().st_size == identity["size"] and sha256_file(target) == identity["sha256"],
                 "Imported artifact byte identity differs")
    core.verify_evaluation(destination / "evaluation")
    location = {"schema_version": "bace_gnn_portable_location_overlay_v1", "state": "PASS",
        "archive_sha256": expected_sha256, "scientific_engine_commit": verified["scientific_engine_commit"],
        "publication_driver_commit": verified["publication_driver_commit"],
        "original_manifest_paths_preserved": True, "historical_hpc_paths_opened": False,
        "model_roots": {name: str(destination / "classifiers" / name) for name in core.BACKBONES},
        "evaluation_root": str(destination / "evaluation"), "main_matrix_write": False,
        "classifier_inference_rerun": False, "ot_recomputed": False}
    atomic_json(destination / "location_overlay.json", location)
    return location
