"""Explicit fixtures only: no scientific PASS directory, models, DB or OT."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.baselines.cm_crem_audit import (ProvenanceAuditError, _Records, _verify_closed_jobs,
    audit_bace_run, fixed_spotcheck_pairs, independent_spotcheck, validate_encoding,
    validate_generation, validate_raw_pair)
from src.baselines.cm_crem_experiment import science_identity
from src.baselines.cm_crem_runtime import digest, file_sha
from src.baselines.cm_crem_selection import select_calibration


def spec_fixture():
    spec = yaml.safe_load((Path(__file__).parents[1]/"configs/baselines/cm_crem_global_v1.yaml").read_text())
    spec["upstream"]["commit"] = "b5816b502cde00ee24c652a02cbc54664583f773"
    spec["resolved_oracle"] = {"backbone": "gine", "model_sha256": "a"*64,
        "temperature": 1.2, "temperature_sha256": "b"*64}
    spec["resolved_wnode"] = {"feature_cost": "cosine", "node_mass": "uniform", "size_penalty_beta": 0.0,
        "numerical_contract_sha256": "c"*64, "molclr_checkpoint_sha256": "d"*64}
    spec["resolved_evaluation"] = {"theta": .008413518173529859, "cap": .02956508038627219}
    spec["resolved_parents"] = {"train": {"count": 386}, "calibration": {"count": 66}, "test": {"count": 141}}
    spec["execution"] = {"execution_commit": "e"*40}
    spec["science_hash"] = science_identity(spec)
    return spec


def encoding(name, spec):
    identity = {"canonical_smiles": name, "graph_sha256": digest(name), "feature_schema_sha256": "f"*64}
    row = {"schema_version": "cm_crem_original_molclr_node_encoding_v1", **identity,
        "candidate_id": digest(identity), "num_atoms": 1, "H": [[1.0, 2.0]], "atom_numbers": [6],
        "embedding_dtype": "float32", "node_extraction_version": "FIXTURE",
        "molclr_checkpoint_sha256": spec["resolved_wnode"]["molclr_checkpoint_sha256"],
        "numerical_contract_sha256": spec["resolved_wnode"]["numerical_contract_sha256"],
        "producer": {"device": "FIXTURE", "torch": "FIXTURE", "numpy": "FIXTURE", "rdkit": "FIXTURE", "architecture": "FIXTURE"}}
    row["encoding_sha256"] = digest(row)
    return row


def raw_pair(left, right, spec):
    body = {"schema_version": "cm_crem_raw_fullgraph_wnode_v1", "encoding_ids": sorted([left["encoding_sha256"], right["encoding_sha256"]]),
        "numerical_contract": {k: spec["resolved_wnode"][k] for k in ("numerical_contract_sha256", "feature_cost", "node_mass", "size_penalty_beta")},
        "solver": "exact_emd2", "ot_producer": {"numpy": "FIXTURE", "dtype": "float64_cost_uniform_mass",
            "implementation_sha256": "1"*64, "POT": "UNIT_TEST_INJECTED_SOLVER"}}
    return {**body, "raw_pair_key": digest(body), "distance": .01, "production_solver_used": False,
        "parent_graph_id": left["candidate_id"], "prototype_graph_id": right["candidate_id"], "distance_is_uncapped": True, "seconds": .01}


def prediction(row, enc, spec, label):
    prob = [.1, .9] if label else [.9, .1]
    return {**row, "full_graph_id": enc["candidate_id"], "oracle_weight_sha256": spec["resolved_oracle"]["model_sha256"],
        "temperature": 1.2, "predicted_label": label, "probabilities": prob, "logits": list(prob),
        "source_probability": prob[1], "confidence": .9}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def full_zero_fixture(root, *, one_target=False):
    spec = spec_fixture()
    cohorts = {}
    for split, count in (("train", 386), ("calibration", 66), ("test", 141)):
        cohorts[split] = [{"parent_id": f"{split}-{i:03d}", "smiles": f"FIXTURE-{split}-{i}"} for i in range(count)]
        path = root/f"fixture_inputs/{split}.json"
        write_json(path, cohorts[split])
        spec["resolved_parents"][split].update(path=str(path), sha256=file_sha(path),
            ordered_ids_sha256=digest([r["parent_id"] for r in cohorts[split]]))
    db = root/"fixture_inputs/db.json"
    write_json(db, {"status": "FIXTURE_STATIC_COPY", "url": spec["upstream"]["database"]["url"],
                    "compressed_sha256": "2"*64, "uncompressed_sha256": "3"*64})
    spec["execution"]["database_receipt"] = str(db)
    spec["science_hash"] = science_identity(spec)
    journal = []
    def put(name, data):
        row = {"science_hash": spec["science_hash"], **data}
        write_json(root/name, row)
        journal.append({"path": name, "sha256": file_sha(root/name), "science_hash": spec["science_hash"],
            "execution_commit": spec["execution"]["execution_commit"], "job_id": "FIXTURE", "pid": 123,
            "created_at": "FIXTURE", "fixture": True})
    preds = [prediction({**r, "split": "train"}, encoding(r["smiles"], spec), spec, int(i < 32))
             for i, r in enumerate(cohorts["train"])]
    attrs = [{"parent_id": p["parent_id"], "input_smiles": p["smiles"], "split": "train", "status": "NO_REPLACEABLE_CONTEXT",
              "generation_allowed": False, "oracle_weight_sha256": "a"*64, "temperature_sha256": "b"*64} for p in preds[:32]]
    target_enc = encoding("FIXTURE-TARGET", spec)
    target = prediction({**{k: target_enc[k] for k in ("candidate_id", "canonical_smiles", "graph_sha256", "feature_schema_sha256")},
                        "smiles": "FIXTURE-TARGET"}, target_enc, spec, 0)
    if one_target:
        attrs[0].update(status="ATTRIBUTION_COMPLETE", generation_allowed=True)
        seed = int(digest([spec["science_hash"], preds[0]["parent_id"], 7])[:16], 16)
        target["origins"] = [{"parent_id": preds[0]["parent_id"], "retained_raw_index": 0,
                              "raw_id": digest([seed, target["smiles"]])}]
    put("attribution.json", {"parents": preds[:32], "records": attrs, "train_predictions": preds})
    for p, attr in zip(preds[:32], attrs):
        suffix = digest(p["parent_id"])[:20]+".json"
        generation = {"parent_id": p["parent_id"], "status": "NO_REPLACEABLE_CONTEXT",
            "retained_raw": [], "top_level_calls": 0, "database_uncompressed_sha256": "3"*64}
        candidate_here = one_target and p["parent_id"] == preds[0]["parent_id"]
        if candidate_here:
            generation.update(status="GENERATED", split="train", seed=seed, top_level_calls=1,
                retained_raw=[{"smiles": target["smiles"], "raw_id": target["origins"][0]["raw_id"], "source_return_index": 0}],
                budget={"radius": 1, "min_max_inc": 3, "max_replacements_per_component": 64,
                        "component_combinations_cap": 500, "retained_raw_limit": 128, "parent_wall_limit_seconds": 900},
                native={"upstream_commit": spec["upstream"]["commit"], "native_function_bodies_changed": False,
                        "original_source_sha256": "b5e485195ede009c560ee397f458794dc22272c14fb17faf3313fbdc64f39e49"},
                environment={"python": "3.11.5", "crem": "0.2.14", "rdkit": "2023.9.6", "numpy": "1.26.4"},
                worker_pid=456, worker_exitcode=0, counts={"native_return_count": 1, "raw_unique_count": 1,
                    "retained_raw_count": 1, "raw_exact_duplicate_count": 0, "raw_truncated_count": 0})
        put("generation_units/"+suffix, generation)
        put("filter_units/"+suffix, {"parent_id": p["parent_id"], "status": "FILTER_COMPLETE", "source_prediction": p,
            "raw_count": int(candidate_here), "strict_flip_count": int(candidate_here),
            "unique_target_count": int(candidate_here), "chemically_valid_nonself_count": int(candidate_here),
            "accepted": [target] if candidate_here else [], "rejected": [], "test_loaded": False})
    enc = [encoding(p["smiles"], spec) for p in preds[:32]]
    pairs = [raw_pair(enc[i], enc[j], spec) for i in range(32) for j in range(i+1,32)][:64]
    put("pilot/oracle.json", {"parents": preds[:32], "encoded_graphs": enc, "nonself_pairs": pairs})
    put("pilot/final_receipt.json", {"status": "PILOT_ENGINEERING_PROTOCOL_ACCEPTED", "complete_end_to_end_pilot": True,
        "parent_count": 32, "real_nonself_wnode_pairs": 64, "scientific_parameters_tuned": False})
    cids = [target["candidate_id"]] if one_target else []
    pool = {"candidate_ids": cids, "candidates": [target] if one_target else [],
            "retained_count": int(one_target), "unique_strict_flip_count_before_cap": int(one_target),
            "test_loaded": False, "calibration_loaded": False}
    pool["pool_sha256"] = digest(pool)
    put("pool_freeze.json", pool)
    put("pool_encodings.json", {"pool_sha256": pool["pool_sha256"], "records": [target_enc] if one_target else []})
    cal_ids = [r["parent_id"] for r in cohorts["calibration"]]
    freeze = select_calibration(np.full((66,len(cids)), np.inf), pair_status=np.full((66,len(cids)), "BEFORE_NOT_SOURCE"), parent_ids=cal_ids,
        candidate_ids=cids, source_mask=np.zeros(66, dtype=bool), **spec["resolved_evaluation"],
        contract_sha256=spec["science_hash"], frozen_pool_sha256=pool["pool_sha256"])
    put("selection_freeze.json", freeze.to_dict())
    all_predictions = list(preds) + ([target] if one_target else [])
    for split in ("calibration", "test"):
        for index, p in enumerate(cohorts[split]):
            left = encoding(p["smiles"], spec)
            source = one_target and split == "test" and index < 12
            pred = prediction({**p, "split": split}, left, spec, int(source))
            all_predictions.append(pred)
            raw = raw_pair(left, target_enc, spec) if source else None
            pairs = [{"parent_id": p["parent_id"], "candidate_id": target["candidate_id"],
                "pred_before": int(source), "pred_after": 0, "strict_flip": source, "kept_in_base_denominator": True,
                "raw_distance": raw, "distance": raw["distance"] if source else None,
                "raw_pair_key": raw["raw_pair_key"] if source else None,
                "pair_status": "OK" if source else "BEFORE_NOT_SOURCE", "failure_reason": None if source else "BEFORE_NOT_SOURCE"}] if one_target else []
            put(split+"/parents/"+digest(p["parent_id"])[:20]+".json", {"parent_id": p["parent_id"],
                "pool_sha256": pool["pool_sha256"], "prediction": pred, "pairs": pairs, "parent_encoding": left if source else None})
    path = root/"producer_receipts/FIXTURE.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(r)+"\n" for r in journal))
    return spec, all_predictions


class FixtureBackend:
    def __init__(self, predictions, encodings=(), spec=None):
        self.predictions = {r["smiles"]: r for r in predictions}
        self.encodings = {r["canonical_smiles"]: r for r in encodings}
        self.spec, self.ot_calls = spec, 0
    def predict(self, rows):
        return [deepcopy(self.predictions[r["smiles"]]) for r in rows]
    def encode(self, rows):
        return [deepcopy(self.encodings[r["smiles"]]) for r in rows]
    def distance(self, left, right):
        self.ot_calls += 1
        return raw_pair(left, right, self.spec)


def fixture_audit(spec, root):
    result = audit_bace_run(spec, root, fixture=True)
    write_json(root/"test_evaluation.json", result["test_evaluation"])
    return result


def test_full_fixture_saved_record_and_zero_scientific_spotcheck(tmp_path):
    spec, preds = full_zero_fixture(tmp_path)
    audit = fixture_audit(spec, tmp_path)
    assert audit["status"] == "FIXTURE_PROVENANCE_VERIFIED"
    assert audit["scientific_pass_claimed"] is False
    assert audit["generation_status_counts"] == {"NO_REPLACEABLE_CONTEXT": 32}
    result = independent_spotcheck(spec, tmp_path, audit, fixture=True, fixture_backend=FixtureBackend(preds))
    assert result["audit_spotcheck_count"] == 0
    assert result["oracle_graph_recomputations"] == 3
    assert result["zero_evidence"]["reason"] == "EMPTY_TRAIN_LIBRARY"
    assert result["scientific_pass_claimed"] is False


def test_db_403_is_not_audit_pass(tmp_path):
    spec = spec_fixture()
    with pytest.raises(ProvenanceAuditError, match="ASSET_BLOCKED"):
        audit_bace_run(spec, tmp_path)


def test_production_rejects_fixture_db(tmp_path):
    spec, _ = full_zero_fixture(tmp_path)
    with pytest.raises(ProvenanceAuditError, match="database provenance"):
        audit_bace_run(spec, tmp_path)


def test_raw_and_reduction_rewrite_cannot_evade_producer_hash(tmp_path):
    spec, _ = full_zero_fixture(tmp_path)
    path = tmp_path/"pilot/oracle.json"
    row = json.loads(path.read_text())
    row["nonself_pairs"][0]["distance"] += 1
    write_json(path, row)
    with pytest.raises(ProvenanceAuditError, match="bytes changed"):
        audit_bace_run(spec, tmp_path, fixture=True)


def test_freeze_checked_before_any_test_inputs(tmp_path):
    spec, _ = full_zero_fixture(tmp_path)
    path = tmp_path/"selection_freeze.json"
    row = json.loads(path.read_text())
    row["theta"] = .9
    write_json(path, row)
    with pytest.raises(ProvenanceAuditError, match="bytes changed"):
        audit_bace_run(spec, tmp_path, fixture=True)


def test_terminal_failed_job_units_can_be_reused(monkeypatch):
    monkeypatch.setattr("src.baselines.cm_crem_audit.subprocess.run", lambda *a, **kw:
        SimpleNamespace(stdout="10|FAILED|1:0\n11|COMPLETED|0:0\n"))
    assert _verify_closed_jobs({"10", "11"}) == {"10": "FAILED:1:0", "11": "COMPLETED:0:0"}


def test_running_producer_not_closed(monkeypatch):
    monkeypatch.setattr("src.baselines.cm_crem_audit.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout="10|RUNNING|0:0\n"))
    with pytest.raises(ProvenanceAuditError, match="not terminal"):
        _verify_closed_jobs({"10"})


@pytest.mark.parametrize("status", ["INFRASTRUCTURE_FAILED", "ENGINEERING_FAILED", "PENDING"])
def test_failure_never_generation_terminal(status):
    with pytest.raises(ProvenanceAuditError, match="legal terminal"):
        validate_generation({"parent_id": "p", "status": status, "retained_raw": []}, {"parent_id": "p"}, "a"*64)


def test_timeout_cannot_adopt_partial_output():
    with pytest.raises(ProvenanceAuditError, match="partial"):
        validate_generation({"parent_id": "p", "status": "TIMEOUT_BUDGETED", "retained_raw": [{}], "partial_adopted": True},
                            {"parent_id": "p", "generation_allowed": True}, "a"*64)


def test_raw_producer_and_nonfinite_values():
    spec = spec_fixture()
    left, right = encoding("LEFT", spec), encoding("RIGHT", spec)
    raw = raw_pair(left, right, spec)
    validate_raw_pair(raw, left, right, spec["resolved_wnode"], fixture=True)
    with pytest.raises(ProvenanceAuditError, match="fixture solver"):
        validate_raw_pair(raw, left, right, spec["resolved_wnode"])
    raw["distance"] = float("nan")
    with pytest.raises(ProvenanceAuditError, match="nonfinite"):
        validate_raw_pair(raw, left, right, spec["resolved_wnode"], fixture=True)


def test_fixed_sampling_not_distance_ranked():
    rows = [{"parent_id": str(i), "prediction": {"full_graph_id": "parent"+str(i)},
        "pairs": [{"candidate_id": "target", "pair_status": "OK", "strict_flip": True,
                   "raw_distance": {"distance": i}}]} for i in range(20)]
    chosen = [r["parent_id"] for r, p in fixed_spotcheck_pairs(rows, ["target"], "a"*64)]
    assert len(chosen) == 8
    for r in rows:
        r["pairs"][0]["raw_distance"]["distance"] *= -123
    assert chosen == [r["parent_id"] for r, p in fixed_spotcheck_pairs(rows[::-1], ["target"], "a"*64)]


def test_spotcheck_refuses_post_provenance_mutation(tmp_path):
    spec, preds = full_zero_fixture(tmp_path)
    audit = fixture_audit(spec, tmp_path)
    path = next((tmp_path/"test/parents").glob("*.json"))
    data = json.loads(path.read_text()); data["prediction"]["predicted_label"] = 1
    write_json(path, data)
    with pytest.raises(ProvenanceAuditError, match="changed after"):
        independent_spotcheck(spec, tmp_path, audit, fixture=True, fixture_backend=FixtureBackend(preds))


def test_zero_spotcheck_detects_actual_prediction_mismatch(tmp_path):
    spec, preds = full_zero_fixture(tmp_path)
    audit = fixture_audit(spec, tmp_path)
    backend = FixtureBackend(preds)
    for p in backend.predictions.values():
        p["logits"] = [0., 0.]
    with pytest.raises(ProvenanceAuditError, match="oracle.logits"):
        independent_spotcheck(spec, tmp_path, audit, fixture=True, fixture_backend=backend)


def test_nonempty_full_audit_and_bounded_recomputation(tmp_path):
    spec, preds = full_zero_fixture(tmp_path, one_target=True)
    audit = fixture_audit(spec, tmp_path)
    backend = FixtureBackend(preds, [encoding(p["smiles"], spec) for p in preds], spec)
    result = independent_spotcheck(spec, tmp_path, audit, fixture=True, fixture_backend=backend)
    assert result["audit_spotcheck_count"] == backend.ot_calls == 8
    assert result["oracle_graph_recomputations"] == result["encoding_graph_recomputations"] == 16
    assert result["cache_production_reuse_claimed"] is False
    assert result["zero_evidence"] is None


def test_recomputed_distance_not_equal_rejected_without_tolerance(tmp_path):
    spec, preds = full_zero_fixture(tmp_path, one_target=True)
    audit = fixture_audit(spec, tmp_path)
    backend = FixtureBackend(preds, [encoding(p["smiles"], spec) for p in preds], spec)
    original = backend.distance
    def mismatch(left, right):
        raw = original(left, right)
        raw["distance"] = np.nextafter(raw["distance"], np.inf).item()
        return raw
    backend.distance = mismatch
    with pytest.raises(ProvenanceAuditError, match="raw exact WNode"):
        independent_spotcheck(spec, tmp_path, audit, fixture=True, fixture_backend=backend)
