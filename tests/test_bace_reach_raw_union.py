from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.bace_frozen_gnn_contracts import atomic_json, sha256_file, stable_sha256
from src.eval.bace_reach_v2 import seal
from src.eval import bace_reach_raw_binding as binding
from src.ablations.gnn import reach_raw_distance_reuse as raw


def fixture(tmp_path, monkeypatch, value=.2):
    campaign = tmp_path/"campaign"
    actual = {"wnode": {"solver": "exact_emd2"}}
    proof = {"unchanged-kernel": "sha"}
    monkeypatch.setattr(binding, "current_raw_contract", lambda *args: actual)
    monkeypatch.setattr(raw, "kernel_identity_proof", lambda *args: proof)
    monkeypatch.setattr(raw, "graph_key", lambda p, r, c: (p+"|"+r, p, r))
    import src.eval.bace_frozen_gnn_contracts as contracts
    monkeypatch.setattr(contracts, "load_bace_parents", lambda *args, **kwargs: [SimpleNamespace(parent_id="cal1", smiles="CCC")])
    contract = seal(campaign/"search_contract.json", {"paths": {"calibration": "unopened.csv"},
        "source_label": 1, "oracle_binding": "new-GINE"})
    pool = seal(campaign/"candidate_freeze.json", {"search_contract_sha256": contract["self_sha256"],
        "candidate_universe_sha256": "pool"})
    seal(campaign/"selector_freeze.json", {"test_opened": False,
        "candidate_freeze_sha256": pool["self_sha256"], "calibration_pairs_sha256": "sealed-pairs"})
    parent = campaign/"calibration"/(stable_sha256("cal1")[:24]+".json")
    match = {"parent_id": "cal1", "parent_smiles": "CCC", "residual_smiles": "CC", "candidate_id": "new-rule",
        "match_index": 0, "match_atom_indices": [0], "oracle_checkpoint_hash": "new-GINE",
        "action_semantics_version": "hard-delete", "distance_ok": True, "wnode_distance": value,
        "delete_valid": True, "sanitize_ok": True, "residual_connected": True}
    seal(parent, {"pool_sha256": "pool", "pairs": [{"parent_id": "cal1"}], "matches": [match]})
    old = {"schema": raw.SCHEMA, "split": "calibration", "raw_contract": actual,
        "kernel_identity": proof, "source_spec": {}, "source_parent_units": 288,
        "source_finite_match_records": 140, "raw_cost_count": 1,
        "graph_costs": {"CCO|CC": {"parent": "CCO", "residual": "CC", "distance": .1, "source_records": []}}}
    old["self_sha256"] = stable_sha256(old)
    files = {"index": old, "source_spec": {"raw_distance_source": {}}, "portable_manifest": {}}
    descriptor = {}
    for name, data in files.items():
        path = tmp_path/(name+".json")
        atomic_json(path, data)
        descriptor[name] = {"path": str(path), "sha256": sha256_file(path)}
    return campaign, descriptor, parent


def test_union_preserves_both_sources_no_flip_or_ot(tmp_path, monkeypatch):
    campaign, descriptor, parent = fixture(tmp_path, monkeypatch)
    output = tmp_path/"union.json"
    result = binding.export_calibration_raw_union(campaign, output, descriptor=descriptor,
        repo=tmp_path, science_commit="a"*40)
    assert result["raw_cost_count"] == 2
    assert result["source_parent_units"] == 289
    assert result["source_finite_match_records"] == 141
    assert result["ours_new_finite_match_records"] == 1
    assert result["ours_requests_overlapping_old_index"] == 0
    assert result["ot_recomputed"] == 0 and not result["source_flip_masks_reused"]
    action = result["graph_costs"]["CCC|CC"]["source_records"][0]["original_action_context"]
    assert action["oracle_checkpoint_hash"] == "new-GINE" and action["candidate_id"] == "new-rule"
    parent.unlink()
    assert binding.export_calibration_raw_union(campaign, output, descriptor=descriptor,
        repo=tmp_path, science_commit="a"*40) == result


def test_invalid_finite_cost_is_not_exported(tmp_path, monkeypatch):
    campaign, descriptor, _ = fixture(tmp_path, monkeypatch, value=float("inf"))
    with pytest.raises(ValueError, match="SOURCE_RECORD_INVALID"):
        binding.export_calibration_raw_union(campaign, tmp_path/"union.json", descriptor=descriptor,
            repo=tmp_path, science_commit="a"*40)


def test_source_parent_self_hash_must_be_valid(tmp_path, monkeypatch):
    campaign, descriptor, parent = fixture(tmp_path, monkeypatch)
    parent.write_text(parent.read_text().replace('"new-GINE"', '"different-model"'))
    with pytest.raises(ValueError, match="PARENT_CONFLICT"):
        binding.export_calibration_raw_union(campaign, tmp_path/"union.json", descriptor=descriptor,
            repo=tmp_path, science_commit="a"*40)
