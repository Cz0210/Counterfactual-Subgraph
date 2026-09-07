import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments import bace_gin_reach_test_raw as m
from src.eval.bace_frozen_gnn_contracts import stable_sha256, sha256_file
from src.eval.bace_reach_v2 import seal


def write(path, data, sealed=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if sealed:
        data = {**data, "self_sha256": stable_sha256(data)}
    path.write_text(json.dumps(data))
    return {"path": str(path), "sha256": sha256_file(path)}, data


def fixture(tmp_path, monkeypatch):
    root, repo = tmp_path/"old", tmp_path/"repo"
    proof = {}
    for rel in m.KERNELS:
        p = repo/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
        proof[rel] = sha256_file(p)
    raw = {"wnode": {"solver": "exact_emd2"}, "feature_schema": {"sha256": "schema"}}
    monkeypatch.setattr(m, "current_raw_contract", lambda c,p: raw)
    monkeypatch.setattr(m, "kernel_identity_proof", lambda r,c: proof)
    ids = [f"rule-{i}" for i in range(21)]
    candidates = [dict(candidate_id=c, canonical_fragment="C") for c in ids]
    root.mkdir()
    poolpath = root/"candidate_universe.jsonl"
    poolpath.write_text("".join(json.dumps(c)+"\n" for c in candidates))
    testpath = root/"test.csv"
    testpath.write_text("fixture never used to choose a policy")
    parent_ids = [f"parent-{i}" for i in range(141)]
    parents = [SimpleNamespace(parent_id=p, smiles="CCO", label=1) for p in parent_ids]
    monkeypatch.setattr(m, "load_bace_parents", lambda *a,**kw: parents)
    portable, _ = write(tmp_path/"portable.json", {"splits": {"test": "test.csv"},
        "files": {"test.csv": {"sha256": sha256_file(testpath)}}}, False)
    source_spec, _ = write(tmp_path/"source_spec.json", {"raw_distance_source": {"correction": "original614"}}, False)
    contract = dict(oracle_binding="old-GINE", source_label=1)
    contract_item, contract = write(root/"search_contract.json", contract)
    pool_item, pool = write(root/"candidate_freeze.json", dict(test_opened=False,
        search_contract_sha256=contract["self_sha256"], candidate_universe_sha256=sha256_file(poolpath)))
    controls = dict(old_pool_old_selector=ids[:20], new_pool_old_selector=ids[1:], new_pool_reach_first=ids[1:])
    selector_item, selector = write(root/"selector_freeze.json", dict(test_opened=False,
        candidate_freeze_sha256=pool["self_sha256"], controls={k:v for k,v in controls.items() if k!="new_pool_reach_first"},
        reach_first={"ordered_rule_ids": controls["new_pool_reach_first"]}))
    gate_item, gate = write(root/"train_reach_gate.json", dict(state="NO_ADDITIONAL_PPO_REQUIRED_BY_TRAIN_GATE",
        candidate_freeze_sha256=pool["self_sha256"]))
    binding_item, binding = write(root/"final_test_binding.json", dict(state="REACH_V2_FINAL_CONFIGURATION_FROZEN",
        campaign=str(root), test_output_root=str(root/"three-control-final"), selected_control="new_pool_reach_first",
        selected_using_test=False, test_opened=False, test_campaigns_max=1, main_matrix_write=False,
        claim_new_untouched_test=False, search_contract_sha256=contract["self_sha256"],
        candidate_freeze_sha256=pool["self_sha256"], selector_freeze_sha256=selector["self_sha256"],
        train_gate_sha256=gate["self_sha256"], controls=controls, test_path=str(testpath), test_sha256=sha256_file(testpath),
        raw_distance_source=dict(portable_manifest=portable, source_spec=source_spec), old_test_pair_source={"path": "accepted-old20"}))
    raw_index = dict(schema=m.SCHEMA, split="test", state="RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS",
        new_test_freeze_sha256=binding_item["sha256"], raw_contract=raw, raw_contract_sha256=stable_sha256(raw),
        kernel_identity=proof, graph_costs={}, raw_cost_count=0, source_spec={"correction": "original614"},
        source_parent_units=614, source_finite_match_records=0, model_inference_performed=False, ot_recomputed=0,
        source_flip_masks_reused=False, source_selected_match_minima_reused=False, old_cache_keys_modified=False)
    index_item, index = write(tmp_path/"old-index.json", raw_index)
    raw_item, raw_reuse = write(root/"three-control-final/raw_distance_reuse.json", dict(
        state="EXPLICIT_RAW_DISTANCE_REUSE_NOT_FLIP_ADOPTION", old_cache_keys_modified=False,
        input_binding=dict(current_raw_contract_sha256=stable_sha256(raw), source_flip_masks_reused=False,
            source_descriptor=dict(index=index_item, portable_manifest=portable, source_spec=source_spec)),
        reuse_records=[], committed_parent_new_raw_graph_requests=1,
        current_process_stats=dict(new_raw_graph_requests=1, pair_distance_cache_misses=1)))
    final_item, final = write(root/"three-control-final/final_audit.json", dict(state="EXECUTION_VALID",
        test_parent_count=141, test_campaigns=1, test_selected_variant=False, main_matrix_write=False,
        final_binding_sha256=binding["self_sha256"], raw_distance_reuse_sha256=raw_reuse["self_sha256"]))
    terminal_item, _ = write(root/"three-control-final/cpu_owner_terminal.json", dict(
        state="DESCRIPTIVE_EVALUATION_EXECUTION_COMPLETE", final_audit_sha256=final["self_sha256"]), False)
    audit_item, _ = write(root/"independent-audit/audit.json", dict(state="BLOCKED_FIRST_INDEPENDENT_EVIDENCE_CONFLICT",
        audit_commit="92b9e5830e537f41ffe8f22a35d7ccfcfe27f198", first_conflict={"message": "fixture:FULL_REACH_WITNESS_CANNOT_BE_REAPPLIED"}))
    row = dict(parent_id=parent_ids[0], parent_smiles="CCO", candidate_id=ids[-1], canonical_fragment="C",
        oracle_checkpoint_hash="old-GINE", distance_ok=True, wnode_distance=.125, match_index=0,
        match_atom_indices=[0], delete_valid=True, sanitize_ok=True, residual_connected=True, residual_smiles="CO",
        action_semantics_version="connected_sanitized_residual_v1", residual_num_components=1, contains_dot=False)
    for i,p in enumerate(parents):
        write(root/"three-control-final/parents"/(stable_sha256(p.parent_id)[:24]+".json"), dict(
            parent_id=p.parent_id, final_binding_sha256=binding["self_sha256"], test_used_for_selection=False,
            old_pair_source_reused=binding["old_test_pair_source"], raw_source_index_sha256=index["self_sha256"],
            new_selected_pairs=[dict(parent_id=p.parent_id, candidate_id=ids[-1])],
            new_selected_match_witnesses=[row] if i==0 else [], raw_distance_adoptions=[], fresh_raw_graph_requests=int(i==0)))
    docs = dict(zip(m.SOURCE_FILES, (contract_item,pool_item,selector_item,gate_item,binding_item,final_item,raw_item,terminal_item)))
    source = dict(schema=m.SOURCE_SCHEMA, campaign=str(root), science_commit="a"*40, documents=docs, independent_audit=audit_item)
    freeze_item, freeze = write(tmp_path/"new-freeze.json", {"state": "actually frozen"})
    callback = lambda f: dict(state="ACTUAL_A_PLUS_FREEZE_VERIFIED", freeze_self_sha256=f["self_sha256"])
    kwargs = dict(repo=repo, new_freeze_path=freeze_item["path"], new_freeze_sha=freeze_item["sha256"], validate_new_freeze=callback)
    return source, kwargs, row


def reseal(path, modifier):
    value = json.loads(path.read_text())
    value.pop("self_sha256", None)
    modifier(value)
    return write(path, value)[0]


def test_gate_precedes_any_old_source_read(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_source_documents", lambda _: pytest.fail("old source was read"))
    with pytest.raises(ValueError, match="BEFORE_NEW_FREEZE"):
        m.export_test_raw({"campaign": "/not/read"}, tmp_path/"out", repo=tmp_path,
            new_freeze_path=None, new_freeze_sha=None, validate_new_freeze=lambda f: True)


def test_actual_freeze_receipt_not_boolean(tmp_path, monkeypatch):
    source, kw, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "_source_documents", lambda _: pytest.fail("old source was read"))
    kw["validate_new_freeze"] = lambda f: True
    with pytest.raises(ValueError, match="VALIDATOR_RECEIPT_REQUIRED"):
        m.export_test_raw(source, tmp_path/"out.json", **kw)


def test_finite_cost_replay_preserves_old_failed_audit_and_no_metrics(tmp_path, monkeypatch):
    source, kw, _ = fixture(tmp_path, monkeypatch)
    old_bytes = Path(source["independent_audit"]["path"]).read_bytes()
    result = m.export_test_raw(source, tmp_path/"out.json", **kw)
    assert result["source_parent_units"] == 141
    assert result["source_finite_match_records"] == result["raw_cost_count"] == 1
    assert result["ot_recomputed"] == 0 and not result["model_inference_performed"]
    assert not result["source_flip_masks_reused"] and not result["source_selected_match_minima_reused"]
    assert not result["source_full_pool_witnesses_used"]
    assert Path(source["independent_audit"]["path"]).read_bytes() == old_bytes
    cost = next(iter(result["graph_costs"].values()))
    assert (cost["parent"], cost["residual"], cost["distance"]) == ("CCO", "CO", .125)
    assert cost["source_records"][0]["original_action_context"]["oracle_checkpoint_hash"] == "old-GINE"
    assert "pred_after" not in cost and "pair_strict_flip" not in cost
    assert m.export_test_raw(source, tmp_path/"out.json", **kw) == result


@pytest.mark.parametrize("key,value", [("match_atom_indices",[1]), ("residual_smiles","CC"),
    ("wnode_distance",True), ("wnode_distance",float("nan")), ("oracle_checkpoint_hash","foreign"),
    ("sanitize_ok",False), ("action_semantics_version","foreign")])
def test_bad_finite_cost_or_mapping_never_adopted(tmp_path,monkeypatch,key,value):
    source, kw, _ = fixture(tmp_path, monkeypatch)
    path = Path(source["campaign"])/"three-control-final/parents"/(stable_sha256("parent-0")[:24]+".json")
    reseal(path, lambda d: d["new_selected_match_witnesses"][0].update({key:value}))
    with pytest.raises(ValueError, match="FINITE_RAW"):
        m.export_test_raw(source, tmp_path/"out.json", **kw)
    assert not (tmp_path/"out.json").exists()


def test_parent_selfhash_corruption_refused(tmp_path,monkeypatch):
    source, kw, _ = fixture(tmp_path,monkeypatch)
    path = Path(source["campaign"])/"three-control-final/parents"/(stable_sha256("parent-0")[:24]+".json")
    data=json.loads(path.read_text()); data["fresh_raw_graph_requests"]=2; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SELF_HASH_CONFLICT"):
        m.export_test_raw(source,tmp_path/"out.json",**kw)


def test_old_raw_related_failure_not_ignored(tmp_path,monkeypatch):
    source, kw, _ = fixture(tmp_path,monkeypatch)
    source["independent_audit"] = reseal(Path(source["independent_audit"]["path"]),
        lambda d:d["first_conflict"].update(message="RAW_GRAPH_COST_SOURCE_CONFLICT:key"))
    with pytest.raises(ValueError, match="EXPLICIT_RAW_SCOPE_REVIEW"):
        m.export_test_raw(source,tmp_path/"out.json",**kw)


def test_old_final_selector_binding_refused(tmp_path,monkeypatch):
    source, kw, _ = fixture(tmp_path,monkeypatch)
    item=source["documents"]["final_binding"]
    source["documents"]["final_binding"] = reseal(Path(item["path"]),lambda d:d.update(selector_freeze_sha256="wrong"))
    with pytest.raises(ValueError, match="FREEZE_DEPENDENCY"):
        m.export_test_raw(source,tmp_path/"out.json",**kw)


def test_new_gate_before_union_old_index_read(tmp_path):
    with pytest.raises(ValueError, match="BEFORE_NEW_FREEZE"):
        m.union_test_indexes({"path":"not-read"},{"path":"also-not-read"},tmp_path/"out",repo=tmp_path,
            new_freeze_path=None,new_freeze_sha=None,validate_new_freeze=None)


def test_union_preserves_both_provenances_and_requires_exact_same_cost(tmp_path,monkeypatch):
    source, kw, _ = fixture(tmp_path,monkeypatch)
    ours=m.export_test_raw(source,tmp_path/"ours.json",**kw)
    original=copy.deepcopy(ours); original.pop("self_sha256")
    original["source_spec"]={"correction":"original614"}; original["source_parent_units"]=614
    original_item,_=write(tmp_path/"original.json",original)
    ours_item={"path":str(tmp_path/"ours.json"),"sha256":sha256_file(tmp_path/"ours.json")}
    result=m.union_test_indexes(original_item,ours_item,tmp_path/"union.json",**kw)
    assert result["raw_cost_count"]==1 and result["overlap_graph_keys"]==1
    assert result["source_parent_units"]==755
    assert len(next(iter(result["graph_costs"].values()))["source_records"])==2
    original_item=reseal(tmp_path/"original.json",lambda d:next(iter(d["graph_costs"].values())).update(distance=.126))
    with pytest.raises(ValueError,match="NUMERICAL_CONFLICT"):
        m.union_test_indexes(original_item,ours_item,tmp_path/"bad-union.json",**kw)


def test_union_different_freeze_rejected(tmp_path,monkeypatch):
    source, kw, _ = fixture(tmp_path,monkeypatch)
    m.export_test_raw(source,tmp_path/"ours.json",**kw)
    ours_item={"path":str(tmp_path/"ours.json"),"sha256":sha256_file(tmp_path/"ours.json")}
    wrong=reseal(tmp_path/"ours.json",lambda d:d.update(new_test_freeze_sha256="other"))
    with pytest.raises(ValueError,match="FREEZE_CONFLICT"):
        m.union_test_indexes(wrong,wrong,tmp_path/"union.json",**kw)


def new_freeze_fixture(root, monkeypatch):
    from src.experiments import bace_gin_reach_v2 as driver
    called=[]
    monkeypatch.setattr(driver,"require_freeze",lambda s,f:called.append((s,f)))
    spec={"test_results_previously_observed":True,"new_control_name":"expanded_pool_new_selector"}
    order=[f"rule-{i}" for i in range(20)]
    _,contract=write(root/"contract.json",dict(main_matrix_write=False,spec_sha256=stable_sha256(spec),old_order=order))
    _,freeze=write(root/"selection_freeze.json",dict(contract_sha256=contract["self_sha256"],
        main_matrix_write=False,calibration_parent_ids=[f"p-{i}" for i in range(66)],
        controls=dict(old66_old_selector=order,old66_new_selector=order,expanded_pool_new_selector=order)))
    return spec,freeze,called


def test_portable_new_gate_calls_authoritative_dynamic_control_validator(tmp_path,monkeypatch):
    spec,freeze,called=new_freeze_fixture(tmp_path,monkeypatch)
    result=m.validate_aplus_freeze(freeze,spec=spec,evidence_root=tmp_path)
    assert result["state"]=="ACTUAL_A_PLUS_FREEZE_VERIFIED" and called==[(spec,freeze)]


@pytest.mark.parametrize("mutate", [
    lambda f:f.update(contract_sha256="wrong"),
    lambda f:f["calibration_parent_ids"].__setitem__(0,"p-1"),
    lambda f:f["controls"]["expanded_pool_new_selector"].__setitem__(0,"rule-1"),
])
def test_portable_new_gate_rejects_dependency_count_or_sequence_gap(tmp_path,monkeypatch,mutate):
    spec,freeze,_=new_freeze_fixture(tmp_path,monkeypatch)
    freeze.pop("self_sha256"); mutate(freeze); freeze["self_sha256"]=stable_sha256(freeze)
    with pytest.raises(ValueError,match="NEW_FREEZE"):
        m.validate_aplus_freeze(freeze,spec=spec,evidence_root=tmp_path)
