"""Freeze-gated raw-cost migration from the completed Ours2607 test records.

This leaf neither selects rules nor evaluates a model/OT problem. A historical
full-pool witness audit failure is preserved; every *adopted* finite match gets
an independent graph/deletion check. Old probabilities, flips, match minima and
reported performance never become inputs to the new GIN evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from src.ablations.gnn.reach_raw_distance_reuse import (
    KERNELS, SCHEMA, graph_key, kernel_identity_proof,
)
from src.eval.bace_frozen_gnn_contracts import (
    load_bace_parents, read_jsonl, sha256_file, stable_sha256,
)
from src.eval.bace_reach_raw_binding import current_raw_contract
from src.eval.bace_reach_v2 import seal

SOURCE_SCHEMA = "completed_ours2607_test_raw_source_v1"
ACTION_FIELDS = ("parent_id", "candidate_id", "match_index", "match_atom_indices",
                 "oracle_checkpoint_hash", "action_semantics_version")
SOURCE_FILES = {
    "search_contract": "search_contract.json", "candidate_freeze": "candidate_freeze.json",
    "selector_freeze": "selector_freeze.json", "train_gate": "train_reach_gate.json",
    "final_binding": "final_test_binding.json", "execution_audit": "three-control-final/final_audit.json",
    "raw_reuse": "three-control-final/raw_distance_reuse.json",
    "terminal": "three-control-final/cpu_owner_terminal.json",
}


def _require(condition, error):
    if not condition:
        raise ValueError(error)


def _sealed(value, label):
    _require(value.get("self_sha256") == stable_sha256(
        {k: v for k, v in value.items() if k != "self_sha256"}), "SELF_HASH_CONFLICT:" + label)
    return value


def bound_json(item, *, sealed=False):
    path = Path(item["path"])
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == item["sha256"], "FILE_BINDING_CONFLICT:" + str(path))
    value = json.loads(data)
    return _sealed(value, str(path)) if sealed else value


def validate_aplus_freeze(freeze, *, spec, evidence_root):
    """Portable new-freeze callback: copied spec, contract and actual freeze.

    No old test source is read here. The root driver's policy validator remains
    authoritative; these extra checks bind its actual sealed dependency/order.
    """
    from src.experiments.bace_gin_reach_v2 import require_freeze
    _sealed(freeze, "new A+ freeze")
    require_freeze(spec, freeze)
    contract = _sealed(json.loads((Path(evidence_root) / "contract.json").read_text()), "new A+ contract")
    _require(contract["self_sha256"] == freeze["contract_sha256"]
        and contract["spec_sha256"] == stable_sha256(spec)
        and contract["main_matrix_write"] is False and freeze["main_matrix_write"] is False
        and spec["test_results_previously_observed"] is True
        and len(freeze["calibration_parent_ids"]) == len(set(freeze["calibration_parent_ids"])) == 66,
        "NEW_FREEZE_ACTUAL_DEPENDENCY_CONFLICT")
    _require(all(len(order) == len(set(order)) == 20 for order in freeze["controls"].values())
        and freeze["controls"]["old66_old_selector"] == contract["old_order"],
        "NEW_FREEZE_FIXED_NESTED_SEQUENCE_CONFLICT")
    return {"state": "ACTUAL_A_PLUS_FREEZE_VERIFIED", "spec_sha256": stable_sha256(spec),
            "freeze_self_sha256": freeze["self_sha256"], "contract_self_sha256": contract["self_sha256"]}


def _new_gate(path, digest, callback):
    _require(path and digest and callable(callback), "TEST_RAW_MIGRATION_BEFORE_NEW_FREEZE")
    freeze = bound_json({"path": str(path), "sha256": digest}, sealed=True)
    receipt = callback(freeze)  # must precede opening any historical test data
    _require(isinstance(receipt, dict) and receipt.get("state") == "ACTUAL_A_PLUS_FREEZE_VERIFIED"
        and receipt.get("freeze_self_sha256") == freeze["self_sha256"], "ACTUAL_FREEZE_VALIDATOR_RECEIPT_REQUIRED")
    return receipt


def _outside_source(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    _require(output != source and source not in output.parents, "CANNOT_WRITE_OLD_SCIENCE_SOURCE")


def _index_checked(index, *, freeze_sha, raw=None, proof=None):
    _sealed(index, "raw index")
    _require(index.get("schema") == SCHEMA and index.get("split") == "test"
        and index.get("state") == "RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS"
        and index.get("new_test_freeze_sha256") == freeze_sha
        and index.get("raw_contract_sha256") == stable_sha256(index["raw_contract"])
        and set(index["kernel_identity"]) == set(KERNELS)
        and index.get("ot_recomputed") == 0 and index.get("model_inference_performed") is False
        and index.get("source_flip_masks_reused") is False
        and index.get("source_selected_match_minima_reused") is False
        and index.get("old_cache_keys_modified") is False,
        "RAW_INDEX_CONTRACT_OR_FREEZE_CONFLICT")
    if raw is not None:
        _require(index["raw_contract"] == raw and index["kernel_identity"] == proof,
                 "RAW_INDEX_KERNEL_OR_INPUT_DRIFT")
    _require(len(index["graph_costs"]) == index["raw_cost_count"], "RAW_INDEX_COUNT_CONFLICT")
    for key, row in index["graph_costs"].items():
        _require(type(row["distance"]) in (float, int) and math.isfinite(row["distance"])
            and row["distance"] >= 0 and row["source_records"]
            and graph_key(row["parent"], row["residual"], index["raw_contract_sha256"])[0] == key,
            "RAW_INDEX_GRAPH_OR_VALUE_CONFLICT:" + key)


def _historical_audit(audit):
    """Do not turn the old failed broader audit into a new whole-result PASS."""
    if audit.get("state") == "OURS_REACH_V2_SELECTED_CONTROLS_SCIENTIFIC_PASS":
        return "HISTORICAL_SELECTED_CONTROL_AUDIT_PASS_NOT_ADOPTING_METRICS"
    error = audit.get("first_conflict", {}).get("message", "")
    _require(audit.get("state") == "BLOCKED_FIRST_INDEPENDENT_EVIDENCE_CONFLICT"
        and audit.get("audit_commit") == "92b9e5830e537f41ffe8f22a35d7ccfcfe27f198"
        and error.endswith(":FULL_REACH_WITNESS_CANNOT_BE_REAPPLIED"),
        "OLD_AUDIT_CONFLICT_REQUIRES_EXPLICIT_RAW_SCOPE_REVIEW")
    return "OLD_FULL_POOL_WITNESS_FAILURE_PRESERVED_EACH_ADOPTED_MATCH_RECHECKED"


def _source_documents(source):
    _require(source.get("schema") == SOURCE_SCHEMA, "UNSUPPORTED_OLD_TEST_RAW_SOURCE")
    campaign = Path(source["campaign"]).resolve(strict=True)
    docs = {}
    for role, relative in SOURCE_FILES.items():
        item = source["documents"][role]
        _require(Path(item["path"]).resolve(strict=True) == campaign / relative,
                 "OLD_SOURCE_LOCATOR_CONFLICT:" + role)
        docs[role] = bound_json(item, sealed=role != "terminal")
    audit = bound_json(source["independent_audit"], sealed=True)
    state = _historical_audit(audit)
    contract, pool, selector, gate, binding, final, raw, terminal = (docs[k] for k in SOURCE_FILES)
    from src.ablations.gnn.reach_raw_distance_reuse import validate_ours_final_freeze
    validate_ours_final_freeze(binding, campaign)
    for name in ("search_contract", "candidate_freeze", "selector_freeze", "train_gate"):
        _require(binding[name + "_sha256"] == docs[name]["self_sha256"], "OLD_FINAL_DEPENDENCY_CONFLICT:" + name)
    _require(binding["campaign"] == str(campaign) and Path(binding["test_output_root"]) == campaign / "three-control-final"
        and final["state"] == "EXECUTION_VALID" and final["test_parent_count"] == 141
        and final["test_campaigns"] == 1 and final["test_selected_variant"] is False
        and final["main_matrix_write"] is False and final["final_binding_sha256"] == binding["self_sha256"]
        and terminal["state"] == "DESCRIPTIVE_EVALUATION_EXECUTION_COMPLETE"
        and terminal["final_audit_sha256"] == final["self_sha256"]
        and final["raw_distance_reuse_sha256"] == raw["self_sha256"]
        and raw["state"] == "EXPLICIT_RAW_DISTANCE_REUSE_NOT_FLIP_ADOPTION"
        and raw["old_cache_keys_modified"] is False, "OLD_COMPLETED_TEST_BINDING_CONFLICT")
    return campaign, docs, state


def _finite_match(row, *, parent, candidate, contract, raw_sha, outcome_cache):
    from src.chem.bace_reach_search import deletion_outcomes
    key = (parent.parent_id, candidate["candidate_id"])
    if key not in outcome_cache:
        outcome_cache[key] = {o.match_id: o for o in deletion_outcomes(parent.smiles, candidate, parent.parent_id)}
    outcome = outcome_cache[key].get(row["match_index"])
    value = row.get("wnode_distance")
    _require(outcome is not None and outcome.valid and type(value) in (float, int)
        and math.isfinite(value) and value >= 0 and row["parent_id"] == parent.parent_id
        and row["parent_smiles"] == parent.smiles and row["canonical_fragment"] == candidate["canonical_fragment"]
        and row["oracle_checkpoint_hash"] == contract["oracle_binding"], "FINITE_RAW_RECORD_BINDING_CONFLICT")
    expected = {"match_atom_indices": list(outcome.match_atom_indices), "delete_valid": outcome.valid,
        "sanitize_ok": outcome.sanitize_ok, "residual_connected": outcome.residual_connected,
        "residual_smiles": outcome.residual_smiles, "action_semantics_version": outcome.action_semantics_version,
        "residual_num_components": outcome.residual_num_components, "contains_dot": outcome.contains_dot}
    _require(all(row.get(k) == v for k, v in expected.items()), "FINITE_RAW_MATCH_GRAPH_REPLAY_CONFLICT")
    # Flip/probability fields are intentionally not consumed by this raw layer.
    return graph_key(parent.smiles, outcome.residual_smiles, raw_sha)


def export_test_raw(source, output, *, repo, new_freeze_path, new_freeze_sha, validate_new_freeze):
    """AutoDL-only completed-source export; no server/job management.

    Caller freezes the new A+ selection first and supplies its actual validator.
    A fresh narrow raw-record audit is embedded in the new index, never written
    over the old failed scientific audit. No legacy aggregate pair/metric file
    or full-pool reach entry is used to construct a distance.
    """
    new_receipt = _new_gate(new_freeze_path, new_freeze_sha, validate_new_freeze)
    _outside_source(output, source["campaign"])
    campaign, docs, audit_scope = _source_documents(source)
    contract, pool, binding, final, raw_reuse = (docs[k] for k in
        ("search_contract", "candidate_freeze", "final_binding", "execution_audit", "raw_reuse"))
    portable = bound_json(binding["raw_distance_source"]["portable_manifest"])
    raw = current_raw_contract(contract, portable)
    proof = kernel_identity_proof(Path(repo), source["science_commit"])
    raw_sha = stable_sha256(raw)
    descriptor = raw_reuse["input_binding"]["source_descriptor"]
    old_index = bound_json(descriptor["index"])
    _index_checked(old_index, freeze_sha=source["documents"]["final_binding"]["sha256"], raw=raw, proof=proof)
    _require(raw_reuse["input_binding"]["current_raw_contract_sha256"] == raw_sha
        and raw_reuse["input_binding"]["source_flip_masks_reused"] is False
        and descriptor["portable_manifest"] == binding["raw_distance_source"]["portable_manifest"]
        and old_index["source_spec"] == bound_json(descriptor["source_spec"])["raw_distance_source"],
        "OLD_RAW_INPUT_SOURCE_BINDING_CONFLICT")
    source_pool = campaign / "candidate_universe.jsonl"
    _require(sha256_file(source_pool) == pool["candidate_universe_sha256"], "OLD_CANDIDATE_POOL_CHANGED")
    candidates = read_jsonl(source_pool)
    by_id = {c["candidate_id"]: c for c in candidates}
    selected = set().union(*map(set, binding["controls"].values()))
    new_ids = selected - set(binding["controls"]["old_pool_old_selector"])
    _require(len(by_id) == len(candidates) and selected <= by_id.keys(), "OLD_SELECTED_POOL_MEMBERS_CONFLICT")
    _require(sha256_file(binding["test_path"]) == binding["test_sha256"], "OLD_TEST_INPUT_CHANGED")
    _require(portable["files"][portable["splits"]["test"]]["sha256"] == binding["test_sha256"],
             "OLD_TEST_NOT_ACCEPTED_BACE_INPUT")
    parents = load_bace_parents(binding["test_path"], source_label=contract["source_label"])
    _require(len(parents) == len({p.parent_id for p in parents}) == 141, "OLD_TEST_PARENT_COVERAGE_GAP")
    binding_sha = stable_sha256(dict(source=source, new_test_freeze_sha256=new_freeze_sha, raw=raw, kernels=proof))
    if Path(output).exists():
        existing = bound_json({"path": str(output), "sha256": sha256_file(output)}, sealed=True)
        _require(existing["binding_sha256"] == binding_sha, "FRESH_RAW_EXPORT_BINDING_CONFLICT")
        _index_checked(existing, freeze_sha=new_freeze_sha, raw=raw, proof=proof)
        return existing
    costs, members, adopted, fresh, finite = {}, [], [], 0, 0
    for parent in parents:
        path = campaign / "three-control-final/parents" / (stable_sha256(parent.parent_id)[:24] + ".json")
        _require(path.resolve(strict=True).parent == (campaign / "three-control-final/parents").resolve(), "PARENT_PATH_ESCAPE")
        data = path.read_bytes()
        record = _sealed(json.loads(data), str(path))
        _require(record["parent_id"] == parent.parent_id and record["final_binding_sha256"] == binding["self_sha256"]
            and record["test_used_for_selection"] is False and record["old_pair_source_reused"] == binding["old_test_pair_source"]
            and record["raw_source_index_sha256"] == old_index["self_sha256"], "OLD_TEST_PARENT_BINDING_CONFLICT")
        pairs = record["new_selected_pairs"]
        _require(len(pairs) == len(new_ids) and {r["candidate_id"] for r in pairs} == new_ids
            and all(r["parent_id"] == parent.parent_id for r in pairs), "OLD_SELECTED_PARENT_COVERAGE_GAP")
        member = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), self_sha256=record["self_sha256"], parent_id=parent.parent_id)
        members.append(member)
        seen, outcomes = set(), {}
        for row in record["new_selected_match_witnesses"]:
            identity = (row["candidate_id"], row["match_index"])
            _require(identity not in seen and row["candidate_id"] in new_ids, "OLD_MATCH_DUPLICATE_OR_FOREIGN_RULE")
            seen.add(identity)
            if row.get("distance_ok") is not True:
                _require(row.get("wnode_distance") is None, "UNVERIFIED_FINITE_RAW_VALUE_PRESENT")
                continue
            key, p, residual = _finite_match(row, parent=parent, candidate=by_id[row["candidate_id"]],
                contract=contract, raw_sha=raw_sha, outcome_cache=outcomes)
            value = row["wnode_distance"]
            if key in costs:
                _require(costs[key]["distance"] == value, "RAW_GRAPH_COST_CONFLICT:" + key)
            item = costs.setdefault(key, dict(parent=p, residual=residual, distance=value, source_records=[]))
            item["source_records"].append(dict(source_parent_member=str(path), source_parent_sha256=member["sha256"],
                source_match_sha256=stable_sha256(row), original_action_context={k: row[k] for k in ACTION_FIELDS},
                source_final_binding_sha256=binding["self_sha256"], deletion_graph_replayed=True))
            finite += 1
        adopted.extend(record["raw_distance_adoptions"])
        fresh += record["fresh_raw_graph_requests"]
    _require(adopted == raw_reuse["reuse_records"] and fresh == raw_reuse["committed_parent_new_raw_graph_requests"]
        and fresh == raw_reuse["current_process_stats"]["new_raw_graph_requests"], "OLD_RAW_EXECUTION_COUNT_CONFLICT")
    # This actual completed run recorded fresh raw misses, not a mixed unknown
    # cache format. Do not silently accept a future nonempty adoption schema.
    _require(not adopted and fresh == raw_reuse["current_process_stats"]["pair_distance_cache_misses"],
             "OLD_MIXED_RAW_ADOPTION_PROVENANCE_REQUIRES_EXPLICIT_ADAPTER")
    return seal(Path(output), dict(schema=SCHEMA, state="RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS",
        binding_sha256=binding_sha, split="test", raw_contract=raw, raw_contract_sha256=raw_sha,
        kernel_identity=proof, graph_costs=costs, raw_cost_count=len(costs), source_parent_units=len(members),
        source_finite_match_records=finite, new_test_freeze_sha256=new_freeze_sha,
        source_spec=dict(kind="COMPLETED_OURS2607_TEST_RAW_ONLY", binding=source, parent_sources=members,
            historical_audit_scope=audit_scope, execution_audit_self_sha256=final["self_sha256"],
            source_test_input_sha256=binding["test_sha256"],
            new_freeze_validation=new_receipt, original_raw_index=descriptor["index"]),
        raw_record_audit="FINITE_GRAPH_COST_SOURCE_AND_DELETION_MAPPING_PASS_NOT_WHOLE_RESULT_PASS",
        source_full_pool_witnesses_used=False, source_reported_metrics_used=False,
        old_cache_keys_modified=False, source_flip_masks_reused=False,
        source_selected_match_minima_reused=False, model_inference_performed=False, ot_recomputed=0))


def union_test_indexes(original, ours, output, *, repo, new_freeze_path, new_freeze_sha, validate_new_freeze):
    """HPC-only two-source union. Does not reread AutoDL parents or compute OT."""
    gate = _new_gate(new_freeze_path, new_freeze_sha, validate_new_freeze)
    _require(Path(output).resolve() not in {Path(original["path"]).resolve(), Path(ours["path"]).resolve()},
             "RAW_UNION_MUST_NOT_OVERWRITE_SOURCE_INDEX")
    old, extra = bound_json(original), bound_json(ours)
    _index_checked(old, freeze_sha=new_freeze_sha)
    _index_checked(extra, freeze_sha=new_freeze_sha, raw=old["raw_contract"], proof=old["kernel_identity"])
    _require(extra["source_spec"]["kind"] == "COMPLETED_OURS2607_TEST_RAW_ONLY"
        and extra["raw_record_audit"] == "FINITE_GRAPH_COST_SOURCE_AND_DELETION_MAPPING_PASS_NOT_WHOLE_RESULT_PASS",
        "OURS_RAW_SOURCE_NARROW_AUDIT_REQUIRED")
    _require(all(sha256_file(Path(repo) / p) == digest for p, digest in old["kernel_identity"].items()),
             "RAW_UNION_CURRENT_KERNEL_DRIFT")
    costs, overlap = copy.deepcopy(old["graph_costs"]), 0
    for key, value in extra["graph_costs"].items():
        if key in costs:
            _require(costs[key]["distance"] == value["distance"], "RAW_UNION_NUMERICAL_CONFLICT:" + key)
            overlap += 1
            costs[key]["source_records"].extend(copy.deepcopy(value["source_records"]))
        else:
            costs[key] = copy.deepcopy(value)
    return seal(Path(output), dict(schema=SCHEMA, state="RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS",
        binding_sha256=stable_sha256(dict(original=original, ours=ours, freeze=new_freeze_sha)), split="test",
        raw_contract=old["raw_contract"], raw_contract_sha256=old["raw_contract_sha256"],
        kernel_identity=old["kernel_identity"], graph_costs=costs, raw_cost_count=len(costs),
        source_parent_units=old["source_parent_units"] + extra["source_parent_units"],
        source_finite_match_records=old["source_finite_match_records"] + extra["source_finite_match_records"],
        new_test_freeze_sha256=new_freeze_sha,
        source_spec=dict(kind="ACCEPTED_GNN614_PLUS_VERIFIED_OURS2607_TEST_RAW_UNION",
            original_index=original, ours_index=ours,
            source_test_input_sha256=extra["source_spec"]["source_test_input_sha256"],
            new_freeze_validation=gate), overlap_graph_keys=overlap,
        old_cache_keys_modified=False, source_flip_masks_reused=False,
        source_selected_match_minima_reused=False, model_inference_performed=False, ot_recomputed=0))
