"""Bind unchanged current BACE raw-distance inputs to a portable old index.

The old index is evidence, not the authority for the current graph schema,
MolCLR weights or source. Small current files are checked; a prior immutable
reference receipt supplies the weights digest, avoiding repeated model hashing.
"""
from pathlib import Path
import copy
import hashlib
import json
import math

from src.eval.bace_frozen_gnn_contracts import read_json, sha256_file, stable_sha256


def bound_json(item):
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError("RAW_REUSE_DESCRIPTOR_CHANGED:" + str(path))
    return read_json(path)


def current_raw_contract(contract, portable):
    """Derive values from actual current paths, never copy an index contract."""
    ref_path = Path(contract["paths"]["reference"])
    if sha256_file(ref_path) != contract["reference_sha256"]:
        raise ValueError("FROZEN_REFERENCE_RECEIPT_CHANGED")
    reference = read_json(ref_path)
    down = reference["frozen_downstream"]
    paths = contract["paths"]
    if (down["wnode_config"] != contract["wnode_config"]
        or down["molclr_sha"] != contract["molclr_sha256"]
        or Path(down["molclr_root"]).resolve() != Path(paths["molclr_checkpoint"]).resolve()
        or Path(down["gine_checkpoint"]).parent.resolve() != Path(paths["oracle"]).resolve()):
        raise ValueError("CURRENT_RAW_INPUTS_NOT_REFERENCE_BOUND")
    schema = Path(paths["oracle"]) / "feature_schema.json"
    weights = Path(paths["molclr_checkpoint"])
    prefix = portable["molclr_source_root"] + "/"
    source = {}
    for rel in portable["files"]:
        if not rel.startswith(prefix):
            continue
        suffix = Path(rel.removeprefix(prefix))
        if suffix.is_absolute() or ".." in suffix.parts:
            raise ValueError("UNSAFE_MOLCLR_SOURCE_SUFFIX")
        actual = Path(paths["molclr_source"]) / suffix
        source[rel] = {"sha256": sha256_file(actual), "size": actual.stat().st_size}
    if prefix + "models/ginet_molclr.py" not in source:
        raise ValueError("CURRENT_MOLCLR_IMPLEMENTATION_NOT_BOUND")
    actual = {"wnode": dict(contract["wnode_config"]),
        "feature_schema": {"sha256": sha256_file(schema), "size": schema.stat().st_size},
        "molclr_checkpoint": {"sha256": down["molclr_sha"], "size": weights.stat().st_size},
        "molclr_source": source}
    accepted = {"wnode": portable["wnode_config"],
        "feature_schema": portable["files"][portable["feature_schema_path"]],
        "molclr_checkpoint": portable["files"][portable["molclr_checkpoint_path"]],
        "molclr_source": {r: value for r, value in portable["files"].items() if r.startswith(prefix)}}
    if actual != accepted:
        raise ValueError("CURRENT_RAW_CONTRACT_DIFFERS_FROM_ACCEPTED_SOURCE")
    return actual


def wrap_raw_distance(delegate, *, contract, descriptor, split, repo, final_freeze=None):
    """A caller supplies the real new freeze for any old-test migration use."""
    if split not in {"calibration", "test"}:
        raise ValueError("OLD_RAW_INDEX_MUST_NOT_GUIDE_TRAIN_SEARCH")
    portable = bound_json(descriptor["portable_manifest"])
    source_spec = bound_json(descriptor["source_spec"])["raw_distance_source"]
    index = bound_json(descriptor["index"])
    if index.get("split") != split or index.get("source_spec") != source_spec:
        raise ValueError("RAW_INDEX_SPLIT_OR_SOURCE_CHANGED")
    if split == "test":
        if not final_freeze:
            raise ValueError("RAW_TEST_REUSE_REQUIRES_FINAL_FREEZE")
        from src.eval.bace_reach_v2 import unseal
        frozen = unseal(Path(final_freeze))
        if (frozen.get("state") != "REACH_V2_FINAL_CONFIGURATION_FROZEN"
            or frozen.get("test_opened") is not False
            or index.get("new_test_freeze_sha256") != sha256_file(final_freeze)):
            raise ValueError("RAW_TEST_INDEX_NOT_BOUND_TO_ACTUAL_NEW_FREEZE")
    actual = current_raw_contract(contract, portable)
    from src.ablations.gnn.reach_raw_distance_reuse import VerifiedRawGraphDistance
    result = VerifiedRawGraphDistance(delegate, index=index, current_raw_contract=actual, repo=Path(repo))
    result.current_input_binding = {"current_raw_contract_sha256": stable_sha256(actual),
        "frozen_reference_sha256": contract["reference_sha256"],
        "weights_digest_source": "EXISTING_IMMUTABLE_REFERENCE_RECEIPT",
        "weights_rehashed": False, "small_current_sources_verified": True,
        "source_descriptor": descriptor, "source_flip_masks_reused": False}
    return result


def export_calibration_raw_union(campaign, output, *, descriptor, repo, science_commit):
    """Migrate sealed Ours raw costs once; preserve both source provenances.

    No model is loaded, no distance is calculated, and no test record is read.
    The same existing raw-index consumer can then reuse this union for each
    newly frozen backbone's independently recomputed strict flips/match minima.
    """
    from src.eval.bace_reach_v2 import unseal, seal
    from src.ablations.gnn.reach_raw_distance_reuse import SCHEMA, graph_key, kernel_identity_proof
    from src.eval.bace_frozen_gnn_contracts import load_bace_parents
    contract = unseal(campaign / "search_contract.json")
    pool = unseal(campaign / "candidate_freeze.json")
    selector = unseal(campaign / "selector_freeze.json")
    if (selector["test_opened"] is not False or selector["candidate_freeze_sha256"] != pool["self_sha256"]
        or pool["search_contract_sha256"] != contract["self_sha256"]):
        raise ValueError("OURS_CALIBRATION_NOT_FROZEN_TO_SEARCH")
    old = bound_json(descriptor["index"])
    raw = current_raw_contract(contract, bound_json(descriptor["portable_manifest"]))
    proof = kernel_identity_proof(Path(repo), science_commit)
    if (old.get("schema") != SCHEMA or old.get("split") != "calibration"
        or old.get("self_sha256") != stable_sha256({k: v for k, v in old.items() if k != "self_sha256"})
        or old["raw_contract"] != raw or old["kernel_identity"] != proof
        or old["source_spec"] != bound_json(descriptor["source_spec"])["raw_distance_source"]):
        raise ValueError("OLD_CALIBRATION_RAW_INDEX_NOT_SAME_BOUND_KERNEL")
    binding = stable_sha256({"old_index": old["self_sha256"], "search": contract["self_sha256"],
        "pool": pool["self_sha256"], "selector": selector["self_sha256"], "science_commit": science_commit})
    if output.exists():
        existing = unseal(output)
        if existing["binding_sha256"] != binding:
            raise ValueError("RAW_CALIBRATION_UNION_ALREADY_EXISTS_DIFFERENT_BINDING")
        return existing
    parents = load_bace_parents(contract["paths"]["calibration"], source_label=contract["source_label"])
    costs, members, finite, overlaps = copy.deepcopy(old["graph_costs"]), [], 0, 0
    contract_sha = stable_sha256(raw)
    for parent in parents:
        path = campaign / "calibration" / (stable_sha256(parent.parent_id)[:24] + ".json")
        data = path.read_bytes()
        record = json.loads(data)
        if (record.get("self_sha256") != stable_sha256({k: v for k, v in record.items() if k != "self_sha256"})
            or record["pool_sha256"] != pool["candidate_universe_sha256"]
            or any(p["parent_id"] != parent.parent_id for p in record["pairs"])):
            raise ValueError("SEALED_OURS_CALIBRATION_PARENT_CONFLICT")
        member = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                  "self_sha256": record["self_sha256"], "parent_id": parent.parent_id}
        members.append(member)
        for row in record["matches"]:
            if row.get("distance_ok") is not True:
                continue
            value = row.get("wnode_distance")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or row.get("delete_valid") is not True
                or row.get("sanitize_ok") is not True or row.get("residual_connected") is not True
                or row["parent_id"] != parent.parent_id
                or row["oracle_checkpoint_hash"] != contract["oracle_binding"]):
                raise ValueError("OURS_FINITE_RAW_COST_SOURCE_RECORD_INVALID")
            key, parent_graph, residual_graph = graph_key(row["parent_smiles"], row["residual_smiles"], contract_sha)
            if key != graph_key(parent.smiles, row["residual_smiles"], contract_sha)[0]:
                raise ValueError("OURS_RAW_COST_PARENT_GRAPH_NOT_SOURCE_BOUND")
            provenance = {"source_parent_member": str(path), "source_parent_sha256": member["sha256"],
                "source_match_sha256": stable_sha256(row), "original_action_context": {k: row[k] for k in
                    ("parent_id", "candidate_id", "match_index", "match_atom_indices",
                     "oracle_checkpoint_hash", "action_semantics_version")}}
            if key in costs:
                if costs[key]["distance"] != value:
                    raise ValueError("RAW_GRAPH_COST_UNION_NUMERICAL_CONFLICT:" + key)
                overlaps += key in old["graph_costs"]
            else:
                costs[key] = {"parent": parent_graph, "residual": residual_graph, "distance": value, "source_records": []}
            costs[key]["source_records"].append(provenance)
            finite += 1
    return seal(output, {"schema": SCHEMA, "state": "RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS",
        "binding_sha256": binding, "split": "calibration", "raw_contract": raw,
        "raw_contract_sha256": contract_sha, "kernel_identity": proof, "graph_costs": costs,
        "raw_cost_count": len(costs), "source_parent_units": old["source_parent_units"] + len(members),
        "source_finite_match_records": old["source_finite_match_records"] + finite,
        "new_test_freeze_sha256": None,
        "source_spec": {"kind": "ACCEPTED_GNN_PLUS_SEALED_OURS_CALIBRATION_RAW_UNION",
            "old_accepted_source": old["source_spec"], "old_index": descriptor["index"],
            "old_index_self_sha256": old["self_sha256"], "ours_science_commit": science_commit,
            "ours_search_contract_sha256": contract["self_sha256"],
            "ours_selector_freeze_sha256": selector["self_sha256"],
            "ours_calibration_pairs_sha256": selector["calibration_pairs_sha256"],
            "ours_parent_sources": members},
        "ours_new_finite_match_records": finite, "ours_unique_graph_costs_added": len(costs) - old["raw_cost_count"],
        "ours_requests_overlapping_old_index": overlaps, "old_cache_keys_modified": False,
        "source_flip_masks_reused": False, "source_selected_match_minima_reused": False,
        "model_inference_performed": False, "ot_recomputed": 0, "test_opened": False})
