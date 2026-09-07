"""One-time raw train-cost migration from the already sealed Reach campaign."""
from pathlib import Path
import math

from src.eval.bace_frozen_gnn_contracts import load_bace_parents, read_json, stable_sha256, sha256_file
from src.eval.bace_reach_raw_binding import current_raw_contract
from src.eval.bace_reach_v2 import unseal, seal
from src.ablations.gnn.reach_raw_distance_reuse import SCHEMA, graph_key, kernel_identity_proof


def migrate_train(campaign, portable, output, source_commit):
    campaign, output = Path(campaign), Path(output)
    contract = unseal(campaign / "search_contract.json")
    pool = unseal(campaign / "candidate_freeze.json")
    if (pool["search_contract_sha256"] != contract["self_sha256"] or pool["test_opened"] is not False
            or pool["calibration_opened_during_search"] is not False):
        raise ValueError("SEALED_TRAIN_ONLY_SOURCE_REQUIRED")
    raw = current_raw_contract(contract, read_json(portable))
    proof = kernel_identity_proof(Path(__file__).resolve().parents[2], source_commit)
    binding = stable_sha256(dict(search=contract["self_sha256"], pool=pool["self_sha256"], raw=raw, kernels=proof))
    if output.exists():
        saved = unseal(output)
        if saved["binding_sha256"] != binding:
            raise ValueError("RAW_TRAIN_INDEX_BINDING_CHANGED")
        return saved
    parents = load_bace_parents(contract["paths"]["train"], source_label=1)
    if len(parents) != pool["train_parent_count"]:
        raise ValueError("TRAIN_PARENT_BINDING")
    raw_sha = stable_sha256(raw)
    costs, members, finite = {}, [], 0
    for parent in parents:
        token = stable_sha256(dict(id=parent.parent_id, smiles=parent.smiles))[:24]
        path = campaign / "train" / (token + ".json")
        saved = unseal(path)
        if saved["search_contract_sha256"] != contract["self_sha256"] or saved["parent_id"] != parent.parent_id:
            raise ValueError("TRAIN_PARENT_RECORD_BINDING")
        source = dict(path=str(path), sha256=sha256_file(path), self_sha256=saved["self_sha256"])
        members.append(source)
        for row in saved["old_pool_matches"]:
            if row.get("distance_ok") is not True:
                continue
            value = row["wnode_distance"]
            if (not math.isfinite(value) or value < 0 or row["delete_valid"] is not True
                    or row["sanitize_ok"] is not True or row["residual_connected"] is not True
                    or row["parent_id"] != parent.parent_id or row["oracle_checkpoint_hash"] != contract["oracle_binding"]):
                raise ValueError("RAW_TRAIN_COST_PROVENANCE_GAP")
            key, p, cf = graph_key(row["parent_smiles"], row["residual_smiles"], raw_sha)
            if key != graph_key(parent.smiles, row["residual_smiles"], raw_sha)[0]:
                raise ValueError("TRAIN_PARENT_GRAPH_DRIFT")
            if key in costs and costs[key]["distance"] != value:
                raise ValueError("RAW_TRAIN_COST_CONFLICT")
            item = costs.setdefault(key, dict(parent=p, residual=cf, distance=value, source_records=[]))
            item["source_records"].append(dict(source_parent_member=str(path), source_parent_sha256=source["sha256"],
                source_match_sha256=stable_sha256(row), original_action_context={k: row[k] for k in
                    ("parent_id", "candidate_id", "match_index", "match_atom_indices", "oracle_checkpoint_hash", "action_semantics_version")}))
            finite += 1
    return seal(output, dict(schema=SCHEMA, split="train", state="RAW_COST_ADOPTION_INDEX_SEALED_NOT_SCIENCE_PASS",
        binding_sha256=binding, raw_contract=raw, raw_contract_sha256=raw_sha, kernel_identity=proof,
        graph_costs=costs, raw_cost_count=len(costs), source_parent_units=len(members),
        source_finite_match_records=finite, source_spec=dict(campaign=str(campaign), members=members),
        new_test_freeze_sha256=None, model_inference_performed=False, ot_recomputed=0,
        source_flip_masks_reused=False, source_selected_match_minima_reused=False, test_opened=False))
