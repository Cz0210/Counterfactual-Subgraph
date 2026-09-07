"""Original BACE native full-graph pools under the adopted corrected GIN.

This is an evaluation adapter, not a generator or new recourse algorithm.
GCF's 21958 graphs and ComRecGC's 44 native medoids are the full original
calibration inputs, not the old GINE-selected twenty. No deletion mapping is
invented for these complete-graph interventions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.counterfactual_semantics import compute_counterfactual_semantics


NATIVE_CONTRACTS = {
    "gcfexplainer": ("full_counterfactual_graph", "official_vrrw_neurosed_greedy_fullgraph_v1", 21958),
    "comrecgc": ("native_common_recourse_fullgraph", "official_comrecgc_lineage_unique_transition_medoid_v1", 44),
}
METHOD_NAMES = {"gcfexplainer": "GCFExplainer", "comrecgc": "ComRecGC"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_original_pool(method: str, candidates: Sequence[Mapping[str, Any]], *, expected_count: int | None = None) -> None:
    """Reject an old selected20, a foreign action, or an altered original pool."""
    kind, semantics, production_count = NATIVE_CONTRACTS[method]
    count = production_count if expected_count is None else expected_count
    if len(candidates) != count:
        raise ValueError(f"ORIGINAL_PRESELECT_POOL_COUNT_MISMATCH:{method}:{len(candidates)}!={count}")
    ids = [str(c.get("candidate_id") or "") for c in candidates]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("ORIGINAL_POOL_IDS_INVALID")
    for candidate in candidates:
        if candidate.get("action_kind") != kind or candidate.get("action_semantics") != semantics:
            raise ValueError("NATIVE_ACTION_CONTRACT_CHANGED")
        if not candidate.get("canonical_smiles"):
            raise ValueError("NATIVE_COMPLETE_GRAPH_MISSING")
        if candidate.get("generation_split") != "train" or candidate.get("test_loaded") is not False:
            raise ValueError("ORIGINAL_POOL_TRAIN_BOUNDARY_MISSING")
        if candidate.get("candidate_set_preselected") is not False:
            raise ValueError("FINAL_SELECTED_POOL_CANNOT_REPLACE_ORIGINAL_CALIBRATION_INPUT")
        if method == "comrecgc" and candidate.get("lineage_validated") is not True:
            raise ValueError("COMRECGC_NATIVE_LINEAGE_BINDING_MISSING")


def _binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    required = ("model_sha256", "temperature_sha256", "feature_schema_sha256")
    if binding.get("backbone") != "gin" or any(not binding.get(k) for k in required):
        raise ValueError("CORRECTED_GIN_PREDICTION_BINDING_REQUIRED")
    return {"backbone": "gin", **{k: str(binding[k]) for k in required}}


def _check_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    p = list(map(float, row["probabilities"]))
    logits = list(map(float, row["logits"]))
    pred = int(row["predicted_label"])
    if len(p) != 2 or len(logits) != 2 or not all(math.isfinite(x) for x in p + logits):
        raise ValueError("GIN_NATIVE_PREDICTION_NOT_FINITE_BINARY")
    if any(x < 0 or x > 1 for x in p) or abs(sum(p) - 1) > 1e-6 or pred != max(range(2), key=p.__getitem__):
        raise ValueError("GIN_NATIVE_PREDICTION_CLASS_MAPPING_INVALID")
    return {"probabilities": p, "logits": logits, "predicted_label": pred}


def prepare_candidate_predictions(candidates: Sequence[Mapping[str, Any]], oracle: Any, featurizer: Any,
                                  *, oracle_binding: Mapping[str, Any], batch_size: int = 64,
                                  graph_builder: Any = None) -> dict[str, Any]:
    """Predict each distinct complete graph once, in bounded GIN-only batches.

    The returned small JSON-compatible cache can be sealed by the existing
    driver and reused across parent chunks. It contains no parent flip mask or
    distance and cannot be adopted with another weight/temperature/schema.
    """
    binding = _binding(oracle_binding)
    if batch_size < 1:
        raise ValueError("POSITIVE_BATCH_SIZE_REQUIRED")
    if graph_builder is None:
        from src.eval.bace_native_baseline_gnn import _graph
        graph_builder = _graph
    ordered = list(dict.fromkeys(str(c["canonical_smiles"]) for c in candidates))
    by_smiles: dict[str, Any] = {}
    for start in range(0, len(ordered), batch_size):
        part = ordered[start:start + batch_size]
        graphs = [graph_builder(featurizer, smiles=s, molecule_id=f"native-gin:{start+i}",
                                split="train_generated_native_fullgraph") for i, s in enumerate(part)]
        rows = oracle.predict_records(graphs, batch_size=batch_size)
        if len(rows) != len(part):
            raise ValueError("GIN_NATIVE_PREDICTION_COUNT_MISMATCH")
        by_smiles.update((s, _check_prediction(row)) for s, row in zip(part, rows, strict=True))
    return {"schema_version": "bace_gin_native_prediction_cache_v1", "oracle_binding": binding,
            "oracle_binding_sha256": _digest(binding), "unique_graph_count": len(ordered),
            "candidate_count": len(candidates),
            "candidate_identity_sha256": _digest([[c["candidate_id"], c["canonical_smiles"]] for c in candidates]),
            "candidate_smiles": {str(c["candidate_id"]): str(c["canonical_smiles"]) for c in candidates},
            "predictions_by_smiles": by_smiles, "parent_flip_masks_reused": False}


def evaluate_parent(parent: Any, candidates: Sequence[Mapping[str, Any]], oracle: Any,
                    featurizer: Any, distance: Any, split: str, output: Any = None,
                    before: Mapping[str, Any] | None = None, batch_size: int = 64, *,
                    method: str, oracle_binding: Mapping[str, Any],
                    candidate_predictions: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Original fullgraph counterfactual semantics, with fresh GIN strict flips.

    The second return contains native application records, not fictional atom
    deletion matches. Complete graph IDs and costs are available to raw-cost
    migration independently of current classifier decisions.
    """
    del output
    kind, semantics_version, _ = NATIVE_CONTRACTS[method]
    binding = _binding(oracle_binding)
    if candidate_predictions.get("oracle_binding") != binding or candidate_predictions.get("oracle_binding_sha256") != _digest(binding):
        raise ValueError("GIN_NATIVE_CACHE_BINDING_MISMATCH")
    if split not in {"calibration", "test"}:
        raise ValueError("EXPLICIT_EVALUATION_SPLIT_REQUIRED")
    if before is None:
        from src.eval.bace_native_baseline_gnn import _graph
        graph = _graph(featurizer, smiles=parent.smiles, molecule_id=parent.parent_id, split=split)
        before = oracle.predict_records([graph], batch_size=batch_size)[0]
    before = _check_prediction(before)
    pairs, applications = [], []
    for candidate in candidates:
        if candidate.get("action_kind") != kind or candidate.get("action_semantics") != semantics_version:
            raise ValueError("NATIVE_ACTION_CONTRACT_CHANGED")
        smiles, candidate_id = str(candidate["canonical_smiles"]), str(candidate["candidate_id"])
        if candidate_predictions.get("candidate_smiles", {}).get(candidate_id) != smiles:
            raise ValueError("GIN_NATIVE_CACHE_CANDIDATE_IDENTITY_MISMATCH")
        if smiles not in candidate_predictions["predictions_by_smiles"]:
            raise ValueError("GIN_NATIVE_CACHE_GRAPH_MISSING")
        after = _check_prediction(candidate_predictions["predictions_by_smiles"][smiles])
        semantics = compute_counterfactual_semantics(source_label=1, pred_before=before["predicted_label"],
                pred_after=after["predicted_label"], probabilities_before=before["probabilities"],
                probabilities_after=after["probabilities"], rule_id=candidate_id)
        value, reason, hit = None, None, False
        if semantics.cf_flip:
            # This is exactly the original complete-graph distance operation.
            # Never pass a fabricated deletion match/context to the provider.
            result = distance.distance(parent.smiles, smiles)
            raw = result.get("distance")
            if result.get("ok") is True and raw is not None and math.isfinite(float(raw)) and float(raw) >= 0:
                value, hit = float(raw), bool(result.get("cache_hit"))
            else:
                # A real strict flip with a missing/invalid cost is an unfinished
                # evaluation, not evidence of zero threshold coverage. Leave the
                # parent uncommitted so a later exact repair can resume it.
                raise ValueError(
                    "STRICT_FLIP_RAW_DISTANCE_FAILURE_NOT_ZERO_COVERAGE:"
                    f"{parent.parent_id}:{candidate_id}:"
                    f"{result.get('error') or 'wnode_distance_failed'}"
                )
        else:
            reason = "frozen_gin_not_strict_flip"
        row = {"dataset": "bace", "method": METHOD_NAMES[method], "method_id": method,
               "parent_id": parent.parent_id, "parent_smiles": parent.smiles, "candidate_id": candidate_id,
               "canonical_smiles": smiles, "canonical_fragment": smiles, "residual_smiles": smiles,
               "candidate_rank": candidate.get("rank"), "native_rank": candidate.get("native_rank"),
               "action_kind": kind, "action_semantics": semantics_version,
               "native_record_kind": "complete_graph_intervention", "applicable": True,
               "pred_before": before["predicted_label"], "pred_after": after["predicted_label"],
               "p_before": before["probabilities"], "p_after": after["probabilities"],
               "logits_before": before["logits"], "logits_after": after["logits"],
               "p1_before": before["probabilities"][1], "p1_after": after["probabilities"][1],
               "cf_drop": float(semantics.cf_drop), "cf_flip": bool(semantics.cf_flip),
               "pair_strict_flip": bool(semantics.cf_flip and value is not None),
               "wnode_distance": value, "distance_for_selection": value if value is not None else "+inf",
               "failure_reason": reason, "cf_mode": "strict_flip", "source_label": 1,
               "oracle_backend": "gnn", "classifier_family": "gin", "classifier_type": "gnn",
               "rf_oracle_used": False, "oracle_checkpoint_hash": binding["model_sha256"],
               "temperature_sha256": binding["temperature_sha256"], "split": split}
        pairs.append(row)
        applications.append({**row, "distance_ok": value is not None, "distance_cache_hit": hit,
                             "match_index": None, "match_atom_indices": None,
                             "delete_valid": None, "teacher_strict_flip": bool(semantics.cf_flip),
                             "operation_is_deletion": False})
    return pairs, applications


def select_order(matrix: str | Path, selector_context: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the original native calibration selector, not Reach-first.

    The common driver supplies its hash-verified frozen original threshold
    bundle/provenance and fresh GIN calibration matrix. Original four-variant
    calibration-only decision and deterministic tie breaks remain unchanged.
    """
    from src.eval.mutagenicity_wnode_selector import run_mutagenicity_wnode_selector, threshold_bundle_from_dict
    method = str(selector_context["method"])
    global_native = (method == "globalgce"
        and selector_context.get("native_attachment_contract") == "bace_globalgce_gin_aplus_attachment_v1"
        and selector_context.get("original_global_selector_verified") is True)
    if (method not in NATIVE_CONTRACTS and not global_native) or selector_context.get("test_loaded") is not False:
        raise ValueError("NATIVE_SELECTOR_CALIBRATION_CONTEXT_REQUIRED")
    config = dict(selector_context["original_selector_config"])
    frozen = {"top_k": 20, "table_k": 10, "seed": 13, "local_swap_passes": 2,
              "parent_limit": 0, "candidate_limit": 0, "forbid_test": True}
    if any(config.get(k) != v for k, v in frozen.items()) or config.get("prefix_weights") != [1.0] * 10 + [0.5] * 10:
        raise ValueError("ORIGINAL_NATIVE_SELECTOR_CONFIG_CHANGED")
    out = Path(selector_context["output_root"])
    run_mutagenicity_wnode_selector(matrix_run_dir=matrix, output_dir=out,
            **frozen, prefix_weights=config["prefix_weights"],
            frozen_thresholds=threshold_bundle_from_dict(selector_context["thresholds"]),
            frozen_threshold_provenance=selector_context["threshold_provenance"])
    decision = json.loads((out / "calibration_decision.json").read_text())
    selected = json.loads((out / "variants" / decision["selected_variant"] / "selected_top20.json").read_text())
    rows = selected["candidates"]
    return {"ordered_rule_ids": [r["candidate_id"] for r in rows], "candidates": rows,
            "selected_variant": decision["selected_variant"], "decision_rule": decision["decision_rule"],
            "selector_implementation": "src.eval.mutagenicity_wnode_selector.run_mutagenicity_wnode_selector",
            "classifier_family": "gin", "test_loaded": False, "original_selector_reused": True}
