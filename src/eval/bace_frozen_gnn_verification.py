"""BACE GNN strict-flip plus MolCLR-WNode verification for B11 and B13."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
    CONNECTED_WNODE_CACHE_NAMESPACE,
    enumerate_connected_hard_deletions,
)
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.eval.bace_frozen_gnn_contracts import (
    BACEParent,
    CF_MODE,
    CLASSIFIER_TYPE,
    DATASET,
    NUM_CLASSES,
    NUM_SHARDS,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    assert_no_rf_provenance,
    assert_stage_data_boundary,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    fresh_output_dir,
    load_bace_parents,
    read_json,
    read_jsonl,
    select_parent_shard,
    sha256_file,
    stable_sha256,
    utc_now,
    validate_materialized_parent_shard,
    validate_pass_manifest,
)
from src.eval.bace_frozen_gnn_pool import _checkpoint_contract
from src.eval.counterfactual_semantics import compute_counterfactual_semantics
from src.eval.molclr_node_embeddings import checkpoint_identity
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.oracles.oracle_factory import build_oracle


VERIFICATION_STAGES = ("B11_CROSS_PARENT_VERIFIED", "B13_FINAL_EVAL")
DISTANCE_IMPLEMENTATION_VERSION = "molclr_node_wasserstein_exact_emd2_v1"


def _graph(
    featurizer: MolecularGraphFeaturizer,
    *,
    smiles: str,
    molecule_id: str,
    split: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=SOURCE_LABEL,
        molecule_id=molecule_id,
        smiles=features.canonical_smiles,
        split=split,
        graph_sha256=features.graph_sha256,
    )


def _candidate_rows_for_stage(
    stage: str,
    *,
    predecessor_root: Path,
    frozen_selection_manifest: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if stage == "B11_CROSS_PARENT_VERIFIED":
        predecessor = validate_pass_manifest(
            predecessor_root / "merge_manifest.json",
            expected_stage="B10_POOL_MERGED",
            require_no_test=True,
        )
        candidates = read_jsonl(predecessor_root / "candidate_universe.jsonl")
        source_hash = sha256_file(predecessor_root / "candidate_universe.jsonl")
    elif stage == "B13_FINAL_EVAL":
        if frozen_selection_manifest is None:
            raise ValueError("B13 candidate loading requires the B12 frozen manifest")
        predecessor = validate_pass_manifest(
            frozen_selection_manifest,
            expected_stage="B12_SELECTOR",
            require_no_test=True,
        )
        top20_path = predecessor_root / "selected_top20.json"
        top20 = read_json(top20_path)
        candidates = [dict(row) for row in top20.get("candidates", [])]
        source_hash = sha256_file(top20_path)
        declared_ids = [str(value) for value in predecessor.get("ordered_rule_ids", [])]
        observed_ids = [str(row.get("candidate_id") or "") for row in candidates]
        if len(candidates) != 20 or declared_ids != observed_ids:
            raise ValueError("B13 selected_top20 differs from frozen B12 ordering")
    else:
        raise ValueError(f"Unsupported verification stage: {stage}")
    candidate_ids = [str(row.get("candidate_id") or "") for row in candidates]
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError("Verification candidate IDs must be non-empty and unique")
    if len(candidates) < 20:
        raise ValueError("BACE verification requires at least 20 candidate rules")
    for candidate in candidates:
        assert_no_rf_provenance(candidate)
        fragment = str(
            candidate.get("canonical_fragment")
            or candidate.get("final_fragment")
            or ""
        ).strip()
        if not fragment:
            raise ValueError(f"Candidate lacks a canonical fragment: {candidate}")
        candidate["canonical_fragment"] = fragment
    return candidates, predecessor, source_hash


def _evaluate_rows(
    parents: Sequence[BACEParent],
    candidates: Sequence[dict[str, Any]],
    *,
    oracle: Any,
    featurizer: MolecularGraphFeaturizer,
    distance_provider: Any,
    oracle_batch_size: int,
    split: str,
    oracle_checkpoint_id: str,
    parent_prediction_cache: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if parent_prediction_cache is None:
        parent_graphs = [
            _graph(
                featurizer,
                smiles=parent.smiles,
                molecule_id=parent.parent_id,
                split=split,
            )
            for parent in parents
        ]
        parent_predictions = oracle.predict_records(
            parent_graphs, batch_size=oracle_batch_size
        )
    else:
        parent_predictions = []
        for parent in parents:
            cached = parent_prediction_cache.get(parent.parent_id)
            if cached is None or str(cached.get("parent_smiles")) != parent.smiles:
                raise ValueError(
                    f"GNN-before cache lacks exact parent identity: {parent.parent_id}"
                )
            probabilities = cached.get("p_before")
            if not isinstance(probabilities, list) or len(probabilities) != 2:
                raise ValueError("GNN-before cache has an invalid probability vector")
            parent_predictions.append(
                {
                    "predicted_label": int(cached["pred_before"]),
                    "probabilities": [float(value) for value in probabilities],
                }
            )
    pair_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for parent, before in zip(parents, parent_predictions, strict=True):
        pending: list[tuple[dict[str, Any], Any]] = []
        residual_graphs: list[MolecularGraphData] = []
        outcomes_by_candidate: dict[str, list[Any]] = {}
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            outcomes = enumerate_connected_hard_deletions(
                parent.smiles,
                str(candidate["canonical_fragment"]),
                parent_id=parent.parent_id,
                candidate_id=candidate_id,
            )
            outcomes_by_candidate[candidate_id] = outcomes
            for outcome in outcomes:
                if not outcome.valid or not outcome.residual_smiles:
                    continue
                pending.append((candidate, outcome))
                residual_graphs.append(
                    _graph(
                        featurizer,
                        smiles=outcome.residual_smiles,
                        molecule_id=(
                            f"{parent.parent_id}:{candidate_id}:match-{outcome.match_id}"
                        ),
                        split=f"{split}_hard_deletion_residual",
                    )
                )
        after_predictions = (
            oracle.predict_records(residual_graphs, batch_size=oracle_batch_size)
            if residual_graphs
            else []
        )
        prediction_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for (candidate, outcome), after in zip(
            pending, after_predictions, strict=True
        ):
            prediction_by_key[(str(candidate["candidate_id"]), int(outcome.match_id))] = after

        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            fragment = str(candidate["canonical_fragment"])
            outcomes = outcomes_by_candidate[candidate_id]
            candidate_matches: list[dict[str, Any]] = []
            strict_finite: list[dict[str, Any]] = []
            for outcome in outcomes:
                after = prediction_by_key.get((candidate_id, int(outcome.match_id)))
                row: dict[str, Any] = {
                    "dataset": DATASET,
                    "parent_id": parent.parent_id,
                    "parent_smiles": parent.smiles,
                    "candidate_id": candidate_id,
                    "canonical_fragment": fragment,
                    "match_index": int(outcome.match_id),
                    "match_atom_indices": list(outcome.match_atom_indices),
                    "delete_valid": bool(outcome.valid),
                    "residual_smiles": outcome.residual_smiles,
                    "residual_num_components": outcome.residual_num_components,
                    "residual_connected": outcome.residual_connected,
                    "sanitize_ok": outcome.sanitize_ok,
                    "contains_dot": outcome.contains_dot,
                    "boundary_bond_count": outcome.boundary_bond_count,
                    "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                    "pred_before": int(before["predicted_label"]),
                    "pred_after": None,
                    "p_before": list(before["probabilities"]),
                    "p_after": None,
                    "p1_before": float(before["probabilities"][SOURCE_LABEL]),
                    "p1_after": None,
                    "cf_drop": None,
                    "cf_flip": False,
                    "teacher_strict_flip": False,
                    "wnode_distance": None,
                    "distance_ok": False,
                    "oracle_backend": ORACLE_BACKEND,
                    "classifier_type": CLASSIFIER_TYPE,
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": oracle_checkpoint_id,
                    "failure_reason": outcome.invalid_reason,
                }
                if not outcome.valid or not outcome.residual_smiles or after is None:
                    row["failure_reason"] = row["failure_reason"] or "invalid_residual"
                    candidate_matches.append(row)
                    match_rows.append(row)
                    continue
                semantics = compute_counterfactual_semantics(
                    source_label=SOURCE_LABEL,
                    pred_before=before["predicted_label"],
                    pred_after=after["predicted_label"],
                    probabilities_before=before["probabilities"],
                    probabilities_after=after["probabilities"],
                    rule_id=candidate_id,
                )
                row.update(
                    {
                        "pred_after": int(after["predicted_label"]),
                        "p_after": list(after["probabilities"]),
                        "p1_after": float(after["probabilities"][SOURCE_LABEL]),
                        "cf_drop": float(semantics.cf_drop),
                        "cf_flip": bool(semantics.cf_flip),
                        "teacher_strict_flip": bool(semantics.cf_flip),
                        "failure_reason": None,
                    }
                )
                if semantics.cf_flip:
                    result = distance_provider.distance_for_action(
                        parent.smiles,
                        outcome.residual_smiles,
                        action_context={
                            "parent_id": parent.parent_id,
                            "candidate_id": candidate_id,
                            "match_index": int(outcome.match_id),
                            "match_atom_indices": list(outcome.match_atom_indices),
                            # Compatibility field required by the existing cache key.
                            # Its value is explicitly the GNN checkpoint identity.
                            "teacher_sha256": oracle_checkpoint_id,
                            "oracle_checkpoint_id": oracle_checkpoint_id,
                            "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                            "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                            "distance_implementation_version": DISTANCE_IMPLEMENTATION_VERSION,
                        },
                    )
                    value = result.get("distance")
                    finite = (
                        result.get("ok") is True
                        and value is not None
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                    )
                    row.update(
                        {
                            "wnode_distance": float(value) if finite else None,
                            "distance_ok": finite,
                            "distance_cache_hit": bool(result.get("cache_hit")),
                            "failure_reason": (
                                None
                                if finite
                                else str(result.get("error") or "wnode_distance_failed")
                            ),
                        }
                    )
                    if finite:
                        strict_finite.append(row)
                candidate_matches.append(row)
                match_rows.append(row)
            strict_finite.sort(
                key=lambda row: (
                    float(row["wnode_distance"]),
                    -float(row["cf_drop"]),
                    tuple(int(value) for value in row["match_atom_indices"]),
                )
            )
            best = strict_finite[0] if strict_finite else None
            pair_rows.append(
                {
                    "dataset": DATASET,
                    "parent_id": parent.parent_id,
                    "parent_smiles": parent.smiles,
                    "candidate_id": candidate_id,
                    "canonical_fragment": fragment,
                    "applicable": bool(outcomes),
                    "num_matches": len(outcomes),
                    "num_valid_residuals": sum(outcome.valid for outcome in outcomes),
                    "num_strict_flip_matches": sum(
                        bool(row["cf_flip"]) for row in candidate_matches
                    ),
                    "pair_strict_flip": best is not None,
                    "best_match_index": best.get("match_index") if best else None,
                    "best_match_atom_indices": (
                        best.get("match_atom_indices") if best else []
                    ),
                    "residual_smiles": best.get("residual_smiles") if best else None,
                    "pred_before": int(before["predicted_label"]),
                    "pred_after": best.get("pred_after") if best else None,
                    "p1_before": float(before["probabilities"][SOURCE_LABEL]),
                    "p1_after": best.get("p1_after") if best else None,
                    "cf_drop": best.get("cf_drop") if best else None,
                    "wnode_distance": best.get("wnode_distance") if best else None,
                    "distance_for_selection": (
                        float(best["wnode_distance"]) if best else "+inf"
                    ),
                    "failure_reason": (
                        None
                        if best
                        else (
                            "no_substructure_match"
                            if not outcomes
                            else "no_valid_strict_flip_with_finite_wnode"
                        )
                    ),
                    "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
                    "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
                    "cf_mode": CF_MODE,
                    "source_label": SOURCE_LABEL,
                    "oracle_backend": ORACLE_BACKEND,
                    "classifier_type": CLASSIFIER_TYPE,
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": oracle_checkpoint_id,
                }
            )
    return pair_rows, match_rows


def run_verification_shard(
    *,
    stage: str,
    split_path: str | Path,
    predecessor_output: str | Path,
    gnn_checkpoint: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    output_dir: str | Path,
    shard_index: int,
    wnode_cache_db: str | Path,
    node_embedding_cache_dir: str | Path,
    frozen_selection_manifest: str | Path | None = None,
    parent_before_cache: str | Path | None = None,
    parent_shard_manifest: str | Path | None = None,
    device: str = "cuda:0",
    oracle_batch_size: int = 256,
) -> dict[str, Any]:
    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in VERIFICATION_STAGES:
        raise ValueError(f"Verification stage must be one of {VERIFICATION_STAGES}")
    selection_path = (
        Path(frozen_selection_manifest).expanduser().resolve(strict=True)
        if frozen_selection_manifest is not None
        else None
    )
    predecessor = Path(predecessor_output).expanduser().resolve(strict=True)
    if normalized_stage == "B13_FINAL_EVAL" and selection_path != (
        predecessor / "frozen_selection_manifest.json"
    ).resolve(strict=True):
        raise ValueError("B13 selection manifest must belong to its B12 predecessor root")
    # This gate runs before resolving or opening the raw held-out test path.
    assert_stage_data_boundary(
        stage=normalized_stage,
        split_path=split_path,
        frozen_selection_manifest=selection_path,
    )
    split = Path(split_path).expanduser().resolve(strict=True)
    checkpoint = Path(gnn_checkpoint).expanduser().resolve(strict=True)
    molclr_project = Path(molclr_root).expanduser().resolve(strict=True)
    molclr_ckpt = Path(molclr_checkpoint).expanduser().resolve(strict=True)
    candidates, predecessor_manifest, candidate_hash = _candidate_rows_for_stage(
        normalized_stage,
        predecessor_root=predecessor,
        frozen_selection_manifest=selection_path,
    )
    assert_no_rf_provenance(predecessor_manifest)
    card, schema = _checkpoint_contract(checkpoint)
    expected_oracle = str(predecessor_manifest.get("oracle_checkpoint_hash") or "")
    if expected_oracle != str(card["checkpoint_id"]):
        raise ValueError("Verification and predecessor use different frozen GNNs")
    policy_checkpoint_hash = str(
        predecessor_manifest.get("policy_checkpoint_hash") or ""
    ).strip()
    if not policy_checkpoint_hash:
        raise ValueError("Verification predecessor lacks the frozen B7 policy hash")
    parent_cache: dict[str, dict[str, Any]] | None = None
    parent_cache_identity: dict[str, Any] | None = None
    if parent_before_cache is not None:
        if normalized_stage != "B11_CROSS_PARENT_VERIFIED":
            raise ValueError("A calibration GNN-before cache is forbidden in B13")
        cache_path = Path(parent_before_cache).expanduser().resolve(strict=True)
        cache_rows = read_jsonl(cache_path)
        expected_temperature_hash = sha256_file(checkpoint / "temperature_scaling.json")
        expected_schema_hash = sha256_file(checkpoint / "feature_schema.json")
        cache_featurizer = MolecularGraphFeaturizer(schema)
        parent_cache = {}
        for row in cache_rows:
            parent_id = str(row.get("parent_id") or "")
            if not parent_id or parent_id in parent_cache:
                raise ValueError("GNN-before cache has empty/duplicate parent identity")
            if (
                row.get("oracle_checkpoint_hash") != card["checkpoint_id"]
                or row.get("temperature_scaling_sha256") != expected_temperature_hash
                or row.get("feature_schema_sha256") != expected_schema_hash
            ):
                raise ValueError("GNN-before cache provenance differs from B11 GNN")
            canonical = cache_featurizer.featurize(
                str(row.get("parent_smiles") or "")
            ).canonical_smiles
            expected_cache_key = stable_sha256(
                {
                    "canonical_smiles": canonical,
                    "oracle_checkpoint_hash": card["checkpoint_id"],
                    "temperature_scaling_sha256": expected_temperature_hash,
                    "feature_schema_sha256": expected_schema_hash,
                }
            )
            if row.get("canonical_smiles") != canonical or row.get(
                "cache_key"
            ) != expected_cache_key:
                raise ValueError("GNN-before cache key/content identity is invalid")
            parent_cache[parent_id] = row
        parent_cache_identity = file_identity(cache_path)
    all_parents = load_bace_parents(split)
    parents = select_parent_shard(all_parents, int(shard_index))
    parent_shard_identity = None
    if parent_shard_manifest is not None:
        validate_materialized_parent_shard(
            parent_shard_manifest,
            parents=all_parents,
            shard_index=int(shard_index),
            split=(
                "calibration"
                if normalized_stage == "B11_CROSS_PARENT_VERIFIED"
                else "test"
            ),
        )
        parent_shard_identity = file_identity(parent_shard_manifest)
    if not parents:
        raise ValueError(f"{normalized_stage} parent shard {shard_index} is empty")
    output = fresh_output_dir(output_dir)
    oracle = build_oracle(
        dataset=DATASET,
        backend=ORACLE_BACKEND,
        checkpoint=checkpoint,
        device=device,
        batch_size=int(oracle_batch_size),
    )
    distance_provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=molclr_project,
            molclr_ckpt=molclr_ckpt,
            cache_db=Path(wnode_cache_db).expanduser().resolve(strict=False),
            node_emb_cache_dir=Path(node_embedding_cache_dir)
            .expanduser()
            .resolve(strict=False),
            device=device,
            distance_namespace=CONNECTED_WNODE_CACHE_NAMESPACE,
        )
    )
    try:
        pair_rows, match_rows = _evaluate_rows(
            parents,
            candidates,
            oracle=oracle,
            featurizer=MolecularGraphFeaturizer(schema),
            distance_provider=distance_provider,
            oracle_batch_size=int(oracle_batch_size),
            split=("calibration" if normalized_stage.startswith("B11") else "test"),
            oracle_checkpoint_id=str(card["checkpoint_id"]),
            parent_prediction_cache=parent_cache,
        )
        provider_stats = distance_provider.stats_dict()
    finally:
        distance_provider.close()
    expected_pairs = len(parents) * len(candidates)
    if len(pair_rows) != expected_pairs:
        raise RuntimeError(
            f"Verification shard is incomplete: {len(pair_rows)} != {expected_pairs}"
        )
    atomic_jsonl(output / "pair_details.jsonl", pair_rows)
    atomic_csv(output / "pair_details.csv", pair_rows)
    atomic_jsonl(output / "match_instances.jsonl", match_rows)
    parent_ids = [parent.parent_id for parent in parents]
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    cohort = "calibration" if normalized_stage == "B11_CROSS_PARENT_VERIFIED" else "test"
    manifest = {
        "schema_version": "bace_frozen_gnn_verification_shard_v1",
        "dataset": DATASET,
        "stage": normalized_stage,
        "status": "PASS",
        "cohort": cohort,
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "policy_checkpoint_hash": policy_checkpoint_hash,
        "oracle_checkpoint_hash": card["checkpoint_id"],
        "gnn_sha256sums_sha256": sha256_file(checkpoint / "sha256sums.txt"),
        "molclr_checkpoint_hash": sha256_file(molclr_ckpt),
        "molclr_embedding_checkpoint_identity": checkpoint_identity(molclr_ckpt),
        "candidate_source_hash": candidate_hash,
        "candidate_ids": candidate_ids,
        "candidate_ids_sha256": stable_sha256(candidate_ids),
        "candidate_count": len(candidates),
        "shard_index": int(shard_index),
        "num_shards": NUM_SHARDS,
        "shard_rule": "sorted(parent_id)_position_mod_4",
        "all_parent_ids_sha256": stable_sha256(
            sorted(parent.parent_id for parent in all_parents)
        ),
        "parent_ids": parent_ids,
        "parent_ids_sha256": stable_sha256(parent_ids),
        "parent_shard_manifest_identity": parent_shard_identity,
        "parent_count": len(parents),
        "pair_count": len(pair_rows),
        "match_count": len(match_rows),
        "strict_flip_pair_count": sum(bool(row["pair_strict_flip"]) for row in pair_rows),
        "pair_details_identity": file_identity(output / "pair_details.jsonl"),
        "match_instances_identity": file_identity(output / "match_instances.jsonl"),
        "split_identity": file_identity(split),
        "selector_frozen_before_split_load": normalized_stage == "B13_FINAL_EVAL",
        "selector_fitted_on_calibration": normalized_stage == "B13_FINAL_EVAL",
        "test_loaded": normalized_stage == "B13_FINAL_EVAL",
        "calibration_loaded": normalized_stage == "B11_CROSS_PARENT_VERIFIED",
        "all_matches_enumerated": True,
        "no_valid_strict_flip_semantics": "+inf",
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "distance_type": "node_wasserstein",
        "distance_provider_stats": provider_stats,
        "parent_before_cache_identity": parent_cache_identity,
        "parent_before_cache_used": parent_cache is not None,
        "created_at": utc_now(),
    }
    atomic_json(output / "verification_manifest.json", manifest)
    atomic_json(
        output / "oracle_provenance.json",
        {
            key: manifest[key]
            for key in (
                "dataset",
                "stage",
                "status",
                "oracle_backend",
                "classifier_type",
                "rf_oracle_used",
                "source_label",
                "num_classes",
                "policy_checkpoint_hash",
                "oracle_checkpoint_hash",
                "test_loaded",
                "calibration_loaded",
            )
        },
    )
    atomic_marker(output / "PASS", "PASS")
    return manifest


def merge_verification_shards(
    *,
    stage: str,
    shard_dirs: Sequence[str | Path],
    predecessor_output: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in VERIFICATION_STAGES:
        raise ValueError(f"Verification stage must be one of {VERIFICATION_STAGES}")
    if len(shard_dirs) != NUM_SHARDS:
        raise ValueError("Verification merge requires exactly four fixed shards")
    entries: dict[int, tuple[Path, dict[str, Any]]] = {}
    pair_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    identity_sets: dict[str, set[str]] = {
        "oracle": set(),
        "molclr": set(),
        "candidates": set(),
        "candidate_source": set(),
        "parents": set(),
        "policy": set(),
    }
    seen_parents: set[str] = set()
    for path_like in shard_dirs:
        root = Path(path_like).expanduser().resolve(strict=True)
        manifest = validate_pass_manifest(
            root / "verification_manifest.json",
            expected_stage=normalized_stage,
            require_no_test=normalized_stage == "B11_CROSS_PARENT_VERIFIED",
        )
        index = int(manifest.get("shard_index", -1))
        if not 0 <= index < NUM_SHARDS or index in entries:
            raise ValueError(f"Duplicate/invalid verification shard index: {index}")
        if int(manifest.get("num_shards", 0)) != NUM_SHARDS:
            raise ValueError("Verification shard count is not frozen to four")
        entries[index] = (root, manifest)
        identity_sets["oracle"].add(str(manifest.get("oracle_checkpoint_hash")))
        identity_sets["molclr"].add(str(manifest.get("molclr_checkpoint_hash")))
        identity_sets["candidates"].add(str(manifest.get("candidate_ids_sha256")))
        identity_sets["candidate_source"].add(
            str(manifest.get("candidate_source_hash"))
        )
        identity_sets["parents"].add(str(manifest.get("all_parent_ids_sha256")))
        identity_sets["policy"].add(str(manifest.get("policy_checkpoint_hash")))
        shard_parents = {str(value) for value in manifest.get("parent_ids", [])}
        if seen_parents & shard_parents:
            raise ValueError("Verification shards overlap in parent identity")
        seen_parents.update(shard_parents)
        local_pairs = read_jsonl(root / "pair_details.jsonl")
        local_matches = read_jsonl(root / "match_instances.jsonl")
        for identity_name, artifact_name in (
            ("pair_details_identity", "pair_details.jsonl"),
            ("match_instances_identity", "match_instances.jsonl"),
        ):
            declared = manifest.get(identity_name)
            if not isinstance(declared, Mapping) or dict(declared) != file_identity(
                root / artifact_name
            ):
                raise ValueError(
                    f"Verification shard artifact bytes differ from manifest: {root}"
                )
        if len(local_pairs) != int(manifest.get("pair_count", -1)):
            raise ValueError(f"Verification pair count differs from manifest: {root}")
        if len(local_matches) != int(manifest.get("match_count", -1)):
            raise ValueError(f"Verification match count differs from manifest: {root}")
        candidate_set = {str(value) for value in manifest.get("candidate_ids", [])}
        if any(
            str(row.get("parent_id")) not in shard_parents
            or str(row.get("candidate_id")) not in candidate_set
            for row in (*local_pairs, *local_matches)
        ):
            raise ValueError(f"Verification rows escaped their frozen shard: {root}")
        for row in (*local_pairs, *local_matches):
            assert_no_rf_provenance(row)
        pair_rows.extend(local_pairs)
        match_rows.extend(local_matches)
    if set(entries) != set(range(NUM_SHARDS)):
        raise ValueError("Verification shard set is incomplete")
    if any(len(values) != 1 for values in identity_sets.values()):
        raise ValueError(f"Verification shard identities differ: {identity_sets}")
    candidate_ids = list(entries[0][1]["candidate_ids"])
    pair_keys = [
        (str(row.get("parent_id")), str(row.get("candidate_id"))) for row in pair_rows
    ]
    if len(pair_keys) != len(set(pair_keys)):
        raise ValueError("Verification merge found duplicate parent-candidate pairs")
    expected_pairs = len(seen_parents) * len(candidate_ids)
    if len(pair_rows) != expected_pairs:
        raise ValueError(
            f"Verification Cartesian product incomplete: {len(pair_rows)} != {expected_pairs}"
        )
    pair_rows.sort(
        key=lambda row: (
            str(row["parent_id"]),
            candidate_ids.index(str(row["candidate_id"])),
        )
    )
    match_rows.sort(
        key=lambda row: (
            str(row["parent_id"]),
            candidate_ids.index(str(row["candidate_id"])),
            int(row["match_index"]),
        )
    )
    predecessor = Path(predecessor_output).expanduser().resolve(strict=True)
    if normalized_stage == "B11_CROSS_PARENT_VERIFIED":
        predecessor_manifest = validate_pass_manifest(
            predecessor / "merge_manifest.json",
            expected_stage="B10_POOL_MERGED",
            require_no_test=True,
        )
        candidates = read_jsonl(predecessor / "candidate_universe.jsonl")
        cohort = "calibration"
        test_loaded = False
    else:
        predecessor_manifest = validate_pass_manifest(
            predecessor / "frozen_selection_manifest.json",
            expected_stage="B12_SELECTOR",
            require_no_test=True,
        )
        top20 = read_json(predecessor / "selected_top20.json")
        candidates = [dict(row) for row in top20["candidates"]]
        cohort = "test"
        test_loaded = True
    if [str(row["candidate_id"]) for row in candidates] != candidate_ids:
        raise ValueError("Verification merge candidate order differs from predecessor")
    if predecessor_manifest.get("oracle_checkpoint_hash") != next(
        iter(identity_sets["oracle"])
    ) or predecessor_manifest.get("policy_checkpoint_hash") != next(
        iter(identity_sets["policy"])
    ):
        raise ValueError("Verification merge changed frozen policy/GNN identity")
    declared_source_hash = (
        predecessor_manifest.get("candidate_universe_hash")
        if normalized_stage == "B11_CROSS_PARENT_VERIFIED"
        else predecessor_manifest.get("selected_top20_hash")
    )
    if declared_source_hash != next(iter(identity_sets["candidate_source"])):
        raise ValueError("Verification candidate bytes differ from predecessor freeze")
    output = fresh_output_dir(output_dir)
    atomic_jsonl(output / "pair_matrix.jsonl", pair_rows)
    atomic_jsonl(output / "match_instances.jsonl", match_rows)
    atomic_csv(output / "pair_details.csv", pair_rows)
    atomic_jsonl(output / "selected_candidate_universe.jsonl", candidates)
    strict_count = sum(bool(row["pair_strict_flip"]) for row in pair_rows)
    summary = {
        "schema_version": "bace_frozen_gnn_verification_summary_v1",
        "dataset": DATASET,
        "stage": normalized_stage,
        "status": "PASS",
        "parent_count": len(seen_parents),
        "selected_candidate_count": len(candidates),
        "pair_count": len(pair_rows),
        "strict_flip_pair_count": strict_count,
        "match_count": len(match_rows),
        "test_loaded": test_loaded,
        "calibration_loaded": not test_loaded,
        "run_complete": True,
    }
    atomic_json(output / "summary.json", summary)
    oracle_hash = next(iter(identity_sets["oracle"]))
    molclr_hash = next(iter(identity_sets["molclr"]))
    policy_hash = next(iter(identity_sets["policy"]))
    manifest = {
        "schema_version": "bace_frozen_gnn_verification_merge_v1",
        "dataset": DATASET,
        "stage": normalized_stage,
        "status": "PASS",
        "oracle_backend": ORACLE_BACKEND,
        "classifier_type": CLASSIFIER_TYPE,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "policy_checkpoint_hash": policy_hash,
        "oracle_checkpoint_hash": oracle_hash,
        "molclr_checkpoint_hash": molclr_hash,
        "test_loaded": test_loaded,
        "calibration_loaded": not test_loaded,
        "selector_fitted_on_calibration": normalized_stage == "B13_FINAL_EVAL",
        "test_used_only_after_freeze": normalized_stage == "B13_FINAL_EVAL",
        "inputs": {
            "cohort_name": cohort,
            "predecessor_manifest": str(predecessor),
            "shard_manifests": [
                file_identity(entries[index][0] / "verification_manifest.json")
                for index in range(NUM_SHARDS)
            ],
        },
        "parent_count": len(seen_parents),
        "selected_candidate_count": len(candidates),
        "pair_count": len(pair_rows),
        "strict_flip_pair_count": strict_count,
        "pair_matrix_hash": sha256_file(output / "pair_matrix.jsonl"),
        "candidate_universe_hash": sha256_file(
            output / "selected_candidate_universe.jsonl"
        ),
        "all_matches_enumerated": True,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "created_at": utc_now(),
        "run_complete": True,
    }
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "matrix_manifest.json", manifest)
    atomic_json(
        output / "oracle_provenance.json",
        {
            key: manifest[key]
            for key in (
                "dataset",
                "stage",
                "status",
                "oracle_backend",
                "classifier_type",
                "rf_oracle_used",
                "source_label",
                "num_classes",
                "policy_checkpoint_hash",
                "oracle_checkpoint_hash",
                "test_loaded",
                "calibration_loaded",
            )
        },
    )
    if normalized_stage == "B13_FINAL_EVAL":
        # Imported lazily to keep the verification kernel independent from the
        # CPU selector. Finalization reads only merged shard artifacts + B12.
        from src.eval.bace_frozen_gnn_selection import finalize_b13_output

        finalize_b13_output(b13_output=output, b12_output=predecessor)
    atomic_marker(output / "PASS", "PASS")
    return manifest


__all__ = [
    "VERIFICATION_STAGES",
    "merge_verification_shards",
    "run_verification_shard",
]
