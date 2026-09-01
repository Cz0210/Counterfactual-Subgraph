"""Frozen-GINE/WNode evaluation for native BACE full-graph baselines.

GCFExplainer and ComRecGC both emit complete molecular graphs, although their
generation/search semantics differ.  This evaluator keeps those native action
identities and only maps their outputs into the common parent/candidate metric
schema.  It never converts a graph into a deletion fragment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.baselines.bace_gnn_baseline_contracts import (
    CF_MODE,
    DATASET,
    SOURCE_LABEL,
    assert_gine_clean_manifest,
    baseline_spec,
    normalize_method,
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    apply_rule_to_parent,
)
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.eval.bace_frozen_gnn_contracts import (
    NUM_SHARDS,
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
)
from src.eval.counterfactual_semantics import compute_counterfactual_semantics
from src.eval.molclr_node_embeddings import checkpoint_identity
from src.eval.mutagenicity_wnode_selector import (
    ThresholdBundle,
    run_mutagenicity_wnode_selector,
    threshold_bundle_from_dict,
)
from src.eval.node_wasserstein_distance import (
    MolCLRNodeWassersteinConfig,
    MolCLRNodeWassersteinDistance,
)
from src.oracles.oracle_factory import build_oracle
from src.oracles.gnn_oracle import (
    EXPECTED_EMPTY_GRAPH_SEQUENCE,
    UNEXPECTED_EMPTY_GRAPH_SEQUENCE,
)


CALIBRATION_STAGE = "BASELINE_CALIBRATION_VERIFY"
TEST_STAGE = "BASELINE_TEST_EVAL"
SELECTION_STAGE = "BASELINE_CALIBRATION_SELECTOR"
FINAL_STAGE = "BASELINE_FINAL_FREEZE"
DISTANCE_NAMESPACE = "bace_native_fullgraph_frozen_gine_wnode_v1"
_CALIBRATION_PREDECESSOR_ROOT_FIELDS = (
    "train_candidates_root",
    "source_train_candidates_root",
    "generation_root",
)
_FORBIDDEN_PREDECESSOR_COMPONENTS = {"merged", "_native_aux"}
BACE_OURS_B12_THRESHOLD_CONFIG_SHA256 = (
    "37d7a265ee53fc0c31edaf59f8b412f41c79c62af4941d4ddf1f3e66c4afa427"
)


def _minimum_candidate_count(method_id: str) -> int:
    """Return the preregistered resource-capped minimum for one baseline."""

    return 10 if method_id in {"comrecgc", "globalgce"} else 20


def _manifest_value(manifest: Mapping[str, Any], field: str) -> Any:
    """Read one lineage field from the shard or its explicit inputs block."""

    if manifest.get(field) not in (None, ""):
        return manifest[field]
    inputs = manifest.get("inputs")
    if isinstance(inputs, Mapping) and inputs.get(field) not in (None, ""):
        return inputs[field]
    return None


def _lineage_path(value: Any, *, field: str) -> Path:
    if isinstance(value, Mapping):
        for key in ("path", "root", "output_root"):
            if value.get(key) not in (None, ""):
                value = value[key]
                break
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Shard lineage field {field} is not a path")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Shard lineage field {field} must be absolute: {path}")
    if path.name in {"candidate_universe.jsonl", "run_manifest.json"}:
        path = path.parent
    return path


def _reject_forbidden_predecessor(path: Path) -> None:
    forbidden = _FORBIDDEN_PREDECESSOR_COMPONENTS.intersection(path.parts)
    if forbidden:
        raise ValueError(
            "BACE GCF predecessor cannot resolve through merged/_native_aux: "
            f"{path}"
        )


def _has_calibration_predecessor_contract(path: Path) -> bool:
    return (path / "run_manifest.json").is_file() and (
        path / "candidate_universe.jsonl"
    ).is_file()


def _resolve_declared_predecessor_root(
    *, field: str, value: Any, generation_attempt_id: str | None
) -> Path:
    declared = _lineage_path(value, field=field)
    options = [declared]
    if generation_attempt_id and declared.name != generation_attempt_id:
        options.append(declared / generation_attempt_id)
        options.append(declared / "train_candidates" / generation_attempt_id)
    resolved: list[Path] = []
    for option in options:
        _reject_forbidden_predecessor(option)
        if _has_calibration_predecessor_contract(option):
            candidate = option.resolve(strict=True)
            _reject_forbidden_predecessor(candidate)
            if candidate not in resolved:
                resolved.append(candidate)
    if len(resolved) != 1:
        raise ValueError(
            f"Shard lineage field {field} does not identify exactly one train "
            f"candidate attempt: {declared}"
        )
    return resolved[0]


def _fallback_predecessor_root(shard_root: Path) -> Path:
    calibration_root = next(
        (
            ancestor
            for ancestor in (shard_root, *shard_root.parents)
            if ancestor.name == "calibration"
        ),
        None,
    )
    if calibration_root is None:
        raise ValueError(
            f"Cannot derive BACE GCF run root from calibration shard: {shard_root}"
        )
    fallback = calibration_root.parent / "train_candidates" / "attempt-0"
    _reject_forbidden_predecessor(fallback)
    if not _has_calibration_predecessor_contract(fallback):
        raise ValueError(
            "Shard manifest lacks predecessor lineage and the only authorized "
            f"fallback is incomplete: {fallback}"
        )
    return fallback.resolve(strict=True)


def _sha256_claim(manifest: Mapping[str, Any], *, shard_index: int) -> str:
    canonical = _manifest_value(manifest, "candidate_pool_sha256")
    legacy = _manifest_value(manifest, "candidate_source_hash")
    canonical_text = str(canonical).strip() if canonical not in (None, "") else ""
    legacy_text = str(legacy).strip() if legacy not in (None, "") else ""
    if canonical_text and legacy_text and canonical_text != legacy_text:
        raise ValueError(
            f"Shard {shard_index} candidate pool hashes conflict within manifest"
        )
    result = canonical_text or legacy_text
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result.lower()
    ):
        raise ValueError(f"Shard {shard_index} lacks a valid candidate pool SHA-256")
    return result.lower()


def _resolve_calibration_predecessor(
    *,
    manifests: Mapping[int, Mapping[str, Any]],
    shard_roots: Mapping[int, Path],
    predecessor_output: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the immutable train universe from shard lineage, never merge output.

    Old PASS shards predate explicit lineage roots.  Their sole compatibility
    path is the verified ``<run_root>/train_candidates/attempt-0`` location.
    The CLI predecessor is retained as a diagnostic hint, not an authority.
    """

    roots: dict[int, Path] = {}
    sources: dict[int, list[str]] = {}
    explicit_attempt_ids: set[str] = set()
    pool_hashes: dict[int, str] = {}
    for index in range(NUM_SHARDS):
        manifest = manifests[index]
        attempt_value = _manifest_value(manifest, "generation_attempt_id")
        attempt_id = (
            str(attempt_value).strip() if attempt_value not in (None, "") else None
        )
        if attempt_id:
            explicit_attempt_ids.add(attempt_id)
        declared_roots: list[tuple[str, Path]] = []
        for field in _CALIBRATION_PREDECESSOR_ROOT_FIELDS:
            value = _manifest_value(manifest, field)
            if value in (None, ""):
                continue
            declared_roots.append(
                (
                    field,
                    _resolve_declared_predecessor_root(
                        field=field,
                        value=value,
                        generation_attempt_id=attempt_id,
                    ),
                )
            )
        if declared_roots:
            distinct = {root for _field, root in declared_roots}
            if len(distinct) != 1:
                raise ValueError(
                    f"Shard {index} declares conflicting predecessor roots: "
                    f"{declared_roots}"
                )
            roots[index] = next(iter(distinct))
            sources[index] = [field for field, _root in declared_roots]
        else:
            roots[index] = _fallback_predecessor_root(shard_roots[index])
            sources[index] = ["verified_run_root_fallback"]
        pool_hashes[index] = _sha256_claim(manifest, shard_index=index)

    if len(set(roots.values())) != 1:
        raise ValueError(f"BACE GCF shards do not share one predecessor root: {roots}")
    if len(explicit_attempt_ids) > 1:
        raise ValueError(
            "BACE GCF shards declare different generation_attempt_id values: "
            f"{sorted(explicit_attempt_ids)}"
        )
    if len(set(pool_hashes.values())) != 1:
        raise ValueError(f"BACE GCF shard candidate pool hashes differ: {pool_hashes}")

    predecessor = next(iter(roots.values()))
    candidate_path = predecessor / "candidate_universe.jsonl"
    candidate_sha256 = sha256_file(candidate_path)
    declared_sha256 = next(iter(pool_hashes.values()))
    if candidate_sha256 != declared_sha256:
        raise ValueError(
            "BACE GCF predecessor candidate bytes differ from all shard manifests"
        )
    source_manifest = read_json(predecessor / "run_manifest.json")
    source_hashes = {
        str(source_manifest.get(field)).strip().lower()
        for field in ("candidate_pool_sha256", "candidate_universe_hash")
        if source_manifest.get(field) not in (None, "")
    }
    if source_hashes and source_hashes != {candidate_sha256}:
        raise ValueError("BACE GCF source manifest candidate hash is inconsistent")
    source_attempt = next(
        (
            str(source_manifest.get(field)).strip()
            for field in ("generation_attempt_id", "attempt_id")
            if source_manifest.get(field) not in (None, "")
        ),
        None,
    )
    if (
        source_attempt
        and explicit_attempt_ids
        and source_attempt not in explicit_attempt_ids
    ):
        raise ValueError("BACE GCF generation attempt identity changed before merge")

    hint = Path(predecessor_output).expanduser()
    hint_status = "ignored_missing"
    if not hint.is_absolute():
        hint_status = "ignored_non_absolute"
    elif _FORBIDDEN_PREDECESSOR_COMPONENTS.intersection(hint.parts):
        hint_status = "ignored_forbidden_merged_or_native_aux"
    elif hint.exists():
        hint_status = (
            "matched_authoritative_root"
            if hint.resolve(strict=True) == predecessor
            else "ignored_mismatch"
        )
    return predecessor, {
        "schema_version": "bace_gcf_predecessor_resolution_v1",
        "status": "PASS",
        "authoritative_root": str(predecessor),
        "candidate_pool_sha256": candidate_sha256,
        "generation_attempt_id": next(iter(explicit_attempt_ids), source_attempt),
        "shard_resolution_sources": {
            str(index): sources[index] for index in range(NUM_SHARDS)
        },
        "cli_hint": str(hint),
        "cli_hint_status": hint_status,
        "forbidden_components": sorted(_FORBIDDEN_PREDECESSOR_COMPONENTS),
    }


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


def _load_candidates(
    *,
    method: str,
    stage: str,
    predecessor_root: Path,
    checkpoint_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    spec = baseline_spec(method)
    if stage == CALIBRATION_STAGE:
        manifest_path = predecessor_root / "run_manifest.json"
        manifest = read_json(manifest_path)
        if (
            manifest.get("stage") != "TRAIN_CANDIDATE_GENERATION"
            or manifest.get("status") != "PASS"
            or manifest.get("run_complete") is not True
        ):
            raise ValueError("Calibration requires a PASS train candidate universe")
        assert_gine_clean_manifest(
            manifest, checkpoint_id=checkpoint_id, require_train_only=True
        )
        candidates = read_jsonl(predecessor_root / "candidate_universe.jsonl")
    else:
        manifest_path = predecessor_root / "frozen_selection_manifest.json"
        manifest = read_json(manifest_path)
        if (
            manifest.get("stage") != SELECTION_STAGE
            or manifest.get("status") != "FROZEN"
            or manifest.get("selection_frozen") is not True
            or manifest.get("test_loaded") is not False
            or manifest.get("selector_fitted_on_calibration") is not True
        ):
            raise ValueError("Held-out test requires a frozen calibration selector")
        assert_gine_clean_manifest(
            manifest, checkpoint_id=checkpoint_id, require_train_only=False
        )
        top20 = read_json(predecessor_root / "selected_top20.json")
        candidates = [dict(row) for row in top20.get("candidates", [])]
        if [str(row.get("candidate_id")) for row in candidates] != list(
            manifest.get("ordered_rule_ids") or []
        ):
            raise ValueError("Frozen selector and selected_top20 ordering differ")
    if str(manifest.get("method_id")) != spec.method_id:
        raise ValueError("Baseline predecessor belongs to a different method")
    minimum_candidates = _minimum_candidate_count(spec.method_id)
    if len(candidates) < minimum_candidates:
        raise ValueError(
            "Native BACE baseline evaluation requires at least "
            f"{minimum_candidates} candidates for {spec.method}"
        )
    ids = [str(row.get("candidate_id") or "") for row in candidates]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Native BACE candidate IDs must be non-empty and unique")
    for row in candidates:
        if row.get("action_kind") != spec.action_kind:
            raise ValueError("Candidate action kind changed from its native method")
        if row.get("action_semantics") != spec.action_semantics:
            raise ValueError("Candidate action semantics changed from its native method")
        if spec.method_id == "globalgce":
            GlobalGCENativeRule.from_payload(row)
        elif not str(row.get("canonical_smiles") or "").strip():
            raise ValueError("Native full-graph candidate lacks canonical_smiles")
        if row.get("rf_oracle_used") is not False:
            raise ValueError("RF candidate provenance is forbidden for BACE")
    return candidates, manifest, manifest_path


def _fullgraph_pair_rows(
    *,
    parents: Sequence[Any],
    before_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[dict[str, Any]],
    featurizer: MolecularGraphFeaturizer,
    oracle: Any,
    provider: MolCLRNodeWassersteinDistance,
    card: Mapping[str, Any],
    spec: Any,
    method_id: str,
    oracle_batch_size: int,
) -> list[dict[str, Any]]:
    candidate_graphs = [
        _graph(
            featurizer,
            smiles=str(candidate["canonical_smiles"]),
            molecule_id=str(candidate["candidate_id"]),
            split="train_generated_native_fullgraph",
        )
        for candidate in candidates
    ]
    after_rows = oracle.predict_records(
        candidate_graphs, batch_size=int(oracle_batch_size)
    )
    pair_rows: list[dict[str, Any]] = []
    for parent, before in zip(parents, before_rows, strict=True):
        for candidate, after in zip(candidates, after_rows, strict=True):
            semantics = compute_counterfactual_semantics(
                source_label=SOURCE_LABEL,
                pred_before=before["predicted_label"],
                pred_after=after["predicted_label"],
                probabilities_before=before["probabilities"],
                probabilities_after=after["probabilities"],
                rule_id=str(candidate["candidate_id"]),
            )
            distance: float | None = None
            failure_reason: str | None = None
            if semantics.cf_flip:
                # Native GCFExplainer and ComRecGC candidates are complete
                # molecular graphs, not matched deletion actions. They have no
                # truthful match indices or match-selection policy, so use the
                # exact full-graph pair key instead of fabricating that context.
                result = provider.distance(
                    parent.smiles,
                    str(candidate["canonical_smiles"]),
                )
                value = result.get("distance")
                if (
                    result.get("ok") is True
                    and value is not None
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                ):
                    distance = float(value)
                else:
                    failure_reason = str(result.get("error") or "wnode_distance_failed")
            else:
                failure_reason = "frozen_gine_not_strict_flip"
            pair_rows.append(
                {
                    "dataset": DATASET,
                    "method": spec.method,
                    "method_id": method_id,
                    "parent_id": parent.parent_id,
                    "parent_smiles": parent.smiles,
                    "candidate_id": candidate["candidate_id"],
                    "canonical_smiles": candidate["canonical_smiles"],
                    "canonical_fragment": candidate["canonical_smiles"],
                    "candidate_rank": candidate.get("rank"),
                    "native_rank": candidate.get("native_rank"),
                    "action_kind": spec.action_kind,
                    "action_semantics": spec.action_semantics,
                    "applicable": True,
                    "pred_before": int(before["predicted_label"]),
                    "pred_after": int(after["predicted_label"]),
                    "p_before": list(before["probabilities"]),
                    "p_after": list(after["probabilities"]),
                    "p1_before": float(before["probabilities"][SOURCE_LABEL]),
                    "p1_after": float(after["probabilities"][SOURCE_LABEL]),
                    "cf_drop": float(semantics.cf_drop),
                    "cf_flip": bool(semantics.cf_flip),
                    "pair_strict_flip": bool(semantics.cf_flip and distance is not None),
                    "wnode_distance": distance,
                    "distance_for_selection": distance if distance is not None else "+inf",
                    "failure_reason": failure_reason,
                    "cf_mode": CF_MODE,
                    "source_label": SOURCE_LABEL,
                    "oracle_backend": "gnn",
                    "classifier_family": "gine",
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": card["checkpoint_id"],
                }
            )
    return pair_rows


def _predict_expected_graph_batch(
    *,
    oracle: Any,
    graphs: Sequence[Any],
    expected_count: int,
    oracle_batch_size: int,
) -> dict[str, Any]:
    """Score one independently counted graph batch without ambiguous emptiness."""

    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("expected_count must be a non-negative int")
    actual_count = len(graphs)
    if actual_count != expected_count:
        if actual_count == 0 and expected_count > 0:
            raise RuntimeError(
                f"{UNEXPECTED_EMPTY_GRAPH_SEQUENCE}: "
                f"expected_count={expected_count}, actual_count=0"
            )
        raise RuntimeError(
            "ORACLE_GRAPH_SEQUENCE_COUNT_MISMATCH: "
            f"expected_count={expected_count}, actual_count={actual_count}"
        )
    num_classes = int(getattr(oracle, "num_classes", 0))
    if num_classes < 2:
        raise RuntimeError("Frozen GINE oracle has an invalid num_classes contract")
    if expected_count == 0:
        return {
            "records": [],
            "logits": np.empty((0, num_classes), dtype=np.float64),
            "probabilities": np.empty((0, num_classes), dtype=np.float64),
            "predictions": np.empty((0,), dtype=np.int64),
            "expected_count": 0,
            "actual_count": 0,
            "oracle_called": False,
            "reason": EXPECTED_EMPTY_GRAPH_SEQUENCE,
        }

    records = oracle.predict_records(
        list(graphs), batch_size=int(oracle_batch_size)
    )
    if len(records) != expected_count:
        raise RuntimeError(
            "Frozen GINE returned an incomplete application prediction batch"
        )
    logits = np.asarray([row["logits"] for row in records], dtype=np.float64)
    probabilities = np.asarray(
        [row["probabilities"] for row in records], dtype=np.float64
    )
    predictions = np.asarray(
        [row["predicted_label"] for row in records], dtype=np.int64
    )
    if logits.shape != (expected_count, num_classes):
        raise RuntimeError("Frozen GINE application logits have an invalid shape")
    if probabilities.shape != (expected_count, num_classes):
        raise RuntimeError(
            "Frozen GINE application probabilities have an invalid shape"
        )
    if predictions.shape != (expected_count,):
        raise RuntimeError("Frozen GINE application predictions have an invalid shape")
    return {
        "records": records,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "expected_count": expected_count,
        "actual_count": actual_count,
        "oracle_called": True,
        "reason": None,
    }


def _prediction_batch_receipt(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce an in-memory prediction batch to JSON-safe shape evidence."""

    return {
        "expected_count": int(batch["expected_count"]),
        "actual_count": int(batch["actual_count"]),
        "oracle_called": bool(batch["oracle_called"]),
        "reason": batch["reason"],
        "logits_shape": list(np.asarray(batch["logits"]).shape),
        "probabilities_shape": list(np.asarray(batch["probabilities"]).shape),
        "predictions_shape": list(np.asarray(batch["predictions"]).shape),
    }


def _globalgce_pair_rows(
    *,
    parents: Sequence[Any],
    before_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[dict[str, Any]],
    featurizer: MolecularGraphFeaturizer,
    oracle: Any,
    provider: MolCLRNodeWassersteinDistance,
    card: Mapping[str, Any],
    spec: Any,
    method_id: str,
    oracle_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = [GlobalGCENativeRule.from_payload(candidate) for candidate in candidates]
    applications: dict[tuple[int, int], list[dict[str, Any]]] = {}
    match_audits: dict[tuple[int, int], list[dict[str, Any]]] = {}
    unique_graphs: dict[str, MolecularGraphData] = {}
    failures: dict[tuple[int, int], str] = {}
    for parent_index, parent in enumerate(parents):
        for candidate_index, rule in enumerate(rules):
            key = (parent_index, candidate_index)
            try:
                rows = apply_rule_to_parent(parent.smiles, rule)
            except Exception as exc:
                rows = []
                failures[key] = f"{type(exc).__name__}:{exc}"
            valid: list[dict[str, Any]] = []
            audited: list[dict[str, Any]] = []
            for row in rows:
                audit_row = dict(row)
                if audit_row.get("valid") is not True:
                    audited.append(audit_row)
                    continue
                canonical = str(audit_row.get("canonical_smiles") or "").strip()
                try:
                    graph = _graph(
                        featurizer,
                        smiles=canonical,
                        molecule_id=(
                            f"{parent.parent_id}:{rule.rule_id}:"
                            f"{audit_row['match_id']}"
                        ),
                        split="native_globalgce_rule_application",
                    )
                except Exception as exc:
                    reason = f"gine_featurize_failed:{type(exc).__name__}:{exc}"
                    failures[key] = reason
                    audit_row["valid"] = False
                    audit_row["failure_reason"] = reason
                    audited.append(audit_row)
                    continue
                record = dict(audit_row)
                record["canonical_smiles"] = graph.smiles
                valid.append(record)
                audited.append(record)
                unique_graphs.setdefault(graph.smiles, graph)
            applications[key] = valid
            match_audits[key] = audited
            if not valid and key not in failures:
                failures[key] = "no_legal_native_lhs_match_or_sanitized_rhs"
    ordered_smiles = sorted(unique_graphs)
    prediction_batch = _predict_expected_graph_batch(
        oracle=oracle,
        graphs=[unique_graphs[smiles] for smiles in ordered_smiles],
        expected_count=len(ordered_smiles),
        oracle_batch_size=int(oracle_batch_size),
    )
    after_predictions = prediction_batch["records"]
    prediction_by_smiles = dict(zip(ordered_smiles, after_predictions, strict=True))
    pair_rows: list[dict[str, Any]] = []
    for parent_index, (parent, before) in enumerate(
        zip(parents, before_rows, strict=True)
    ):
        for candidate_index, (candidate, rule) in enumerate(
            zip(candidates, rules, strict=True)
        ):
            key = (parent_index, candidate_index)
            valid = applications[key]
            audited_matches = match_audits[key]
            evaluated: list[dict[str, Any]] = []
            for application in valid:
                after = prediction_by_smiles[str(application["canonical_smiles"])]
                semantics = compute_counterfactual_semantics(
                    source_label=SOURCE_LABEL,
                    pred_before=before["predicted_label"],
                    pred_after=after["predicted_label"],
                    probabilities_before=before["probabilities"],
                    probabilities_after=after["probabilities"],
                    rule_id=str(candidate["candidate_id"]),
                )
                distance: float | None = None
                distance_failure: str | None = None
                if semantics.cf_flip:
                    distance_result = provider.distance_for_action(
                        parent.smiles,
                        str(application["canonical_smiles"]),
                        action_context={
                            "parent_id": parent.parent_id,
                            "candidate_id": candidate["candidate_id"],
                            "match_id": application["match_id"],
                            "teacher_sha256": card["checkpoint_id"],
                            "oracle_checkpoint_id": card["checkpoint_id"],
                            "action_kind": spec.action_kind,
                            "action_semantics": spec.action_semantics,
                        },
                    )
                    value = distance_result.get("distance")
                    if (
                        distance_result.get("ok") is True
                        and value is not None
                        and math.isfinite(float(value))
                        and float(value) >= 0.0
                    ):
                        distance = float(value)
                    else:
                        distance_failure = str(
                            distance_result.get("error") or "wnode_distance_failed"
                        )
                evaluated.append(
                    {
                        "application": application,
                        "after": after,
                        "semantics": semantics,
                        "distance": distance,
                        "distance_failure": distance_failure,
                    }
                )
            legal = [row for row in evaluated if row["distance"] is not None]
            legal.sort(
                key=lambda row: (
                    float(row["distance"]),
                    str(row["application"]["match_id"]),
                    str(row["application"]["canonical_smiles"]),
                )
            )
            if legal:
                chosen = legal[0]
            elif evaluated:
                chosen = min(
                    evaluated,
                    key=lambda row: (
                        not bool(row["semantics"].cf_flip),
                        str(row["application"]["match_id"]),
                        str(row["application"]["canonical_smiles"]),
                    ),
                )
            else:
                chosen = None
            distance = None if not legal else float(legal[0]["distance"])
            after = chosen["after"] if chosen is not None else None
            semantics = chosen["semantics"] if chosen is not None else None
            application = chosen["application"] if chosen is not None else None
            pair_rows.append(
                {
                    "dataset": DATASET,
                    "method": spec.method,
                    "method_id": method_id,
                    "parent_id": parent.parent_id,
                    "parent_smiles": parent.smiles,
                    "candidate_id": candidate["candidate_id"],
                    "canonical_smiles": (
                        application["canonical_smiles"] if application is not None else ""
                    ),
                    "canonical_fragment": "N/A",
                    "canonical_fragment_reason": (
                        "GlobalGCE action is an attachment-aware LHS-to-RHS rule"
                    ),
                    "candidate_rank": candidate.get("rank"),
                    "native_rank": candidate.get("native_rank"),
                    "native_rule_index": rule.native_rule_index,
                    "rule_content_hash": candidate.get("rule_content_hash"),
                    "action_kind": spec.action_kind,
                    "action_semantics": spec.action_semantics,
                    "applicable": bool(valid),
                    "native_match_attempt_count": len(audited_matches),
                    "native_match_count": len(valid),
                    "native_invalid_match_count": sum(
                        row.get("valid") is not True for row in audited_matches
                    ),
                    "native_match_audit": audited_matches,
                    "selected_match_id": (
                        application.get("match_id") if application is not None else None
                    ),
                    "selected_mapping": (
                        application.get("mapping") if application is not None else None
                    ),
                    "boundary_attachments_preserved": (
                        application.get("boundary_attachments_preserved")
                        if application is not None
                        else None
                    ),
                    "pred_before": int(before["predicted_label"]),
                    "pred_after": (
                        int(after["predicted_label"]) if after is not None else None
                    ),
                    "p_before": list(before["probabilities"]),
                    "p_after": list(after["probabilities"]) if after is not None else [],
                    "p1_before": float(before["probabilities"][SOURCE_LABEL]),
                    "p1_after": (
                        float(after["probabilities"][SOURCE_LABEL])
                        if after is not None
                        else None
                    ),
                    "cf_drop": float(semantics.cf_drop) if semantics is not None else None,
                    "cf_flip": bool(semantics.cf_flip) if semantics is not None else False,
                    "pair_strict_flip": bool(distance is not None),
                    "wnode_distance": distance,
                    "distance_for_selection": distance if distance is not None else "+inf",
                    "failure_reason": (
                        None
                        if distance is not None
                        else (
                            chosen.get("distance_failure")
                            if chosen is not None and chosen.get("distance_failure")
                            else (
                                "frozen_gine_not_strict_flip"
                                if chosen is not None
                                else failures.get(key)
                            )
                        )
                    ),
                    "cf_mode": CF_MODE,
                    "source_label": SOURCE_LABEL,
                    "oracle_backend": "gnn",
                    "classifier_family": "gine",
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": card["checkpoint_id"],
                }
            )
    return pair_rows, _prediction_batch_receipt(prediction_batch)


def _authorize_split(
    *, stage: str, split_path: str | Path, selection_manifest: Path | None
) -> Path:
    raw = Path(split_path).expanduser()
    name = raw.name.lower()
    if stage == CALIBRATION_STAGE:
        if "calib" not in name or "test" in name:
            raise ValueError("Calibration verification requires an explicit calibration split")
    elif stage == TEST_STAGE:
        # Validate the freeze before resolving/opening the held-out split.
        if selection_manifest is None:
            raise ValueError("Test access requires a frozen selection manifest")
        frozen = read_json(selection_manifest)
        if (
            frozen.get("stage") != SELECTION_STAGE
            or frozen.get("selection_frozen") is not True
            or frozen.get("test_loaded") is not False
        ):
            raise ValueError("Test access rejected an incomplete selection freeze")
        if "test" not in name:
            raise ValueError("Held-out evaluation requires an explicitly named test split")
    else:
        raise ValueError(f"Unsupported native baseline evaluation stage: {stage}")
    return raw.resolve(strict=True)


def run_fullgraph_verification_shard(
    *,
    method: str,
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
    device: str = "cuda:0",
    oracle_batch_size: int = 256,
) -> dict[str, Any]:
    method_id = normalize_method(method)
    spec = baseline_spec(method_id)
    if not spec.native_route_available:
        raise ValueError(f"{spec.blocker_code}: {spec.blocker_reason}")
    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in {CALIBRATION_STAGE, TEST_STAGE}:
        raise ValueError("stage must be calibration or held-out test")
    predecessor = Path(predecessor_output).expanduser().resolve(strict=True)
    selection_path = (
        predecessor / "frozen_selection_manifest.json"
        if normalized_stage == TEST_STAGE
        else None
    )
    split = _authorize_split(
        stage=normalized_stage,
        split_path=split_path,
        selection_manifest=selection_path,
    )
    checkpoint, card, schema = validate_bace_frozen_gine(gnn_checkpoint)
    candidates, predecessor_manifest, predecessor_manifest_path = _load_candidates(
        method=method_id,
        stage=normalized_stage,
        predecessor_root=predecessor,
        checkpoint_id=str(card["checkpoint_id"]),
    )
    all_parents = load_bace_parents(split)
    parents = select_parent_shard(all_parents, int(shard_index))
    if not parents:
        raise ValueError(f"Native BACE shard {shard_index} is empty")
    output = fresh_output_dir(output_dir)
    featurizer = MolecularGraphFeaturizer(schema)
    oracle = build_oracle(
        dataset=DATASET,
        backend="gnn",
        checkpoint=checkpoint,
        device=device,
        batch_size=int(oracle_batch_size),
    )
    parent_graphs = [
        _graph(
            featurizer,
            smiles=parent.smiles,
            molecule_id=parent.parent_id,
            split="calibration" if normalized_stage == CALIBRATION_STAGE else "test",
        )
        for parent in parents
    ]
    before_rows = oracle.predict_records(parent_graphs, batch_size=int(oracle_batch_size))
    provider = MolCLRNodeWassersteinDistance(
        MolCLRNodeWassersteinConfig(
            molclr_root=Path(molclr_root).expanduser().resolve(strict=True),
            molclr_ckpt=Path(molclr_checkpoint).expanduser().resolve(strict=True),
            cache_db=Path(wnode_cache_db).expanduser().resolve(strict=False),
            node_emb_cache_dir=Path(node_embedding_cache_dir)
            .expanduser()
            .resolve(strict=False),
            device=device,
            distance_namespace=DISTANCE_NAMESPACE,
        )
    )
    try:
        evaluation_kwargs = {
            "parents": parents,
            "before_rows": before_rows,
            "candidates": candidates,
            "featurizer": featurizer,
            "oracle": oracle,
            "provider": provider,
            "card": card,
            "spec": spec,
            "method_id": method_id,
            "oracle_batch_size": int(oracle_batch_size),
        }
        if method_id == "globalgce":
            pair_rows, application_prediction_batch = _globalgce_pair_rows(
                **evaluation_kwargs
            )
        else:
            pair_rows = _fullgraph_pair_rows(**evaluation_kwargs)
            application_prediction_batch = None
        provider_stats = provider.stats_dict()
    finally:
        provider.close()
    expected = len(parents) * len(candidates)
    if len(pair_rows) != expected:
        raise RuntimeError("Native full-graph shard is not a complete Cartesian product")
    atomic_jsonl(output / "pair_details.jsonl", pair_rows)
    atomic_csv(output / "pair_details.csv", pair_rows)
    provenance = oracle_provenance(card, checkpoint)
    manifest = {
        "schema_version": "bace_native_baseline_verification_shard_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": method_id,
        "stage": normalized_stage,
        "status": "PASS",
        "action_kind": spec.action_kind,
        "action_semantics": spec.action_semantics,
        **provenance,
        "molclr_checkpoint_hash": sha256_file(molclr_checkpoint),
        "molclr_embedding_checkpoint_identity": checkpoint_identity(
            molclr_checkpoint
        ),
        "predecessor_manifest_identity": file_identity(predecessor_manifest_path),
        "candidate_source_hash": (
            predecessor_manifest.get("candidate_universe_hash")
            if normalized_stage == CALIBRATION_STAGE
            else predecessor_manifest.get("selected_top20_hash")
        ),
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "candidate_ids_sha256": stable_sha256(
            [str(row["candidate_id"]) for row in candidates]
        ),
        "shard_index": int(shard_index),
        "num_shards": NUM_SHARDS,
        "shard_rule": "sorted(parent_id)_position_mod_4",
        "all_parent_ids_sha256": stable_sha256(
            sorted(parent.parent_id for parent in all_parents)
        ),
        "parent_ids": [parent.parent_id for parent in parents],
        "parent_count": len(parents),
        "pair_count": len(pair_rows),
        "strict_flip_pair_count": sum(
            bool(row["pair_strict_flip"]) for row in pair_rows
        ),
        "pair_details_identity": file_identity(output / "pair_details.jsonl"),
        "split_identity": file_identity(split),
        "calibration_loaded": normalized_stage == CALIBRATION_STAGE,
        "test_loaded": normalized_stage == TEST_STAGE,
        "selection_frozen_before_test": normalized_stage == TEST_STAGE,
        "distance_provider_stats": provider_stats,
        "created_at": utc_now(),
        "run_complete": True,
    }
    if application_prediction_batch is not None:
        manifest["application_prediction_batch"] = application_prediction_batch
    if normalized_stage == CALIBRATION_STAGE and method_id == "gcfexplainer":
        candidate_pool_sha256 = sha256_file(predecessor / "candidate_universe.jsonl")
        generation_attempt_id = str(
            predecessor_manifest.get("generation_attempt_id")
            or predecessor_manifest.get("attempt_id")
            or predecessor.name
        )
        manifest.update(
            {
                "train_candidates_root": str(predecessor),
                "source_train_candidates_root": str(predecessor),
                "generation_root": str(predecessor),
                "generation_attempt_id": generation_attempt_id,
                "candidate_pool_sha256": candidate_pool_sha256,
            }
        )
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "oracle_provenance.json", provenance)
    atomic_marker(output / "PASS", "PASS")
    return manifest


def merge_fullgraph_verification_shards(
    *,
    method: str,
    stage: str,
    shard_dirs: Sequence[str | Path],
    predecessor_output: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    method_id = normalize_method(method)
    spec = baseline_spec(method_id)
    normalized_stage = str(stage).strip().upper()
    if normalized_stage not in {CALIBRATION_STAGE, TEST_STAGE}:
        raise ValueError("Unsupported native baseline merge stage")
    if len(shard_dirs) != NUM_SHARDS:
        raise ValueError("Native baseline merge requires exactly four fixed shards")
    manifests: dict[int, dict[str, Any]] = {}
    shard_roots: dict[int, Path] = {}
    pair_rows: list[dict[str, Any]] = []
    parents: set[str] = set()
    identities: dict[str, set[str]] = {
        "oracle": set(),
        "molclr": set(),
        "candidates": set(),
        "source": set(),
        "all_parents": set(),
        "split": set(),
    }
    for path_like in shard_dirs:
        root = Path(path_like).expanduser().resolve(strict=True)
        if not (root / "PASS").is_file():
            raise ValueError(f"Native baseline shard lacks PASS marker: {root}")
        manifest = read_json(root / "run_manifest.json")
        if (
            manifest.get("status") != "PASS"
            or manifest.get("run_complete") is not True
            or manifest.get("stage") != normalized_stage
            or manifest.get("method_id") != method_id
            or manifest.get("action_kind") != spec.action_kind
            or manifest.get("rf_oracle_used") is not False
        ):
            raise ValueError(f"Native baseline shard contract mismatch: {root}")
        index = int(manifest.get("shard_index", -1))
        if index in manifests or not 0 <= index < NUM_SHARDS:
            raise ValueError("Duplicate or invalid native baseline shard")
        if int(manifest.get("num_shards", 0)) != NUM_SHARDS:
            raise ValueError("Native baseline shard count is not frozen to four")
        if manifest.get("shard_rule") != "sorted(parent_id)_position_mod_4":
            raise ValueError("Native baseline shard partition rule changed")
        local_parent_list = [str(value) for value in manifest.get("parent_ids", [])]
        local_parents = set(local_parent_list)
        if (
            not local_parents
            or len(local_parents) != len(local_parent_list)
            or len(local_parent_list) != int(manifest.get("parent_count", -1))
        ):
            raise ValueError("Native baseline shard parent range is invalid")
        if parents & local_parents:
            raise ValueError("Native baseline shards overlap in parent identity")
        parents.update(local_parents)
        local_rows = read_jsonl(root / "pair_details.jsonl")
        if len(local_rows) != int(manifest.get("pair_count", -1)):
            raise ValueError("Native baseline shard row count mismatch")
        if file_identity(root / "pair_details.jsonl") != manifest.get(
            "pair_details_identity"
        ):
            raise ValueError("Native baseline shard bytes changed after PASS")
        local_candidate_ids = {
            str(value) for value in manifest.get("candidate_ids", [])
        }
        if any(
            str(row.get("parent_id")) not in local_parents
            or str(row.get("candidate_id")) not in local_candidate_ids
            for row in local_rows
        ):
            raise ValueError("Native baseline shard rows escaped their frozen range")
        pair_rows.extend(local_rows)
        manifests[index] = manifest
        shard_roots[index] = root
        identities["oracle"].add(str(manifest.get("oracle_checkpoint_hash")))
        identities["molclr"].add(str(manifest.get("molclr_checkpoint_hash")))
        identities["candidates"].add(str(manifest.get("candidate_ids_sha256")))
        identities["source"].add(str(manifest.get("candidate_source_hash")))
        identities["all_parents"].add(str(manifest.get("all_parent_ids_sha256")))
        split_identity = manifest.get("split_identity")
        if not isinstance(split_identity, Mapping):
            raise ValueError("Native baseline shard lacks split identity")
        identities["split"].add(stable_sha256(dict(split_identity)))
    if set(manifests) != set(range(NUM_SHARDS)):
        raise ValueError("Native baseline shard set is incomplete")
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError(f"Native baseline shard identities differ: {identities}")
    candidate_ids = list(manifests[0]["candidate_ids"])
    if any(
        [str(value) for value in manifests[index].get("candidate_ids", [])]
        != candidate_ids
        for index in range(NUM_SHARDS)
    ):
        raise ValueError("Native baseline shard candidate ordering differs")
    sorted_parents = sorted(parents)
    if stable_sha256(sorted_parents) != next(iter(identities["all_parents"])):
        raise ValueError("Native baseline shards do not cover the frozen parent range")
    for index in range(NUM_SHARDS):
        expected = [
            parent_id
            for position, parent_id in enumerate(sorted_parents)
            if position % NUM_SHARDS == index
        ]
        observed = [str(value) for value in manifests[index]["parent_ids"]]
        if observed != expected:
            raise ValueError(
                f"Native baseline shard {index} differs from its frozen parent range"
            )
    keys = [(str(row["parent_id"]), str(row["candidate_id"])) for row in pair_rows]
    if len(keys) != len(set(keys)) or len(keys) != len(parents) * len(candidate_ids):
        raise ValueError("Native baseline merged Cartesian product is incomplete")
    candidate_rank = {
        candidate_id: rank for rank, candidate_id in enumerate(candidate_ids)
    }
    pair_rows.sort(
        key=lambda row: (
            str(row["parent_id"]),
            candidate_rank[str(row["candidate_id"])],
        )
    )
    if normalized_stage == CALIBRATION_STAGE and method_id == "gcfexplainer":
        predecessor, predecessor_resolution = _resolve_calibration_predecessor(
            manifests=manifests,
            shard_roots=shard_roots,
            predecessor_output=predecessor_output,
        )
        candidates, _predecessor_manifest, predecessor_manifest_path = _load_candidates(
            method=method_id,
            stage=normalized_stage,
            predecessor_root=predecessor,
            checkpoint_id=next(iter(identities["oracle"])),
        )
        cohort = "calibration"
    elif normalized_stage == CALIBRATION_STAGE:
        predecessor = Path(predecessor_output).expanduser().resolve(strict=True)
        candidates, _predecessor_manifest, predecessor_manifest_path = _load_candidates(
            method=method_id,
            stage=normalized_stage,
            predecessor_root=predecessor,
            checkpoint_id=next(iter(identities["oracle"])),
        )
        predecessor_resolution = {
            "schema_version": "bace_native_predecessor_resolution_v1",
            "status": "PASS",
            "authoritative_root": str(predecessor),
            "resolution": "explicit_calibration_cli",
        }
        cohort = "calibration"
    else:
        predecessor = Path(predecessor_output).expanduser().resolve(strict=True)
        candidates, _predecessor_manifest, predecessor_manifest_path = _load_candidates(
            method=method_id,
            stage=normalized_stage,
            predecessor_root=predecessor,
            checkpoint_id=next(iter(identities["oracle"])),
        )
        predecessor_resolution = {
            "schema_version": "bace_gcf_predecessor_resolution_v1",
            "status": "PASS",
            "authoritative_root": str(predecessor),
            "resolution": "frozen_selection_cli",
        }
        cohort = "test"
    if [str(row["candidate_id"]) for row in candidates] != candidate_ids:
        raise ValueError("Native baseline merged candidate order changed")
    output = fresh_output_dir(output_dir)
    atomic_jsonl(output / "pair_matrix.jsonl", pair_rows)
    atomic_jsonl(output / "selected_candidate_universe.jsonl", candidates)
    atomic_csv(output / "pair_details.csv", pair_rows)
    strict_count = sum(bool(row["pair_strict_flip"]) for row in pair_rows)
    summary = {
        "method": spec.method,
        "parent_count": len(parents),
        "selected_candidate_count": len(candidates),
        "pair_count": len(pair_rows),
        "strict_flip_pair_count": strict_count,
        "calibration_loaded": cohort == "calibration",
        "test_loaded": cohort == "test",
        "run_complete": True,
    }
    atomic_json(output / "summary.json", summary)
    manifest = {
        "schema_version": "bace_native_baseline_verification_merge_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": method_id,
        "stage": normalized_stage,
        "status": "PASS",
        "action_kind": spec.action_kind,
        "action_semantics": spec.action_semantics,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "cf_mode": CF_MODE,
        "oracle_checkpoint_hash": next(iter(identities["oracle"])),
        "molclr_checkpoint_hash": next(iter(identities["molclr"])),
        "candidate_universe_hash": sha256_file(output / "selected_candidate_universe.jsonl"),
        "pair_matrix_hash": sha256_file(output / "pair_matrix.jsonl"),
        "parent_count": len(parents),
        "selected_candidate_count": len(candidates),
        "pair_count": len(pair_rows),
        "strict_flip_pair_count": strict_count,
        "inputs": {
            "cohort_name": cohort,
            "predecessor_manifest": file_identity(predecessor_manifest_path),
            "predecessor_resolution": predecessor_resolution,
            "shard_manifests": [
                file_identity(shard_roots[index] / "run_manifest.json")
                for index in range(NUM_SHARDS)
            ],
        },
        "calibration_loaded": cohort == "calibration",
        "test_loaded": cohort == "test",
        "selection_frozen_before_test": cohort == "test",
        "created_at": utc_now(),
        "run_complete": True,
    }
    atomic_json(output / "run_manifest.json", manifest)
    atomic_marker(output / "PASS", "PASS")
    return manifest


def _contains_path_route(path: Path, route: Sequence[str]) -> bool:
    parts = tuple(part.lower() for part in path.parts)
    expected = tuple(part.lower() for part in route)
    width = len(expected)
    return any(parts[index : index + width] == expected for index in range(len(parts)))


def _verify_declared_file_identity(identity: Any, *, label: str) -> Path:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} identity is missing")
    raw_path = str(identity.get("path") or "").strip()
    path = Path(raw_path).expanduser()
    if not raw_path or not path.is_absolute():
        raise ValueError(f"{label} identity path must be absolute")
    if path.is_symlink():
        raise ValueError(f"{label} identity must name one physical file")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} identity must name one physical file")
    if int(identity.get("size", -1)) != path.stat().st_size:
        raise ValueError(f"{label} identity size changed")
    if str(identity.get("sha256") or "").lower() != sha256_file(path):
        raise ValueError(f"{label} identity SHA-256 changed")
    return path


def _load_ours_b12_thresholds(
    thresholds_json: str | Path,
    *,
    matrix_manifest: Mapping[str, Any],
) -> tuple[ThresholdBundle, dict[str, Any], dict[str, Any]]:
    """Adopt the immutable Ours B12 grid without consulting held-out test data."""

    unresolved_thresholds = Path(thresholds_json).expanduser()
    if unresolved_thresholds.is_symlink():
        raise ValueError("--thresholds-json must name one physical JSON file")
    thresholds_path = unresolved_thresholds.resolve(strict=True)
    if not thresholds_path.is_file():
        raise ValueError("--thresholds-json must name one physical JSON file")
    if thresholds_path.name != "thresholds.json" or not _contains_path_route(
        thresholds_path, ("bace", "ours", "b12-selector")
    ):
        raise ValueError(
            "BACE baseline thresholds must come from bace/ours/b12-selector"
        )
    source_manifest_path = thresholds_path.parent / "frozen_selection_manifest.json"
    if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
        raise ValueError("Ours B12 frozen_selection_manifest.json is missing")
    thresholds_payload = read_json(thresholds_path)
    thresholds = threshold_bundle_from_dict(thresholds_payload)
    threshold_hash = stable_sha256(
        [level.threshold for level in thresholds.levels]
    )
    if threshold_hash != BACE_OURS_B12_THRESHOLD_CONFIG_SHA256:
        raise ValueError(
            "Ours B12 threshold grid is not the preregistered BACE contract"
        )

    source_manifest = read_json(source_manifest_path)
    required = {
        "schema_version": "bace_frozen_gnn_selection_manifest_v1",
        "dataset": DATASET,
        "stage": "B12_SELECTOR",
        "status": "FROZEN",
        "selection_frozen": True,
        "selector_fitted_on_calibration": True,
        "calibration_loaded": True,
        "test_loaded": False,
        "test_used": False,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": 2,
        "cf_mode": CF_MODE,
    }
    mismatches = [
        f"{field}={source_manifest.get(field)!r}"
        for field, expected in required.items()
        if source_manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            "Ours B12 frozen selection contract mismatch: " + ", ".join(mismatches)
        )
    if source_manifest.get("thresholds") != thresholds_payload:
        raise ValueError("Ours B12 manifest does not bind thresholds.json exactly")
    for field in ("oracle_checkpoint_hash", "molclr_checkpoint_hash"):
        if source_manifest.get(field) != matrix_manifest.get(field):
            raise ValueError(
                f"Ours B12/baseline calibration identity mismatch: {field}"
            )

    b11_path = _verify_declared_file_identity(
        source_manifest.get("matrix_manifest_identity"),
        label="Ours B11 matrix manifest",
    )
    if not _contains_path_route(b11_path, ("bace", "ours", "b11-merged")):
        raise ValueError("Ours B12 source is not bound to bace/ours/b11-merged")
    b11_manifest = read_json(b11_path)
    b11_required = {
        "schema_version": "bace_frozen_gnn_verification_merge_v1",
        "dataset": DATASET,
        "stage": "B11_CROSS_PARENT_VERIFIED",
        "status": "PASS",
        "run_complete": True,
        "calibration_loaded": True,
        "test_loaded": False,
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": 2,
        "cf_mode": CF_MODE,
    }
    if any(b11_manifest.get(field) != value for field, value in b11_required.items()):
        raise ValueError("Ours B11 threshold source is not a PASS calibration matrix")
    if (b11_manifest.get("inputs") or {}).get("cohort_name") != "calibration":
        raise ValueError("Ours B11 threshold source cohort is not calibration")
    for field in ("oracle_checkpoint_hash", "molclr_checkpoint_hash"):
        if b11_manifest.get(field) != source_manifest.get(field):
            raise ValueError(f"Ours B11/B12 identity mismatch: {field}")

    provenance = {
        "schema_version": "bace_ours_b12_threshold_adoption_v1",
        "mode": "frozen_ours_b12_selector",
        "dataset": DATASET,
        "source_method": "Ours",
        "source_stage": "B12_SELECTOR",
        "threshold_config_hash": threshold_hash,
        "thresholds_json": file_identity(thresholds_path),
        "source_selection_manifest": file_identity(source_manifest_path),
        "source_matrix_manifest": file_identity(b11_path),
        "test_loaded": False,
        "test_used": False,
    }
    return thresholds, thresholds_payload, provenance


def run_native_baseline_selector(
    *,
    method: str,
    matrix_output: str | Path,
    output_dir: str | Path,
    seed: int = 13,
    thresholds_json: str | Path | None = None,
) -> dict[str, Any]:
    method_id = normalize_method(method)
    spec = baseline_spec(method_id)
    matrix_root = Path(matrix_output).expanduser().resolve(strict=True)
    matrix_manifest = read_json(matrix_root / "run_manifest.json")
    if (
        matrix_manifest.get("stage") != CALIBRATION_STAGE
        or matrix_manifest.get("status") != "PASS"
        or matrix_manifest.get("test_loaded") is not False
        or matrix_manifest.get("method_id") != method_id
    ):
        raise ValueError("Native selector requires a PASS calibration matrix")
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"Native selector output must be fresh: {output}")
    calibration_rows = read_jsonl(matrix_root / "pair_matrix.jsonl")
    calibration_candidate_ids = {
        str(row.get("candidate_id") or "") for row in calibration_rows
    }
    if "" in calibration_candidate_ids:
        raise ValueError("Calibration matrix contains an empty candidate ID")
    effective_k = min(20, len(calibration_candidate_ids))
    minimum_k = _minimum_candidate_count(method_id)
    if effective_k < minimum_k:
        raise ValueError(
            f"Native selector has {effective_k} candidates, below minimum {minimum_k}"
        )
    frozen_thresholds: ThresholdBundle | None = None
    threshold_payload: dict[str, Any] | None = None
    threshold_provenance: dict[str, Any] | None = None
    if thresholds_json is not None:
        frozen_thresholds, threshold_payload, threshold_provenance = (
            _load_ours_b12_thresholds(
                thresholds_json,
                matrix_manifest=matrix_manifest,
            )
        )
    elif method_id == "gcfexplainer":
        raise ValueError(
            "GCFExplainer selection requires explicit --thresholds-json "
            "from the frozen Ours B12 selector"
        )
    run_mutagenicity_wnode_selector(
        matrix_run_dir=matrix_root,
        output_dir=output,
        top_k=effective_k,
        table_k=10,
        seed=int(seed),
        forbid_test=True,
        frozen_thresholds=frozen_thresholds,
        frozen_threshold_provenance=threshold_provenance,
    )
    decision = read_json(output / "calibration_decision.json")
    variant = str(decision.get("selected_variant") or "")
    selected = read_json(output / "variants" / variant / "selected_top20.json")
    candidates = [dict(row) for row in selected.get("candidates", [])]
    ids = [str(row.get("candidate_id") or "") for row in candidates]
    if len(ids) != effective_k or len(ids) != len(set(ids)):
        raise ValueError(
            "Native selector did not freeze the complete effective unique prefix"
        )
    thresholds = read_json(output / "thresholds.json")
    if threshold_payload is not None and thresholds != threshold_payload:
        raise ValueError("GCFExplainer selector changed the frozen Ours thresholds")
    top20 = {
        "schema_version": "bace_native_baseline_selected_top20_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": method_id,
        "stage": SELECTION_STAGE,
        "status": "FROZEN",
        "candidates": candidates,
        "candidate_ids": ids,
        "test_loaded": False,
    }
    atomic_json(output / "selected_top20.json", top20)
    frozen = {
        "schema_version": "bace_native_baseline_selection_manifest_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": method_id,
        "stage": SELECTION_STAGE,
        "status": "FROZEN",
        "action_kind": spec.action_kind,
        "action_semantics": spec.action_semantics,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "cf_mode": CF_MODE,
        "oracle_checkpoint_hash": matrix_manifest["oracle_checkpoint_hash"],
        "molclr_checkpoint_hash": matrix_manifest["molclr_checkpoint_hash"],
        "selector_fitted_on_calibration": True,
        "selection_frozen": True,
        "calibration_loaded": True,
        "test_loaded": False,
        "K": effective_k,
        "K_MAX": 20,
        "effective_rule_count": effective_k,
        "ordered_rule_ids": ids,
        "ordered_rule_ids_sha256": stable_sha256(ids),
        "prefixes": {
            str(k): ids[: min(k, effective_k)] for k in range(1, 21)
        },
        "thresholds": thresholds,
        "selected_variant": variant,
        "calibration_input_hash": sha256_file(matrix_root / "pair_matrix.jsonl"),
        "candidate_pool_hash": matrix_manifest["candidate_universe_hash"],
        "selected_top20_hash": sha256_file(output / "selected_top20.json"),
        "created_at": utc_now(),
    }
    if threshold_provenance is not None:
        frozen["threshold_provenance"] = threshold_provenance
        frozen["threshold_config_hash"] = threshold_provenance[
            "threshold_config_hash"
        ]
    atomic_json(output / "frozen_selection_manifest.json", frozen)
    atomic_marker(output / "PASS", "PASS")
    return frozen


def freeze_native_baseline_final(
    *, method: str, selection_output: str | Path, test_output: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    method_id = normalize_method(method)
    spec = baseline_spec(method_id)
    selection_root = Path(selection_output).expanduser().resolve(strict=True)
    test_root = Path(test_output).expanduser().resolve(strict=True)
    frozen = read_json(selection_root / "frozen_selection_manifest.json")
    test_manifest = read_json(test_root / "run_manifest.json")
    if (
        frozen.get("stage") != SELECTION_STAGE
        or frozen.get("status") != "FROZEN"
        or frozen.get("method_id") != method_id
        or test_manifest.get("stage") != TEST_STAGE
        or test_manifest.get("status") != "PASS"
        or test_manifest.get("method_id") != method_id
        or test_manifest.get("selection_frozen_before_test") is not True
        or test_manifest.get("test_loaded") is not True
    ):
        raise ValueError("Native baseline final freeze dependencies are incomplete")
    for field in ("oracle_checkpoint_hash", "molclr_checkpoint_hash"):
        if frozen.get(field) != test_manifest.get(field):
            raise ValueError(f"Native baseline selection/test identity changed: {field}")
    ids = list(frozen["ordered_rule_ids"])
    pair_rows = read_jsonl(test_root / "pair_matrix.jsonl")
    by_parent: dict[str, dict[str, float]] = {}
    for row in pair_rows:
        parent = str(row["parent_id"])
        candidate = str(row["candidate_id"])
        if candidate not in ids:
            raise ValueError("Test pair escaped frozen candidate ordering")
        distance = math.inf
        if row.get("pair_strict_flip"):
            distance = float(row["wnode_distance"])
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("Strict test pair lacks finite WNode")
        by_parent.setdefault(parent, {})[candidate] = distance
    if not by_parent or any(set(values) != set(ids) for values in by_parent.values()):
        raise ValueError("Held-out test matrix is not the frozen Cartesian product")
    matrix = np.asarray(
        [[by_parent[parent][candidate] for candidate in ids] for parent in sorted(by_parent)],
        dtype=np.float64,
    )
    thresholds = frozen["thresholds"]
    theta_star = float(thresholds["theta_star"])
    prefix_metrics = []
    for k in range(1, 21):
        effective_prefix_k = min(k, len(ids))
        best = np.min(matrix[:, :effective_prefix_k], axis=1)
        finite = best[np.isfinite(best)]
        prefix_metrics.append(
            {
                "K": k,
                "effective_rule_count": effective_prefix_k,
                "plateau_after_effective_k": k > len(ids),
                "SuppCov": float(np.mean(np.isfinite(best))),
                "CCRCov": float(np.mean(best <= theta_star)),
                "avg_cost": float(np.mean(finite)) if finite.size else None,
                "median_cost": float(np.median(finite)) if finite.size else None,
            }
        )
    output = fresh_output_dir(output_dir)
    metrics = {
        "schema_version": "bace_native_baseline_test_metrics_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": method_id,
        "stage": TEST_STAGE,
        "status": "PASS",
        "parent_count": len(by_parent),
        "ordered_rule_ids": ids,
        "effective_rule_count": len(ids),
        "K_MAX": 20,
        "theta_star": theta_star,
        "prefix_metrics": prefix_metrics,
        "selector_refit_on_test": False,
        "threshold_refit_on_test": False,
        "test_loaded": True,
    }
    atomic_json(output / "final_metrics.json", metrics)
    atomic_csv(output / "prefix_metrics.csv", prefix_metrics)
    final = {
        "schema_version": "bace_native_baseline_final_freeze_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": method_id,
        "stage": FINAL_STAGE,
        "status": "PASS",
        "action_kind": spec.action_kind,
        "action_semantics": spec.action_semantics,
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "cf_mode": CF_MODE,
        "oracle_checkpoint_hash": frozen["oracle_checkpoint_hash"],
        "molclr_checkpoint_hash": frozen["molclr_checkpoint_hash"],
        "selector_fitted_on_calibration": True,
        "selection_frozen_before_test": True,
        "test_used_only_after_freeze": True,
        "all_hashes_frozen": True,
        "ordered_rule_ids": ids,
        "effective_rule_count": len(ids),
        "K_MAX": 20,
        "selection_manifest_identity": file_identity(selection_root / "frozen_selection_manifest.json"),
        "test_manifest_identity": file_identity(test_root / "run_manifest.json"),
        "test_pair_matrix_identity": file_identity(test_root / "pair_matrix.jsonl"),
        "final_metrics_identity": file_identity(output / "final_metrics.json"),
        "created_at": utc_now(),
        "run_complete": True,
    }
    atomic_json(output / "FINAL_PASS.json", final)
    atomic_json(output / "run_manifest.json", final)
    atomic_marker(output / "PASS", "PASS")
    return final


__all__ = [
    "CALIBRATION_STAGE",
    "FINAL_STAGE",
    "SELECTION_STAGE",
    "TEST_STAGE",
    "freeze_native_baseline_final",
    "merge_fullgraph_verification_shards",
    "run_fullgraph_verification_shard",
    "run_native_baseline_selector",
]
