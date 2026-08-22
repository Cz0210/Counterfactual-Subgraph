"""Freeze train-generated native full-graph candidates with the frozen GINE.

This is a validity/identity gate, not a calibration selector.  Native order is
retained, all connected unique strict-flip graphs are exported, and the held-
out calibration/test cohorts are never opened.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rdkit import Chem, rdBase

from src.baselines.bace_gnn_baseline_contracts import (
    CF_MODE,
    DATASET,
    SOURCE_LABEL,
    assert_gine_clean_manifest,
    baseline_spec,
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.eval.bace_frozen_gnn_contracts import (
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    atomic_marker,
    file_identity,
    fresh_output_dir,
    sha256_file,
    stable_sha256,
    utc_now,
)
from src.oracles.oracle_factory import build_oracle


def _graph(
    featurizer: MolecularGraphFeaturizer,
    *,
    smiles: str,
    candidate_id: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=0,
        molecule_id=candidate_id,
        smiles=features.canonical_smiles,
        split="train_native_baseline_candidate",
        graph_sha256=features.graph_sha256,
    )


def _candidate_id(method_id: str, smiles: str) -> str:
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20].upper()
    return f"{method_id.upper()}_BACE_{digest}"


def freeze_gine_candidate_universe(
    *,
    method: str,
    decoded_candidates: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    source_manifest_path: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    oracle_batch_size: int = 256,
    minimum_candidates: int = 20,
) -> dict[str, Any]:
    """Validate, GINE-score, and freeze one native train-generated universe."""

    spec = baseline_spec(method)
    if not spec.native_route_available:
        raise ValueError(f"{spec.blocker_code}: {spec.blocker_reason}")
    checkpoint, card, schema = validate_bace_frozen_gine(gnn_checkpoint)
    assert_gine_clean_manifest(
        source_manifest,
        checkpoint_id=str(card["checkpoint_id"]),
        require_train_only=True,
    )
    source_path = Path(source_manifest_path).expanduser().resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("Native source manifest must be a regular file")
    if int(minimum_candidates) <= 0:
        raise ValueError("minimum_candidates must be positive")

    output = fresh_output_dir(output_dir)
    featurizer = MolecularGraphFeaturizer(schema)
    prepared: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    reasons: Counter[str] = Counter()
    for index, raw_value in enumerate(decoded_candidates):
        raw = dict(raw_value)
        native_rank = int(raw.get("native_rank") or raw.get("rank") or index + 1)
        raw_candidate_id = str(raw.get("candidate_id") or f"native-{native_rank}")
        smiles = str(
            raw.get("canonical_smiles") or raw.get("smiles") or ""
        ).strip()
        reason = "eligible_for_gnn"
        canonical = ""
        graph = None
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            reason = "invalid_or_empty_smiles"
        else:
            try:
                Chem.SanitizeMol(molecule)
                canonical = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                )
            except Exception:
                reason = "sanitize_or_canonicalize_failed"
            if reason == "eligible_for_gnn" and (
                "." in canonical or len(Chem.GetMolFrags(molecule)) != 1
            ):
                reason = "disconnected_fullgraph_candidate"
            if reason == "eligible_for_gnn" and canonical in seen:
                reason = "duplicate_canonical_smiles"
            if reason == "eligible_for_gnn":
                candidate_id = _candidate_id(spec.method_id, canonical)
                try:
                    graph = _graph(
                        featurizer, smiles=canonical, candidate_id=candidate_id
                    )
                except Exception:
                    reason = "gine_featurization_failed"
        audit_rows.append(
            {
                "native_candidate_id": raw_candidate_id,
                "native_rank": native_rank,
                "canonical_smiles": canonical,
                "decode_ok": bool(raw.get("decode_ok", bool(smiles))),
                "connected": reason not in {
                    "invalid_or_empty_smiles",
                    "sanitize_or_canonicalize_failed",
                    "disconnected_fullgraph_candidate",
                },
                "gine_inference_attempted": False,
                "pred_after": None,
                "p_after": None,
                "strict_flip_target": False,
                "selected": False,
                "rejection_reason": reason,
            }
        )
        reasons[reason] += 1
        if graph is None:
            continue
        seen[canonical] = raw_candidate_id
        prepared.append(
            {
                "native": raw,
                "native_candidate_id": raw_candidate_id,
                "native_rank": native_rank,
                "canonical_smiles": canonical,
                "candidate_id": _candidate_id(spec.method_id, canonical),
                "graph": graph,
                "audit_index": len(audit_rows) - 1,
            }
        )

    oracle = build_oracle(
        dataset=DATASET,
        backend="gnn",
        checkpoint=checkpoint,
        device=device,
        batch_size=int(oracle_batch_size),
    )
    predictions = oracle.predict_records(
        [row["graph"] for row in prepared], batch_size=int(oracle_batch_size)
    )
    selected: list[dict[str, Any]] = []
    for prepared_row, prediction in zip(prepared, predictions, strict=True):
        audit = audit_rows[int(prepared_row["audit_index"])]
        pred_after = int(prediction["predicted_label"])
        probabilities = [float(value) for value in prediction["probabilities"]]
        finite = len(probabilities) == 2 and all(math.isfinite(v) for v in probabilities)
        strict_target = bool(finite and pred_after != SOURCE_LABEL)
        audit.update(
            {
                "gine_inference_attempted": True,
                "pred_after": pred_after,
                "p_after": probabilities,
                "strict_flip_target": strict_target,
                "rejection_reason": (
                    "selected" if strict_target else "frozen_gine_not_target_label"
                ),
            }
        )
        reasons[audit["rejection_reason"]] += 1
        if not strict_target:
            continue
        native = dict(prepared_row["native"])
        row = {
            **native,
            "candidate_id": prepared_row["candidate_id"],
            "native_candidate_id": prepared_row["native_candidate_id"],
            "native_rank": int(prepared_row["native_rank"]),
            "rank": len(selected) + 1,
            "canonical_smiles": prepared_row["canonical_smiles"],
            "smiles": prepared_row["canonical_smiles"],
            # Compatibility field consumed only by the common structural-
            # redundancy selector.  The action kind below remains full graph.
            "canonical_fragment": prepared_row["canonical_smiles"],
            "action_kind": spec.action_kind,
            "action_semantics": spec.action_semantics,
            "source_method": spec.method,
            "pred_after_generation": pred_after,
            "p_after_generation": probabilities,
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "oracle_checkpoint_hash": card["checkpoint_id"],
            "source_label": SOURCE_LABEL,
            "cf_mode": CF_MODE,
            "generation_split": "train",
            "calibration_loaded": False,
            "test_loaded": False,
            "candidate_set_preselected": False,
            "selection_performed_in_eval": False,
        }
        selected.append(row)
        audit["selected"] = True

    atomic_jsonl(output / "candidate_filter_audit.jsonl", audit_rows)
    atomic_jsonl(output / "candidate_universe.jsonl", selected)
    if selected:
        atomic_csv(output / "candidate_universe.csv", selected)
    provenance = oracle_provenance(card, checkpoint)
    manifest = {
        "schema_version": "bace_native_baseline_candidate_universe_v1",
        "dataset": DATASET,
        "method": spec.method,
        "method_id": spec.method_id,
        "stage": "TRAIN_CANDIDATE_GENERATION",
        "status": "PASS" if len(selected) >= int(minimum_candidates) else "BLOCKED",
        "action_kind": spec.action_kind,
        "action_semantics": spec.action_semantics,
        **provenance,
        "source_manifest_identity": file_identity(source_path),
        "source_manifest_stage": source_manifest.get("stage"),
        "source_run_complete": source_manifest.get("run_complete"),
        "native_input_count": len(decoded_candidates),
        "gine_scored_count": len(prepared),
        "candidate_count": len(selected),
        "minimum_candidates": int(minimum_candidates),
        "candidate_ids": [row["candidate_id"] for row in selected],
        "candidate_ids_sha256": stable_sha256(
            [row["candidate_id"] for row in selected]
        ),
        "candidate_universe_hash": sha256_file(output / "candidate_universe.jsonl"),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "native_order_preserved": True,
        "candidate_set_preselected": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "run_complete": len(selected) >= int(minimum_candidates),
        "created_at": utc_now(),
    }
    atomic_json(output / "oracle_provenance.json", provenance)
    atomic_json(output / "run_manifest.json", manifest)
    if manifest["run_complete"] is not True:
        failure = {
            **manifest,
            "status": "BLOCKED",
            "blocker_code": "BLOCKED_INSUFFICIENT_GINE_VALID_NATIVE_CANDIDATES",
            "message": (
                f"Only {len(selected)} frozen-GINE strict candidates; "
                f"required {minimum_candidates}."
            ),
        }
        atomic_json(output / "BLOCKED.json", failure)
        atomic_marker(output / "BLOCKED", failure["blocker_code"])
        raise RuntimeError(failure["blocker_code"] + ": " + failure["message"])
    atomic_marker(output / "PASS", "PASS")
    return manifest


__all__ = ["freeze_gine_candidate_universe"]
