"""Frozen-GINE forward evaluation for native GlobalGCE rule applications.

This adapter intentionally owns only the *forward* scientific boundary.  A
rule is applied with the audited attachment-aware LHS->RHS action engine, the
resulting sanitized molecule is featurized with the frozen BACE schema, and
the same calibrated GINE used by every BACE method scores before/after graphs.
It never substitutes RF, the official GTGNN, a full-graph candidate, or a
deletion action.  It also does not pretend that this discrete forward path is
differentiable with respect to the official continuous RHS decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.baselines.bace_gnn_baseline_contracts import (
    CF_MODE,
    CLASSIFIER_FAMILY,
    DATASET,
    NUM_CLASSES,
    ORACLE_BACKEND,
    SOURCE_LABEL,
    baseline_spec,
    oracle_provenance,
    validate_bace_frozen_gine,
)
from src.baselines.globalgce_bace_native_rules import (
    ACTION_ENGINE_VERSION,
    GlobalGCENativeRule,
    apply_rule_to_parent,
)
from src.data.molecular_graph_dataset import MolecularGraphData
from src.data.molecular_graph_featurizer import MolecularGraphFeaturizer
from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_marker,
    file_identity,
    fresh_output_dir,
    utc_now,
)
from src.eval.counterfactual_semantics import compute_counterfactual_semantics
from src.oracles.oracle_factory import build_oracle


FORWARD_EVAL_VERSION = "bace_globalgce_native_frozen_gine_forward_v1"
_ALLOWED_SPLITS = {"train", "calibration", "preflight_canary", "test"}


def _graph(
    featurizer: MolecularGraphFeaturizer,
    *,
    smiles: str,
    molecule_id: str,
    split_role: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=SOURCE_LABEL,
        molecule_id=molecule_id,
        smiles=features.canonical_smiles,
        split=split_role,
        graph_sha256=features.graph_sha256,
    )


def _validate_oracle(oracle: Any, provenance: Mapping[str, Any]) -> None:
    required = {
        "oracle_backend": ORACLE_BACKEND,
        "classifier_family": CLASSIFIER_FAMILY,
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
    }
    mismatches = [
        f"{key}={provenance.get(key)!r}"
        for key, expected in required.items()
        if provenance.get(key) != expected
    ]
    actual = {
        "backbone": str(getattr(oracle, "backbone", "")).strip().lower(),
        "source_label": getattr(oracle, "source_label", None),
        "num_classes": getattr(oracle, "num_classes", None),
        "checkpoint_id": str(getattr(oracle, "checkpoint_id", "")),
    }
    expected_checkpoint = str(provenance.get("oracle_checkpoint_hash") or "")
    if actual["backbone"] != CLASSIFIER_FAMILY:
        mismatches.append(f"oracle.backbone={actual['backbone']!r}")
    if actual["source_label"] != SOURCE_LABEL:
        mismatches.append(f"oracle.source_label={actual['source_label']!r}")
    if actual["num_classes"] != NUM_CLASSES:
        mismatches.append(f"oracle.num_classes={actual['num_classes']!r}")
    if not expected_checkpoint or actual["checkpoint_id"] != expected_checkpoint:
        mismatches.append(
            "oracle.checkpoint_id="
            f"{actual['checkpoint_id']!r}, expected={expected_checkpoint!r}"
        )
    if mismatches:
        raise ValueError(
            "Native GlobalGCE forward evaluator rejected non-frozen-GINE oracle: "
            + ", ".join(mismatches)
        )


def _validate_prediction(
    record: Mapping[str, Any], checkpoint_id: str, expected_temperature: float
) -> None:
    probabilities = tuple(float(value) for value in record.get("probabilities") or ())
    if len(probabilities) != NUM_CLASSES:
        raise ValueError("Frozen GINE prediction has the wrong probability width")
    if any(not math.isfinite(value) for value in probabilities):
        raise ValueError("Frozen GINE prediction contains non-finite probabilities")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("Frozen GINE prediction contains out-of-range probabilities")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Frozen GINE prediction probabilities do not sum to one")
    if str(record.get("checkpoint_id")) != checkpoint_id:
        raise ValueError("Frozen GINE prediction checkpoint identity drifted")
    if str(record.get("backbone") or "").strip().lower() != CLASSIFIER_FAMILY:
        raise ValueError("Frozen GINE prediction backbone drifted")
    if record.get("source_label") != SOURCE_LABEL:
        raise ValueError("Frozen GINE prediction source label drifted")
    if record.get("num_classes") != NUM_CLASSES:
        raise ValueError("Frozen GINE prediction class count drifted")
    temperature = float(record.get("temperature", float("nan")))
    if not math.isfinite(temperature) or not math.isclose(
        temperature, expected_temperature, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Frozen GINE prediction temperature drifted")
    predicted = int(record.get("predicted_label", -1))
    if predicted != max(range(NUM_CLASSES), key=probabilities.__getitem__):
        raise ValueError("Frozen GINE predicted label disagrees with probabilities")


@dataclass(slots=True)
class BACEGlobalGCEFrozenGINEForwardEvaluator:
    """Loaded-once, provenance-checked evaluator for native rule matches."""

    oracle: Any
    featurizer: MolecularGraphFeaturizer
    provenance: Mapping[str, Any]
    oracle_batch_size: int = 256

    def __post_init__(self) -> None:
        if int(self.oracle_batch_size) <= 0:
            raise ValueError("oracle_batch_size must be positive")
        self.provenance = dict(self.provenance)
        _validate_oracle(self.oracle, self.provenance)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str = "cpu",
        oracle_batch_size: int = 256,
    ) -> "BACEGlobalGCEFrozenGINEForwardEvaluator":
        checkpoint, card, schema = validate_bace_frozen_gine(checkpoint_dir)
        oracle = build_oracle(
            dataset=DATASET,
            backend=ORACLE_BACKEND,
            checkpoint=checkpoint,
            device=device,
            batch_size=int(oracle_batch_size),
            num_classes=NUM_CLASSES,
            source_label=SOURCE_LABEL,
        )
        return cls(
            oracle=oracle,
            featurizer=MolecularGraphFeaturizer(schema),
            provenance=oracle_provenance(card, checkpoint),
            oracle_batch_size=int(oracle_batch_size),
        )

    def score_parent_rule(
        self,
        *,
        parent_id: str,
        parent_smiles: str,
        rule: GlobalGCENativeRule,
        split_role: str = "preflight_canary",
        selector_frozen_before_test: bool = False,
        frozen_selector_hash: str | None = None,
    ) -> dict[str, Any]:
        """Apply all matches and batch one exact calibrated-GINE forward pass."""

        role = str(split_role).strip().lower().replace("-", "_")
        if role not in _ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split_role for native GINE forward: {role}")
        selector_hash = str(frozen_selector_hash or "").strip()
        if role == "test" and (
            selector_frozen_before_test is not True or not selector_hash
        ):
            raise ValueError(
                "Held-out test forward requires an already-frozen selector hash"
            )
        if not str(parent_id).strip():
            raise ValueError("Native GlobalGCE forward requires a parent_id")

        applications = apply_rule_to_parent(parent_smiles, rule)
        valid_positions = [
            index for index, row in enumerate(applications) if row.get("valid") is True
        ]
        graphs = [
            _graph(
                self.featurizer,
                smiles=parent_smiles,
                molecule_id=str(parent_id),
                split_role=role,
            )
        ]
        graphs.extend(
            _graph(
                self.featurizer,
                smiles=str(applications[index]["canonical_smiles"]),
                molecule_id=f"{parent_id}:{rule.rule_id}:{applications[index]['match_id']}",
                split_role=role,
            )
            for index in valid_positions
        )
        predictions = self.oracle.predict_records(
            graphs, batch_size=int(self.oracle_batch_size)
        )
        if len(predictions) != len(graphs):
            raise ValueError("Frozen GINE returned an incomplete prediction batch")
        checkpoint_id = str(self.provenance["oracle_checkpoint_hash"])
        expected_temperature = float(getattr(self.oracle, "temperature", float("nan")))
        if not math.isfinite(expected_temperature) or expected_temperature <= 0.0:
            raise ValueError("Frozen GINE oracle has an invalid calibration temperature")
        for prediction in predictions:
            _validate_prediction(prediction, checkpoint_id, expected_temperature)
        before = predictions[0]
        by_position = {
            position: predictions[offset + 1]
            for offset, position in enumerate(valid_positions)
        }

        scored_rows: list[dict[str, Any]] = []
        for position, application in enumerate(applications):
            row = dict(application)
            row.update(
                {
                    "dataset": DATASET,
                    "method": "GlobalGCE",
                    "parent_id": str(parent_id),
                    "parent_smiles": graphs[0].smiles,
                    "rule_id": rule.rule_id,
                    "rule_hash": rule.content_hash(),
                    "action_kind": "lhs_rhs_graph_transformation_rule",
                    "action_semantics": "native_lhs_to_rhs_attachment_aware_v1",
                    "action_engine_version": ACTION_ENGINE_VERSION,
                    "oracle_backend": ORACLE_BACKEND,
                    "classifier_family": CLASSIFIER_FAMILY,
                    "rf_oracle_used": False,
                    "oracle_checkpoint_hash": checkpoint_id,
                    "temperature": float(getattr(self.oracle, "temperature", 1.0)),
                    "cf_mode": CF_MODE,
                    "split_role": role,
                    "selector_frozen_before_test": bool(
                        selector_frozen_before_test
                    ),
                    "frozen_selector_hash": selector_hash or None,
                    "test_loaded": role == "test",
                    "gnn_scored": False,
                }
            )
            after = by_position.get(position)
            if after is None:
                row.update(
                    {
                        "pred_before": int(before["predicted_label"]),
                        "p_before_all_classes": list(before["probabilities"]),
                        "pred_after": None,
                        "p_after_all_classes": None,
                        "cf_drop": None,
                        "cf_flip": False,
                        "destination_label": None,
                    }
                )
            else:
                semantics = compute_counterfactual_semantics(
                    source_label=SOURCE_LABEL,
                    pred_before=before["predicted_label"],
                    pred_after=after["predicted_label"],
                    probabilities_before=before["probabilities"],
                    probabilities_after=after["probabilities"],
                    rule_id=rule.rule_id,
                )
                row.update(semantics.to_dict())
                row["gnn_scored"] = True
            scored_rows.append(row)

        return {
            "schema_version": FORWARD_EVAL_VERSION,
            "status": "PASS",
            "dataset": DATASET,
            "method": "GlobalGCE",
            "parent_id": str(parent_id),
            "rule_id": rule.rule_id,
            "rule_hash": rule.content_hash(),
            "split_role": role,
            "selector_frozen_before_test": bool(selector_frozen_before_test),
            "frozen_selector_hash": selector_hash or None,
            "test_loaded": role == "test",
            "application_count": len(applications),
            "valid_application_count": len(valid_positions),
            "gnn_scored_application_count": len(valid_positions),
            "strict_flip_count": sum(bool(row.get("cf_flip")) for row in scored_rows),
            "oracle_provenance": dict(self.provenance),
            "applications": scored_rows,
        }


def run_native_gine_forward_canary(
    *,
    parent_id: str,
    parent_smiles: str,
    rule_json: str | Path,
    gnn_checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    oracle_batch_size: int = 256,
) -> dict[str, Any]:
    """Persist a bounded action+GINE forward proof without releasing training."""

    rule_path = Path(rule_json).expanduser().resolve(strict=True)
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("GlobalGCE canary rule JSON must contain one object")
    rule = GlobalGCENativeRule.from_payload(payload)
    evaluator = BACEGlobalGCEFrozenGINEForwardEvaluator.from_checkpoint(
        gnn_checkpoint,
        device=device,
        oracle_batch_size=int(oracle_batch_size),
    )
    scored = evaluator.score_parent_rule(
        parent_id=parent_id,
        parent_smiles=parent_smiles,
        rule=rule,
        split_role="preflight_canary",
    )
    if int(scored["gnn_scored_application_count"]) < 1:
        raise ValueError(
            "GlobalGCE forward canary requires at least one valid native application"
        )
    blocker = baseline_spec("GlobalGCE")
    output = fresh_output_dir(output_dir)
    manifest = {
        "schema_version": FORWARD_EVAL_VERSION,
        "dataset": DATASET,
        "method": "GlobalGCE",
        "stage": "NATIVE_RULE_FROZEN_GINE_FORWARD_CANARY",
        "status": "PASS",
        "native_action_status": "PASS",
        "exact_frozen_gine_forward_status": "PASS",
        "full_rule_training_released": False,
        "full_rule_training_status": "BLOCKED_CODE",
        "full_rule_training_blocker": blocker.blocker_code,
        "rf_oracle_used": False,
        "test_loaded": False,
        "rule_input": file_identity(rule_path),
        "oracle_provenance": dict(evaluator.provenance),
        "parent_id": str(parent_id),
        "parent_smiles": str(parent_smiles),
        "rule_id": rule.rule_id,
        "rule_hash": rule.content_hash(),
        "application_count": scored["application_count"],
        "valid_application_count": scored["valid_application_count"],
        "gnn_scored_application_count": scored["gnn_scored_application_count"],
        "strict_flip_count": scored["strict_flip_count"],
        "created_at": utc_now(),
    }
    atomic_json(output / "native_gine_forward.json", scored)
    atomic_json(output / "run_manifest.json", manifest)
    atomic_json(output / "state.json", manifest)
    atomic_marker(output / "FORWARD_EVAL_PASS", "FORWARD_EVAL_PASS")
    return manifest


__all__ = [
    "BACEGlobalGCEFrozenGINEForwardEvaluator",
    "FORWARD_EVAL_VERSION",
    "run_native_gine_forward_canary",
]
