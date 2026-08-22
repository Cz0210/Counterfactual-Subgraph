"""Fail-closed multiclass adapters for future TasteMolNet baselines.

This module is deliberately free of training and I/O.  It freezes the common
classifier and counterfactual semantics that GCFExplainer, GlobalGCE, and
ComRecGC must obey after the exact TasteMolNet data license is approved.  The
current AutoDL task fragment remains blocked and never calls these adapters.

The adapters preserve each baseline's native action:

* GCFExplainer accepts complete counterfactual graphs predicted outside Sweet;
* GlobalGCE merges the Sweet->Bitter and Sweet->Tasteless native rule branches
  before calibration selection;
* ComRecGC accepts only one globally identified, pinned-upstream single-edit
  transition and never treats parent metadata as graph content identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from src.data.dataset_registry import assert_oracle_backend_allowed
from src.eval.counterfactual_semantics import (
    compute_counterfactual_semantics,
    destination_distribution,
    strict_flip,
)


DATASET = "tastemolnet"
DISPLAY_NAME = "TasteMolNet"
NUM_CLASSES = 3
SOURCE_LABEL = 1
SOURCE_LABEL_NAME = "Sweet"
LABEL_MAP = {0: "Bitter", 1: "Sweet", 2: "Tasteless"}
DESTINATION_LABELS = (0, 2)
ORACLE_BACKEND = "gnn"
CLASSIFIER_FAMILY = "gine"
CF_MODE = "untargeted_strict_flip"
RF_ORACLE_USED = False
GLOBAL_GRAPH_IDENTITY = "canonical_global_graph_hash"
GLOBALGCE_TARGET_BRANCHES = DESTINATION_LABELS
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class TasteMulticlassContractError(ValueError):
    """A TasteMolNet baseline would weaken a frozen scientific contract."""


@dataclass(frozen=True, slots=True)
class TasteFrozenGINEIdentity:
    """Minimal immutable identity shared by all four TasteMolNet methods."""

    checkpoint_hash: str
    temperature_hash: str
    feature_schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": DATASET,
            "oracle_backend": ORACLE_BACKEND,
            "classifier_family": CLASSIFIER_FAMILY,
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "source_label_name": SOURCE_LABEL_NAME,
            "rf_oracle_used": RF_ORACLE_USED,
            "cf_mode": CF_MODE,
            "oracle_checkpoint_hash": self.checkpoint_hash,
            "temperature_calibration_hash": self.temperature_hash,
            "feature_schema_hash": self.feature_schema_hash,
        }


def _text(value: Any, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise TasteMulticlassContractError(f"{field} is required")
    return result


def _hash(value: Any, *, field: str) -> str:
    result = _text(value, field=field).lower()
    if _HEX_64.fullmatch(result) is None:
        raise TasteMulticlassContractError(f"{field} must be one SHA-256 digest")
    return result


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TasteMulticlassContractError(f"{field} must be an integer class")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TasteMulticlassContractError(
            f"{field} must be an integer class"
        ) from exc
    if str(value).strip() not in {str(result), f"+{result}"} and not isinstance(
        value, int
    ):
        raise TasteMulticlassContractError(f"{field} is not an exact integer")
    if result < 0 or result >= NUM_CLASSES:
        raise TasteMulticlassContractError(
            f"{field}={result} is outside [0, {NUM_CLASSES - 1}]"
        )
    return result


def validate_frozen_gine_manifest(
    manifest: Mapping[str, Any],
) -> TasteFrozenGINEIdentity:
    """Validate the only classifier identity allowed by TasteMolNet methods.

    The model may be selected and calibrated with validation data, but a frozen
    classifier manifest used to construct baseline routes must explicitly prove
    that held-out test data was not loaded for fitting or selection.
    """

    dataset = _text(manifest.get("dataset"), field="dataset").lower()
    backend = _text(manifest.get("oracle_backend"), field="oracle_backend").lower()
    family = _text(
        manifest.get("classifier_family") or manifest.get("backbone"),
        field="classifier_family",
    ).lower()
    assert_oracle_backend_allowed(dataset, backend)
    failures: list[str] = []
    if dataset != DATASET:
        failures.append(f"dataset={dataset!r}")
    if backend != ORACLE_BACKEND:
        failures.append(f"oracle_backend={backend!r}")
    if family != CLASSIFIER_FAMILY:
        failures.append(f"classifier_family={family!r}")
    if manifest.get("rf_oracle_used") is not False:
        failures.append("rf_oracle_used_not_false")
    if manifest.get("num_classes") != NUM_CLASSES:
        failures.append(f"num_classes={manifest.get('num_classes')!r}")
    if manifest.get("source_label") != SOURCE_LABEL:
        failures.append(f"source_label={manifest.get('source_label')!r}")
    if manifest.get("test_loaded") is not False:
        failures.append("test_loaded_not_false")
    if manifest.get("test_used_for_selection") not in {None, False}:
        failures.append("test_used_for_selection_not_false")
    serialized = json.dumps(dict(manifest), sort_keys=True).lower()
    if any(
        token in serialized
        for token in (
            "randomforestclassifier",
            "rf_model.pkl",
            "morgan-rf",
            '"oracle_backend": "rf"',
            '"teacher_backend": "rf"',
        )
    ):
        failures.append("forbidden_rf_provenance")
    if failures:
        raise TasteMulticlassContractError(
            "TasteMolNet frozen-GINE contract failed: " + ", ".join(failures)
        )
    return TasteFrozenGINEIdentity(
        checkpoint_hash=_hash(
            manifest.get("oracle_checkpoint_hash") or manifest.get("checkpoint_id"),
            field="oracle_checkpoint_hash",
        ),
        temperature_hash=_hash(
            manifest.get("temperature_calibration_hash")
            or manifest.get("temperature_scaling_sha256"),
            field="temperature_calibration_hash",
        ),
        feature_schema_hash=_hash(
            manifest.get("feature_schema_hash")
            or manifest.get("feature_schema_sha256"),
            field="feature_schema_hash",
        ),
    )


def is_taste_strict_flip(*, pred_before: Any, pred_after: Any) -> bool:
    """Return the untargeted Sweet-to-non-Sweet strict-flip decision."""

    before = _integer(pred_before, field="pred_before")
    after = _integer(pred_after, field="pred_after")
    return strict_flip(before, after, SOURCE_LABEL)


def gcf_candidate_condition(pred_candidate: Any) -> bool:
    """Native GCFExplainer full-graph counterfactual candidate condition."""

    return _integer(pred_candidate, field="pred_candidate") != SOURCE_LABEL


def adapt_gcf_fullgraph_score(
    *,
    pred_before: Any,
    pred_candidate: Any,
    probabilities_before: Sequence[float],
    probabilities_candidate: Sequence[float],
    candidate_id: str,
) -> dict[str, Any]:
    """Return a complete three-class score without projecting the full graph."""

    record = compute_counterfactual_semantics(
        source_label=SOURCE_LABEL,
        pred_before=_integer(pred_before, field="pred_before"),
        pred_after=_integer(pred_candidate, field="pred_candidate"),
        probabilities_before=probabilities_before,
        probabilities_after=probabilities_candidate,
        rule_id=_text(candidate_id, field="candidate_id"),
    )
    return {
        **record.to_dict(),
        "dataset": DATASET,
        "method": "GCFExplainer",
        "action_kind": "full_counterfactual_graph",
        "candidate_condition": "pred_candidate != source_label",
        "native_action_preserved": True,
    }


def _canonical_action_identity(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    rule_hash = _hash(row.get("rule_hash"), field="rule_hash")
    action = {
        "rule_hash": rule_hash,
        "lhs_hash": _hash(row.get("lhs_hash"), field="lhs_hash"),
        "rhs_hash": _hash(row.get("rhs_hash"), field="rhs_hash"),
        "attachment_map_hash": _hash(
            row.get("attachment_map_hash"), field="attachment_map_hash"
        ),
        "action_kind": _text(row.get("action_kind"), field="action_kind"),
    }
    if action["action_kind"] != "lhs_rhs_graph_transformation_rule":
        raise TasteMulticlassContractError(
            "GlobalGCE must preserve LHS->RHS transformation-rule actions"
        )
    return rule_hash, action


def merge_globalgce_target_branches(
    branches: Mapping[int, Iterable[Mapping[str, Any]]],
    *,
    oracle_checkpoint_hash: str,
) -> list[dict[str, Any]]:
    """Merge target-0/target-2 rules before any calibration selection.

    Duplicate native rules are collapsed by their frozen ``rule_hash``.  A hash
    collision or any action mismatch fails closed instead of taking the first
    representative.  Every row must come from train-only generation against
    the same three-class GINE.
    """

    expected_hash = _hash(
        oracle_checkpoint_hash, field="oracle_checkpoint_hash"
    )
    branch_ids = set(branches)
    if branch_ids != set(GLOBALGCE_TARGET_BRANCHES):
        raise TasteMulticlassContractError(
            "GlobalGCE requires exactly target branches {0, 2} before calibration"
        )
    merged: dict[str, dict[str, Any]] = {}
    action_by_hash: dict[str, dict[str, Any]] = {}
    provenance: dict[str, set[int]] = {}
    for raw_target in GLOBALGCE_TARGET_BRANCHES:
        rows = list(branches[raw_target])
        for raw in rows:
            row = dict(raw)
            target = _integer(row.get("target_label"), field="target_label")
            if target != raw_target or target not in DESTINATION_LABELS:
                raise TasteMulticlassContractError(
                    "GlobalGCE row is stored under the wrong target branch"
                )
            if row.get("source_label") != SOURCE_LABEL:
                raise TasteMulticlassContractError(
                    "GlobalGCE row must declare Sweet source_label=1"
                )
            if str(row.get("data_split_used") or "").lower() != "train":
                raise TasteMulticlassContractError(
                    "GlobalGCE target pools must be generated from train only"
                )
            if row.get("calibration_loaded") is not False or row.get(
                "test_loaded"
            ) is not False:
                raise TasteMulticlassContractError(
                    "GlobalGCE branches must merge before calibration/test access"
                )
            if row.get("rf_oracle_used") is not False or str(
                row.get("oracle_backend") or ""
            ).lower() != ORACLE_BACKEND:
                raise TasteMulticlassContractError(
                    "GlobalGCE branches must use the frozen GINE and no RF"
                )
            observed_hash = _hash(
                row.get("oracle_checkpoint_hash"),
                field="oracle_checkpoint_hash",
            )
            if observed_hash != expected_hash:
                raise TasteMulticlassContractError(
                    "GlobalGCE target branches use different GINE checkpoints"
                )
            rule_hash, action = _canonical_action_identity(row)
            if rule_hash in action_by_hash and action_by_hash[rule_hash] != action:
                raise TasteMulticlassContractError(
                    "GLOBALGCE_RULE_HASH_COLLISION_OR_CORRUPTION"
                )
            action_by_hash[rule_hash] = action
            provenance.setdefault(rule_hash, set()).add(target)
            if rule_hash not in merged:
                merged[rule_hash] = {
                    **row,
                    **action,
                    "dataset": DATASET,
                    "method": "GlobalGCE",
                    "source_label": SOURCE_LABEL,
                    "num_classes": NUM_CLASSES,
                    "cf_mode": CF_MODE,
                    "branch_merge_stage": "before_calibration",
                    "native_action_preserved": True,
                }
    result: list[dict[str, Any]] = []
    for rule_hash in sorted(merged):
        row = dict(merged[rule_hash])
        row["target_branches"] = sorted(provenance[rule_hash])
        row.pop("target_label", None)
        result.append(row)
    return result


def adapt_comrecgc_transition(
    row: Mapping[str, Any],
    *,
    pred_before: Any,
    pred_after: Any,
    probabilities_before: Sequence[float],
    probabilities_after: Sequence[float],
) -> dict[str, Any]:
    """Validate global lineage and score one native ComRecGC transition."""

    required_true = (
        "transition_uniqueness_enforced",
        "lineage_unique",
        "upstream_identity_matches",
        "downstream_hash_matches",
    )
    failures = [field for field in required_true if row.get(field) is not True]
    if row.get("graph_content_identity") != GLOBAL_GRAPH_IDENTITY:
        failures.append("graph_content_identity")
    if row.get("parent_metadata_is_graph_identity") is not False:
        failures.append("parent_metadata_is_graph_identity")
    if row.get("single_edit_count") != 1:
        failures.append("single_edit_count")
    if row.get("true_transition_count") != 1:
        failures.append("true_transition_count")
    if row.get("graph_hash_collision_or_corruption") is not False:
        failures.append("graph_hash_collision_or_corruption")
    if str(row.get("oracle_backend") or "").lower() != ORACLE_BACKEND:
        failures.append("oracle_backend")
    if row.get("rf_oracle_used") is not False:
        failures.append("rf_oracle_used_not_false")
    if row.get("num_classes") != NUM_CLASSES:
        failures.append("num_classes")
    if row.get("source_label") != SOURCE_LABEL:
        failures.append("source_label")
    if failures:
        raise TasteMulticlassContractError(
            "ComRecGC global-lineage gate failed: " + ", ".join(failures)
        )
    transition_id = _text(row.get("transition_id"), field="transition_id")
    record = compute_counterfactual_semantics(
        source_label=SOURCE_LABEL,
        pred_before=_integer(pred_before, field="pred_before"),
        pred_after=_integer(pred_after, field="pred_after"),
        probabilities_before=probabilities_before,
        probabilities_after=probabilities_after,
        rule_id=transition_id,
    )
    return {
        **dict(row),
        **record.to_dict(),
        "dataset": DATASET,
        "method": "ComRecGC",
        "action_kind": "native_common_recourse_transition",
        "graph_content_identity": GLOBAL_GRAPH_IDENTITY,
        "parent_metadata_is_graph_identity": False,
        "native_action_preserved": True,
    }


def authorize_split_access(
    *,
    split: str,
    selector_manifest: Mapping[str, Any] | None = None,
    oracle_checkpoint_hash: str,
) -> None:
    """Enforce calibration selection and one-shot test-after-freeze access."""

    normalized = str(split or "").strip().lower()
    expected_hash = _hash(
        oracle_checkpoint_hash, field="oracle_checkpoint_hash"
    )
    if normalized in {"train", "validation", "calibration"}:
        if normalized != "calibration" and selector_manifest is not None:
            raise TasteMulticlassContractError(
                "Selector evidence is accepted only at/after calibration"
            )
        return
    if normalized != "test":
        raise TasteMulticlassContractError(f"Unsupported split: {split!r}")
    if selector_manifest is None:
        raise TasteMulticlassContractError(
            "Held-out test is unavailable before selector freeze"
        )
    failures: list[str] = []
    if selector_manifest.get("selection_frozen") is not True:
        failures.append("selection_frozen")
    if selector_manifest.get("selector_fitted_on_calibration") is not True:
        failures.append("selector_fitted_on_calibration")
    if selector_manifest.get("calibration_loaded") is not True:
        failures.append("calibration_loaded")
    if selector_manifest.get("test_loaded") is not False:
        failures.append("test_loaded_not_false")
    if selector_manifest.get("source_label") != SOURCE_LABEL:
        failures.append("source_label")
    if selector_manifest.get("num_classes") != NUM_CLASSES:
        failures.append("num_classes")
    if selector_manifest.get("cf_mode") != CF_MODE:
        failures.append("cf_mode")
    observed_hash = selector_manifest.get("oracle_checkpoint_hash")
    if observed_hash != expected_hash:
        failures.append("oracle_checkpoint_hash")
    ordered = selector_manifest.get("ordered_rule_ids")
    if not isinstance(ordered, list) or not ordered or len(ordered) > 20:
        failures.append("ordered_rule_ids")
    if failures:
        raise TasteMulticlassContractError(
            "Held-out test is unavailable before a complete calibration freeze: "
            + ", ".join(failures)
        )


def taste_destination_distribution(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return Bitter/Tasteless destinations overall and per native rule."""

    return destination_distribution(
        rows,
        source_label=SOURCE_LABEL,
        num_classes=NUM_CLASSES,
        label_map=LABEL_MAP,
    )


def multiclass_extension_manifest(method: str) -> dict[str, Any]:
    """Return the frozen, method-specific extension contract."""

    normalized = str(method or "").strip().lower().replace("_", "")
    contracts = {
        "gcfexplainer": {
            "method": "GCFExplainer",
            "action_kind": "full_counterfactual_graph",
            "candidate_condition": "pred_candidate != source_label",
            "importance": "1 - p_source_or_max_non_source",
        },
        "globalgce": {
            "method": "GlobalGCE",
            "action_kind": "lhs_rhs_graph_transformation_rule",
            "target_branches": list(GLOBALGCE_TARGET_BRANCHES),
            "branch_merge": "deduplicate_before_calibration_selector",
        },
        "comrecgc": {
            "method": "ComRecGC",
            "action_kind": "native_common_recourse_transition",
            "candidate_condition": "pred_after != source_label",
            "graph_content_identity": GLOBAL_GRAPH_IDENTITY,
            "transition_uniqueness_enforced": True,
        },
    }
    try:
        method_contract = contracts[normalized]
    except KeyError as exc:
        raise TasteMulticlassContractError(
            f"Unsupported TasteMolNet baseline method: {method!r}"
        ) from exc
    return {
        "schema_version": "tastemolnet_multiclass_baseline_adapter_v1",
        "dataset": DATASET,
        "num_classes": NUM_CLASSES,
        "label_map": {str(key): value for key, value in LABEL_MAP.items()},
        "source_label": SOURCE_LABEL,
        "source_label_name": SOURCE_LABEL_NAME,
        "destination_labels": list(DESTINATION_LABELS),
        "cf_mode": CF_MODE,
        "strict_flip_definition": (
            "pred_before == source_label and pred_after != source_label"
        ),
        "oracle_backend": ORACLE_BACKEND,
        "classifier_family": CLASSIFIER_FAMILY,
        "rf_oracle_used": False,
        "separate_binary_explainee_forbidden": True,
        "calibration_merge_before_selection": True,
        "test_loaded_only_after_selector_freeze": True,
        "native_action_preserved": True,
        **method_contract,
    }


def canonical_manifest_hash(payload: Mapping[str, Any]) -> str:
    """Hash a contract without relying on dictionary insertion order."""

    encoded = json.dumps(
        dict(payload), separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CF_MODE",
    "CLASSIFIER_FAMILY",
    "DATASET",
    "DESTINATION_LABELS",
    "GLOBALGCE_TARGET_BRANCHES",
    "GLOBAL_GRAPH_IDENTITY",
    "LABEL_MAP",
    "NUM_CLASSES",
    "ORACLE_BACKEND",
    "RF_ORACLE_USED",
    "SOURCE_LABEL",
    "SOURCE_LABEL_NAME",
    "TasteFrozenGINEIdentity",
    "TasteMulticlassContractError",
    "adapt_comrecgc_transition",
    "adapt_gcf_fullgraph_score",
    "authorize_split_access",
    "canonical_manifest_hash",
    "gcf_candidate_condition",
    "is_taste_strict_flip",
    "merge_globalgce_target_branches",
    "multiclass_extension_manifest",
    "taste_destination_distribution",
    "validate_frozen_gine_manifest",
]
