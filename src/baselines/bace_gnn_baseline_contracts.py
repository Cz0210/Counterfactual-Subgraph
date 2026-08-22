"""Fail-closed contracts shared by the BACE native-baseline GINE routes.

The module intentionally does not know how a baseline generates candidates.
It only freezes the classifier identity, native action semantics, split order,
fresh-output policy, and controller-facing terminal markers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from src.eval.bace_frozen_gnn_contracts import (
    atomic_json,
    atomic_marker,
    fresh_output_dir,
    sha256_file,
    utc_now,
)
from src.eval.bace_frozen_gnn_pool import _checkpoint_contract


DATASET = "bace"
SOURCE_LABEL = 1
NUM_CLASSES = 2
ORACLE_BACKEND = "gnn"
CLASSIFIER_FAMILY = "gine"
CF_MODE = "strict_flip"
ROUTE_ID = "bace_native_baseline_frozen_gine_v1"


@dataclass(frozen=True, slots=True)
class NativeBaselineSpec:
    method: str
    method_id: str
    action_kind: str
    action_semantics: str
    generation_resource: str
    verification_resource: str = "gpu"
    selector_resource: str = "cpu"
    native_route_available: bool = True
    blocker_code: str | None = None
    blocker_reason: str | None = None


BASELINE_SPECS: dict[str, NativeBaselineSpec] = {
    "gcfexplainer": NativeBaselineSpec(
        method="GCFExplainer",
        method_id="gcfexplainer",
        action_kind="full_counterfactual_graph",
        action_semantics="official_vrrw_neurosed_greedy_fullgraph_v1",
        generation_resource="gpu",
    ),
    "comrecgc": NativeBaselineSpec(
        method="ComRecGC",
        method_id="comrecgc",
        action_kind="native_common_recourse_fullgraph",
        action_semantics="official_comrecgc_lineage_unique_transition_medoid_v1",
        generation_resource="gpu",
    ),
    "globalgce": NativeBaselineSpec(
        method="GlobalGCE",
        method_id="globalgce",
        action_kind="lhs_rhs_graph_transformation_rule",
        action_semantics="native_lhs_to_rhs_attachment_aware_v1",
        generation_resource="cpu",
        verification_resource="cpu",
        native_route_available=False,
        blocker_code="BLOCKED_GLOBALGCE_LHS_RHS_ATTACHMENT_MAPPING_UNAVAILABLE",
        blocker_reason=(
            "The current project adapter exports LHS/RHS rule identities but has no "
            "audited attachment-aware rule application engine. A full-graph or deletion "
            "substitution would change GlobalGCE's native action semantics."
        ),
    ),
}


def normalize_method(method: str) -> str:
    value = str(method).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "gcfexplainer": "gcfexplainer",
        "gcf": "gcfexplainer",
        "comrecgc": "comrecgc",
        "globalgce": "globalgce",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported BACE baseline method: {method!r}") from exc


def baseline_spec(method: str) -> NativeBaselineSpec:
    return BASELINE_SPECS[normalize_method(method)]


def validate_bace_frozen_gine(
    checkpoint_dir: str | Path,
) -> tuple[Path, dict[str, Any], Any]:
    """Validate the one classifier allowed for all BACE baseline cells."""

    checkpoint = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError("BACE baseline GINE checkpoint must be a frozen bundle directory")
    card, schema = _checkpoint_contract(checkpoint)
    required = {
        "dataset": DATASET,
        "backbone": CLASSIFIER_FAMILY,
        "oracle_backend": ORACLE_BACKEND,
        "rf_oracle_used": False,
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
    }
    failures = [
        f"{field}={card.get(field)!r}"
        for field, expected in required.items()
        if card.get(field) != expected
    ]
    if failures:
        raise ValueError(
            "BACE baseline classifier provenance is not the frozen GINE contract: "
            + ", ".join(failures)
        )
    return checkpoint, dict(card), schema


def assert_gine_clean_manifest(
    manifest: Mapping[str, Any],
    *,
    checkpoint_id: str,
    require_train_only: bool,
) -> None:
    forbidden_text = json.dumps(dict(manifest), sort_keys=True).lower()
    forbidden_value = False

    def visit(value: Any, key: str = "") -> None:
        nonlocal forbidden_value
        if forbidden_value:
            return
        normalized_key = key.strip().lower().replace("-", "_")
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            if normalized in {
                "rf",
                "random_forest",
                "randomforestclassifier",
                "morgan_rf",
                "rf_contaminated",
            }:
                forbidden_value = True
        elif "rf" in normalized_key and value not in (False, 0, None, ""):
            forbidden_value = True

    visit(manifest)
    if forbidden_value or any(
        token in forbidden_text
        for token in ("randomforestclassifier", "rf_model.pkl", "morgan-rf")
    ):
        raise ValueError("BACE baseline manifest contains forbidden RF provenance")
    required = {
        "oracle_backend": ORACLE_BACKEND,
        "classifier_family": CLASSIFIER_FAMILY,
        "rf_oracle_used": False,
        "oracle_checkpoint_hash": checkpoint_id,
    }
    failures = [
        f"{field}={manifest.get(field)!r}"
        for field, expected in required.items()
        if manifest.get(field) != expected
    ]
    if require_train_only:
        if manifest.get("calibration_loaded") is not False:
            failures.append("calibration_loaded_not_false")
        if manifest.get("test_loaded") is not False:
            failures.append("test_loaded_not_false")
    if failures:
        raise ValueError("BACE GINE-clean manifest gate failed: " + ", ".join(failures))


def oracle_provenance(card: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "oracle_backend": ORACLE_BACKEND,
        "classifier_family": CLASSIFIER_FAMILY,
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "cf_mode": CF_MODE,
        "oracle_checkpoint": str(checkpoint),
        "oracle_checkpoint_hash": str(card["checkpoint_id"]),
        "model_pt_sha256": sha256_file(checkpoint / "model.pt"),
        "temperature_scaling_sha256": sha256_file(
            checkpoint / "temperature_scaling.json"
        ),
        "feature_schema_sha256": sha256_file(checkpoint / "feature_schema.json"),
    }


def write_route_preflight(
    *,
    method: str,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Publish a controller-readable READY or BLOCKED_CODE terminal contract."""

    spec = baseline_spec(method)
    checkpoint, card, _schema = validate_bace_frozen_gine(checkpoint_dir)
    output = fresh_output_dir(output_dir)
    provenance = oracle_provenance(card, checkpoint)
    payload = {
        "schema_version": ROUTE_ID,
        "dataset": DATASET,
        "method": spec.method,
        "method_id": spec.method_id,
        "stage": "PREFLIGHT",
        "status": "READY" if spec.native_route_available else "BLOCKED_CODE",
        "route_available": spec.native_route_available,
        "native_action": asdict(spec),
        **provenance,
        "split_order": [
            "train_generation",
            "calibration_verification",
            "calibration_selection_freeze",
            "heldout_test_verification",
            "final_freeze",
        ],
        "fresh_output_required": True,
        "pass_marker_published_last": True,
        "blocker_code": spec.blocker_code,
        "blocker_reason": spec.blocker_reason,
        "created_at": utc_now(),
    }
    atomic_json(output / "route_contract.json", payload)
    atomic_json(output / "oracle_provenance.json", provenance)
    atomic_json(output / "state.json", payload)
    if spec.native_route_available:
        atomic_marker(output / "READY", "READY")
    else:
        atomic_json(output / "BLOCKED_CODE.json", payload)
        atomic_marker(output / "BLOCKED_CODE", str(spec.blocker_code))
    return payload


__all__ = [
    "BASELINE_SPECS",
    "CF_MODE",
    "CLASSIFIER_FAMILY",
    "DATASET",
    "NativeBaselineSpec",
    "NUM_CLASSES",
    "ORACLE_BACKEND",
    "ROUTE_ID",
    "SOURCE_LABEL",
    "assert_gine_clean_manifest",
    "baseline_spec",
    "normalize_method",
    "oracle_provenance",
    "validate_bace_frozen_gine",
    "write_route_preflight",
]
