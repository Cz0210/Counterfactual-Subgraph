"""Release-candidate contracts for official-semantics fixed-budget NeuroSED.

These gates validate metadata and the GCF call direction without declaring a
scientific PASS.  Real GEDLIB labels, a trained checkpoint, and an independent
managed verifier remain mandatory external evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from src.data.tastemolnet_neurosed_fixed_budget import ALLOWED_TRAIN_PAIR_BUDGETS
from src.eval.tastemolnet_neurosed_fixed_budget import (
    OFFICIAL_GED_DIRECTION,
    OFFICIAL_SED_EDIT_COSTS,
    PAIR_LABELS_MANIFEST_SCHEMA,
    validation_pair_budget,
)
from src.eval.tastemolnet_neurosed_gate import STRICT_OFFICIAL_PROVENANCE
from src.train.tastemolnet_neurosed_official_selector import SELECTOR_TRACE_SCHEMA


OFFICIAL_FIXED_MODEL_CARD_SCHEMA = (
    "tastemolnet_gcf_neurosed_official_fixed_budget_model_card_v1"
)
DISTANCE_DIRECTION_SCHEMA = "tastemolnet_gcf_distance_direction_trace_v1"
READINESS_SCHEMA = "tastemolnet_neurosed_official_fixed_budget_readiness_v1"
GENERATED_QUERY_ROLE = "generated_counterfactual_candidate"
ORIGINAL_TARGET_ROLE = "original_input_graph"
VENDORED_GCF_SOURCE_SHA256 = {
    "neurosed/models.py": (
        "8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60"
    ),
    "distance.py": (
        "d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3"
    ),
    "importance.py": (
        "5e364634fcf6fac9c5e16b5d9dc2f53837ab67508421e5076010c1e9cdac33be"
    ),
    "vrrw.py": (
        "89ff1a9dbb9561d33dd4fbc1bffe84e60deeb069948778b39b75dc5c93a59fce"
    ),
    "summary.py": (
        "371ca30b9672bd17b472d261327dc343b989b52150257de8a8ce1c868389af44"
    ),
}


class OfficialFixedBudgetGateError(RuntimeError):
    """A fixed-budget release-candidate contract is incomplete or changed."""


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise OfficialFixedBudgetGateError(f"{label} is not a lowercase SHA256")
    return digest


def _commit(value: Any, *, label: str) -> str:
    commit = str(value or "")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise OfficialFixedBudgetGateError(f"{label} is not a full Git commit")
    return commit


def _matrix_shape(value: Any) -> tuple[int, int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            return None
        return dimensions if len(dimensions) == 2 else None
    if type(value) is list:
        if not value or any(type(row) is not list for row in value):
            return None
        widths = {len(row) for row in value}
        if len(widths) != 1:
            return None
        return len(value), widths.pop()
    return None


@dataclass(slots=True)
class GeneratedQueryOriginalTargetBinding:
    """Bind official embedded targets and expose only generated-query calls."""

    model: Any
    original_target_hashes: tuple[str, ...]
    _records: list[dict[str, Any]]
    _call_count: int = 0

    @classmethod
    def create(
        cls,
        model: Any,
        *,
        original_targets: Sequence[Any],
        original_target_hashes: Sequence[str],
    ) -> "GeneratedQueryOriginalTargetBinding":
        targets = list(original_targets)
        hashes = tuple(
            _sha256(value, label="original target graph hash")
            for value in original_target_hashes
        )
        if not targets or len(targets) != len(hashes):
            raise OfficialFixedBudgetGateError(
                "original target graphs/hashes must be non-empty and aligned"
            )
        embed = getattr(model, "embed_targets", None)
        predict = getattr(model, "predict_outer_with_queries", None)
        if not callable(embed) or not callable(predict):
            raise OfficialFixedBudgetGateError(
                "NeuroSED model lacks the official target/query API"
            )
        embed(targets)
        return cls(model=model, original_target_hashes=hashes, _records=[])

    def predict_generated_queries(
        self,
        generated_queries: Sequence[Any],
        *,
        generated_query_hashes: Sequence[str],
        batch_size: int | None = None,
    ) -> Any:
        queries = list(generated_queries)
        hashes = tuple(
            _sha256(value, label="generated query graph hash")
            for value in generated_query_hashes
        )
        if not queries or len(queries) != len(hashes):
            raise OfficialFixedBudgetGateError(
                "generated query graphs/hashes must be non-empty and aligned"
            )
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise OfficialFixedBudgetGateError("distance batch size must be positive")
        result = self.model.predict_outer_with_queries(queries, batch_size=batch_size)
        expected_shape = (len(queries), len(self.original_target_hashes))
        if _matrix_shape(result) != expected_shape:
            raise OfficialFixedBudgetGateError(
                "generated-query/original-target distance matrix shape changed"
            )
        for query_hash in hashes:
            for target_hash in self.original_target_hashes:
                self._records.append(
                    {
                        "distance_call_index": self._call_count,
                        "query_graph_hash": query_hash,
                        "target_graph_hash": target_hash,
                        "query_role": GENERATED_QUERY_ROLE,
                        "target_role": ORIGINAL_TARGET_ROLE,
                        "direction": OFFICIAL_GED_DIRECTION,
                    }
                )
        self._call_count += 1
        return result

    def direction_manifest(self) -> dict[str, Any]:
        if self._call_count <= 0 or not self._records:
            raise OfficialFixedBudgetGateError("no official GCF distance call was recorded")
        if any(
            row.get("query_role") != GENERATED_QUERY_ROLE
            or row.get("target_role") != ORIGINAL_TARGET_ROLE
            or row.get("direction") != OFFICIAL_GED_DIRECTION
            for row in self._records
        ):
            raise OfficialFixedBudgetGateError("GCF distance direction was reversed")
        payload = {
            "schema_version": DISTANCE_DIRECTION_SCHEMA,
            "status": "READY_FOR_INDEPENDENT_VERIFICATION",
            "distance_api": "NormGEDModel.predict_outer_with_queries",
            "targets_embedded_before_query_calls": True,
            "query_role": GENERATED_QUERY_ROLE,
            "target_role": ORIGINAL_TARGET_ROLE,
            "direction": "generated_query_to_original_target",
            "reverse_direction_used": False,
            "distance_call_count": self._call_count,
            "distance_pair_record_count": len(self._records),
            "original_target_hashes_sha256": _stable_sha256(
                list(self.original_target_hashes)
            ),
            "records": [dict(row) for row in self._records],
        }
        payload["trace_sha256"] = _stable_sha256(payload)
        return payload


def validate_official_fixed_budget_model_card(
    model_card: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the model-card claims required before independent verification."""

    card = dict(model_card)
    exact = {
        "schema_version": OFFICIAL_FIXED_MODEL_CARD_SCHEMA,
        "dataset": "tastemolnet",
        "role": "GCF_AUXILIARY_DISTANCE_MODEL",
        "classifier": False,
        "source_label_independent": True,
        "train_only_fit": True,
        "validation_only_selection": True,
        "calibration_loaded": False,
        "test_loaded": False,
        "pair_budget_strategy": "fixed_budget_resource_control",
        "fixed_pair_budget_is_project_extension": True,
        "official_pair_semantics": True,
        "fixed_budget_extension_documented": True,
        "upstream_greed_independent_pair_role_semantics_unchanged": True,
        "upstream_greed_sampler_byte_for_byte_unchanged": False,
        "exhaustive_pairs": False,
        "cartesian_product_materialized": False,
        "independent_query_target_pairs": True,
        "query_graph_id_differs_from_target_graph_id": True,
        "parent_own_subgraph_shortcut": False,
        "class_label_used_as_supervision": False,
        "real_pyged_gedlib_labels": True,
        "ged_method": "f2",
        "ged_method_switched_from_official": False,
        "approximate_or_neural_labels_used": False,
        "timeout_or_error_rows_used_as_labels": False,
        "label_representation": "ordered_query_target_lower_upper_interval",
        "pyged_return_dtype": "float64",
        "label_dtype": "float32",
        "label_transform": "official_torch_float32_storage_cast_only",
        "bound_average_used": False,
        "single_bound_substitution_used": False,
        "training_loop_authority": (
            "neuro.train.train_full_batch_interleaved_validation"
        ),
        "upstream_greed_batch_interleaved_selection_loop_unchanged": True,
        "strict_official_batch_interleaved_selector_implemented": True,
        "gcf_runtime_direction": "generated_query_to_original_target",
        "training_direction_matches_gcf_runtime": True,
        "checkpoint_reload_passed": True,
        "batch_single_inference_passed": True,
        "finite_labels": True,
        "all_lower_bounds_le_upper_bounds": True,
        "official_selection_trace_authenticated": True,
        "gcf_runner_load_passed": True,
        "feature_schema_compatible": True,
        "pair_sampling_seed": 7,
        "deterministic_reserve_fraction": 0.10,
        "disk_reservation_pass": True,
        "cpu_contention_gate_pass": True,
        "worker_wrote_pass": False,
        "scientific_release_eligible": True,
        "full_official_neurosed_semantics_claimed": True,
    }
    if any(
        type(card.get(key)) is not type(value) or card.get(key) != value
        for key, value in exact.items()
    ):
        raise OfficialFixedBudgetGateError(
            "official fixed-budget NeuroSED model-card contract changed"
        )
    train_budget = card.get("train_pair_budget")
    validation_budget = card.get("validation_pair_budget")
    if (
        type(train_budget) is not int
        or train_budget not in ALLOWED_TRAIN_PAIR_BUDGETS
        or type(validation_budget) is not int
        or validation_budget != validation_pair_budget(train_budget)
        or card.get("successful_train_pair_count") != train_budget
        or card.get("successful_validation_pair_count") != validation_budget
    ):
        raise OfficialFixedBudgetGateError("fixed train/validation pair budget changed")
    if card.get("edit_cost_contract") != OFFICIAL_SED_EDIT_COSTS:
        raise OfficialFixedBudgetGateError("official SED edit-cost contract changed")
    if card.get("strict_official_provenance") != STRICT_OFFICIAL_PROVENANCE:
        raise OfficialFixedBudgetGateError("strict official GREED provenance changed")
    if card.get("vendored_gcf_source_sha256") != VENDORED_GCF_SOURCE_SHA256:
        raise OfficialFixedBudgetGateError("vendored GCF source authority changed")
    _commit(card.get("official_gcf_commit"), label="official GCF commit")
    if card.get("official_greed_commit") != STRICT_OFFICIAL_PROVENANCE["greed_commit"]:
        raise OfficialFixedBudgetGateError("official GREED commit changed")
    _commit(card.get("gedlib_commit"), label="GEDLIB commit")
    for field in (
        "pyged_module_sha256",
        "gedlib_build_manifest_sha256",
        "gedlib_config_sha256",
        "feature_schema_sha256",
        "gedlib_benchmark_summary_sha256",
        "pair_budget_plan_sha256",
        "train_pair_labels_manifest_sha256",
        "validation_pair_labels_manifest_sha256",
        "train_pair_sampler_manifest_sha256",
        "validation_pair_sampler_manifest_sha256",
        "selector_trace_sha256",
        "distance_direction_trace_sha256",
        "selected_checkpoint_sha256",
    ):
        _sha256(card.get(field), label=field)
    return card


def verify_official_fixed_budget_readiness(
    *,
    model_card: Mapping[str, Any],
    train_pair_labels_manifest: Mapping[str, Any],
    validation_pair_labels_manifest: Mapping[str, Any],
    selector_trace: Mapping[str, Any],
    distance_direction_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-bind local contracts; return readiness, never a scientific PASS."""

    card = validate_official_fixed_budget_model_card(model_card)
    train_labels = dict(train_pair_labels_manifest)
    validation_labels = dict(validation_pair_labels_manifest)
    selector = dict(selector_trace)
    direction = dict(distance_direction_trace)
    for split, manifest, budget in (
        ("train", train_labels, card["train_pair_budget"]),
        ("validation", validation_labels, card["validation_pair_budget"]),
    ):
        if (
            manifest.get("schema_version") != PAIR_LABELS_MANIFEST_SCHEMA
            or manifest.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
            or manifest.get("split") != split
            or manifest.get("requested_pair_count") != budget
            or manifest.get("successful_pair_count") != budget
            or manifest.get("real_pyged_gedlib_labels") is not True
            or manifest.get("timeout_or_error_rows_used_as_labels") is not False
            or manifest.get("selected_in_sampler_order") is not True
            or manifest.get("ged_value_based_selection_used") is not False
            or manifest.get("finite_labels") is not True
            or manifest.get("all_lower_bounds_le_upper_bounds") is not True
            or manifest.get("cache_symmetric") is not False
            or manifest.get("reverse_cache_shared") is not False
            or manifest.get("query_target_order_in_cache_key") is not True
            or manifest.get("large_per_pair_json_debug_dump_used") is not False
            or manifest.get("compact_storage_format")
            not in ("parquet", "arrow_ipc", "numpy_npz")
            or manifest.get("gedlib_commit") != card["gedlib_commit"]
            or manifest.get("pyged_module_sha256")
            != card["pyged_module_sha256"]
            or manifest.get("gedlib_build_manifest_sha256")
            != card["gedlib_build_manifest_sha256"]
            or manifest.get("gedlib_config_sha256")
            != card["gedlib_config_sha256"]
            or manifest.get("feature_schema_sha256")
            != card["feature_schema_sha256"]
            or manifest.get("pair_sampler_manifest_sha256")
            != card[f"{split}_pair_sampler_manifest_sha256"]
            or manifest.get("calibration_loaded") is not False
            or manifest.get("test_loaded") is not False
        ):
            raise OfficialFixedBudgetGateError(
                f"{split} official pair-label manifest is not release-ready"
            )
        exact_count = manifest.get("exact_bound_pair_count")
        interval_count = manifest.get("interval_bound_pair_count")
        if (
            type(exact_count) is not int
            or type(interval_count) is not int
            or min(exact_count, interval_count) < 0
            or exact_count + interval_count != budget
        ):
            raise OfficialFixedBudgetGateError(
                f"{split} exact/interval label accounting changed"
            )
        _sha256(manifest.get("compact_labels_sha256"), label=f"{split} labels")
        if manifest.get("manifest_sha256") != _stable_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ):
            raise OfficialFixedBudgetGateError(
                f"{split} pair-label manifest hash changed"
            )
    if (
        selector.get("schema_version") != SELECTOR_TRACE_SCHEMA
        or selector.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
        or selector.get("selector_contract")
        != "neuro.train.train_full_batch_interleaved_validation"
        or selector.get("validation_before_every_training_batch") is not True
        or selector.get("stopped_before_paired_training_batch") is not True
        or selector.get("epoch_end_validation_used") is not False
        or selector.get("selected_checkpoint_sha256")
        != card["selected_checkpoint_sha256"]
        or selector.get("trace_sha256")
        != _stable_sha256(
            {key: value for key, value in selector.items() if key != "trace_sha256"}
        )
    ):
        raise OfficialFixedBudgetGateError("official selector trace changed")
    if (
        direction.get("schema_version") != DISTANCE_DIRECTION_SCHEMA
        or direction.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION"
        or direction.get("query_role") != GENERATED_QUERY_ROLE
        or direction.get("target_role") != ORIGINAL_TARGET_ROLE
        or direction.get("direction") != "generated_query_to_original_target"
        or direction.get("reverse_direction_used") is not False
        or type(direction.get("distance_call_count")) is not int
        or direction["distance_call_count"] <= 0
        or direction.get("trace_sha256")
        != _stable_sha256(
            {key: value for key, value in direction.items() if key != "trace_sha256"}
        )
    ):
        raise OfficialFixedBudgetGateError("GCF generated-query direction changed")
    bindings = {
        "train_pair_labels_manifest_sha256": train_labels["manifest_sha256"],
        "validation_pair_labels_manifest_sha256": validation_labels[
            "manifest_sha256"
        ],
        "selector_trace_sha256": str(selector.get("trace_sha256") or ""),
        "distance_direction_trace_sha256": str(direction.get("trace_sha256") or ""),
    }
    if any(card.get(field) != digest for field, digest in bindings.items()):
        raise OfficialFixedBudgetGateError("model card does not bind fixed-budget evidence")
    return {
        "schema_version": READINESS_SCHEMA,
        "status": "READY_FOR_MANAGED_INDEPENDENT_VERIFICATION",
        "marker": None,
        "scientific_pass_claimed": False,
        "real_gedlib_execution_required": True,
        "checkpoint_execution_required": True,
        "model_card_contract_valid": True,
        "train_pair_labels_contract_valid": True,
        "validation_pair_labels_contract_valid": True,
        "official_selector_contract_valid": True,
        "generated_query_original_target_direction_valid": True,
        "evidence_bindings": bindings,
    }


__all__ = [
    "DISTANCE_DIRECTION_SCHEMA",
    "GENERATED_QUERY_ROLE",
    "GeneratedQueryOriginalTargetBinding",
    "OFFICIAL_FIXED_MODEL_CARD_SCHEMA",
    "ORIGINAL_TARGET_ROLE",
    "OfficialFixedBudgetGateError",
    "READINESS_SCHEMA",
    "VENDORED_GCF_SOURCE_SHA256",
    "validate_official_fixed_budget_model_card",
    "verify_official_fixed_budget_readiness",
]
