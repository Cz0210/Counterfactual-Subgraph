"""Declarative fixed-GINE downstream plan for the proposer ablation.

The plan is intentionally non-executable.  It proves that every proposer will
feed the same classifier, split, selector, threshold, WNode, and evaluator
identities, while leaving all scientific execution to an independently
reviewed future runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import (
    ArtifactPin,
    LLMAblationContractError,
    LLMProposerVariant,
    canonical_json_sha256,
)


@dataclass(frozen=True, slots=True)
class CommonDownstreamPlan:
    dataset: str
    source_label: int
    num_classes: int
    train_split: ArtifactPin
    validation_split: ArtifactPin
    calibration_split: ArtifactPin
    test_split: ArtifactPin
    gine_checkpoint: ArtifactPin
    temperature_manifest: ArtifactPin
    feature_schema: ArtifactPin
    selector_config: ArtifactPin
    threshold_config: ArtifactPin
    evaluator_config: ArtifactPin
    molclr_checkpoint: ArtifactPin
    k_max: int = 20
    table2_k: int = 10
    counterfactual_mode: str = "strict_flip"
    schema_version: str = "llm_ablation_common_downstream_plan_v1"

    def __post_init__(self) -> None:
        if str(self.dataset).lower() != "bace":
            raise LLMAblationContractError("LLM proposer ablation is fixed to BACE")
        for field in ("source_label", "num_classes", "k_max", "table2_k"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise LLMAblationContractError(f"{field} must be an integer, not bool")
        if self.source_label != 1 or self.num_classes != 2:
            raise LLMAblationContractError(
                "BACE common downstream requires source_label=1 and num_classes=2"
            )
        if self.counterfactual_mode != "strict_flip":
            raise LLMAblationContractError("counterfactual_mode must be strict_flip")
        if self.k_max != 20 or self.table2_k != 10:
            raise LLMAblationContractError("fixed downstream requires K_MAX=20/Table2 K=10")
        artifact_names = (
            "train_split",
            "validation_split",
            "calibration_split",
            "test_split",
            "gine_checkpoint",
            "temperature_manifest",
            "feature_schema",
            "selector_config",
            "threshold_config",
            "evaluator_config",
            "molclr_checkpoint",
        )
        if tuple(getattr(self, name).role for name in artifact_names) != artifact_names:
            raise LLMAblationContractError("common downstream artifact roles are not exact")
        split_paths = {
            self.train_split.resolved_path,
            self.validation_split.resolved_path,
            self.calibration_split.resolved_path,
            self.test_split.resolved_path,
        }
        if len(split_paths) != 4:
            raise LLMAblationContractError("train/validation/calibration/test paths must differ")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CommonDownstreamPlan":
        if not isinstance(payload, Mapping):
            raise LLMAblationContractError("common_downstream must be an object")
        expected_keys = {
            "dataset",
            "source_label",
            "num_classes",
            "counterfactual_mode",
            "k_max",
            "table2_k",
            "train_split",
            "validation_split",
            "calibration_split",
            "test_split",
            "gine_checkpoint",
            "temperature_manifest",
            "feature_schema",
            "selector_config",
            "threshold_config",
            "evaluator_config",
            "molclr_checkpoint",
        }
        if set(payload) != expected_keys:
            missing = sorted(expected_keys - set(payload))
            extra = sorted(set(payload) - expected_keys)
            raise LLMAblationContractError(
                f"common_downstream keys mismatch; missing={missing}, extra={extra}"
            )

        def pin(name: str) -> ArtifactPin:
            return ArtifactPin.from_mapping(payload.get(name), role=name)

        return cls(
            dataset=str(payload.get("dataset") or ""),
            source_label=payload.get("source_label"),
            num_classes=payload.get("num_classes"),
            train_split=pin("train_split"),
            validation_split=pin("validation_split"),
            calibration_split=pin("calibration_split"),
            test_split=pin("test_split"),
            gine_checkpoint=pin("gine_checkpoint"),
            temperature_manifest=pin("temperature_manifest"),
            feature_schema=pin("feature_schema"),
            selector_config=pin("selector_config"),
            threshold_config=pin("threshold_config"),
            evaluator_config=pin("evaluator_config"),
            molclr_checkpoint=pin("molclr_checkpoint"),
            k_max=payload.get("k_max", 20),
            table2_k=payload.get("table2_k", 10),
            counterfactual_mode=str(payload.get("counterfactual_mode") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        pins = {
            name: getattr(self, name).to_dict()
            for name in (
                "train_split",
                "validation_split",
                "calibration_split",
                "test_split",
                "gine_checkpoint",
                "temperature_manifest",
                "feature_schema",
                "selector_config",
                "threshold_config",
                "evaluator_config",
                "molclr_checkpoint",
            )
        }
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset": "BACE",
            "source_label": self.source_label,
            "num_classes": self.num_classes,
            "counterfactual_mode": self.counterfactual_mode,
            "k_max": self.k_max,
            "table2_k": self.table2_k,
            "variant_order": [variant.value for variant in LLMProposerVariant],
            "shared_identity_for_all_variants": True,
            "generation_access": ["train"],
            "generation_oracle_ranking": False,
            "selector_access": ["calibration"],
            "selector_uses_test": False,
            "test_access_after_selector_freeze_only": True,
            "evaluation_action": "connected_sanitized_hard_deletion",
            "distance": "MolCLR-Node-Wasserstein",
            "science_execution_requested": False,
            "science_executed": False,
            "artifacts": pins,
        }
        payload["plan_sha256"] = canonical_json_sha256(payload)
        return payload


def build_common_downstream_plan(
    payload: Mapping[str, Any],
    *,
    execute_science: bool = False,
) -> CommonDownstreamPlan:
    """Validate and return a plan; executing science is not an allowed mode."""

    if execute_science is not False:
        raise LLMAblationContractError(
            "this framework may construct/validate the common downstream plan only"
        )
    plan = CommonDownstreamPlan.from_mapping(payload)
    plan_payload = plan.to_dict()
    if plan_payload["science_executed"] is not False:
        raise LLMAblationContractError("common downstream plan must remain non-executing")
    return plan


__all__ = ["CommonDownstreamPlan", "build_common_downstream_plan"]
