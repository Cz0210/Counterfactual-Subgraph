"""Scientific identity and cohort contracts for GNN backbone ablations.

The proposal axis is a frozen pool of rules produced from the training split.
Those rules are deliberately independent of the calibration/test source
cohorts on which each frozen classifier is compared.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from src.data.dataset_registry import get_dataset_spec, normalize_dataset_id
from src.models.gnn_backbone_registry import (
    get_gnn_backbone_spec,
    normalize_gnn_backbone,
)


CANDIDATE_IDENTITY_SCHEMA = "gnn_train_rule_candidate_identity_v2"
PROPOSAL_UNIVERSE_SCHEMA = "gnn_train_rule_proposal_universe_v2"
PARENT_PREDICTION_SCHEMA = "gnn_backbone_parent_prediction_v2"
COHORT_SCHEMA = "gnn_backbone_source_cohort_v2"
COHORT_FREEZE_SCHEMA = "gnn_backbone_split_cohort_freeze_v2"
COHORT_SPLIT_AUTHORITY_SCHEMA = "gnn_backbone_split_authority_v1"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PROPOSAL_FIELDS = frozenset(
    {
        "backbone",
        "backbone_name",
        "checkpoint_id",
        "checkpoint_sha256",
        "classifier_score",
        "oracle_score",
        "logits",
        "probabilities",
        "predicted_label",
        "selected",
        "strict_flip",
        "calibration_parent_id",
        "test_parent_id",
    }
)
_PROPOSAL_INPUT_FIELDS = frozenset(
    {
        "dataset",
        "proposal_index",
        "proposal_source_sha256",
        "rule_id",
        "rule_sha256",
        "fragment_graph_sha256",
        "source_split",
        "action_type",
    }
)
_PROPOSAL_DERIVED_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "candidate_identity_sha256",
        "identity_excludes",
        "proposal_record_sha256",
    }
)


class GNNAblationContractError(ValueError):
    """One GNN-ablation identity or cohort contract is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GNNAblationContractError("payload is not canonical JSON") from exc


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_cohort_split_authority(
    *,
    dataset: str,
    split: str,
    split_sha256: str,
    feature_schema_sha256: str,
    backbones: Sequence[str],
    checkpoint_sha256s: Mapping[str, str],
    temperature_scaling_sha256s: Mapping[str, str],
    parent_ids: Sequence[str],
    parent_graph_sha256s: Mapping[str, str],
    true_labels: Mapping[str, int],
) -> dict[str, Any]:
    """Create the single self-hashed authority consumed by cohort freezing."""

    payload: dict[str, Any] = {
        "schema_version": COHORT_SPLIT_AUTHORITY_SCHEMA,
        "dataset": normalize_dataset_id(dataset, allow_historical=False),
        "split": _text(split, field="split").lower(),
        "source_split_manifest_sha256": _hex64(
            split_sha256, field="source_split_manifest_sha256"
        ),
        "feature_schema_sha256": _hex64(
            feature_schema_sha256, field="feature_schema_sha256"
        ),
        "backbone_order": [normalize_gnn_backbone(value) for value in backbones],
        "checkpoint_sha256s": {
            normalize_gnn_backbone(key): _hex64(value, field=f"{key}.checkpoint")
            for key, value in checkpoint_sha256s.items()
        },
        "temperature_scaling_sha256s": {
            normalize_gnn_backbone(key): _hex64(value, field=f"{key}.temperature")
            for key, value in temperature_scaling_sha256s.items()
        },
        "parent_ids": [_text(value, field="parent_id") for value in parent_ids],
        "parent_graph_sha256s": {
            str(key): _hex64(value, field=f"{key}.parent_graph")
            for key, value in parent_graph_sha256s.items()
        },
        "true_labels": {
            str(key): _native_int(value, field=f"{key}.true_label")
            for key, value in true_labels.items()
        },
    }
    if payload["split"] not in {"calibration", "test"}:
        raise GNNAblationContractError("split authority must be calibration or test")
    payload["split_authority_sha256"] = stable_sha256(payload)
    return payload


def _hex64(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _HEX64.fullmatch(normalized):
        raise GNNAblationContractError(f"{field} must be one lowercase SHA-256")
    return normalized


def _text(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise GNNAblationContractError(f"{field} must be non-empty")
    return normalized


def _native_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GNNAblationContractError(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ProposalCandidateIdentity:
    """One backbone-independent deletion rule mined from training only."""

    dataset: str
    proposal_index: int
    proposal_source_sha256: str
    rule_id: str
    rule_sha256: str
    fragment_graph_sha256: str
    source_split: str = "train"
    action_type: str = "connected_fragment_deletion_rule"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset",
            normalize_dataset_id(self.dataset, allow_historical=False),
        )
        object.__setattr__(
            self,
            "proposal_index",
            _native_int(self.proposal_index, field="proposal_index"),
        )
        object.__setattr__(
            self,
            "proposal_source_sha256",
            _hex64(self.proposal_source_sha256, field="proposal_source_sha256"),
        )
        object.__setattr__(self, "rule_id", _text(self.rule_id, field="rule_id"))
        object.__setattr__(
            self, "rule_sha256", _hex64(self.rule_sha256, field="rule_sha256")
        )
        object.__setattr__(
            self,
            "fragment_graph_sha256",
            _hex64(self.fragment_graph_sha256, field="fragment_graph_sha256"),
        )
        split = _text(self.source_split, field="source_split").lower()
        if split != "train":
            raise GNNAblationContractError(
                "proposal rules must be produced from the train split"
            )
        object.__setattr__(self, "source_split", split)
        action = _text(self.action_type, field="action_type")
        if action != "connected_fragment_deletion_rule":
            raise GNNAblationContractError(
                "GNN ablation proposals must be connected deletion rules"
            )
        object.__setattr__(self, "action_type", action)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProposalCandidateIdentity":
        if not isinstance(value, Mapping):
            raise GNNAblationContractError("proposal record must be a mapping")
        forbidden = sorted(_FORBIDDEN_PROPOSAL_FIELDS.intersection(value))
        if forbidden:
            raise GNNAblationContractError(
                "proposal identity contains classifier/cohort fields: "
                + ", ".join(forbidden)
            )
        allowed = _PROPOSAL_INPUT_FIELDS | _PROPOSAL_DERIVED_FIELDS
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise GNNAblationContractError(
                "proposal record contains unknown fields: " + ", ".join(unknown)
            )
        required = _PROPOSAL_INPUT_FIELDS.difference({"source_split", "action_type"})
        missing = sorted(required.difference(value))
        if missing:
            raise GNNAblationContractError(
                "proposal identity is missing: " + ", ".join(missing)
            )
        candidate = cls(
            dataset=value["dataset"],
            proposal_index=value["proposal_index"],
            proposal_source_sha256=value["proposal_source_sha256"],
            rule_id=value["rule_id"],
            rule_sha256=value["rule_sha256"],
            fragment_graph_sha256=value["fragment_graph_sha256"],
            source_split=value.get("source_split", "train"),
            action_type=value.get(
                "action_type", "connected_fragment_deletion_rule"
            ),
        )
        expected = candidate.to_dict()
        for field in _PROPOSAL_DERIVED_FIELDS.intersection(value):
            if value[field] != expected[field]:
                raise GNNAblationContractError(
                    f"proposal record {field} differs from recomputed identity"
                )
        return candidate

    @property
    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_IDENTITY_SCHEMA,
            "dataset": self.dataset,
            "rule_sha256": self.rule_sha256,
            "fragment_graph_sha256": self.fragment_graph_sha256,
            "source_split": self.source_split,
            "action_type": self.action_type,
        }

    @property
    def candidate_identity_sha256(self) -> str:
        return stable_sha256(self.semantic_payload)

    @property
    def candidate_id(self) -> str:
        return f"gnnrule-{self.candidate_identity_sha256}"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.semantic_payload,
            "proposal_index": self.proposal_index,
            "proposal_source_sha256": self.proposal_source_sha256,
            "rule_id": self.rule_id,
            "candidate_id": self.candidate_id,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "identity_excludes": sorted(_FORBIDDEN_PROPOSAL_FIELDS),
        }
        payload["proposal_record_sha256"] = stable_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ProposalUniverse:
    """The one ordered train-rule universe shared by every backbone and split."""

    dataset: str
    proposal_source_sha256: str
    candidates: tuple[ProposalCandidateIdentity, ...]

    @classmethod
    def freeze(
        cls, candidates: Iterable[ProposalCandidateIdentity]
    ) -> "ProposalUniverse":
        ordered = tuple(sorted(tuple(candidates), key=lambda item: item.proposal_index))
        if not ordered:
            raise GNNAblationContractError("proposal universe must not be empty")
        if tuple(item.proposal_index for item in ordered) != tuple(range(len(ordered))):
            raise GNNAblationContractError(
                "proposal indices must be unique and contiguous from zero"
            )
        datasets = {item.dataset for item in ordered}
        sources = {item.proposal_source_sha256 for item in ordered}
        splits = {item.source_split for item in ordered}
        candidate_ids = [item.candidate_id for item in ordered]
        rule_ids = [item.rule_id for item in ordered]
        if len(datasets) != 1 or len(sources) != 1 or splits != {"train"}:
            raise GNNAblationContractError(
                "proposal universe mixes datasets, sources, or non-train rules"
            )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise GNNAblationContractError(
                "proposal universe contains duplicate structural rules"
            )
        if len(rule_ids) != len(set(rule_ids)):
            raise GNNAblationContractError("proposal universe contains duplicate rule ids")
        return cls(
            dataset=next(iter(datasets)),
            proposal_source_sha256=next(iter(sources)),
            candidates=ordered,
        )

    @property
    def ordered_candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)

    @property
    def ordered_rule_ids(self) -> tuple[str, ...]:
        return tuple(item.rule_id for item in self.candidates)

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROPOSAL_UNIVERSE_SCHEMA,
            "dataset": self.dataset,
            "proposal_mode": "proposal_fixed_train_rule_pool",
            "proposal_source_split": "train",
            "proposal_source_sha256": self.proposal_source_sha256,
            "generation_per_backbone": False,
            "calibration_or_test_parent_in_candidate_identity": False,
            "classifier_outputs_in_candidate_identity": False,
            "candidate_count": len(self.candidates),
            "ordered_candidate_ids": list(self.ordered_candidate_ids),
            "ordered_rule_ids": list(self.ordered_rule_ids),
            "candidate_records_sha256": stable_sha256(
                [item.to_dict() for item in self.candidates]
            ),
        }

    @property
    def identity_sha256(self) -> str:
        return stable_sha256(self.identity_payload)

    def to_manifest(self) -> dict[str, Any]:
        return {
            **self.identity_payload,
            "proposal_universe_sha256": self.identity_sha256,
            "candidate_ids_sha256": stable_sha256(list(self.ordered_candidate_ids)),
            "rule_ids_sha256": stable_sha256(list(self.ordered_rule_ids)),
        }


@dataclass(frozen=True, slots=True)
class ParentPrediction:
    """One frozen pre-intervention source-parent prediction on one split."""

    dataset: str
    split: str
    split_sha256: str
    feature_schema_sha256: str
    temperature_scaling_sha256: str
    backbone: str
    edge_feature_mode: str
    checkpoint_sha256: str
    parent_id: str
    parent_graph_sha256: str
    true_label: int
    predicted_label: int
    source_label: int
    num_classes: int

    def __post_init__(self) -> None:
        dataset = normalize_dataset_id(self.dataset, allow_historical=False)
        object.__setattr__(self, "dataset", dataset)
        split = _text(self.split, field="split").lower()
        if split not in {"calibration", "test"}:
            raise GNNAblationContractError(
                "cohort predictions must come from calibration or test"
            )
        object.__setattr__(self, "split", split)
        for field in (
            "split_sha256",
            "feature_schema_sha256",
            "temperature_scaling_sha256",
            "checkpoint_sha256",
            "parent_graph_sha256",
        ):
            object.__setattr__(self, field, _hex64(getattr(self, field), field=field))
        backbone = normalize_gnn_backbone(self.backbone)
        object.__setattr__(self, "backbone", backbone)
        expected_mode = get_gnn_backbone_spec(backbone).edge_feature_mode
        mode = _text(self.edge_feature_mode, field="edge_feature_mode")
        if mode != expected_mode:
            raise GNNAblationContractError(
                f"{backbone} edge-feature disclosure differs from registry"
            )
        object.__setattr__(self, "edge_feature_mode", mode)
        object.__setattr__(self, "parent_id", _text(self.parent_id, field="parent_id"))
        classes = _native_int(self.num_classes, field="num_classes", minimum=2)
        object.__setattr__(self, "num_classes", classes)
        for field in ("true_label", "predicted_label", "source_label"):
            label = _native_int(getattr(self, field), field=field)
            if label >= classes:
                raise GNNAblationContractError(f"{field} falls outside num_classes")
            object.__setattr__(self, field, label)
        spec = get_dataset_spec(dataset, allow_historical=False)
        if classes != spec.num_classes or self.source_label != spec.source_label:
            raise GNNAblationContractError(
                "prediction class semantics conflict with dataset registry"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ParentPrediction":
        """Load one prediction without accepting ignored or forged fields."""

        if not isinstance(value, Mapping):
            raise GNNAblationContractError("parent prediction must be a mapping")
        input_fields = {
            "dataset",
            "split",
            "split_sha256",
            "feature_schema_sha256",
            "temperature_scaling_sha256",
            "backbone",
            "edge_feature_mode",
            "checkpoint_sha256",
            "parent_id",
            "parent_graph_sha256",
            "true_label",
            "predicted_label",
            "source_label",
            "num_classes",
        }
        derived_fields = {
            "schema_version",
            "native_eligible",
            "prediction_record_sha256",
        }
        unknown = sorted(set(value).difference(input_fields | derived_fields))
        missing = sorted(input_fields.difference(value))
        if unknown:
            raise GNNAblationContractError(
                "parent prediction contains unknown fields: " + ", ".join(unknown)
            )
        if missing:
            raise GNNAblationContractError(
                "parent prediction is missing: " + ", ".join(missing)
            )
        prediction = cls(**{field: value[field] for field in input_fields})
        expected = prediction.to_dict()
        for field in derived_fields.intersection(value):
            if value[field] != expected[field]:
                raise GNNAblationContractError(
                    f"parent prediction {field} differs from recomputed value"
                )
        return prediction

    @property
    def native_eligible(self) -> bool:
        return (
            self.true_label == self.source_label
            and self.predicted_label == self.source_label
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": PARENT_PREDICTION_SCHEMA,
            "dataset": self.dataset,
            "split": self.split,
            "split_sha256": self.split_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "temperature_scaling_sha256": self.temperature_scaling_sha256,
            "backbone": self.backbone,
            "edge_feature_mode": self.edge_feature_mode,
            "checkpoint_sha256": self.checkpoint_sha256,
            "parent_id": self.parent_id,
            "parent_graph_sha256": self.parent_graph_sha256,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "source_label": self.source_label,
            "num_classes": self.num_classes,
            "native_eligible": self.native_eligible,
        }
        payload["prediction_record_sha256"] = stable_sha256(payload)
        return payload


def _cohort_manifest(
    *,
    universe: ProposalUniverse,
    split: str,
    split_sha256: str,
    feature_schema_sha256: str,
    kind: str,
    backbone: str | None,
    parent_ids: Sequence[str],
    checkpoint_sha256s: Mapping[str, str],
    temperature_sha256s: Mapping[str, str],
    edge_feature_modes: Mapping[str, str],
) -> dict[str, Any]:
    candidate_ids = universe.ordered_candidate_ids
    payload: dict[str, Any] = {
        "schema_version": COHORT_SCHEMA,
        "dataset": universe.dataset,
        "split": split,
        "split_sha256": split_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "cohort_kind": kind,
        "backbone": backbone,
        "proposal_mode": "proposal_fixed_train_rule_pool",
        "proposal_source_split": "train",
        "proposal_universe_sha256": universe.identity_sha256,
        "parent_ids": list(parent_ids),
        "parent_count": len(parent_ids),
        "parent_ids_sha256": stable_sha256(list(parent_ids)),
        "candidate_ids": list(candidate_ids),
        "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": stable_sha256(list(candidate_ids)),
        "expected_application_count": len(parent_ids) * len(candidate_ids),
        "checkpoint_sha256s": dict(sorted(checkpoint_sha256s.items())),
        "temperature_scaling_sha256s": dict(sorted(temperature_sha256s.items())),
        "edge_feature_modes": dict(sorted(edge_feature_modes.items())),
        "candidate_regeneration_per_backbone": False,
        "cohort_parent_identity_in_proposal": False,
    }
    payload["cohort_identity_sha256"] = stable_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class CohortFreeze:
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise GNNAblationContractError("cohort freeze must be one mapping")
        normalized = json.loads(canonical_json_bytes(self.payload).decode("utf-8"))
        expected_fields = {
            "schema_version",
            "dataset",
            "split",
            "split_sha256",
            "feature_schema_sha256",
            "source_label",
            "num_classes",
            "backbone_order",
            "proposal_source_split",
            "proposal_universe_sha256",
            "proposal_candidate_regeneration_per_backbone",
            "prediction_records_sha256",
            "checkpoint_sha256s",
            "temperature_scaling_sha256s",
            "edge_feature_modes",
            "split_authority_sha256",
            "authoritative_parent_ids_sha256",
            "authoritative_parent_graphs_sha256",
            "authoritative_true_labels_sha256",
            "true_labels_consistent_across_backbones",
            "parent_graphs_consistent_across_backbones",
            "authoritative_split_manifest_bound",
            "authoritative_checkpoint_manifests_bound",
            "authoritative_temperature_manifests_bound",
            "common_cohort",
            "native_cohorts",
            "common_cohort_primary",
            "native_cohorts_secondary",
            "cohort_freeze_sha256",
        }
        if set(normalized) != expected_fields:
            raise GNNAblationContractError(
                "cohort freeze fields differ from the closed schema"
            )
        if normalized.get("schema_version") != COHORT_FREEZE_SCHEMA:
            raise GNNAblationContractError("cohort freeze schema changed")
        dataset = normalize_dataset_id(
            normalized.get("dataset"), allow_historical=False
        )
        if dataset != normalized["dataset"]:
            raise GNNAblationContractError("cohort freeze dataset is not canonical")
        if normalized.get("split") not in {"calibration", "test"}:
            raise GNNAblationContractError("cohort freeze split is invalid")
        for field in (
            "split_sha256",
            "feature_schema_sha256",
            "proposal_universe_sha256",
            "split_authority_sha256",
            "authoritative_parent_ids_sha256",
            "authoritative_parent_graphs_sha256",
            "authoritative_true_labels_sha256",
        ):
            _hex64(normalized.get(field), field=f"cohort_freeze.{field}")
        spec = get_dataset_spec(dataset, allow_historical=False)
        if (
            normalized.get("source_label") != spec.source_label
            or normalized.get("num_classes") != spec.num_classes
        ):
            raise GNNAblationContractError(
                "cohort freeze class semantics differ from dataset registry"
            )
        claimed = normalized["cohort_freeze_sha256"]
        freeze_body = dict(normalized)
        freeze_body.pop("cohort_freeze_sha256")
        if claimed != stable_sha256(freeze_body):
            raise GNNAblationContractError("cohort freeze SHA differs from payload")
        common = normalized.get("common_cohort")
        native = normalized.get("native_cohorts")
        if not isinstance(common, Mapping) or not isinstance(native, Mapping):
            raise GNNAblationContractError("cohort freeze manifests are missing")
        backbone_order = normalized.get("backbone_order")
        if type(backbone_order) is not list or set(native) != set(backbone_order):
            raise GNNAblationContractError(
                "native cohort keys differ from the frozen backbone order"
            )
        if len(backbone_order) != len(set(backbone_order)):
            raise GNNAblationContractError("frozen backbone order is duplicated")
        if [normalize_gnn_backbone(value) for value in backbone_order] != backbone_order:
            raise GNNAblationContractError("frozen backbone order is not canonical")
        for field in (
            "prediction_records_sha256",
            "checkpoint_sha256s",
            "temperature_scaling_sha256s",
            "edge_feature_modes",
        ):
            value = normalized.get(field)
            if not isinstance(value, Mapping) or set(value) != set(backbone_order):
                raise GNNAblationContractError(
                    f"cohort freeze {field} differs from backbone authority"
                )
        for backbone in backbone_order:
            _hex64(
                normalized["prediction_records_sha256"][backbone],
                field=f"{backbone}.prediction_records_sha256",
            )
            _hex64(
                normalized["checkpoint_sha256s"][backbone],
                field=f"{backbone}.checkpoint_sha256",
            )
            _hex64(
                normalized["temperature_scaling_sha256s"][backbone],
                field=f"{backbone}.temperature_scaling_sha256",
            )
            if (
                normalized["edge_feature_modes"][backbone]
                != get_gnn_backbone_spec(backbone).edge_feature_mode
            ):
                raise GNNAblationContractError(
                    f"{backbone} edge-feature mode differs from registry"
                )
        cohort_fields = {
            "schema_version",
            "dataset",
            "split",
            "split_sha256",
            "feature_schema_sha256",
            "cohort_kind",
            "backbone",
            "proposal_mode",
            "proposal_source_split",
            "proposal_universe_sha256",
            "parent_ids",
            "parent_count",
            "parent_ids_sha256",
            "candidate_ids",
            "candidate_count",
            "candidate_ids_sha256",
            "expected_application_count",
            "checkpoint_sha256s",
            "temperature_scaling_sha256s",
            "edge_feature_modes",
            "candidate_regeneration_per_backbone",
            "cohort_parent_identity_in_proposal",
            "cohort_identity_sha256",
        }
        for manifest in (common, *native.values()):
            if not isinstance(manifest, Mapping):
                raise GNNAblationContractError("cohort manifest must be one mapping")
            if set(manifest) != cohort_fields:
                raise GNNAblationContractError(
                    "cohort manifest fields differ from the closed schema"
                )
            body = dict(manifest)
            identity = body.pop("cohort_identity_sha256", None)
            if identity != stable_sha256(body):
                raise GNNAblationContractError("cohort identity SHA differs from payload")
            for field in (
                "dataset",
                "split",
                "split_sha256",
                "feature_schema_sha256",
                "proposal_universe_sha256",
            ):
                if manifest[field] != normalized[field]:
                    raise GNNAblationContractError(
                        f"cohort manifest {field} differs from its freeze"
                    )
            parent_ids = manifest.get("parent_ids")
            candidate_ids = manifest.get("candidate_ids")
            if type(parent_ids) is not list or type(candidate_ids) is not list:
                raise GNNAblationContractError(
                    "cohort parent/candidate identities must be ordered lists"
                )
            if (
                len(parent_ids) != len(set(parent_ids))
                or len(candidate_ids) != len(set(candidate_ids))
                or manifest.get("parent_count") != len(parent_ids)
                or manifest.get("candidate_count") != len(candidate_ids)
                or manifest.get("parent_ids_sha256") != stable_sha256(parent_ids)
                or manifest.get("candidate_ids_sha256")
                != stable_sha256(candidate_ids)
                or manifest.get("expected_application_count")
                != len(parent_ids) * len(candidate_ids)
            ):
                raise GNNAblationContractError(
                    "cohort ordered identity/count contract is malformed"
                )
            if (
                manifest.get("proposal_mode")
                != "proposal_fixed_train_rule_pool"
                or manifest.get("proposal_source_split") != "train"
                or manifest.get("candidate_regeneration_per_backbone") is not False
                or manifest.get("cohort_parent_identity_in_proposal") is not False
            ):
                raise GNNAblationContractError(
                    "cohort proposal/source separation contract changed"
                )
        if common.get("cohort_kind") != "common" or common.get("backbone") is not None:
            raise GNNAblationContractError("common cohort identity is malformed")
        for backbone, manifest in native.items():
            if manifest.get("cohort_kind") != "native" or manifest.get("backbone") != backbone:
                raise GNNAblationContractError("native cohort identity is malformed")
            for field, top_field in (
                ("checkpoint_sha256s", "checkpoint_sha256s"),
                ("temperature_scaling_sha256s", "temperature_scaling_sha256s"),
                ("edge_feature_modes", "edge_feature_modes"),
            ):
                if manifest[field] != {backbone: normalized[top_field][backbone]}:
                    raise GNNAblationContractError(
                        f"native cohort {field} differs from frozen backbone authority"
                    )
        if (
            common["checkpoint_sha256s"] != normalized["checkpoint_sha256s"]
            or common["temperature_scaling_sha256s"]
            != normalized["temperature_scaling_sha256s"]
            or common["edge_feature_modes"] != normalized["edge_feature_modes"]
            or any(
                common["candidate_ids"] != manifest["candidate_ids"]
                for manifest in native.values()
            )
        ):
            raise GNNAblationContractError(
                "common cohort differs from frozen model/proposal authority"
            )
        native_intersection = set(common["parent_ids"])
        if native:
            native_intersection = set.intersection(
                *(set(manifest["parent_ids"]) for manifest in native.values())
            )
        if set(common["parent_ids"]) != native_intersection:
            raise GNNAblationContractError(
                "common cohort is not the intersection of native cohorts"
            )
        if (
            normalized["proposal_source_split"] != "train"
            or normalized["proposal_candidate_regeneration_per_backbone"] is not False
            or normalized["true_labels_consistent_across_backbones"] is not True
            or normalized["parent_graphs_consistent_across_backbones"] is not True
            or normalized["authoritative_split_manifest_bound"] is not True
            or normalized["authoritative_checkpoint_manifests_bound"] is not True
            or normalized["authoritative_temperature_manifests_bound"] is not True
            or normalized["common_cohort_primary"] is not True
            or normalized["native_cohorts_secondary"] is not True
        ):
            raise GNNAblationContractError("cohort freeze authority flags changed")
        object.__setattr__(self, "payload", normalized)

    @property
    def common(self) -> Mapping[str, Any]:
        return self.payload["common_cohort"]

    @property
    def native(self) -> Mapping[str, Mapping[str, Any]]:
        return self.payload["native_cohorts"]

    @property
    def identity_sha256(self) -> str:
        return str(self.payload["cohort_freeze_sha256"])

    def to_manifest(self) -> dict[str, Any]:
        return dict(self.payload)


def freeze_common_and_native_cohorts(
    *,
    universe: ProposalUniverse,
    predictions: Iterable[ParentPrediction],
    backbones: Sequence[str],
    split: str,
    expected_parent_ids: Sequence[str],
    expected_dataset: str,
    expected_split_sha256: str,
    expected_feature_schema_sha256: str,
    expected_checkpoint_sha256s: Mapping[str, str],
    expected_temperature_scaling_sha256s: Mapping[str, str],
    expected_parent_graph_sha256s: Mapping[str, str],
    expected_true_labels: Mapping[str, int],
    split_authority: Mapping[str, Any],
    minimum_common_parents: int = 1,
) -> CohortFreeze:
    """Freeze one split's native/common source cohorts over one train-rule pool."""

    ordered_backbones = tuple(normalize_gnn_backbone(value) for value in backbones)
    if not ordered_backbones or len(ordered_backbones) != len(set(ordered_backbones)):
        raise GNNAblationContractError("backbone order is empty or duplicated")
    normalized_split = _text(split, field="split").lower()
    if normalized_split not in {"calibration", "test"}:
        raise GNNAblationContractError("cohort split must be calibration or test")
    normalized_dataset = normalize_dataset_id(
        expected_dataset, allow_historical=False
    )
    if normalized_dataset != universe.dataset:
        raise GNNAblationContractError(
            "cohort dataset differs from train-rule universe"
        )
    minimum = _native_int(
        minimum_common_parents, field="minimum_common_parents", minimum=1
    )
    parent_order = tuple(
        _text(value, field="expected_parent_id") for value in expected_parent_ids
    )
    if not parent_order or len(parent_order) != len(set(parent_order)):
        raise GNNAblationContractError("expected parent order is empty or duplicated")
    expected_parents = set(parent_order)
    expected_split_sha = _hex64(
        expected_split_sha256, field="expected_split_sha256"
    )
    expected_feature_sha = _hex64(
        expected_feature_schema_sha256,
        field="expected_feature_schema_sha256",
    )
    expected_checkpoints = {
        normalize_gnn_backbone(backbone): _hex64(value, field=f"{backbone}.checkpoint")
        for backbone, value in expected_checkpoint_sha256s.items()
    }
    expected_temperatures = {
        normalize_gnn_backbone(backbone): _hex64(value, field=f"{backbone}.temperature")
        for backbone, value in expected_temperature_scaling_sha256s.items()
    }
    if set(expected_checkpoints) != set(ordered_backbones):
        raise GNNAblationContractError("expected checkpoint map differs from backbones")
    if set(expected_temperatures) != set(ordered_backbones):
        raise GNNAblationContractError("expected temperature map differs from backbones")
    if set(expected_parent_graph_sha256s) != expected_parents:
        raise GNNAblationContractError("expected parent graph map differs from cohort")
    authoritative_graphs = {
        parent: _hex64(value, field=f"{parent}.parent_graph")
        for parent, value in expected_parent_graph_sha256s.items()
    }
    if set(expected_true_labels) != expected_parents:
        raise GNNAblationContractError("expected true-label map differs from cohort")
    authoritative_labels = {
        parent: _native_int(value, field=f"{parent}.true_label")
        for parent, value in expected_true_labels.items()
    }
    expected_authority = build_cohort_split_authority(
        dataset=normalized_dataset,
        split=normalized_split,
        split_sha256=expected_split_sha,
        feature_schema_sha256=expected_feature_sha,
        backbones=ordered_backbones,
        checkpoint_sha256s=expected_checkpoints,
        temperature_scaling_sha256s=expected_temperatures,
        parent_ids=parent_order,
        parent_graph_sha256s=authoritative_graphs,
        true_labels=authoritative_labels,
    )
    if not isinstance(split_authority, Mapping) or dict(split_authority) != expected_authority:
        raise GNNAblationContractError(
            "cohort inputs differ from the self-hashed split authority"
        )
    grouped: dict[str, dict[str, ParentPrediction]] = {
        backbone: {} for backbone in ordered_backbones
    }
    for prediction in predictions:
        if prediction.dataset != normalized_dataset or prediction.split != normalized_split:
            raise GNNAblationContractError(
                "prediction dataset/split differs from cohort contract"
            )
        if prediction.feature_schema_sha256 != expected_feature_sha:
            raise GNNAblationContractError(
                "prediction feature schema differs from shared schema"
            )
        if prediction.split_sha256 != expected_split_sha:
            raise GNNAblationContractError(
                "prediction split SHA differs from authoritative split"
            )
        if prediction.backbone not in grouped:
            raise GNNAblationContractError(
                f"prediction uses unconfigured backbone: {prediction.backbone}"
            )
        if prediction.parent_id not in expected_parents:
            raise GNNAblationContractError(
                f"prediction escapes expected source cohort: {prediction.parent_id}"
            )
        if prediction.checkpoint_sha256 != expected_checkpoints[prediction.backbone]:
            raise GNNAblationContractError(
                f"{prediction.backbone} checkpoint differs from frozen authority"
            )
        if (
            prediction.temperature_scaling_sha256
            != expected_temperatures[prediction.backbone]
        ):
            raise GNNAblationContractError(
                f"{prediction.backbone} temperature differs from frozen authority"
            )
        if prediction.parent_graph_sha256 != authoritative_graphs[prediction.parent_id]:
            raise GNNAblationContractError(
                f"parent graph differs from split manifest for {prediction.parent_id}"
            )
        if prediction.true_label != authoritative_labels[prediction.parent_id]:
            raise GNNAblationContractError(
                f"true label differs from split manifest for {prediction.parent_id}"
            )
        bucket = grouped[prediction.backbone]
        if prediction.parent_id in bucket:
            raise GNNAblationContractError(
                f"duplicate parent prediction for {prediction.backbone}/{prediction.parent_id}"
            )
        bucket[prediction.parent_id] = prediction

    checkpoint_sha256s: dict[str, str] = {}
    temperature_sha256s: dict[str, str] = {}
    edge_feature_modes: dict[str, str] = {}
    native_parent_ids: dict[str, tuple[str, ...]] = {}
    prediction_sha256s: dict[str, str] = {}
    split_hashes: set[str] = set()
    source_labels: set[int] = set()
    class_counts: set[int] = set()
    labels_by_parent: dict[str, set[int]] = {parent: set() for parent in parent_order}
    graph_hashes_by_parent: dict[str, set[str]] = {
        parent: set() for parent in parent_order
    }
    for backbone in ordered_backbones:
        bucket = grouped[backbone]
        if set(bucket) != expected_parents:
            missing = sorted(expected_parents.difference(bucket))
            raise GNNAblationContractError(
                f"{backbone} prediction coverage differs from source cohort: {missing}"
            )
        ordered = tuple(bucket[parent_id] for parent_id in parent_order)
        for field, target in (
            ("checkpoint_sha256", checkpoint_sha256s),
            ("temperature_scaling_sha256", temperature_sha256s),
            ("edge_feature_mode", edge_feature_modes),
        ):
            values = {getattr(item, field) for item in ordered}
            if len(values) != 1:
                raise GNNAblationContractError(f"{backbone} records mix {field}")
            target[backbone] = next(iter(values))
        split_hashes.update(item.split_sha256 for item in ordered)
        source_labels.update(item.source_label for item in ordered)
        class_counts.update(item.num_classes for item in ordered)
        for item in ordered:
            labels_by_parent[item.parent_id].add(item.true_label)
            graph_hashes_by_parent[item.parent_id].add(item.parent_graph_sha256)
        native_parent_ids[backbone] = tuple(
            item.parent_id for item in ordered if item.native_eligible
        )
        prediction_sha256s[backbone] = stable_sha256(
            [item.to_dict() for item in ordered]
        )
    if len(split_hashes) != 1:
        raise GNNAblationContractError("backbone records disagree on split SHA")
    if split_hashes != {expected_split_sha}:
        raise GNNAblationContractError("cohort split SHA differs from authority")
    if len(source_labels) != 1 or len(class_counts) != 1:
        raise GNNAblationContractError(
            "backbone records disagree on class semantics"
        )
    source_label = next(iter(source_labels))
    num_classes = next(iter(class_counts))
    spec = get_dataset_spec(universe.dataset, allow_historical=False)
    if source_label != spec.source_label or num_classes != spec.num_classes:
        raise GNNAblationContractError(
            "cohort records conflict with active dataset registry"
        )
    if checkpoint_sha256s != expected_checkpoints:
        raise GNNAblationContractError("checkpoint map differs from frozen authority")
    if temperature_sha256s != expected_temperatures:
        raise GNNAblationContractError("temperature map differs from frozen authority")
    for parent_id in parent_order:
        if len(labels_by_parent[parent_id]) != 1:
            raise GNNAblationContractError(
                f"true label differs across backbones for parent {parent_id}"
            )
        if labels_by_parent[parent_id] != {source_label}:
            raise GNNAblationContractError(
                f"source cohort contains non-source parent {parent_id}"
            )
        if len(graph_hashes_by_parent[parent_id]) != 1:
            raise GNNAblationContractError(
                f"parent graph identity differs across backbones for {parent_id}"
            )

    common_set = set(parent_order)
    for values in native_parent_ids.values():
        common_set.intersection_update(values)
    common_parent_ids = tuple(
        parent_id for parent_id in parent_order if parent_id in common_set
    )
    if len(common_parent_ids) < minimum:
        raise GNNAblationContractError(
            "common cohort is below the configured minimum: "
            f"observed={len(common_parent_ids)}, required={minimum}"
        )
    split_sha256 = next(iter(split_hashes))
    common = _cohort_manifest(
        universe=universe,
        split=normalized_split,
        split_sha256=split_sha256,
        feature_schema_sha256=expected_feature_sha,
        kind="common",
        backbone=None,
        parent_ids=common_parent_ids,
        checkpoint_sha256s=checkpoint_sha256s,
        temperature_sha256s=temperature_sha256s,
        edge_feature_modes=edge_feature_modes,
    )
    native = {
        backbone: _cohort_manifest(
            universe=universe,
            split=normalized_split,
            split_sha256=split_sha256,
            feature_schema_sha256=expected_feature_sha,
            kind="native",
            backbone=backbone,
            parent_ids=native_parent_ids[backbone],
            checkpoint_sha256s={backbone: checkpoint_sha256s[backbone]},
            temperature_sha256s={backbone: temperature_sha256s[backbone]},
            edge_feature_modes={backbone: edge_feature_modes[backbone]},
        )
        for backbone in ordered_backbones
    }
    payload: dict[str, Any] = {
        "schema_version": COHORT_FREEZE_SCHEMA,
        "dataset": universe.dataset,
        "split": normalized_split,
        "split_sha256": split_sha256,
        "feature_schema_sha256": expected_feature_sha,
        "source_label": source_label,
        "num_classes": num_classes,
        "backbone_order": list(ordered_backbones),
        "proposal_source_split": "train",
        "proposal_universe_sha256": universe.identity_sha256,
        "proposal_candidate_regeneration_per_backbone": False,
        "prediction_records_sha256": prediction_sha256s,
        "checkpoint_sha256s": checkpoint_sha256s,
        "temperature_scaling_sha256s": temperature_sha256s,
        "edge_feature_modes": edge_feature_modes,
        "split_authority_sha256": expected_authority["split_authority_sha256"],
        "authoritative_parent_ids_sha256": stable_sha256(list(parent_order)),
        "authoritative_parent_graphs_sha256": stable_sha256(
            {parent: authoritative_graphs[parent] for parent in parent_order}
        ),
        "authoritative_true_labels_sha256": stable_sha256(
            {parent: authoritative_labels[parent] for parent in parent_order}
        ),
        "true_labels_consistent_across_backbones": True,
        "parent_graphs_consistent_across_backbones": True,
        "authoritative_split_manifest_bound": True,
        "authoritative_checkpoint_manifests_bound": True,
        "authoritative_temperature_manifests_bound": True,
        "common_cohort": common,
        "native_cohorts": native,
        "common_cohort_primary": True,
        "native_cohorts_secondary": True,
    }
    payload["cohort_freeze_sha256"] = stable_sha256(payload)
    return CohortFreeze(payload=payload)


__all__ = [
    "CANDIDATE_IDENTITY_SCHEMA",
    "COHORT_FREEZE_SCHEMA",
    "COHORT_SPLIT_AUTHORITY_SCHEMA",
    "COHORT_SCHEMA",
    "PARENT_PREDICTION_SCHEMA",
    "PROPOSAL_UNIVERSE_SCHEMA",
    "CohortFreeze",
    "GNNAblationContractError",
    "ParentPrediction",
    "ProposalCandidateIdentity",
    "ProposalUniverse",
    "canonical_json_bytes",
    "build_cohort_split_authority",
    "freeze_common_and_native_cohorts",
    "stable_sha256",
]
