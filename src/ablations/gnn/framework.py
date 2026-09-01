"""Config-only, non-executing planner for molecular GNN ablations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.ablations.gnn.contracts import (
    CANDIDATE_IDENTITY_SCHEMA,
    COHORT_FREEZE_SCHEMA,
    COHORT_SPLIT_AUTHORITY_SCHEMA,
    GNNAblationContractError,
    PARENT_PREDICTION_SCHEMA,
    PROPOSAL_UNIVERSE_SCHEMA,
    stable_sha256,
)
from src.ablations.output_schema import output_inventory
from src.data.dataset_registry import get_dataset_spec, normalize_dataset_id
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.models.gnn_backbone_registry import (
    available_gnn_backbones,
    get_gnn_backbone_spec,
    normalize_gnn_backbone,
    required_backbone_bundle_files,
)
from src.utils.env import load_yaml_config


CONFIG_SCHEMA = "gnn_backbone_ablation_config_v2"
PLAN_SCHEMA = "gnn_backbone_ablation_plan_v3"
TASK_SCHEMA = "gnn_backbone_ablation_task_v2"
OUTPUT_CONTRACT_SCHEMA = "gnn_backbone_ablation_output_contract_v2"
FINAL_MANIFEST_SCHEMA = "gnn_backbone_ablation_final_manifest_v2"

REQUIRED_BACKBONES = ("gine", "gin", "gcn", "gatv2")
TEST_SPLIT_NAMES = frozenset({"test"})


class GNNAblationConfigError(GNNAblationContractError):
    """The checked-in ablation configuration is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GNNAblationConfigError(f"{field} must be a mapping")
    return dict(value)


def _exact_keys(
    value: Mapping[str, Any], *, field: str, expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unknown = sorted(actual.difference(expected))
        raise GNNAblationConfigError(
            f"{field} fields differ from the closed schema: "
            f"missing={missing}, unknown={unknown}"
        )


def _text(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise GNNAblationConfigError(f"{field} must be non-empty")
    return normalized


def _bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise GNNAblationConfigError(f"{field} must be boolean")
    return value


def _int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GNNAblationConfigError(f"{field} must be an int >= {minimum}")
    return value


def _hex64(value: Any, *, field: str) -> str:
    normalized = _text(value, field=field).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise GNNAblationConfigError(f"{field} must be one SHA-256")
    return normalized


def _project_file(project_root: Path, raw: Any, *, field: str) -> Path:
    value = Path(_text(raw, field=field)).expanduser()
    if value.is_absolute():
        raise GNNAblationConfigError(f"{field} must be repository-relative")
    path = (project_root / value).resolve(strict=True)
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise GNNAblationConfigError(f"{field} escapes project_root") from exc
    if not path.is_file():
        raise GNNAblationConfigError(f"{field} is not a file: {path}")
    return path


@dataclass(frozen=True, slots=True)
class BackbonePlanConfig:
    name: str
    role: str
    checkpoint_policy: str
    model_config: str
    model_config_sha256: str
    comparison_config_sha256: str
    edge_feature_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "checkpoint_policy": self.checkpoint_policy,
            "model_config": self.model_config,
            "model_config_sha256": self.model_config_sha256,
            "comparison_config_sha256": self.comparison_config_sha256,
            "edge_feature_mode": self.edge_feature_mode,
        }


@dataclass(frozen=True, slots=True)
class GNNAblationConfig:
    source_path: str
    source_sha256: str
    project_root: str
    experiment_id: str
    dataset: str
    num_classes: int
    source_label: int
    dataset_config: str
    dataset_config_sha256: str
    shared_feature_schema_provider: str
    shared_feature_schema_sha256: str
    primary_results_gate: str
    proposal_mode: str
    fit_split: str
    selection_split: str
    proposal_split: str
    calibration_split: str
    evaluation_split: str
    minimum_common_parents: int
    output_root_template: str
    evaluation_policy: Mapping[str, Any]
    policy: Mapping[str, Any]
    data_policy: Mapping[str, Any]
    backbones: tuple[BackbonePlanConfig, ...]

    @property
    def backbone_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.backbones)

    def scientific_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "num_classes": self.num_classes,
            "source_label": self.source_label,
            "dataset_config": self.dataset_config,
            "dataset_config_sha256": self.dataset_config_sha256,
            "shared_feature_schema_provider": self.shared_feature_schema_provider,
            "shared_feature_schema_sha256": self.shared_feature_schema_sha256,
            "primary_results_gate": self.primary_results_gate,
            "proposal_mode": self.proposal_mode,
            "splits": {
                "fit": self.fit_split,
                "selection": self.selection_split,
                "proposal": self.proposal_split,
                "calibration": self.calibration_split,
                "evaluation": self.evaluation_split,
            },
            "minimum_common_parents": self.minimum_common_parents,
            "output_root_template": self.output_root_template,
            "evaluation_policy": dict(self.evaluation_policy),
            "policy": dict(self.policy),
            "data_policy": dict(self.data_policy),
            "backbones": [item.to_dict() for item in self.backbones],
        }

    @property
    def scientific_sha256(self) -> str:
        return stable_sha256(self.scientific_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scientific_payload(),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "project_root": self.project_root,
            "config_scientific_sha256": self.scientific_sha256,
        }


@dataclass(frozen=True, slots=True)
class AblationTask:
    task_id: str
    stage: str
    component: str
    depends_on: tuple[str, ...]
    split_access: tuple[str, ...]
    outputs: tuple[str, ...]
    backbone: str | None = None
    cohort_kind: str | None = None
    checkpoint_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TASK_SCHEMA,
            "task_id": self.task_id,
            "stage": self.stage,
            "component": self.component,
            "depends_on": list(self.depends_on),
            "split_access": list(self.split_access),
            "outputs": list(self.outputs),
            "backbone": self.backbone,
            "cohort_kind": self.cohort_kind,
            "checkpoint_policy": self.checkpoint_policy,
            "execution_mode": "CONFIG_ONLY_CONTRACT",
            "launches_science": False,
            "writes_autodl": False,
            "writes_main_matrix": False,
        }


def _parse_model_config(*, name: str, path: Path) -> tuple[str, str]:
    payload = _mapping(load_yaml_config(path), field=f"model_config.{name}")
    _exact_keys(
        payload,
        field=f"model_config.{name}",
        expected={"gnn", "training", "calibration"},
    )
    gnn = _mapping(payload.get("gnn"), field=f"model_config.{name}.gnn")
    _exact_keys(
        gnn,
        field=f"model_config.{name}.gnn",
        expected={
            "backbone",
            "num_layers",
            "hidden_dim",
            "dropout",
            "pooling",
            "readout_layers",
            "normalization",
            "residual",
        },
    )
    configured = normalize_gnn_backbone(
        _text(gnn.get("backbone"), field=f"model_config.{name}.gnn.backbone")
    )
    if configured != name:
        raise GNNAblationConfigError(f"{name} model config selects {configured}")
    calibration = _mapping(
        payload.get("calibration"), field=f"model_config.{name}.calibration"
    )
    _exact_keys(
        calibration,
        field=f"model_config.{name}.calibration",
        expected={"method", "split", "max_iter"},
    )
    if (
        calibration.get("method") != "temperature_scaling"
        or calibration.get("split") != "validation"
    ):
        raise GNNAblationConfigError(
            f"{name} calibration must be validation-only temperature scaling"
        )
    training = _mapping(
        payload.get("training"), field=f"model_config.{name}.training"
    )
    _exact_keys(
        training,
        field=f"model_config.{name}.training",
        expected={
            "optimizer",
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "early_stopping_patience",
            "batch_size",
            "primary_seed",
            "selection_metric",
            "class_weighted_loss",
            "weighted_sampler",
            "gradient_clip_norm",
        },
    )
    comparable = {
        "gnn": {key: value for key, value in gnn.items() if key != "backbone"},
        "training": training,
        "calibration": calibration,
    }
    return _sha256_file(path), stable_sha256(comparable)


def load_ablation_config(
    path_like: str | Path,
    *,
    project_root: str | Path | None = None,
) -> GNNAblationConfig:
    """Load and close one dataset's no-execution ablation specification."""

    path = Path(path_like).expanduser().resolve(strict=True)
    root = (
        Path(project_root).expanduser().resolve(strict=True)
        if project_root is not None
        else path.parents[3]
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GNNAblationConfigError("config path escapes project_root") from exc
    raw = _mapping(load_yaml_config(path), field="root")
    _exact_keys(
        raw,
        field="root",
        expected={
            "schema_version",
            "experiment",
            "dataset",
            "feature_schema",
            "splits",
            "cohorts",
            "evaluation",
            "backbones",
            "outputs",
            "policy",
            "data_policy",
        },
    )
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise GNNAblationConfigError("GNN ablation config schema changed")

    experiment = _mapping(raw.get("experiment"), field="experiment")
    dataset_raw = _mapping(raw.get("dataset"), field="dataset")
    feature_raw = _mapping(raw.get("feature_schema"), field="feature_schema")
    splits = _mapping(raw.get("splits"), field="splits")
    cohorts = _mapping(raw.get("cohorts"), field="cohorts")
    evaluation_policy = _mapping(raw.get("evaluation"), field="evaluation")
    outputs = _mapping(raw.get("outputs"), field="outputs")
    policy = _mapping(raw.get("policy"), field="policy")
    data_policy = _mapping(raw.get("data_policy"), field="data_policy")
    backbone_values = _mapping(raw.get("backbones"), field="backbones")

    _exact_keys(
        experiment,
        field="experiment",
        expected={
            "id",
            "mode",
            "primary_results_gate",
            "proposal_mode",
            "generation_per_backbone",
        },
    )
    _exact_keys(
        dataset_raw,
        field="dataset",
        expected={"id", "num_classes", "source_label", "config"},
    )
    _exact_keys(
        feature_raw,
        field="feature_schema",
        expected={"provider", "sha256"},
    )
    _exact_keys(
        splits,
        field="splits",
        expected={
            "fit",
            "selection",
            "proposal",
            "calibration",
            "evaluation",
            "test_open_after_selector_freeze",
        },
    )
    _exact_keys(
        cohorts,
        field="cohorts",
        expected={"common", "native", "minimum_common_parents"},
    )
    _exact_keys(
        evaluation_policy,
        field="evaluation",
        expected={
            "distance_metric",
            "wnode_config_source",
            "wnode_shared_across_backbones",
            "selector_fit_split",
            "held_out_evaluation_split",
        },
    )
    _exact_keys(outputs, field="outputs", expected={"root_template"})
    _exact_keys(
        policy,
        field="policy",
        expected={
            "matrix_write_allowed",
            "science_launch_allowed_by_framework",
            "require_fresh_attempt",
        },
    )
    _exact_keys(
        data_policy,
        field="data_policy",
        expected={
            "mode",
            "policy_receipt_required",
            "upstream_terms_status",
            "redistribution_allowed",
            "graph_cache_mode",
        },
    )
    if tuple(backbone_values) != REQUIRED_BACKBONES:
        raise GNNAblationConfigError(
            "backbones must appear exactly as gine, gin, gcn, gatv2"
        )

    experiment_id = _text(experiment.get("id"), field="experiment.id")
    primary_gate = _text(
        experiment.get("primary_results_gate"), field="experiment.primary_results_gate"
    )
    if primary_gate != "fast16_matrix_16_of_16":
        raise GNNAblationConfigError("ablation requires the fast16 16/16 gate")
    proposal_mode = _text(
        experiment.get("proposal_mode"), field="experiment.proposal_mode"
    )
    if proposal_mode != "proposal_fixed_train_rule_pool":
        raise GNNAblationConfigError(
            "only proposal_fixed_train_rule_pool is supported"
        )
    if _bool(
        experiment.get("generation_per_backbone"),
        field="experiment.generation_per_backbone",
    ):
        raise GNNAblationConfigError("proposal generation may not vary by backbone")
    if _text(experiment.get("mode"), field="experiment.mode") != "plan_only":
        raise GNNAblationConfigError("framework must remain plan_only")

    dataset = normalize_dataset_id(
        _text(dataset_raw.get("id"), field="dataset.id"), allow_historical=False
    )
    spec = get_dataset_spec(dataset, allow_historical=False)
    num_classes = _int(
        dataset_raw.get("num_classes"), field="dataset.num_classes", minimum=2
    )
    source_label = _int(dataset_raw.get("source_label"), field="dataset.source_label")
    if num_classes != spec.num_classes or source_label != spec.source_label:
        raise GNNAblationConfigError("dataset class contract conflicts with registry")
    dataset_config_path = _project_file(
        root, dataset_raw.get("config"), field="dataset.config"
    )

    provider = _text(feature_raw.get("provider"), field="feature_schema.provider")
    if provider != "default_molecular_feature_schema":
        raise GNNAblationConfigError("unsupported feature-schema provider")
    configured_feature_sha = _hex64(
        feature_raw.get("sha256"), field="feature_schema.sha256"
    )
    live_feature_sha = default_molecular_feature_schema().to_dict()["schema_sha256"]
    if configured_feature_sha != live_feature_sha:
        raise GNNAblationConfigError(
            "configured shared feature schema differs from implementation"
        )

    fit_split = _text(splits.get("fit"), field="splits.fit")
    selection_split = _text(splits.get("selection"), field="splits.selection")
    proposal_split = _text(splits.get("proposal"), field="splits.proposal")
    calibration_split = _text(splits.get("calibration"), field="splits.calibration")
    evaluation_split = _text(splits.get("evaluation"), field="splits.evaluation")
    if (fit_split, selection_split, proposal_split) != (
        "train",
        "validation",
        "train",
    ):
        raise GNNAblationConfigError(
            "fit/selection/proposal must remain train/validation/train"
        )
    if (calibration_split, evaluation_split) != ("calibration", "test"):
        raise GNNAblationConfigError(
            "cohort selection/evaluation must remain calibration/test"
        )
    if _bool(
        splits.get("test_open_after_selector_freeze"),
        field="splits.test_open_after_selector_freeze",
    ) is not True:
        raise GNNAblationConfigError("test must remain closed until selectors freeze")

    if _bool(cohorts.get("common"), field="cohorts.common") is not True:
        raise GNNAblationConfigError("common cohort is required")
    if _bool(cohorts.get("native"), field="cohorts.native") is not True:
        raise GNNAblationConfigError("native cohort reporting is required")
    minimum_common = _int(
        cohorts.get("minimum_common_parents"),
        field="cohorts.minimum_common_parents",
        minimum=1,
    )
    expected_evaluation_policy = {
        "distance_metric": "WNode",
        "wnode_config_source": "main_matrix_frozen_contract",
        "wnode_shared_across_backbones": True,
        "selector_fit_split": "calibration",
        "held_out_evaluation_split": "test",
    }
    if evaluation_policy != expected_evaluation_policy:
        raise GNNAblationConfigError("shared WNode/selector evaluation policy changed")

    output_root = _text(outputs.get("root_template"), field="outputs.root_template")
    if "{attempt_id}" not in output_root:
        raise GNNAblationConfigError("output root must contain {attempt_id}")
    if _bool(policy.get("matrix_write_allowed"), field="policy.matrix_write_allowed"):
        raise GNNAblationConfigError("ablation may not write the main matrix")
    if _bool(
        policy.get("science_launch_allowed_by_framework"),
        field="policy.science_launch_allowed_by_framework",
    ):
        raise GNNAblationConfigError("config-only framework may not launch science")
    if _bool(
        policy.get("require_fresh_attempt"), field="policy.require_fresh_attempt"
    ) is not True:
        raise GNNAblationConfigError("every run requires a fresh attempt")

    if dataset == "tastemolnet":
        required_taste = {
            "mode": "scoped_research_no_redistribution_v2",
            "policy_receipt_required": True,
            "upstream_terms_status": "NOT_EXPLICITLY_STATED",
            "redistribution_allowed": False,
            "graph_cache_mode": "read_only_existing",
        }
        changed = sorted(
            key for key, expected in required_taste.items() if data_policy.get(key) != expected
        )
        if changed:
            raise GNNAblationConfigError(
                "Taste data-policy contract changed: " + ", ".join(changed)
            )
    elif data_policy.get("redistribution_allowed") is not False:
        raise GNNAblationConfigError("ablation outputs never redistribute datasets")

    normalized_backbones: list[BackbonePlanConfig] = []
    comparison_hashes: set[str] = set()
    for raw_name, raw_value in backbone_values.items():
        name = normalize_gnn_backbone(str(raw_name))
        values = _mapping(raw_value, field=f"backbones.{name}")
        _exact_keys(
            values,
            field=f"backbones.{name}",
            expected={"role", "checkpoint_policy", "model_config"},
        )
        role = _text(values.get("role"), field=f"backbones.{name}.role")
        checkpoint_policy = _text(
            values.get("checkpoint_policy"),
            field=f"backbones.{name}.checkpoint_policy",
        )
        model_path = _project_file(
            root, values.get("model_config"), field=f"backbones.{name}.model_config"
        )
        if model_path.stem != name:
            raise GNNAblationConfigError(
                f"{name} points to a different model config: {model_path.stem}"
            )
        if role not in {"reference", "ablation"}:
            raise GNNAblationConfigError(f"unsupported backbone role: {role}")
        allowed_policies = {
            "reference": {"adopt_if_compatible_else_train"},
            "ablation": {"train"},
        }
        if checkpoint_policy not in allowed_policies[role]:
            raise GNNAblationConfigError(
                f"checkpoint policy {checkpoint_policy!r} is invalid for {role}"
            )
        model_sha, comparison_sha = _parse_model_config(name=name, path=model_path)
        comparison_hashes.add(comparison_sha)
        normalized_backbones.append(
            BackbonePlanConfig(
                name=name,
                role=role,
                checkpoint_policy=checkpoint_policy,
                model_config=str(model_path.relative_to(root)),
                model_config_sha256=model_sha,
                comparison_config_sha256=comparison_sha,
                edge_feature_mode=get_gnn_backbone_spec(name).edge_feature_mode,
            )
        )
    names = tuple(item.name for item in normalized_backbones)
    if names != REQUIRED_BACKBONES or set(names) != set(available_gnn_backbones()):
        raise GNNAblationConfigError(
            "backbones must appear exactly as gine, gin, gcn, gatv2"
        )
    if [item.name for item in normalized_backbones if item.role == "reference"] != [
        "gine"
    ]:
        raise GNNAblationConfigError("GINE must be the sole reference backbone")
    if len(comparison_hashes) != 1:
        raise GNNAblationConfigError(
            "backbone model/training/calibration configs differ beyond backbone"
        )

    return GNNAblationConfig(
        source_path=str(path),
        source_sha256=_sha256_file(path),
        project_root=str(root),
        experiment_id=experiment_id,
        dataset=dataset,
        num_classes=num_classes,
        source_label=source_label,
        dataset_config=str(dataset_config_path.relative_to(root)),
        dataset_config_sha256=_sha256_file(dataset_config_path),
        shared_feature_schema_provider=provider,
        shared_feature_schema_sha256=configured_feature_sha,
        primary_results_gate=primary_gate,
        proposal_mode=proposal_mode,
        fit_split=fit_split,
        selection_split=selection_split,
        proposal_split=proposal_split,
        calibration_split=calibration_split,
        evaluation_split=evaluation_split,
        minimum_common_parents=minimum_common,
        output_root_template=output_root,
        evaluation_policy=evaluation_policy,
        policy=policy,
        data_policy=data_policy,
        backbones=tuple(normalized_backbones),
    )


def _output_contract(config: GNNAblationConfig) -> dict[str, Any]:
    variant_files = output_inventory("gnn")
    aggregate_files = output_inventory("gnn", aggregate=True)
    required_bundle = required_backbone_bundle_files(config.dataset)
    return {
        "schema_version": OUTPUT_CONTRACT_SCHEMA,
        "root_template": config.output_root_template,
        "fresh_attempt_required": True,
        "science_execution_allowed": False,
        "autodl_write_allowed": False,
        "main_matrix_artifacts_allowed": False,
        "shared_feature_schema_sha256": config.shared_feature_schema_sha256,
        "wnode_contract": {
            "distance_metric": "WNode",
            "config_source": "main_matrix_frozen_contract",
            "shared_across_backbones_and_splits": True,
            "runtime_config_sha256_required": True,
        },
        "artifacts": {
            "plan": {"path": "plan.json", "schema_version": PLAN_SCHEMA},
            "proposal_universe": {
                "path": "proposal/train_rule_universe.json",
                "schema_version": PROPOSAL_UNIVERSE_SCHEMA,
                "source_split": "train",
            },
            "proposal_rows": {
                "path": "proposal/train_rules.jsonl",
                "schema_version": CANDIDATE_IDENTITY_SCHEMA,
                "source_split": "train",
                "forbidden_parent_splits": ["calibration", "test"],
            },
            "model_bundle": {
                "path_template": "models/{backbone}/frozen",
                "schema_version": "molecular_gnn_checkpoint_v2",
                "required_files": list(required_bundle),
                "shared_feature_schema_sha256": config.shared_feature_schema_sha256,
                "temperature_contract": {
                    "status": "fit",
                    "selection_split": "validation",
                    "test_used_for_fit": False,
                    "required_provenance": [
                        "dataset",
                        "validation_split_sha256",
                        "ordered_parent_ids_sha256",
                        "ordered_labels_sha256",
                        "selected_checkpoint_sha256",
                        "feature_schema_sha256",
                    ],
                    "self_hash_required": True,
                    "validation_predictions_binding_required": True,
                },
                "edge_feature_mode_must_match_registry": True,
                "hash_verification_required": True,
                "taste_closure_required": True,
                "registry_save_state": (
                    "BLOCKED_UNIMPLEMENTED_FULL_CLOSURE"
                    if config.dataset == "tastemolnet"
                    else "AVAILABLE"
                ),
                "taste_bundle_source_policy": (
                    "adopt_or_train_via_existing_complete_taste_pipeline"
                    if config.dataset == "tastemolnet"
                    else "registry_save_allowed"
                ),
            },
            "parent_predictions": {
                "path_template": "cohorts/{split}/predictions/{backbone}.jsonl",
                "schema_version": PARENT_PREDICTION_SCHEMA,
                "required_bindings": [
                    "dataset",
                    "split",
                    "split_sha256",
                    "feature_schema_sha256",
                    "temperature_scaling_sha256",
                    "checkpoint_sha256",
                    "edge_feature_mode",
                ],
            },
            "cohort_freeze": {
                "path_template": "cohorts/{split}/cohort_freeze.json",
                "schema_version": COHORT_FREEZE_SCHEMA,
                "splits": ["calibration", "test"],
                "split_authority_schema": COHORT_SPLIT_AUTHORITY_SCHEMA,
                "split_authority_self_hash_required": True,
            },
            "calibration_evaluation": {
                "path_template": "calibration/{backbone}/{cohort}/{artifact}",
                "artifacts": ["rows.jsonl", "metrics.json", "wnode.json"],
            },
            "selector_freeze": {
                "path_template": "selectors/{backbone}/frozen.json",
                "schema_version": "gnn_calibration_only_selector_freeze_v1",
                "input_split": "calibration",
                "test_loaded": False,
            },
            "test_evaluation": {
                "path_template": "test/{backbone}/{cohort}/{artifact}",
                "artifacts": ["rows.jsonl", "metrics.json", "wnode.json"],
                "selector_must_be_frozen": True,
            },
            "variant_output_inventory": {
                "path_template": "variants/{backbone}/{filename}",
                "files": list(variant_files),
                "files_sha256": stable_sha256(list(variant_files)),
                "metrics_must_come_from_science": True,
            },
            "aggregate_output_inventory": {
                "path_template": "aggregate/{filename}",
                "files": list(aggregate_files),
                "files_sha256": stable_sha256(list(aggregate_files)),
                "metrics_must_come_from_science": True,
            },
            "final_manifest": {
                "path": "final_manifest.json",
                "schema_version": FINAL_MANIFEST_SCHEMA,
            },
            "artifact_inventory": {"path": "artifact_inventory.json"},
            "sha256sums": {"path": "sha256sums.txt"},
        },
    }


def _model_tasks(config: GNNAblationConfig) -> tuple[list[AblationTask], list[str]]:
    tasks: list[AblationTask] = []
    freezes: list[str] = []
    for backbone in config.backbones:
        prepare = f"model:{backbone.name}:prepare"
        calibrate = f"model:{backbone.name}:temperature"
        freeze = f"model:{backbone.name}:freeze"
        tasks.extend(
            [
                AblationTask(
                    task_id=prepare,
                    stage="MODEL_PREPARE",
                    component="shared_schema_molecular_gnn_trainer_or_reference_adopter",
                    depends_on=("gate:fast16-16of16",),
                    split_access=(config.fit_split, config.selection_split),
                    outputs=(f"models/{backbone.name}/prepared",),
                    backbone=backbone.name,
                    checkpoint_policy=backbone.checkpoint_policy,
                ),
                AblationTask(
                    task_id=calibrate,
                    stage="VALIDATION_TEMPERATURE_CALIBRATE",
                    component="validation_only_temperature_scaling",
                    depends_on=(prepare,),
                    split_access=(config.selection_split,),
                    outputs=(f"models/{backbone.name}/temperature_scaling.json",),
                    backbone=backbone.name,
                ),
                AblationTask(
                    task_id=freeze,
                    stage="CHECKPOINT_FREEZE",
                    component="complete_bundle_schema_temperature_edge_mode_freezer",
                    depends_on=(calibrate,),
                    split_access=(),
                    outputs=(f"models/{backbone.name}/frozen",),
                    backbone=backbone.name,
                ),
            ]
        )
        freezes.append(freeze)
    return tasks, freezes


def _tasks(config: GNNAblationConfig) -> tuple[AblationTask, ...]:
    tasks = [
        AblationTask(
            task_id="gate:fast16-16of16",
            stage="PRIMARY_RESULTS_GATE",
            component="hash_closed_fast16_gate",
            depends_on=(),
            split_access=(),
            outputs=("gate/primary_results_gate.json",),
        )
    ]
    model_tasks, model_freezes = _model_tasks(config)
    tasks.extend(model_tasks)
    proposal_task = "proposal:train-rules:freeze"
    tasks.append(
        AblationTask(
            task_id=proposal_task,
            stage="TRAIN_RULE_PROPOSAL_FREEZE",
            component="backbone_independent_train_rule_pool_freezer",
            depends_on=tuple(model_freezes),
            split_access=(config.proposal_split,),
            outputs=("proposal/train_rules.jsonl", "proposal/train_rule_universe.json"),
        )
    )

    calibration_scores: list[str] = []
    for backbone in config.backbones:
        task_id = f"calibration:{backbone.name}:parents"
        calibration_scores.append(task_id)
        tasks.append(
            AblationTask(
                task_id=task_id,
                stage="CALIBRATION_PARENT_SCORE",
                component="frozen_parent_oracle_scorer",
                depends_on=(f"model:{backbone.name}:freeze", proposal_task),
                split_access=(config.calibration_split,),
                outputs=(f"cohorts/calibration/predictions/{backbone.name}.jsonl",),
                backbone=backbone.name,
            )
        )
    calibration_freeze = "cohorts:calibration:freeze"
    tasks.append(
        AblationTask(
            task_id=calibration_freeze,
            stage="CALIBRATION_COMMON_NATIVE_FREEZE",
            component="split_hash_schema_temperature_bound_cohort_freezer",
            depends_on=tuple(calibration_scores),
            split_access=(),
            outputs=("cohorts/calibration/cohort_freeze.json",),
        )
    )

    selector_tasks: list[str] = []
    calibration_eval_by_backbone: dict[str, list[str]] = {}
    for backbone in config.backbones:
        evaluation_ids: list[str] = []
        for cohort in ("common", "native"):
            task_id = f"calibration:{backbone.name}:{cohort}:evaluate"
            evaluation_ids.append(task_id)
            tasks.append(
                AblationTask(
                    task_id=task_id,
                    stage="CALIBRATION_NATIVE_COMMON_WNODE",
                    component="calibration_rule_validation_and_wnode",
                    depends_on=(
                        f"model:{backbone.name}:freeze",
                        proposal_task,
                        calibration_freeze,
                    ),
                    split_access=(config.calibration_split,),
                    outputs=(
                        f"calibration/{backbone.name}/{cohort}/rows.jsonl",
                        f"calibration/{backbone.name}/{cohort}/metrics.json",
                        f"calibration/{backbone.name}/{cohort}/wnode.json",
                    ),
                    backbone=backbone.name,
                    cohort_kind=cohort,
                )
            )
        calibration_eval_by_backbone[backbone.name] = evaluation_ids
        selector_id = f"selector:{backbone.name}:freeze"
        selector_tasks.append(selector_id)
        tasks.append(
            AblationTask(
                task_id=selector_id,
                stage="CALIBRATION_ONLY_SELECTOR_FREEZE",
                component="calibration_only_rule_selector",
                depends_on=tuple(evaluation_ids),
                split_access=(),
                outputs=(f"selectors/{backbone.name}/frozen.json",),
                backbone=backbone.name,
            )
        )
    selector_barrier = "selectors:all:freeze"
    tasks.append(
        AblationTask(
            task_id=selector_barrier,
            stage="ALL_SELECTORS_FROZEN",
            component="held_out_test_open_barrier",
            depends_on=tuple(selector_tasks),
            split_access=(),
            outputs=("selectors/all_frozen.json",),
        )
    )

    test_scores: list[str] = []
    for backbone in config.backbones:
        task_id = f"test:{backbone.name}:parents"
        test_scores.append(task_id)
        tasks.append(
            AblationTask(
                task_id=task_id,
                stage="HELD_OUT_TEST_PARENT_SCORE",
                component="frozen_parent_oracle_scorer",
                depends_on=(
                    f"model:{backbone.name}:freeze",
                    proposal_task,
                    selector_barrier,
                ),
                split_access=(config.evaluation_split,),
                outputs=(f"cohorts/test/predictions/{backbone.name}.jsonl",),
                backbone=backbone.name,
            )
        )
    test_freeze = "cohorts:test:freeze"
    tasks.append(
        AblationTask(
            task_id=test_freeze,
            stage="HELD_OUT_TEST_COMMON_NATIVE_FREEZE",
            component="split_hash_schema_temperature_bound_cohort_freezer",
            depends_on=tuple(test_scores),
            split_access=(),
            outputs=("cohorts/test/cohort_freeze.json",),
        )
    )

    test_eval_by_backbone: dict[str, list[str]] = {}
    for backbone in config.backbones:
        evaluation_ids: list[str] = []
        for cohort in ("common", "native"):
            task_id = f"test:{backbone.name}:{cohort}:evaluate"
            evaluation_ids.append(task_id)
            tasks.append(
                AblationTask(
                    task_id=task_id,
                    stage="HELD_OUT_TEST_NATIVE_COMMON_EVALUATION",
                    component="frozen_selector_strict_flip_and_wnode_evaluator",
                    depends_on=(
                        f"model:{backbone.name}:freeze",
                        proposal_task,
                        f"selector:{backbone.name}:freeze",
                        selector_barrier,
                        test_freeze,
                    ),
                    split_access=(config.evaluation_split,),
                    outputs=(
                        f"test/{backbone.name}/{cohort}/rows.jsonl",
                        f"test/{backbone.name}/{cohort}/metrics.json",
                        f"test/{backbone.name}/{cohort}/wnode.json",
                    ),
                    backbone=backbone.name,
                    cohort_kind=cohort,
                )
            )
        test_eval_by_backbone[backbone.name] = evaluation_ids

    variant_tasks: list[str] = []
    for backbone in config.backbones:
        task_id = f"variant:{backbone.name}:finalize"
        variant_tasks.append(task_id)
        tasks.append(
            AblationTask(
                task_id=task_id,
                stage="VARIANT_OUTPUT_FINALIZE",
                component="gnn_common_output_schema_publisher",
                depends_on=tuple(
                    calibration_eval_by_backbone[backbone.name]
                    + [f"selector:{backbone.name}:freeze"]
                    + test_eval_by_backbone[backbone.name]
                ),
                split_access=(),
                outputs=tuple(
                    f"variants/{backbone.name}/{filename}"
                    for filename in output_inventory("gnn")
                ),
                backbone=backbone.name,
            )
        )
    tasks.extend(
        [
            AblationTask(
                task_id="aggregate",
                stage="ABLATION_AGGREGATE",
                component="common_primary_native_secondary_aggregator",
                depends_on=tuple(variant_tasks),
                split_access=(),
                outputs=tuple(
                    f"aggregate/{filename}"
                    for filename in output_inventory("gnn", aggregate=True)
                ),
            ),
            AblationTask(
                task_id="manifest:final",
                stage="FINAL_MANIFEST_PLAN",
                component="hash_closed_manifest_publisher",
                depends_on=("aggregate",),
                split_access=(),
                outputs=("final_manifest.json", "artifact_inventory.json", "sha256sums.txt"),
            ),
        ]
    )
    return tuple(tasks)


def _dependency_closure(tasks: Sequence[AblationTask], task_id: str) -> set[str]:
    by_id = {task.task_id: task for task in tasks}
    result: set[str] = set()
    stack = list(by_id[task_id].depends_on)
    while stack:
        current = stack.pop()
        if current in result:
            continue
        result.add(current)
        stack.extend(by_id[current].depends_on)
    return result


def validate_plan(config: GNNAblationConfig, tasks: Sequence[AblationTask]) -> None:
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise GNNAblationConfigError("plan contains duplicate task ids")
    known: set[str] = set()
    outputs: set[str] = set()
    for task in tasks:
        unknown = set(task.depends_on).difference(known)
        if unknown:
            raise GNNAblationConfigError(
                f"task {task.task_id} has forward/unknown dependencies: {sorted(unknown)}"
            )
        known.add(task.task_id)
        overlap = outputs.intersection(task.outputs)
        if overlap:
            raise GNNAblationConfigError(
                f"multiple tasks own output paths: {sorted(overlap)}"
            )
        outputs.update(task.outputs)
        serialized = task.to_dict()
        if (
            serialized["launches_science"] is not False
            or serialized["writes_autodl"] is not False
            or serialized["writes_main_matrix"] is not False
        ):
            raise GNNAblationConfigError("config-only task gained execution authority")
        if task.stage == "TRAIN_RULE_PROPOSAL_FREEZE" and task.split_access != ("train",):
            raise GNNAblationConfigError("proposal rule pool is not train-only")
        if task.stage.startswith("CALIBRATION") and TEST_SPLIT_NAMES.intersection(
            task.split_access
        ):
            raise GNNAblationConfigError("test leaked into calibration or selector")

    freeze_ids = {f"model:{name}:freeze" for name in config.backbone_names}
    selector_ids = {f"selector:{name}:freeze" for name in config.backbone_names}
    proposal = next(task for task in tasks if task.stage == "TRAIN_RULE_PROPOSAL_FREEZE")
    if set(proposal.depends_on) != freeze_ids:
        raise GNNAblationConfigError("proposal freeze does not follow every model freeze")
    for task in tasks:
        if TEST_SPLIT_NAMES.intersection(task.split_access):
            closure = _dependency_closure(tasks, task.task_id)
            if not freeze_ids.issubset(closure) or not selector_ids.issubset(closure):
                raise GNNAblationConfigError(
                    f"test-opening task {task.task_id} precedes model/selector freeze"
                )
    for split, stage in (
        ("calibration", "CALIBRATION_NATIVE_COMMON_WNODE"),
        ("test", "HELD_OUT_TEST_NATIVE_COMMON_EVALUATION"),
    ):
        observed = {
            (task.backbone, task.cohort_kind)
            for task in tasks
            if task.stage == stage
        }
        expected = {
            (backbone, cohort)
            for backbone in config.backbone_names
            for cohort in ("common", "native")
        }
        if observed != expected:
            raise GNNAblationConfigError(f"{split} common/native grid is incomplete")


@dataclass(frozen=True, slots=True)
class GNNAblationPlan:
    payload: Mapping[str, Any]

    @property
    def plan_sha256(self) -> str:
        return str(self.payload["plan_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def build_ablation_plan(config: GNNAblationConfig) -> GNNAblationPlan:
    tasks = _tasks(config)
    validate_plan(config, tasks)
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "PLANNED_NOT_RUN",
        "execution_mode": "CONFIG_ONLY",
        "science_executed": False,
        "autodl_modified": False,
        "main_matrix_modified": False,
        "config": config.to_dict(),
        "candidate_identity_contract": {
            "schema_version": CANDIDATE_IDENTITY_SCHEMA,
            "proposal_mode": config.proposal_mode,
            "proposal_source_split": "train",
            "generation_per_backbone": False,
            "calibration_or_test_parent_in_identity": False,
            "identity_fields": [
                "dataset",
                "rule_sha256",
                "fragment_graph_sha256",
                "source_split",
                "action_type",
            ],
        },
        "model_contract": {
            "shared_feature_schema_sha256": config.shared_feature_schema_sha256,
            "temperature_selection_split": "validation",
            "temperature_test_used_for_fit": False,
            "edge_feature_mode_must_match_registry": True,
            "complete_bundle_files": list(required_backbone_bundle_files(config.dataset)),
        },
        "cohort_contract": {
            "schema_version": COHORT_FREEZE_SCHEMA,
            "splits": ["calibration", "test"],
            "native_eligibility": "true_label == source_label and pred_before == source_label",
            "common_definition": "ordered_intersection_of_all_native_parent_ids_per_split",
            "common_cohort_primary": True,
            "native_cohorts_secondary": True,
            "true_labels_must_match_across_backbones": True,
            "required_bindings": [
                "dataset",
                "split",
                "split_sha256",
                "feature_schema_sha256",
                "temperature_scaling_sha256s",
                "checkpoint_sha256s",
                "edge_feature_modes",
            ],
            "same_train_rule_universe_for_every_backbone_and_split": True,
        },
        "backbone_registry": [
            {
                "name": item.name,
                "display_name": get_gnn_backbone_spec(item.name).display_name,
                "edge_feature_mode": item.edge_feature_mode,
                "role": item.role,
            }
            for item in config.backbones
        ],
        "tasks": [task.to_dict() for task in tasks],
        "output_contract": _output_contract(config),
        "final_manifest_plan": {
            "schema_version": FINAL_MANIFEST_SCHEMA,
            "required_status": "PASS",
            "required_bindings": [
                "config_scientific_sha256",
                "primary_results_gate_sha256",
                "shared_feature_schema_sha256",
                "checkpoint_sha256_by_backbone",
                "temperature_scaling_sha256_by_backbone",
                "edge_feature_mode_by_backbone",
                "wnode_config_sha256",
                "train_rule_proposal_universe_sha256",
                "calibration_cohort_freeze_sha256",
                "calibration_evaluation_sha256_by_backbone_and_cohort",
                "selector_freeze_sha256_by_backbone",
                "test_cohort_freeze_sha256",
                "test_evaluation_sha256_by_backbone_and_cohort",
                "variant_output_inventory_sha256_by_backbone",
                "aggregate_output_inventory_sha256",
                "artifact_inventory_sha256",
            ],
            "required_disclosures": {
                "proposal_fixed": True,
                "proposal_source_split": "train",
                "calibration_or_test_parent_in_proposal_identity": False,
                "candidate_generation_per_backbone": False,
                "selector_fit_split": "calibration",
                "test_used_for_fit_selection_or_temperature": False,
                "common_cohort_primary": True,
                "native_cohorts_secondary": True,
                "shared_feature_schema": True,
                "edge_feature_modes_verified": True,
                "main_matrix_modified": False,
                "dataset_redistributed": False,
            },
            "publication_order": [
                "science_artifacts",
                "artifact_inventory",
                "sha256sums",
                "final_manifest",
                "PASS",
            ],
        },
    }
    payload["plan_sha256"] = stable_sha256(payload)
    return GNNAblationPlan(payload=payload)


def build_ablation_plan_from_config(
    path_like: str | Path,
    *,
    project_root: str | Path | None = None,
) -> GNNAblationPlan:
    return build_ablation_plan(
        load_ablation_config(path_like, project_root=project_root)
    )


__all__ = [
    "AblationTask",
    "BackbonePlanConfig",
    "CONFIG_SCHEMA",
    "FINAL_MANIFEST_SCHEMA",
    "GNNAblationConfig",
    "GNNAblationConfigError",
    "GNNAblationPlan",
    "OUTPUT_CONTRACT_SCHEMA",
    "PLAN_SCHEMA",
    "TASK_SCHEMA",
    "build_ablation_plan",
    "build_ablation_plan_from_config",
    "load_ablation_config",
    "validate_plan",
]
