"""Strict five-backbone BACE proposal-fixed ablation contract.

This module only builds and validates a plan.  It never trains a classifier,
opens the test split, acquires a GPU lease, or mutates the main matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.ablations.gnn.contracts import stable_sha256
from src.models.gnn_backbone_registry import (
    get_gnn_backbone_spec,
    normalize_gnn_backbone,
)
from src.models.gatedgcn_plus_backbone import (
    GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS,
    GATEDGCN_PLUS_DROPOUT,
    GATEDGCN_PLUS_NUM_LAYERS,
    GATEDGCN_PLUS_OFFICIAL_COMMIT,
    GATEDGCN_PLUS_OFFICIAL_REPOSITORY,
    GATEDGCN_PLUS_LICENSE_SHA256,
    GATEDGCN_PLUS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE,
    GATEDGCN_PLUS_RWPE_DIM,
    GATEDGCN_PLUS_RWPE_WALK_LENGTH,
    match_gatedgcn_plus_hidden_dim,
)


FIVE_BACKBONES = ("gine", "gin", "gcn", "gatv2", "gatedgcn_plus")
FIVE_BACKBONE_CONFIG_SCHEMA = "gnn_five_backbone_proposal_fixed_v1"
FIVE_BACKBONE_PLAN_SCHEMA = "gnn_five_backbone_plan_v1"
GINE_REFERENCE_PARAMETERS = 1_432_583
GATEDGCN_PLUS_SELECTED_HIDDEN_DIM = 160


class FiveBackboneConfigError(ValueError):
    """The five-backbone plan is scientifically incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FiveBackboneConfigError(f"{field} must be a mapping")
    return dict(value)


def _project_file(project_root: Path, raw: Any, *, field: str) -> Path:
    value = Path(str(raw or "").strip())
    if not str(value) or value.is_absolute():
        raise FiveBackboneConfigError(f"{field} must be repository-relative")
    resolved = (project_root / value).resolve(strict=True)
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise FiveBackboneConfigError(f"{field} escapes project root") from exc
    if not resolved.is_file():
        raise FiveBackboneConfigError(f"{field} is not a file")
    return resolved


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiveBackboneConfigError(f"{field} is not valid JSON") from exc
    return _mapping(payload, field=field)


def _load_yaml(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FiveBackboneConfigError(f"{field} is not valid YAML") from exc
    return _mapping(payload, field=field)


def _load_gatedgcn_plus_receipts(
    *, project_root: Path, gatedgcn_plus: Mapping[str, Any]
) -> dict[str, Any]:
    reference_path = _project_file(
        project_root,
        gatedgcn_plus.get("reference_parameter_receipt"),
        field="gatedgcn_plus.reference_parameter_receipt",
    )
    match_path = _project_file(
        project_root,
        gatedgcn_plus.get("parameter_match_receipt"),
        field="gatedgcn_plus.parameter_match_receipt",
    )
    source_path = _project_file(
        project_root,
        gatedgcn_plus.get("source_mapping"),
        field="gatedgcn_plus.source_mapping",
    )
    source = _load_yaml(source_path, field="GatedGCN+ source mapping")
    if (
        source.get("schema_version") != "gatedgcn_plus_source_mapping_v1"
        or source.get("status") != "PASS"
        or source.get("official_repository")
        != f"https://github.com/{GATEDGCN_PLUS_OFFICIAL_REPOSITORY}"
        or source.get("official_commit") != GATEDGCN_PLUS_OFFICIAL_COMMIT
        or source.get("license") != "MIT"
        or source.get("license_sha256") != GATEDGCN_PLUS_LICENSE_SHA256
        or source.get("adapted_hyperparameters_not_official_bace_recipe") is not True
        or source.get("moving_main_executed") is not False
        or not isinstance(source.get("relevant_files"), list)
        or not source.get("relevant_files")
    ):
        raise FiveBackboneConfigError("GatedGCN+ pinned source mapping changed")
    reference = _load_json(reference_path, field="GINE parameter receipt")
    if (
        reference.get("schema_version")
        != "bace_gine_reference_parameter_receipt_v1"
        or reference.get("status") != "PASS"
        or reference.get("dataset") != "bace"
        or reference.get("method") != "ours"
        or reference.get("backbone") != "gine"
        or reference.get("source") != "ACTUAL_LOADED_WEIGHTS"
        or reference.get("total_parameters") != GINE_REFERENCE_PARAMETERS
        or reference.get("trainable_parameters") != GINE_REFERENCE_PARAMETERS
        or reference.get("validation_metrics_loaded_for_parameter_count") is not False
        or reference.get("test_metrics_loaded_for_parameter_count") is not False
    ):
        raise FiveBackboneConfigError("GINE reference parameter receipt changed")
    match = _load_json(match_path, field="GatedGCN+ parameter-match receipt")
    recomputed = match_gatedgcn_plus_hidden_dim(reference["total_parameters"])
    candidate_projection = [
        {
            "hidden_dim": candidate.hidden_dim,
            "parameter_count": candidate.parameter_count,
            "absolute_difference": candidate.absolute_difference,
            "relative_difference": candidate.relative_difference,
            "within_tolerance": candidate.within_tolerance,
        }
        for candidate in recomputed.candidates
    ]
    if (
        match.get("schema_version") != "gatedgcn_plus_parameter_match_v1"
        or match.get("status") != "PASS"
        or match.get("reference_parameter_count") != recomputed.reference_parameter_count
        or match.get("candidate_backbone") != "gatedgcn_plus"
        or match.get("official_repository")
        != f"https://github.com/{GATEDGCN_PLUS_OFFICIAL_REPOSITORY}"
        or match.get("official_commit") != GATEDGCN_PLUS_OFFICIAL_COMMIT
        or match.get("adapted_hyperparameters_not_official_bace_recipe") is not True
        or match.get("allowed_hidden_dims")
        != list(GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS)
        or match.get("max_relative_difference")
        != GATEDGCN_PLUS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE
        or match.get("selected_hidden_dim") != recomputed.selected_hidden_dim
        or match.get("selected_parameter_count") != recomputed.selected_parameter_count
        or match.get("selected_relative_difference")
        != recomputed.selected_relative_difference
        or match.get("candidates") != candidate_projection
        or match.get("validation_metrics_loaded") is not False
        or match.get("test_metrics_loaded") is not False
    ):
        raise FiveBackboneConfigError(
            "GatedGCN+ parameter-match receipt differs from the parameter-only recomputation"
        )
    return {
        "reference_receipt_path": str(reference_path.relative_to(project_root)),
        "reference_receipt_sha256": _sha256_file(reference_path),
        "match_receipt_path": str(match_path.relative_to(project_root)),
        "match_receipt_sha256": _sha256_file(match_path),
        "source_mapping_path": str(source_path.relative_to(project_root)),
        "source_mapping_sha256": _sha256_file(source_path),
        "official_commit": GATEDGCN_PLUS_OFFICIAL_COMMIT,
        "reference_parameters": recomputed.reference_parameter_count,
        "selected_hidden_dim": recomputed.selected_hidden_dim,
        "selected_parameters": recomputed.selected_parameter_count,
        "relative_difference": recomputed.selected_relative_difference,
    }


def _load_graph_mamba_metadata(
    *, project_root: Path, graph_mamba: Mapping[str, Any]
) -> dict[str, Any]:
    metadata_path = _project_file(
        project_root, graph_mamba.get("metadata"), field="graph_mamba.metadata"
    )
    payload = _load_yaml(metadata_path, field="Graph-Mamba metadata")
    if (
        payload.get("schema_version") != "graph_mamba_optional_metadata_v1"
        or payload.get("registered") is not True
        or payload.get("run_enabled") is not False
        or payload.get("weights_downloaded") is not False
        or payload.get("code_downloaded") is not False
        or payload.get("science_started") is not False
        or payload.get("gpu_lock_allowed") is not False
        or payload.get("official_commit")
        != "acb4a2321d46f4044cb5e073a9fadd47eb4f343f"
    ):
        raise FiveBackboneConfigError("Graph-Mamba must remain metadata-only")
    if graph_mamba.get("registered") is not True or graph_mamba.get("run_enabled") is not False:
        raise FiveBackboneConfigError("Graph-Mamba run policy changed")
    return {
        "metadata_path": str(metadata_path.relative_to(project_root)),
        "metadata_sha256": _sha256_file(metadata_path),
        "official_commit": payload["official_commit"],
        "license_status": payload.get("license_status"),
        "run_enabled": False,
        "science_started": False,
    }


@dataclass(frozen=True, slots=True)
class FiveBackboneConfig:
    source_path: str
    source_sha256: str
    project_root: str
    experiment_id: str
    backbones: tuple[str, ...]
    primary_seed: int
    optional_seeds: tuple[int, ...]
    max_concurrent_gpus: int
    model_configs: Mapping[str, str]
    model_config_sha256s: Mapping[str, str]
    gatedgcn_plus_receipts: Mapping[str, Any]
    graph_mamba_metadata: Mapping[str, Any]
    output_root_template: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIVE_BACKBONE_CONFIG_SCHEMA,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "project_root": self.project_root,
            "experiment_id": self.experiment_id,
            "dataset": "bace",
            "method": "ours",
            "mode": "proposal_fixed",
            "backbones": list(self.backbones),
            "primary_seed": self.primary_seed,
            "optional_seeds": list(self.optional_seeds),
            "max_concurrent_gpus": self.max_concurrent_gpus,
            "model_configs": dict(self.model_configs),
            "model_config_sha256s": dict(self.model_config_sha256s),
            "gatedgcn_plus_receipts": dict(self.gatedgcn_plus_receipts),
            "graph_mamba_metadata": dict(self.graph_mamba_metadata),
            "output_root_template": self.output_root_template,
        }


def load_five_backbone_config(
    path_like: str | Path,
    *,
    project_root: str | Path | None = None,
) -> FiveBackboneConfig:
    path = Path(path_like).expanduser().resolve(strict=True)
    root = (
        Path(project_root).expanduser().resolve(strict=True)
        if project_root is not None
        else path.parents[3]
    )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FiveBackboneConfigError("config path escapes project root") from exc
    raw = _load_yaml(path, field="config")
    if raw.get("schema_version") != FIVE_BACKBONE_CONFIG_SCHEMA:
        raise FiveBackboneConfigError("five-backbone config schema changed")
    experiment = _mapping(raw.get("experiment"), field="experiment")
    if (
        experiment.get("dataset") != "bace"
        or experiment.get("method") != "ours"
        or experiment.get("mode") != "proposal_fixed"
        or experiment.get("primary_seed") != 7
        or experiment.get("optional_seeds") != [17, 27]
        or experiment.get("max_concurrent_gpus") != 2
        or experiment.get("science_started") is not False
    ):
        raise FiveBackboneConfigError("experiment contract changed")
    raw_backbones = raw.get("backbones")
    if not isinstance(raw_backbones, list) or len(raw_backbones) != 5:
        raise FiveBackboneConfigError("exactly five backbone mappings are required")
    model_configs: dict[str, str] = {}
    model_hashes: dict[str, str] = {}
    model_payloads: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    for index, entry in enumerate(raw_backbones):
        value = _mapping(entry, field=f"backbones[{index}]")
        name = normalize_gnn_backbone(value.get("name"))
        names.append(name)
        expected_role = "reference" if name == "gine" else "ablation"
        expected_policy = (
            "adopt_exact_bace_main_if_verified" if name == "gine" else "train"
        )
        if value.get("role") != expected_role or value.get("checkpoint_policy") != expected_policy:
            raise FiveBackboneConfigError(f"{name} role/checkpoint policy changed")
        model_path = _project_file(
            root, value.get("model_config"), field=f"backbones.{name}.model_config"
        )
        if model_path.stem != name:
            raise FiveBackboneConfigError(f"{name} model config points elsewhere")
        model_configs[name] = str(model_path.relative_to(root))
        model_hashes[name] = _sha256_file(model_path)
        model_payloads[name] = _load_yaml(model_path, field=f"model config {name}")
    if tuple(names) != FIVE_BACKBONES:
        raise FiveBackboneConfigError(
            f"backbones must appear exactly as {FIVE_BACKBONES}"
        )

    reference_training = _mapping(
        model_payloads["gine"].get("training"), field="gine.training"
    )
    reference_calibration = _mapping(
        model_payloads["gine"].get("calibration"), field="gine.calibration"
    )
    common_gnn = {
        "num_layers": 5,
        "dropout": 0.2,
        "pooling": "mean",
        "readout_layers": 2,
        "normalization": "batch_norm",
        "residual": True,
    }
    for name, payload in model_payloads.items():
        gnn = _mapping(payload.get("gnn"), field=f"{name}.gnn")
        if normalize_gnn_backbone(gnn.get("backbone")) != name:
            raise FiveBackboneConfigError(f"{name} model config selects another backbone")
        changed_common = [
            field for field, expected in common_gnn.items() if gnn.get(field) != expected
        ]
        if changed_common:
            raise FiveBackboneConfigError(
                f"{name} shared architecture fields changed: {changed_common}"
            )
        if _mapping(payload.get("training"), field=f"{name}.training") != reference_training:
            raise FiveBackboneConfigError(f"{name} training policy differs from GINE")
        if _mapping(payload.get("calibration"), field=f"{name}.calibration") != reference_calibration:
            raise FiveBackboneConfigError(f"{name} calibration policy differs from GINE")
        if name == "gatedgcn_plus":
            gated_expected = {
                "hidden_dim": GATEDGCN_PLUS_SELECTED_HIDDEN_DIM,
                "rwpe_walk_length": GATEDGCN_PLUS_RWPE_WALK_LENGTH,
                "rwpe_dim": GATEDGCN_PLUS_RWPE_DIM,
                "rwpe_raw_normalization": "batch_norm",
                "ffn": True,
            }
            changed_gated = [
                field
                for field, expected in gated_expected.items()
                if gnn.get(field) != expected
            ]
            if changed_gated:
                raise FiveBackboneConfigError(
                    "GatedGCN+ executable config differs from the frozen match: "
                    f"{changed_gated}"
                )
        elif gnn.get("hidden_dim") != 256:
            raise FiveBackboneConfigError(f"{name} hidden_dim differs from reference")

    gatedgcn_plus = _mapping(raw.get("gatedgcn_plus"), field="gatedgcn_plus")
    frozen_gated = {
        "official_commit": GATEDGCN_PLUS_OFFICIAL_COMMIT,
        "rwpe_walk_length": GATEDGCN_PLUS_RWPE_WALK_LENGTH,
        "rwpe_dim": GATEDGCN_PLUS_RWPE_DIM,
        "rwpe_source": "topology_only_preprocessing",
        "edge_feature_integration": "native_residual_edge_gates",
        "ffn": True,
        "num_layers": GATEDGCN_PLUS_NUM_LAYERS,
        "dropout": GATEDGCN_PLUS_DROPOUT,
        "pooling": "mean",
        "allowed_hidden_dims": list(GATEDGCN_PLUS_ALLOWED_HIDDEN_DIMS),
        "selected_hidden_dim": GATEDGCN_PLUS_SELECTED_HIDDEN_DIM,
        "max_parameter_difference_fraction": (
            GATEDGCN_PLUS_PARAMETER_MATCH_MAX_RELATIVE_DIFFERENCE
        ),
    }
    changed = [
        key for key, value in frozen_gated.items() if gatedgcn_plus.get(key) != value
    ]
    if changed:
        raise FiveBackboneConfigError(f"frozen GatedGCN+ config changed: {changed}")
    gated_receipts = _load_gatedgcn_plus_receipts(
        project_root=root, gatedgcn_plus=gatedgcn_plus
    )

    proposal = _mapping(raw.get("proposal_fixed"), field="proposal_fixed")
    expected_proposal = {
        "source": "bace_ours_main_candidate_pool",
        "candidate_identity_hash_required": True,
        "runtime_candidate_manifest_required": True,
        "rerun_chemllm": False,
        "rerun_ppo": False,
        "selector_shared": True,
        "verifier_shared": True,
        "evaluation_shared": True,
    }
    if proposal != expected_proposal:
        raise FiveBackboneConfigError("proposal-fixed candidate contract changed")
    splits = _mapping(raw.get("splits"), field="splits")
    if splits != {
        "classifier_fit": "train",
        "checkpoint_selection": "validation",
        "temperature_calibration": "calibration",
        "final_evaluation": "test",
        "test_open_after_selector_freeze": True,
    }:
        raise FiveBackboneConfigError("split isolation contract changed")
    cohorts = _mapping(raw.get("cohorts"), field="cohorts")
    if cohorts != {"native": True, "common_five_backbone_intersection": True}:
        raise FiveBackboneConfigError("native/common cohort contract changed")
    resources = _mapping(raw.get("resource_reporting"), field="resource_reporting")
    if resources != {
        "total_parameters": "required",
        "estimated_flops": "required",
        "peak_vram_bytes": "required",
        "training_gpu_hours": "required",
    }:
        raise FiveBackboneConfigError("resource reporting contract changed")
    gate = _mapping(raw.get("launch_gate"), field="launch_gate")
    if gate != {
        "matrix_complete_cells": 16,
        "matrix_total_cells": 16,
        "final_matrix_audit": "PASS",
        "final_figure3": "PASS",
        "final_figure4": "PASS",
        "final_table2": "PASS",
        "explicit_gnn_run_authorization": True,
        "no_main_task_waiting_for_gpu": True,
        "run_gnn_ablation_default": False,
    }:
        raise FiveBackboneConfigError("16/16 launch gate changed")
    scheduling = _mapping(raw.get("scheduling"), field="scheduling")
    if scheduling != {
        "gpu0": ["gine", "gin", "gatedgcn_plus"],
        "gpu1": ["gcn", "gatv2"],
        "phase1_seeds": [7],
        "optional_seed_expansion": [17, 27],
    }:
        raise FiveBackboneConfigError("five-backbone scheduling contract changed")
    graph_mamba = _mapping(raw.get("graph_mamba"), field="graph_mamba")
    graph_mamba_metadata = _load_graph_mamba_metadata(
        project_root=root, graph_mamba=graph_mamba
    )
    outputs = _mapping(raw.get("outputs"), field="outputs")
    root_template = str(outputs.get("root_template") or "")
    if (
        outputs.get("main_matrix_write_allowed") is not False
        or "{backbone}" not in root_template
        or "{seed}" not in root_template
        or "{attempt_id}" not in root_template
    ):
        raise FiveBackboneConfigError("fresh isolated output contract changed")
    return FiveBackboneConfig(
        source_path=str(path),
        source_sha256=_sha256_file(path),
        project_root=str(root),
        experiment_id=str(experiment.get("id")),
        backbones=tuple(names),
        primary_seed=7,
        optional_seeds=(17, 27),
        max_concurrent_gpus=2,
        model_configs=model_configs,
        model_config_sha256s=model_hashes,
        gatedgcn_plus_receipts=gated_receipts,
        graph_mamba_metadata=graph_mamba_metadata,
        output_root_template=root_template,
    )


def build_five_backbone_plan(config: FiveBackboneConfig) -> dict[str, Any]:
    """Build a no-execution DAG with the held-out test behind all selectors."""

    tasks: list[dict[str, Any]] = [
        {
            "task_id": "gate:main-16of16",
            "stage": "MAIN_RESULTS_GATE",
            "depends_on": [],
            "split_access": [],
        },
        {
            "task_id": "proposal:freeze",
            "stage": "PROPOSAL_FIXED_FREEZE",
            "depends_on": ["gate:main-16of16"],
            "split_access": ["train"],
        },
    ]
    model_freezes: list[str] = []
    selector_freezes: list[str] = []
    for name in config.backbones:
        model = f"model:{name}:freeze"
        selector = f"selector:{name}:freeze"
        model_freezes.append(model)
        selector_freezes.append(selector)
        tasks.extend(
            [
                {
                    "task_id": model,
                    "stage": "MODEL_TRAIN_VALIDATE_CALIBRATE_FREEZE",
                    "backbone": name,
                    "depends_on": ["gate:main-16of16"],
                    "split_access": ["train", "validation", "calibration"],
                },
                {
                    "task_id": selector,
                    "stage": "CALIBRATION_ONLY_SELECTOR_FREEZE",
                    "backbone": name,
                    "depends_on": [model, "proposal:freeze"],
                    "split_access": ["calibration"],
                },
            ]
        )
    tasks.append(
        {
            "task_id": "cohort:common-five:freeze",
            "stage": "COMMON_FIVE_BACKBONE_COHORT_FREEZE",
            "depends_on": model_freezes,
            "split_access": ["calibration"],
        }
    )
    tasks.append(
        {
            "task_id": "selectors:all:freeze",
            "stage": "ALL_SELECTORS_FROZEN",
            "depends_on": selector_freezes,
            "split_access": [],
        }
    )
    for name in config.backbones:
        tasks.append(
            {
                "task_id": f"test:{name}:native-common",
                "stage": "HELD_OUT_TEST_EVALUATION",
                "backbone": name,
                "depends_on": [
                    f"model:{name}:freeze",
                    f"selector:{name}:freeze",
                    "selectors:all:freeze",
                    "cohort:common-five:freeze",
                    "proposal:freeze",
                ],
                "split_access": ["test"],
            }
        )
    for task in tasks:
        task.update(
            {
                "science_started": False,
                "gpu_lock_acquired": False,
                "main_matrix_write": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": FIVE_BACKBONE_PLAN_SCHEMA,
        "status": "CONFIG_ONLY_BLOCKED_WAITING_MAIN_16_OF_16",
        "config": config.to_dict(),
        "proposal_contract": {
            "source": "bace_ours_main_candidate_pool",
            "same_candidate_identity_sha256_for_all_backbones": True,
            "chemllm_rerun": False,
            "ppo_rerun": False,
        },
        "cohort_contract": {
            "native": True,
            "common": (
                "intersection_of_gine_gin_gcn_gatv2_"
                "gatedgcn_plus_correct_parents"
            ),
            "common_is_primary": True,
        },
        "resource_receipt_required_per_backbone": [
            "total_parameters",
            "estimated_flops",
            "peak_vram_bytes",
            "training_gpu_hours",
        ],
        "graph_mamba": dict(config.graph_mamba_metadata),
        "tasks": tasks,
        "science_started": False,
        "gpu_lock_acquired": False,
        "main_matrix_modified": False,
    }
    payload["plan_sha256"] = stable_sha256(payload)
    return payload


def validate_proposal_fixed_runtime_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless one BACE train candidate universe is hash bound."""

    value = _mapping(payload, field="proposal manifest")
    sha = str(value.get("candidate_universe_sha256") or "").lower()
    if (
        value.get("dataset") != "bace"
        or value.get("method") != "ours"
        or value.get("source_split") != "train"
        or value.get("generation_per_backbone") is not False
        or value.get("calibration_loaded") is not False
        or value.get("test_loaded") is not False
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise FiveBackboneConfigError("proposal-fixed runtime manifest is not closed")
    return dict(value)


__all__ = [
    "FIVE_BACKBONES",
    "FIVE_BACKBONE_CONFIG_SCHEMA",
    "FIVE_BACKBONE_PLAN_SCHEMA",
    "FiveBackboneConfig",
    "FiveBackboneConfigError",
    "build_five_backbone_plan",
    "load_five_backbone_config",
    "validate_proposal_fixed_runtime_manifest",
]
