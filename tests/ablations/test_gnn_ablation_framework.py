from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from scripts.ablations.gnn.build_gnn_ablation_plan import (
    build_plan_document,
    write_plan_document,
)
from src.ablations.gnn import framework as framework_module
from src.ablations.gnn import (
    CohortFreeze,
    GNNAblationConfigError,
    GNNAblationContractError,
    ParentPrediction,
    ProposalCandidateIdentity,
    ProposalUniverse,
    build_ablation_plan_from_config,
    build_cohort_split_authority,
    freeze_common_and_native_cohorts,
    load_ablation_config,
)
from src.ablations.output_schema import output_inventory
from src.data.molecular_graph_featurizer import default_molecular_feature_schema
from src.models import gnn_backbone_registry as backbone_registry
from src.models.gnn_backbone_registry import get_gnn_backbone_spec
from src.utils.env import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs/ablations/gnn"
BACKBONES = ("gine", "gin", "gcn", "gatv2")
FEATURE_SCHEMA = default_molecular_feature_schema()
FEATURE_SHA = FEATURE_SCHEMA.to_dict()["schema_sha256"]


def _sha(character: str) -> str:
    return character * 64


def _candidate(
    index: int, *, rule: str, fragment: str, source: str = "a"
) -> ProposalCandidateIdentity:
    return ProposalCandidateIdentity(
        dataset="bace",
        proposal_index=index,
        proposal_source_sha256=_sha(source),
        rule_id=f"rule-{rule}",
        rule_sha256=_sha(rule),
        fragment_graph_sha256=_sha(fragment),
    )


def _universe() -> ProposalUniverse:
    return ProposalUniverse.freeze(
        [
            _candidate(0, rule="1", fragment="3"),
            _candidate(1, rule="2", fragment="4"),
        ]
    )


def _predictions(
    parent_ids: tuple[str, ...],
    native: dict[str, set[str]],
    *,
    split: str,
) -> list[ParentPrediction]:
    result: list[ParentPrediction] = []
    graph_hashes = {parent: _sha(str(5 + index)) for index, parent in enumerate(parent_ids)}
    for offset, backbone in enumerate(BACKBONES):
        for parent_id in parent_ids:
            result.append(
                ParentPrediction(
                    dataset="bace",
                    split=split,
                    split_sha256=_sha("9" if split == "calibration" else "8"),
                    feature_schema_sha256=FEATURE_SHA,
                    temperature_scaling_sha256=_sha(str(offset + 1)),
                    backbone=backbone,
                    edge_feature_mode=get_gnn_backbone_spec(backbone).edge_feature_mode,
                    checkpoint_sha256=_sha(chr(ord("a") + offset)),
                    parent_id=parent_id,
                    parent_graph_sha256=graph_hashes[parent_id],
                    true_label=1,
                    predicted_label=1 if parent_id in native[backbone] else 0,
                    source_label=1,
                    num_classes=2,
                )
            )
    return result


def _cohort_authority(
    parent_ids: tuple[str, ...], *, split: str
) -> dict[str, object]:
    values: dict[str, object] = {
        "expected_dataset": "bace",
        "expected_split_sha256": _sha(
            "9" if split == "calibration" else "8"
        ),
        "expected_feature_schema_sha256": FEATURE_SHA,
        "expected_checkpoint_sha256s": {
            backbone: _sha(chr(ord("a") + offset))
            for offset, backbone in enumerate(BACKBONES)
        },
        "expected_temperature_scaling_sha256s": {
            backbone: _sha(str(offset + 1))
            for offset, backbone in enumerate(BACKBONES)
        },
        "expected_parent_graph_sha256s": {
            parent: _sha(str(5 + index))
            for index, parent in enumerate(parent_ids)
        },
        "expected_true_labels": {parent: 1 for parent in parent_ids},
    }
    values["split_authority"] = build_cohort_split_authority(
        dataset=values["expected_dataset"],
        split=split,
        split_sha256=values["expected_split_sha256"],
        feature_schema_sha256=values["expected_feature_schema_sha256"],
        backbones=BACKBONES,
        checkpoint_sha256s=values["expected_checkpoint_sha256s"],
        temperature_scaling_sha256s=values[
            "expected_temperature_scaling_sha256s"
        ],
        parent_ids=parent_ids,
        parent_graph_sha256s=values["expected_parent_graph_sha256s"],
        true_labels=values["expected_true_labels"],
    )
    return values


def _closure(tasks: dict[str, dict[str, object]], task_id: str) -> set[str]:
    result: set[str] = set()
    stack = list(tasks[task_id]["depends_on"])
    while stack:
        current = str(stack.pop())
        if current in result:
            continue
        result.add(current)
        stack.extend(tasks[current]["depends_on"])
    return result


@pytest.mark.parametrize(
    ("name", "dataset", "num_classes"),
    (("bace.yaml", "bace", 2), ("tastemolnet.yaml", "tastemolnet", 3)),
)
def test_checked_in_configs_build_complete_config_only_plans(
    name: str,
    dataset: str,
    num_classes: int,
) -> None:
    config = load_ablation_config(CONFIG_ROOT / name, project_root=PROJECT_ROOT)
    plan = build_ablation_plan_from_config(
        CONFIG_ROOT / name, project_root=PROJECT_ROOT
    ).to_dict()

    assert config.dataset == dataset
    assert config.num_classes == num_classes
    assert config.backbone_names == BACKBONES
    assert config.shared_feature_schema_sha256 == FEATURE_SHA
    assert config.evaluation_policy["distance_metric"] == "WNode"
    assert config.evaluation_policy["wnode_shared_across_backbones"] is True
    assert plan["status"] == "PLANNED_NOT_RUN"
    assert plan["execution_mode"] == "CONFIG_ONLY"
    assert plan["science_executed"] is False
    assert plan["autodl_modified"] is False
    assert plan["main_matrix_modified"] is False
    assert len(plan["tasks"]) == 51
    assert all(task["launches_science"] is False for task in plan["tasks"])
    assert all(task["writes_autodl"] is False for task in plan["tasks"])
    assert all(task["writes_main_matrix"] is False for task in plan["tasks"])


def test_train_rule_identity_is_parent_and_backbone_independent_and_strict() -> None:
    first = _candidate(0, rule="1", fragment="3")
    same_rule_new_universe = ProposalCandidateIdentity(
        dataset="bace",
        proposal_index=99,
        proposal_source_sha256=_sha("b"),
        rule_id="different-row-id",
        rule_sha256=first.rule_sha256,
        fragment_graph_sha256=first.fragment_graph_sha256,
    )
    assert first.candidate_id == same_rule_new_universe.candidate_id
    assert "parent_id" not in first.semantic_payload
    assert "backbone" not in first.semantic_payload
    assert ProposalCandidateIdentity.from_mapping(first.to_dict()) == first

    tampered = first.to_dict()
    tampered["candidate_id"] = "gnnrule-" + _sha("f")
    with pytest.raises(GNNAblationContractError, match="differs from recomputed"):
        ProposalCandidateIdentity.from_mapping(tampered)
    with pytest.raises(GNNAblationContractError, match="unknown fields"):
        ProposalCandidateIdentity.from_mapping({**first.to_dict(), "mystery": 1})
    with pytest.raises(GNNAblationContractError, match="classifier/cohort"):
        ProposalCandidateIdentity.from_mapping({**first.to_dict(), "backbone": "gine"})
    with pytest.raises(GNNAblationContractError, match="train split"):
        replace(first, source_split="calibration")


def test_proposal_universe_is_one_ordered_train_rule_pool() -> None:
    second = _candidate(1, rule="2", fragment="4")
    first = _candidate(0, rule="1", fragment="3")
    universe = ProposalUniverse.freeze([second, first])

    assert universe.ordered_rule_ids == ("rule-1", "rule-2")
    manifest = universe.to_manifest()
    assert manifest["proposal_source_split"] == "train"
    assert manifest["generation_per_backbone"] is False
    assert manifest["calibration_or_test_parent_in_candidate_identity"] is False

    with pytest.raises(GNNAblationContractError, match="duplicate rule ids"):
        ProposalUniverse.freeze(
            [first, replace(second, rule_id=first.rule_id)]
        )
    with pytest.raises(GNNAblationContractError, match="duplicate structural rules"):
        ProposalUniverse.freeze(
            [
                first,
                replace(
                    second,
                    rule_id="different-row-id",
                    rule_sha256=first.rule_sha256,
                    fragment_graph_sha256=first.fragment_graph_sha256,
                ),
            ]
        )


@pytest.mark.parametrize("split", ("calibration", "test"))
def test_common_native_cohorts_bind_split_schema_temperature_and_rule_pool(
    split: str,
) -> None:
    universe = _universe()
    parents = ("p1", "p2", "p3")
    native = {
        "gine": {"p1", "p2"},
        "gin": {"p1", "p3"},
        "gcn": {"p1", "p2", "p3"},
        "gatv2": {"p1"},
    }
    frozen = freeze_common_and_native_cohorts(
        universe=universe,
        predictions=_predictions(parents, native, split=split),
        backbones=BACKBONES,
        split=split,
        expected_parent_ids=parents,
        **_cohort_authority(parents, split=split),
    ).to_manifest()

    assert frozen["split"] == split
    assert frozen["feature_schema_sha256"] == FEATURE_SHA
    assert frozen["true_labels_consistent_across_backbones"] is True
    assert frozen["common_cohort"]["parent_ids"] == ["p1"]
    assert frozen["common_cohort"]["candidate_ids"] == list(
        universe.ordered_candidate_ids
    )
    assert frozen["common_cohort"]["expected_application_count"] == 2
    assert set(frozen["temperature_scaling_sha256s"]) == set(BACKBONES)
    assert frozen["native_cohorts"]["gine"]["parent_ids"] == ["p1", "p2"]


def test_cohort_freeze_fails_on_label_split_schema_or_coverage_drift() -> None:
    universe = _universe()
    parents = ("p1", "p2")
    native = {backbone: set(parents) for backbone in BACKBONES}
    predictions = _predictions(parents, native, split="calibration")
    kwargs = {
        "universe": universe,
        "backbones": BACKBONES,
        "split": "calibration",
        "expected_parent_ids": parents,
        **_cohort_authority(parents, split="calibration"),
    }
    with pytest.raises(GNNAblationContractError, match="coverage"):
        freeze_common_and_native_cohorts(predictions=predictions[:-1], **kwargs)

    label_drift = list(predictions)
    index = next(
        i
        for i, row in enumerate(label_drift)
        if row.backbone == "gcn" and row.parent_id == "p1"
    )
    label_drift[index] = replace(label_drift[index], true_label=0)
    with pytest.raises(GNNAblationContractError, match="true label differs"):
        freeze_common_and_native_cohorts(predictions=label_drift, **kwargs)

    split_drift = list(predictions)
    split_drift[0] = replace(split_drift[0], split_sha256=_sha("7"))
    with pytest.raises(GNNAblationContractError, match="split SHA"):
        freeze_common_and_native_cohorts(predictions=split_drift, **kwargs)

    schema_drift = list(predictions)
    schema_drift[0] = replace(schema_drift[0], feature_schema_sha256=_sha("7"))
    with pytest.raises(GNNAblationContractError, match="shared schema"):
        freeze_common_and_native_cohorts(predictions=schema_drift, **kwargs)

    stale_checkpoint = dict(kwargs)
    stale_checkpoint["expected_checkpoint_sha256s"] = {
        **kwargs["expected_checkpoint_sha256s"],
        "gine": _sha("f"),
    }
    with pytest.raises(GNNAblationContractError, match="self-hashed split authority"):
        freeze_common_and_native_cohorts(
            predictions=predictions, **stale_checkpoint
        )

    serialized = freeze_common_and_native_cohorts(
        predictions=predictions, **kwargs
    ).to_manifest()
    serialized["authoritative_parent_ids_sha256"] = _sha("f")
    with pytest.raises(GNNAblationContractError, match="freeze SHA"):
        CohortFreeze(payload=serialized)


def test_dag_freezes_train_rules_then_calibration_selectors_then_test() -> None:
    plan = build_ablation_plan_from_config(
        CONFIG_ROOT / "bace.yaml", project_root=PROJECT_ROOT
    ).to_dict()
    tasks = {task["task_id"]: task for task in plan["tasks"]}
    freeze_ids = {f"model:{backbone}:freeze" for backbone in BACKBONES}
    selector_ids = {f"selector:{backbone}:freeze" for backbone in BACKBONES}
    proposal = tasks["proposal:train-rules:freeze"]
    assert proposal["split_access"] == ["train"]
    assert set(proposal["depends_on"]) == freeze_ids

    calibration_grid = {
        (task["backbone"], task["cohort_kind"])
        for task in tasks.values()
        if task["stage"] == "CALIBRATION_NATIVE_COMMON_WNODE"
    }
    test_grid = {
        (task["backbone"], task["cohort_kind"])
        for task in tasks.values()
        if task["stage"] == "HELD_OUT_TEST_NATIVE_COMMON_EVALUATION"
    }
    expected_grid = {
        (backbone, cohort)
        for backbone in BACKBONES
        for cohort in ("common", "native")
    }
    assert calibration_grid == expected_grid
    assert test_grid == expected_grid
    for backbone in BACKBONES:
        assert set(tasks[f"selector:{backbone}:freeze"]["depends_on"]) == {
            f"calibration:{backbone}:common:evaluate",
            f"calibration:{backbone}:native:evaluate",
        }
    for task in tasks.values():
        if "test" in task["split_access"]:
            closure = _closure(tasks, str(task["task_id"]))
            assert freeze_ids.issubset(closure)
            assert selector_ids.issubset(closure)
    assert all(
        "test" not in task["split_access"]
        for task in tasks.values()
        if task["stage"] != "HELD_OUT_TEST_PARENT_SCORE"
        and task["stage"] != "HELD_OUT_TEST_NATIVE_COMMON_EVALUATION"
    )


@pytest.mark.parametrize("name", ("bace.yaml", "tastemolnet.yaml"))
def test_output_contract_matches_real_loader_bundle_inventory(name: str) -> None:
    config = load_ablation_config(CONFIG_ROOT / name, project_root=PROJECT_ROOT)
    plan = build_ablation_plan_from_config(
        CONFIG_ROOT / name, project_root=PROJECT_ROOT
    ).to_dict()
    output = plan["output_contract"]
    required = output["artifacts"]["model_bundle"]["required_files"]
    assert required == list(backbone_registry.required_backbone_bundle_files(config.dataset))
    assert output["shared_feature_schema_sha256"] == FEATURE_SHA
    temperature = output["artifacts"]["model_bundle"]["temperature_contract"]
    assert temperature["status"] == "fit"
    assert temperature["selection_split"] == "validation"
    assert temperature["test_used_for_fit"] is False
    assert set(temperature["required_provenance"]) == {
        "dataset",
        "validation_split_sha256",
        "ordered_parent_ids_sha256",
        "ordered_labels_sha256",
        "selected_checkpoint_sha256",
        "feature_schema_sha256",
    }
    assert temperature["self_hash_required"] is True
    assert temperature["validation_predictions_binding_required"] is True
    assert output["artifacts"]["variant_output_inventory"]["files"] == list(
        output_inventory("gnn")
    )
    assert output["artifacts"]["aggregate_output_inventory"]["files"] == list(
        output_inventory("gnn", aggregate=True)
    )
    if config.dataset == "tastemolnet":
        assert "checkpoint_reload.json" in required
        assert "data_use_policy_binding.json" in required


def test_final_manifest_binds_schema_temperature_edge_cohorts_and_selectors() -> None:
    plan = build_ablation_plan_from_config(
        CONFIG_ROOT / "bace.yaml", project_root=PROJECT_ROOT
    ).to_dict()
    final = plan["final_manifest_plan"]
    for required in (
        "shared_feature_schema_sha256",
        "temperature_scaling_sha256_by_backbone",
        "edge_feature_mode_by_backbone",
        "wnode_config_sha256",
        "train_rule_proposal_universe_sha256",
        "calibration_cohort_freeze_sha256",
        "selector_freeze_sha256_by_backbone",
        "test_cohort_freeze_sha256",
    ):
        assert required in final["required_bindings"]
    assert final["required_disclosures"]["proposal_source_split"] == "train"
    assert final["required_disclosures"]["selector_fit_split"] == "calibration"
    assert final["required_disclosures"]["test_used_for_fit_selection_or_temperature"] is False
    assert final["publication_order"][-1] == "PASS"


def test_taste_config_preserves_policy_v2_without_license_pass_claim() -> None:
    plan = build_ablation_plan_from_config(
        CONFIG_ROOT / "tastemolnet.yaml", project_root=PROJECT_ROOT
    ).to_dict()
    serialized = json.dumps(plan, sort_keys=True)
    policy = plan["config"]["data_policy"]
    assert policy["mode"] == "scoped_research_no_redistribution_v2"
    assert policy["policy_receipt_required"] is True
    assert policy["upstream_terms_status"] == "NOT_EXPLICITLY_STATED"
    assert policy["redistribution_allowed"] is False
    assert "LICENSE_PASS" not in serialized


def test_registry_builds_checked_in_model_config_with_shared_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return "model-with-forward"

    import src.models.molecular_gnn as molecular_gnn

    monkeypatch.setattr(molecular_gnn, "build_molecular_gnn", fake_builder)
    model = backbone_registry.build_backbone(
        "gatv2",
        load_yaml_config(PROJECT_ROOT / "configs/gnn/gatv2.yaml"),
        feature_schema=FEATURE_SCHEMA,
        expected_feature_schema_sha256=FEATURE_SHA,
        num_classes=2,
    )
    assert model == "model-with-forward"
    assert captured["backbone"] == "gatv2"
    assert captured["num_classes"] == 2
    assert captured["node_feature_schema"] is FEATURE_SCHEMA
    assert captured["edge_feature_schema"] is FEATURE_SCHEMA


class _FakeConfig:
    backbone = "gine"


class _FakeModel:
    config = _FakeConfig()


def _valid_temperature() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "temperature_scaling_v1",
        "status": "fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "temperature": 1.25,
        "num_examples": 1,
        "num_classes": 2,
        "nll_before": 0.5,
        "nll_after": 0.4,
        "ece_before": 0.2,
        "ece_after": 0.1,
        "brier_before": 0.2,
        "brier_after": 0.1,
        "argmax_invariant": True,
        "dataset": "bace",
        "validation_split_sha256": _sha("9"),
        "ordered_parent_ids_sha256": backbone_registry._canonical_sha256(
            {"ordered_parent_ids": ["p1"]}
        ),
        "ordered_labels_sha256": backbone_registry._canonical_sha256(
            {"ordered_true_labels": [1]}
        ),
        "selected_checkpoint_sha256": _sha("a"),
        "feature_schema_sha256": FEATURE_SHA,
    }
    payload["temperature_contract_sha256"] = backbone_registry._canonical_sha256(
        payload
    )
    return payload


def _model_card() -> dict[str, object]:
    return {
        "dataset": "bace",
        "selected_checkpoint_sha256": _sha("a"),
        "backbone": "gine",
        "edge_feature_mode": get_gnn_backbone_spec("gine").edge_feature_mode,
        "feature_schema_sha256": FEATURE_SHA,
    }


def _split_manifest() -> dict[str, object]:
    return {
        "dataset": "bace",
        "files": {"validation": {"sha256": _sha("9")}},
    }


def _validation_predictions() -> list[dict[str, object]]:
    return [{"molecule_id": "p1", "label": 1}]


def test_registry_save_and_load_close_temperature_schema_and_edge_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.oracles.gnn_oracle as oracle

    saved: dict[str, object] = {}

    def fake_save(**kwargs):
        saved.update(kwargs)
        return {"status": "saved", "checkpoint_id": _sha("a")}

    monkeypatch.setattr(oracle, "save_gnn_checkpoint_bundle", fake_save)
    result = backbone_registry.save_backbone_bundle(
        model=_FakeModel(),
        feature_schema=FEATURE_SCHEMA,
        temperature_scaling=_valid_temperature(),
        model_card=_model_card(),
        split_manifest=_split_manifest(),
        validation_predictions=_validation_predictions(),
        expected_backbone="gine",
        expected_feature_schema_sha256=FEATURE_SHA,
    )
    assert result == {"status": "saved", "checkpoint_id": _sha("a")}
    assert saved["feature_schema"] is FEATURE_SCHEMA

    monkeypatch.setattr(
        oracle,
        "save_gnn_checkpoint_bundle",
        lambda **_kwargs: {"status": "saved", "checkpoint_id": _sha("b")},
    )
    with pytest.raises(ValueError, match="actual model.pt"):
        backbone_registry.save_backbone_bundle(
            model=_FakeModel(),
            feature_schema=FEATURE_SCHEMA,
            temperature_scaling=_valid_temperature(),
            model_card=_model_card(),
            split_manifest=_split_manifest(),
            validation_predictions=_validation_predictions(),
        )
    monkeypatch.setattr(oracle, "save_gnn_checkpoint_bundle", fake_save)

    with pytest.raises(RuntimeError, match="BLOCKED_UNIMPLEMENTED_FULL_CLOSURE"):
        backbone_registry.save_backbone_bundle(
            model=_FakeModel(),
            feature_schema=FEATURE_SCHEMA,
            temperature_scaling=_valid_temperature(),
            model_card={**_model_card(), "dataset": "tastemolnet", "profile": "full"},
            split_manifest=_split_manifest(),
            validation_predictions=_validation_predictions(),
        )

    invalid_temperature = _valid_temperature()
    invalid_temperature["selection_split"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        backbone_registry.save_backbone_bundle(
            model=_FakeModel(),
            feature_schema=FEATURE_SCHEMA,
            temperature_scaling=invalid_temperature,
            model_card=_model_card(),
            split_manifest=_split_manifest(),
            validation_predictions=_validation_predictions(),
        )

    with pytest.raises(ValueError, match="may not defer"):
        backbone_registry.save_backbone_bundle(
            model=_FakeModel(),
            feature_schema=FEATURE_SCHEMA,
            temperature_scaling=_valid_temperature(),
            model_card=_model_card(),
            split_manifest=_split_manifest(),
            validation_predictions=_validation_predictions(),
            defer_tastemolnet_closure=True,
        )

    checkpoint = tmp_path / "bundle"
    checkpoint.mkdir()
    (checkpoint / "validation_predictions.csv").write_text(
        "molecule_id,label\np1,1\n", encoding="utf-8"
    )

    def fake_load(*_args, **_kwargs):
        return _FakeModel(), {
            "checkpoint_id": _sha("a"),
            "feature_schema": FEATURE_SCHEMA,
            "temperature_scaling": _valid_temperature(),
            "model_card": _model_card(),
            "split_manifest": _split_manifest(),
        }

    monkeypatch.setattr(oracle, "load_gnn_checkpoint_bundle", fake_load)
    _model, metadata = backbone_registry.load_backbone_bundle(
        checkpoint,
        expected_backbone="gine",
        expected_feature_schema_sha256=FEATURE_SHA,
    )
    assert metadata["edge_feature_mode"] == get_gnn_backbone_spec("gine").edge_feature_mode

    def wrong_checkpoint_load(*_args, **_kwargs):
        _model, loaded = fake_load()
        loaded["checkpoint_id"] = _sha("b")
        return _model, loaded

    monkeypatch.setattr(oracle, "load_gnn_checkpoint_bundle", wrong_checkpoint_load)
    with pytest.raises(ValueError, match="actual model.pt"):
        backbone_registry.load_backbone_bundle(checkpoint)

    def bad_edge_load(*_args, **_kwargs):
        model, metadata = fake_load()
        metadata["model_card"] = dict(metadata["model_card"])
        metadata["model_card"]["edge_feature_mode"] = "wrong"
        return model, metadata

    monkeypatch.setattr(oracle, "load_gnn_checkpoint_bundle", bad_edge_load)
    with pytest.raises(ValueError, match="edge/schema disclosure"):
        backbone_registry.load_backbone_bundle(checkpoint)
    with pytest.raises(ValueError, match="hash verification"):
        backbone_registry.load_backbone_bundle(checkpoint, verify_hashes=False)
    with pytest.raises(ValueError, match="Taste closure"):
        backbone_registry.load_backbone_bundle(
            checkpoint, require_taste_closure=False
        )


def test_backbone_temperature_gate_rejects_non_validation_split() -> None:
    with pytest.raises(ValueError, match="validation-only"):
        backbone_registry.fit_backbone_temperature(
            [],
            [],
            split="test",
            dataset="bace",
            validation_split_sha256=_sha("9"),
            ordered_parent_ids=(),
            selected_checkpoint_sha256=_sha("a"),
            feature_schema_sha256=FEATURE_SHA,
        )


def test_backbone_temperature_fit_adds_hash_closed_validation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.oracles.gnn_oracle as oracle

    base = {
        key: value
        for key, value in _valid_temperature().items()
        if key in backbone_registry._TEMPERATURE_BASE_FIELDS
    }
    monkeypatch.setattr(
        oracle, "fit_temperature_scaling", lambda *_args, **_kwargs: dict(base)
    )
    result = backbone_registry.fit_backbone_temperature(
        [[0.0, 1.0]],
        [1],
        split="validation",
        dataset="bace",
        validation_split_sha256=_sha("9"),
        ordered_parent_ids=("p1",),
        selected_checkpoint_sha256=_sha("a"),
        feature_schema_sha256=FEATURE_SHA,
    )
    assert result["dataset"] == "bace"
    assert result["test_used_for_fit"] is False
    assert len(result["temperature_contract_sha256"]) == 64

    result["unknown_default"] = True
    with pytest.raises(ValueError, match="closed schema"):
        backbone_registry._validate_temperature_contract(result)


def test_config_rejects_per_backbone_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = framework_module.load_yaml_config
    invalid = original(CONFIG_ROOT / "bace.yaml")
    invalid["experiment"]["generation_per_backbone"] = True

    def fake_load(path: Path):
        if Path(path).resolve() == (CONFIG_ROOT / "bace.yaml").resolve():
            return invalid
        return original(path)

    monkeypatch.setattr(framework_module, "load_yaml_config", fake_load)
    with pytest.raises(GNNAblationConfigError, match="may not vary by backbone"):
        load_ablation_config(CONFIG_ROOT / "bace.yaml", project_root=PROJECT_ROOT)


def test_config_rejects_unknown_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    original = framework_module.load_yaml_config
    invalid = original(CONFIG_ROOT / "bace.yaml")
    invalid["unknown_future_default"] = True

    def fake_load(path: Path):
        if Path(path).resolve() == (CONFIG_ROOT / "bace.yaml").resolve():
            return invalid
        return original(path)

    monkeypatch.setattr(framework_module, "load_yaml_config", fake_load)
    with pytest.raises(GNNAblationConfigError, match="unknown"):
        load_ablation_config(CONFIG_ROOT / "bace.yaml", project_root=PROJECT_ROOT)


def test_config_only_cli_document_and_paired_slurm(tmp_path: Path) -> None:
    document = build_plan_document(
        ablation_config=CONFIG_ROOT / "bace.yaml",
        runtime_config=PROJECT_ROOT / "configs/hpc.yaml",
    )
    output = tmp_path / "gnn-plan.json"
    write_plan_document(output, document)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["status"] == "PLANNED_NOT_RUN"
    assert stored["science_executed"] is False
    assert stored["autodl_modified"] is False
    assert stored["main_matrix_modified"] is False
    assert all("command" not in task for task in stored["plan"]["tasks"])
    assert len(stored["document_sha256"]) == 64

    with pytest.raises(ValueError, match="AutoDL or main-matrix"):
        write_plan_document(
            Path("/autodl-fs/data/paper_matrix/gnn-plan.json"), document
        )

    slurm = (
        PROJECT_ROOT / "scripts/slurm/build_gnn_ablation_plan.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "conda activate smiles_pip118",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
    ):
        assert required in slurm
