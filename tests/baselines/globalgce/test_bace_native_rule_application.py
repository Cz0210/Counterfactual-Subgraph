from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.baselines.globalgce_bace_native_rules import (
    GlobalGCENativeRule,
    GlobalGCENativeRuleError,
    apply_official_rule_tensors,
    apply_rule_to_parent,
    build_parent_native_tensors,
    enumerate_labeled_rule_matches,
    run_official_tensor_parity,
)
from src.data.molecular_graph_featurizer import (
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.eval.bace_globalgce_native_gine import (
    BACEGlobalGCEFrozenGINEForwardEvaluator,
    run_native_gine_forward_canary,
)


def _edge_rows(torch: object, maximum: int, labels: dict[tuple[int, int], int]) -> object:
    rows = torch.zeros((maximum * (maximum - 1) // 2, 4), dtype=torch.float32)
    rows[:, 0] = 1.0
    for (left, right), label in labels.items():
        high, low = max(left, right), min(left, right)
        position = (high - 1) * high // 2 + low
        rows[position, 0] = 0.0
        rows[position, label] = 1.0
    return rows


def _rule(*, add_node: bool = False, invalid_bond: bool = False) -> GlobalGCENativeRule:
    torch = pytest.importorskip("torch")
    maximum = 3 if add_node else 2
    lhs_feature = torch.zeros((maximum, 3), dtype=torch.float32)
    lhs_feature[:, 0] = 1.0
    lhs_feature[0] = torch.tensor([0.0, 1.0, 0.0])
    lhs_feature[1] = torch.tensor([0.0, 1.0, 0.0])
    lhs_adjacency = torch.zeros((maximum, maximum), dtype=torch.float32)
    lhs_adjacency[0, 1] = lhs_adjacency[1, 0] = 1.0
    lhs_edges = _edge_rows(torch, maximum, {(0, 1): 1})
    rhs_feature = lhs_feature.clone()
    rhs_feature[1] = torch.tensor([0.0, 0.0, 1.0])
    rhs_adjacency = lhs_adjacency.clone()
    rhs_labels = {(0, 1): 0 if invalid_bond else 1}
    if add_node:
        rhs_feature[2] = torch.tensor([0.0, 1.0, 0.0])
        rhs_adjacency[1, 2] = rhs_adjacency[2, 1] = 1.0
        rhs_labels[(1, 2)] = 1
    rhs_edges = _edge_rows(torch, maximum, rhs_labels)
    rule = GlobalGCENativeRule(
        rule_id="rule-add" if add_node else "rule-rewrite",
        native_rule_index=0,
        lhs_feature=lhs_feature,
        lhs_adjacency=lhs_adjacency,
        lhs_edge_attr=lhs_edges,
        rhs_feature=rhs_feature,
        rhs_adjacency=rhs_adjacency,
        rhs_edge_attr=rhs_edges,
        atom_symbols=("C", "O"),
        bond_names=("no_edge", "single", "double", "triple"),
    )
    rule.validate()
    return rule


def _official_reference(parent: object, rule: GlobalGCENativeRule, mapping: object) -> tuple:
    """Literal reference for generate_fs_mask + concatenate upstream writes."""

    torch = pytest.importorskip("torch")
    parent_nodes = int(parent.feature.shape[0])
    total = parent_nodes + rule.maximum_nodes - len(rule.lhs_nodes)
    feature = torch.zeros((total, parent.feature.shape[1]))
    feature[:, 0] = 1.0
    feature[:parent_nodes] = parent.feature
    adjacency = torch.zeros((total, total))
    adjacency[:parent_nodes, :parent_nodes] = parent.adjacency
    edge = torch.zeros((total * (total - 1) // 2, parent.edge_attr.shape[1]))
    edge[:, 0] = 1.0
    for left in range(parent_nodes):
        for right in range(left + 1, parent_nodes):
            old = (right - 1) * right // 2 + left
            edge[old] = parent.edge_attr[old]
    mask_order = tuple(mapping.keys()) + tuple(range(parent_nodes, total))
    for i, target in enumerate(mask_order):
        feature[target] = rule.rhs_feature[i]
    for i, left in enumerate(mask_order):
        for j, right in enumerate(mask_order):
            adjacency[left, right] = rule.rhs_adjacency[i, j]
    for i in range(rule.maximum_nodes):
        for j in range(i + 1, rule.maximum_nodes):
            source = (j - 1) * j // 2 + i
            left, right = sorted((mask_order[i], mask_order[j]))
            target = (right - 1) * right // 2 + left
            edge[target] = rule.rhs_edge_attr[source]
    return feature, adjacency, edge, mask_order


def test_official_tensor_application_parity_and_mask_order() -> None:
    torch = pytest.importorskip("torch")
    rule = _rule(add_node=True)
    parent = build_parent_native_tensors("CCC", atom_symbols=rule.atom_symbols)
    mapping = OrderedDict(((1, 0), (2, 1)))
    actual = apply_official_rule_tensors(parent, rule, mapping)
    expected = _official_reference(parent, rule, mapping)
    assert actual[3] == (1, 2, 3)
    for left, right in zip(actual[:3], expected[:3], strict=True):
        assert torch.equal(left, right)


def test_pinned_official_source_tensor_parity_when_checkout_is_available() -> None:
    configured = os.environ.get("GLOBALGCE_OFFICIAL_TEST_ROOT")
    root = Path(configured or "/private/tmp/globalgce-official-audit")
    if not root.is_dir():
        pytest.skip("pinned GlobalGCE checkout is not available in this environment")
    parity = run_official_tensor_parity(root)
    assert parity["status"] == "PASS"
    assert parity["official_function_loading"] == (
        "ast_extracted_from_hash_verified_source"
    )
    assert parity["comparisons"] == {
        "mask": True,
        "feature": True,
        "adjacency": True,
        "edge_attr": True,
        "boundary_attachment": True,
    }


def test_multiple_exact_subgraph_matches_are_retained() -> None:
    rule = _rule()
    parent = build_parent_native_tensors("CCC", atom_symbols=rule.atom_symbols)
    matches = enumerate_labeled_rule_matches(parent, rule)
    assert len(matches) >= 2
    assert len({tuple(row.items()) for row in matches}) == len(matches)
    applications = apply_rule_to_parent("CCC", rule)
    assert len(applications) == len(matches)
    assert any(row["valid"] for row in applications)
    assert applications == apply_rule_to_parent("CCC", rule)


def test_boundary_attachment_and_appended_node_are_preserved() -> None:
    rule = _rule(add_node=True)
    rows = apply_rule_to_parent("CCC", rule)
    valid = [row for row in rows if row["valid"]]
    assert valid
    assert all(row["boundary_attachments_preserved"] is True for row in valid)
    assert all(
        pair["bond_label_before"] == pair["bond_label_after"]
        for row in valid
        for pair in row["boundary_pairs"]
    )
    assert any(row["num_atoms"] == 4 for row in valid)
    assert all(row["connected"] and row["sanitized"] for row in valid)


def test_ambiguous_mapping_and_invalid_shapes_fail_closed() -> None:
    torch = pytest.importorskip("torch")
    rule = _rule()
    parent = build_parent_native_tensors("CCC", atom_symbols=rule.atom_symbols)
    with pytest.raises(GlobalGCENativeRuleError, match="bijection"):
        apply_official_rule_tensors(parent, rule, OrderedDict(((0, 0), (1, 0))))
    payload = rule.to_payload()
    payload["lhs_adjacency"] = [[0.0]]
    with pytest.raises(GlobalGCENativeRuleError, match="shape"):
        GlobalGCENativeRule.from_payload(payload)
    asymmetric = rule.to_payload()
    asymmetric["rhs_adjacency"][0][1] = 0.25
    asymmetric["rhs_adjacency"][1][0] = 0.75
    with pytest.raises(GlobalGCENativeRuleError, match="asymmetric"):
        GlobalGCENativeRule.from_payload(asymmetric)
    ambiguous_label = rule.to_payload()
    ambiguous_label["rhs_feature"][0] = [0.0, 0.5, 0.5]
    with pytest.raises(GlobalGCENativeRuleError, match="ambiguous hard label"):
        GlobalGCENativeRule.from_payload(ambiguous_label)
    ambiguous_edge = rule.to_payload()
    ambiguous_edge["rhs_adjacency"][0][1] = 0.5
    ambiguous_edge["rhs_adjacency"][1][0] = 0.5
    with pytest.raises(GlobalGCENativeRuleError, match="ambiguous 0.5"):
        GlobalGCENativeRule.from_payload(ambiguous_edge)


def test_adjacency_with_no_edge_label_fails_chemistry_closed() -> None:
    rule = _rule(invalid_bond=True)
    rows = apply_rule_to_parent("CCC", rule)
    assert rows
    assert all(row["valid"] is False for row in rows)
    assert all("no-edge bond label" in row["failure_reason"] for row in rows)


def test_blocked_training_contract_forbids_rf_gtgnn_and_ste() -> None:
    from src.baselines.bace_gnn_baseline_contracts import baseline_spec

    spec = baseline_spec("GlobalGCE")
    assert spec.native_route_available is False
    assert spec.blocker_code == (
        "BLOCKED_GLOBALGCE_FROZEN_GINE_DIFFERENTIABLE_RULE_TRAINING_UNAVAILABLE"
    )
    text = str(spec.blocker_reason).lower()
    assert "categorical long" in text
    assert "gtgnn/rf" in text
    assert "straight-through" in text


class _FakeBACEGINE:
    checkpoint_id = "frozen-bace-gine"
    backbone = "gine"
    num_classes = 2
    source_label = 1
    temperature = 1.25

    def predict_records(self, graphs: object, *, batch_size: int | None = None) -> list[dict]:
        del batch_size
        rows = []
        for graph in graphs:
            # The native fixture changes one C into O.  Use this deterministic
            # fake solely to prove before/after batching and strict-flip wiring.
            probabilities = np.asarray(
                [0.9, 0.1] if "O" in graph.smiles else [0.1, 0.9],
                dtype=np.float64,
            )
            rows.append(
                {
                    "predicted_label": int(probabilities.argmax()),
                    "probabilities": probabilities.tolist(),
                    "logits": np.log(probabilities).tolist(),
                    "checkpoint_id": self.checkpoint_id,
                    "backbone": self.backbone,
                    "num_classes": self.num_classes,
                    "source_label": self.source_label,
                    "temperature": self.temperature,
                }
            )
        return rows


def _forward_evaluator() -> BACEGlobalGCEFrozenGINEForwardEvaluator:
    return BACEGlobalGCEFrozenGINEForwardEvaluator(
        oracle=_FakeBACEGINE(),
        featurizer=MolecularGraphFeaturizer(default_molecular_feature_schema()),
        provenance={
            "oracle_backend": "gnn",
            "classifier_family": "gine",
            "rf_oracle_used": False,
            "source_label": 1,
            "num_classes": 2,
            "oracle_checkpoint_hash": "frozen-bace-gine",
        },
        oracle_batch_size=32,
    )


def test_native_applications_are_scored_by_same_frozen_gine_forward() -> None:
    payload = _forward_evaluator().score_parent_rule(
        parent_id="bace-parent-1",
        parent_smiles="CCC",
        rule=_rule(),
    )
    assert payload["status"] == "PASS"
    assert payload["test_loaded"] is False
    assert payload["application_count"] >= 2
    assert payload["gnn_scored_application_count"] >= 1
    assert payload["strict_flip_count"] >= 1
    for row in payload["applications"]:
        assert row["action_kind"] == "lhs_rhs_graph_transformation_rule"
        assert row["oracle_backend"] == "gnn"
        assert row["classifier_family"] == "gine"
        assert row["rf_oracle_used"] is False
        if row["gnn_scored"]:
            assert row["pred_before"] == 1
            assert row["pred_after"] == 0
            assert row["cf_flip"] is True


def test_native_gine_forward_rejects_rf_and_test_before_freeze() -> None:
    bad_oracle = _FakeBACEGINE()
    bad_oracle.backbone = "random_forest"
    with pytest.raises(ValueError, match="non-frozen-GINE"):
        BACEGlobalGCEFrozenGINEForwardEvaluator(
            oracle=bad_oracle,
            featurizer=MolecularGraphFeaturizer(default_molecular_feature_schema()),
            provenance={
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
                "source_label": 1,
                "num_classes": 2,
                "oracle_checkpoint_hash": "frozen-bace-gine",
            },
        )
    evaluator = _forward_evaluator()
    with pytest.raises(ValueError, match="already-frozen selector"):
        evaluator.score_parent_rule(
            parent_id="bace-parent-1",
            parent_smiles="CCC",
            rule=_rule(),
            split_role="test",
        )


def test_forward_canary_publishes_pass_without_releasing_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(json.dumps(_rule().to_payload()), encoding="utf-8")
    evaluator = _forward_evaluator()
    monkeypatch.setattr(
        BACEGlobalGCEFrozenGINEForwardEvaluator,
        "from_checkpoint",
        classmethod(lambda cls, *_args, **_kwargs: evaluator),
    )
    output = tmp_path / "canary"
    manifest = run_native_gine_forward_canary(
        parent_id="bace-parent-1",
        parent_smiles="CCC",
        rule_json=rule_path,
        gnn_checkpoint=tmp_path / "unused-checkpoint",
        output_dir=output,
    )
    assert manifest["exact_frozen_gine_forward_status"] == "PASS"
    assert manifest["full_rule_training_released"] is False
    assert manifest["full_rule_training_status"] == "BLOCKED_CODE"
    assert (output / "FORWARD_EVAL_PASS").read_text().strip() == "FORWARD_EVAL_PASS"
    assert not (output / "PASS").exists()
