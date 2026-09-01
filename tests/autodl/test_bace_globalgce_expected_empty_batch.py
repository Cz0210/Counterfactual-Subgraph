from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.data.molecular_graph_dataset import MolecularGraphBatch
from src.eval import bace_native_baseline_gnn as native
from src.oracles.gnn_oracle import (
    GNNOracle,
    UNEXPECTED_EMPTY_GRAPH_SEQUENCE,
)


def _unloaded_oracle(*, num_classes: int = 3) -> GNNOracle:
    """Build only the state needed to prove the pre-model empty contract."""

    oracle = object.__new__(GNNOracle)
    oracle.num_classes = num_classes
    oracle.temperature = 1.25
    return oracle


def test_gnn_oracle_default_empty_sequence_remains_fail_closed() -> None:
    oracle = _unloaded_oracle()

    with pytest.raises(ValueError, match="cannot predict an empty graph sequence"):
        oracle.predict_logits([])
    with pytest.raises(ValueError, match="cannot predict an empty graph sequence"):
        oracle.predict_proba([], allow_empty=True)
    with pytest.raises(ValueError, match="cannot predict an empty graph sequence"):
        oracle.predict_label([], expected_count=0)


def test_gnn_oracle_explicit_expected_empty_returns_typed_shapes() -> None:
    oracle = _unloaded_oracle(num_classes=3)

    logits = oracle.predict_logits([], allow_empty=True, expected_count=0)
    probabilities = oracle.predict_proba([], allow_empty=True, expected_count=0)
    predictions = oracle.predict_label([], allow_empty=True, expected_count=0)

    assert logits.shape == (0, 3)
    assert probabilities.shape == (0, 3)
    assert predictions.shape == (0,)
    assert logits.dtype == np.float64
    assert probabilities.dtype == np.float64
    assert predictions.dtype == np.int64
    assert oracle.predict_records([], allow_empty=True, expected_count=0) == []

    empty_batch = MolecularGraphBatch(
        x=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
        y=None,
        molecule_ids=(),
        smiles=(),
        splits=(),
        graph_sha256s=(),
    )
    assert oracle.predict_logits(
        empty_batch, allow_empty=True, expected_count=0
    ).shape == (0, 3)


def test_gnn_oracle_rejects_empty_when_expected_count_is_positive() -> None:
    oracle = _unloaded_oracle(num_classes=2)

    with pytest.raises(ValueError, match=UNEXPECTED_EMPTY_GRAPH_SEQUENCE):
        oracle.predict_logits([], allow_empty=True, expected_count=1)


class _BombOracle:
    num_classes = 2

    def predict_records(self, _graphs: Any, *, batch_size: int) -> list[dict[str, Any]]:
        raise AssertionError(f"oracle must not be called; batch_size={batch_size}")


def test_caller_bypasses_oracle_for_independently_expected_empty_batch() -> None:
    batch = native._predict_expected_graph_batch(
        oracle=_BombOracle(),
        graphs=[],
        expected_count=0,
        oracle_batch_size=256,
    )

    assert batch["records"] == []
    assert batch["logits"].shape == (0, 2)
    assert batch["probabilities"].shape == (0, 2)
    assert batch["predictions"].shape == (0,)
    assert batch["oracle_called"] is False
    assert batch["reason"] == "NO_EVALUABLE_GRAPHS_AFTER_PRE_ORACLE_FILTERS"
    assert native._prediction_batch_receipt(batch) == {
        "expected_count": 0,
        "actual_count": 0,
        "oracle_called": False,
        "reason": "NO_EVALUABLE_GRAPHS_AFTER_PRE_ORACLE_FILTERS",
        "logits_shape": [0, 2],
        "probabilities_shape": [0, 2],
        "predictions_shape": [0],
    }


def test_caller_rejects_unexpected_empty_batch_before_oracle() -> None:
    with pytest.raises(RuntimeError, match=UNEXPECTED_EMPTY_GRAPH_SEQUENCE):
        native._predict_expected_graph_batch(
            oracle=_BombOracle(),
            graphs=[],
            expected_count=1,
            oracle_batch_size=256,
        )


def test_globalgce_zero_application_shard_records_explicit_empty_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_rule = SimpleNamespace(rule_id="rule-1", native_rule_index=7)
    monkeypatch.setattr(
        native,
        "GlobalGCENativeRule",
        SimpleNamespace(from_payload=lambda _payload: fake_rule),
    )
    monkeypatch.setattr(native, "apply_rule_to_parent", lambda *_args: [])
    spec = SimpleNamespace(
        method="GlobalGCE",
        action_kind="lhs_rhs_graph_transformation_rule",
        action_semantics="native_globalgce_rule_application",
    )

    rows, receipt = native._globalgce_pair_rows(
        parents=[SimpleNamespace(parent_id="parent-1", smiles="CC")],
        before_rows=[{"predicted_label": 1, "probabilities": [0.1, 0.9]}],
        candidates=[
            {
                "candidate_id": "rule-1",
                "rank": 1,
                "native_rank": 1,
                "rule_content_hash": "a" * 64,
            }
        ],
        featurizer=object(),
        oracle=_BombOracle(),
        provider=object(),
        card={"checkpoint_id": "frozen-bace-gine"},
        spec=spec,
        method_id="globalgce",
        oracle_batch_size=256,
    )

    assert len(rows) == 1
    assert rows[0]["applicable"] is False
    assert rows[0]["failure_reason"] == "no_legal_native_lhs_match_or_sanitized_rhs"
    assert receipt["oracle_called"] is False
    assert receipt["reason"] == "NO_EVALUABLE_GRAPHS_AFTER_PRE_ORACLE_FILTERS"
    assert receipt["logits_shape"] == [0, 2]
    assert receipt["probabilities_shape"] == [0, 2]
    assert receipt["predictions_shape"] == [0]
