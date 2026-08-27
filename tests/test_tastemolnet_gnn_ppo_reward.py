from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.data.molecular_graph_featurizer import (
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.oracles.base_oracle import BaseOracle
from src.rewards.gnn_ppo_reward import (
    BatchedGNNPPORewardAdapter,
    GNNPPORewardConfig,
    TASTE_GNN_PPO_REWARD_SCHEMA,
)


class _FakeTasteFrozenGINE(BaseOracle):
    checkpoint_id = "taste-frozen-gine-checkpoint-hash"
    backbone = "gine"
    num_classes = 3
    source_label = 1
    temperature = 1.5

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @staticmethod
    def _probabilities(graph: object) -> list[float]:
        smiles = str(getattr(graph, "smiles", ""))
        molecule_id = str(getattr(graph, "molecule_id", ""))
        if "O" in smiles:
            return [0.05, 0.90, 0.05]
        if molecule_id.startswith("to-tasteless:"):
            return [0.05, 0.05, 0.90]
        return [0.90, 0.05, 0.05]

    def predict_logits(self, graphs, *, batch_size=None):
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        return np.log(probabilities)

    def predict_proba(self, graphs, *, batch_size=None):
        del batch_size
        self.batch_sizes.append(len(graphs))
        return np.asarray(
            [self._probabilities(graph) for graph in graphs], dtype=np.float32
        )

    def predict_records(self, graphs, *, batch_size=None):
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        return [
            {
                "predicted_label": int(row.argmax()),
                "probabilities": row.tolist(),
                "logits": np.log(row).tolist(),
                "source_probability": float(row[self.source_label]),
                "confidence": float(row.max()),
                "checkpoint_id": self.checkpoint_id,
                "backbone": self.backbone,
                "num_classes": self.num_classes,
                "temperature": self.temperature,
                "source_label": self.source_label,
            }
            for row in probabilities
        ]


def _adapter(tmp_path: Path) -> BatchedGNNPPORewardAdapter:
    checkpoint = tmp_path / "taste-gine"
    checkpoint.mkdir()
    (checkpoint / "temperature_scaling.json").write_text(
        json.dumps({"temperature": 1.5}), encoding="utf-8"
    )
    return BatchedGNNPPORewardAdapter(
        oracle=_FakeTasteFrozenGINE(),
        featurizer=MolecularGraphFeaturizer(default_molecular_feature_schema()),
        checkpoint_dir=checkpoint,
        policy_initializer_hash="a" * 64,
        reference_policy_hash="b" * 64,
        config=GNNPPORewardConfig(
            dataset="tastemolnet",
            num_classes=3,
            source_label=1,
            oracle_batch_size=32,
        ),
    )


def test_taste_reward_preserves_three_classes_and_both_destinations(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    rows = adapter.score_batch(
        parent_smiles=["CCCO", "CCCO"],
        generated_fragments=["O", "O"],
        labels=[1, 1],
        metas=[
            {"molecule_id": "to-bitter"},
            {"molecule_id": "to-tasteless"},
        ],
    )

    assert [row["destination_label"] for row in rows] == [0, 2]
    assert all(row["schema_version"] == TASTE_GNN_PPO_REWARD_SCHEMA for row in rows)
    assert all(row["dataset"] == "tastemolnet" for row in rows)
    assert all(row["source_label"] == 1 for row in rows)
    assert all(row["pred_before"] == 1 for row in rows)
    assert all(row["pred_after"] in {0, 2} for row in rows)
    assert all(row["cf_flip"] is True for row in rows)
    assert all(len(row["p_before_all_classes"]) == 3 for row in rows)
    assert all(len(row["p_after_all_classes"]) == 3 for row in rows)
    assert all(len(row["logits_before_all_classes"]) == 3 for row in rows)
    assert all(len(row["logits_after_all_classes"]) == 3 for row in rows)
    assert all(row["source_probability_before"] == pytest.approx(0.9) for row in rows)
    assert all(row["source_probability_after"] == pytest.approx(0.05) for row in rows)
    assert all(row["margin_drop"] > 0.0 for row in rows)
    assert all(row["rf_oracle_used"] is False for row in rows)
    assert all(row["calibration_dataset_loaded"] is False for row in rows)
    assert all(row["test_loaded"] is False for row in rows)

    provenance = adapter.provenance()
    assert provenance["dataset"] == "tastemolnet"
    assert provenance["num_classes"] == 3
    assert provenance["source_label"] == 1
    assert provenance["oracle_load_count"] == 1


def test_parent_preflight_selects_only_predicted_source_without_second_oracle(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    records = adapter.predict_parent_records(
        parent_smiles=["CCCO", "CCC"],
        metas=[{"molecule_id": "sweet"}, {"molecule_id": "bitter"}],
    )
    assert [row["predicted_label"] for row in records] == [1, 0]
    selected = [row for row in records if row["predicted_label"] == 1]
    assert len(selected) == 1
    assert len(selected[0]["probabilities"]) == 3
    assert adapter.oracle_load_count == 1


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"dataset": "tastemolnet", "num_classes": 2}, "class contract"),
        ({"dataset": "bace", "num_classes": 3}, "class contract"),
        ({"dataset": True}, "dataset is unsupported"),
        ({"num_classes": True}, "class contract"),
        ({"source_label": True}, "source_label=1"),
        ({"enable_projection": 1}, "native bool"),
        ({"oracle_batch_size": 1.0}, "positive native int"),
        ({"cf_drop_weight": "3.0"}, "finite native numeric"),
        ({"strict_flip_bonus": True}, "finite native numeric"),
    ),
)
def test_reward_config_rejects_dataset_and_native_type_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    defaults: dict[str, object] = {
        "dataset": "bace",
        "num_classes": 2,
        "source_label": 1,
    }
    with pytest.raises(ValueError, match=message):
        GNNPPORewardConfig(**{**defaults, **changes}).validate()


def test_three_class_oracle_cannot_enter_default_bace_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "taste-gine"
    checkpoint.mkdir()
    (checkpoint / "temperature_scaling.json").write_text(
        json.dumps({"temperature": 1.5}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="class/source contract differs"):
        BatchedGNNPPORewardAdapter(
            oracle=_FakeTasteFrozenGINE(),
            featurizer=MolecularGraphFeaturizer(
                default_molecular_feature_schema()
            ),
            checkpoint_dir=checkpoint,
            policy_initializer_hash="a" * 64,
            reference_policy_hash="b" * 64,
        )


def test_reward_adapter_from_payloads_uses_in_memory_file_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "held-checkpoint"
    checkpoint.mkdir()
    schema_bytes = (
        json.dumps(default_molecular_feature_schema().to_dict(), sort_keys=True) + "\n"
    ).encode()
    temperature_bytes = b'{"temperature":1.5}\n'
    payloads = {
        "model.pt": b"model",
        "model_card.json": b"{}",
        "feature_schema.json": schema_bytes,
        "label_map.json": b"{}",
        "split_manifest.json": b"{}",
        "test_evaluation_status.json": b"{}",
        "temperature_scaling.json": temperature_bytes,
    }
    monkeypatch.setattr(
        "src.rewards.gnn_ppo_reward.GNNOracle.from_payloads",
        lambda *_args, **_kwargs: _FakeTasteFrozenGINE(),
    )
    adapter = BatchedGNNPPORewardAdapter.from_payloads(
        payloads,
        checkpoint_dir=checkpoint,
        device="cpu",
        policy_initializer_hash="a" * 64,
        reference_policy_hash="b" * 64,
        config=GNNPPORewardConfig(
            dataset="tastemolnet", num_classes=3, source_label=1
        ),
    )
    assert adapter.temperature_calibration_hash == hashlib.sha256(
        temperature_bytes
    ).hexdigest()
    assert adapter.feature_schema_hash == hashlib.sha256(schema_bytes).hexdigest()


@pytest.mark.parametrize("label", (1.0, True, "1", None))
def test_taste_reward_rejects_non_native_source_labels(
    tmp_path: Path,
    label: object,
) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="train source-class parents only"):
        adapter.score_batch(
            parent_smiles=["CCCO"],
            generated_fragments=["O"],
            labels=[label],  # type: ignore[list-item]
            metas=[{"molecule_id": "typed-source"}],
        )
