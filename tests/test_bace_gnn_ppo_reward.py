from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.data.molecular_graph_featurizer import (
    MolecularGraphFeaturizer,
    default_molecular_feature_schema,
)
from src.oracles.base_oracle import BaseOracle
from src.rewards.gnn_ppo_reward import (
    BatchedGNNPPORewardAdapter,
    GNNPPORewardConfig,
)
from src.train.bace_gnn_ppo import run_canary_connected_deletion_preflight


class _FakeFrozenGINE(BaseOracle):
    checkpoint_id = "frozen-gine-checkpoint-hash"
    backbone = "gine"
    num_classes = 2
    source_label = 1
    temperature = 1.25

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_logits(self, graphs, *, batch_size=None):
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        return np.log(probabilities)

    def predict_proba(self, graphs, *, batch_size=None):
        self.batch_sizes.append(len(graphs))
        rows = []
        for graph in graphs:
            # Parents contain O; deleting terminal O yields CCC and flips class.
            rows.append([0.1, 0.9] if "O" in graph.smiles else [0.9, 0.1])
        return np.asarray(rows, dtype=np.float32)

    def predict_records(self, graphs, *, batch_size=None):
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        return [
            {
                "predicted_label": int(row.argmax()),
                "probabilities": row.tolist(),
                "logits": np.log(row).tolist(),
                "source_probability": float(row[1]),
                "confidence": float(row.max()),
                "checkpoint_id": self.checkpoint_id,
                "backbone": self.backbone,
                "num_classes": self.num_classes,
                "temperature": self.temperature,
                "source_label": self.source_label,
            }
            for row in probabilities
        ]


def _adapter(tmp_path: Path) -> tuple[BatchedGNNPPORewardAdapter, _FakeFrozenGINE]:
    checkpoint = tmp_path / "gnn"
    checkpoint.mkdir()
    (checkpoint / "temperature_scaling.json").write_text(
        json.dumps({"temperature": 1.25}), encoding="utf-8"
    )
    oracle = _FakeFrozenGINE()
    adapter = BatchedGNNPPORewardAdapter(
        oracle=oracle,
        featurizer=MolecularGraphFeaturizer(default_molecular_feature_schema()),
        checkpoint_dir=checkpoint,
        policy_initializer_hash="clean-policy-hash",
        reference_policy_hash="frozen-reference-hash",
        config=GNNPPORewardConfig(oracle_batch_size=32),
    )
    return adapter, oracle


def test_batched_gnn_reward_caches_before_and_batches_after(tmp_path: Path) -> None:
    adapter, oracle = _adapter(tmp_path)
    rows = adapter.score_batch(
        parent_smiles=["CCCO", "CCCO"],
        generated_fragments=["O", "O"],
        labels=[1, 1],
        metas=[{"molecule_id": "p0"}, {"molecule_id": "p1"}],
    )

    # One unique parent batch and one residual batch; the checkpoint was supplied
    # once to the adapter rather than loaded per candidate.
    assert oracle.batch_sizes == [1, 2]
    assert adapter.oracle_load_count == 1
    assert adapter.parent_cache_misses == 1
    assert all(row["gnn_scored_deletion"] for row in rows)
    assert all(row["cf_flip"] for row in rows)
    assert all(row["cf_drop"] > 0.0 for row in rows)
    assert all(row["oracle_backend"] == "gnn" for row in rows)
    assert all(row["rf_oracle_used"] is False for row in rows)
    assert all(row["calibration_loaded"] is False for row in rows)
    assert all(row["calibration_dataset_loaded"] is False for row in rows)
    assert all(row["frozen_temperature_calibration_loaded"] is True for row in rows)
    assert all(row["test_loaded"] is False for row in rows)

    adapter.score_batch(
        parent_smiles=["CCCO"],
        generated_fragments=["O"],
        labels=[1],
        metas=[{"molecule_id": "p2"}],
    )
    assert adapter.parent_cache_hits == 1
    assert oracle.batch_sizes == [1, 2, 1]


def test_invalid_generation_fails_candidate_closed_without_skipping_batch(
    tmp_path: Path,
) -> None:
    adapter, _oracle = _adapter(tmp_path)
    rows = adapter.score_batch(
        parent_smiles=["CCCO", "CCCO"],
        generated_fragments=["not-smiles", "O"],
        labels=[1, 1],
        metas=[{"id": "bad"}, {"id": "good"}],
    )
    bad, good = rows
    assert bad["parse_ok"] is False
    assert bad["gnn_scored_deletion"] is False
    assert bad["reward_total"] < good["reward_total"]
    assert good["cf_flip"] is True


def test_reward_provenance_contains_every_required_scientific_field(
    tmp_path: Path,
) -> None:
    from src.train.bace_gnn_ppo import REWARD_PROVENANCE_FIELDS

    adapter, _oracle = _adapter(tmp_path)
    row = adapter.score_batch(
        parent_smiles=["CCCO"],
        generated_fragments=["O"],
        labels=[1],
        metas=[{"id": "p0"}],
    )[0]
    assert not (set(REWARD_PROVENANCE_FIELDS) - set(row))
    assert row["policy_initializer_hash"] == "clean-policy-hash"
    assert row["reference_policy_hash"] == "frozen-reference-hash"


def test_canary_preflight_uses_same_real_adapter_on_eight_train_parents(
    tmp_path: Path,
) -> None:
    adapter, oracle = _adapter(tmp_path)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("molecule_id,parent_smiles,label\n", encoding="utf-8")
    examples = [
        SimpleNamespace(
            index=index,
            molecule_id=f"train-{index}",
            parent_smiles="CCCO",
            original_label=1,
        )
        for index in range(8)
    ]
    manifest = run_canary_connected_deletion_preflight(
        reward_adapter=adapter,
        examples=examples,
        train_csv=train_csv,
        frozen_train_contract={
            "checkpoint_split_manifest": str(tmp_path / "split_manifest.json"),
            "checkpoint_split_manifest_sha256": "a" * 64,
            "train_csv": str(train_csv.resolve()),
            "train_csv_sha256": hashlib.sha256(train_csv.read_bytes()).hexdigest(),
            "calibration_loaded": False,
            "test_loaded": False,
        },
    )
    assert manifest["status"] == "PASS"
    assert manifest["source_parent_count"] == 8
    assert manifest["source_split"] == "train"
    assert manifest["real_gnn_inference_observed"] is True
    assert manifest["adapter_instance_reused"] is True
    assert manifest["gnn_scored_deletion_count"] >= 1
    assert manifest["calibration_loaded"] is False
    assert manifest["test_loaded"] is False
    assert adapter.oracle_load_count == 1
    assert oracle.batch_sizes[0] == 1  # one canonical parent cache miss
    assert oracle.batch_sizes[-1] >= 8  # real residual GNN batch
