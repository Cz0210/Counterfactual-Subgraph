"""Single dataset-aware factory for GNN and legacy RF classifier oracles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.dataset_registry import (
    assert_oracle_backend_allowed,
    get_dataset_spec,
    normalize_dataset_id,
)
from src.oracles.base_oracle import BaseOracle, OraclePredictionRecord
from src.oracles.gnn_oracle import GNNOracle, sha256_file


class LegacyRFOracle(BaseOracle):
    """Compatibility adapter for already-frozen AIDS/Mutagenicity RF bundles."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        num_classes: int,
        source_label: int,
    ) -> None:
        from src.rewards.reward_calculator import load_oracle_bundle

        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        bundle = load_oracle_bundle(self.checkpoint_path)
        self.model = bundle["model"]
        self.radius = int(bundle["fingerprint_radius"])
        self.n_bits = int(bundle["fingerprint_bits"])
        self.num_classes = int(num_classes)
        self.source_label = int(source_label)
        self.temperature = 1.0
        self.checkpoint_id = sha256_file(self.checkpoint_path)
        self.backbone = "legacy_random_forest"
        classes = getattr(self.model, "classes_", tuple(range(self.num_classes)))
        self.class_labels = tuple(int(value) for value in classes)
        if self.class_labels != tuple(range(self.num_classes)):
            raise ValueError(
                "Legacy RF oracle class ordering must match contiguous class indices."
            )
        self.validate_contract()

    @staticmethod
    def _smiles_values(graphs: Any) -> list[str]:
        if isinstance(graphs, str):
            return [graphs]
        if isinstance(graphs, Mapping):
            return [str(graphs.get("model_smiles") or graphs.get("smiles") or "")]
        if hasattr(graphs, "smiles") and isinstance(graphs.smiles, str):
            return [str(graphs.smiles)]
        if not isinstance(graphs, Sequence):
            raise TypeError("Legacy RF oracle expects SMILES strings or records.")
        values: list[str] = []
        for item in graphs:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, Mapping):
                values.append(str(item.get("model_smiles") or item.get("smiles") or ""))
            else:
                values.append(str(getattr(item, "smiles", "") or ""))
        if not values or any(not value.strip() for value in values):
            raise ValueError("Legacy RF oracle received an empty SMILES value.")
        return values

    def predict_proba(
        self, graphs: Any, *, batch_size: int | None = None
    ) -> np.ndarray:
        del batch_size
        from src.rewards.reward_calculator import smiles_to_morgan_array

        features = []
        for smiles in self._smiles_values(graphs):
            feature = smiles_to_morgan_array(
                smiles,
                radius=self.radius,
                n_bits=self.n_bits,
                clean_dummy_atoms=True,
            )
            if feature is None:
                raise ValueError(f"Legacy RF oracle could not featurize SMILES: {smiles!r}")
            features.append(feature)
        probabilities = np.asarray(
            self.model.predict_proba(np.asarray(features, dtype=np.float32)),
            dtype=np.float64,
        )
        if probabilities.shape[1] != self.num_classes:
            raise ValueError("Legacy RF probability width differs from dataset num_classes.")
        return probabilities

    def predict_logits(
        self, graphs: Any, *, batch_size: int | None = None
    ) -> np.ndarray:
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        return np.log(np.clip(probabilities, 1e-12, 1.0))

    def predict_records(
        self, graphs: Any, *, batch_size: int | None = None
    ) -> list[dict[str, Any]]:
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        result: list[dict[str, Any]] = []
        for row_logits, row_probabilities in zip(logits, probabilities, strict=True):
            predicted = int(row_probabilities.argmax())
            result.append(
                OraclePredictionRecord(
                    predicted_label=predicted,
                    probabilities=tuple(float(value) for value in row_probabilities),
                    logits=tuple(float(value) for value in row_logits),
                    source_probability=float(row_probabilities[self.source_label]),
                    confidence=float(row_probabilities[predicted]),
                    checkpoint_id=self.checkpoint_id,
                    backbone=self.backbone,
                    num_classes=self.num_classes,
                    temperature=1.0,
                    source_label=self.source_label,
                ).to_dict()
            )
        return result


def build_oracle(
    *,
    dataset: str,
    backend: str,
    checkpoint: str | Path,
    device: str | Any = "cpu",
    batch_size: int = 256,
    num_classes: int | None = None,
    source_label: int | None = None,
    verify_hashes: bool = True,
) -> BaseOracle:
    """Build one allowed oracle and validate it against dataset semantics."""

    dataset_id = normalize_dataset_id(dataset)
    normalized_backend = str(backend or "").strip().lower().replace("-", "_")
    assert_oracle_backend_allowed(dataset_id, normalized_backend)
    spec = get_dataset_spec(dataset_id)
    resolved_classes = spec.num_classes if num_classes is None else int(num_classes)
    resolved_source = spec.source_label if source_label is None else int(source_label)
    if resolved_classes != spec.num_classes:
        raise ValueError(
            f"Oracle num_classes={resolved_classes} conflicts with {dataset_id} "
            f"registry num_classes={spec.num_classes}."
        )
    if resolved_source != spec.source_label:
        raise ValueError(
            f"Oracle source_label={resolved_source} conflicts with {dataset_id} "
            f"registry source_label={spec.source_label}."
        )

    if normalized_backend == "gnn":
        oracle = GNNOracle.from_checkpoint(
            checkpoint,
            device=device,
            batch_size=batch_size,
            verify_hashes=verify_hashes,
        )
        if oracle.num_classes != resolved_classes:
            raise ValueError("GNN checkpoint num_classes conflicts with dataset registry.")
        if oracle.source_label != resolved_source:
            raise ValueError("GNN checkpoint source_label conflicts with dataset registry.")
        # The model card is already validated by the bundle loader. Normalize
        # aliases so capitalization in historical manifests remains harmless.
        card_dataset = get_dataset_spec(
            str(
                json.loads(
                    (Path(checkpoint) / "model_card.json").read_text(encoding="utf-8")
                ).get("dataset", dataset_id)
            )
        ).dataset_id
        if card_dataset != dataset_id:
            raise ValueError(
                f"GNN checkpoint dataset={card_dataset!r} does not match {dataset_id!r}."
            )
        return oracle
    if normalized_backend == "rf":
        return LegacyRFOracle(
            checkpoint,
            num_classes=resolved_classes,
            source_label=resolved_source,
        )
    raise ValueError(f"Unsupported oracle backend: {backend!r}")


def oracle_from_config(
    config: Mapping[str, Any],
    *,
    device: str | Any | None = None,
) -> BaseOracle:
    oracle = config.get("oracle", config)
    dataset = str(oracle.get("dataset") or config.get("dataset") or "")
    backend = str(oracle.get("backend") or oracle.get("oracle_backend") or "")
    checkpoint = oracle.get("checkpoint") or oracle.get("gnn_checkpoint")
    if not dataset or not backend or not checkpoint:
        raise ValueError("Oracle config requires dataset, backend, and checkpoint.")
    return build_oracle(
        dataset=dataset,
        backend=backend,
        checkpoint=checkpoint,
        device=device or oracle.get("device", "cpu"),
        batch_size=int(oracle.get("batch_size", 256)),
        num_classes=(
            None if oracle.get("num_classes") is None else int(oracle["num_classes"])
        ),
        source_label=(
            None if oracle.get("source_label") is None else int(oracle["source_label"])
        ),
        verify_hashes=bool(oracle.get("verify_hashes", True)),
    )


create_oracle = build_oracle


__all__ = [
    "LegacyRFOracle",
    "build_oracle",
    "create_oracle",
    "oracle_from_config",
]
