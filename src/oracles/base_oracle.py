"""Backend-neutral classifier oracle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class OraclePredictionRecord:
    """JSON-safe prediction record shared by binary and multiclass tasks."""

    predicted_label: int
    probabilities: tuple[float, ...]
    logits: tuple[float, ...]
    source_probability: float
    confidence: float
    checkpoint_id: str
    backbone: str
    num_classes: int
    temperature: float
    source_label: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["probabilities"] = list(self.probabilities)
        payload["logits"] = list(self.logits)
        return payload


class BaseOracle(ABC):
    """One frozen classifier loaded once and reused for batch inference."""

    checkpoint_id: str
    backbone: str
    num_classes: int
    source_label: int
    temperature: float

    @abstractmethod
    def predict_logits(self, graphs: Any, *, batch_size: int | None = None) -> Any:
        """Return a ``[num_graphs, num_classes]`` float array."""

    @abstractmethod
    def predict_proba(self, graphs: Any, *, batch_size: int | None = None) -> Any:
        """Return calibrated class probabilities for every graph."""

    def predict_label(self, graphs: Any, *, batch_size: int | None = None) -> Any:
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        return probabilities.argmax(axis=1)

    def predict_records(
        self, graphs: Any, *, batch_size: int | None = None
    ) -> list[dict[str, Any]]:
        logits = self.predict_logits(graphs, batch_size=batch_size)
        probabilities = self.predict_proba(graphs, batch_size=batch_size)
        if logits.shape != probabilities.shape:
            raise RuntimeError("Oracle logits/probabilities shapes differ.")
        records: list[dict[str, Any]] = []
        for row_logits, row_probabilities in zip(logits, probabilities, strict=True):
            predicted_label = int(row_probabilities.argmax())
            record = OraclePredictionRecord(
                predicted_label=predicted_label,
                probabilities=tuple(float(value) for value in row_probabilities),
                logits=tuple(float(value) for value in row_logits),
                source_probability=float(row_probabilities[self.source_label]),
                confidence=float(row_probabilities[predicted_label]),
                checkpoint_id=self.checkpoint_id,
                backbone=self.backbone,
                num_classes=self.num_classes,
                temperature=float(self.temperature),
                source_label=self.source_label,
            )
            records.append(record.to_dict())
        return records

    def validate_contract(self) -> None:
        if int(self.num_classes) < 2:
            raise ValueError("Oracle num_classes must be at least two.")
        if not 0 <= int(self.source_label) < int(self.num_classes):
            raise ValueError("Oracle source_label falls outside the class range.")
        if float(self.temperature) <= 0.0:
            raise ValueError("Oracle temperature must be positive.")
        if not str(self.checkpoint_id).strip():
            raise ValueError("Oracle checkpoint_id must be non-empty.")


__all__ = ["BaseOracle", "OraclePredictionRecord"]
