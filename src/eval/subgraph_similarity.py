"""Similarity helpers for selector redundancy terms."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_EMBEDDING_FIELD = "final_fragment_embedding"
EMBEDDING_FIELD_FALLBACKS = (
    "embedding",
    "fragment_embedding",
    "subgraph_embedding",
    "graph_embedding",
)


@dataclass(frozen=True, slots=True)
class CandidateEmbedding:
    """Parsed embedding plus the source field used to retrieve it."""

    vector: np.ndarray
    field_name: str


def parse_embedding(value: Any) -> np.ndarray:
    """Parse an embedding from a list, JSON string, or comma-separated string."""

    if value is None:
        raise ValueError("embedding value is missing")

    raw_value = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("embedding value is empty")
        if text.startswith("["):
            try:
                raw_value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"embedding JSON string could not be parsed: {exc}") from exc
        else:
            raw_value = [part.strip() for part in text.split(",") if part.strip()]

    try:
        array = np.asarray(raw_value, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"embedding value could not be converted to floats: {exc}") from exc

    if array.ndim != 1:
        raise ValueError(f"embedding must be one-dimensional, got shape={array.shape}")
    if array.size == 0:
        raise ValueError("embedding dimension is zero")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding contains NaN or Inf")
    return array


def l2_normalize(vec: Any) -> np.ndarray:
    """Return an L2-normalized vector, leaving zero vectors safely at zero."""

    array = np.asarray(vec, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"embedding must be a non-empty one-dimensional vector, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding contains NaN or Inf")
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return np.zeros_like(array, dtype=np.float64)
    return array / norm


def cosine_embedding_similarity(vec_a: Any, vec_b: Any) -> float:
    """Return max(0, cosine(vec_a, vec_b)) clipped to the [0, 1] range."""

    norm_a = l2_normalize(vec_a)
    norm_b = l2_normalize(vec_b)
    if norm_a.shape != norm_b.shape:
        raise ValueError(f"embedding dimensions differ: {norm_a.shape[0]} vs {norm_b.shape[0]}")
    cosine = float(np.dot(norm_a, norm_b))
    if not math.isfinite(cosine):
        return 0.0
    return max(0.0, min(1.0, cosine))


def get_candidate_embedding(
    candidate: dict[str, Any],
    embedding_field: str = DEFAULT_EMBEDDING_FIELD,
) -> CandidateEmbedding:
    """Read and parse a candidate embedding from the preferred field or fallbacks."""

    fields = [embedding_field]
    for fallback in EMBEDDING_FIELD_FALLBACKS:
        if fallback not in fields:
            fields.append(fallback)

    for field_name in fields:
        if field_name not in candidate:
            continue
        value = candidate.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        vector = parse_embedding(value)
        return CandidateEmbedding(vector=vector, field_name=field_name)

    raise ValueError(
        "candidate_pool.jsonl row is missing an embedding field. "
        f"Tried fields={fields}; add embeddings or use --embedding-missing-policy skip."
    )


__all__ = [
    "CandidateEmbedding",
    "DEFAULT_EMBEDDING_FIELD",
    "EMBEDDING_FIELD_FALLBACKS",
    "cosine_embedding_similarity",
    "get_candidate_embedding",
    "l2_normalize",
    "parse_embedding",
]
