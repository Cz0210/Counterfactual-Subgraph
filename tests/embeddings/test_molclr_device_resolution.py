from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embeddings import molclr_gnn_embedding
from src.embeddings.molclr_gnn_embedding import MolCLREmbeddingError


def _fake_torch(*, cuda_available: bool, device_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: device_count,
        )
    )


def test_resolve_device_accepts_visible_indexed_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        molclr_gnn_embedding,
        "_require_torch",
        lambda: _fake_torch(cuda_available=True, device_count=4),
    )

    assert molclr_gnn_embedding._resolve_device("cuda:0") == "cuda:0"
    assert molclr_gnn_embedding._resolve_device("cuda:3") == "cuda:3"


def test_resolve_device_rejects_indexed_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        molclr_gnn_embedding,
        "_require_torch",
        lambda: _fake_torch(cuda_available=False, device_count=0),
    )

    with pytest.raises(MolCLREmbeddingError, match="CUDA is not available"):
        molclr_gnn_embedding._resolve_device("cuda:0")


def test_resolve_device_rejects_non_visible_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        molclr_gnn_embedding,
        "_require_torch",
        lambda: _fake_torch(cuda_available=True, device_count=2),
    )

    with pytest.raises(MolCLREmbeddingError, match="only 2 CUDA device"):
        molclr_gnn_embedding._resolve_device("cuda:2")


@pytest.mark.parametrize("device", ["cuda:", "cuda:-1", "cuda:gpu0", "mps"])
def test_resolve_device_rejects_malformed_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setattr(
        molclr_gnn_embedding,
        "_require_torch",
        lambda: _fake_torch(cuda_available=True, device_count=4),
    )

    with pytest.raises(ValueError, match="Expected auto/cpu/cuda/cuda:N"):
        molclr_gnn_embedding._resolve_device(device)
