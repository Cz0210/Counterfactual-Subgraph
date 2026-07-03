"""HIV.csv graph dataset utilities for the adapted GCFExplainer path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class HIVCSVGraphDataset:
    """Lightweight PyG-compatible dataset backed by ``graphs.pt``.

    The class intentionally avoids external graph benchmark loaders.  It
    provides the minimal interface used by the official GCFExplainer code:
    ``num_features``, ``num_classes``, ``__len__``, ``__getitem__``, and
    slicing/list indexing.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.graphs_path = self.root / "graphs.pt"
        self.summary_path = self.root / "dataset_summary.json"
        if not self.graphs_path.exists():
            raise FileNotFoundError(f"HIVCSV graphs.pt not found: {self.graphs_path}")
        from src.baselines.gcf_hiv_csv_model import torch_load

        self.graphs = list(torch_load(str(self.graphs_path), map_location="cpu"))
        self._summary: dict[str, Any] = {}
        if self.summary_path.exists():
            try:
                self._summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
            except Exception:
                self._summary = {}

    @property
    def num_features(self) -> int:
        if self._summary.get("num_features") is not None:
            return int(self._summary["num_features"])
        if not self.graphs:
            return 0
        return int(self.graphs[0].x.shape[1])

    @property
    def num_classes(self) -> int:
        if self._summary.get("num_classes") is not None:
            return int(self._summary["num_classes"])
        labels = {int(graph.y.item()) for graph in self.graphs if hasattr(graph, "y")}
        return max(labels) + 1 if labels else 2

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Any:
        return self.graphs[int(idx)]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return self.graphs[index]
        if isinstance(index, (list, tuple)):
            return [self.graphs[int(i)] for i in index]
        try:
            import torch

            if torch.is_tensor(index):
                if index.ndim == 0:
                    return self.graphs[int(index.item())]
                return [self.graphs[int(i)] for i in index.detach().cpu().tolist()]
        except Exception:
            pass
        return self.graphs[int(index)]
