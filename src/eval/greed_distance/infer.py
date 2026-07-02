"""Inference wrapper for GREED-style normalized-GED prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.greed_distance.graph_conversion import graph_from_smiles
from src.eval.greed_distance.model import load_checkpoint, make_pair_batch


class GreedDistancePredictor:
    """Load a GREED checkpoint and predict normalized graph distance."""

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cuda") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("GREED distance inference requires PyTorch.") from exc
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model, self.payload = load_checkpoint(checkpoint_path, device=device)

    def predict_graphs(self, graph_a: dict[str, Any], graph_b: dict[str, Any]) -> float | None:
        try:
            import torch

            batch = make_pair_batch([graph_a], [graph_b], device=self.device)
            with torch.no_grad():
                pred = self.model(batch).detach().cpu().tolist()[0]
            return max(0.0, min(1.0, float(pred)))
        except Exception:
            return None

    def predict_smiles(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        graph_a = graph_from_smiles(smiles_a)
        graph_b = graph_from_smiles(smiles_b)
        if not graph_a.get("parse_ok") or not graph_b.get("parse_ok"):
            return {
                "distance": None,
                "ok": False,
                "error": f"parse_failed:{graph_a.get('error')};{graph_b.get('error')}",
            }
        distance = self.predict_graphs(graph_a, graph_b)
        return {
            "distance": distance,
            "ok": distance is not None,
            "error": None if distance is not None else "greed_prediction_failed",
        }
