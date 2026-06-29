"""Training utilities for the GREED-style HIV normalized-GED model."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from src.eval.greed_distance.graph_conversion import graph_from_smiles
from src.eval.greed_distance.model import GreedGEDModel, make_pair_batch, save_checkpoint
from src.utils.io import ensure_directory


def _require_torch() -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GREED distance training requires PyTorch.") from exc
    return torch, DataLoader


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def read_labeled_pairs(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    clean: list[dict[str, Any]] = []
    for row in rows:
        target = _as_float(row.get("ged_norm"))
        if str(row.get("ged_label_ok")).lower() not in {"true", "1", "yes"} or target is None:
            continue
        clean.append(row)
    return clean


class PairDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.graph_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _graph(self, smiles: str) -> dict[str, Any]:
        if smiles not in self.graph_cache:
            self.graph_cache[smiles] = graph_from_smiles(smiles)
        return self.graph_cache[smiles]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "graph_a": self._graph(str(row.get("smiles_a") or "")),
            "graph_b": self._graph(str(row.get("smiles_b") or "")),
            "target": float(row["ged_norm"]),
            "pair_id": row.get("pair_id"),
        }


def _collate(items: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    torch, _DataLoader = _require_torch()
    batch = make_pair_batch(
        [item["graph_a"] for item in items],
        [item["graph_b"] for item in items],
        device=device,
    )
    batch["target"] = torch.tensor([float(item["target"]) for item in items], dtype=torch.float32, device=device)
    batch["pair_id"] = [item.get("pair_id") for item in items]
    return batch


def _metrics(preds: list[float], targets: list[float]) -> dict[str, float | None]:
    if not preds:
        return {"mae": None, "rmse": None, "mse": None}
    errors = [float(p) - float(t) for p, t in zip(preds, targets)]
    mse = sum(error * error for error in errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    return {"mae": float(mae), "rmse": float(math.sqrt(mse)), "mse": float(mse)}


def _evaluate(model: Any, rows: list[dict[str, Any]], *, batch_size: int, device: str) -> dict[str, Any]:
    torch, DataLoader = _require_torch()
    dataset = PairDataset(rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda items: _collate(items, device=device))
    preds: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch).detach().cpu().tolist()
            target = batch["target"].detach().cpu().tolist()
            preds.extend(float(item) for item in pred)
            targets.extend(float(item) for item in target)
    return {"num_pairs": len(rows), **_metrics(preds, targets)}


def train_greed_distance_model(
    *,
    train_pairs_csv: str | Path,
    val_pairs_csv: str | Path,
    test_pairs_csv: str | Path | None,
    checkpoint_path: str | Path,
    train_metrics_json: str | Path,
    test_metrics_csv: str | Path,
    num_layers: int = 8,
    hidden_dim: int = 64,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cuda",
    seed: int = 13,
) -> dict[str, Any]:
    torch, DataLoader = _require_torch()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    train_rows = read_labeled_pairs(train_pairs_csv)
    val_rows = read_labeled_pairs(val_pairs_csv)
    test_rows = read_labeled_pairs(test_pairs_csv) if test_pairs_csv else []
    if not train_rows:
        raise ValueError(f"no labeled train pairs found: {train_pairs_csv}")
    if not val_rows:
        val_rows = train_rows[: min(len(train_rows), max(1, int(batch_size)))]

    model_config = {"num_layers": int(num_layers), "hidden_dim": int(hidden_dim)}
    model = GreedGEDModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MSELoss()
    dataset = PairDataset(train_rows)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=lambda items: _collate(items, device=device),
    )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch["target"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_metrics = _evaluate(model, val_rows, batch_size=int(batch_size), device=device)
        train_loss = sum(losses) / len(losses) if losses else None
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"[GREED_TRAIN] epoch={epoch} train_loss={train_loss} val_mae={val_metrics.get('mae')}", flush=True)
        val_mse = float(val_metrics.get("mse") or float("inf"))
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                model_config=model_config,
                metrics={"best_epoch": best_epoch, "val_mse": best_val},
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    final_metrics = {
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "test_pairs": len(test_rows),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "history": history,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
    }
    ensure_directory(Path(train_metrics_json).expanduser().resolve().parent)
    Path(train_metrics_json).expanduser().resolve().write_text(
        json.dumps(final_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    test_metrics = _evaluate(model, test_rows, batch_size=int(batch_size), device=device) if test_rows else {}
    ensure_directory(Path(test_metrics_csv).expanduser().resolve().parent)
    with Path(test_metrics_csv).expanduser().resolve().open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["split", "num_pairs", "mae", "rmse", "mse"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"split": "test", **test_metrics})
    return final_metrics
