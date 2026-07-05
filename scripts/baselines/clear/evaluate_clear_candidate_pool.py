#!/usr/bin/env python3
"""Evaluate a CLEAR candidate/action pool under the unified CCRCov contract.

CLEAR exports local counterfactual graph actions. This evaluator keeps the
official CLEAR fields as diagnostics, but final flip/drop metrics are populated
only from a project teacher prediction source. When the pool lacks SMILES or
full graph arrays needed by a teacher adapter, the script fails clearly unless
``--allow-action-only`` is set for cost-only smoke diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.ccrcov_distance_eval import CF_MODES, normalize_cf_mode  # noqa: E402
from src.eval.close_counterfactual_coverage import predict_with_teacher  # noqa: E402
from src.rewards.teacher_semantic import TeacherSemanticScorer  # noqa: E402


DEFAULT_CANDIDATE_POOL = (
    "outputs/hpc/baselines/clear/aids/candidate_pool/"
    "clear_aids_candidate_pool.with_graphs.jsonl"
)
DEFAULT_TEACHER_PATH = ""
DEFAULT_CLEAR_GRAPHPRED_PATH = "baselines/clear_official/models_save/prediction/weights_graphPred__aids.pt"
DEFAULT_OUT_DIR = "outputs/hpc/baselines/clear/aids/eval"

SMILES_ORIGINAL_FIELDS = ("original_smiles", "parent_smiles", "smiles")
SMILES_CF_FIELDS = ("cf_smiles", "counterfactual_smiles", "candidate_smiles", "action_smiles")
FULL_GRAPH_FIELDS = ("original_adj", "cf_adj", "original_x", "cf_x")
TEACHER_KINDS = ("none", "action_only", "smiles", "smiles_rf", "clear_graphpred")
PRECOMPUTED_TEACHER_ORIG = ("teacher_original_pred", "teacher_original_pred_label", "teacher_pred_before")
PRECOMPUTED_TEACHER_CF = ("teacher_cf_pred", "teacher_cf_pred_label", "teacher_pred_after")
PRECOMPUTED_TEACHER_P_ORIG = ("teacher_original_p_label", "teacher_p_before", "teacher_original_prob_label")
PRECOMPUTED_TEACHER_P_CF = ("teacher_cf_p_label", "teacher_p_after", "teacher_cf_prob_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CLEAR candidate/action pools with unified teacher-facing metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-pool", default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--dataset", default="aids")
    parser.add_argument("--teacher-kind", choices=TEACHER_KINDS, default="none")
    parser.add_argument("--teacher-path", default=DEFAULT_TEACHER_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--cf-mode", choices=CF_MODES, default="strict_flip")
    parser.add_argument("--min-cf-drop", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu.")
    parser.add_argument(
        "--clear-graphpred-h-dim",
        type=int,
        default=32,
        help="Hidden dimension used by CLEAR train_pred.py for Graph_pred_model.",
    )
    parser.add_argument("--top-k", default="1,5,10,20")
    parser.add_argument(
        "--thresholds",
        default="5,10,20,50,100,200",
        help="Cost thresholds for action-distance CCRCov summaries. Use values matching distance-method scale.",
    )
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--deduplicate-by", choices=("none", "instance_index"), default="none")
    parser.add_argument("--distance-method", choices=("action", "ged", "molclr"), default="action")
    parser.add_argument("--rank-by", choices=("total_cost", "edge_cost", "input_order"), default="total_cost")
    parser.add_argument(
        "--allow-action-only",
        action="store_true",
        help=(
            "Allow cost-only diagnostics when no unified teacher prediction source is available. "
            "Final strict FlipRate/CFDrop/CCRCov will be null/zero and must not be reported as final."
        ),
    )
    parser.add_argument("--config", default=None, help="Accepted for Slurm/config compatibility; not used.")
    parser.add_argument("--set", action="append", default=[], help="Accepted for Slurm/config compatibility; not used.")
    return parser.parse_args()


def parse_csv_numbers(raw: str, *, as_int: bool = False) -> list[Any]:
    values: list[Any] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() in {"inf", "infinity"}:
            values.append(math.inf)
        elif as_int:
            values.append(int(float(token)))
        else:
            values.append(float(token))
    return values


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                payload["_input_order"] = len(rows)
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else ("" if row.get(key) is None else row.get(key))
                    for key in fields
                }
            )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None if math.isnan(value) else "inf"
    return value


def first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def numeric_values(values: Iterable[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        number = as_float(value)
        if number is not None:
            clean.append(number)
    return clean


def mean(values: Iterable[Any]) -> float | None:
    clean = numeric_values(values)
    return float(statistics.mean(clean)) if clean else None


def median(values: Iterable[Any]) -> float | None:
    clean = numeric_values(values)
    return float(statistics.median(clean)) if clean else None


def rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def normalize_prob_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    probs: list[float] = []
    for item in value:
        number = as_float(item)
        if number is None:
            return None
        probs.append(number)
    return probs


def p_for_label(prob_vector: list[float] | None, label: int | None) -> float | None:
    if prob_vector is None or label is None or label < 0 or label >= len(prob_vector):
        return None
    return float(prob_vector[label])


def compute_action_distance(row: dict[str, Any]) -> tuple[float | None, str | None]:
    total_cost = as_float(row.get("total_cost"))
    if total_cost is not None:
        return total_cost, None
    edge_cost = as_float(row.get("edge_cost"))
    feature_cost = as_float(row.get("feature_l1_cost"))
    if edge_cost is not None and feature_cost is not None:
        return float(edge_cost + feature_cost), None
    if edge_cost is not None:
        return edge_cost, None
    return None, "missing_action_cost"


def has_full_graph_arrays(row: dict[str, Any]) -> bool:
    return all(name in row for name in FULL_GRAPH_FIELDS)


def has_smiles_pair(row: dict[str, Any]) -> bool:
    return first_present(row, SMILES_ORIGINAL_FIELDS) is not None and first_present(row, SMILES_CF_FIELDS) is not None


def has_precomputed_teacher(row: dict[str, Any]) -> bool:
    return first_present(row, PRECOMPUTED_TEACHER_ORIG) is not None and first_present(row, PRECOMPUTED_TEACHER_CF) is not None


def graph_array_shape(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return (0,)
    if isinstance(value[0], list):
        inner = len(value[0])
        return (len(value), inner)
    return (len(value),)


def collect_missing_field_counts(rows: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for field in fields:
        counts[field] = sum(1 for row in rows if row.get(field) in (None, ""))
    return counts


def resolve_device(device_arg: str | None):
    import torch

    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_clear_models_module() -> Any:
    clear_src = REPO_ROOT / "baselines" / "clear_official" / "src"
    models_path = clear_src / "models.py"
    if not models_path.exists():
        raise FileNotFoundError(f"CLEAR official models.py not found: {models_path}")
    spec = importlib.util.spec_from_file_location("clear_official_models", models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import CLEAR models.py from {models_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)
        return strip_module_prefix(payload)
    raise TypeError(f"Unsupported CLEAR graphPred checkpoint payload type: {type(payload)!r}")


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int:
    weight = state_dict.get("predictor.0.weight")
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return 2


def build_clear_graphpred_compat_model(
    *,
    x_dim: int,
    h_dim: int,
    num_classes: int,
    max_num_nodes: int,
    dataset: str,
    device: Any,
) -> Any:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import DenseGraphConv

    class ClearGraphPredCompat(nn.Module):
        """Project-side CPU-safe mirror of CLEAR Graph_pred_model for evaluation."""

        def __init__(self) -> None:
            super().__init__()
            self.num_graph_models = 3
            self.dataset = dataset
            self.graph_model = nn.ModuleList([DenseGraphConv(x_dim, h_dim) for _ in range(self.num_graph_models)])
            self.encoder = nn.Sequential(nn.Linear(2 * h_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU())
            self.predictor = nn.Sequential(nn.Linear(h_dim, num_classes))
            self.max_num_nodes = max_num_nodes
            self.mask = nn.Parameter(torch.ones(max_num_nodes), requires_grad=True)

        @staticmethod
        def graph_pooling(x: Any, pool_type: str = "mean", mask: Any = None) -> Any:
            if mask is not None:
                mask_feat = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
                x = x * mask_feat
            if pool_type == "max":
                out, _ = torch.max(x, dim=1, keepdim=False)
                return out
            return torch.sum(x, dim=1, keepdim=False)

        def forward(self, x: Any, adj: Any) -> dict[str, Any]:
            if self.dataset in {"synthetic", "community", "imdb_b"}:
                x = torch.ones_like(x).to(device)
            elif self.dataset in {"ogbg_molhiv", "aids"}:
                x = x.clone()
                x[:, :, 2:] = 0.0
                x[:, :, 0] = 0.0
            mask = None
            rep_graphs = []
            for graph_model in self.graph_model:
                rep = graph_model(x, adj, mask=mask)
                graph_rep = torch.cat(
                    [self.graph_pooling(rep, "mean", mask=mask), self.graph_pooling(rep, "max", mask=mask)],
                    dim=-1,
                )
                graph_rep = self.encoder(graph_rep)
                rep_graphs.append(graph_rep.unsqueeze(0))
            rep_graph_agg = torch.mean(torch.cat(rep_graphs, dim=0), dim=0)
            y_pred = self.predictor(rep_graph_agg)
            return {"y_pred": y_pred, "rep_graph": rep_graph_agg}

    return ClearGraphPredCompat().to(device)


def softmax_probabilities(logits: Any) -> list[list[float]]:
    import torch

    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    probs = torch.softmax(logits.float(), dim=-1)
    return [[float(item) for item in row] for row in probs.detach().cpu().tolist()]


def load_clear_graphpred_model(
    *,
    teacher_path: Path,
    dataset: str,
    x_dim: int,
    max_num_nodes: int,
    h_dim: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import torch

    if not teacher_path.exists():
        raise FileNotFoundError(f"CLEAR graphPred checkpoint not found: {teacher_path}")
    torch.manual_seed(1)
    np.random.seed(1)
    state_payload = torch.load(str(teacher_path), map_location=device)
    state_dict = extract_state_dict(state_payload)
    num_classes = infer_num_classes_from_state_dict(state_dict)
    model_class = "baselines.clear_official.src.models.Graph_pred_model"
    try:
        clear_models = load_clear_models_module()
        model = clear_models.Graph_pred_model(x_dim, h_dim, num_classes, max_num_nodes, dataset)
    except Exception as exc:  # noqa: BLE001 - keep CPU diagnostics usable without editing official source
        model_class = "project_compat.ClearGraphPredCompat"
        model = build_clear_graphpred_compat_model(
            x_dim=x_dim,
            h_dim=h_dim,
            num_classes=num_classes,
            max_num_nodes=max_num_nodes,
            dataset=dataset,
            device=device,
        )
        construction_warning = f"official_graphpred_constructor_failed:{type(exc).__name__}:{exc}"
    else:
        construction_warning = None
    model = model.to(device)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    info = {
        "teacher_kind": "clear_graphpred",
        "teacher_path": str(teacher_path),
        "model_class": model_class,
        "construction_warning": construction_warning,
        "x_dim": int(x_dim),
        "h_dim": int(h_dim),
        "num_classes": int(num_classes),
        "max_num_nodes": int(max_num_nodes),
        "device": str(device),
        "missing_state_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_state_keys": list(getattr(load_result, "unexpected_keys", [])),
    }
    return model, info


def batched_indices(total: int, batch_size: int) -> Iterable[range]:
    step = max(1, int(batch_size))
    for start in range(0, total, step):
        yield range(start, min(total, start + step))


def predict_clear_graphpred_batch(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    results: list[dict[str, Any]] = [
        {
            "teacher_eval_ok": False,
            "teacher_eval_source": "clear_graphpred",
            "teacher_error": "not_evaluated",
        }
        for _ in rows
    ]
    for batch_range in batched_indices(len(rows), batch_size):
        batch = [rows[index] for index in batch_range]
        try:
            orig_x = torch.tensor([row["original_x"] for row in batch], dtype=torch.float32, device=device)
            cf_x = torch.tensor([row["cf_x"] for row in batch], dtype=torch.float32, device=device)
            orig_adj = torch.tensor([row["original_adj"] for row in batch], dtype=torch.float32, device=device)
            cf_adj = torch.tensor([row["cf_adj"] for row in batch], dtype=torch.float32, device=device)
            with torch.no_grad():
                orig_logits = model(orig_x, orig_adj)["y_pred"]
                cf_logits = model(cf_x, cf_adj)["y_pred"]
            orig_probs = softmax_probabilities(orig_logits)
            cf_probs = softmax_probabilities(cf_logits)
            orig_preds = [int(max(range(len(prob)), key=lambda idx: prob[idx])) for prob in orig_probs]
            cf_preds = [int(max(range(len(prob)), key=lambda idx: prob[idx])) for prob in cf_probs]
        except Exception as exc:  # noqa: BLE001 - per-batch diagnostics must not hide the reason
            for index in batch_range:
                results[index] = {
                    "teacher_eval_ok": False,
                    "teacher_eval_source": "clear_graphpred",
                    "teacher_error": f"clear_graphpred_batch_failed:{type(exc).__name__}:{exc}",
                }
            continue

        for offset, index in enumerate(batch_range):
            row = rows[index]
            label = as_int(row.get("original_label"))
            original_prob_label = p_for_label(orig_probs[offset], label)
            cf_prob_label = p_for_label(cf_probs[offset], label)
            cf_drop = (
                original_prob_label - cf_prob_label
                if original_prob_label is not None and cf_prob_label is not None
                else None
            )
            strict_flip_eval = bool(cf_preds[offset] != orig_preds[offset])
            strict_flip_vs_label = bool(cf_preds[offset] != label) if label is not None else None
            results[index] = {
                "teacher_eval_ok": True,
                "teacher_eval_source": "clear_graphpred",
                "teacher_original_pred": orig_preds[offset],
                "teacher_cf_pred": cf_preds[offset],
                "teacher_p_before": original_prob_label,
                "teacher_p_after": cf_prob_label,
                "teacher_flip": strict_flip_eval,
                "cf_drop": cf_drop,
                "teacher_error": None,
                "label_used_for_p_label": label,
                "original_pred_label_eval": orig_preds[offset],
                "cf_pred_label_eval": cf_preds[offset],
                "original_pred_prob_eval": orig_probs[offset],
                "cf_pred_prob_eval": cf_probs[offset],
                "original_prob_label_eval": original_prob_label,
                "cf_prob_label_eval": cf_prob_label,
                "strict_flip_eval": strict_flip_eval,
                "strict_flip_vs_original_label_eval": strict_flip_vs_label,
                "cf_drop_eval": cf_drop,
            }
    return results


def attach_clear_graphpred_predictions(
    rows: list[dict[str, Any]],
    *,
    teacher_path: Path,
    dataset: str,
    batch_size: int,
    device_arg: str | None,
    h_dim: int,
) -> dict[str, Any]:
    missing = collect_missing_field_counts(rows, FULL_GRAPH_FIELDS)
    if any(count > 0 for count in missing.values()):
        raise RuntimeError(
            "teacher-kind=clear_graphpred requires full graph arrays in every candidate. "
            f"Missing counts: {missing}. Re-run CLEAR conversion with INCLUDE_FULL_GRAPHS=1."
        )
    first = rows[0]
    x_shape = graph_array_shape(first.get("original_x"))
    adj_shape = graph_array_shape(first.get("original_adj"))
    if x_shape is None or len(x_shape) != 2 or adj_shape is None or len(adj_shape) != 2:
        raise RuntimeError(
            "teacher-kind=clear_graphpred could not infer graph tensor dimensions from original_x/original_adj."
        )
    max_num_nodes = int(adj_shape[0])
    x_dim = int(x_shape[1])
    device = resolve_device(device_arg)
    model, info = load_clear_graphpred_model(
        teacher_path=teacher_path,
        dataset=dataset,
        x_dim=x_dim,
        max_num_nodes=max_num_nodes,
        h_dim=h_dim,
        device=device,
    )
    predictions = predict_clear_graphpred_batch(model, rows, device=device, batch_size=batch_size)
    for row, prediction in zip(rows, predictions):
        row["_clear_graphpred_eval"] = prediction
    info.update(
        {
            "batch_size": int(batch_size),
            "num_rows_evaluated": len(rows),
            "teacher_eval_ok_count": sum(1 for item in predictions if item.get("teacher_eval_ok") is True),
            "teacher_eval_error_count": sum(1 for item in predictions if item.get("teacher_eval_ok") is not True),
        }
    )
    return info


def teacher_eval_from_precomputed(row: dict[str, Any], label: int | None) -> dict[str, Any]:
    pred_before = as_int(first_present(row, PRECOMPUTED_TEACHER_ORIG))
    pred_after = as_int(first_present(row, PRECOMPUTED_TEACHER_CF))
    p_before = as_float(first_present(row, PRECOMPUTED_TEACHER_P_ORIG))
    p_after = as_float(first_present(row, PRECOMPUTED_TEACHER_P_CF))
    return {
        "teacher_eval_ok": pred_before is not None and pred_after is not None,
        "teacher_eval_source": "precomputed_teacher_fields",
        "teacher_original_pred": pred_before,
        "teacher_cf_pred": pred_after,
        "teacher_p_before": p_before,
        "teacher_p_after": p_after,
        "teacher_flip": bool(pred_before != pred_after) if pred_before is not None and pred_after is not None else None,
        "cf_drop": (p_before - p_after) if p_before is not None and p_after is not None else None,
        "teacher_error": None,
        "label_used_for_p_label": label,
    }


def teacher_eval_from_smiles(
    row: dict[str, Any],
    *,
    teacher: TeacherSemanticScorer,
    label: int,
) -> dict[str, Any]:
    original_smiles = str(first_present(row, SMILES_ORIGINAL_FIELDS) or "")
    cf_smiles = str(first_present(row, SMILES_CF_FIELDS) or "")
    before = predict_with_teacher(teacher, original_smiles, label)
    after = predict_with_teacher(teacher, cf_smiles, label)
    if not before.get("ok"):
        return {
            "teacher_eval_ok": False,
            "teacher_eval_source": "smiles_teacher",
            "teacher_original_pred": None,
            "teacher_cf_pred": None,
            "teacher_p_before": None,
            "teacher_p_after": None,
            "teacher_flip": None,
            "cf_drop": None,
            "teacher_error": f"before_failed:{before.get('error')}",
            "label_used_for_p_label": label,
        }
    if not after.get("ok"):
        return {
            "teacher_eval_ok": False,
            "teacher_eval_source": "smiles_teacher",
            "teacher_original_pred": before.get("pred_label"),
            "teacher_cf_pred": None,
            "teacher_p_before": before.get("p_label"),
            "teacher_p_after": None,
            "teacher_flip": None,
            "cf_drop": None,
            "teacher_error": f"after_failed:{after.get('error')}",
            "label_used_for_p_label": label,
        }
    p_before = before.get("p_label")
    p_after = after.get("p_label")
    return {
        "teacher_eval_ok": True,
        "teacher_eval_source": "smiles_teacher",
        "teacher_original_pred": before.get("pred_label"),
        "teacher_cf_pred": after.get("pred_label"),
        "teacher_p_before": p_before,
        "teacher_p_after": p_after,
        "teacher_flip": bool(before.get("pred_label") != after.get("pred_label")),
        "cf_drop": (float(p_before) - float(p_after)) if p_before is not None and p_after is not None else None,
        "teacher_error": None,
        "label_used_for_p_label": label,
    }


def teacher_eval_from_clear_graphpred(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("_clear_graphpred_eval")
    if isinstance(value, dict):
        return value
    return {
        "teacher_eval_ok": False,
        "teacher_eval_source": "clear_graphpred",
        "teacher_original_pred": None,
        "teacher_cf_pred": None,
        "teacher_p_before": None,
        "teacher_p_after": None,
        "teacher_flip": None,
        "cf_drop": None,
        "teacher_error": "missing_clear_graphpred_prediction",
        "label_used_for_p_label": as_int(row.get("original_label")),
    }


def cf_condition(row: dict[str, Any], *, cf_mode: str, min_cf_drop: float) -> bool | None:
    mode = normalize_cf_mode(cf_mode)
    teacher_flip = as_bool(row.get("teacher_flip"))
    cf_drop = as_float(row.get("cf_drop"))
    drop_ok = cf_drop is not None and cf_drop >= float(min_cf_drop)
    if teacher_flip is None and mode in {"strict_flip", "drop_or_flip"} and not drop_ok:
        return None
    if mode == "strict_flip":
        return bool(teacher_flip)
    if mode == "drop_or_flip":
        return bool(teacher_flip or drop_ok)
    if mode == "drop_only":
        return bool(drop_ok)
    raise ValueError(f"Unsupported cf_mode={mode}")


def evaluate_candidate(
    row: dict[str, Any],
    *,
    teacher: TeacherSemanticScorer | None,
    teacher_kind: str,
    distance_method: str,
) -> dict[str, Any]:
    label = as_int(row.get("original_label"))
    if label is None:
        label = as_int(row.get("label"))

    distance_error: str | None = None
    if distance_method == "action":
        distance, distance_error = compute_action_distance(row)
    elif distance_method in {"ged", "molclr"}:
        if not has_smiles_pair(row) and not has_full_graph_arrays(row):
            raise RuntimeError(
                f"distance-method={distance_method} requires original/counterfactual graph content. "
                "The current CLEAR candidate pool does not contain full graph arrays by default. "
                "Regenerate it with: scripts/baselines/clear/convert_clear_exports_to_candidate_pool.py "
                "--include-full-graphs, then add the corresponding graph-distance adapter."
            )
        raise RuntimeError(
            f"distance-method={distance_method} is reserved for a future CLEAR graph-distance adapter. "
            "Use --distance-method action for the current CLEAR action pool."
        )
    else:
        raise ValueError(f"Unsupported distance_method={distance_method}")

    teacher_eval: dict[str, Any]
    if teacher_kind == "clear_graphpred":
        teacher_eval = teacher_eval_from_clear_graphpred(row)
    elif has_precomputed_teacher(row):
        teacher_eval = teacher_eval_from_precomputed(row, label)
    elif teacher_kind in {"smiles", "smiles_rf"} and has_smiles_pair(row) and teacher is not None and teacher.available and label is not None:
        teacher_eval = teacher_eval_from_smiles(row, teacher=teacher, label=label)
    else:
        teacher_eval = {
            "teacher_eval_ok": False,
            "teacher_eval_source": teacher_kind if teacher_kind in {"none", "action_only"} else "unavailable",
            "teacher_original_pred": None,
            "teacher_cf_pred": None,
            "teacher_p_before": None,
            "teacher_p_after": None,
            "teacher_flip": None,
            "cf_drop": None,
            "teacher_error": (
                "no_unified_teacher_prediction_source: candidate lacks original_smiles/cf_smiles "
                "or precomputed teacher fields; default CLEAR pool also omits full graph arrays"
            ),
            "label_used_for_p_label": label,
        }

    candidate = {
        "candidate_id": row.get("candidate_id"),
        "record_index": row.get("_input_order"),
        "source": row.get("source", "CLEAR"),
        "dataset": row.get("dataset"),
        "exp_id": row.get("exp_id"),
        "split": row.get("split"),
        "instance_index": row.get("instance_index"),
        "original_label": label,
        "target_cf_label": row.get("target_cf_label"),
        "official_original_pred_label": row.get("official_original_pred_label"),
        "official_cf_pred_label": row.get("official_cf_pred_label"),
        "official_flip": row.get("official_flip"),
        "official_target_success": row.get("official_target_success"),
        "official_original_correct": row.get("official_original_correct"),
        "official_original_pred_prob": row.get("official_original_pred_prob"),
        "official_cf_pred_prob": row.get("official_cf_pred_prob"),
        "teacher_kind": teacher_kind,
        "distance_method": distance_method,
        "distance": distance,
        "distance_ok": distance is not None,
        "distance_error": distance_error,
        "edge_cost": row.get("edge_cost"),
        "total_cost": row.get("total_cost"),
        "num_edge_added": row.get("num_edge_added"),
        "num_edge_deleted": row.get("num_edge_deleted"),
        "num_edge_changed": row.get("num_edge_changed"),
        "edge_added_count": row.get("num_edge_added"),
        "edge_deleted_count": row.get("num_edge_deleted"),
        "edge_changed_count": row.get("num_edge_changed"),
        "action_edges_added": row.get("action_edges_added"),
        "action_edges_deleted": row.get("action_edges_deleted"),
        "feature_l1_cost": row.get("feature_l1_cost"),
        "feature_l2_cost": row.get("feature_l2_cost"),
        "num_node_feature_changed": row.get("num_node_feature_changed"),
        "changed_node_indices": row.get("changed_node_indices"),
        **teacher_eval,
        "has_smiles_pair": has_smiles_pair(row),
        "has_full_graph_arrays": has_full_graph_arrays(row),
    }
    return candidate


def load_and_prepare_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.candidate_pool).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CLEAR candidate pool not found: {path}")
    raw_rows = read_jsonl(path)
    if args.max_candidates is not None:
        raw_rows = raw_rows[: int(args.max_candidates)]

    if args.deduplicate_by == "instance_index":
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for row in raw_rows:
            key = str(row.get("instance_index"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        raw_rows = deduped
    return raw_rows


def rank_candidates(rows: list[dict[str, Any]], rank_by: str) -> list[dict[str, Any]]:
    if rank_by == "input_order":
        return sorted(rows, key=lambda row: int(row.get("_eval_order", row.get("_input_order", 0)) or 0))
    if rank_by == "edge_cost":
        return sorted(rows, key=lambda row: (as_float(row.get("edge_cost")) is None, as_float(row.get("edge_cost")) or 1e18))
    return sorted(rows, key=lambda row: (as_float(row.get("distance")) is None, as_float(row.get("distance")) or 1e18))


def action_signature(row: dict[str, Any]) -> set[str]:
    signature: set[str] = set()
    for edge in row.get("action_edges_added") or []:
        signature.add(f"+e:{edge}")
    for edge in row.get("action_edges_deleted") or []:
        signature.add(f"-e:{edge}")
    for node in row.get("changed_node_indices") or []:
        signature.add(f"xf:{node}")
    return signature


def structural_redundancy(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return 0.0
    signatures = [action_signature(row) for row in rows]
    total = 0.0
    count = 0
    for i, left in enumerate(signatures):
        for right in signatures[i + 1 :]:
            union = len(left | right)
            total += (len(left & right) / union) if union else 0.0
            count += 1
    return float(total / count) if count else 0.0


def coverage_redundancy(rows: list[dict[str, Any]], *, threshold: float, cf_mode: str, min_cf_drop: float) -> float:
    cover_sets: list[set[str]] = []
    for row in rows:
        distance = as_float(row.get("distance"))
        if distance is None or distance > threshold:
            continue
        condition = cf_condition(row, cf_mode=cf_mode, min_cf_drop=min_cf_drop)
        if condition is not True:
            continue
        instance = row.get("instance_index")
        if instance is not None:
            cover_sets.append({str(instance)})
    if len(cover_sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i, left in enumerate(cover_sets):
        for right in cover_sets[i + 1 :]:
            union = len(left | right)
            total += (len(left & right) / union) if union else 0.0
            count += 1
    return float(total / count) if count else 0.0


def build_summary(
    evaluated_rows: list[dict[str, Any]],
    ranked_rows: list[dict[str, Any]],
    *,
    top_ks: list[int],
    thresholds: list[float],
    cf_mode: str,
    min_cf_drop: float,
    dataset: str,
    distance_method: str,
    teacher_kind: str,
    teacher_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_ids = {str(row.get("instance_index")) for row in evaluated_rows if row.get("instance_index") is not None}
    num_parents = len(parent_ids)
    teacher_eval_count = sum(1 for row in evaluated_rows if row.get("teacher_eval_ok") is True)
    teacher_available = teacher_eval_count > 0
    threshold_rows: list[dict[str, Any]] = []

    for k in top_ks:
        top_rows = ranked_rows[: min(int(k), len(ranked_rows))]
        for threshold in thresholds:
            close_rows = [
                row
                for row in top_rows
                if as_float(row.get("distance")) is not None and float(row["distance"]) <= float(threshold)
            ]
            close_parent_ids = {str(row.get("instance_index")) for row in close_rows if row.get("instance_index") is not None}
            cf_rows = [
                row
                for row in close_rows
                if cf_condition(row, cf_mode=cf_mode, min_cf_drop=min_cf_drop) is True
            ]
            cf_parent_ids = {str(row.get("instance_index")) for row in cf_rows if row.get("instance_index") is not None}
            threshold_rows.append(
                {
                    "method": "CLEAR",
                    "dataset": dataset,
                    "teacher_kind": teacher_kind,
                    "teacher_path": teacher_path,
                    "distance_method": distance_method,
                    "cf_mode": cf_mode,
                    "min_cf_drop": float(min_cf_drop),
                    "top_k": int(k),
                    "threshold": float(threshold) if math.isfinite(float(threshold)) else "inf",
                    "num_parents": num_parents,
                    "num_candidates": len(evaluated_rows),
                    "num_selected_candidates": len(top_rows),
                    "num_close_only_covered": len(close_parent_ids),
                    "close_only_coverage": rate(len(close_parent_ids), num_parents),
                    "num_close_cf_covered": len(cf_parent_ids),
                    "close_cf_coverage": rate(len(cf_parent_ids), num_parents) if teacher_available else None,
                    "SuppCov": rate(len(close_parent_ids), num_parents),
                    "CCRCov@K": rate(len(cf_parent_ids), num_parents) if teacher_available else None,
                    "FlipRate": mean(1.0 if row.get("teacher_flip") is True else 0.0 for row in close_rows if row.get("teacher_eval_ok") is True),
                    "CFDrop": mean(row.get("cf_drop") for row in cf_rows),
                    "CostMean": mean(row.get("distance") for row in cf_rows) if cf_rows else mean(row.get("distance") for row in close_rows),
                    "CostMedian": median(row.get("distance") for row in cf_rows) if cf_rows else median(row.get("distance") for row in close_rows),
                    "StructRed": structural_redundancy(top_rows),
                    "CovRed": coverage_redundancy(top_rows, threshold=float(threshold), cf_mode=cf_mode, min_cf_drop=min_cf_drop),
                    "ValidRate": rate(sum(1 for row in top_rows if row.get("distance_ok")), len(top_rows)),
                    "teacher_eval_ok_rate": rate(sum(1 for row in top_rows if row.get("teacher_eval_ok")), len(top_rows)),
                    "official_flip_rate_selected": mean(1.0 if as_bool(row.get("official_flip")) else 0.0 for row in top_rows),
                    "mean_edge_changed_selected": mean(row.get("num_edge_changed") for row in top_rows),
                    "mean_feature_l1_selected": mean(row.get("feature_l1_cost") for row in top_rows),
                }
            )

    overall = {
        "method": "CLEAR",
        "dataset": dataset,
        "teacher_kind": teacher_kind,
        "teacher_path": teacher_path,
        "distance_method": distance_method,
        "cf_mode": cf_mode,
        "min_cf_drop": float(min_cf_drop),
        "num_candidates": len(evaluated_rows),
        "num_parents": num_parents,
        "teacher_eval_count": teacher_eval_count,
        "teacher_eval_ok_rate": rate(teacher_eval_count, len(evaluated_rows)),
        "teacher_available_for_final_metrics": teacher_available,
        "official_flip_count": sum(1 for row in evaluated_rows if as_bool(row.get("official_flip")) is True),
        "official_flip_rate": mean(1.0 if as_bool(row.get("official_flip")) else 0.0 for row in evaluated_rows),
        "official_target_success_rate": mean(
            1.0 if as_bool(row.get("official_target_success")) else 0.0 for row in evaluated_rows
        ),
        "official_original_correct_rate": mean(
            1.0 if as_bool(row.get("official_original_correct")) else 0.0 for row in evaluated_rows
        ),
        "eval_original_correct_rate": mean(
            1.0
            if as_int(row.get("original_pred_label_eval")) == as_int(row.get("original_label"))
            else 0.0
            for row in evaluated_rows
            if as_int(row.get("original_pred_label_eval")) is not None and as_int(row.get("original_label")) is not None
        ),
        "strict_flip_rate_eval": mean(
            1.0 if as_bool(row.get("strict_flip_eval")) else 0.0
            for row in evaluated_rows
            if as_bool(row.get("strict_flip_eval")) is not None
        ),
        "strict_flip_vs_original_label_rate_eval": mean(
            1.0 if as_bool(row.get("strict_flip_vs_original_label_eval")) else 0.0
            for row in evaluated_rows
            if as_bool(row.get("strict_flip_vs_original_label_eval")) is not None
        ),
        "mean_cf_drop_eval": mean(row.get("cf_drop_eval") for row in evaluated_rows),
        "mean_original_prob_label_eval": mean(row.get("original_prob_label_eval") for row in evaluated_rows),
        "mean_cf_prob_label_eval": mean(row.get("cf_prob_label_eval") for row in evaluated_rows),
        "eval_vs_official_flip_agreement": mean(
            1.0 if as_bool(row.get("official_flip")) == as_bool(row.get("strict_flip_eval")) else 0.0
            for row in evaluated_rows
            if as_bool(row.get("official_flip")) is not None and as_bool(row.get("strict_flip_eval")) is not None
        ),
        "mean_distance": mean(row.get("distance") for row in evaluated_rows),
        "median_distance": median(row.get("distance") for row in evaluated_rows),
        "mean_edge_changed": mean(row.get("num_edge_changed") for row in evaluated_rows),
        "mean_edge_added": mean(row.get("num_edge_added") for row in evaluated_rows),
        "mean_edge_deleted": mean(row.get("num_edge_deleted") for row in evaluated_rows),
        "mean_feature_l1_cost": mean(row.get("feature_l1_cost") for row in evaluated_rows),
        "mean_total_cost": mean(row.get("total_cost") for row in evaluated_rows),
        "missing_field_counts": collect_missing_field_counts(evaluated_rows, FULL_GRAPH_FIELDS),
        "note": (
            "Final FlipRate/CFDrop/CCRCov use unified teacher fields only. "
            "CLEAR official_flip is retained as a diagnostic and never used as final strict flip."
        ),
    }
    return threshold_rows, overall


def write_report(path: Path, *, overall: dict[str, Any], threshold_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CLEAR Unified Candidate Pool Evaluation",
        "",
        f"- dataset: {overall.get('dataset')}",
        f"- teacher_kind: {overall.get('teacher_kind')}",
        f"- teacher_path: {overall.get('teacher_path')}",
        f"- distance_method: {overall.get('distance_method')}",
        f"- CF mode: {overall.get('cf_mode')}",
        f"- num_candidates: {overall.get('num_candidates')}",
        f"- num_parents: {overall.get('num_parents')}",
        f"- teacher_eval_ok_rate: {overall.get('teacher_eval_ok_rate')}",
        f"- official_flip_rate_diagnostic_only: {overall.get('official_flip_rate')}",
        f"- strict_flip_rate_eval: {overall.get('strict_flip_rate_eval')}",
        f"- strict_flip_vs_original_label_rate_eval: {overall.get('strict_flip_vs_original_label_rate_eval')}",
        f"- mean_cf_drop_eval: {overall.get('mean_cf_drop_eval')}",
        "",
        "CLEAR official flip/validity fields are diagnostics only. Final FlipRate, CFDrop, and CCRCov must use the unified teacher/oracle.",
        "",
        "## Top-K Summary",
        "",
        "| top_k | threshold | SuppCov | CCRCov@K | FlipRate | CFDrop | CostMean | teacher_eval_ok_rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_rows:
        lines.append(
            "| {top_k} | {threshold} | {SuppCov} | {ccrcov} | {flip} | {drop} | {cost} | {teacher} |".format(
                top_k=row.get("top_k"),
                threshold=row.get("threshold"),
                SuppCov=_fmt(row.get("SuppCov")),
                ccrcov=_fmt(row.get("CCRCov@K")),
                flip=_fmt(row.get("FlipRate")),
                drop=_fmt(row.get("CFDrop")),
                cost=_fmt(row.get("CostMean")),
                teacher=_fmt(row.get("teacher_eval_ok_rate")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def main() -> int:
    args = parse_args()
    candidate_pool = Path(args.candidate_pool).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cf_mode = normalize_cf_mode(args.cf_mode)
    teacher_kind = str(args.teacher_kind).strip().lower()
    action_only_mode = bool(args.allow_action_only or teacher_kind in {"none", "action_only"})
    top_ks = parse_csv_numbers(args.top_k, as_int=True)
    thresholds = parse_csv_numbers(args.thresholds)
    if not top_ks:
        raise ValueError("--top-k must contain at least one K value")
    if not thresholds:
        thresholds = [math.inf]

    print("[CLEAR_EVAL_CONFIG]")
    print(f"candidate_pool={candidate_pool}")
    print(f"dataset={args.dataset}")
    print(f"teacher_kind={teacher_kind}")
    print(f"teacher_path={args.teacher_path}")
    print(f"batch_size={args.batch_size}")
    print(f"device={args.device or 'auto'}")
    print(f"out_dir={out_dir}")
    print(f"cf_mode={cf_mode}")
    print(f"top_k={top_ks}")
    print(f"thresholds={thresholds}")
    print(f"distance_method={args.distance_method}")
    print(f"allow_action_only={args.allow_action_only}")
    print(f"effective_action_only_mode={action_only_mode}")

    raw_rows = load_and_prepare_candidates(args)
    if not raw_rows:
        raise ValueError(f"No candidates found in {candidate_pool}")

    has_any_smiles = any(has_smiles_pair(row) for row in raw_rows)
    has_any_full_graph = any(has_full_graph_arrays(row) for row in raw_rows)
    has_any_precomputed_teacher = any(has_precomputed_teacher(row) for row in raw_rows)

    teacher = None
    graphpred_info: dict[str, Any] = {}
    if teacher_kind == "clear_graphpred":
        teacher_path = Path(args.teacher_path or DEFAULT_CLEAR_GRAPHPRED_PATH).expanduser()
        if not teacher_path.is_absolute():
            teacher_path = REPO_ROOT / teacher_path
        graphpred_info = attach_clear_graphpred_predictions(
            raw_rows,
            teacher_path=teacher_path,
            dataset=args.dataset,
            batch_size=int(args.batch_size),
            device_arg=args.device,
            h_dim=int(args.clear_graphpred_h_dim),
        )
    elif teacher_kind in {"smiles", "smiles_rf"} and has_any_smiles and args.teacher_path:
        teacher = TeacherSemanticScorer(args.teacher_path)

    if args.distance_method != "action" and not (has_any_smiles or has_any_full_graph):
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] CLEAR candidate pool lacks full graph arrays or SMILES required for non-action "
            "distance evaluation. Regenerate the pool with --include-full-graphs, or use --distance-method action."
        )

    if (
        teacher_kind not in {"clear_graphpred", "smiles", "smiles_rf"}
        and not has_any_precomputed_teacher
        and not action_only_mode
    ):
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] teacher-kind must be clear_graphpred, smiles/smiles_rf, or a candidate pool must provide "
            "precomputed teacher fields. Use --teacher-kind none or --allow-action-only for diagnostics only."
        )

    if not has_any_smiles and not has_any_precomputed_teacher and teacher_kind in {"smiles", "smiles_rf"} and not action_only_mode:
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] Unified teacher/oracle evaluation cannot run on this CLEAR candidate pool because it lacks "
            "original_smiles/cf_smiles and precomputed teacher_* fields. The default conversion omits full graph arrays, "
            "so no graph-teacher adapter can reconstruct predictions from this JSONL. Re-run conversion with "
            "--include-full-graphs and provide a graph-teacher adapter, or use --allow-action-only for diagnostic "
            "cost/SuppCov summaries only. CLEAR official_flip is not used as final flip."
        )

    evaluated: list[dict[str, Any]] = []
    for eval_order, row in enumerate(raw_rows):
        result = evaluate_candidate(row, teacher=teacher, teacher_kind=teacher_kind, distance_method=args.distance_method)
        result["_eval_order"] = eval_order
        evaluated.append(result)

    if not action_only_mode and not any(row.get("teacher_eval_ok") for row in evaluated):
        raise SystemExit(
            "[CLEAR_EVAL_ERROR] No candidate received a successful unified teacher evaluation. This usually means the pool has no SMILES "
            "or the supplied teacher cannot score the candidate representation. Final CLEAR metrics are not available."
        )

    ranked = rank_candidates(evaluated, args.rank_by)
    threshold_rows, overall = build_summary(
        evaluated,
        ranked,
        top_ks=[int(k) for k in top_ks],
        thresholds=[float(t) for t in thresholds],
        cf_mode=cf_mode,
        min_cf_drop=float(args.min_cf_drop),
        dataset=args.dataset,
        distance_method=args.distance_method,
        teacher_kind=teacher_kind,
        teacher_path=args.teacher_path or (DEFAULT_CLEAR_GRAPHPRED_PATH if teacher_kind == "clear_graphpred" else ""),
    )
    overall.update(
        {
            "candidate_pool": str(candidate_pool),
            "out_dir": str(out_dir),
            "teacher_kind": teacher_kind,
            "teacher_path": args.teacher_path or (DEFAULT_CLEAR_GRAPHPRED_PATH if teacher_kind == "clear_graphpred" else ""),
            "clear_graphpred_info": graphpred_info,
            "rank_by": args.rank_by,
            "deduplicate_by": args.deduplicate_by,
            "max_candidates": args.max_candidates,
            "allow_action_only": args.allow_action_only,
            "effective_action_only_mode": action_only_mode,
            "has_any_smiles_pair": has_any_smiles,
            "has_any_full_graph_arrays": has_any_full_graph,
            "has_any_precomputed_teacher": has_any_precomputed_teacher,
            "missing_field_counts": collect_missing_field_counts(raw_rows, FULL_GRAPH_FIELDS),
        }
    )

    per_candidate_path = out_dir / "per_candidate_eval.jsonl"
    threshold_summary_path = out_dir / "threshold_summary.csv"
    summary_json_path = out_dir / "summary.json"
    summary_csv_path = out_dir / "summary.csv"
    report_path = out_dir / "report.md"
    write_jsonl(per_candidate_path, evaluated)
    write_csv(threshold_summary_path, threshold_rows)
    write_json(summary_json_path, overall)
    write_csv(summary_csv_path, [overall])
    write_report(report_path, overall=overall, threshold_rows=threshold_rows)

    print("[CLEAR_EVAL_PREVIEW]")
    for row in evaluated[:3]:
        print(
            json.dumps(
                {
                    "candidate_id": row.get("candidate_id"),
                    "instance_index": row.get("instance_index"),
                    "distance": row.get("distance"),
                    "official_flip": row.get("official_flip"),
                    "teacher_eval_ok": row.get("teacher_eval_ok"),
                    "teacher_flip": row.get("teacher_flip"),
                    "strict_flip_eval": row.get("strict_flip_eval"),
                    "strict_flip_vs_original_label_eval": row.get("strict_flip_vs_original_label_eval"),
                    "cf_drop": row.get("cf_drop"),
                    "cf_drop_eval": row.get("cf_drop_eval"),
                },
                sort_keys=True,
            )
        )
    print("[CLEAR_EVAL_DONE]")
    print(f"per_candidate_eval={per_candidate_path}")
    print(f"threshold_summary={threshold_summary_path}")
    print(f"summary_json={summary_json_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
