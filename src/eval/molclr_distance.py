"""MolCLR embedding precompute and distance lookup for CCRCov."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.embeddings.molclr_gnn_embedding import encode_smiles_list_with_failures
from src.eval.close_counterfactual_coverage import (
    hard_delete_substructure_any_match,
    _load_candidate_records,
    _load_parent_records,
)
from src.eval.greed_distance.pair_generation import GT_FULLGRAPH_FIELDS, OURS_FRAGMENT_FIELDS
from src.utils.io import ensure_directory


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(f"embedding dimensions differ: {len(vec_a)} vs {len(vec_b)}")
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("zero-norm embedding")
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def embedding_distance(vec_a: list[float], vec_b: list[float]) -> tuple[float, float]:
    cosine = cosine_similarity(vec_a, vec_b)
    return float(1.0 - cosine), float(cosine)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_embedding(value: Any) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    try:
        vector = [float(item) for item in value]
    except Exception:
        return None
    return vector if vector else None


def _load_embedding_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return []


class MolCLREmbeddingDistanceLookup:
    """Lookup precomputed embeddings and return 1 - cosine distance."""

    def __init__(self, embedding_dir: str | Path) -> None:
        root = Path(embedding_dir).expanduser().resolve()
        self.embedding_dir = root
        rows: list[dict[str, Any]] = []
        for name in (
            "parents.jsonl",
            "ours_residuals.jsonl",
            "gt_fullgraph_candidates.jsonl",
            "clear_rf_fullgraph_candidates.jsonl",
            "all_embeddings.jsonl",
        ):
            rows.extend(_load_embedding_rows(root / name))
        self.embeddings: dict[str, list[float]] = {}
        for row in rows:
            smiles = str(row.get("smiles") or "").strip()
            vector = _parse_embedding(row.get("embedding"))
            if smiles and vector:
                self.embeddings[smiles] = vector

    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        vec_a = self.embeddings.get(str(smiles_a).strip())
        vec_b = self.embeddings.get(str(smiles_b).strip())
        if vec_a is None or vec_b is None:
            return {"distance": None, "cosine_similarity": None, "ok": False, "error": "embedding_missing"}
        try:
            dist, cosine = embedding_distance(vec_a, vec_b)
            return {"distance": dist, "cosine_similarity": cosine, "ok": True, "error": None}
        except Exception as exc:
            return {"distance": None, "cosine_similarity": None, "ok": False, "error": str(exc)}


def _collect_unique_smiles(rows: Iterable[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        smiles = str(row.get("smiles") or "").strip()
        if smiles and smiles not in seen:
            seen.add(smiles)
            out.append(smiles)
    return out


def precompute_molclr_embeddings_for_ccrcov(
    *,
    dataset_csv: str | Path,
    ours_selected_path: str | Path | None,
    gt_fullgraph_candidates_path: str | Path | None,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    output_dir: str | Path,
    clear_fullgraph_candidates_path: str | Path | None = None,
    label: int = 1,
    smiles_col: str = "smiles",
    label_col: str = "label",
    max_parents: int | None = None,
    max_candidates: int | None = None,
    encoder_type: str = "gin",
    batch_size: int = 64,
    device: str = "cuda",
    invalid_policy: str = "skip",
) -> dict[str, Any]:
    output_root = ensure_directory(Path(output_dir).expanduser().resolve())
    _dataset_path, parents, _actual_label_col = _load_parent_records(
        dataset_csv,
        label=int(label),
        smiles_col=smiles_col,
        label_col=label_col,
        max_parents=max_parents,
    )
    ours_candidates = []
    gt_candidates = []
    clear_candidates = []
    if ours_selected_path:
        _ours_path, ours_candidates = _load_candidate_records(
            ours_selected_path,
            fields=OURS_FRAGMENT_FIELDS,
            directory_candidates=("selected_subgraphs.csv", "selected_subgraphs.json", "selected_subgraphs.jsonl", "candidate_pool.jsonl"),
        )
    if gt_fullgraph_candidates_path:
        _gt_path, gt_candidates = _load_candidate_records(
            gt_fullgraph_candidates_path,
            fields=GT_FULLGRAPH_FIELDS,
            directory_candidates=("gt_selected_fullgraphs.csv", "selected_fullgraphs.csv", "selected_subgraphs.csv", "candidate_pool.jsonl", "candidate_pool.csv"),
        )
    if clear_fullgraph_candidates_path:
        _clear_path, clear_candidates = _load_candidate_records(
            clear_fullgraph_candidates_path,
            fields=GT_FULLGRAPH_FIELDS,
            directory_candidates=("gt_selected_fullgraphs.csv", "selected_fullgraphs.csv", "selected_subgraphs.csv", "candidate_pool.jsonl", "candidate_pool.csv"),
        )
    if max_candidates is not None:
        ours_candidates = ours_candidates[: int(max_candidates)]
        gt_candidates = gt_candidates[: int(max_candidates)]
        clear_candidates = clear_candidates[: int(max_candidates)]

    parent_rows = [{"graph_id": parent.parent_id, "smiles": parent.smiles, "kind": "parent"} for parent in parents]
    residual_rows: list[dict[str, Any]] = []
    for parent in parents:
        for candidate in ours_candidates:
            for deletion in hard_delete_substructure_any_match(parent.smiles, candidate.smiles):
                if deletion.get("delete_valid") and deletion.get("residual_smiles"):
                    residual_rows.append(
                        {
                            "graph_id": f"{parent.parent_id}:{candidate.candidate_id}:{deletion.get('match_index')}",
                            "smiles": deletion["residual_smiles"],
                            "kind": "ours_residual",
                        }
                    )
    gt_rows = [
        {"graph_id": candidate.candidate_id, "smiles": candidate.smiles, "kind": "gt_fullgraph_candidate"}
        for candidate in gt_candidates
    ]
    clear_rows = [
        {"graph_id": candidate.candidate_id, "smiles": candidate.smiles, "kind": "clear_rf_fullgraph_candidate"}
        for candidate in clear_candidates
    ]
    all_smiles = _collect_unique_smiles(parent_rows + residual_rows + gt_rows + clear_rows)
    result = encode_smiles_list_with_failures(
        all_smiles,
        molclr_root=molclr_root,
        molclr_ckpt=molclr_checkpoint,
        encoder_type=encoder_type,
        batch_size=int(batch_size),
        device=device,
        invalid_policy=invalid_policy,  # type: ignore[arg-type]
    )

    def attach_embeddings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            embedding = result.embeddings.get(row["smiles"])
            out.append(
                {
                    **row,
                    "embedding": embedding,
                    "embedding_dim": len(embedding) if embedding else None,
                    "parse_ok": embedding is not None,
                    "error": None if embedding is not None else "embedding_missing_or_invalid_smiles",
                }
            )
        return out

    parent_out = attach_embeddings(parent_rows)
    residual_out = attach_embeddings(residual_rows)
    gt_out = attach_embeddings(gt_rows)
    clear_out = attach_embeddings(clear_rows)
    all_out = attach_embeddings([{"graph_id": smiles, "smiles": smiles, "kind": "unique"} for smiles in all_smiles])
    _write_jsonl(output_root / "parents.jsonl", parent_out)
    _write_jsonl(output_root / "ours_residuals.jsonl", residual_out)
    _write_jsonl(output_root / "gt_fullgraph_candidates.jsonl", gt_out)
    _write_jsonl(output_root / "clear_rf_fullgraph_candidates.jsonl", clear_out)
    _write_jsonl(output_root / "all_embeddings.jsonl", all_out)
    failed_rows = [{"smiles": item.smiles, "error": item.error, "failure_reason": item.failure_reason} for item in result.failed_smiles]
    _write_jsonl(output_root / "failed_smiles.jsonl", failed_rows)
    summary = {
        "output_dir": str(output_root),
        "num_parents": len(parent_rows),
        "num_ours_residuals": len(residual_rows),
        "num_gt_fullgraph_candidates": len(gt_rows),
        "num_clear_rf_fullgraph_candidates": len(clear_rows),
        "num_unique_smiles": len(all_smiles),
        "num_embeddings": len(result.embeddings),
        "embedding_dim": result.embedding_dim,
        "num_failed_smiles": len(result.failed_smiles),
        "invalid_policy": invalid_policy,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
