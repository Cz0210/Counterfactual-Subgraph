"""Pair generation for GREED-style HIV distance supervision."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

from src.eval.close_counterfactual_coverage import hard_delete_substructure_any_match
from src.eval.greed_distance.graph_conversion import graph_from_smiles, read_graphs_jsonl
from src.utils.io import ensure_directory

OURS_FRAGMENT_FIELDS = (
    "final_fragment",
    "core_fragment",
    "fragment",
    "selected_fragment",
    "smiles",
    "subgraph_smiles",
)
GT_FULLGRAPH_FIELDS = (
    "fullgraph_smiles",
    "counterfactual_smiles",
    "cf_smiles",
    "graph_smiles",
    "candidate_smiles",
    "final_smiles",
    "smiles",
)
PAIR_FIELDS = [
    "pair_id",
    "split",
    "pair_type",
    "graph_a_id",
    "graph_b_id",
    "smiles_a",
    "smiles_b",
    "label_a",
    "label_b",
    "source",
    "num_atoms_a",
    "num_atoms_b",
    "num_bonds_a",
    "num_bonds_b",
    "num_removed_atoms",
    "num_removed_bonds",
    "candidate_id",
    "fragment_smiles",
    "residual_smiles",
]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        for name in (
            "selected_subgraphs.csv",
            "selected_subgraphs.jsonl",
            "selected_subgraphs.json",
            "gt_selected_fullgraphs.csv",
            "selected_fullgraphs.csv",
            "candidate_pool.jsonl",
            "candidate_pool.csv",
        ):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("selected_rows", "candidates", "rows", "items", "data"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key] if isinstance(row, dict)]
            return [payload]
    raise ValueError(f"unsupported candidate file: {path}")


def _coalesce(row: dict[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def load_candidate_smiles(path: str | Path | None, fields: Sequence[str]) -> list[dict[str, Any]]:
    if path is None or not str(path).strip():
        return []
    rows = _read_rows(Path(path).expanduser())
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        smiles = _coalesce(row, fields)
        if not smiles:
            continue
        key = smiles
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "candidate_id": str(row.get("candidate_id") or row.get("id") or row.get("rank") or index),
                "smiles": smiles,
                "raw": row,
            }
        )
    return candidates


def _pair_row(
    *,
    pair_id: str,
    split: str,
    pair_type: str,
    graph_a: dict[str, Any],
    graph_b: dict[str, Any],
    source: str,
    candidate_id: str = "",
    fragment_smiles: str = "",
    residual_smiles: str = "",
    num_removed_atoms: int | None = None,
    num_removed_bonds: int | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "split": split,
        "pair_type": pair_type,
        "graph_a_id": graph_a.get("graph_id"),
        "graph_b_id": graph_b.get("graph_id"),
        "smiles_a": graph_a.get("smiles"),
        "smiles_b": graph_b.get("smiles"),
        "label_a": graph_a.get("label"),
        "label_b": graph_b.get("label"),
        "source": source,
        "num_atoms_a": graph_a.get("num_atoms"),
        "num_atoms_b": graph_b.get("num_atoms"),
        "num_bonds_a": graph_a.get("num_bonds"),
        "num_bonds_b": graph_b.get("num_bonds"),
        "num_removed_atoms": num_removed_atoms,
        "num_removed_bonds": num_removed_bonds,
        "candidate_id": candidate_id,
        "fragment_smiles": fragment_smiles,
        "residual_smiles": residual_smiles,
    }


def generate_pair_rows(
    *,
    graphs_jsonl: str | Path,
    split: str,
    num_pairs: int,
    seed: int,
    include_deletion_pairs: bool,
    include_fullgraph_pairs: bool,
    include_random_pairs: bool,
    ours_selected_path: str | Path | None = None,
    gt_fullgraph_candidates_path: str | Path | None = None,
    max_parents: int | None = None,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    graphs = read_graphs_jsonl(graphs_jsonl, parse_ok_only=True)
    if max_parents is not None:
        graphs = graphs[: int(max_parents)]
    if not graphs:
        return []

    ours_candidates = load_candidate_smiles(ours_selected_path, OURS_FRAGMENT_FIELDS)
    gt_candidates = load_candidate_smiles(gt_fullgraph_candidates_path, GT_FULLGRAPH_FIELDS)
    if max_candidates is not None:
        ours_candidates = ours_candidates[: int(max_candidates)]
        gt_candidates = gt_candidates[: int(max_candidates)]

    rows: list[dict[str, Any]] = []
    pair_index = 0

    if include_deletion_pairs and ours_candidates:
        for graph in graphs:
            for candidate in ours_candidates:
                deletions = hard_delete_substructure_any_match(str(graph.get("smiles") or ""), candidate["smiles"])
                for deletion in deletions:
                    if not deletion.get("delete_valid") or not deletion.get("residual_smiles"):
                        continue
                    graph_b = graph_from_smiles(
                        str(deletion["residual_smiles"]),
                        graph_id=f"{graph.get('graph_id')}:del:{candidate['candidate_id']}:{deletion.get('match_index')}",
                        label=graph.get("label"),
                    )
                    if not graph_b.get("parse_ok"):
                        continue
                    rows.append(
                        _pair_row(
                            pair_id=f"{split}_{pair_index}",
                            split=split,
                            pair_type="ours_deletion",
                            graph_a=graph,
                            graph_b=graph_b,
                            source="ours_selected_hard_deletion",
                            candidate_id=str(candidate["candidate_id"]),
                            fragment_smiles=str(candidate["smiles"]),
                            residual_smiles=str(deletion["residual_smiles"]),
                            num_removed_atoms=int(deletion.get("num_removed_atoms") or 0),
                            num_removed_bonds=int(deletion.get("num_removed_bonds") or 0),
                        )
                    )
                    pair_index += 1
                    if len(rows) >= int(num_pairs):
                        return rows

    if include_fullgraph_pairs and gt_candidates:
        for graph in graphs:
            for candidate in gt_candidates:
                graph_b = graph_from_smiles(
                    str(candidate["smiles"]),
                    graph_id=f"gt:{candidate['candidate_id']}",
                    label=None,
                )
                if not graph_b.get("parse_ok"):
                    continue
                rows.append(
                    _pair_row(
                        pair_id=f"{split}_{pair_index}",
                        split=split,
                        pair_type="gt_fullgraph",
                        graph_a=graph,
                        graph_b=graph_b,
                        source="gt_fullgraph_candidate",
                        candidate_id=str(candidate["candidate_id"]),
                    )
                )
                pair_index += 1
                if len(rows) >= int(num_pairs):
                    return rows

    if include_random_pairs:
        attempts = 0
        while len(rows) < int(num_pairs) and attempts < int(num_pairs) * 20:
            graph_a, graph_b = rng.sample(graphs, 2) if len(graphs) >= 2 else (graphs[0], graphs[0])
            rows.append(
                _pair_row(
                    pair_id=f"{split}_{pair_index}",
                    split=split,
                    pair_type="random_hiv_pair",
                    graph_a=graph_a,
                    graph_b=graph_b,
                    source="random_hiv_pair",
                )
            )
            pair_index += 1
            attempts += 1
    rng.shuffle(rows)
    return rows[: int(num_pairs)]


def write_pairs_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path).expanduser().resolve()
    ensure_directory(destination.parent)
    fieldnames = list(PAIR_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
