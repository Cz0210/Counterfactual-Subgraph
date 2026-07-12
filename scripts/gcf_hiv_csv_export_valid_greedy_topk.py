#!/usr/bin/env python3
"""Export a validity-filtered greedy Top-K GCFExplainer-HIVCSV candidate set."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gcf_hiv_csv_export_summary import (  # noqa: E402
    _finite_float_or_default,
    _greedy,
    _indices_from_sparse,
)
from src.baselines.gcf_hiv_csv_model import torch_load  # noqa: E402


SORTING_KEY = "(marginal_coverage_gain, frequency, -min_distance_seen)"
SMILES_FIELDS = ("smiles", "candidate_smiles", "canonical_smiles")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _candidate_metadata_maps(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_id: dict[str, dict[str, str]] = {}
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        graph_hash = str(row.get("graph_hash") or "").strip()
        if candidate_id and candidate_id not in by_id:
            by_id[candidate_id] = row
        if graph_hash and graph_hash not in by_hash:
            by_hash[graph_hash] = row
    return by_id, by_hash


def load_raw_candidate_records(
    payload: dict[str, Any],
    candidate_metadata_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Join raw candidate records to graphs without changing payload order."""

    candidates = list(payload.get("counterfactual_candidates") or [])
    graph_map = payload.get("graph_map") if isinstance(payload.get("graph_map"), dict) else {}
    by_id, by_hash = _candidate_metadata_maps(candidate_metadata_rows or [])
    records: list[dict[str, Any]] = []
    for index, value in enumerate(candidates):
        row = dict(value) if isinstance(value, dict) else {}
        graph_hash = str(row.get("graph_hash") or "")
        fallback_id = f"gcf_hiv_csv_{index}"
        metadata = by_hash.get(graph_hash) or by_id.get(fallback_id) or {}
        candidate_id = str(metadata.get("candidate_id") or row.get("candidate_id") or fallback_id)
        records.append(
            {
                "candidate_id": candidate_id,
                "graph_hash": graph_hash,
                "frequency": int(row.get("frequency") or metadata.get("frequency") or 0),
                "covered_indices": _indices_from_sparse(row.get("input_graphs_covering_list")),
                "min_distance_seen": _finite_float_or_default(
                    row.get("min_distance_seen", metadata.get("min_distance_seen")),
                    999.0,
                ),
                "candidate_pred": row.get("candidate_pred", metadata.get("candidate_pred", "")),
                "candidate_score_label1": row.get(
                    "candidate_score_label1", metadata.get("candidate_score_label1", "")
                ),
                "graph": graph_map.get(graph_hash),
                "input_order": index,
            }
        )
    return records


def load_validity_map(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    return _candidate_metadata_maps(rows)


def validate_smiles(smiles: str) -> tuple[bool, str, str, int]:
    try:
        from rdkit import Chem
    except ImportError:
        return False, "rdkit_unavailable", "", 0
    if not smiles.strip():
        return False, "empty_smiles", "", 0
    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as exc:
        return False, f"rdkit_parse_failed:{exc}", "", 0
    if molecule is None:
        return False, "rdkit_parse_failed", "", 0
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        return False, f"rdkit_sanitize_failed:{exc}", "", 0
    atom_count = int(molecule.GetNumAtoms())
    if atom_count <= 0:
        return False, "empty_molecule", "", 0
    return True, "ok", Chem.MolToSmiles(molecule, canonical=True), atom_count


def filter_valid_candidates(
    records: list[dict[str, Any]],
    validity_rows: list[dict[str, str]],
    *,
    validate_smiles_fn: Callable[[str], tuple[bool, str, str, int]] = validate_smiles,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Filter before greedy so invalid coverage sets can never affect selection."""

    by_id, by_hash = load_validity_map(validity_rows)
    valid: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for record in records:
        mapping = by_hash.get(str(record.get("graph_hash") or "")) or by_id.get(
            str(record.get("candidate_id") or "")
        )
        if mapping is None:
            reasons["missing_validity_mapping"] += 1
            continue
        smiles = next(
            (str(mapping.get(field) or "").strip() for field in SMILES_FIELDS if mapping.get(field)),
            "",
        )
        if not smiles:
            reasons["empty_smiles"] += 1
            continue
        if "convert_ok" in mapping and not _as_bool(mapping.get("convert_ok")):
            reasons["graph_to_smiles_failed"] += 1
            continue
        if "sanitize_ok" in mapping and not _as_bool(mapping.get("sanitize_ok")):
            reasons["sanitize_mapping_failed"] += 1
            continue
        ok, reason, canonical_smiles, atom_count = validate_smiles_fn(smiles)
        if not ok:
            reasons[reason.split(":", 1)[0]] += 1
            continue
        if record.get("graph") is None:
            reasons["missing_graph"] += 1
            continue
        accepted = dict(record)
        accepted.update(
            {
                "smiles": canonical_smiles,
                "sanitize_ok": True,
                "atom_count": int(atom_count),
                "validity_reason": "ok",
            }
        )
        valid.append(accepted)
    return valid, reasons


def select_valid_greedy_topk(
    valid_records: list[dict[str, Any]],
    *,
    top_k: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run the existing deterministic greedy on the already-filtered pool."""

    del seed  # The existing greedy has no random branch; retained for run provenance.
    return _greedy(valid_records, int(top_k))


def build_aligned_outputs(
    selected: list[dict[str, Any]],
    *,
    num_parents: int,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    graphs: list[Any] = []
    metadata: list[dict[str, Any]] = []
    smiles_rows: list[dict[str, Any]] = []
    for row in selected:
        rank = int(row["selected_rank"])
        covered_count = int(row["covered_count_at_rank"])
        common = {
            "rank": rank,
            "candidate_id": row["candidate_id"],
            "graph_hash": row["graph_hash"],
            "smiles": row["smiles"],
            "frequency": int(row.get("frequency") or 0),
            "min_distance_seen": float(row["min_distance_seen"]),
            "marginal_coverage_gain": int(row.get("marginal_coverage_gain") or 0),
            "cumulative_covered_count": covered_count,
            "cumulative_native_coverage": covered_count / num_parents if num_parents else 0.0,
            "atom_count": int(row["atom_count"]),
        }
        graphs.append(row["graph"])
        metadata.append(common)
        smiles_rows.append(
            {
                "candidate_smiles": row["smiles"],
                "method": "GCFExplainer-HIVCSV",
                "fullgraph_method": "gcf_hiv_csv_valid_greedy_topk",
                **common,
            }
        )
    metadata_order = [(row["candidate_id"], row["graph_hash"]) for row in metadata]
    smiles_order = [(row["candidate_id"], row["graph_hash"]) for row in smiles_rows]
    selected_order = [(row["candidate_id"], row["graph_hash"]) for row in selected]
    if not (metadata_order == smiles_order == selected_order):
        raise AssertionError("Selected graph, metadata, and SMILES orders are inconsistent.")
    return graphs, metadata, smiles_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--counterfactuals", required=True)
    parser.add_argument("--candidate-records", default=None)
    parser.add_argument("--validity-csv", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--train-theta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if int(args.top_k) <= 0:
        raise SystemExit("[ERROR] --top-k must be positive.")
    import torch

    counterfactuals_path = Path(args.counterfactuals).expanduser().resolve()
    validity_path = Path(args.validity_csv).expanduser().resolve()
    candidate_records_path = (
        Path(args.candidate_records).expanduser().resolve() if args.candidate_records else None
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load(str(counterfactuals_path), map_location="cpu")
    raw_records = load_raw_candidate_records(payload, _read_csv(candidate_records_path))
    validity_rows = _read_csv(validity_path)
    valid_records, invalid_reasons = filter_valid_candidates(raw_records, validity_rows)
    selected = select_valid_greedy_topk(
        valid_records,
        top_k=int(args.top_k),
        seed=int(args.seed),
    )
    if len(selected) < int(args.top_k):
        raise SystemExit(
            f"[ERROR] Only {len(selected)} legal candidates are available; requested top_k={args.top_k}."
        )

    num_parents = len(list(payload.get("target_parent_indices") or []))
    if not num_parents:
        covered_values = [index for row in raw_records for index in row["covered_indices"]]
        num_parents = max(covered_values, default=-1) + 1
    graphs, metadata_rows, smiles_rows = build_aligned_outputs(
        selected,
        num_parents=num_parents,
    )
    prefix = f"valid_greedy_top{int(args.top_k)}"
    metadata_path = out_dir / f"{prefix}_metadata.csv"
    graphs_path = out_dir / f"{prefix}_graphs.pt"
    smiles_path = out_dir / f"{prefix}_smiles_for_fgw.csv"
    report_path = out_dir / "valid_greedy_selection_report.json"

    metadata_fields = [
        "rank",
        "candidate_id",
        "graph_hash",
        "smiles",
        "frequency",
        "min_distance_seen",
        "marginal_coverage_gain",
        "cumulative_covered_count",
        "cumulative_native_coverage",
        "atom_count",
    ]
    _write_csv(metadata_path, metadata_rows, metadata_fields)
    _write_csv(
        smiles_path,
        smiles_rows,
        ["candidate_smiles", "method", "fullgraph_method", *metadata_fields],
    )
    selected_records = []
    for row in selected:
        serialized = {key: value for key, value in row.items() if key != "graph"}
        serialized["covered_indices"] = sorted(row["covered_indices"])
        selected_records.append(serialized)
    torch.save(
        {
            "selected_graphs": graphs,
            "selected_records": selected_records,
            "GCF_MODE": "hiv_csv_adapted",
            "DATASET_SOURCE": "HIV_CSV",
            "CF_MODE": "strict_flip",
            "selection_mode": "valid_greedy_native_coverage",
            "sorting_key": SORTING_KEY,
            "train_theta": float(args.train_theta),
        },
        graphs_path,
    )

    sanitize_failed = sum(
        count
        for reason, count in invalid_reasons.items()
        if "sanitize" in reason or reason == "rdkit_parse_failed"
    )
    report = {
        "method": "GCFExplainer-HIVCSV",
        "GCF_MODE": "hiv_csv_adapted",
        "DATASET_SOURCE": "HIV_CSV",
        "CF_MODE": "strict_flip",
        "counterfactuals": str(counterfactuals_path),
        "candidate_records": str(candidate_records_path) if candidate_records_path else None,
        "validity_csv": str(validity_path),
        "original_candidate_count": len(raw_records),
        "valid_candidate_count": len(valid_records),
        "sanitize_failure_count": sanitize_failed,
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "top_k": int(args.top_k),
        "selected_count": len(selected),
        "sorting_key": SORTING_KEY,
        "train_theta": float(args.train_theta),
        "seed": int(args.seed),
        "num_target_parents": int(num_parents),
        "cumulative_native_coverage_by_rank": [
            {
                "rank": row["rank"],
                "marginal_coverage_gain": row["marginal_coverage_gain"],
                "covered_count": row["cumulative_covered_count"],
                "coverage": row["cumulative_native_coverage"],
            }
            for row in metadata_rows
        ],
        "selected_candidates": [
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "graph_hash": row["graph_hash"],
            }
            for row in metadata_rows
        ],
        "git_commit": _git_commit(),
        "outputs": {
            "metadata_csv": str(metadata_path),
            "graphs_pt": str(graphs_path),
            "smiles_for_fgw_csv": str(smiles_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[GCF_HIV_CSV_VALID_GREEDY_DONE]", flush=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
