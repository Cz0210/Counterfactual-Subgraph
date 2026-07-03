#!/usr/bin/env python3
"""Convert selected GCFExplainer-HIVCSV graphs back to SMILES when possible."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.gcf_hiv_csv_model import torch_load  # noqa: E402
from scripts.gcf_hiv_csv_run_vrrw import _unique_edge_data  # noqa: E402


BOND_TO_RDKIT = {
    "SINGLE": "SINGLE",
    "DOUBLE": "DOUBLE",
    "TRIPLE": "TRIPLE",
    "AROMATIC": "AROMATIC",
}


def _bond_type(Chem: Any, name: str) -> Any:
    if name == "DOUBLE":
        return Chem.BondType.DOUBLE
    if name == "TRIPLE":
        return Chem.BondType.TRIPLE
    if name == "AROMATIC":
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def _graph_to_smiles(graph: Any) -> tuple[str, bool, bool, str]:
    try:
        from rdkit import Chem
    except Exception:
        return "", False, False, "rdkit_unavailable"
    symbols = list(getattr(graph, "atom_symbols", []) or [])
    if len(symbols) != int(graph.num_nodes):
        return "", False, False, "missing_atom_symbols"
    try:
        mol = Chem.RWMol()
        for symbol in symbols:
            mol.AddAtom(Chem.Atom(str(symbol)))
        pairs, _attr_by_pair = _unique_edge_data(graph)
        bond_types = list(getattr(graph, "bond_types", []) or [])
        for idx, (a, b) in enumerate(pairs):
            name = str(bond_types[idx]) if idx < len(bond_types) else "SINGLE"
            mol.AddBond(int(a), int(b), _bond_type(Chem, name))
        built = mol.GetMol()
    except Exception as exc:
        return "", False, False, f"rdkit_build_failed:{exc}"
    try:
        Chem.SanitizeMol(built)
    except Exception as exc:
        return "", True, False, f"rdkit_sanitize_failed:{exc}"
    try:
        return Chem.MolToSmiles(built, canonical=True), True, True, "ok"
    except Exception as exc:
        return "", True, False, f"mol_to_smiles_failed:{exc}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--selected-graphs", required=True)
    parser.add_argument("--out-csv", default="outputs/hpc/gcfexplainer_hiv_csv/graph_to_smiles/gcf_hiv_csv_graph_smiles_candidates.csv")
    parser.add_argument("--out-report", default="outputs/hpc/gcfexplainer_hiv_csv/graph_to_smiles/gcf_hiv_csv_graph_to_smiles_report.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = torch_load(str(Path(args.selected_graphs).expanduser().resolve()), map_location="cpu")
    graphs = list(payload.get("selected_graphs") or [])
    records = [dict(row) for row in payload.get("selected_records") or [] if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for idx, graph in enumerate(graphs):
        record = records[idx] if idx < len(records) else {}
        smiles, convert_ok, sanitize_ok, reason = _graph_to_smiles(graph)
        reasons[reason.split(":", 1)[0]] += 1
        rows.append(
            {
                "candidate_id": record.get("candidate_id", f"gcf_hiv_csv_{idx}"),
                "smiles": smiles,
                "convert_ok": bool(convert_ok),
                "sanitize_ok": bool(sanitize_ok),
                "reason": reason,
                "GCF_MODE": "hiv_csv_adapted",
                "DATASET_SOURCE": "HIV_CSV",
                "CF_MODE": "strict_flip",
            }
        )
    out_csv = Path(args.out_csv).expanduser().resolve()
    _write_csv(out_csv, rows, ["candidate_id", "smiles", "convert_ok", "sanitize_ok", "reason", "GCF_MODE", "DATASET_SOURCE", "CF_MODE"])
    total = len(rows)
    report = {
        "method": "GCFExplainer-HIVCSV",
        "GCF_MODE": "hiv_csv_adapted",
        "DATASET_SOURCE": "HIV_CSV",
        "CF_MODE": "strict_flip",
        "selected_graphs": str(Path(args.selected_graphs).expanduser().resolve()),
        "out_csv": str(out_csv),
        "num_candidates": total,
        "convert_ok": sum(1 for row in rows if row["convert_ok"]),
        "sanitize_ok": sum(1 for row in rows if row["sanitize_ok"]),
        "convert_ok_rate": sum(1 for row in rows if row["convert_ok"]) / total if total else 0.0,
        "sanitize_ok_rate": sum(1 for row in rows if row["sanitize_ok"]) / total if total else 0.0,
        "reason_counts": dict(reasons),
    }
    out_report = Path(args.out_report).expanduser().resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[GCF_HIV_CSV_GRAPH_TO_SMILES_DONE]", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

