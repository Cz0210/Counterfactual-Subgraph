"""Normalized GED labeling helpers for GREED-style pair training."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from src.eval.close_counterfactual_coverage import mol_from_smiles, normalized_networkx_ged_distance
from src.utils.io import ensure_directory


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _denominator(row: dict[str, Any]) -> int:
    atoms_a = _as_int(row.get("num_atoms_a")) or 0
    atoms_b = _as_int(row.get("num_atoms_b")) or 0
    bonds_a = _as_int(row.get("num_bonds_a")) or 0
    bonds_b = _as_int(row.get("num_bonds_b")) or 0
    return max(1, atoms_a + atoms_b + bonds_a + bonds_b)


def _histogram_count_approx(smiles_a: str, smiles_b: str) -> tuple[float | None, str | None]:
    mol_a = mol_from_smiles(smiles_a)
    mol_b = mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None, "rdkit_parse_failed"
    atom_counts: dict[int, int] = {}
    for atom in mol_a.GetAtoms():
        atom_counts[int(atom.GetAtomicNum())] = atom_counts.get(int(atom.GetAtomicNum()), 0) + 1
    for atom in mol_b.GetAtoms():
        atom_counts[int(atom.GetAtomicNum())] = atom_counts.get(int(atom.GetAtomicNum()), 0) - 1
    atom_label_cost = sum(abs(value) for value in atom_counts.values()) / 2.0
    bond_count_cost = abs(int(mol_a.GetNumBonds()) - int(mol_b.GetNumBonds()))
    atom_count_cost = abs(int(mol_a.GetNumAtoms()) - int(mol_b.GetNumAtoms()))
    return float(atom_count_cost + bond_count_cost + atom_label_cost), None


def label_pair_row(
    row: dict[str, Any],
    *,
    allow_networkx_debug: bool = False,
    networkx_timeout: float = 1.0,
    fullgraph_label_mode: str = "bounded_approx",
) -> dict[str, Any]:
    """Add normalized GED labels to one pair row.

    Full-graph/random pairs deliberately avoid default NetworkX exact GED. The
    default fallback is a bounded atom/bond-count approximation suitable for
    smoke/full GREED-style training when GEDLIB is unavailable.
    """

    out = dict(row)
    out.update(
        {
            "ged_raw": None,
            "ged_norm": None,
            "ged_label_ok": False,
            "ged_label_source": "failed",
            "ged_label_error": None,
        }
    )
    denom = _denominator(row)
    if str(row.get("pair_type")) == "ours_deletion":
        removed_atoms = _as_int(row.get("num_removed_atoms")) or 0
        removed_bonds = _as_int(row.get("num_removed_bonds")) or 0
        raw = max(0, removed_atoms) + max(0, removed_bonds)
        out.update(
            {
                "ged_raw": float(raw),
                "ged_norm": max(0.0, min(1.0, float(raw) / float(denom))),
                "ged_label_ok": True,
                "ged_label_source": "deletion_exact",
                "ged_label_error": None,
            }
        )
        return out

    smiles_a = str(row.get("smiles_a") or "")
    smiles_b = str(row.get("smiles_b") or "")
    if fullgraph_label_mode == "networkx_debug" or allow_networkx_debug:
        distance = normalized_networkx_ged_distance(smiles_a, smiles_b, timeout=float(networkx_timeout))
        out.update(
            {
                "ged_raw": None,
                "ged_norm": distance,
                "ged_label_ok": distance is not None and math.isfinite(float(distance)),
                "ged_label_source": "networkx_timeout_debug" if distance is not None else "failed",
                "ged_label_error": None if distance is not None else "networkx_timeout_or_unavailable",
            }
        )
        return out

    if fullgraph_label_mode == "fail":
        out["ged_label_error"] = "fullgraph_label_mode_fail"
        return out

    raw, error = _histogram_count_approx(smiles_a, smiles_b)
    if raw is None:
        out["ged_label_error"] = error or "bounded_approx_failed"
        return out
    out.update(
        {
            "ged_raw": float(raw),
            "ged_norm": max(0.0, min(1.0, float(raw) / float(denom))),
            "ged_label_ok": True,
            "ged_label_source": "bounded_count_approx",
            "ged_label_error": None,
        }
    )
    return out


def label_pairs_csv(
    *,
    input_csv: str | Path,
    output_csv: str | Path,
    allow_networkx_debug: bool = False,
    networkx_timeout: float = 1.0,
    fullgraph_label_mode: str = "bounded_approx",
) -> dict[str, Any]:
    source = Path(input_csv).expanduser().resolve()
    destination = Path(output_csv).expanduser().resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    labeled = [
        label_pair_row(
            row,
            allow_networkx_debug=allow_networkx_debug,
            networkx_timeout=networkx_timeout,
            fullgraph_label_mode=fullgraph_label_mode,
        )
        for row in rows
    ]
    ensure_directory(destination.parent)
    fieldnames: list[str] = []
    for row in labeled:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(labeled)
    ok_count = sum(1 for row in labeled if str(row.get("ged_label_ok")).lower() == "true")
    return {
        "input_csv": str(source),
        "output_csv": str(destination),
        "num_pairs": len(labeled),
        "num_label_ok": ok_count,
        "label_ok_rate": (ok_count / len(labeled)) if labeled else 0.0,
        "fullgraph_label_mode": fullgraph_label_mode,
        "networkx_debug_enabled": bool(allow_networkx_debug),
    }
