#!/usr/bin/env python3
"""Build a calibration-only BACE Round-2 source cohort from hard parent groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _scaffold(smiles: str, declared: str = "") -> str:
    if declared.strip():
        return declared.strip()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def _fingerprint(smiles: str) -> Any:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid source SMILES: {smiles!r}")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)


def build_round2_cohort(
    *,
    pair_matrix: Path,
    thresholds_json: Path,
    calibration_csv: Path,
    train_csv: Path,
    output_csv: Path,
    manifest_path: Path,
    nearest_per_hard_parent: int = 2,
) -> dict[str, Any]:
    pairs = _read_jsonl(pair_matrix)
    thresholds = json.loads(thresholds_json.read_text(encoding="utf-8"))
    theta = float(thresholds["theta_star"])
    calibration = {str(row["molecule_id"]): row for row in _read_csv(calibration_csv)}
    train = _read_csv(train_csv)
    if any(str(row.get("split") or "").lower() == "test" for row in train):
        raise ValueError("Round-2 source cohort cannot contain test rows.")
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_parent[str(row["parent_id"])].append(row)
    hard_groups: dict[str, str] = {}
    for parent_id, rows in sorted(by_parent.items()):
        connected = any(int(row.get("num_connected_valid_matches") or 0) > 0 for row in rows)
        strict_rows = [row for row in rows if bool(row.get("pair_strict_flip"))]
        close = any(
            row.get("wnode_distance") is not None
            and math.isfinite(float(row["wnode_distance"]))
            and float(row["wnode_distance"]) <= theta
            for row in strict_rows
        )
        if close:
            continue
        if strict_rows:
            hard_groups[parent_id] = "B_only_high_threshold"
        elif connected:
            hard_groups[parent_id] = "C_applicable_not_flip"
        else:
            hard_groups[parent_id] = "D_no_connected_valid_candidate"
    train_fingerprints = {
        str(row["molecule_id"]): _fingerprint(str(row["smiles"])) for row in train
    }
    selected: dict[str, dict[str, Any]] = {}
    lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for calibration_id, group in sorted(hard_groups.items()):
        source = calibration.get(calibration_id)
        if source is None:
            raise ValueError(f"Matrix parent missing from calibration CSV: {calibration_id}")
        query_scaffold = _scaffold(str(source["smiles"]), str(source.get("scaffold") or ""))
        query_fp = _fingerprint(query_scaffold or str(source["smiles"]))
        ranked: list[tuple[int, float, str, dict[str, Any]]] = []
        for row in train:
            train_id = str(row["molecule_id"])
            train_scaffold = _scaffold(str(row["smiles"]), str(row.get("scaffold") or ""))
            exact = int(bool(query_scaffold) and train_scaffold == query_scaffold)
            similarity = float(DataStructs.TanimotoSimilarity(query_fp, train_fingerprints[train_id]))
            ranked.append((exact, similarity, train_id, row))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for exact, similarity, train_id, row in ranked[: int(nearest_per_hard_parent)]:
            selected[train_id] = dict(row)
            lineage[train_id].append(
                {
                    "calibration_parent_id": calibration_id,
                    "hard_group": group,
                    "exact_scaffold": bool(exact),
                    "morgan_tanimoto": similarity,
                }
            )
    rows = [selected[key] for key in sorted(selected)]
    if not rows:
        raise ValueError("Calibration hard groups produced an empty Round-2 source cohort.")
    fieldnames = list(rows[0])
    for field in ("source_cluster", "round2_hard_group_lineage"):
        if field not in fieldnames:
            fieldnames.append(field)
    temporary = output_csv.with_name(f".{output_csv.name}.tmp")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            value = dict(row)
            value["source_cluster"] = "calibration_hard_group_nearest_v1"
            value["round2_hard_group_lineage"] = json.dumps(
                lineage[str(row["molecule_id"])], sort_keys=True
            )
            writer.writerow(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_csv)
    payload = {
        "schema_version": "bace_round2_source_cohort_v4",
        "selection_split": "calibration",
        "test_loaded": False,
        "test_source_parent_count": 0,
        "theta": theta,
        "theta_source": str(thresholds_json),
        "hard_parent_count": len(hard_groups),
        "hard_group_counts": dict(
            sorted((group, list(hard_groups.values()).count(group)) for group in set(hard_groups.values()))
        ),
        "selected_train_parent_count": len(rows),
        "nearest_per_hard_parent": int(nearest_per_hard_parent),
        "source_selection": "exact_scaffold_then_morgan_nearest_v1",
        "molclr_cluster_artifact_available": False,
        "pair_matrix_sha256": _sha(pair_matrix),
        "calibration_csv_sha256": _sha(calibration_csv),
        "train_csv_sha256": _sha(train_csv),
        "output_csv": str(output_csv),
        "output_csv_sha256": _sha(output_csv),
        "run_complete": True,
    }
    _atomic(manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--pair-matrix", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--nearest-per-hard-parent", type=int, default=2)
    args = parser.parse_args()
    payload = build_round2_cohort(
        pair_matrix=Path(args.pair_matrix).expanduser().resolve(),
        thresholds_json=Path(args.thresholds_json).expanduser().resolve(),
        calibration_csv=Path(args.calibration_csv).expanduser().resolve(),
        train_csv=Path(args.train_csv).expanduser().resolve(),
        output_csv=Path(args.output_csv).expanduser().resolve(),
        manifest_path=Path(args.manifest_path).expanduser().resolve(),
        nearest_per_hard_parent=int(args.nearest_per_hard_parent),
    )
    print(json.dumps(payload, sort_keys=True))
    print("[BACE_ROUND2_SOURCE_COHORT_PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
