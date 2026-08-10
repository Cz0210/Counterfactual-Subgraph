#!/usr/bin/env python3
"""Freeze BACE GlobalGCE Frequency-Top20 from its train-only native pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chem.hard_deletion import (  # noqa: E402
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(payload: Any) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(dict(payload))
    return rows


def _atomic(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def _csv_text(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def freeze_frequency_top20(
    *,
    run_dir: str | Path,
    teacher_path: str | Path,
    molclr_checkpoint: str | Path,
    thresholds_json: str | Path,
    output_dir: str | Path,
    target_k: int = 20,
) -> dict[str, Any]:
    source = Path(run_dir).expanduser().resolve()
    teacher = Path(teacher_path).expanduser().resolve()
    molclr = Path(molclr_checkpoint).expanduser().resolve()
    thresholds = Path(thresholds_json).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Frozen BACE GlobalGCE selector already exists: {output}")
    manifest_path = source / "run_manifest.json"
    summary_path = source / "summary.json"
    universe_path = source / "candidate_universe.jsonl"
    for path in (manifest_path, summary_path, universe_path, teacher, molclr, thresholds):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    manifest = _json(manifest_path)
    summary = _json(summary_path)
    if manifest.get("dataset") != "BACE" or manifest.get("run_complete") is not True:
        raise ValueError("BACE GlobalGCE train-pool manifest is incomplete or misidentified.")
    if manifest.get("calibration_used") is not False or manifest.get("test_used") is not False:
        raise ValueError("BACE GlobalGCE pool does not prove calibration/test exclusion.")
    expected_teacher = ((manifest.get("inputs") or {}).get("teacher_path") or {}).get("sha256")
    if expected_teacher != _sha(teacher):
        raise ValueError("BACE GlobalGCE pool teacher differs from the frozen BACE teacher.")
    ranked: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    for row in _jsonl(universe_path):
        smiles = str(row.get("canonical_smiles") or "").strip()
        reason = ""
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() <= 0:
            reason = "parse_or_empty"
        else:
            try:
                Chem.SanitizeMol(molecule)
            except Exception:
                reason = "sanitize_failed"
            if not reason and ("." in smiles or len(Chem.GetMolFrags(molecule)) != 1):
                reason = "disconnected"
        if not bool(row.get("teacher_target_ok")) or int(row.get("teacher_pred", -1)) != 0:
            reason = reason or "not_teacher_counterfactual"
        if reason:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        ranked.append(dict(row))
    ranked.sort(
        key=lambda row: (
            -int(row.get("source_parent_count") or 0),
            -int(row.get("source_occurrence_count") or 0),
            str(row.get("canonical_smiles") or ""),
            str(row.get("candidate_id") or ""),
        )
    )
    ranked_ids = [str(row.get("candidate_id") or "") for row in ranked]
    ranked_smiles = [str(row.get("canonical_smiles") or "") for row in ranked]
    if not all(ranked_ids) or len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("BACE GlobalGCE candidate universe IDs are missing or duplicated.")
    if not all(ranked_smiles) or len(ranked_smiles) != len(set(ranked_smiles)):
        raise ValueError("BACE GlobalGCE candidate universe SMILES are missing or duplicated.")
    if len(ranked) < int(target_k):
        raise RuntimeError(
            "INSUFFICIENT_VALID_CONNECTED_GLOBALGCE_CANDIDATES: "
            f"available={len(ranked)}, required={target_k}, rejections={rejection_counts}"
        )
    selected: list[dict[str, Any]] = []
    for rank, source_row in enumerate(ranked[: int(target_k)], start=1):
        selected.append(
            {
                "rank": rank,
                "candidate_id": str(source_row["candidate_id"]),
                "canonical_smiles": str(source_row["canonical_smiles"]),
                "smiles": str(source_row["canonical_smiles"]),
                "frequency": int(source_row.get("source_occurrence_count") or 0),
                "source_parent_count": int(source_row.get("source_parent_count") or 0),
                "source_occurrence_count": int(source_row.get("source_occurrence_count") or 0),
                "rf_strict_flip": True,
                "candidate_set_preselected": True,
                "selection_performed_in_eval": False,
                "selection_mode": "globalgce_frequency_top20_train_support_v1",
                "source_split": "train",
                "connected": True,
            }
        )
    candidate_ids = [row["candidate_id"] for row in selected]
    output.mkdir(parents=True, exist_ok=False)
    fields = tuple(selected[0])
    ranked_export = [
        {
            "rank": index,
            "candidate_id": str(row["candidate_id"]),
            "canonical_smiles": str(row["canonical_smiles"]),
            "smiles": str(row["canonical_smiles"]),
            "frequency": int(row.get("source_occurrence_count") or 0),
            "source_parent_count": int(row.get("source_parent_count") or 0),
            "source_occurrence_count": int(row.get("source_occurrence_count") or 0),
            "rf_strict_flip": True,
            "candidate_set_preselected": False,
            "selection_performed_in_eval": False,
            "selection_mode": "globalgce_frequency_train_support_v1",
            "source_split": "train",
            "connected": True,
        }
        for index, row in enumerate(ranked, start=1)
    ]
    _atomic(
        output / "frequency_ranked_candidates.csv",
        _csv_text(ranked_export, fields),
    )
    selected_text = _csv_text(selected, fields)
    _atomic(output / "selected_top20.csv", selected_text)
    _atomic(output / "selected_top20_for_eval.csv", selected_text)
    frozen = {
        "schema_version": "bace_globalgce_frequency_top20_v1",
        "dataset": "BACE",
        "method": "GlobalGCE",
        "selection_frozen": True,
        "selection_method": "globalgce_frequency_top20_train_support_v1",
        "selection_split": "train",
        "selected_candidate_ids": candidate_ids,
        "selected_sequence_sha256": _stable(candidate_ids),
        "target_k": int(target_k),
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "action_semantics_version": CONNECTED_ACTION_SEMANTICS,
        "match_selection_policy": CONNECTED_MATCH_SELECTION_POLICY,
        "calibration_used": False,
        "test_used": False,
        "gcf_result_used": False,
        "source_run_dir": str(source),
        "source_manifest_sha256": _sha(manifest_path),
        "source_universe_sha256": _sha(universe_path),
        "teacher_path": str(teacher),
        "teacher_sha256": _sha(teacher),
        "teacher_identity": {"path": str(teacher), "sha256": _sha(teacher)},
        "molclr_identity": {"path": str(molclr), "sha256": _sha(molclr)},
        "threshold_manifest_sha256": _sha(thresholds),
        "raw_unique_candidate_count": int(summary.get("canonical_unique_candidates") or 0),
        "valid_connected_candidate_count": len(ranked),
        "rejection_reason_counts": rejection_counts,
        "all_selected_candidates_connected": True,
        "native_order_modified": False,
        "native_generation_outputs_modified": False,
        "frequency_ranking_applied_once": True,
        "frequency_order_preserved": True,
    }
    _atomic(output / "frozen_selection.json", json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    audit = {
        **frozen,
        "passed": True,
        "selected_count": len(selected),
        "rank_contiguous": [row["rank"] for row in selected] == list(range(1, int(target_k) + 1)),
        "candidate_ids_unique": len(candidate_ids) == len(set(candidate_ids)),
        "smiles_unique": len({row["canonical_smiles"] for row in selected}) == len(selected),
    }
    _atomic(output / "selection_audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--teacher-path", required=True)
    parser.add_argument("--molclr-checkpoint", required=True)
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-k", type=int, default=20)
    args = parser.parse_args(argv)
    result = freeze_frequency_top20(
        run_dir=args.run_dir,
        teacher_path=args.teacher_path,
        molclr_checkpoint=args.molclr_checkpoint,
        thresholds_json=args.thresholds_json,
        output_dir=args.output_dir,
        target_k=int(args.target_k),
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_GLOBALGCE_FREQUENCY_TOP20_FROZEN]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
