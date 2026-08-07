"""Attach stable BACE parent lineage to an existing Ours candidate pool."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_parents(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"BACE generation parent CSV is empty: {path}")
    required = {"molecule_id", "smiles", "label", "source_graph_hash"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"BACE generation parent CSV is missing {missing}: {path}")
    parent_ids = [str(row["molecule_id"]).strip() for row in rows]
    if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("BACE generation parent IDs must be non-empty and unique.")
    return rows


def attach_bace_candidate_lineage(
    *,
    raw_pool_jsonl: str | Path,
    parent_csv: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path,
    expected_candidates_per_parent: int = 4,
) -> dict[str, Any]:
    """Add IDs only; candidate payload values and row order remain untouched."""

    raw_path = Path(raw_pool_jsonl).expanduser().resolve()
    parents_path = Path(parent_csv).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not raw_path.is_file() or not parents_path.is_file():
        raise FileNotFoundError("BACE raw candidate pool and parent CSV must exist.")
    if output_path.exists() or manifest.exists():
        raise FileExistsError("BACE lineage output already exists; refusing to overwrite.")
    expected_per_parent = int(expected_candidates_per_parent)
    if expected_per_parent <= 0:
        raise ValueError("expected_candidates_per_parent must be positive.")
    parents = _load_parents(parents_path)
    enriched: list[dict[str, Any]] = []
    counts = {index: 0 for index in range(len(parents))}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank BACE candidate row at line {line_number}.")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"BACE candidate row {line_number} is not an object.")
            try:
                parent_index = int(row["parent_index"])
                candidate_index = int(row["candidate_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"BACE candidate row {line_number} lacks integer lineage indices."
                ) from exc
            if not 0 <= parent_index < len(parents):
                raise ValueError(
                    f"BACE parent_index out of range at line {line_number}: {parent_index}"
                )
            if not 0 <= candidate_index < expected_per_parent:
                raise ValueError(
                    f"BACE candidate_index out of range at line {line_number}: "
                    f"{candidate_index}"
                )
            parent = parents[parent_index]
            parent_smiles = str(parent["smiles"]).strip()
            if str(row.get("parent_smiles") or "").strip() != parent_smiles:
                raise ValueError(
                    f"BACE parent SMILES lineage mismatch at line {line_number}."
                )
            if int(row.get("label", -1)) != int(parent["label"]):
                raise ValueError(f"BACE parent label mismatch at line {line_number}.")
            parent_id = str(parent["molecule_id"]).strip()
            additions = {
                "parent_id": parent_id,
                "molecule_id": parent_id,
                "source_graph_hash": str(parent["source_graph_hash"]).strip(),
                "candidate_id": f"BACE_OURS_{parent_id}_{candidate_index:02d}",
            }
            for key, value in additions.items():
                existing = row.get(key)
                if existing is not None and str(existing).strip() != str(value):
                    raise ValueError(
                        f"BACE candidate lineage would overwrite {key} at line {line_number}."
                    )
            enriched.append({**row, **additions})
            counts[parent_index] += 1
    expected_rows = len(parents) * expected_per_parent
    if len(enriched) != expected_rows:
        raise ValueError(
            f"BACE candidate count mismatch: actual={len(enriched)} expected={expected_rows}"
        )
    wrong_counts = {
        index: count for index, count in counts.items() if count != expected_per_parent
    }
    if wrong_counts:
        raise ValueError(f"BACE per-parent candidate counts differ: {wrong_counts}")
    _atomic_text(
        output_path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in enriched
        ),
    )
    payload = {
        "schema_version": "bace_ours_candidate_lineage_v1",
        "dataset": "BACE",
        "raw_pool_jsonl": str(raw_path),
        "raw_pool_sha256": sha256_file(raw_path),
        "parent_csv": str(parents_path),
        "parent_csv_sha256": sha256_file(parents_path),
        "output_jsonl": str(output_path),
        "output_sha256": sha256_file(output_path),
        "num_parents": len(parents),
        "num_candidates": len(enriched),
        "candidates_per_parent": expected_per_parent,
        "candidate_order_unchanged": True,
        "scientific_fields_changed": False,
        "added_fields": [
            "parent_id",
            "molecule_id",
            "source_graph_hash",
            "candidate_id",
        ],
    }
    _atomic_text(
        manifest,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return payload


__all__ = ["attach_bace_candidate_lineage", "sha256_file"]
