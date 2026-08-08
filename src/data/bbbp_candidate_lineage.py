"""Attach stable BBBP parent lineage to an existing Ours candidate pool."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
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
        raise ValueError(f"BBBP generation parent CSV is empty: {path}")
    required = {
        "molecule_id",
        "smiles",
        "label",
        "split",
        "source_graph_hash",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"BBBP generation parent CSV is missing {missing}: {path}")
    parent_ids = [str(row["molecule_id"]).strip() for row in rows]
    if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("BBBP generation parent IDs must be non-empty and unique.")
    return rows


def attach_bbbp_candidate_lineage(
    *,
    raw_pool_jsonl: str | Path,
    parent_csv: str | Path,
    output_jsonl: str | Path,
    manifest_path: str | Path,
    expected_candidates_per_parent: int = 4,
    candidate_source: str = "chemllm_ppo",
    candidate_source_variant: str = "stable300",
    generation_seed: int = 13,
    checkpoint_path: str | Path | None = None,
    checkpoint_kind: str = "ppo",
) -> dict[str, Any]:
    """Add IDs only; candidate payload values and row order remain untouched."""

    raw_path = Path(raw_pool_jsonl).expanduser().resolve()
    parents_path = Path(parent_csv).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not raw_path.is_file() or not parents_path.is_file():
        raise FileNotFoundError("BBBP raw candidate pool and parent CSV must exist.")
    if output_path.exists() or manifest.exists():
        raise FileExistsError("BBBP lineage output already exists; refusing to overwrite.")
    expected_per_parent = int(expected_candidates_per_parent)
    if expected_per_parent <= 0:
        raise ValueError("expected_candidates_per_parent must be positive.")
    parents = _load_parents(parents_path)
    enriched: list[dict[str, Any]] = []
    counts = {index: 0 for index in range(len(parents))}
    preserved_generator_parent_ids = 0
    added_project_parent_ids = 0
    source_git_commit = _git_commit()
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank BBBP candidate row at line {line_number}.")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"BBBP candidate row {line_number} is not an object.")
            try:
                parent_index = int(row["parent_index"])
                candidate_index = int(row["candidate_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"BBBP candidate row {line_number} lacks integer lineage indices."
                ) from exc
            if not 0 <= parent_index < len(parents):
                raise ValueError(
                    f"BBBP parent_index out of range at line {line_number}: {parent_index}"
                )
            if not 0 <= candidate_index < expected_per_parent:
                raise ValueError(
                    f"BBBP candidate_index out of range at line {line_number}: "
                    f"{candidate_index}"
                )
            parent = parents[parent_index]
            parent_smiles = str(parent["smiles"]).strip()
            if str(row.get("parent_smiles") or "").strip() != parent_smiles:
                raise ValueError(
                    f"BBBP parent SMILES lineage mismatch at line {line_number}."
                )
            if int(row.get("label", -1)) != int(parent["label"]):
                raise ValueError(f"BBBP parent label mismatch at line {line_number}.")
            parent_id = str(parent["molecule_id"]).strip()
            additions = {
                "molecule_id": parent_id,
                "parent_split": str(parent["split"]),
                "source_graph_hash": str(parent["source_graph_hash"]).strip(),
                "candidate_id": f"BBBP_OURS_{parent_id}_{candidate_index:02d}",
                "candidate_source": str(candidate_source),
                "candidate_source_variant": str(candidate_source_variant),
                "generation_seed": int(generation_seed),
                "generation_rank": int(candidate_index + 1),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_kind": str(checkpoint_kind),
                "source_git_commit": source_git_commit,
            }
            existing_parent_id = row.get("parent_id")
            if existing_parent_id is None:
                additions["parent_id"] = parent_id
                added_project_parent_ids += 1
            else:
                rendered_parent_id = str(existing_parent_id).strip()
                try:
                    generator_parent_id_matches = int(rendered_parent_id) == parent_index
                except ValueError:
                    generator_parent_id_matches = False
                if rendered_parent_id != parent_id and not generator_parent_id_matches:
                    raise ValueError(
                        "BBBP generator parent_id does not match parent_index or "
                        f"project molecule_id at line {line_number}."
                    )
                preserved_generator_parent_ids += 1
            for key, value in additions.items():
                existing = row.get(key)
                if existing is not None and str(existing).strip() != str(value):
                    raise ValueError(
                        f"BBBP candidate lineage would overwrite {key} at line {line_number}."
                    )
            enriched.append({**row, **additions})
            counts[parent_index] += 1
    expected_rows = len(parents) * expected_per_parent
    if len(enriched) != expected_rows:
        raise ValueError(
            f"BBBP candidate count mismatch: actual={len(enriched)} expected={expected_rows}"
        )
    wrong_counts = {
        index: count for index, count in counts.items() if count != expected_per_parent
    }
    if wrong_counts:
        raise ValueError(f"BBBP per-parent candidate counts differ: {wrong_counts}")
    _atomic_text(
        output_path,
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
            for row in enriched
        ),
    )
    payload = {
        "schema_version": "bbbp_ours_candidate_lineage_v1",
        "dataset": "BBBP",
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
        "parent_id_policy": "preserve_generator_parent_id_or_add_project_molecule_id",
        "preserved_generator_parent_id_count": preserved_generator_parent_ids,
        "added_project_parent_id_count": added_project_parent_ids,
        "added_fields": [
            *(["parent_id"] if added_project_parent_ids else []),
            "molecule_id",
            "source_graph_hash",
            "candidate_id",
            "parent_split",
            "candidate_source",
            "candidate_source_variant",
            "generation_seed",
            "generation_rank",
            "checkpoint_path",
            "checkpoint_kind",
            "source_git_commit",
        ],
        "candidate_source": str(candidate_source),
        "candidate_source_variant": str(candidate_source_variant),
        "generation_seed": int(generation_seed),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_kind": str(checkpoint_kind),
        "candidate_source_splits": sorted({str(row["split"]) for row in parents}),
        "test_used_for_generation": any(str(row["split"]) == "test" for row in parents),
    }
    _atomic_text(
        manifest,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return payload


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


__all__ = ["attach_bbbp_candidate_lineage", "sha256_file"]
