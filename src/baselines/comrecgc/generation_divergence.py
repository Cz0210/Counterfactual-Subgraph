"""Safe JSON-only first-divergence diagnostics for COMRECGC generation.

The formal equivalence gate must load the complete generation payload, but a
failed payload should not hide an earlier actionable difference.  This module
therefore consumes only the hash-bound JSON trace artifacts written by the
generation runtime.  It never unpickles ``counterfactuals.pt``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import sha256_file, stable_json_sha256, write_json


DIVERGENCE_SCHEMA = "bace_comrecgc_generation_first_divergence_v1"
UNSTABLE_OFFICIAL_HASH_FIELDS = frozenset(
    {"source_official_hash", "target_official_hash", "official_graph_hash"}
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _trace_root(run_root: Path) -> Path:
    for candidate in (
        run_root / "_native_aux/trace",
        run_root / "trace",
    ):
        if (candidate / "selected_action_trace_manifest.json").is_file():
            return candidate
    raise FileNotFoundError(f"Selected-action trace is missing below {run_root}.")


def _selected_rows(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_root = _trace_root(run_root)
    manifest_path = trace_root / "selected_action_trace_manifest.json"
    manifest = _json(manifest_path)
    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for chunk in manifest.get("chunks") or ():
        relative = str(chunk["path"])
        path = trace_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        chunk_rows = _jsonl(path)
        expected_count = int(chunk["row_count"])
        if len(chunk_rows) != expected_count:
            raise ValueError(
                f"Selected trace row count mismatch for {path}: "
                f"actual={len(chunk_rows)}, expected={expected_count}."
            )
        expected_sha = str(chunk.get("sha256") or "")
        observed_sha = sha256_file(path)
        if expected_sha and expected_sha != observed_sha:
            raise ValueError(f"Selected trace SHA256 mismatch: {path}")
        rows.extend(chunk_rows)
        chunks.append(
            {
                "path": relative,
                "row_count": len(chunk_rows),
                "sha256": observed_sha,
            }
        )
    expected_total = int(manifest.get("row_count", -1))
    if len(rows) != expected_total:
        raise ValueError(
            "Selected trace manifest total mismatch: "
            f"actual={len(rows)}, expected={expected_total}."
        )
    return rows, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(rows),
        "chunks": chunks,
    }


def _lineage_rows(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_root = _trace_root(run_root)
    path = trace_root / "candidate_action_lineage_index.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = _jsonl(path)
    return rows, {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
    }


def _field_differences(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    ignored: Iterable[str] = (),
) -> dict[str, dict[str, Any]]:
    excluded = set(ignored)
    return {
        key: {"legacy": left.get(key), "optimized": right.get(key)}
        for key in sorted((set(left) | set(right)) - excluded)
        if left.get(key) != right.get(key)
    }


def _first_row_difference(
    legacy: list[dict[str, Any]],
    optimized: list[dict[str, Any]],
    *,
    ignored: Iterable[str] = (),
) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(legacy, optimized, strict=False)):
        differences = _field_differences(left, right, ignored=ignored)
        if differences:
            return {
                "row_index": index,
                "head_index": left.get("head_index", right.get("head_index")),
                "move_index": left.get("move_index", right.get("move_index")),
                "parent_id": left.get("parent_id", right.get("parent_id")),
                "differences": differences,
                "legacy_row": left,
                "optimized_row": right,
            }
    if len(legacy) != len(optimized):
        index = min(len(legacy), len(optimized))
        return {
            "row_index": index,
            "differences": {
                "row_count": {
                    "legacy": len(legacy),
                    "optimized": len(optimized),
                }
            },
            "legacy_row": legacy[index] if index < len(legacy) else None,
            "optimized_row": optimized[index] if index < len(optimized) else None,
        }
    return None


def _lineage_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("parent_id"),
        row.get("stable_graph_sha256"),
        row.get("action_count"),
        row.get("lineage_root_status"),
        row.get("lineage_storage"),
    )


def diagnose_generation_divergence(
    *,
    legacy_root: str | Path,
    optimized_root: str | Path,
) -> dict[str, Any]:
    """Return the earliest safe trace divergence between two completed runs."""

    legacy_dir = Path(legacy_root).expanduser().resolve()
    optimized_dir = Path(optimized_root).expanduser().resolve()
    legacy_selected, legacy_selected_manifest = _selected_rows(legacy_dir)
    optimized_selected, optimized_selected_manifest = _selected_rows(optimized_dir)
    legacy_lineage, legacy_lineage_manifest = _lineage_rows(legacy_dir)
    optimized_lineage, optimized_lineage_manifest = _lineage_rows(optimized_dir)

    first_any_selected = _first_row_difference(
        legacy_selected,
        optimized_selected,
    )
    first_stable_selected = _first_row_difference(
        legacy_selected,
        optimized_selected,
        ignored=UNSTABLE_OFFICIAL_HASH_FIELDS,
    )
    first_lineage: dict[str, Any] | None = None
    for index, (left, right) in enumerate(
        zip(legacy_lineage, optimized_lineage, strict=False)
    ):
        if _lineage_identity(left) != _lineage_identity(right):
            first_lineage = {
                "candidate_index": index,
                "legacy_identity": list(_lineage_identity(left)),
                "optimized_identity": list(_lineage_identity(right)),
                "legacy_row": left,
                "optimized_row": right,
            }
            break
    if first_lineage is None and len(legacy_lineage) != len(optimized_lineage):
        first_lineage = {
            "candidate_index": min(len(legacy_lineage), len(optimized_lineage)),
            "legacy_identity": None,
            "optimized_identity": None,
            "row_count_only": True,
        }

    legacy_set = {_lineage_identity(row) for row in legacy_lineage}
    optimized_set = {_lineage_identity(row) for row in optimized_lineage}
    identified = any(
        value is not None
        for value in (first_any_selected, first_stable_selected, first_lineage)
    )
    report: dict[str, Any] = {
        "schema_version": DIVERGENCE_SCHEMA,
        "status": "DIVERGENCE_IDENTIFIED" if identified else "NO_DIVERGENCE",
        "legacy_root": str(legacy_dir),
        "optimized_root": str(optimized_dir),
        "legacy_selected_trace": legacy_selected_manifest,
        "optimized_selected_trace": optimized_selected_manifest,
        "legacy_candidate_lineage": legacy_lineage_manifest,
        "optimized_candidate_lineage": optimized_lineage_manifest,
        "first_any_selected_trace_difference": first_any_selected,
        "first_stable_selected_transition_difference": first_stable_selected,
        "first_candidate_sequence_difference": first_lineage,
        "candidate_counts": {
            "legacy": len(legacy_lineage),
            "optimized": len(optimized_lineage),
            "delta_optimized_minus_legacy": len(optimized_lineage)
            - len(legacy_lineage),
        },
        "candidate_identity_set_difference": {
            "legacy_only_count": len(legacy_set - optimized_set),
            "optimized_only_count": len(optimized_set - legacy_set),
        },
        "official_hash_caveat": (
            "Official COMRECGC hashes Python hash(graph_embedding.tobytes()); "
            "the stable transition comparison therefore separately excludes "
            "official hash fields and requires canonical graph SHA256."
        ),
        "paper_eligible": False,
    }
    report["report_sha256"] = stable_json_sha256(report)
    return report


def write_generation_divergence_report(
    *,
    legacy_root: str | Path,
    optimized_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = diagnose_generation_divergence(
        legacy_root=legacy_root,
        optimized_root=optimized_root,
    )
    write_json(output / "first_divergence.json", report)
    if report["status"] == "DIVERGENCE_IDENTIFIED":
        (output / "FIRST_DIVERGENCE_IDENTIFIED").write_text(
            "COMRECGC first deterministic divergence identified.\n",
            encoding="utf-8",
        )
    else:
        (output / "NO_DIVERGENCE").write_text(
            "No JSON-trace divergence was identified.\n",
            encoding="utf-8",
        )
    return report


__all__ = [
    "DIVERGENCE_SCHEMA",
    "diagnose_generation_divergence",
    "write_generation_divergence_report",
]
