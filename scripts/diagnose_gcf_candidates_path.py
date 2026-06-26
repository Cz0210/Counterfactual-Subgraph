#!/usr/bin/env python3
"""Diagnose which file, if any, is a valid GCF_CANDIDATES_PATH.

The target file for GCF_CANDIDATES_PATH must contain complete counterfactual
graph/molecule candidates from an official GCFExplainer run. It must not be one
of this repository's fragment candidate pools or a CAMC motif-pool artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


try:  # pragma: no cover - depends on HPC runtime environment.
    from rdkit import Chem, RDLogger

    RDKIT_AVAILABLE = True
    RDKIT_IMPORT_ERROR = ""
    RDLogger.DisableLog("rdApp.*")
except Exception as exc:  # pragma: no cover - depends on runtime environment.
    Chem = None
    RDKIT_AVAILABLE = False
    RDKIT_IMPORT_ERROR = repr(exc)


SCANNED_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt"}
REFERENCE_EXTENSIONS = {".log", ".out", ".err", ".pkl", ".pt"}

FULL_CANDIDATE_FIELDS = (
    "counterfactual_smiles",
    "cf_smiles",
    "graph_smiles",
    "candidate_smiles",
    "final_smiles",
    "molecule_smiles",
    "smiles",
)
STRONG_FULL_CANDIDATE_FIELDS = (
    "counterfactual_smiles",
    "cf_smiles",
    "graph_smiles",
    "candidate_smiles",
    "final_smiles",
    "molecule_smiles",
)
PARENT_FIELDS = (
    "parent_smiles",
    "original_smiles",
    "input_smiles",
    "query_smiles",
    "source_smiles",
)
FRAGMENT_DANGER_FIELDS = (
    "final_fragment",
    "core_fragment",
    "raw_fragment",
    "fragment",
    "subgraph",
    "motif",
    "selected_motif",
    "selected_fragment",
    "atom_indices",
    "bond_indices",
    "match_atoms",
)
CAMC_DANGER_TOKENS = (
    "camc",
    "motif_pool",
    "selected_motifs",
    "fullgraph_motif",
    "gt_fullgraph_motif",
)
OURS_DANGER_TOKENS = (
    "candidate_pool.jsonl",
    "stable300",
    "sftv3",
    "ppo",
    "reward_total",
    "cf_drop",
    "cf_flip",
    "projection_used",
    "final_fragment",
    "core_fragment",
)
EXPLICIT_BAD_FILENAMES = (
    "camc_gt_fullgraph_motif_pool.csv",
    "camc_gt_fullgraph_selected_motifs.csv",
)
PRED_LABEL_DISTANCE_FIELDS = (
    "recourse",
    "cost",
    "distance",
    "ged",
    "pred",
    "prediction",
    "label",
    "teacher",
    "score",
)


@dataclass
class SmilesStats:
    field: str = ""
    sample_size: int = 0
    parse_ok_count: int = 0
    parse_ok_rate_sample: float | None = None
    avg_atom_count_sample: float | None = None
    min_atom_count_sample: int | None = None
    max_atom_count_sample: int | None = None
    contains_dummy_atom_rate: float | None = None
    contains_dot_rate: float | None = None
    likely_full_molecule_score: float = 0.0
    likely_fragment_score: float = 0.0
    rdkit_available: bool = RDKIT_AVAILABLE
    rdkit_error: str = RDKIT_IMPORT_ERROR


@dataclass
class CandidateDiagnosis:
    path: str
    absolute_path: str
    file_size: int
    modified_time: str
    file_type: str
    num_rows_estimated: int | None
    columns: list[str]
    sample_rows: list[dict[str, Any]]
    smiles_field: str
    parent_field: str
    score: int
    positive_score: int
    negative_score: int
    positive_flags: list[str] = field(default_factory=list)
    negative_flags: list[str] = field(default_factory=list)
    reason: str = ""
    is_valid_gcf_candidate: bool = False
    parse_ok_rate_sample: float | None = None
    avg_atom_count_sample: float | None = None
    min_atom_count_sample: int | None = None
    max_atom_count_sample: int | None = None
    contains_dummy_atom_rate: float | None = None
    contains_dot_rate: float | None = None
    likely_full_molecule_score: float = 0.0
    likely_fragment_score: float = 0.0
    candidate_smiles_stats_by_field: dict[str, dict[str, Any]] = field(default_factory=dict)


def _compact_json(value: Any, max_chars: int = 2000) -> str:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _safe_text(value[0])
        return "|".join(_safe_text(item) for item in value if _safe_text(item))
    return str(value).strip()


def _normalize_columns(columns: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in columns:
        column = _safe_text(item)
        if not column or column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def _read_csv(path: Path, max_sample_rows: int) -> tuple[list[str], list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    num_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _normalize_columns(reader.fieldnames or [])
        for row in reader:
            num_rows += 1
            if len(rows) < max_sample_rows:
                rows.append({key: row.get(key, "") for key in columns})
    return columns, rows, num_rows


def _read_jsonl(path: Path, max_sample_rows: int) -> tuple[list[str], list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    column_seen: set[str] = set()
    num_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            num_rows += 1
            if len(rows) >= max_sample_rows:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                obj = {"line": stripped}
            if not isinstance(obj, dict):
                obj = {"value": obj}
            for key in obj:
                key_s = _safe_text(key)
                if key_s not in column_seen:
                    column_seen.add(key_s)
                    columns.append(key_s)
            rows.append(obj)
    return columns, rows, num_rows


def _extract_json_rows(payload: Any) -> tuple[list[dict[str, Any]], list[str] | None]:
    if isinstance(payload, list):
        rows = [item if isinstance(item, dict) else {"value": item} for item in payload]
        return rows, None
    if isinstance(payload, dict):
        for key in ("rows", "data", "candidates", "records", "results", "selected", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = [item if isinstance(item, dict) else {"value": item} for item in value]
                return rows, [str(key)]
        return [payload], None
    return [{"value": payload}], None


def _read_json(path: Path, max_sample_rows: int) -> tuple[list[str], list[dict[str, Any]], int | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return ["json_error"], [{"json_error": str(exc)}], None
    rows, container_keys = _extract_json_rows(payload)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows[:max_sample_rows]:
        for key in row:
            key_s = _safe_text(key)
            if key_s not in seen:
                seen.add(key_s)
                columns.append(key_s)
    if container_keys:
        columns = container_keys + columns
    return columns, rows[:max_sample_rows], len(rows)


def _read_txt(path: Path, max_sample_rows: int) -> tuple[list[str], list[dict[str, Any]], int | None]:
    rows: list[dict[str, Any]] = []
    num_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            num_rows += 1
            if len(rows) < max_sample_rows:
                rows.append({"line": stripped})
    return ["line"], rows, num_rows


def read_table_sample(path: Path, max_sample_rows: int) -> tuple[list[str], list[dict[str, Any]], int | None]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return _read_csv(path, max_sample_rows)
        if suffix == ".jsonl":
            return _read_jsonl(path, max_sample_rows)
        if suffix == ".json":
            return _read_json(path, max_sample_rows)
        if suffix == ".txt":
            return _read_txt(path, max_sample_rows)
    except OSError as exc:
        return ["read_error"], [{"read_error": str(exc)}], None
    return [], [], None


def _candidate_smiles_fields(columns: list[str]) -> list[str]:
    column_set = {column.lower(): column for column in columns}
    fields: list[str] = []
    for field_name in FULL_CANDIDATE_FIELDS:
        if field_name in column_set:
            fields.append(column_set[field_name])
    for column in columns:
        lowered = column.lower()
        if "smiles" not in lowered:
            continue
        if lowered in {field.lower() for field in PARENT_FIELDS}:
            continue
        if column not in fields:
            fields.append(column)
    return fields


def _first_matching_field(columns: list[str], choices: Iterable[str]) -> str:
    column_set = {column.lower(): column for column in columns}
    for choice in choices:
        if choice.lower() in column_set:
            return column_set[choice.lower()]
    return ""


def _smiles_values(rows: list[dict[str, Any]], field_name: str, max_values: int) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = _safe_text(row.get(field_name))
        if value:
            values.append(value)
        if len(values) >= max_values:
            break
    return values


def _compute_smiles_stats(field_name: str, values: list[str]) -> SmilesStats:
    stats = SmilesStats(field=field_name, sample_size=len(values))
    if not values:
        return stats
    dot_count = sum(1 for value in values if "." in value)
    stats.contains_dot_rate = dot_count / len(values)

    if not RDKIT_AVAILABLE:
        return stats

    atom_counts: list[int] = []
    dummy_count = 0
    parse_ok = 0
    for value in values:
        try:
            mol = Chem.MolFromSmiles(value)
        except Exception:
            mol = None
        if mol is None:
            continue
        parse_ok += 1
        atom_count = int(mol.GetNumAtoms())
        atom_counts.append(atom_count)
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            dummy_count += 1

    stats.parse_ok_count = parse_ok
    stats.parse_ok_rate_sample = parse_ok / len(values)
    stats.contains_dummy_atom_rate = dummy_count / len(values)
    if atom_counts:
        stats.avg_atom_count_sample = sum(atom_counts) / len(atom_counts)
        stats.min_atom_count_sample = min(atom_counts)
        stats.max_atom_count_sample = max(atom_counts)

    avg_atoms = stats.avg_atom_count_sample
    parse_rate = stats.parse_ok_rate_sample or 0.0
    if parse_rate >= 0.8:
        stats.likely_full_molecule_score += 1.0
    if avg_atoms is not None and avg_atoms >= 8:
        stats.likely_full_molecule_score += 1.0
    if avg_atoms is not None and avg_atoms >= 12:
        stats.likely_full_molecule_score += 0.5
    if avg_atoms is not None and avg_atoms < 6:
        stats.likely_fragment_score += 1.0
    if (stats.contains_dummy_atom_rate or 0.0) > 0.2:
        stats.likely_fragment_score += 1.0
    if field_name.lower() in {"final_fragment", "core_fragment", "raw_fragment", "fragment"}:
        stats.likely_fragment_score += 1.0
    return stats


def _choose_best_smiles_stats(stats_by_field: dict[str, SmilesStats]) -> SmilesStats:
    if not stats_by_field:
        return SmilesStats()

    field_priority = {name: idx for idx, name in enumerate(FULL_CANDIDATE_FIELDS)}

    def key(item: tuple[str, SmilesStats]) -> tuple[float, float, float, int]:
        field_name, stats = item
        parse_rate = stats.parse_ok_rate_sample
        parse_score = -1.0 if parse_rate is None else parse_rate
        avg_atoms = stats.avg_atom_count_sample
        avg_score = -1.0 if avg_atoms is None else avg_atoms
        full_score = stats.likely_full_molecule_score - stats.likely_fragment_score
        priority = -field_priority.get(field_name.lower(), 999)
        return (parse_score, avg_score, full_score, priority)

    return max(stats_by_field.items(), key=key)[1]


def _contains_any(text: str, tokens: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _matching_columns(columns: list[str], choices: Iterable[str]) -> list[str]:
    lowered_choices = {choice.lower() for choice in choices}
    return [column for column in columns if column.lower() in lowered_choices]


def diagnose_file(path: Path, project_root: Path, max_sample_rows: int) -> CandidateDiagnosis:
    stat = path.stat()
    rel_path = path.relative_to(project_root).as_posix()
    columns, rows, num_rows = read_table_sample(path, max_sample_rows=max_sample_rows)
    candidate_fields = _candidate_smiles_fields(columns)
    parent_field = _first_matching_field(columns, PARENT_FIELDS)

    stats_by_field: dict[str, SmilesStats] = {}
    for field_name in candidate_fields:
        values = _smiles_values(rows, field_name, max_sample_rows)
        stats_by_field[field_name] = _compute_smiles_stats(field_name, values)
    best_stats = _choose_best_smiles_stats(stats_by_field)

    positive_score = 0
    negative_score = 0
    positive_flags: list[str] = []
    negative_flags: list[str] = []
    path_lower = rel_path.lower()
    file_name_lower = path.name.lower()
    column_lower_text = " ".join(column.lower() for column in columns)

    if "gcfexplainer" in path_lower or "gcfexplainer" in column_lower_text:
        positive_score += 5
        positive_flags.append("path_or_columns_contains_gcfexplainer")
    elif "gcf" in path_lower:
        positive_score += 5
        positive_flags.append("path_contains_gcf")
    if "official_aids" in path_lower:
        positive_score += 4
        positive_flags.append("path_contains_official_aids")

    present_strong_fields = _matching_columns(columns, STRONG_FULL_CANDIDATE_FIELDS)
    present_full_fields = _matching_columns(columns, FULL_CANDIDATE_FIELDS)
    if present_strong_fields:
        positive_score += 5
        positive_flags.append("has_complete_candidate_smiles_field:" + ",".join(present_strong_fields))
    elif present_full_fields:
        positive_score += 3
        positive_flags.append("has_generic_smiles_field:" + ",".join(present_full_fields))

    if best_stats.parse_ok_rate_sample is not None and best_stats.parse_ok_rate_sample >= 0.8:
        positive_score += 3
        positive_flags.append("sample_parse_ok_rate_ge_0.8")
    if best_stats.avg_atom_count_sample is not None and best_stats.avg_atom_count_sample >= 8:
        positive_score += 3
        positive_flags.append("avg_atom_count_sample_ge_8")

    pred_related = [
        column
        for column in columns
        if any(token in column.lower() for token in PRED_LABEL_DISTANCE_FIELDS)
    ]
    if pred_related:
        positive_score += 2
        positive_flags.append("has_recourse_cost_distance_pred_label_fields:" + ",".join(pred_related[:8]))
    if parent_field and present_full_fields:
        positive_score += 2
        positive_flags.append(f"has_parent_and_candidate_fields:{parent_field}+{present_full_fields[0]}")

    if "full_candidate_pools" in path_lower:
        negative_score += 10
        negative_flags.append("path_contains_full_candidate_pools")
    if any(token in path_lower for token in ("stable300", "sftv3", "ppo", "candidate_pool")):
        negative_score += 10
        negative_flags.append("path_contains_ours_candidate_pool_or_training_token")
    if file_name_lower in EXPLICIT_BAD_FILENAMES:
        negative_score += 10
        negative_flags.append("filename_is_explicit_camc_motif_artifact")
    if any(token in path_lower for token in CAMC_DANGER_TOKENS):
        negative_flags.append("path_contains_camc_or_motif_token")
    if any(token in path_lower for token in OURS_DANGER_TOKENS):
        negative_flags.append("path_contains_ours_danger_token")

    fragment_fields = _matching_columns(columns, FRAGMENT_DANGER_FIELDS)
    if fragment_fields and not present_strong_fields:
        negative_score += 8
        negative_flags.append("fragment_or_motif_fields_without_complete_candidate_field:" + ",".join(fragment_fields))
    elif fragment_fields:
        negative_flags.append("has_fragment_or_motif_fields:" + ",".join(fragment_fields))

    camc_columns = [column for column in columns if any(token in column.lower() for token in CAMC_DANGER_TOKENS)]
    if camc_columns:
        negative_flags.append("columns_contain_camc_or_motif_token:" + ",".join(camc_columns[:8]))

    ours_columns = [column for column in columns if any(token in column.lower() for token in OURS_DANGER_TOKENS)]
    if ours_columns:
        negative_flags.append("columns_contain_ours_danger_token:" + ",".join(ours_columns[:8]))

    if (best_stats.contains_dummy_atom_rate or 0.0) > 0.2:
        negative_score += 5
        negative_flags.append("dummy_atom_rate_high")
    if best_stats.avg_atom_count_sample is not None and best_stats.avg_atom_count_sample < 6:
        negative_score += 5
        negative_flags.append("avg_atom_count_sample_lt_6")
    if RDKIT_AVAILABLE and candidate_fields and (best_stats.parse_ok_rate_sample or 0.0) < 0.8:
        negative_score += 5
        negative_flags.append("sample_rdkit_parse_ok_rate_lt_0.8")
    if not RDKIT_AVAILABLE and candidate_fields:
        negative_flags.append("rdkit_unavailable_parse_checks_skipped")

    score = positive_score - negative_score

    from_gcf = (
        "gcfexplainer" in path_lower
        or "official_aids" in path_lower
        or "baselines/gcfexplainer_official" in path_lower
    )
    has_complete_candidate_field = bool(present_full_fields)
    parse_ok = RDKIT_AVAILABLE and best_stats.parse_ok_rate_sample is not None and best_stats.parse_ok_rate_sample >= 0.8
    not_fragment_primary = not (fragment_fields and not present_strong_fields)
    not_ours_pool = "full_candidate_pools" not in path_lower and not any(
        token in path_lower for token in ("stable300", "sftv3", "ppo", "candidate_pool")
    )
    not_explicit_camc = file_name_lower not in EXPLICIT_BAD_FILENAMES
    not_camc_primary = not any(token in path_lower for token in CAMC_DANGER_TOKENS)

    disqualifiers: list[str] = []
    if not from_gcf:
        disqualifiers.append("not from a GCFExplainer/official_aids path")
    if not has_complete_candidate_field:
        disqualifiers.append("no recognized complete counterfactual molecule SMILES field")
    if not parse_ok:
        if RDKIT_AVAILABLE:
            disqualifiers.append("sample RDKit parse_ok_rate is below 0.8 or unavailable")
        else:
            disqualifiers.append("RDKit unavailable; parse_ok_rate cannot satisfy required threshold")
    if not not_fragment_primary:
        disqualifiers.append("fragment/motif/subgraph fields dominate without complete candidate field")
    if not not_ours_pool:
        disqualifiers.append("looks like ours full_candidate_pools/training candidate pool")
    if not not_explicit_camc or not not_camc_primary:
        disqualifiers.append("looks like CAMC/motif output rather than GCF full-graph candidates")

    is_valid = not disqualifiers
    if is_valid:
        reason = "Valid: official GCFExplainer-related path with parseable complete candidate SMILES."
    else:
        reason = "Excluded: " + "; ".join(disqualifiers)

    return CandidateDiagnosis(
        path=rel_path,
        absolute_path=str(path.resolve()),
        file_size=int(stat.st_size),
        modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        file_type=path.suffix.lower().lstrip("."),
        num_rows_estimated=num_rows,
        columns=columns,
        sample_rows=rows,
        smiles_field=best_stats.field,
        parent_field=parent_field,
        score=score,
        positive_score=positive_score,
        negative_score=negative_score,
        positive_flags=positive_flags,
        negative_flags=negative_flags,
        reason=reason,
        is_valid_gcf_candidate=is_valid,
        parse_ok_rate_sample=best_stats.parse_ok_rate_sample,
        avg_atom_count_sample=best_stats.avg_atom_count_sample,
        min_atom_count_sample=best_stats.min_atom_count_sample,
        max_atom_count_sample=best_stats.max_atom_count_sample,
        contains_dummy_atom_rate=best_stats.contains_dummy_atom_rate,
        contains_dot_rate=best_stats.contains_dot_rate,
        likely_full_molecule_score=best_stats.likely_full_molecule_score,
        likely_fragment_score=best_stats.likely_fragment_score,
        candidate_smiles_stats_by_field={
            field_name: asdict(stats) for field_name, stats in stats_by_field.items()
        },
    )


def iter_scored_files(project_root: Path, search_roots: list[str], out_dir: Path) -> list[Path]:
    files: list[Path] = []
    out_dir_resolved = out_dir.resolve()
    for root_item in search_roots:
        root = (project_root / root_item).resolve()
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]
        for path in candidates:
            suffix = path.suffix.lower()
            if suffix not in SCANNED_EXTENSIONS:
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(out_dir_resolved)
                continue
            except ValueError:
                pass
            files.append(path)
    return sorted(set(files))


def iter_reference_artifacts(project_root: Path, search_roots: list[str], out_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    out_dir_resolved = out_dir.resolve()
    for root_item in search_roots:
        root = (project_root / root_item).resolve()
        if not root.exists():
            continue
        for path in root.rglob("*") if root.is_dir() else [root]:
            if not path.is_file() or path.suffix.lower() not in REFERENCE_EXTENSIONS:
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(out_dir_resolved)
                continue
            except ValueError:
                pass
            text = path.as_posix().lower()
            if not any(
                token in text
                for token in (
                    "gcf",
                    "gcfexplainer",
                    "official_aids",
                    "counterfactual",
                    "candidate",
                    "recourse",
                    "summary",
                    "result",
                    "selected",
                    "baseline",
                    "camc",
                )
            ):
                continue
            stat = path.stat()
            artifacts.append(
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "absolute_path": str(path.resolve()),
                    "file_size": int(stat.st_size),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    "file_type": path.suffix.lower().lstrip("."),
                    "note": "reference artifact; not scored as GCF_CANDIDATES_PATH because scanner expects CSV/JSON/JSONL/TXT",
                }
            )
    return sorted(artifacts, key=lambda item: (item["modified_time"], item["path"]), reverse=True)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return _compact_json(value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if value is None:
        return ""
    return value


def write_candidates_csv(candidates: list[CandidateDiagnosis], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "score",
        "positive_score",
        "negative_score",
        "path",
        "absolute_path",
        "is_valid_gcf_candidate",
        "reason",
        "file_type",
        "file_size",
        "modified_time",
        "num_rows_estimated",
        "columns",
        "smiles_field",
        "parent_field",
        "parse_ok_rate_sample",
        "avg_atom_count_sample",
        "min_atom_count_sample",
        "max_atom_count_sample",
        "contains_dummy_atom_rate",
        "contains_dot_rate",
        "likely_full_molecule_score",
        "likely_fragment_score",
        "negative_flags",
        "positive_flags",
        "sample_rows",
        "candidate_smiles_stats_by_field",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            payload = asdict(item)
            writer.writerow({field_name: _csv_cell(payload.get(field_name)) for field_name in fields})


def write_candidates_json(
    candidates: list[CandidateDiagnosis],
    reference_artifacts: list[dict[str, Any]],
    args: argparse.Namespace,
    out_json: Path,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(Path(args.project_root).resolve()),
        "search_roots": args.search_roots,
        "max_sample_rows": args.max_sample_rows,
        "config": args.config,
        "set": args.set,
        "rdkit_available": RDKIT_AVAILABLE,
        "rdkit_import_error": RDKIT_IMPORT_ERROR,
        "candidate_count": len(candidates),
        "valid_candidate_count": sum(1 for item in candidates if item.is_valid_gcf_candidate),
        "candidates": [asdict(item) for item in candidates],
        "reference_artifacts": reference_artifacts,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _sample_excerpt(item: CandidateDiagnosis) -> str:
    if not item.sample_rows:
        return "{}"
    row = item.sample_rows[0]
    keep_keys = []
    for key in (
        item.smiles_field,
        item.parent_field,
        "counterfactual_smiles",
        "cf_smiles",
        "graph_smiles",
        "candidate_smiles",
        "final_smiles",
        "molecule_smiles",
        "smiles",
        "final_fragment",
        "core_fragment",
        "fragment",
        "motif",
        "label",
        "pred",
        "cost",
        "distance",
        "coverage",
    ):
        if key and key in row and key not in keep_keys:
            keep_keys.append(key)
    if not keep_keys:
        keep_keys = list(row.keys())[:8]
    return _compact_json({key: row.get(key, "") for key in keep_keys}, max_chars=1000)


def _known_bad_matches(candidates: list[CandidateDiagnosis]) -> list[CandidateDiagnosis]:
    category_hits: dict[str, list[CandidateDiagnosis]] = {
        "ours_full_candidate_pool": [],
        "generic_candidate_pool": [],
        "camc_motif_pool": [],
        "camc_selected_motifs": [],
        "gt_selected_fullgraphs": [],
        "official_summary_not_candidates": [],
    }
    for item in candidates:
        path_lower = item.path.lower()
        filename = Path(item.path).name.lower()
        if "outputs/hpc/full_candidate_pools" in path_lower and "candidate_pool" in filename:
            category_hits["ours_full_candidate_pool"].append(item)
        if "candidate_pool.jsonl" in path_lower:
            category_hits["generic_candidate_pool"].append(item)
        if "camc_gt_fullgraph_motif_pool.csv" in path_lower:
            category_hits["camc_motif_pool"].append(item)
        if "camc_gt_fullgraph_selected_motifs.csv" in path_lower:
            category_hits["camc_selected_motifs"].append(item)
        if "gt_selected_fullgraphs.csv" in path_lower:
            category_hits["gt_selected_fullgraphs"].append(item)
        if "official_aids_summary" in path_lower:
            category_hits["official_summary_not_candidates"].append(item)

    bad: list[CandidateDiagnosis] = []
    seen: set[str] = set()
    for matches in category_hits.values():
        for item in sorted(matches, key=lambda candidate: candidate.score, reverse=True)[:4]:
            if item.path in seen:
                continue
            seen.add(item.path)
            bad.append(item)
    return bad[:30]


def render_report(
    candidates: list[CandidateDiagnosis],
    reference_artifacts: list[dict[str, Any]],
    project_root: Path,
    search_roots: list[str],
) -> str:
    valid_candidates = [item for item in candidates if item.is_valid_gcf_candidate]
    top_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[:20]
    lines: list[str] = []
    lines.append("# GCF_CANDIDATES_PATH Diagnosis")
    lines.append("")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- Search roots: `{', '.join(search_roots)}`")
    lines.append(f"- Scored files: `{len(candidates)}`")
    lines.append(f"- Valid GCF candidate files: `{len(valid_candidates)}`")
    lines.append(f"- RDKit available: `{str(RDKIT_AVAILABLE).lower()}`")
    if not RDKIT_AVAILABLE:
        lines.append(f"- RDKit import error: `{RDKIT_IMPORT_ERROR}`")
    lines.append("")

    lines.append("## Top ranked candidate files")
    lines.append("")
    if not top_candidates:
        lines.append("No scored candidate-like files were found.")
    else:
        lines.append(
            "| score | valid | path | smiles_field | parse_ok | avg_atoms | reason |"
        )
        lines.append("|---:|:---:|---|---|---:|---:|---|")
        for item in top_candidates:
            lines.append(
                "| {score} | {valid} | `{path}` | `{field}` | {parse_ok} | {avg_atoms} | {reason} |".format(
                    score=item.score,
                    valid="yes" if item.is_valid_gcf_candidate else "no",
                    path=item.path,
                    field=item.smiles_field or "",
                    parse_ok=_format_metric(item.parse_ok_rate_sample),
                    avg_atoms=_format_metric(item.avg_atom_count_sample),
                    reason=item.reason.replace("|", "/"),
                )
            )
    lines.append("")

    for idx, item in enumerate(top_candidates[:8], start=1):
        lines.append(f"### Top {idx}: `{item.path}`")
        lines.append("")
        lines.append(f"- Absolute path: `{item.absolute_path}`")
        lines.append(f"- Score: `{item.score}` (`+{item.positive_score}` / `-{item.negative_score}`)")
        lines.append(f"- Valid GCF candidate: `{str(item.is_valid_gcf_candidate).lower()}`")
        lines.append(f"- Candidate SMILES field: `{item.smiles_field}`")
        lines.append(f"- Parent field: `{item.parent_field}`")
        lines.append(f"- Columns: `{', '.join(item.columns[:30])}`")
        lines.append(f"- Positive flags: `{'; '.join(item.positive_flags)}`")
        lines.append(f"- Negative flags: `{'; '.join(item.negative_flags)}`")
        lines.append(f"- Reason: {item.reason}")
        lines.append(f"- Sample: `{_sample_excerpt(item)}`")
        lines.append("")

    lines.append("## Recommended GCF_CANDIDATES_PATH")
    lines.append("")
    if len(valid_candidates) == 1:
        recommended = valid_candidates[0]
        lines.append(f"`{recommended.absolute_path}`")
        lines.append("")
        lines.append(
            "This is the only file that satisfies the GCFExplainer provenance, complete candidate SMILES field, RDKit parseability, and non-fragment/non-CAMC checks."
        )
    elif len(valid_candidates) > 1:
        lines.append("Unable to uniquely determine one recommended path.")
        lines.append("")
        lines.append("Top valid candidates:")
        for item in sorted(valid_candidates, key=lambda candidate: candidate.score, reverse=True)[:3]:
            lines.append(f"- `{item.absolute_path}`: {item.reason}")
        lines.append("")
        lines.append("Please provide the head/columns of the intended file among the top valid candidates.")
    else:
        lines.append(
            "No valid GCF_CANDIDATES_PATH found. Existing files appear to be ours fragment pools, CAMC motif pools, or logs, not complete GCFExplainer counterfactual graph candidates."
        )
        lines.append("")
        lines.append("No `recommended_gcf_candidates.env` file or `final/gcf_candidates.csv` symlink should be created from the current artifacts.")
    lines.append("")

    lines.append("## Explicitly excluded paths")
    lines.append("")
    bad_matches = _known_bad_matches(candidates)
    if not bad_matches:
        lines.append("No known bad candidate-pool/CAMC paths were present among scored files.")
    else:
        for item in bad_matches:
            lines.append(f"- `{item.path}`: {item.reason}")
    lines.append("")

    lines.append("## Non-scored reference artifacts")
    lines.append("")
    if not reference_artifacts:
        lines.append("No matching log/pkl/pt reference artifacts were found.")
    else:
        for item in reference_artifacts[:30]:
            lines.append(f"- `{item['path']}` ({item['file_type']}, {item['file_size']} bytes): {item['note']}")
    lines.append("")

    official_pt = [
        item for item in reference_artifacts if item["path"].endswith("results/aids/runs/counterfactuals.pt")
    ]
    if official_pt:
        lines.append("## Official GCFExplainer raw artifact")
        lines.append("")
        lines.append(
            "`counterfactuals.pt` exists under the official AIDS reproduction output. It is the raw upstream artifact, but it is not a CSV/JSONL full-molecule SMILES table and was not promoted to `GCF_CANDIDATES_PATH` by this diagnostic."
        )
        for item in official_pt:
            lines.append(f"- `{item['absolute_path']}`")
        lines.append("")

    close_eval = project_root / "scripts" / "evaluate_close_counterfactual_coverage.py"
    lines.append("## Close CF coverage smoke readiness")
    lines.append("")
    if close_eval.exists():
        lines.append(f"- Found evaluator: `{close_eval}`")
    else:
        lines.append("- `scripts/evaluate_close_counterfactual_coverage.py` was not found, so no smoke run was attempted.")
    if not valid_candidates:
        lines.append("- Smoke run skipped because no valid GCF_CANDIDATES_PATH was identified.")
    lines.append("")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan project artifacts and score possible GCF_CANDIDATES_PATH files."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory.",
    )
    parser.add_argument(
        "--search-roots",
        nargs="+",
        default=["outputs/hpc", "logs"],
        help="Project-relative roots to scan.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/hpc/diagnostics/gcf_candidate_path_search"),
        help="Output directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--max-sample-rows",
        type=int,
        default=20,
        help="Maximum rows sampled from each candidate file.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Accepted for Slurm/config compatibility; not used by this read-only diagnostic.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Accepted for config-override compatibility; not used by this read-only diagnostic.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_sample_rows <= 0:
        raise ValueError("--max-sample-rows must be positive")

    scored_files = iter_scored_files(project_root, args.search_roots, out_dir=out_dir)
    candidates = [
        diagnose_file(path.resolve(), project_root=project_root, max_sample_rows=args.max_sample_rows)
        for path in scored_files
    ]
    candidates = sorted(candidates, key=lambda item: (item.score, item.modified_time, item.path), reverse=True)
    reference_artifacts = iter_reference_artifacts(project_root, args.search_roots, out_dir=out_dir)

    out_csv = out_dir / "gcf_candidate_path_candidates.csv"
    out_json = out_dir / "gcf_candidate_path_candidates.json"
    out_report = out_dir / "gcf_candidate_path_report.md"
    write_candidates_csv(candidates, out_csv)
    write_candidates_json(candidates, reference_artifacts, args, out_json)
    out_report.write_text(
        render_report(candidates, reference_artifacts, project_root=project_root, search_roots=args.search_roots),
        encoding="utf-8",
    )

    valid_count = sum(1 for item in candidates if item.is_valid_gcf_candidate)
    print(f"[GCF_DIAG] scored_files={len(candidates)} valid_gcf_candidates={valid_count}")
    print(f"[GCF_DIAG] wrote_csv={out_csv}")
    print(f"[GCF_DIAG] wrote_json={out_json}")
    print(f"[GCF_DIAG] wrote_report={out_report}")
    if valid_count == 1:
        valid = next(item for item in candidates if item.is_valid_gcf_candidate)
        print(f"[GCF_DIAG] recommended_gcf_candidates_path={valid.absolute_path}")
    elif valid_count == 0:
        print("[GCF_DIAG] recommended_gcf_candidates_path=NONE")
    else:
        print("[GCF_DIAG] recommended_gcf_candidates_path=AMBIGUOUS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
