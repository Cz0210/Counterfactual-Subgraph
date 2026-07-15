"""Read-only audit for GlobalGCE Frequency-Top20 Node-FGW evaluation.

The audit consumes saved candidate rankings, pair details, summaries, and
paper-report CSVs. It never loads MolCLR, runs POT, or writes to distance
caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.eval.gcf_style_recourse_report import (
    REPO_ROOT,
    best_recourse_by_parent,
    load_method_run,
)


DEFAULT_RUN_DIR = (
    "outputs/hpc/eval/"
    "ccrcov_molclr_node_fgw_globalgce_frequency_top20_lam05"
)
DEFAULT_SELECTED = (
    "outputs/hpc/selectors/globalgce_fullgraph_frequency_top20/"
    "selected_top20_for_eval.csv"
)
DEFAULT_OUTPUT = "outputs/hpc/audits/globalgce_frequency_top20_fgw_v2"
DEFAULT_REPORT_DIR = "outputs/hpc/eval/paper/molclr_node_fgw_gcf_style"
DEFAULT_THETA = 0.0328
DEFAULT_COMPARISON_RUNS = {
    "Ours": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_ours_top20_lam05_final"
    ),
    "CLEAR": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_"
        "clear_parent_frequency_top20_lam05_final"
    ),
    "GCFExplainer": (
        "outputs/hpc/eval/"
        "ccrcov_molclr_node_fgw_full_fixed_oursref1283_"
        "gcfexplainer_top20_normalized_lam05_final"
    ),
}
TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
STRICT_CONFUSION_KEYS = (
    "recorded_true_expected_true",
    "recorded_true_expected_false",
    "recorded_false_expected_true",
    "recorded_false_expected_false",
)


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    rank: int
    candidate_id: str
    candidate_smiles: str
    canonical_smiles: str
    row_index: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class PairAudit:
    method: str
    rows: list[dict[str, str]]
    parents: set[str]
    raw_method_parents: set[str]
    teacher_target_parents: set[str]
    recorded_flip_pairs: int
    expected_flip_pairs: int
    mismatch_rows: list[dict[str, Any]]
    mismatch_by_method: dict[str, int]
    confusion: dict[str, int]
    rows_before_method_filter: int
    rows_after_method_filter: int
    rows_after_reference_filter: int
    invalid_pred_before_rows: int
    invalid_pred_after_rows: int
    duplicate_parent_candidate_rows: int
    first_candidate_order: list[str]
    finite_strict_by_candidate_parent: dict[str, dict[str, tuple[float, float | None]]]


def _text(value: Any) -> str:
    if value is None:
        return ""
    rendered = str(value).strip()
    return "" if rendered.lower() in {"none", "null", "nan"} else rendered


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in TRUE_VALUES


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def normalize_strict_flip_confusion_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return one arithmetic-checked strict-flip confusion summary.

    Historical audit JSONs stored the four confusion cells under ``confusion``
    and sometimes omitted redundant totals. A complete matrix is sufficient to
    recover those totals; callers should not treat the missing aliases as a
    failed core experiment.
    """

    nested = payload.get("confusion")
    nested_confusion = nested if isinstance(nested, dict) else {}
    confusion: dict[str, int] = {}
    for key in STRICT_CONFUSION_KEYS:
        value = payload.get(key, nested_confusion.get(key))
        parsed = _as_int(value)
        if parsed is None or parsed < 0:
            raise ValueError(f"Missing or invalid strict-flip confusion field: {key}")
        confusion[key] = parsed

    true_true = confusion["recorded_true_expected_true"]
    true_false = confusion["recorded_true_expected_false"]
    false_true = confusion["recorded_false_expected_true"]
    false_false = confusion["recorded_false_expected_false"]
    derived_total = true_true + true_false + false_true + false_false
    derived_recorded_true = true_true + true_false
    derived_expected_strict = true_true + false_true
    derived_mismatch = true_false + false_true

    aliases = {
        "total_pair_rows": ("total_pair_rows", "num_pair_rows", "rows_after_reference_filter"),
        "recorded_true_pairs": ("recorded_true_pairs", "recorded_cf_flip_pairs"),
        "expected_strict_pairs": ("expected_strict_pairs", "expected_strict_flip_pairs"),
        "mismatch_rows": ("mismatch_rows",),
    }
    derived = {
        "total_pair_rows": derived_total,
        "recorded_true_pairs": derived_recorded_true,
        "expected_strict_pairs": derived_expected_strict,
        "mismatch_rows": derived_mismatch,
    }
    warnings: list[str] = []
    errors: list[str] = []
    normalized_totals: dict[str, int] = {}
    for canonical, candidate_keys in aliases.items():
        provided_key = next((key for key in candidate_keys if key in payload), None)
        provided_value = payload.get(provided_key) if provided_key is not None else None
        if provided_key is None:
            provided_key = next(
                (key for key in candidate_keys if key in nested_confusion),
                None,
            )
            provided_value = (
                nested_confusion.get(provided_key) if provided_key is not None else None
            )
        if provided_key is None:
            normalized_totals[canonical] = derived[canonical]
            warnings.append(f"inferred_missing_field:{canonical}")
            continue
        parsed = _as_int(provided_value)
        if parsed is None:
            errors.append(f"invalid_field:{provided_key}")
            normalized_totals[canonical] = derived[canonical]
        else:
            normalized_totals[canonical] = parsed
            if parsed != derived[canonical]:
                errors.append(
                    f"arithmetic_mismatch:{provided_key}:provided={parsed}:derived={derived[canonical]}"
                )

    status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    result = {
        **confusion,
        **normalized_totals,
        # Preserve legacy names while making the canonical names unambiguous.
        "recorded_cf_flip_pairs": normalized_totals["recorded_true_pairs"],
        "expected_strict_flip_pairs": normalized_totals["expected_strict_pairs"],
        "consistency_status": status,
        "consistency_warnings": warnings,
        "consistency_errors": errors,
    }
    if not errors:
        assert true_true + true_false + false_true + false_false == result["total_pair_rows"]
        assert true_true + true_false == result["recorded_true_pairs"]
        assert true_true + false_true == result["expected_strict_pairs"]
        assert true_false + false_true == result["mismatch_rows"]
    return result


def _normalize_id(value: Any) -> str:
    text = _text(value)
    number = _as_float(text)
    return str(int(number)) if number is not None and number.is_integer() else text


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _natural_key(value: str) -> tuple[Any, ...]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def load_reference_parent_ids(
    path: Path,
    *,
    parent_id_col: str = "parent_id",
) -> tuple[tuple[str, ...], dict[str, Any]]:
    fields, rows = _read_csv(path)
    if parent_id_col not in fields:
        raise ValueError(
            f"Reference parent file lacks {parent_id_col!r}; fields={fields} path={path}"
        )
    values = [_text(row.get(parent_id_col)) for row in rows]
    if any(not value for value in values):
        raise ValueError(f"Reference parent file contains empty IDs: {path}")
    if len(set(values)) != len(values):
        raise ValueError(f"Reference parent file contains duplicate IDs: {path}")
    ordered = tuple(sorted(values, key=_natural_key))
    return ordered, {
        "source_path": str(path),
        "parent_id_column": parent_id_col,
        "source_kind": "explicit_reference_parent_ids",
    }


def load_reference_from_ours_run(run_dir: Path) -> tuple[tuple[str, ...], dict[str, Any]]:
    detail_path = run_dir / "details" / "pair_details.csv"
    fields, rows = _read_csv(detail_path)
    if "parent_id" not in fields:
        raise ValueError(f"Ours reference pair details lacks parent_id: {detail_path}")
    methods = sorted({_text(row.get("method")) for row in rows if _text(row.get("method"))})
    ours_methods = [method for method in methods if "ours" in method.lower()]
    selected_method = ours_methods[0] if len(ours_methods) == 1 else (methods[0] if len(methods) == 1 else None)
    if selected_method is None:
        raise ValueError(f"Cannot identify Ours method in reference run: methods={methods}")
    parent_ids = {
        _text(row.get("parent_id"))
        for row in rows
        if _text(row.get("method")) == selected_method and _text(row.get("parent_id"))
    }
    ordered = tuple(sorted(parent_ids, key=_natural_key))
    return ordered, {
        "source_path": str(detail_path),
        "parent_id_column": "parent_id",
        "source_kind": "auto_from_final_ours_reference_run",
        "method_filter": selected_method,
    }


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else ("" if row.get(field) is None else row.get(field))
                    for field in fields
                }
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonicalize(smiles: str) -> str:
    normalized = _text(smiles)
    if not normalized:
        return ""
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(normalized)
        if molecule is not None:
            return str(Chem.MolToSmiles(molecule, canonical=True))
    except Exception:
        pass
    return normalized


def _parent_smiles_by_id(
    rows: Sequence[dict[str, str]],
    *,
    method: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        if _text(row.get("method")) != method:
            continue
        parent_id = _text(row.get("parent_id"))
        canonical = _canonicalize(_text(row.get("parent_smiles")))
        if not parent_id or not canonical:
            continue
        previous = result.get(parent_id)
        if previous is not None and previous != canonical:
            raise ValueError(
                f"Parent {parent_id!r} has inconsistent SMILES for method {method!r}"
            )
        result[parent_id] = canonical
    return result


def load_reference_from_ours_run_with_crosswalk(
    run_dir: Path,
    *,
    current_rows: Sequence[dict[str, str]],
    current_method: str,
) -> tuple[tuple[str, ...], dict[str, Any], list[dict[str, Any]]]:
    """Load an explicitly requested Ours cohort into the current ID namespace."""

    source_ids, metadata = load_reference_from_ours_run(run_dir)
    source_path = Path(metadata["source_path"])
    _source_fields, source_rows = _read_csv(source_path)
    source_method = str(metadata["method_filter"])
    source_smiles = _parent_smiles_by_id(source_rows, method=source_method)
    current_smiles = _parent_smiles_by_id(current_rows, method=current_method)
    current_ids = {
        _text(row.get("parent_id"))
        for row in current_rows
        if _text(row.get("method")) == current_method and _text(row.get("parent_id"))
    }
    current_by_smiles: dict[str, set[str]] = defaultdict(set)
    for parent_id, canonical in current_smiles.items():
        current_by_smiles[canonical].add(parent_id)

    mapped_ids: list[str] = []
    crosswalk_rows: list[dict[str, Any]] = []
    for source_id in source_ids:
        canonical = source_smiles.get(source_id, "")
        matches = sorted(current_by_smiles.get(canonical, ()), key=_natural_key) if canonical else []
        if len(matches) == 1:
            target_id = matches[0]
            match_type = "canonical_smiles"
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous parent crosswalk for Ours parent {source_id!r}: "
                f"canonical_smiles={canonical!r} current_ids={matches[:10]}"
            )
        elif not canonical and source_id in current_ids:
            target_id = source_id
            match_type = "exact_parent_id_without_smiles"
        else:
            raise ValueError(
                f"Cannot map Ours parent {source_id!r} into current GlobalGCE namespace; "
                f"canonical_smiles={canonical!r}"
            )
        mapped_ids.append(target_id)
        crosswalk_rows.append(
            {
                "parent_id": target_id,
                "source_ours_parent_id": source_id,
                "parent_smiles": canonical,
                "canonical_smiles": canonical,
                "match_type": match_type,
            }
        )

    if len(set(mapped_ids)) != len(mapped_ids):
        raise ValueError("Ours-to-GlobalGCE parent crosswalk is not one-to-one")
    ordered = tuple(sorted(mapped_ids, key=_natural_key))
    metadata.update(
        {
            "source_kind": "explicit_auto_reference_from_ours_with_crosswalk",
            "source_ours_run": str(run_dir),
            "target_parent_id_namespace": "current_globalgce_run",
            "crosswalk_applied": True,
            "crosswalk_match_counts": dict(
                Counter(row["match_type"] for row in crosswalk_rows)
            ),
        }
    )
    return ordered, metadata, crosswalk_rows


def _smiles_from_row(row: dict[str, Any]) -> str:
    for field in (
        "candidate_smiles",
        "canonical_smiles",
        "counterfactual_smiles",
        "cf_smiles",
        "smiles",
    ):
        value = _text(row.get(field))
        if value:
            return value
    return ""


def _metadata_maps(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_smiles: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = _normalize_id(row.get("candidate_id") or row.get("id"))
        canonical = _canonicalize(_smiles_from_row(row))
        if candidate_id:
            by_id.setdefault(candidate_id, row)
        if canonical:
            by_smiles.setdefault(canonical, row)
    return by_id, by_smiles


def load_selected_candidates(
    selected_path: Path,
    *,
    metadata_paths: Sequence[Path] = (),
    expected_top_k: int = 20,
) -> tuple[list[SelectedCandidate], str, list[str]]:
    _fields, selected_rows = _read_csv(selected_path)
    metadata_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in metadata_paths:
        if path.is_file():
            metadata_rows.extend(_read_csv(path)[1])
        else:
            warnings.append(f"missing_candidate_metadata:{path}")
    metadata_by_id, metadata_by_smiles = _metadata_maps(metadata_rows)
    has_rank = any(_text(row.get("rank")) for row in selected_rows)
    candidates: list[SelectedCandidate] = []
    for row_index, row in enumerate(selected_rows):
        rank = _as_int(row.get("rank")) if has_rank else row_index + 1
        if rank is None:
            raise ValueError(f"Invalid rank in {selected_path} row {row_index + 2}")
        smiles = _smiles_from_row(row)
        canonical = _canonicalize(smiles)
        candidate_id = _normalize_id(
            row.get("candidate_id")
            or row.get("id")
            or row.get("rank")
            or row_index
        )
        metadata = metadata_by_id.get(candidate_id) or metadata_by_smiles.get(canonical) or {}
        candidates.append(
            SelectedCandidate(
                rank=rank,
                candidate_id=candidate_id,
                candidate_smiles=smiles,
                canonical_smiles=canonical,
                row_index=row_index,
                metadata={**metadata, **row},
            )
        )
    if has_rank:
        candidates.sort(key=lambda candidate: candidate.rank)
        rank_source = "rank"
    else:
        rank_source = "row_order"
    if len(candidates) != int(expected_top_k):
        raise ValueError(
            f"Expected {expected_top_k} selected candidates, found {len(candidates)}: {selected_path}"
        )
    if [candidate.rank for candidate in candidates] != list(range(1, expected_top_k + 1)):
        raise ValueError(f"Candidate ranks are not exactly 1..{expected_top_k}: {selected_path}")
    if len({candidate.candidate_id for candidate in candidates}) != expected_top_k:
        raise ValueError(f"Duplicate selected candidate_id values: {selected_path}")
    if len({candidate.canonical_smiles for candidate in candidates}) != expected_top_k:
        raise ValueError(f"Duplicate selected canonical SMILES: {selected_path}")
    return candidates, rank_source, warnings


def _expected_strict_flip(row: dict[str, Any], target_label: int) -> bool:
    label = _as_int(row.get("label"))
    target = int(target_label if label is None else label)
    pred_before = _as_int(row.get("pred_before"))
    pred_after = _as_int(row.get("pred_after"))
    return pred_before == target and pred_after is not None and pred_after != target


def choose_method(rows: Sequence[dict[str, str]], requested: str | None, config: dict[str, Any]) -> str:
    available = Counter(_text(row.get("method")) for row in rows if _text(row.get("method")))
    candidates = [requested, _text(config.get("fullgraph_method")), "globalgce_frequency_top20"]
    for candidate in candidates:
        if candidate and candidate in available:
            return candidate
    global_methods = [method for method in available if "globalgce" in method.lower()]
    if len(global_methods) == 1:
        return global_methods[0]
    if len(available) == 1:
        return next(iter(available))
    raise ValueError(
        f"Cannot choose one GlobalGCE method from pair details: available={dict(available)} requested={requested!r}"
    )


def audit_pairs(
    rows: Sequence[dict[str, str]],
    *,
    method: str,
    selected: Sequence[SelectedCandidate],
    target_label: int,
    reference_parent_ids: set[str] | None = None,
) -> PairAudit:
    selected_by_id = {candidate.candidate_id: candidate for candidate in selected}
    selected_by_smiles = {candidate.canonical_smiles: candidate for candidate in selected}
    rows_after_method = [row for row in rows if _text(row.get("method")) == method]
    raw_method_parents = {
        _text(row.get("parent_id"))
        for row in rows_after_method
        if _text(row.get("parent_id"))
    }
    method_rows = [
        row
        for row in rows_after_method
        if reference_parent_ids is None or _text(row.get("parent_id")) in reference_parent_ids
    ]
    parents: set[str] = set()
    teacher_target_parents: set[str] = set()
    recorded_flip_pairs = 0
    expected_flip_pairs = 0
    mismatch_rows: list[dict[str, Any]] = []
    confusion: Counter[str] = Counter()
    first_candidate_order: list[str] = []
    seen_candidate_order: set[str] = set()
    finite_strict: dict[str, dict[str, tuple[float, float | None]]] = defaultdict(dict)
    invalid_pred_before_rows = 0
    invalid_pred_after_rows = 0
    pair_keys: list[tuple[str, str]] = []
    for row_index, row in enumerate(method_rows):
        expected = _expected_strict_flip(row, target_label)
        recorded = _as_bool(row.get("cf_flip"))
        confusion[
            f"recorded_{str(recorded).lower()}_expected_{str(expected).lower()}"
        ] += 1
        pred_before = _as_int(row.get("pred_before"))
        pred_after = _as_int(row.get("pred_after"))
        invalid_pred_before_rows += int(pred_before is None)
        invalid_pred_after_rows += int(pred_after is None)
        if recorded != expected:
            mismatch_rows.append(
                {
                    "row_index": row_index,
                    "method": method,
                    "parent_id": row.get("parent_id"),
                    "candidate_id": row.get("candidate_id"),
                    "label": row.get("label"),
                    "pred_before": row.get("pred_before"),
                    "pred_after": row.get("pred_after"),
                    "recorded_cf_flip": recorded,
                    "recorded_teacher_strict_flip": row.get("teacher_strict_flip"),
                    "expected_teacher_strict_flip": expected,
                    "old_weak_flip": (
                        pred_after is not None
                        and pred_after
                        != int(
                            _as_int(row.get("label"))
                            if _as_int(row.get("label")) is not None
                            else target_label
                        )
                    ),
                }
            )
        parent_id = _text(row.get("parent_id"))
        if parent_id:
            parents.add(parent_id)
        label = _as_int(row.get("label"))
        target = int(target_label if label is None else label)
        if pred_before == target and parent_id:
            teacher_target_parents.add(parent_id)
        recorded_flip_pairs += int(recorded)
        expected_flip_pairs += int(expected)
        raw_id = _normalize_id(row.get("candidate_id"))
        pair_identity = raw_id or _canonicalize(_text(row.get("candidate_smiles")))
        pair_keys.append((parent_id, pair_identity))
        candidate = selected_by_id.get(raw_id)
        if candidate is None:
            candidate = selected_by_smiles.get(
                _canonicalize(_text(row.get("candidate_smiles")))
            )
        if candidate is None:
            continue
        if candidate.candidate_id not in seen_candidate_order:
            seen_candidate_order.add(candidate.candidate_id)
            first_candidate_order.append(candidate.candidate_id)
        distance = _as_float(row.get("distance"))
        if not expected or distance is None:
            continue
        cf_drop = _as_float(row.get("cf_drop"))
        previous = finite_strict[candidate.candidate_id].get(parent_id)
        if previous is None or distance < previous[0]:
            finite_strict[candidate.candidate_id][parent_id] = (distance, cf_drop)
    strict_confusion = {
        key: int(confusion.get(key, 0))
        for key in STRICT_CONFUSION_KEYS
    }
    true_true = strict_confusion["recorded_true_expected_true"]
    true_false = strict_confusion["recorded_true_expected_false"]
    false_true = strict_confusion["recorded_false_expected_true"]
    false_false = strict_confusion["recorded_false_expected_false"]
    assert recorded_flip_pairs == true_true + true_false
    assert expected_flip_pairs == true_true + false_true
    assert len(mismatch_rows) == true_false + false_true
    assert len(method_rows) == true_true + true_false + false_true + false_false
    if recorded_flip_pairs == len(method_rows):
        assert len(mismatch_rows) == len(method_rows) - expected_flip_pairs
    return PairAudit(
        method=method,
        rows=method_rows,
        parents=parents,
        raw_method_parents=raw_method_parents,
        teacher_target_parents=teacher_target_parents,
        recorded_flip_pairs=recorded_flip_pairs,
        expected_flip_pairs=expected_flip_pairs,
        mismatch_rows=mismatch_rows,
        mismatch_by_method={method: len(mismatch_rows)},
        confusion=strict_confusion,
        rows_before_method_filter=len(rows),
        rows_after_method_filter=len(rows_after_method),
        rows_after_reference_filter=len(method_rows),
        invalid_pred_before_rows=invalid_pred_before_rows,
        invalid_pred_after_rows=invalid_pred_after_rows,
        duplicate_parent_candidate_rows=len(pair_keys) - len(set(pair_keys)),
        first_candidate_order=first_candidate_order,
        finite_strict_by_candidate_parent=dict(finite_strict),
    )


def _candidate_metadata_number(candidate: SelectedCandidate, fields: Sequence[str]) -> float | None:
    for field in fields:
        value = _as_float(candidate.metadata.get(field))
        if value is not None:
            return value
    return None


def _morgan_fingerprints(candidates: Sequence[SelectedCandidate]) -> dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        return {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output: dict[str, Any] = {}
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate.canonical_smiles)
        if molecule is not None:
            output[candidate.candidate_id] = generator.GetFingerprint(molecule)
    return output


def build_prefix_marginal_rows(
    pair_audit: PairAudit,
    selected: Sequence[SelectedCandidate],
    *,
    theta: float,
) -> list[dict[str, Any]]:
    fingerprints = _morgan_fingerprints(selected)
    covered: set[str] = set()
    rows: list[dict[str, Any]] = []
    previous_ids: list[str] = []
    try:
        from rdkit import DataStructs
    except ImportError:
        DataStructs = None
    for candidate in selected:
        strict_entries = pair_audit.finite_strict_by_candidate_parent.get(candidate.candidate_id, {})
        strict_distances = [value[0] for value in strict_entries.values()]
        standalone = {
            parent_id
            for parent_id, (distance, _drop) in strict_entries.items()
            if distance <= float(theta)
        }
        newly_covered = standalone - covered
        close_distances = [
            distance
            for parent_id, (distance, _drop) in strict_entries.items()
            if parent_id in standalone
        ]
        covered.update(standalone)
        similarities: list[float] = []
        if DataStructs is not None and candidate.candidate_id in fingerprints:
            similarities = [
                float(
                    DataStructs.TanimotoSimilarity(
                        fingerprints[candidate.candidate_id], fingerprints[previous_id]
                    )
                )
                for previous_id in previous_ids
                if previous_id in fingerprints
            ]
        rows.append(
            {
                "rank": candidate.rank,
                "candidate_id": candidate.candidate_id,
                "candidate_smiles": candidate.candidate_smiles,
                "frequency": _candidate_metadata_number(
                    candidate, ("frequency", "raw_frequency", "source_frequency")
                ),
                "graph_support": _candidate_metadata_number(
                    candidate, ("graph_support", "support_count", "covered_count")
                ),
                "standalone_covered_parent_count": len(standalone),
                "standalone_coverage": (
                    len(standalone) / len(pair_audit.parents) if pair_audit.parents else 0.0
                ),
                "marginal_newly_covered_parent_count": len(newly_covered),
                "cumulative_covered_parent_count": len(covered),
                "cumulative_coverage": (
                    len(covered) / len(pair_audit.parents) if pair_audit.parents else 0.0
                ),
                "strict_flip_applicable_parent_count": len(strict_entries),
                "mean_fgw_distance_strict_applicable": (
                    statistics.mean(strict_distances) if strict_distances else None
                ),
                "min_fgw_distance_strict_applicable": (
                    min(strict_distances) if strict_distances else None
                ),
                "median_fgw_distance_strict_applicable": (
                    statistics.median(strict_distances) if strict_distances else None
                ),
                "mean_fgw_distance_on_standalone_covered": (
                    statistics.mean(close_distances) if close_distances else None
                ),
                "min_fgw_distance_on_standalone_covered": (
                    min(close_distances) if close_distances else None
                ),
                "median_fgw_distance_on_standalone_covered": (
                    statistics.median(close_distances) if close_distances else None
                ),
                "max_morgan_tanimoto_to_previous": max(similarities) if similarities else 0.0,
                "theta": float(theta),
            }
        )
        previous_ids.append(candidate.candidate_id)
    return rows


def _prefix_metrics(
    pair_audit: PairAudit,
    selected: Sequence[SelectedCandidate],
    *,
    k: int,
    theta: float,
) -> dict[str, Any]:
    best: dict[str, float] = {parent_id: math.inf for parent_id in pair_audit.parents}
    for candidate in selected[: int(k)]:
        for parent_id, (distance, _drop) in pair_audit.finite_strict_by_candidate_parent.get(
            candidate.candidate_id, {}
        ).items():
            best[parent_id] = min(best[parent_id], distance)
    applicable = [distance for distance in best.values() if math.isfinite(distance)]
    covered = [distance for distance in best.values() if distance <= float(theta)]
    unconditional = list(best.values())
    return {
        "k": int(k),
        "theta": float(theta),
        "num_parents": len(best),
        "num_covered": len(covered),
        "coverage": len(covered) / len(best) if best else 0.0,
        "num_method_applicable_parents": len(applicable),
        "strict_recourse_applicable_rate": len(applicable) / len(best) if best else 0.0,
        "unconditional_median_cost": statistics.median(unconditional) if unconditional else None,
        "applicable_parent_median_cost": statistics.median(applicable) if applicable else None,
        "covered_parent_median_cost": statistics.median(covered) if covered else None,
        "covered_distance_min": min(covered) if covered else None,
        "covered_distance_max": max(covered) if covered else None,
        "covered_distance_mean": statistics.mean(covered) if covered else None,
    }


def corrected_pair_rows(
    pair_audit: PairAudit,
    *,
    target_label: int,
) -> list[dict[str, Any]]:
    corrected: list[dict[str, Any]] = []
    definition = "pred_before == target_label and pred_after != target_label"
    for row in pair_audit.rows:
        output = dict(row)
        label = _as_int(row.get("label"))
        target = int(target_label if label is None else label)
        pred_before = _as_int(row.get("pred_before"))
        pred_after = _as_int(row.get("pred_after"))
        strict = pred_before == target and pred_after is not None and pred_after != target
        weak = pred_after is not None and pred_after != target
        output.update(
            {
                "teacher_strict_flip": strict,
                "old_weak_flip": weak,
                "cf_flip": strict,
                "flip_definition": definition,
            }
        )
        corrected.append(output)
    return corrected


def _threshold_summary_rows(
    pair_audit: PairAudit,
    selected: Sequence[SelectedCandidate],
    *,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        metrics = _prefix_metrics(
            pair_audit,
            selected,
            k=len(selected),
            theta=threshold,
        )
        rows.append(
            {
                "method": pair_audit.method,
                "distance_type": "node_fgw",
                "distance_line": "MolCLR-Node-FGW",
                "cf_mode": "strict_flip",
                "main_ccrcov_uses": "teacher_strict_flip",
                "threshold": threshold,
                "num_parents": len(pair_audit.parents),
                "num_teacher_target_parents": len(pair_audit.teacher_target_parents),
                "num_candidates": len(selected),
                "num_close_cf_covered": metrics["num_covered"],
                "close_cf_coverage": metrics["coverage"],
                "theta_covered_conditional_median_cost": metrics[
                    "covered_parent_median_cost"
                ],
                "applicable_parent_median_cost": metrics[
                    "applicable_parent_median_cost"
                ],
                "strict_recourse_applicable_rate": metrics[
                    "strict_recourse_applicable_rate"
                ],
                "flip_definition": (
                    "pred_before == target_label and pred_after != target_label"
                ),
            }
        )
    return rows


def write_corrected_teacher_strict_outputs(
    *,
    output_dir: Path,
    source_fields: Sequence[str],
    pair_audit: PairAudit,
    selected: Sequence[SelectedCandidate],
    source_config: dict[str, Any],
    source_pair_details: Path,
    target_label: int,
    thresholds: Sequence[float],
    reference_metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected = corrected_pair_rows(pair_audit, target_label=target_label)
    corrected_fields = list(source_fields)
    for field in ("teacher_strict_flip", "old_weak_flip", "cf_flip", "flip_definition"):
        if field not in corrected_fields:
            corrected_fields.append(field)
    _write_csv(
        output_dir / "pair_details_teacher_strict.csv",
        corrected,
        corrected_fields,
    )
    summary_rows = _threshold_summary_rows(
        pair_audit,
        selected,
        thresholds=thresholds,
    )
    _write_csv(
        output_dir / "combined_threshold_summary.csv",
        summary_rows,
        list(summary_rows[0]) if summary_rows else [],
    )
    strict_audit = {
        "method_filter": pair_audit.method,
        "target_label": int(target_label),
        "rows_before_method_filter": pair_audit.rows_before_method_filter,
        "rows_after_method_filter": pair_audit.rows_after_method_filter,
        "rows_after_reference_filter": pair_audit.rows_after_reference_filter,
        "recorded_cf_flip_pairs": pair_audit.recorded_flip_pairs,
        "expected_strict_flip_pairs": pair_audit.expected_flip_pairs,
        "mismatch_rows": len(pair_audit.mismatch_rows),
        "confusion": pair_audit.confusion,
        "invalid_pred_before_rows": pair_audit.invalid_pred_before_rows,
        "invalid_pred_after_rows": pair_audit.invalid_pred_after_rows,
        "duplicate_parent_candidate_rows": pair_audit.duplicate_parent_candidate_rows,
        "reference_cohort": reference_metadata,
    }
    _write_json(output_dir / "strict_flip_audit.json", strict_audit)
    corrected_config = {
        **source_config,
        "source_pair_details": str(source_pair_details),
        "corrected_pair_details": str(output_dir / "pair_details_teacher_strict.csv"),
        "cf_mode": "strict_flip",
        "main_ccrcov_uses": "teacher_strict_flip",
        "teacher_strict_flip_definition": (
            "pred_before == target_label and pred_after != target_label"
        ),
        "old_weak_flip_status": "audit_only",
        "distance_recomputed": False,
        "reference_cohort": reference_metadata,
    }
    _write_json(output_dir / "run_config.json", corrected_config)
    return {
        "output_dir": str(output_dir),
        "num_corrected_rows": len(corrected),
        "num_threshold_rows": len(summary_rows),
    }


def audit_candidate_order(
    selected: Sequence[SelectedCandidate],
    pair_audit: PairAudit,
    *,
    frequency_groups: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluator_index = {
        candidate_id: index for index, candidate_id in enumerate(pair_audit.first_candidate_order)
    }
    frequency_groups = frequency_groups or {}
    rows: list[dict[str, Any]] = []
    for candidate in selected:
        group = frequency_groups.get(candidate.canonical_smiles, {})
        rows.append(
            {
                "rank": candidate.rank,
                "candidate_id": candidate.candidate_id,
                "candidate_smiles": candidate.candidate_smiles,
                "canonical_smiles": candidate.canonical_smiles,
                "frequency": _candidate_metadata_number(
                    candidate, ("frequency", "raw_frequency", "source_frequency")
                ),
                "graph_support": _candidate_metadata_number(
                    candidate, ("graph_support", "support_count", "covered_count")
                ),
                "computed_raw_frequency": group.get("raw_frequency"),
                "source_raw_index_count": group.get("source_raw_index_count"),
                "source_path_count": group.get("source_path_count"),
                "evaluator_read": candidate.candidate_id in evaluator_index,
                "evaluator_internal_index": evaluator_index.get(candidate.candidate_id),
                "expected_internal_index": candidate.rank - 1,
                "order_matches": evaluator_index.get(candidate.candidate_id) == candidate.rank - 1,
                "duplicate_selected_id": False,
                "duplicate_selected_smiles": False,
            }
        )
    summary = {
        "unique_candidates": len({candidate.candidate_id for candidate in selected}),
        "unique_canonical_smiles": len(
            {candidate.canonical_smiles for candidate in selected}
        ),
        "evaluator_unique_candidates": len(pair_audit.first_candidate_order),
        "all_candidates_read": all(row["evaluator_read"] for row in rows),
        "rank_order_preserved": all(row["order_matches"] for row in rows),
        "candidate_order_selected": [candidate.candidate_id for candidate in selected],
        "candidate_order_evaluator": pair_audit.first_candidate_order,
        "raw_14032_pool_misused": len(pair_audit.first_candidate_order) != len(selected),
    }
    return rows, summary


def build_frequency_groups(valid_rows: Sequence[dict[str, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        canonical = _canonicalize(_smiles_from_row(row))
        if canonical:
            grouped[canonical].append(row)
    result: dict[str, dict[str, Any]] = {}
    duplicate_source_records = 0
    for canonical, rows in grouped.items():
        source_keys = [
            (_text(row.get("source_path")), _text(row.get("raw_index")))
            for row in rows
            if _text(row.get("raw_index"))
        ]
        duplicate_source_records += len(source_keys) - len(set(source_keys))
        graph_ids = {
            _text(row.get("graph_idx")) for row in rows if _text(row.get("graph_idx"))
        }
        result[canonical] = {
            "raw_frequency": len(rows),
            "graph_support": len(graph_ids),
            "source_raw_index_count": len(
                {_text(row.get("raw_index")) for row in rows if _text(row.get("raw_index"))}
            ),
            "source_path_count": len(
                {_text(row.get("source_path")) for row in rows if _text(row.get("source_path"))}
            ),
            "cf_index_count": len(
                {_text(row.get("cf_index")) for row in rows if _text(row.get("cf_index"))}
            ),
            "duplicate_source_record_count": len(source_keys) - len(set(source_keys)),
            "missing_source_raw_index_count": sum(
                not bool(_text(row.get("raw_index"))) for row in rows
            ),
        }
    return result, {
        "raw_rows": len(valid_rows),
        "unique_canonical_smiles": len(result),
        "duplicate_source_record_count": duplicate_source_records,
        "frequency_group_key": "canonical_smiles",
        "frequency_counts_selected_only": False,
        "frequency_counts_duplicate_false_only": False,
    }


def _find_optional_csv(directory: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _auto_valid_candidates(selector_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        return _resolve(explicit)
    summary = _read_json(selector_dir / "selector_summary.json", required=False)
    for key, value in summary.items():
        if "valid" in key.lower() and "candidate" in key.lower() and _text(value):
            path = _resolve(_text(value))
            if path.is_file():
                return path
    common = (
        "outputs/hpc/eval/globalgce/aids_hiv_native_cf_fullgraph/label1/valid_candidates.csv",
        "outputs/hpc/eval/globalgce/aids_hiv_native_cf_fullgraph_all_valid/label1/valid_candidates.csv",
    )
    for value in common:
        path = _resolve(value)
        if path.is_file():
            return path
    return None


def _nearest_summary_row(
    rows: Sequence[dict[str, str]], method: str, theta: float, tolerance: float
) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if _text(row.get("method")) == method and _as_float(row.get("threshold")) is not None
    ]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: abs(float(row["threshold"]) - theta))
    return nearest if abs(float(nearest["threshold"]) - theta) <= tolerance else None


def _report_csv(report_dir: Path, suffix: str) -> Path | None:
    matches = sorted(report_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _find_report_row(
    path: Path | None,
    *,
    method: str,
    k: int | None = None,
    theta: float | None = None,
    tolerance: float = 1e-8,
) -> dict[str, str] | None:
    if path is None or not path.is_file():
        return None
    _fields, rows = _read_csv(path)
    filtered = [row for row in rows if _text(row.get("method") or row.get("Method")) == method]
    if k is not None:
        filtered = [row for row in filtered if _as_int(row.get("k") or row.get("K")) == int(k)]
    if theta is not None:
        exact = [
            row
            for row in filtered
            if _as_float(row.get("threshold") or row.get("theta") or row.get("Theta")) is not None
            and abs(
                float(row.get("threshold") or row.get("theta") or row.get("Theta"))
                - float(theta)
            )
            <= tolerance
        ]
        return exact[0] if exact else None
    return filtered[0] if filtered else None


def build_consistency_audit(
    pair_audit: PairAudit,
    selected: Sequence[SelectedCandidate],
    *,
    theta: float,
    table_k: int,
    max_k: int,
    combined_rows: Sequence[dict[str, str]],
    report_dir: Path,
    tolerance: float,
) -> dict[str, Any]:
    prefix = [_prefix_metrics(pair_audit, selected, k=k, theta=theta) for k in range(1, max_k + 1)]
    prefix_monotone = all(
        left["coverage"] <= right["coverage"] + tolerance
        for left, right in zip(prefix, prefix[1:])
    )
    thresholds = sorted(
        {
            theta,
            *[
                float(row["threshold"])
                for row in combined_rows
                if _text(row.get("method")) == pair_audit.method
                and _as_float(row.get("threshold")) is not None
            ],
        }
    )
    threshold_values = [
        _prefix_metrics(pair_audit, selected, k=table_k, theta=value)["coverage"]
        for value in thresholds
    ]
    threshold_monotone = all(
        left <= right + tolerance
        for left, right in zip(threshold_values, threshold_values[1:])
    )
    computed_k10 = prefix[table_k - 1]
    computed_k20 = prefix[max_k - 1]
    table_path = _report_csv(report_dir, "table2_gcf_style_fgw.csv")
    figure3_path = _report_csv(report_dir, "figure3_fgw_coverage_cost_vs_k.csv")
    figure4_path = _report_csv(report_dir, "figure4_fgw_coverage_vs_threshold.csv")
    table_row = _find_report_row(table_path, method="GlobalGCE", k=table_k, theta=theta, tolerance=tolerance)
    figure3_k10 = _find_report_row(figure3_path, method="GlobalGCE", k=table_k)
    figure3_k20 = _find_report_row(figure3_path, method="GlobalGCE", k=max_k)
    figure4_k10 = _find_report_row(figure4_path, method="GlobalGCE", k=table_k, theta=theta, tolerance=tolerance)
    figure4_k20 = _find_report_row(figure4_path, method="GlobalGCE", k=max_k, theta=theta, tolerance=tolerance)
    combined_row = _nearest_summary_row(combined_rows, pair_audit.method, theta, tolerance)

    def coverage(row: dict[str, str] | None) -> float | None:
        if not row:
            return None
        return _as_float(row.get("coverage") or row.get("Coverage") or row.get("close_cf_coverage"))

    def check(value: float | None, expected: float) -> bool | None:
        return None if value is None else abs(value - expected) <= tolerance

    table_fields = _read_csv(table_path)[0] if table_path and table_path.is_file() else []
    figure3_rows = _read_csv(figure3_path)[1] if figure3_path and figure3_path.is_file() else []
    plotted_prefix = sorted(
        (
            (_as_int(row.get("k")), _as_float(row.get("coverage")))
            for row in figure3_rows
            if _text(row.get("method")) == "GlobalGCE"
        ),
        key=lambda item: (-1 if item[0] is None else item[0]),
    )
    plotted_prefix_values = [
        value for k_value, value in plotted_prefix if k_value is not None and value is not None
    ]
    plotted_prefix_monotone = all(
        left <= right + tolerance
        for left, right in zip(plotted_prefix_values, plotted_prefix_values[1:])
    )
    figure4_rows = _read_csv(figure4_path)[1] if figure4_path and figure4_path.is_file() else []
    plotted_threshold = sorted(
        (
            (_as_float(row.get("threshold")), _as_float(row.get("coverage")))
            for row in figure4_rows
            if _text(row.get("method")) == "GlobalGCE"
            and _as_int(row.get("k")) == table_k
        ),
        key=lambda item: (-math.inf if item[0] is None else item[0]),
    )
    plotted_threshold_values = [
        value
        for threshold_value, value in plotted_threshold
        if threshold_value is not None and value is not None
    ]
    plotted_threshold_monotone = all(
        left <= right + tolerance
        for left, right in zip(plotted_threshold_values, plotted_threshold_values[1:])
    )

    return {
        "theta": theta,
        "table_k": table_k,
        "max_k": max_k,
        "prefix_coverage_monotone": prefix_monotone,
        "threshold_coverage_monotone": threshold_monotone,
        "plotted_prefix_coverage_monotone": plotted_prefix_monotone,
        "plotted_threshold_coverage_monotone": plotted_threshold_monotone,
        "computed_prefix_rows": prefix,
        "computed_k10_coverage": computed_k10["coverage"],
        "computed_k20_coverage": computed_k20["coverage"],
        "table2_path": str(table_path) if table_path else None,
        "figure3_path": str(figure3_path) if figure3_path else None,
        "figure4_path": str(figure4_path) if figure4_path else None,
        "table2_k10_coverage": coverage(table_row),
        "figure3_k10_coverage": coverage(figure3_k10),
        "figure3_k20_coverage": coverage(figure3_k20),
        "figure4_exact_theta_k10_coverage": coverage(figure4_k10),
        "figure4_exact_theta_k20_coverage": coverage(figure4_k20),
        "combined_summary_k20_coverage": coverage(combined_row),
        "table2_k10_matches": check(coverage(table_row), computed_k10["coverage"]),
        "figure3_k10_matches": check(coverage(figure3_k10), computed_k10["coverage"]),
        "figure3_k20_matches": check(coverage(figure3_k20), computed_k20["coverage"]),
        "figure4_exact_theta_k10_matches": check(coverage(figure4_k10), computed_k10["coverage"]),
        "figure4_exact_theta_k20_matches": check(coverage(figure4_k20), computed_k20["coverage"]),
        "combined_summary_k20_matches": check(coverage(combined_row), computed_k20["coverage"]),
        "figure4_exact_theta_present": figure4_k10 is not None,
        "figure4_note": (
            "The uniform plotting grid does not contain theta_star exactly; exact-theta coverage was recomputed from saved pair details."
            if figure4_k10 is None
            else "Exact theta is present in the plotted CSV grid."
        ),
        "table_uses_legacy_conditional_median_name": "Conditional median cost" in table_fields,
        "table_uses_legacy_applicable_rate_name": "Applicable rate" in table_fields,
    }


def build_metric_definition_audit(
    pair_audit: PairAudit,
    selected: Sequence[SelectedCandidate],
    *,
    theta: float,
    k: int,
) -> dict[str, Any]:
    metrics = _prefix_metrics(pair_audit, selected, k=k, theta=theta)
    covered_median = metrics["covered_parent_median_cost"]
    covered_assertion = covered_median is None or covered_median <= theta + 1e-12
    return {
        "k": k,
        "theta": theta,
        "coverage_definition": (
            "count(parent with min distance among prefix-K teacher-strict-flip candidates <= theta) / all evaluated parents"
        ),
        "unconditional_median_cost_definition": (
            "median over all evaluated parents of best prefix-K strict-flip distance; unavailable parents are +inf"
        ),
        "applicable_parent_median_cost_definition": (
            "median best prefix-K strict-flip distance among parents with any finite strict-flip recourse; no theta conditioning"
        ),
        "covered_parent_median_cost_definition": (
            "median best prefix-K strict-flip distance among parents whose best distance <= theta"
        ),
        "legacy_conditional_median_label_status": "ambiguous_name_for_applicable_parent_median",
        "applicable_rate_definition": (
            "parents with any finite prefix-K teacher-strict-flip recourse / all evaluated parents"
        ),
        "teacher_target_parent_rate_definition": (
            "parents with pred_before == target_label / all evaluated parents"
        ),
        "metrics": metrics,
        "num_covered_parents_in_cost_distribution": metrics["num_covered"],
        "covered_parent_median_le_theta": covered_assertion,
    }


def build_applicable_rows(
    current_pair: PairAudit,
    selected: Sequence[SelectedCandidate],
    *,
    current_run_dir: Path,
    comparison_specs: dict[str, str],
    expected_top_k: int,
    expected_parents: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    current_metrics = _prefix_metrics(
        current_pair, selected, k=expected_top_k, theta=float("inf")
    )
    rows = [
        {
            "method": "GlobalGCE",
            "run_dir": str(current_run_dir),
            "num_all_parents": len(current_pair.parents),
            "num_teacher_target_parents": len(current_pair.teacher_target_parents),
            "teacher_target_parent_rate": (
                len(current_pair.teacher_target_parents) / len(current_pair.parents)
                if current_pair.parents
                else 0.0
            ),
            "num_method_applicable_parents": current_metrics["num_method_applicable_parents"],
            "applicable_rate_numerator": current_metrics["num_method_applicable_parents"],
            "applicable_rate_denominator": len(current_pair.parents),
            "applicable_rate": current_metrics["strict_recourse_applicable_rate"],
            "applicable_equals_teacher_target_rate": (
                current_metrics["num_method_applicable_parents"]
                == len(current_pair.teacher_target_parents)
            ),
            "fullgraph_candidates_default_applicable": True,
        }
    ]
    for display_name, path_like in comparison_specs.items():
        run_dir = _resolve(path_like)
        if not (run_dir / "run_config.json").is_file():
            warnings.append(f"missing_comparison_run:{display_name}:{run_dir}")
            continue
        try:
            run = load_method_run(
                display_name,
                run_dir,
                expected_top_k=expected_top_k,
                expected_num_parents=expected_parents,
            )
            distances, _drops = best_recourse_by_parent(run, k=expected_top_k)
            method_applicable = sum(math.isfinite(value) for value in distances.values())
            detail_rows = _read_csv(run_dir / "details" / "pair_details.csv")[1]
            teacher_targets = {
                _text(row.get("parent_id"))
                for row in detail_rows
                if _as_int(row.get("pred_before")) == (_as_int(row.get("label")) or 0)
            }
            rows.append(
                {
                    "method": display_name,
                    "run_dir": str(run_dir),
                    "num_all_parents": len(run.parent_ids),
                    "num_teacher_target_parents": len(teacher_targets),
                    "teacher_target_parent_rate": len(teacher_targets) / len(run.parent_ids),
                    "num_method_applicable_parents": method_applicable,
                    "applicable_rate_numerator": method_applicable,
                    "applicable_rate_denominator": len(run.parent_ids),
                    "applicable_rate": method_applicable / len(run.parent_ids),
                    "applicable_equals_teacher_target_rate": method_applicable == len(teacher_targets),
                    "fullgraph_candidates_default_applicable": display_name != "Ours",
                }
            )
        except Exception as exc:
            warnings.append(f"comparison_run_audit_failed:{display_name}:{exc}")
    return rows


def _raw_parent_ids_for_run(run_dir: Path, display_name: str) -> tuple[set[str], str, Path]:
    detail_path = run_dir / "details" / "pair_details.csv"
    _fields, rows = _read_csv(detail_path)
    methods = sorted({_text(row.get("method")) for row in rows if _text(row.get("method"))})
    preferred = [method for method in methods if display_name.lower() in method.lower()]
    if display_name == "Ours":
        preferred = [method for method in methods if "ours" in method.lower()]
    method = preferred[0] if len(preferred) == 1 else (methods[0] if len(methods) == 1 else "")
    if not method:
        raise ValueError(f"Cannot choose method for {display_name}: methods={methods} path={detail_path}")
    parent_ids = {
        _text(row.get("parent_id"))
        for row in rows
        if _text(row.get("method")) == method and _text(row.get("parent_id"))
    }
    return parent_ids, method, detail_path


def build_parent_cohort_audit(
    *,
    pair_audit: PairAudit,
    current_run_dir: Path,
    reference_parent_ids: Sequence[str] | None,
    comparison_specs: dict[str, str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    reference = set(reference_parent_ids or ())
    cohorts: list[tuple[str, Path, str, set[str]]] = [
        ("GlobalGCE", current_run_dir, pair_audit.method, set(pair_audit.raw_method_parents))
    ]
    for display_name, path_like in comparison_specs.items():
        run_dir = _resolve(path_like)
        if not (run_dir / "details" / "pair_details.csv").is_file():
            warnings.append(f"missing_parent_cohort_run:{display_name}:{run_dir}")
            continue
        try:
            parent_ids, method, _detail_path = _raw_parent_ids_for_run(run_dir, display_name)
            cohorts.append((display_name, run_dir, method, parent_ids))
        except Exception as exc:
            warnings.append(f"parent_cohort_audit_failed:{display_name}:{exc}")
    rows: list[dict[str, Any]] = []
    for display_name, run_dir, method, parent_ids in cohorts:
        missing = sorted(reference - parent_ids, key=_natural_key) if reference else []
        extra = sorted(parent_ids - reference, key=_natural_key) if reference else []
        intersection = parent_ids & reference if reference else set(parent_ids)
        rows.append(
            {
                "method_display": display_name,
                "evaluator_method": method,
                "run_dir": str(run_dir),
                "num_parent_ids_raw": len(parent_ids),
                "num_reference_parent_ids": len(reference) if reference else None,
                "intersection_size": len(intersection),
                "missing_id_count": len(missing),
                "extra_id_count": len(extra),
                "missing_ids": missing,
                "extra_ids": extra,
                "exact_set_match": parent_ids == reference if reference else None,
                "reference_filter_applied": bool(reference),
                "num_parent_ids_after_reference_filter": len(intersection),
                "final_cohort_usable": (not missing) if reference else False,
            }
        )
    return rows


def render_report(summary: dict[str, Any]) -> str:
    strict = summary["strict_flip_audit"]
    order = summary["candidate_order_audit"]
    consistency = summary["metric_consistency_audit"]
    frequency = summary["frequency_audit"]
    metric = summary["metric_definition_audit"]
    fullgraph = summary["fullgraph_semantics_audit"]
    applicable_rows = summary["applicable_rate_audit"]
    marginal_rows = summary["prefix_marginal_audit"]
    cohort_rows = summary["parent_cohort_audit"]
    lines = [
        "GlobalGCE Frequency-Top20 MolCLR-Node-FGW audit",
        "=" * 54,
        "",
        "1. Scope",
        f"- Run directory: {summary['run_dir']}",
        f"- Pair details: {summary['pair_details']}",
        "- Audit is read-only; no MolCLR embedding or FGW distance was recomputed.",
        "",
        "2. Strict flip",
        f"- Pair rows: {strict['num_pair_rows']}",
        f"- Method filter: {strict['method_filter']}",
        f"- Target label: {strict['target_label']}",
        f"- Rows before method filter: {strict['rows_before_method_filter']}",
        f"- Rows after method filter: {strict['rows_after_method_filter']}",
        f"- Rows after reference filter: {strict['rows_after_reference_filter']}",
        f"- Teacher-target parents: {strict['num_teacher_target_parents']}",
        f"- Recorded cf_flip pairs: {strict['recorded_cf_flip_pairs']}",
        f"- Expected teacher-strict pairs: {strict['expected_strict_flip_pairs']}",
        f"- Mismatch rows: {strict['mismatch_rows']}",
        f"- Confusion: {strict['confusion']}",
        f"- Invalid pred_before rows: {strict['invalid_pred_before_rows']}",
        f"- Invalid pred_after rows: {strict['invalid_pred_after_rows']}",
        f"- Duplicate parent-candidate rows: {strict['duplicate_parent_candidate_rows']}",
        "- Definition: pred_before == target_label and pred_after != target_label.",
        f"- Cohort mode: {summary['cohort_mode']}",
        f"- Reference source: {summary['reference_cohort']}",
        "- Parent cohort rows:",
        *[
            (
                f"  {row['method_display']}: raw={row['num_parent_ids_raw']} "
                f"intersection={row['intersection_size']} missing={row['missing_id_count']} "
                f"extra={row['extra_id_count']} exact={row['exact_set_match']}"
            )
            for row in cohort_rows
        ],
        "",
        "3. Candidate order",
        f"- Unique candidates: {order['unique_candidates']}",
        f"- Evaluator unique candidates: {order['evaluator_unique_candidates']}",
        f"- Rank order preserved: {order['rank_order_preserved']}",
        f"- Selected path matches run_config: {order['selected_path_matches_run_config']}",
        f"- Frequency is non-increasing by rank: {order['frequency_non_increasing']}",
        f"- Raw 14032-candidate pool misused: {order['raw_14032_pool_misused']}",
        "",
        "4. Metric consistency",
        f"- Prefix coverage monotone: {consistency['prefix_coverage_monotone']}",
        f"- Threshold coverage monotone: {consistency['threshold_coverage_monotone']}",
        f"- Plotted prefix coverage monotone: {consistency['plotted_prefix_coverage_monotone']}",
        f"- Plotted threshold coverage monotone: {consistency['plotted_threshold_coverage_monotone']}",
        f"- Computed K=10 coverage: {consistency['computed_k10_coverage']}",
        f"- Computed K=20 coverage: {consistency['computed_k20_coverage']}",
        f"- Figure 4 exact theta present: {consistency['figure4_exact_theta_present']}",
        f"- Figure 4 note: {consistency['figure4_note']}",
        "",
        "5. Frequency definition",
        f"- Raw valid rows: {frequency.get('raw_rows')}",
        f"- Unique canonical SMILES: {frequency.get('unique_canonical_smiles')}",
        f"- Frequency definition confirmed: {frequency.get('frequency_definition_confirmed')}",
        f"- Duplicate source records: {frequency.get('duplicate_source_record_count')}",
        f"- Source paths represented: {frequency.get('num_source_paths')}",
        "",
        "6. Cost definitions",
        f"- Legacy conditional label: {metric['legacy_conditional_median_label_status']}",
        f"- Applicable-parent median: {metric['metrics']['applicable_parent_median_cost']}",
        f"- Covered-parent median: {metric['metrics']['covered_parent_median_cost']}",
        f"- Covered-parent median <= theta: {metric['covered_parent_median_le_theta']}",
        f"- Corrected Table 2 metrics: {summary['corrected_table2_metrics']}",
        "",
        "7. Applicable rate",
        *[
            (
                f"- {row['method']}: {row['applicable_rate_numerator']}/"
                f"{row['applicable_rate_denominator']}={row['applicable_rate']}; "
                f"teacher-target={row['num_teacher_target_parents']}; "
                f"equal={row['applicable_equals_teacher_target_rate']}"
            )
            for row in applicable_rows
        ],
        "",
        "8. Fullgraph semantics",
        f"- Definition: {fullgraph['path_definition']}",
        f"- Uses subgraph deletion: {fullgraph['uses_subgraph_deletion']}",
        f"- match=false rows: {fullgraph['match_false_rows']}",
        f"- delete_valid=false rows: {fullgraph['delete_valid_false_rows']}",
        "",
        "9. Prefix marginal coverage",
        *[
            (
                f"- rank={row['rank']} candidate={row['candidate_id']} "
                f"standalone={row['standalone_covered_parent_count']} "
                f"marginal={row['marginal_newly_covered_parent_count']} "
                f"cumulative={row['cumulative_covered_parent_count']}"
            )
            for row in marginal_rows
        ],
        "",
        "10. Classification",
        f"- Result class: {summary['result_class']}",
        f"- Implementation findings: {summary['implementation_findings']}",
        f"- Adaptation findings: {summary['adaptation_findings']}",
        "",
        "11. Conclusion",
        summary["conclusion"],
    ]
    if summary.get("warnings"):
        lines.extend(("", "Warnings", *[f"- {warning}" for warning in summary["warnings"]]))
    return "\n".join(lines) + "\n"


def _parse_specs(raw_specs: Sequence[str]) -> dict[str, str]:
    # These are data inputs, not display defaults. Only paths explicitly
    # supplied by the caller may be opened by the audit.
    specs: dict[str, str] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"--comparison-run requires METHOD=PATH, got {raw!r}")
        method, path = raw.split("=", 1)
        specs[method.strip()] = path.strip()
    return specs


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _resolve(args.run_dir)
    output_dir = _resolve(args.output_dir)
    selected_path = _resolve(args.selected_top20)
    selector_dir = selected_path.parent
    pair_details = _resolve(args.pair_details) if args.pair_details else run_dir / "details" / "pair_details.csv"
    combined_path = run_dir / "combined" / "combined_threshold_summary.csv"
    config = _read_json(run_dir / "run_config.json")
    _cache_stats = _read_json(run_dir / "cache_stats.json", required=False)
    _pair_fields, all_pair_rows = _read_csv(pair_details)
    _summary_fields, combined_rows = _read_csv(combined_path)
    metadata_paths = [
        _resolve(args.selected_top20_metadata)
        if args.selected_top20_metadata
        else selector_dir / "selected_top20.csv",
        _resolve(args.frequency_ranked_candidates)
        if args.frequency_ranked_candidates
        else selector_dir / "frequency_ranked_candidates.csv",
    ]
    selected, rank_source, warnings = load_selected_candidates(
        selected_path,
        metadata_paths=metadata_paths,
        expected_top_k=int(args.max_k),
    )
    method = choose_method(all_pair_rows, args.method_name, config)
    comparison_specs = _parse_specs(args.comparison_run)
    reference_parent_ids: tuple[str, ...] | None = None
    reference_crosswalk_rows: list[dict[str, Any]] = []
    reference_metadata: dict[str, Any] = {
        "source_kind": "all_label_parent_diagnostic",
        "source_path": None,
        "parent_id_column": "parent_id",
        "num_reference_parents": None,
    }
    reference_parent_arg = _text(args.reference_parent_ids)
    auto_reference_sentinel = "auto_from_final_ours_reference_run"
    explicit_reference_path = bool(
        reference_parent_arg and reference_parent_arg != auto_reference_sentinel
    )
    explicit_ours_comparison = "Ours" in comparison_specs
    explicit_auto_reference = bool(
        reference_parent_arg == auto_reference_sentinel
        or args.auto_reference_from_ours
        or args.reference_ours_run
        or explicit_ours_comparison
    )
    if explicit_reference_path and (args.auto_reference_from_ours or args.reference_ours_run):
        raise ValueError(
            "Use either an explicit --reference-parent-ids CSV or an Ours auto-reference option, not both"
        )
    if explicit_reference_path:
        reference_parent_ids, reference_metadata = load_reference_parent_ids(
            _resolve(reference_parent_arg),
            parent_id_col=args.reference_parent_id_col,
        )
    elif explicit_auto_reference:
        ours_path = (
            _text(args.reference_ours_run)
            or comparison_specs.get("Ours")
            or DEFAULT_COMPARISON_RUNS["Ours"]
        )
        ours_run = _resolve(ours_path)
        reference_parent_ids, reference_metadata, reference_crosswalk_rows = (
            load_reference_from_ours_run_with_crosswalk(
                ours_run,
                current_rows=all_pair_rows,
                current_method=method,
            )
        )
    else:
        warnings.append(
            "reference_parent_cohort_unavailable_running_all_label_parent_diagnostic"
        )
    if reference_parent_ids is not None:
        reference_metadata["num_reference_parents"] = len(reference_parent_ids)
        expected_reference = int(args.expected_reference_parents)
        if expected_reference > 0 and len(reference_parent_ids) != expected_reference:
            raise ValueError(
                f"Expected {expected_reference} reference parents, found "
                f"{len(reference_parent_ids)} from {reference_metadata['source_path']}"
            )
    reference_set = set(reference_parent_ids) if reference_parent_ids is not None else None
    pair_audit = audit_pairs(
        all_pair_rows,
        method=method,
        selected=selected,
        target_label=int(args.target_label),
        reference_parent_ids=reference_set,
    )
    if reference_set is not None:
        missing_current = sorted(reference_set - pair_audit.raw_method_parents, key=_natural_key)
        if missing_current:
            raise ValueError(
                f"GlobalGCE run is missing {len(missing_current)} reference parent IDs: "
                f"sample={missing_current[:10]}"
            )
    valid_path = _auto_valid_candidates(selector_dir, args.valid_candidates)
    frequency_groups: dict[str, dict[str, Any]] = {}
    frequency_summary: dict[str, Any] = {
        "valid_candidates_path": str(valid_path) if valid_path else None,
        "frequency_definition_confirmed": None,
    }
    if valid_path and valid_path.is_file():
        _valid_fields, valid_rows = _read_csv(valid_path)
        frequency_groups, frequency_counts = build_frequency_groups(valid_rows)
        frequency_summary.update(frequency_counts)
        top20_matches: list[bool] = []
        for candidate in selected:
            expected_frequency = frequency_groups.get(candidate.canonical_smiles, {}).get("raw_frequency")
            recorded_frequency = _candidate_metadata_number(
                candidate, ("frequency", "raw_frequency", "source_frequency")
            )
            if expected_frequency is not None and recorded_frequency is not None:
                top20_matches.append(int(recorded_frequency) == int(expected_frequency))
        frequency_summary["top20_frequency_match_count"] = sum(top20_matches)
        frequency_summary["top20_frequency_compared_count"] = len(top20_matches)
        frequency_summary["frequency_definition_confirmed"] = bool(top20_matches) and all(top20_matches)
        frequency_summary["num_source_paths"] = len(
            {
                _text(row.get("source_path"))
                for row in valid_rows
                if _text(row.get("source_path"))
            }
        )
        frequency_summary["top20_frequency"] = [
            {
                "rank": candidate.rank,
                "candidate_id": candidate.candidate_id,
                "canonical_smiles": candidate.canonical_smiles,
                "recorded_frequency": _candidate_metadata_number(
                    candidate, ("frequency", "raw_frequency", "source_frequency")
                ),
                "computed_frequency": frequency_groups.get(
                    candidate.canonical_smiles, {}
                ).get("raw_frequency"),
                "graph_support": _candidate_metadata_number(
                    candidate, ("graph_support", "support_count", "covered_count")
                ),
                "source_raw_index_count": frequency_groups.get(
                    candidate.canonical_smiles, {}
                ).get("source_raw_index_count"),
                "source_path_count": frequency_groups.get(
                    candidate.canonical_smiles, {}
                ).get("source_path_count"),
            }
            for candidate in selected
        ]
    else:
        warnings.append("valid_candidates.csv_not_found_frequency_definition_not_fully_verifiable")
    order_rows, order_summary = audit_candidate_order(
        selected, pair_audit, frequency_groups=frequency_groups
    )
    configured_candidate_raw = _text(
        config.get("fullgraph_candidates_path")
        or config.get("gt_fullgraph_candidates_path")
        or config.get("gcf_candidates_path")
    )
    configured_candidate_path = (
        _resolve(configured_candidate_raw) if configured_candidate_raw else None
    )
    selected_frequencies = [
        _candidate_metadata_number(
            candidate, ("frequency", "raw_frequency", "source_frequency")
        )
        for candidate in selected
    ]
    frequency_non_increasing = (
        all(left >= right for left, right in zip(selected_frequencies, selected_frequencies[1:]))
        if all(value is not None for value in selected_frequencies)
        else None
    )
    order_summary.update(
        {
            "rank_source": rank_source,
            "run_config_candidate_path": (
                str(configured_candidate_path) if configured_candidate_path else None
            ),
            "selected_path_matches_run_config": (
                configured_candidate_path == selected_path
                if configured_candidate_path is not None
                else None
            ),
            "frequency_non_increasing": frequency_non_increasing,
        }
    )
    if configured_candidate_path is None:
        warnings.append("run_config_does_not_record_fullgraph_candidates_path")
    prefix_rows = build_prefix_marginal_rows(
        pair_audit, selected, theta=float(args.theta)
    )
    consistency = build_consistency_audit(
        pair_audit,
        selected,
        theta=float(args.theta),
        table_k=int(args.table_k),
        max_k=int(args.max_k),
        combined_rows=combined_rows,
        report_dir=_resolve(args.report_dir),
        tolerance=float(args.tolerance),
    )
    metric_definitions = build_metric_definition_audit(
        pair_audit,
        selected,
        theta=float(args.theta),
        k=int(args.table_k),
    )
    applicable_rows = build_applicable_rows(
        pair_audit,
        selected,
        current_run_dir=run_dir,
        comparison_specs=comparison_specs,
        expected_top_k=int(args.max_k),
        expected_parents=len(pair_audit.parents),
        warnings=warnings,
    )
    parent_cohort_rows = build_parent_cohort_audit(
        pair_audit=pair_audit,
        current_run_dir=run_dir,
        reference_parent_ids=reference_parent_ids,
        comparison_specs=comparison_specs,
        warnings=warnings,
    )
    fullgraph_semantics = {
        "num_rows": len(pair_audit.rows),
        "match_false_rows": sum(not _as_bool(row.get("match")) for row in pair_audit.rows),
        "delete_valid_false_rows": sum(not _as_bool(row.get("delete_valid")) for row in pair_audit.rows),
        "nonempty_fragment_smiles_rows": sum(bool(_text(row.get("fragment_smiles"))) for row in pair_audit.rows),
        "residual_differs_candidate_rows": sum(
            _canonicalize(_text(row.get("residual_smiles")))
            != _canonicalize(_text(row.get("candidate_smiles")))
            for row in pair_audit.rows
        ),
        "path_definition": "distance(parent_smiles, complete candidate_smiles)",
        "uses_subgraph_deletion": False,
    }
    implementation_findings: list[str] = []
    if pair_audit.mismatch_rows:
        implementation_findings.append("strict_flip_mismatch_in_saved_pair_details")
    if not order_summary["rank_order_preserved"] or order_summary["raw_14032_pool_misused"]:
        implementation_findings.append("candidate_order_or_pool_mismatch")
    if order_summary["selected_path_matches_run_config"] is False:
        implementation_findings.append("audit_selected_path_differs_from_run_config_candidate_path")
    if not consistency["prefix_coverage_monotone"] or not consistency["threshold_coverage_monotone"]:
        implementation_findings.append("coverage_monotonicity_violation")
    for field in (
        "table2_k10_matches",
        "figure3_k10_matches",
        "figure3_k20_matches",
        "combined_summary_k20_matches",
    ):
        if consistency.get(field) is False:
            implementation_findings.append(f"metric_output_mismatch:{field}")
    if not metric_definitions["covered_parent_median_le_theta"]:
        implementation_findings.append("covered_parent_median_exceeds_theta")
    if consistency["table_uses_legacy_conditional_median_name"]:
        implementation_findings.append("legacy_conditional_median_cost_label_is_ambiguous")
    if consistency["table_uses_legacy_applicable_rate_name"]:
        implementation_findings.append("legacy_applicable_rate_label_is_ambiguous")
    if fullgraph_semantics["match_false_rows"] or fullgraph_semantics["delete_valid_false_rows"]:
        implementation_findings.append("fullgraph_rows_unexpectedly_marked_inapplicable")
    adaptation_findings: list[str] = []
    zero_marginal = sum(row["marginal_newly_covered_parent_count"] == 0 for row in prefix_rows)
    if zero_marginal:
        adaptation_findings.append(f"frequency_top20_has_{zero_marginal}_zero_marginal_candidates")
    max_similarity = max(
        (float(row["max_morgan_tanimoto_to_previous"]) for row in prefix_rows),
        default=0.0,
    )
    if max_similarity >= 0.8:
        adaptation_findings.append("frequency_top20_contains_high_morgan_redundancy")
    adaptation_findings.append("frequency_ranking_does_not_optimize_project_strict_close_coverage")
    adaptation_findings.append("fullgraph_molecule_adaptation_does_not_reproduce_globalgce_rule_application")
    result_class = "C" if implementation_findings and adaptation_findings else ("A" if implementation_findings else "B")
    conclusion = (
        "The audit separates saved-artifact correctness from adaptation quality. "
        "A plateau or jump is accepted as data-driven only when strict flip, candidate order, and metric consistency checks pass. "
        "The historical conditional-median label is ambiguous because it conditions on strict-recourse applicability, not theta coverage."
    )
    table_metrics = _prefix_metrics(
        pair_audit,
        selected,
        k=int(args.table_k),
        theta=float(args.theta),
    )
    theta_covered_median = table_metrics["covered_parent_median_cost"]
    assert theta_covered_median is None or theta_covered_median <= float(args.theta) + 1e-12
    corrected_table_rows = [
        {
            "Method": "GlobalGCE",
            "K": int(args.table_k),
            "Theta": float(args.theta),
            "Coverage": table_metrics["coverage"],
            "Num covered": table_metrics["num_covered"],
            "Theta-covered conditional median cost": theta_covered_median,
        }
    ]
    summary = {
        "run_dir": str(run_dir),
        "pair_details": str(pair_details),
        "selected_top20": str(selected_path),
        "rank_source": rank_source,
        "method": method,
        "theta": float(args.theta),
        "table_k": int(args.table_k),
        "max_k": int(args.max_k),
        "distance_recomputed": False,
        "cohort_mode": (
            "reference_parent_ids" if reference_parent_ids is not None else "all_label_parent_diagnostic"
        ),
        "reference_cohort": reference_metadata,
        "code_provenance": {
            "node_fgw_evaluator": "scripts/evaluate_ccrcov_with_molclr_node_fgw.py",
            "fullgraph_pair_evaluator": "src/eval/ccrcov_distance_eval.py::_evaluate_gt_fullgraph",
            "paper_table_and_prefix_threshold_plots": "src/eval/gcf_style_recourse_report.py",
            "paper_report_entrypoint": "scripts/generate_gcf_style_recourse_report.py",
            "tracked_globalgce_frequency_selector_implementation": None,
            "frequency_selector_note": (
                "No tracked GlobalGCE canonical-frequency Top20 generator was found; "
                "ranking provenance is audited from frequency_ranked_candidates.csv, "
                "selected_top20.csv, selected_top20_for_eval.csv, and valid_candidates.csv."
            ),
        },
        "strict_flip_audit": {
            "num_pair_rows": len(pair_audit.rows),
            "num_all_pair_rows_in_file": len(all_pair_rows),
            "method_filter": method,
            "target_label": int(args.target_label),
            "rows_before_method_filter": pair_audit.rows_before_method_filter,
            "rows_after_method_filter": pair_audit.rows_after_method_filter,
            "rows_after_reference_filter": pair_audit.rows_after_reference_filter,
            "num_parents": len(pair_audit.parents),
            "num_teacher_target_parents": len(pair_audit.teacher_target_parents),
            "teacher_target_parent_rate": (
                len(pair_audit.teacher_target_parents) / len(pair_audit.parents)
                if pair_audit.parents
                else 0.0
            ),
            "recorded_cf_flip_pairs": pair_audit.recorded_flip_pairs,
            "expected_strict_flip_pairs": pair_audit.expected_flip_pairs,
            "mismatch_rows": len(pair_audit.mismatch_rows),
            "mismatch_rows_by_method": pair_audit.mismatch_by_method,
            "confusion": pair_audit.confusion,
            "invalid_pred_before_rows": pair_audit.invalid_pred_before_rows,
            "invalid_pred_after_rows": pair_audit.invalid_pred_after_rows,
            "duplicate_parent_candidate_rows": pair_audit.duplicate_parent_candidate_rows,
            "definition": "pred_before == target_label and pred_after != target_label",
        },
        "parent_cohort_audit": parent_cohort_rows,
        "candidate_order_audit": order_summary,
        "frequency_audit": frequency_summary,
        "metric_consistency_audit": consistency,
        "metric_definition_audit": metric_definitions,
        "applicable_rate_audit": applicable_rows,
        "prefix_marginal_audit": prefix_rows,
        "corrected_table2_metrics": corrected_table_rows,
        "fullgraph_semantics_audit": fullgraph_semantics,
        "implementation_findings": implementation_findings,
        "adaptation_findings": adaptation_findings,
        "result_class": result_class,
        "conclusion": conclusion,
        "warnings": warnings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if reference_parent_ids is not None:
        _write_csv(
            output_dir / "reference_parent_ids.csv",
            ({"parent_id": parent_id} for parent_id in reference_parent_ids),
            ["parent_id"],
        )
    else:
        _write_csv(output_dir / "reference_parent_ids.csv", [], ["parent_id"])
    _write_csv(
        output_dir / "reference_parent_crosswalk.csv",
        reference_crosswalk_rows,
        [
            "parent_id",
            "source_ours_parent_id",
            "parent_smiles",
            "canonical_smiles",
            "match_type",
        ],
    )
    _write_csv(
        output_dir / "globalgce_prefix_marginal_coverage.csv",
        prefix_rows,
        list(prefix_rows[0]) if prefix_rows else [],
    )
    _write_csv(
        output_dir / "candidate_order_audit.csv",
        order_rows,
        list(order_rows[0]) if order_rows else [],
    )
    mismatch_fields = [
        "row_index",
        "method",
        "parent_id",
        "candidate_id",
        "label",
        "pred_before",
        "pred_after",
        "recorded_cf_flip",
        "recorded_teacher_strict_flip",
        "expected_teacher_strict_flip",
        "old_weak_flip",
    ]
    _write_csv(output_dir / "strict_flip_mismatches.csv", pair_audit.mismatch_rows, mismatch_fields)
    strict_confusion_totals = normalize_strict_flip_confusion_payload(
        {
            **pair_audit.confusion,
            "total_pair_rows": len(pair_audit.rows),
            "recorded_true_pairs": pair_audit.recorded_flip_pairs,
            "expected_strict_pairs": pair_audit.expected_flip_pairs,
            "mismatch_rows": len(pair_audit.mismatch_rows),
        }
    )
    assert strict_confusion_totals["consistency_status"] == "PASS"
    strict_confusion_payload = {
        "method_filter": method,
        "target_label": int(args.target_label),
        "rows_before_method_filter": pair_audit.rows_before_method_filter,
        "rows_after_method_filter": pair_audit.rows_after_method_filter,
        "rows_after_reference_filter": pair_audit.rows_after_reference_filter,
        "invalid_pred_before_rows": pair_audit.invalid_pred_before_rows,
        "invalid_pred_after_rows": pair_audit.invalid_pred_after_rows,
        "duplicate_parent_candidate_rows": pair_audit.duplicate_parent_candidate_rows,
        # Duplicate the normalized summary under ``confusion`` for readers that
        # historically descend into that object before looking up totals.
        "confusion": strict_confusion_totals,
        "arithmetic_definitions": {
            "total_pair_rows": "TT + TF + FT + FF",
            "recorded_true_pairs": "recorded_true_expected_true + recorded_true_expected_false",
            "expected_strict_pairs": "recorded_true_expected_true + recorded_false_expected_true",
            "mismatch_rows": "recorded_true_expected_false + recorded_false_expected_true",
        },
        **strict_confusion_totals,
    }
    _write_json(output_dir / "strict_flip_confusion.json", strict_confusion_payload)
    _write_csv(
        output_dir / "parent_cohort_audit.csv",
        parent_cohort_rows,
        list(parent_cohort_rows[0]) if parent_cohort_rows else [],
    )
    _write_csv(
        output_dir / "corrected_table2_metrics.csv",
        corrected_table_rows,
        list(corrected_table_rows[0]),
    )
    _write_json(output_dir / "metric_definition_audit.json", metric_definitions)
    applicable_fields = list(applicable_rows[0]) if applicable_rows else []
    _write_csv(output_dir / "applicable_rate_audit.csv", applicable_rows, applicable_fields)
    _write_json(output_dir / "audit_summary.json", summary)
    (output_dir / "audit_report.txt").write_text(render_report(summary), encoding="utf-8")
    corrected_output_dir = (
        _resolve(args.corrected_output_dir)
        if args.corrected_output_dir
        else run_dir / "corrected_teacher_strict"
    )
    corrected_thresholds = [
        float(row["threshold"])
        for row in combined_rows
        if _as_float(row.get("threshold")) is not None
    ]
    corrected_thresholds.append(float(args.theta))
    corrected_outputs = write_corrected_teacher_strict_outputs(
        output_dir=corrected_output_dir,
        source_fields=_pair_fields,
        pair_audit=pair_audit,
        selected=selected,
        source_config=config,
        source_pair_details=pair_details,
        target_label=int(args.target_label),
        thresholds=corrected_thresholds,
        reference_metadata=reference_metadata,
    )
    summary["corrected_teacher_strict_outputs"] = corrected_outputs
    _write_json(output_dir / "audit_summary.json", summary)
    if args.fail_on_critical and implementation_findings:
        raise RuntimeError(f"Critical audit findings: {implementation_findings}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--pair-details", default=None)
    parser.add_argument("--selected-top20", default=DEFAULT_SELECTED)
    parser.add_argument("--selected-top20-metadata", default=None)
    parser.add_argument("--frequency-ranked-candidates", default=None)
    parser.add_argument("--valid-candidates", default=None)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--corrected-output-dir",
        default=None,
        help="Defaults to RUN_DIR/corrected_teacher_strict.",
    )
    parser.add_argument(
        "--reference-parent-ids",
        default=None,
        help=(
            "CSV containing exact current-namespace parent IDs, or the explicit "
            "sentinel auto_from_final_ours_reference_run."
        ),
    )
    parser.add_argument("--reference-parent-id-col", default="parent_id")
    parser.add_argument(
        "--auto-reference-from-ours",
        action="store_true",
        help="Explicitly request an Ours-run cohort and SMILES-based ID crosswalk.",
    )
    parser.add_argument(
        "--reference-ours-run",
        default=None,
        help="Explicit Ours run used for reference cohort construction and crosswalk.",
    )
    parser.add_argument("--expected-reference-parents", type=int, default=1283)
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    parser.add_argument("--table-k", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--comparison-run",
        action="append",
        default=[],
        metavar="METHOD=PATH",
    )
    parser.add_argument("--fail-on-critical", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print("[GLOBALGCE_FREQUENCY_FGW_AUDIT]", flush=True)
    print(f"run_dir={args.run_dir}", flush=True)
    print(f"selected_top20={args.selected_top20}", flush=True)
    print(f"theta={args.theta}", flush=True)
    print("distance_recomputed=false", flush=True)
    summary = run_audit(args)
    print(f"result_class={summary['result_class']}", flush=True)
    print(f"strict_flip_mismatches={summary['strict_flip_audit']['mismatch_rows']}", flush=True)
    print(f"output_dir={_resolve(args.output_dir)}", flush=True)
    return 0
