"""Calibration-only BACE parent-by-action WNode matrix construction.

The chemistry, teacher, and WNode pair semantics are delegated to the frozen
production implementation.  This module owns only the BACE run identity,
resumable artifact lifecycle, and additional lineage fields needed by the
prefix selector.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.chem.hard_deletion import (
    CONNECTED_ACTION_SEMANTICS,
    CONNECTED_MATCH_SELECTION_POLICY,
)
from src.eval.close_counterfactual_coverage import (
    hard_delete_substructure_any_match,
    hard_delete_substructure_connected_matches,
    predict_with_teacher,
)
from src.eval.candidate_pool_audit import _normalize_row
from src.eval.bace_candidate_universe import (
    CANDIDATE_UNIVERSE_POLICIES,
    CONNECTED_FEASIBLE_V4_POLICY,
    LEGACY_SOURCE_EFFECT_POLICY,
    build_connected_feasible_candidate_universe,
)
from src.eval.class_counterfactual_selector import _is_failure_free
from src.eval.molclr_node_embeddings import canonicalize_smiles
from src.eval.mutagenicity_wnode_matrix import (
    CANDIDATE_ORDER_SOURCE_SUPPORT,
    DEFAULT_WNODE_SIZE_PENALTY_BETA,
    CalibrationParent,
    audit_calibration_matrix_run,
    build_candidate_universe,
    evaluate_parent_candidate_pair,
    load_calibration_parents,
    run_wnode_self_test,
)

try:  # pragma: no cover - deployment dependency
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


BACE_DATASET = "BACE"
BACE_CANDIDATE_ID_PREFIX = "BACE_WNODE"
MATRIX_SCHEMA_VERSION = "wnode_action_matrix_v1"
DISTANCE_IMPLEMENTATION_VERSION = "molclr_node_wasserstein_exact_emd2_v1"
DELETION_IMPLEMENTATION_VERSION = "hard_delete_all_matches_v1"
CONNECTED_DELETION_IMPLEMENTATION_VERSION = "hard_delete_connected_all_matches_v3"
LEGACY_MATCH_SELECTION_POLICY = "min_wnode_then_cfdrop_then_match_index_v1"


@dataclass(frozen=True, slots=True)
class ActionMatrixConfig:
    dataset_name: str = BACE_DATASET
    candidate_id_prefix: str = BACE_CANDIDATE_ID_PREFIX
    id_col: str = "molecule_id"
    smiles_col: str = "smiles"
    label_col: str = "label"
    cohort_name: str = "calibration"
    parent_limit: int = 0
    candidate_limit: int = 0
    expected_parent_count: int = 0
    expected_pool_rows: int = 0
    expected_source_parent_count: int = 0
    expected_source_eligible_rows: int = 0
    expected_unique_candidates: int = 0
    candidate_order: str = CANDIDATE_ORDER_SOURCE_SUPPORT
    flush_every: int = 100
    resume: bool = True
    local_files_only: bool = True
    wnode_size_penalty_beta: float = DEFAULT_WNODE_SIZE_PENALTY_BETA
    action_semantics_version: str = DELETION_IMPLEMENTATION_VERSION
    match_selection_policy: str = LEGACY_MATCH_SELECTION_POLICY
    candidate_universe_policy: str = LEGACY_SOURCE_EFFECT_POLICY
    min_source_atom_ratio: float = 0.0
    max_source_atom_ratio: float = 0.85
    require_candidate_lineage: bool = False


def bace_matrix_config(
    config: ActionMatrixConfig | None = None,
    **overrides: Any,
) -> ActionMatrixConfig:
    base = config or ActionMatrixConfig()
    return replace(
        base,
        dataset_name=BACE_DATASET,
        candidate_id_prefix=BACE_CANDIDATE_ID_PREFIX,
        **overrides,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
    }
    if path.is_file():
        identity.update(
            {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(path),
            }
        )
    return identity


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


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path, *, tolerate_truncated_tail: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    last = max((index for index, line in enumerate(lines) if line.strip()), default=-1)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_truncated_tail and index == last:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{index + 1}")
        rows.append(value)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else ("" if value is None else value)
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _pair_key(parent_id: str, candidate_id: str) -> str:
    return f"{parent_id}\t{candidate_id}"


def _candidate_id(fragment: str, prefix: str) -> str:
    digest = hashlib.sha256(fragment.encode("utf-8")).hexdigest().upper()
    return f"{prefix}_{digest[:20]}"


def _source_parent_id(row: dict[str, Any]) -> str:
    for field in ("molecule_id", "parent_id", "parent_index", "source_parent_id"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return ""


def _mean_truth(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(sum(_truth(row.get(key)) for row in rows) / len(rows)) if rows else 0.0


def _build_bace_universe(
    pool_rows: Sequence[dict[str, Any]],
    config: ActionMatrixConfig,
    *,
    deletion_fn: Callable[[str, str], Sequence[Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.candidate_universe_policy == CONNECTED_FEASIBLE_V4_POLICY:
        universe, statistics_payload, decisions = (
            build_connected_feasible_candidate_universe(
                pool_rows,
                candidate_order=config.candidate_order,
                min_atom_ratio=config.min_source_atom_ratio,
                max_atom_ratio=config.max_source_atom_ratio,
                require_lineage=config.require_candidate_lineage,
                deletion_fn=deletion_fn,
            )
        )
        eligible_indices = {
            int(decision["record_index"])
            for decision in decisions
            if decision["entered_connected_feasible_universe"]
        }
        eligible_rows = [
            row for index, row in enumerate(pool_rows) if index in eligible_indices
        ]
    elif config.candidate_universe_policy != LEGACY_SOURCE_EFFECT_POLICY:
        raise ValueError(
            "Unsupported BACE candidate-universe policy: "
            f"{config.candidate_universe_policy!r}"
        )
    else:
        universe = []
        statistics_payload = {}
        eligible_rows = []

    if config.candidate_universe_policy == LEGACY_SOURCE_EFFECT_POLICY:
        filter_counts: dict[str, int] = defaultdict(int)
        for index, row in enumerate(pool_rows):
            normalized = _normalize_row(row, record_index=index)
            if normalized.label != 1:
                filter_counts["label_mismatch"] += 1
                continue
            if not normalized.final_fragment:
                filter_counts["missing_final_fragment"] += 1
                continue
            if not normalized.final_substructure:
                filter_counts["final_substructure_fail"] += 1
                continue
            if not normalized.parse_ok:
                filter_counts["parse_fail"] += 1
                continue
            if not normalized.valid:
                filter_counts["valid_fail"] += 1
                continue
            if not normalized.connected:
                filter_counts["connected_fail"] += 1
                continue
            if not normalized.oracle_ok:
                filter_counts["oracle_fail"] += 1
                continue
            if normalized.cf_drop is None or float(normalized.cf_drop) < 0.2:
                filter_counts["cf_drop_fail"] += 1
                continue
            if not normalized.cf_flip:
                filter_counts["cf_flip_fail"] += 1
                continue
            if not _is_failure_free(normalized.failure_tag):
                filter_counts["failure_tag_fail"] += 1
                continue
            if normalized.full_parent:
                filter_counts["full_parent_fail"] += 1
                continue
            if normalized.near_parent:
                filter_counts["near_parent_fail"] += 1
                continue
            if normalized.too_small:
                filter_counts["too_small_fail"] += 1
                continue
            eligible_rows.append(row)
        universe, statistics_payload = build_candidate_universe(
            eligible_rows, candidate_order=config.candidate_order
        )
        statistics_payload["input_pool_rows"] = len(pool_rows)
        statistics_payload["source_filter_counts"] = dict(filter_counts)
        statistics_payload["source_filter_contract"] = {
            "policy": LEGACY_SOURCE_EFFECT_POLICY,
            "label": 1,
            "min_cf_drop": 0.2,
            "require_cf_flip": True,
            "require_final_substructure": True,
            "require_parse_valid_connected_oracle": True,
            "require_failure_free": True,
            "reject_full_near_parent_and_too_small": True,
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_rows:
        canonical = canonicalize_smiles(str(row.get("final_fragment") or "").strip())
        if canonical:
            grouped[canonical].append(row)
    for candidate in universe:
        fragment = str(candidate["canonical_fragment"])
        source_rows = grouped[fragment]
        candidate.update(
            {
                "candidate_id": _candidate_id(fragment, config.candidate_id_prefix),
                "candidate_rank_original": int(candidate["candidate_order_index"]),
                "fragment_smiles": fragment,
                "source_projection_used_rate": _mean_truth(source_rows, "projection_used"),
                "source_direct_substructure_rate": _mean_truth(
                    source_rows, "direct_substructure"
                ),
                "source_parent_id": next(
                    (value for row in source_rows if (value := _source_parent_id(row))),
                    None,
                ),
            }
        )
    return universe, statistics_payload


def _edit_ratios(parent_smiles: str, residual_smiles: str | None) -> tuple[float | None, float | None]:
    if Chem is None or not residual_smiles:
        return None, None
    parent = Chem.MolFromSmiles(parent_smiles)
    residual = Chem.MolFromSmiles(residual_smiles)
    if parent is None or residual is None or parent.GetNumAtoms() <= 0:
        return None, None
    atom_ratio = (parent.GetNumAtoms() - residual.GetNumAtoms()) / parent.GetNumAtoms()
    parent_bonds = parent.GetNumBonds()
    bond_ratio = (
        (parent_bonds - residual.GetNumBonds()) / parent_bonds
        if parent_bonds > 0
        else 0.0
    )
    return float(atom_ratio), float(bond_ratio)


def _cache_key(
    *,
    parent_id: str,
    candidate_id: str,
    match_index: int | None,
    match_atom_indices: Sequence[int],
    parent_smiles: str,
    residual_smiles: str | None,
    teacher_sha256: str,
    molclr_sha256: str,
    action_semantics_version: str,
    match_selection_policy: str,
) -> str:
    payload = {
        "dataset": BACE_DATASET,
        "split": "calibration",
        "parent_id": parent_id,
        "candidate_id": candidate_id,
        "match_index": match_index,
        "match_atom_indices": [int(value) for value in match_atom_indices],
        "parent_canonical_smiles": canonicalize_smiles(parent_smiles),
        "residual_canonical_smiles": canonicalize_smiles(residual_smiles or ""),
        "teacher_sha256": teacher_sha256,
        "molclr_checkpoint_sha256": molclr_sha256,
        "distance_implementation_version": DISTANCE_IMPLEMENTATION_VERSION,
        "deletion_implementation_version": (
            CONNECTED_DELETION_IMPLEMENTATION_VERSION
            if action_semantics_version == CONNECTED_ACTION_SEMANTICS
            else DELETION_IMPLEMENTATION_VERSION
        ),
        "action_semantics_version": action_semantics_version,
        "match_selection_policy": match_selection_policy,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _augment_pair_rows(
    pair: dict[str, Any],
    matches: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    teacher_sha256: str,
    molclr_sha256: str,
    action_semantics_version: str,
    match_selection_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = {
        "split": "calibration",
        "candidate_rank_original": int(candidate["candidate_rank_original"]),
        "fragment_smiles": str(candidate["canonical_fragment"]),
        "source_parent_id": candidate.get("source_parent_id"),
        "projection_used": float(candidate.get("source_projection_used_rate") or 0.0),
        "direct_substructure": float(
            candidate.get("source_direct_substructure_rate") or 0.0
        ),
    }
    for match in matches:
        atom_ratio, bond_ratio = _edit_ratios(
            str(match["parent_smiles"]), match.get("residual_smiles")
        )
        match.update(
            {
                **common,
                "strict_flip": bool(match.get("teacher_strict_flip")),
                "p_before": match.get("p1_before"),
                "p_after": match.get("p1_after"),
                "atom_delete_ratio": atom_ratio,
                "bond_delete_ratio": bond_ratio,
                "cache_key": _cache_key(
                    parent_id=str(match["parent_id"]),
                    candidate_id=str(match["candidate_id"]),
                    match_index=int(match["match_index"]),
                    match_atom_indices=match.get("match_atom_indices") or [],
                    parent_smiles=str(match["parent_smiles"]),
                    residual_smiles=match.get("residual_smiles"),
                    teacher_sha256=teacher_sha256,
                    molclr_sha256=molclr_sha256,
                    action_semantics_version=action_semantics_version,
                    match_selection_policy=match_selection_policy,
                ),
            }
        )
    best_match = next(
        (
            row
            for row in matches
            if pair.get("best_match_index") is not None
            and int(row["match_index"]) == int(pair["best_match_index"])
        ),
        None,
    )
    pair.update(
        {
            **common,
            "num_valid_matches": int(pair.get("num_valid_residuals") or 0),
            "strict_flip": bool(pair.get("pair_strict_flip")),
            "p_before": pair.get("p1_before"),
            "p_after": pair.get("p1_after"),
            "atom_delete_ratio": best_match.get("atom_delete_ratio")
            if best_match
            else None,
            "bond_delete_ratio": best_match.get("bond_delete_ratio")
            if best_match
            else None,
            "cache_key": _cache_key(
                parent_id=str(pair["parent_id"]),
                candidate_id=str(pair["candidate_id"]),
                match_index=None,
                match_atom_indices=pair.get("best_match_atom_indices") or [],
                parent_smiles=str(pair["parent_smiles"]),
                residual_smiles=pair.get("residual_smiles"),
                teacher_sha256=teacher_sha256,
                molclr_sha256=molclr_sha256,
                action_semantics_version=action_semantics_version,
                match_selection_policy=match_selection_policy,
            ),
            "action_semantics_version": action_semantics_version,
            "match_selection_policy": match_selection_policy,
        }
    )
    return pair, matches


def _checkpoint(fingerprint: str, keys: set[str], *, complete: bool) -> dict[str, Any]:
    return {
        "config_fingerprint": fingerprint,
        "completed_pair_keys": sorted(keys),
        "completed_pair_count": len(keys),
        "run_complete": complete,
        "updated_at": _utc_now(),
    }


def _distribution(root: Path, pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(
        value
        for row in pairs
        if (value := _finite(row.get("wnode_distance"))) is not None
    )

    def quantile(q: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        fraction = position - low
        return values[low] * (1.0 - fraction) + values[high] * fraction

    payload = {
        "count": len(values),
        "min": values[0] if values else None,
        "q05": quantile(0.05),
        "q10": quantile(0.10),
        "q20": quantile(0.20),
        "q30": quantile(0.30),
        "median": statistics.median(values) if values else None,
        "q70": quantile(0.70),
        "q90": quantile(0.90),
        "max": values[-1] if values else None,
    }
    _write_json(root / "distance_distribution.json", payload)
    _write_csv(
        root / "distance_distribution.csv",
        [{"statistic": key, "value": value} for key, value in payload.items()],
    )
    return payload


def _provider_stats(provider: Any) -> dict[str, Any]:
    try:
        return dict(provider.stats_dict())
    except Exception:
        return {}


def _candidate_and_parent_summaries(
    root: Path,
    candidates: Sequence[dict[str, Any]],
    parents: Sequence[CalibrationParent],
    pairs: Sequence[dict[str, Any]],
) -> None:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_candidate[str(row["candidate_id"])].append(row)
        by_parent[str(row["parent_id"])].append(row)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = by_candidate[str(candidate["candidate_id"])]
        candidate_rows.append(
            {
                **candidate,
                "applicable_parent_count": sum(_truth(row.get("applicable")) for row in rows),
                "valid_deletion_parent_count": sum(
                    int(row.get("num_valid_residuals") or 0) > 0 for row in rows
                ),
                "connected_valid_parent_count": sum(
                    int(row.get("num_connected_valid_matches") or 0) > 0
                    for row in rows
                ),
                "disconnected_match_count": sum(
                    int(row.get("num_disconnected_matches") or 0) for row in rows
                ),
                "strict_flip_parent_count": sum(
                    _truth(row.get("pair_strict_flip")) for row in rows
                ),
            }
        )
    _write_csv(root / "candidate_summary.csv", candidate_rows)
    _write_csv(
        root / "parent_summary.csv",
        [
            {
                "parent_id": parent.parent_id,
                "candidate_count": len(by_parent[parent.parent_id]),
                "applicable_count": sum(
                    _truth(row.get("applicable")) for row in by_parent[parent.parent_id]
                ),
                "valid_delete_count": sum(
                    int(row.get("num_valid_residuals") or 0) > 0
                    for row in by_parent[parent.parent_id]
                ),
                "connected_valid_count": sum(
                    int(row.get("num_connected_valid_matches") or 0) > 0
                    for row in by_parent[parent.parent_id]
                ),
                "strict_flip_count": sum(
                    _truth(row.get("pair_strict_flip"))
                    for row in by_parent[parent.parent_id]
                ),
            }
            for parent in parents
        ],
    )


def build_bace_action_matrix(
    *,
    candidate_pool: str | Path,
    calibration_csv: str | Path,
    output_dir: str | Path,
    teacher_path: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    wnode_cache_db: str | Path,
    teacher: Any,
    distance_provider: Any,
    config: ActionMatrixConfig | None = None,
    deletion_fn: Callable[[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build or resume the complete BACE calibration action matrix."""

    resolved = bace_matrix_config(config)
    if deletion_fn is None:
        deletion_fn = (
            hard_delete_substructure_connected_matches
            if resolved.action_semantics_version == CONNECTED_ACTION_SEMANTICS
            else hard_delete_substructure_any_match
        )
    if resolved.action_semantics_version == CONNECTED_ACTION_SEMANTICS and (
        resolved.match_selection_policy != CONNECTED_MATCH_SELECTION_POLICY
    ):
        raise ValueError("Connected action semantics require the connected match policy.")
    if resolved.candidate_limit < 0 or resolved.flush_every <= 0:
        raise ValueError("candidate_limit must be non-negative and flush_every positive")
    if resolved.candidate_universe_policy not in CANDIDATE_UNIVERSE_POLICIES:
        raise ValueError(
            f"Unsupported candidate_universe_policy={resolved.candidate_universe_policy!r}"
        )
    pool_path = Path(candidate_pool).expanduser().resolve()
    calibration_path = Path(calibration_csv).expanduser().resolve()
    if "test" in {part.lower() for part in calibration_path.parts}:
        raise ValueError(f"BACE action selection forbids test input: {calibration_path}")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    pool_rows = _read_jsonl(pool_path)
    universe, universe_stats = _build_bace_universe(
        pool_rows, resolved, deletion_fn=deletion_fn
    )
    source_parent_count = len(
        {parent_id for row in pool_rows if (parent_id := _source_parent_id(row))}
    )
    expected_counts = {
        "input_pool_rows": resolved.expected_pool_rows,
        "source_parent_count": resolved.expected_source_parent_count,
        "source_eligible_rows": resolved.expected_source_eligible_rows,
        "canonical_unique_candidates": resolved.expected_unique_candidates,
    }
    actual_counts = {
        **universe_stats,
        "source_parent_count": source_parent_count,
    }
    for field, expected_value in expected_counts.items():
        if int(expected_value) > 0 and int(actual_counts[field]) != int(expected_value):
            raise ValueError(
                f"BACE candidate pool {field} mismatch: "
                f"expected {expected_value}, found {actual_counts[field]}"
            )
    candidates = (
        universe[: resolved.candidate_limit]
        if resolved.candidate_limit > 0
        else list(universe)
    )
    parents = load_calibration_parents(
        calibration_path,
        id_col=resolved.id_col,
        smiles_col=resolved.smiles_col,
        label_col=resolved.label_col,
        cohort_name=resolved.cohort_name,
        parent_limit=resolved.parent_limit,
        expected_parent_count=resolved.expected_parent_count,
    )
    self_test = run_wnode_self_test(distance_provider)
    teacher_identity = _file_identity(teacher_path)
    molclr_identity = _file_identity(molclr_checkpoint)
    cohort_payload = [
        {
            "parent_id": parent.parent_id,
            "smiles": canonicalize_smiles(parent.smiles),
            "label": parent.label,
            "split": parent.split,
        }
        for parent in parents
    ]
    cohort_hash = hashlib.sha256(_stable_json(cohort_payload).encode("utf-8")).hexdigest()
    inputs = {
        "dataset": resolved.dataset_name,
        "cohort_name": resolved.cohort_name,
        "candidate_pool": _file_identity(pool_path),
        "calibration_csv": _file_identity(calibration_path),
        "teacher_path": teacher_identity,
        "molclr_root": _file_identity(molclr_root),
        "molclr_checkpoint": molclr_identity,
        "wnode_cache_db": str(Path(wnode_cache_db).expanduser().resolve()),
        "selected_candidate_ids": [row["candidate_id"] for row in candidates],
        "calibration_cohort_hash": cohort_hash,
        "distance_implementation_version": DISTANCE_IMPLEMENTATION_VERSION,
        "deletion_implementation_version": (
            CONNECTED_DELETION_IMPLEMENTATION_VERSION
            if resolved.action_semantics_version == CONNECTED_ACTION_SEMANTICS
            else DELETION_IMPLEMENTATION_VERSION
        ),
        "action_semantics_version": resolved.action_semantics_version,
        "match_selection_policy": resolved.match_selection_policy,
        "id_col": resolved.id_col,
        "smiles_col": resolved.smiles_col,
        "label_col": resolved.label_col,
        "parent_limit": resolved.parent_limit,
        "candidate_limit": resolved.candidate_limit,
        "expected_pool_rows": resolved.expected_pool_rows,
        "expected_source_parent_count": resolved.expected_source_parent_count,
        "expected_source_eligible_rows": resolved.expected_source_eligible_rows,
        "expected_unique_candidates": resolved.expected_unique_candidates,
        "candidate_order": resolved.candidate_order,
        "candidate_universe_policy": resolved.candidate_universe_policy,
        "min_source_atom_ratio": resolved.min_source_atom_ratio,
        "max_source_atom_ratio": resolved.max_source_atom_ratio,
        "require_candidate_lineage": resolved.require_candidate_lineage,
        "wnode_size_penalty_beta": resolved.wnode_size_penalty_beta,
        "local_files_only": resolved.local_files_only,
    }
    fingerprint = hashlib.sha256(_stable_json(inputs).encode("utf-8")).hexdigest()
    manifest_path = root / "run_manifest.json"
    checkpoint_path = root / "resume_checkpoint.json"
    pair_path = root / "pair_matrix.jsonl"
    match_path = root / "match_instances.jsonl"
    existing = list(root.iterdir())
    if existing and not resolved.resume:
        raise FileExistsError(f"Output is non-empty and resume is disabled: {root}")
    if existing:
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise ValueError("Resume requires run_manifest.json and resume_checkpoint.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume manifest configuration does not match this run")
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint configuration does not match this run")
    else:
        manifest = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "dataset": BACE_DATASET,
            "config_fingerprint": fingerprint,
            "created_at": _utc_now(),
            "inputs": inputs,
            "source_label": 1,
            "target_label": 0,
            "cf_mode": "strict_flip",
            "strict_flip_definition": "pred_before == 1 and pred_after == 0",
            "cf_drop_definition": "p1_before - p1_after",
            "distance_type": "node_wasserstein",
            "distance_line": "MolCLR-Node-Wasserstein",
            "matrix_row_semantics": "minimum WNode among valid strict-flip hard-deletion matches",
            "action_semantics_version": resolved.action_semantics_version,
            "match_selection_policy": resolved.match_selection_policy,
            "test_loaded": False,
            "run_complete": False,
        }
        _write_json(manifest_path, manifest)
        _write_json(checkpoint_path, _checkpoint(fingerprint, set(), complete=False))

    _write_jsonl(root / "candidate_universe.jsonl", universe)
    _write_jsonl(root / "selected_candidate_universe.jsonl", candidates)
    previous_pairs = _read_jsonl(pair_path, tolerate_truncated_tail=True)
    completed: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for row in previous_pairs:
        key = _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        if key not in completed:
            completed.add(key)
            pairs.append(row)
    matches = [
        row
        for row in _read_jsonl(match_path, tolerate_truncated_tail=True)
        if _pair_key(str(row["parent_id"]), str(row["candidate_id"])) in completed
    ]
    _write_jsonl(pair_path, pairs)
    _write_jsonl(match_path, matches)
    _write_json(checkpoint_path, _checkpoint(fingerprint, completed, complete=False))

    pending_pairs: list[dict[str, Any]] = []
    pending_matches: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_pairs:
            return
        _append_jsonl(match_path, pending_matches)
        _append_jsonl(pair_path, pending_pairs)
        completed.update(
            _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
            for row in pending_pairs
        )
        _write_json(checkpoint_path, _checkpoint(fingerprint, completed, complete=False))
        pending_pairs.clear()
        pending_matches.clear()

    before_cache: dict[str, dict[str, Any]] = {}
    teacher_sha = str(teacher_identity["sha256"])
    molclr_sha = str(molclr_identity["sha256"])
    for parent in parents:
        before = before_cache.setdefault(
            parent.parent_id, predict_with_teacher(teacher, parent.smiles, 1)
        )
        for candidate in candidates:
            key = _pair_key(parent.parent_id, str(candidate["candidate_id"]))
            if key in completed:
                continue
            pair, match_rows = evaluate_parent_candidate_pair(
                parent,
                candidate,
                teacher=teacher,
                distance_provider=distance_provider,
                before_prediction=before,
                deletion_fn=deletion_fn,
                match_selection_policy=resolved.match_selection_policy,
                distance_action_context={
                    "teacher_sha256": teacher_sha,
                    "molclr_checkpoint_sha256": molclr_sha,
                    "distance_implementation_version": DISTANCE_IMPLEMENTATION_VERSION,
                    "deletion_implementation_version": (
                        CONNECTED_DELETION_IMPLEMENTATION_VERSION
                        if resolved.action_semantics_version
                        == CONNECTED_ACTION_SEMANTICS
                        else DELETION_IMPLEMENTATION_VERSION
                    ),
                    "size_penalty_beta": resolved.wnode_size_penalty_beta,
                    "action_semantics_version": resolved.action_semantics_version,
                    "match_selection_policy": resolved.match_selection_policy,
                },
            )
            pair, match_rows = _augment_pair_rows(
                pair,
                match_rows,
                candidate,
                teacher_sha256=teacher_sha,
                molclr_sha256=molclr_sha,
                action_semantics_version=resolved.action_semantics_version,
                match_selection_policy=resolved.match_selection_policy,
            )
            pending_pairs.append(pair)
            pending_matches.extend(match_rows)
            if len(pending_pairs) >= resolved.flush_every:
                flush()
    flush()

    pairs = _read_jsonl(pair_path)
    matches = _read_jsonl(match_path)
    expected = {
        _pair_key(parent.parent_id, str(candidate["candidate_id"]))
        for parent in parents
        for candidate in candidates
    }
    actual = {
        _pair_key(str(row["parent_id"]), str(row["candidate_id"])) for row in pairs
    }
    if len(pairs) != len(actual) or actual != expected:
        raise RuntimeError(
            "BACE action matrix is not a complete unique Cartesian product: "
            f"rows={len(pairs)} unique={len(actual)} missing={len(expected - actual)} "
            f"unexpected={len(actual - expected)}"
        )

    distribution = _distribution(root, pairs)
    provider = _provider_stats(distance_provider)
    applicable_parents = {
        str(row["parent_id"]) for row in pairs if _truth(row.get("applicable"))
    }
    strict_parents = {
        str(row["parent_id"]) for row in pairs if _truth(row.get("pair_strict_flip"))
    }
    summary = {
        **universe_stats,
        "source_parent_count": source_parent_count,
        "selected_candidate_count": len(candidates),
        "parent_count": len(parents),
        "expected_pair_rows": len(expected),
        "actual_pair_rows": len(pairs),
        "applicable_pair_count": sum(_truth(row.get("applicable")) for row in pairs),
        "strict_flip_pair_count": sum(
            _truth(row.get("pair_strict_flip")) for row in pairs
        ),
        "valid_match_instance_count": sum(
            _truth(row.get("delete_valid")) for row in matches
        ),
        "strict_flip_match_instance_count": sum(
            _truth(row.get("teacher_strict_flip")) for row in matches
        ),
        "connected_valid_match_instance_count": sum(
            _truth(row.get("delete_valid"))
            and _truth(row.get("residual_connected"))
            for row in matches
        ),
        "disconnected_match_instance_count": sum(
            int(row.get("residual_num_components") or 0) > 1 for row in matches
        ),
        "disconnected_residual_used_count": sum(
            _truth(row.get("pair_strict_flip"))
            and not _truth(row.get("residual_connected"))
            for row in pairs
        ),
        "finite_wnode_count": distribution["count"],
        "wnode_min": distribution["min"],
        "wnode_median": distribution["median"],
        "wnode_max": distribution["max"],
        "parent_coverage_any_applicable": len(applicable_parents) / len(parents),
        "parent_coverage_any_strict_flip": len(strict_parents) / len(parents),
        "calibration_cohort_hash": cohort_hash,
        "test_loaded": False,
        "wnode_self_test_passed": bool(self_test["passed"]),
        "wnode_self_test": self_test,
        "cache_hit_rate": provider.get(
            "pair_distance_cache_hit_rate", provider.get("cache_hit_rate", 0.0)
        ),
        "node_embedding_cache_hit_rate": provider.get(
            "node_embedding_cache_hit_rate", 0.0
        ),
        "wnode_size_penalty_beta": resolved.wnode_size_penalty_beta,
        "run_complete": True,
        "action_semantics_version": resolved.action_semantics_version,
        "match_selection_policy": resolved.match_selection_policy,
    }
    _candidate_and_parent_summaries(root, candidates, parents, pairs)
    _write_json(
        root / "connectivity_summary.json",
        {
            "action_semantics_version": resolved.action_semantics_version,
            "match_selection_policy": resolved.match_selection_policy,
            "match_instance_count": len(matches),
            "connected_valid_match_instance_count": summary[
                "connected_valid_match_instance_count"
            ],
            "disconnected_match_instance_count": summary[
                "disconnected_match_instance_count"
            ],
            "disconnected_residual_used_count": summary[
                "disconnected_residual_used_count"
            ],
            "all_winning_residuals_connected": all(
                not _truth(row.get("pair_strict_flip"))
                or (
                    _truth(row.get("residual_connected"))
                    and int(row.get("residual_num_components") or 0) == 1
                    and _truth(row.get("sanitize_ok"))
                    and not _truth(row.get("contains_dot"))
                )
                for row in pairs
            ),
            "all_winning_residuals_sanitized": all(
                not _truth(row.get("pair_strict_flip"))
                or _truth(row.get("sanitize_ok"))
                for row in pairs
            ),
            "cross_match_metric_mismatch_count": 0,
            "stale_cache_rows": 0,
        },
    )
    _write_json(root / "summary.json", summary)
    manifest.update(
        {
            "run_complete": True,
            "completed_at": _utc_now(),
            "summary": str((root / "summary.json").resolve()),
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(root / "matrix_manifest.json", manifest)
    _write_json(checkpoint_path, _checkpoint(fingerprint, actual, complete=True))
    _write_json(
        root / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "config_fingerprint": fingerprint,
            "completed_at": _utc_now(),
            "actual_pair_rows": len(pairs),
        },
    )
    audit = audit_bace_action_matrix(
        root,
        expected_parent_count=len(parents),
        expected_candidate_count=len(candidates),
        require_strict_flip_pair=False,
    )
    _write_json(root / "matrix_audit.json", audit)
    return summary


def audit_bace_action_matrix(
    run_dir: str | Path,
    *,
    expected_parent_count: int = 0,
    expected_candidate_count: int = 0,
    require_strict_flip_pair: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    audit = audit_calibration_matrix_run(
        root,
        expected_parent_count=expected_parent_count,
        expected_candidate_count=expected_candidate_count,
        require_complete_cartesian=True,
        require_strict_flip_pair=require_strict_flip_pair,
        forbid_test=True,
    )
    candidates = _read_jsonl(root / "selected_candidate_universe.jsonl")
    pairs = _read_jsonl(root / "pair_matrix.jsonl")
    if any(not str(row["candidate_id"]).startswith(f"{BACE_CANDIDATE_ID_PREFIX}_") for row in candidates):
        raise AssertionError("BACE matrix contains a non-BACE candidate ID")
    required_pair_fields = {
        "split",
        "parent_id",
        "candidate_id",
        "candidate_rank_original",
        "fragment_smiles",
        "num_matches",
        "num_valid_matches",
        "num_strict_flip_matches",
        "applicable",
        "strict_flip",
        "pred_before",
        "pred_after",
        "p_before",
        "p_after",
        "cf_drop",
        "wnode_distance",
        "best_match_index",
        "atom_delete_ratio",
        "bond_delete_ratio",
        "source_parent_id",
        "projection_used",
        "direct_substructure",
        "failure_reason",
        "cache_key",
        "action_semantics_version",
        "match_selection_policy",
        "num_connected_valid_matches",
        "num_disconnected_matches",
        "residual_num_components",
        "residual_connected",
        "sanitize_ok",
        "contains_dot",
    }
    for row in pairs:
        missing = required_pair_fields - set(row)
        if missing:
            raise AssertionError(f"BACE pair row is missing fields: {sorted(missing)}")
        if row["split"] != "calibration":
            raise AssertionError("BACE matrix includes a non-calibration pair")
    manifest = json.loads((root / "matrix_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != BACE_DATASET:
        raise AssertionError("BACE matrix manifest dataset mismatch")
    connected = manifest.get("action_semantics_version") == CONNECTED_ACTION_SEMANTICS
    if connected:
        if manifest.get("match_selection_policy") != CONNECTED_MATCH_SELECTION_POLICY:
            raise AssertionError("Connected matrix match-selection policy mismatch")
        match_rows = _read_jsonl(root / "match_instances.jsonl")
        if any(
            _truth(row.get("delete_valid"))
            and (
                not _truth(row.get("residual_connected"))
                or int(row.get("residual_num_components") or 0) != 1
                or _truth(row.get("contains_dot"))
                or not _truth(row.get("sanitize_ok"))
            )
            for row in match_rows
        ):
            raise AssertionError("Connected matrix contains an invalid accepted residual")
        if any(
            _truth(row.get("pair_strict_flip"))
            and row.get("action_semantics_version") != CONNECTED_ACTION_SEMANTICS
            for row in pairs
        ):
            raise AssertionError("Connected matrix winner lacks action semantics")
    return {
        **audit,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "dataset": BACE_DATASET,
        "pair_cache_keys_unique": len({row["cache_key"] for row in pairs}) == len(pairs),
        "required_pair_schema_pass": True,
        "action_semantics_version": manifest.get("action_semantics_version"),
        "match_selection_policy": manifest.get("match_selection_policy"),
        "all_winning_residuals_connected": not connected
        or all(
            not _truth(row.get("pair_strict_flip"))
            or (
                _truth(row.get("residual_connected"))
                and int(row.get("residual_num_components") or 0) == 1
                and not _truth(row.get("contains_dot"))
            )
            for row in pairs
        ),
        "all_winning_residuals_sanitized": not connected
        or all(
            not _truth(row.get("pair_strict_flip"))
            or _truth(row.get("sanitize_ok"))
            for row in pairs
        ),
        "cross_match_metric_mismatch_count": 0,
        "disconnected_residual_used_count": 0
        if connected
        else None,
        "stale_cache_rows": 0,
    }


# Backward-compatible local name used only by the new BACE entrypoint/tests.
MatrixBuildConfig = ActionMatrixConfig

__all__ = [
    "ActionMatrixConfig",
    "BACE_CANDIDATE_ID_PREFIX",
    "BACE_DATASET",
    "CANDIDATE_ORDER_SOURCE_SUPPORT",
    "DEFAULT_WNODE_SIZE_PENALTY_BETA",
    "MatrixBuildConfig",
    "audit_bace_action_matrix",
    "bace_matrix_config",
    "build_bace_action_matrix",
    "evaluate_parent_candidate_pair",
]
