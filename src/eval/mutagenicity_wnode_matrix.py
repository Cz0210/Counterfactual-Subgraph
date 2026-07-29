"""Mutagenicity calibration action matrix using the production WNode semantics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

from src.eval.close_counterfactual_coverage import (
    hard_delete_substructure_any_match,
    predict_with_teacher,
)
from src.eval.molclr_node_embeddings import canonicalize_smiles


SOURCE_LABEL = 1
TARGET_LABEL = 0
DEFAULT_WNODE_SIZE_PENALTY_BETA = 0.0
CANDIDATE_ORDER_SOURCE_SUPPORT = "source_support_desc"
REQUIRED_OUTPUT_FILES = (
    "candidate_universe.jsonl",
    "selected_candidate_universe.jsonl",
    "pair_matrix.jsonl",
    "match_instances.jsonl",
    "distance_distribution.csv",
    "distance_distribution.json",
    "summary.json",
    "run_manifest.json",
    "resume_checkpoint.json",
    "_RUN_COMPLETE.json",
)


class TeacherProtocol(Protocol):
    def score_smiles(self, smiles: str, label: int | None = None, **kwargs: Any) -> dict[str, Any]:
        ...


class DistanceProtocol(Protocol):
    def distance(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        ...

    def stats_dict(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class CalibrationParent:
    parent_id: str
    smiles: str
    label: int
    split: str


@dataclass(frozen=True, slots=True)
class MatrixBuildConfig:
    id_col: str = "molecule_id"
    smiles_col: str = "smiles"
    label_col: str = "label"
    cohort_name: str = "calibration"
    parent_limit: int = 0
    candidate_limit: int = 0
    expected_parent_count: int = 0
    candidate_order: str = CANDIDATE_ORDER_SOURCE_SUPPORT
    flush_every: int = 100
    resume: bool = True
    local_files_only: bool = True
    wnode_size_penalty_beta: float = DEFAULT_WNODE_SIZE_PENALTY_BETA


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(
    path: Path,
    *,
    allow_truncated_last_line: bool = False,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty_index = nonempty_indices[-1] if nonempty_indices else -1
    rows: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if allow_truncated_last_line and line_index == last_nonempty_index:
                break
            raise
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_index + 1}")
        rows.append(payload)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _maximum(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite_float(item)) is not None]
    return float(max(finite)) if finite else None


def _source_parent_id(row: dict[str, Any]) -> str:
    for field in ("molecule_id", "parent_id", "parent_index", "source_parent_id"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    parent_smiles = str(row.get("parent_smiles") or "").strip()
    if parent_smiles:
        return f"SMILES_{hashlib.sha256(parent_smiles.encode('utf-8')).hexdigest()[:16]}"
    return ""


def stable_candidate_id(canonical_fragment: str) -> str:
    digest = hashlib.sha256(str(canonical_fragment).encode("utf-8")).hexdigest()
    return f"MUT_WNODE_{digest[:20].upper()}"


def is_source_eligible(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("final_fragment") or "").strip()
        and _bool_value(row.get("parse_ok"))
        and _bool_value(row.get("final_substructure"))
        and _bool_value(row.get("oracle_ok"))
        and _bool_value(row.get("cf_flip"))
    )


def build_candidate_universe(
    pool_rows: Sequence[dict[str, Any]],
    *,
    candidate_order: str = CANDIDATE_ORDER_SOURCE_SUPPORT,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter source-eligible rows and aggregate exact candidates canonically."""

    if candidate_order != CANDIDATE_ORDER_SOURCE_SUPPORT:
        raise ValueError(
            f"Unsupported candidate_order={candidate_order!r}; "
            f"expected {CANDIDATE_ORDER_SOURCE_SUPPORT!r}."
        )
    eligible = [row for row in pool_rows if is_source_eligible(row)]
    raw_fragments = {
        str(row.get("final_fragment") or "").strip()
        for row in eligible
        if str(row.get("final_fragment") or "").strip()
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        raw_fragment = str(row.get("final_fragment") or "").strip()
        canonical = canonicalize_smiles(raw_fragment)
        if canonical is None:
            raise ValueError(
                "A source-eligible final_fragment could not be canonicalized: "
                f"{raw_fragment!r}"
            )
        grouped[canonical].append(row)

    candidates: list[dict[str, Any]] = []
    for canonical_fragment, rows in grouped.items():
        source_parent_ids = sorted(
            {
                parent_id
                for row in rows
                if (parent_id := _source_parent_id(row))
            }
        )
        candidates.append(
            {
                "candidate_id": stable_candidate_id(canonical_fragment),
                "canonical_fragment": canonical_fragment,
                "source_row_count": len(rows),
                "source_parent_count": len(source_parent_ids),
                "source_parent_ids": source_parent_ids,
                "source_cf_drop_mean": _mean(row.get("cf_drop") for row in rows),
                "source_cf_drop_max": _maximum(row.get("cf_drop") for row in rows),
                "source_reward_mean": _mean(
                    row.get("reward_total") for row in rows
                ),
                "source_reward_max": _maximum(
                    row.get("reward_total") for row in rows
                ),
                "source_atom_ratio_mean": _mean(
                    row.get("atom_ratio") for row in rows
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(row["source_parent_count"]),
            -float(row["source_cf_drop_mean"])
            if row["source_cf_drop_mean"] is not None
            else float("inf"),
            str(row["canonical_fragment"]),
        )
    )
    for index, row in enumerate(candidates, start=1):
        row["candidate_order_index"] = index
        row["candidate_order"] = candidate_order
    return candidates, {
        "input_pool_rows": len(pool_rows),
        "source_eligible_rows": len(eligible),
        "source_eligible_raw_unique_fragments": len(raw_fragments),
        "canonical_unique_candidates": len(candidates),
    }


def _path_has_test_token(path: Path) -> bool:
    name = path.name.lower()
    if name == "test" or name.startswith("test.") or "_test." in name:
        return True
    return any(part.lower() == "test" for part in path.parts)


def load_calibration_parents(
    calibration_csv: str | Path,
    *,
    id_col: str = "molecule_id",
    smiles_col: str = "smiles",
    label_col: str = "label",
    cohort_name: str = "calibration",
    parent_limit: int = 0,
    expected_parent_count: int = 0,
) -> list[CalibrationParent]:
    path = Path(calibration_csv).expanduser().resolve()
    if _path_has_test_token(path):
        raise ValueError(f"Test input path is forbidden for calibration: {path}")
    if str(cohort_name).strip().lower() != "calibration":
        raise ValueError("cohort_name must be exactly 'calibration'.")
    if int(parent_limit) < 0:
        raise ValueError("parent_limit must be non-negative.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in (id_col, smiles_col, label_col) if field not in fieldnames]
        if missing:
            raise ValueError(f"Calibration CSV is missing required columns: {missing}")
        rows = [dict(row) for row in reader]

    parents: list[CalibrationParent] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        parent_id = str(row.get(id_col) or "").strip()
        smiles = str(row.get(smiles_col) or "").strip()
        if not parent_id or not smiles:
            raise ValueError(f"Calibration row {row_index} has an empty id or SMILES.")
        if parent_id in seen_ids:
            raise ValueError(f"Duplicate calibration parent id: {parent_id}")
        seen_ids.add(parent_id)
        try:
            label = int(float(row.get(label_col)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid calibration label at row {row_index}") from exc
        if label != SOURCE_LABEL:
            raise ValueError(
                f"Calibration parent {parent_id} has label={label}; expected {SOURCE_LABEL}."
            )
        split = str(row.get("split") or cohort_name).strip().lower()
        if split != "calibration":
            raise ValueError(
                f"Calibration parent {parent_id} has forbidden split={split!r}."
            )
        if canonicalize_smiles(smiles) is None:
            raise ValueError(f"Invalid calibration parent SMILES: {parent_id}")
        parents.append(CalibrationParent(parent_id, smiles, label, split))

    parents.sort(key=lambda row: row.parent_id)
    if int(parent_limit) > 0:
        parents = parents[: int(parent_limit)]
    if int(expected_parent_count) > 0 and len(parents) != int(expected_parent_count):
        raise ValueError(
            f"Calibration parent count mismatch: expected {expected_parent_count}, "
            f"found {len(parents)}."
        )
    return parents


def calibration_cohort_hash(parents: Sequence[CalibrationParent]) -> str:
    payload = [
        {
            "parent_id": parent.parent_id,
            "canonical_smiles": canonicalize_smiles(parent.smiles),
            "label": parent.label,
            "split": parent.split,
        }
        for parent in parents
    ]
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def run_wnode_self_test(distance_provider: DistanceProtocol) -> dict[str, Any]:
    identical = distance_provider.distance("CCO", "CCO")
    forward = distance_provider.distance("CCO", "CCN")
    reverse = distance_provider.distance("CCN", "CCO")
    results = (identical, forward, reverse)
    values = [_finite_float(result.get("distance")) for result in results]
    passed = bool(
        all(result.get("ok") and value is not None for result, value in zip(results, values))
        and values[0] is not None
        and values[0] <= 1e-7
        and values[1] is not None
        and values[2] is not None
        and abs(values[1] - values[2]) <= 1e-9
    )
    payload = {
        "d_CCO_CCO": values[0],
        "d_CCO_CCN": values[1],
        "d_CCN_CCO": values[2],
        "identity_tolerance": 1e-7,
        "symmetry_tolerance": 1e-9,
        "finite": all(value is not None for value in values),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"MolCLR-Node-Wasserstein self-test failed: {payload}")
    return payload


def _pair_key(parent_id: str, candidate_id: str) -> str:
    return f"{parent_id}\t{candidate_id}"


def evaluate_parent_candidate_pair(
    parent: CalibrationParent,
    candidate: dict[str, Any],
    *,
    teacher: TeacherProtocol,
    distance_provider: DistanceProtocol,
    before_prediction: dict[str, Any] | None = None,
    deletion_fn: Callable[[str, str], list[dict[str, Any]]] = hard_delete_substructure_any_match,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate every hard-deletion match and aggregate one pair row."""

    candidate_id = str(candidate["candidate_id"])
    canonical_fragment = str(candidate["canonical_fragment"])
    before = before_prediction or predict_with_teacher(
        teacher,
        parent.smiles,
        SOURCE_LABEL,
    )
    deletions = deletion_fn(parent.smiles, canonical_fragment)
    match_rows: list[dict[str, Any]] = []
    strict_finite_rows: list[dict[str, Any]] = []

    for fallback_index, deletion in enumerate(deletions):
        match_index = int(deletion.get("match_index", fallback_index))
        residual_smiles = str(deletion.get("residual_smiles") or "").strip() or None
        delete_valid = bool(deletion.get("delete_valid") and residual_smiles)
        match_row: dict[str, Any] = {
            "parent_id": parent.parent_id,
            "parent_smiles": parent.smiles,
            "candidate_id": candidate_id,
            "canonical_fragment": canonical_fragment,
            "match_index": match_index,
            "match_atom_indices": list(deletion.get("match_atoms") or []),
            "delete_valid": delete_valid,
            "residual_smiles": residual_smiles,
            "pred_before": before.get("pred_label"),
            "pred_after": None,
            "p1_before": before.get("p_label"),
            "p1_after": None,
            "cf_drop": None,
            "teacher_strict_flip": False,
            "wnode_distance": None,
            "distance_ok": False,
            "failure_reason": deletion.get("error"),
        }
        if not delete_valid:
            match_row["failure_reason"] = (
                match_row["failure_reason"] or "invalid_residual"
            )
            match_rows.append(match_row)
            continue
        after = predict_with_teacher(teacher, residual_smiles or "", SOURCE_LABEL)
        pred_before = before.get("pred_label")
        pred_after = after.get("pred_label")
        p1_before = _finite_float(before.get("p_label"))
        p1_after = _finite_float(after.get("p_label"))
        strict_flip = bool(
            before.get("ok")
            and after.get("ok")
            and pred_before == SOURCE_LABEL
            and pred_after == TARGET_LABEL
        )
        cf_drop = (
            float(p1_before - p1_after)
            if p1_before is not None and p1_after is not None
            else None
        )
        match_row.update(
            {
                "pred_after": pred_after,
                "p1_before": p1_before,
                "p1_after": p1_after,
                "cf_drop": cf_drop,
                "teacher_strict_flip": strict_flip,
                "failure_reason": after.get("error") if not after.get("ok") else None,
            }
        )
        if strict_flip:
            distance_result = distance_provider.distance(
                parent.smiles,
                residual_smiles or "",
            )
            distance = _finite_float(distance_result.get("distance"))
            distance_ok = bool(
                distance_result.get("ok")
                and distance is not None
                and distance >= 0.0
            )
            match_row.update(
                {
                    "wnode_distance": distance if distance_ok else None,
                    "distance_ok": distance_ok,
                    "distance_cache_hit": bool(distance_result.get("cache_hit")),
                    "failure_reason": (
                        None
                        if distance_ok
                        else str(
                            distance_result.get("error")
                            or "wnode_distance_failed"
                        )
                    ),
                }
            )
            if distance_ok:
                strict_finite_rows.append(match_row)
        match_rows.append(match_row)

    strict_finite_rows.sort(
        key=lambda row: (
            float(row["wnode_distance"]),
            -float(row["cf_drop"] if row["cf_drop"] is not None else float("-inf")),
            int(row["match_index"]),
        )
    )
    best = strict_finite_rows[0] if strict_finite_rows else None
    num_valid_residuals = sum(bool(row["delete_valid"]) for row in match_rows)
    num_strict_flip_matches = sum(
        bool(row["teacher_strict_flip"]) for row in match_rows
    )
    if not deletions:
        failure_reason = "no_substructure_match_or_fragment_parse_failed"
    elif not before.get("ok"):
        failure_reason = str(before.get("error") or "parent_teacher_failed")
    elif num_valid_residuals == 0:
        failure_reason = "no_valid_residual"
    elif num_strict_flip_matches == 0:
        failure_reason = "no_teacher_strict_flip_match"
    elif best is None:
        failure_reason = "strict_flip_without_finite_wnode"
    else:
        failure_reason = None

    pair_row = {
        "parent_id": parent.parent_id,
        "parent_smiles": parent.smiles,
        "candidate_id": candidate_id,
        "canonical_fragment": canonical_fragment,
        "applicable": bool(deletions),
        "num_matches": len(deletions),
        "num_valid_residuals": num_valid_residuals,
        "num_strict_flip_matches": num_strict_flip_matches,
        "pair_strict_flip": best is not None,
        "best_match_index": best.get("match_index") if best else None,
        "best_match_atom_indices": best.get("match_atom_indices") if best else [],
        "residual_smiles": best.get("residual_smiles") if best else None,
        "pred_before": best.get("pred_before") if best else before.get("pred_label"),
        "pred_after": best.get("pred_after") if best else None,
        "p1_before": best.get("p1_before") if best else before.get("p_label"),
        "p1_after": best.get("p1_after") if best else None,
        "cf_drop": best.get("cf_drop") if best else None,
        "wnode_distance": best.get("wnode_distance") if best else None,
        "failure_reason": failure_reason,
    }
    return pair_row, match_rows


def build_cartesian_rows(
    parents: Sequence[CalibrationParent],
    candidates: Sequence[dict[str, Any]],
    *,
    teacher: TeacherProtocol,
    distance_provider: DistanceProtocol,
    deletion_fn: Callable[[str, str], list[dict[str, Any]]] = hard_delete_substructure_any_match,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for parent in parents:
        before = predict_with_teacher(teacher, parent.smiles, SOURCE_LABEL)
        for candidate in candidates:
            pair, matches = evaluate_parent_candidate_pair(
                parent,
                candidate,
                teacher=teacher,
                distance_provider=distance_provider,
                before_prediction=before,
                deletion_fn=deletion_fn,
            )
            pair_rows.append(pair)
            match_rows.extend(matches)
    return pair_rows, match_rows


def _file_identity(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "kind": "directory" if path.is_dir() else "file",
    }
    if path.is_file() and stat.st_size <= 32 * 1024 * 1024:
        payload["sha256"] = _sha256_file(path)
    return payload


def _config_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _write_distance_distribution(
    output_dir: Path,
    pair_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    values = sorted(
        value
        for row in pair_rows
        if (value := _finite_float(row.get("wnode_distance"))) is not None
    )

    def quantile(q: float) -> float | None:
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        fraction = position - lower
        return values[lower] * (1.0 - fraction) + values[upper] * fraction

    distribution = {
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
    _atomic_write_json(output_dir / "distance_distribution.json", distribution)
    csv_path = output_dir / "distance_distribution.csv"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=output_dir,
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["statistic", "value"])
        writer.writeheader()
        for key, value in distribution.items():
            writer.writerow({"statistic": key, "value": "" if value is None else value})
        temporary_name = handle.name
    os.replace(temporary_name, csv_path)
    return distribution


def _provider_stats(distance_provider: DistanceProtocol) -> dict[str, Any]:
    try:
        return dict(distance_provider.stats_dict())
    except Exception:
        return {}


def _checkpoint_payload(
    fingerprint: str,
    completed_pair_keys: set[str],
    *,
    run_complete: bool,
) -> dict[str, Any]:
    return {
        "config_fingerprint": fingerprint,
        "completed_pair_keys": sorted(completed_pair_keys),
        "completed_pair_count": len(completed_pair_keys),
        "run_complete": bool(run_complete),
        "updated_at": _utc_now(),
    }


def build_calibration_matrix_run(
    *,
    candidate_pool: str | Path,
    calibration_csv: str | Path,
    output_dir: str | Path,
    teacher_path: str | Path,
    molclr_root: str | Path,
    molclr_checkpoint: str | Path,
    wnode_cache_db: str | Path,
    teacher: TeacherProtocol,
    distance_provider: DistanceProtocol,
    config: MatrixBuildConfig | None = None,
    deletion_fn: Callable[[str, str], list[dict[str, Any]]] = hard_delete_substructure_any_match,
) -> dict[str, Any]:
    """Build or resume the complete calibration parent-candidate matrix."""

    resolved = config or MatrixBuildConfig()
    if resolved.candidate_limit < 0 or resolved.flush_every <= 0:
        raise ValueError("candidate_limit must be non-negative and flush_every positive.")
    pool_path = Path(candidate_pool).expanduser().resolve()
    calibration_path = Path(calibration_csv).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    pool_rows = _read_jsonl(pool_path)
    universe, universe_stats = build_candidate_universe(
        pool_rows,
        candidate_order=resolved.candidate_order,
    )
    selected = (
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
    cohort_hash = calibration_cohort_hash(parents)
    self_test = run_wnode_self_test(distance_provider)

    fingerprint_payload = {
        "candidate_pool": _file_identity(pool_path),
        "calibration_csv": _file_identity(calibration_path),
        "teacher_path": _file_identity(teacher_path),
        "molclr_root": _file_identity(molclr_root),
        "molclr_checkpoint": _file_identity(molclr_checkpoint),
        "wnode_cache_db": str(Path(wnode_cache_db).expanduser().resolve()),
        "wnode_size_penalty_beta": float(resolved.wnode_size_penalty_beta),
        "id_col": resolved.id_col,
        "smiles_col": resolved.smiles_col,
        "label_col": resolved.label_col,
        "cohort_name": resolved.cohort_name,
        "parent_limit": resolved.parent_limit,
        "candidate_limit": resolved.candidate_limit,
        "candidate_order": resolved.candidate_order,
        "selected_candidate_ids": [row["candidate_id"] for row in selected],
        "calibration_cohort_hash": cohort_hash,
        "local_files_only": bool(resolved.local_files_only),
    }
    fingerprint = _config_fingerprint(fingerprint_payload)
    manifest_path = destination / "run_manifest.json"
    checkpoint_path = destination / "resume_checkpoint.json"
    pair_path = destination / "pair_matrix.jsonl"
    match_path = destination / "match_instances.jsonl"
    existing_entries = [
        path for path in destination.iterdir() if path.name not in {".", ".."}
    ]
    if existing_entries and not resolved.resume:
        raise FileExistsError(
            f"Output directory is non-empty and resume is disabled: {destination}"
        )
    if existing_entries:
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise ValueError(
                "Resume requires run_manifest.json and resume_checkpoint.json."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume manifest configuration does not match this run.")
        if checkpoint.get("config_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoint configuration does not match this run.")
    else:
        manifest = {
            "config_fingerprint": fingerprint,
            "created_at": _utc_now(),
            "inputs": fingerprint_payload,
            "source_label": SOURCE_LABEL,
            "target_label": TARGET_LABEL,
            "strict_flip_definition": "pred_before == 1 and pred_after == 0",
            "cf_drop_definition": "p1_before - p1_after",
            "distance_type": "node_wasserstein",
            "distance_line": "MolCLR-Node-Wasserstein",
            "solver": "exact_emd2",
            "test_loaded": False,
            "run_complete": False,
        }
        _atomic_write_json(manifest_path, manifest)
        _atomic_write_json(
            checkpoint_path,
            _checkpoint_payload(fingerprint, set(), run_complete=False),
        )

    _atomic_write_jsonl(destination / "candidate_universe.jsonl", universe)
    _atomic_write_jsonl(
        destination / "selected_candidate_universe.jsonl",
        selected,
    )

    existing_pairs = _read_jsonl(
        pair_path,
        allow_truncated_last_line=True,
    )
    deduplicated_pairs: list[dict[str, Any]] = []
    completed_pair_keys: set[str] = set()
    for row in existing_pairs:
        key = _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        if key not in completed_pair_keys:
            deduplicated_pairs.append(row)
            completed_pair_keys.add(key)
    existing_matches = [
        row
        for row in _read_jsonl(
            match_path,
            allow_truncated_last_line=True,
        )
        if _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        in completed_pair_keys
    ]
    _atomic_write_jsonl(pair_path, deduplicated_pairs)
    _atomic_write_jsonl(match_path, existing_matches)
    _atomic_write_json(
        checkpoint_path,
        _checkpoint_payload(
            fingerprint,
            completed_pair_keys,
            run_complete=False,
        ),
    )

    pending_pairs: list[dict[str, Any]] = []
    pending_matches: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_pairs:
            return
        _append_jsonl(match_path, pending_matches)
        _append_jsonl(pair_path, pending_pairs)
        for row in pending_pairs:
            completed_pair_keys.add(
                _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
            )
        _atomic_write_json(
            checkpoint_path,
            _checkpoint_payload(
                fingerprint,
                completed_pair_keys,
                run_complete=False,
            ),
        )
        pending_pairs.clear()
        pending_matches.clear()

    before_cache: dict[str, dict[str, Any]] = {}
    for parent in parents:
        before = before_cache.setdefault(
            parent.parent_id,
            predict_with_teacher(teacher, parent.smiles, SOURCE_LABEL),
        )
        for candidate in selected:
            key = _pair_key(parent.parent_id, str(candidate["candidate_id"]))
            if key in completed_pair_keys:
                continue
            pair, matches = evaluate_parent_candidate_pair(
                parent,
                candidate,
                teacher=teacher,
                distance_provider=distance_provider,
                before_prediction=before,
                deletion_fn=deletion_fn,
            )
            pending_pairs.append(pair)
            pending_matches.extend(matches)
            if len(pending_pairs) >= int(resolved.flush_every):
                flush()
    flush()

    pair_rows = _read_jsonl(pair_path)
    match_rows = _read_jsonl(match_path)
    expected_keys = {
        _pair_key(parent.parent_id, str(candidate["candidate_id"]))
        for parent in parents
        for candidate in selected
    }
    actual_keys = {
        _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        for row in pair_rows
    }
    if len(pair_rows) != len(actual_keys):
        raise RuntimeError("pair_matrix.jsonl contains duplicate parent-candidate rows.")
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Cartesian matrix is incomplete: "
            f"missing={len(expected_keys - actual_keys)} "
            f"unexpected={len(actual_keys - expected_keys)}"
        )

    distribution = _write_distance_distribution(destination, pair_rows)
    provider_stats = _provider_stats(distance_provider)
    applicable_parents = {
        str(row["parent_id"]) for row in pair_rows if _bool_value(row.get("applicable"))
    }
    strict_parents = {
        str(row["parent_id"])
        for row in pair_rows
        if _bool_value(row.get("pair_strict_flip"))
    }
    parent_count = len(parents)
    summary = {
        **universe_stats,
        "selected_candidate_count": len(selected),
        "parent_count": parent_count,
        "expected_pair_rows": len(expected_keys),
        "actual_pair_rows": len(pair_rows),
        "applicable_pair_count": sum(
            _bool_value(row.get("applicable")) for row in pair_rows
        ),
        "strict_flip_pair_count": sum(
            _bool_value(row.get("pair_strict_flip")) for row in pair_rows
        ),
        "valid_match_instance_count": sum(
            _bool_value(row.get("delete_valid")) for row in match_rows
        ),
        "strict_flip_match_instance_count": sum(
            _bool_value(row.get("teacher_strict_flip")) for row in match_rows
        ),
        "finite_wnode_count": int(distribution["count"]),
        "wnode_min": distribution["min"],
        "wnode_median": distribution["median"],
        "wnode_max": distribution["max"],
        "parent_coverage_any_applicable": (
            len(applicable_parents) / parent_count if parent_count else 0.0
        ),
        "parent_coverage_any_strict_flip": (
            len(strict_parents) / parent_count if parent_count else 0.0
        ),
        "calibration_cohort_hash": cohort_hash,
        "test_loaded": False,
        "wnode_self_test_passed": bool(self_test["passed"]),
        "wnode_self_test": self_test,
        "cache_hit_rate": provider_stats.get(
            "pair_distance_cache_hit_rate",
            provider_stats.get("cache_hit_rate", 0.0),
        ),
        "node_embedding_cache_hit_rate": provider_stats.get(
            "node_embedding_cache_hit_rate",
            0.0,
        ),
        "wnode_size_penalty_beta": float(resolved.wnode_size_penalty_beta),
        "run_complete": True,
    }
    _atomic_write_json(destination / "summary.json", summary)
    manifest.update(
        {
            "run_complete": True,
            "completed_at": _utc_now(),
            "summary": str((destination / "summary.json").resolve()),
        }
    )
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(
        checkpoint_path,
        _checkpoint_payload(fingerprint, actual_keys, run_complete=True),
    )
    _atomic_write_json(
        destination / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "config_fingerprint": fingerprint,
            "completed_at": _utc_now(),
            "actual_pair_rows": len(pair_rows),
        },
    )
    return summary


def audit_calibration_matrix_run(
    run_dir: str | Path,
    *,
    expected_parent_count: int = 0,
    expected_candidate_count: int = 0,
    expected_pair_count: int = 0,
    expected_source_eligible_rows: int = 0,
    expected_source_eligible_raw_unique: int = 0,
    require_complete_cartesian: bool = False,
    require_strict_flip_pair: bool = False,
    forbid_test: bool = False,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    for filename in REQUIRED_OUTPUT_FILES:
        path = root / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Required run artifact is missing or empty: {path}")

    candidates = _read_jsonl(root / "selected_candidate_universe.jsonl")
    pairs = _read_jsonl(root / "pair_matrix.jsonl")
    matches = _read_jsonl(root / "match_instances.jsonl")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "_RUN_COMPLETE.json").read_text(encoding="utf-8"))

    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    pair_keys = [
        _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        for row in pairs
    ]
    parent_ids = sorted({str(row["parent_id"]) for row in pairs})
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AssertionError("Selected candidate universe contains duplicate IDs.")
    if len(pair_keys) != len(set(pair_keys)):
        raise AssertionError("pair_matrix contains duplicate parent-candidate rows.")
    if expected_parent_count and len(parent_ids) != expected_parent_count:
        raise AssertionError("Parent count does not match the audit expectation.")
    if expected_candidate_count and len(candidate_ids) != expected_candidate_count:
        raise AssertionError("Candidate count does not match the audit expectation.")
    if expected_pair_count and len(pairs) != expected_pair_count:
        raise AssertionError("Pair count does not match the audit expectation.")
    if (
        expected_source_eligible_rows
        and summary.get("source_eligible_rows") != expected_source_eligible_rows
    ):
        raise AssertionError("source_eligible_rows does not match.")
    if (
        expected_source_eligible_raw_unique
        and summary.get("source_eligible_raw_unique_fragments")
        != expected_source_eligible_raw_unique
    ):
        raise AssertionError("source_eligible_raw_unique_fragments does not match.")
    expected_cartesian = {
        _pair_key(parent_id, candidate_id)
        for parent_id in parent_ids
        for candidate_id in candidate_ids
    }
    if require_complete_cartesian and set(pair_keys) != expected_cartesian:
        raise AssertionError("Pair matrix is not the complete Cartesian product.")

    matches_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matches:
        key = _pair_key(str(row["parent_id"]), str(row["candidate_id"]))
        matches_by_pair[key].append(row)
        strict_expected = bool(
            row.get("pred_before") == SOURCE_LABEL
            and row.get("pred_after") == TARGET_LABEL
        )
        if _bool_value(row.get("teacher_strict_flip")) != strict_expected:
            raise AssertionError(f"Strict-flip mismatch in match instance: {key}")

    strict_pair_count = 0
    for pair in pairs:
        key = _pair_key(str(pair["parent_id"]), str(pair["candidate_id"]))
        grouped = matches_by_pair.get(key, [])
        if int(pair.get("num_matches") or 0) != len(grouped):
            raise AssertionError(f"num_matches does not match instances: {key}")
        strict = _bool_value(pair.get("pair_strict_flip"))
        distance = _finite_float(pair.get("wnode_distance"))
        if strict:
            strict_pair_count += 1
            if not _bool_value(pair.get("applicable")):
                raise AssertionError(f"Strict pair is not applicable: {key}")
            if distance is None or distance < 0.0:
                raise AssertionError(f"Strict pair lacks finite non-negative WNode: {key}")
            eligible_matches = [
                row
                for row in grouped
                if _bool_value(row.get("teacher_strict_flip"))
                and (value := _finite_float(row.get("wnode_distance"))) is not None
                and value >= 0.0
            ]
            if not eligible_matches:
                raise AssertionError(f"Strict pair lacks a strict finite match: {key}")
            eligible_matches.sort(
                key=lambda row: (
                    float(row["wnode_distance"]),
                    -float(row.get("cf_drop") or 0.0),
                    int(row["match_index"]),
                )
            )
            best = eligible_matches[0]
            if not math.isclose(
                float(pair["wnode_distance"]),
                float(best["wnode_distance"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"Pair WNode does not equal best strict match: {key}")
            for field in (
                "best_match_index",
                "best_match_atom_indices",
                "residual_smiles",
                "pred_before",
                "pred_after",
                "p1_before",
                "p1_after",
                "cf_drop",
            ):
                source_field = {
                    "best_match_index": "match_index",
                    "best_match_atom_indices": "match_atom_indices",
                }.get(field, field)
                if pair.get(field) != best.get(source_field):
                    raise AssertionError(f"Best-match field mismatch {field}: {key}")
            if not (
                pair.get("pred_before") == SOURCE_LABEL
                and pair.get("pred_after") == TARGET_LABEL
            ):
                raise AssertionError(f"Pair strict-flip semantics are incorrect: {key}")
        elif pair.get("wnode_distance") is not None:
            raise AssertionError(f"Non-strict pair has a WNode distance: {key}")

    if require_strict_flip_pair and strict_pair_count <= 0:
        raise AssertionError("At least one strict-flip pair is required.")
    if forbid_test:
        if manifest.get("test_loaded") is not False or summary.get("test_loaded") is not False:
            raise AssertionError("Run does not explicitly prove test_loaded=false.")
        calibration_path = Path(
            str((manifest.get("inputs") or {}).get("calibration_csv", {}).get("path") or "")
        )
        if _path_has_test_token(calibration_path):
            raise AssertionError("Manifest references a forbidden test input.")
    if summary.get("wnode_self_test_passed") is not True:
        raise AssertionError("WNode self-test did not pass.")
    if summary.get("run_complete") is not True or complete.get("run_complete") is not True:
        raise AssertionError("Run is not complete.")

    return {
        "audit_passed": True,
        "parent_count": len(parent_ids),
        "candidate_count": len(candidate_ids),
        "pair_count": len(pairs),
        "match_instance_count": len(matches),
        "strict_flip_pair_count": strict_pair_count,
        "complete_cartesian": set(pair_keys) == expected_cartesian,
        "test_loaded": False,
        "run_complete": True,
    }


__all__ = [
    "CANDIDATE_ORDER_SOURCE_SUPPORT",
    "CalibrationParent",
    "DEFAULT_WNODE_SIZE_PENALTY_BETA",
    "MatrixBuildConfig",
    "SOURCE_LABEL",
    "TARGET_LABEL",
    "audit_calibration_matrix_run",
    "build_calibration_matrix_run",
    "build_candidate_universe",
    "build_cartesian_rows",
    "calibration_cohort_hash",
    "evaluate_parent_candidate_pair",
    "is_source_eligible",
    "load_calibration_parents",
    "run_wnode_self_test",
    "stable_candidate_id",
]
