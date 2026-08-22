"""Calibration-only WNode-aware prefix selection for Mutagenicity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - import availability is environment dependent
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:  # pragma: no cover
    Chem = None
    DataStructs = None
    rdFingerprintGenerator = None


DEFAULT_THRESHOLD_QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
DEFAULT_THRESHOLD_WEIGHTS = (4.0, 4.0, 3.0, 3.0, 2.0, 1.0, 1.0)
DEFAULT_PREFIX_WEIGHTS = (1.0,) * 10 + (0.5,) * 10
DEFAULT_THETA_STAR_QUANTILE = 0.30
DEFAULT_COST_CAP_QUANTILE = 0.90
VARIANT_NAMES = (
    "A1_SingleTheta",
    "A2_MultiThreshold",
    "A3_MultiThresholdPrefix",
    "A4_MultiThresholdPrefixCovRedSwap",
)
FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ThresholdLevel:
    threshold_id: str
    threshold: float
    weight: float
    quantiles: tuple[float, ...]
    quantile_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ThresholdBundle:
    finite_distance_count: int
    requested_quantiles: tuple[float, ...]
    requested_weights: tuple[float, ...]
    raw_thresholds: tuple[float, ...]
    quantile_labels: tuple[str, ...]
    levels: tuple[ThresholdLevel, ...]
    theta_star_quantile: float
    theta_star: float
    cost_cap_quantile: float
    cost_cap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "finite_strict_flip_distance_count": self.finite_distance_count,
            "quantile_method": "linear",
            "dtype": "float64",
            "requested_quantiles": list(self.requested_quantiles),
            "requested_weights": list(self.requested_weights),
            "raw_quantile_thresholds": [
                {
                    "quantile": quantile,
                    "quantile_label": label,
                    "threshold": threshold,
                    "weight": weight,
                }
                for quantile, label, threshold, weight in zip(
                    self.requested_quantiles,
                    self.quantile_labels,
                    self.raw_thresholds,
                    self.requested_weights,
                )
            ],
            "merged_thresholds": [
                {
                    "threshold_id": level.threshold_id,
                    "threshold": level.threshold,
                    "weight": level.weight,
                    "quantiles": list(level.quantiles),
                    "quantile_labels": list(level.quantile_labels),
                }
                for level in self.levels
            ],
            "duplicate_thresholds_merged": len(self.levels)
            < len(self.requested_quantiles),
            "theta_star_quantile": self.theta_star_quantile,
            "theta_star": self.theta_star,
            "cost_cap_quantile": self.cost_cap_quantile,
            "cost_cap": self.cost_cap,
            "threshold_source": "calibration_all_finite_strict_flip_pairs",
            "test_used": False,
        }


@dataclass(frozen=True, slots=True)
class VariantConfig:
    name: str
    single_theta: bool
    lambda_cost: float
    lambda_covred: float
    lambda_struct: float
    lambda_size: float
    insertion_reorder: bool
    local_swap: bool


@dataclass(slots=True)
class MatrixData:
    matrix_run_dir: Path
    parent_ids: tuple[str, ...]
    candidate_rows: tuple[dict[str, Any], ...]
    distances: np.ndarray
    cf_drops: np.ndarray
    applicable: np.ndarray
    full_finite_distances: np.ndarray
    full_parent_count: int
    full_candidate_count: int
    full_pair_count: int
    full_strict_flip_pair_count: int
    summary: dict[str, Any]
    manifest: dict[str, Any]
    full_candidate_rows: tuple[dict[str, Any], ...] | None = None

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(str(row["candidate_id"]) for row in self.candidate_rows)

    @property
    def candidate_index(self) -> dict[str, int]:
        return {
            candidate_id: index
            for index, candidate_id in enumerate(self.candidate_ids)
        }


@dataclass(slots=True)
class ChemistryData:
    heavy_atom_counts: np.ndarray
    normalized_sizes: np.ndarray
    structural_similarity: np.ndarray


def preregistered_variant_configs() -> tuple[VariantConfig, ...]:
    return (
        VariantConfig(
            name="A1_SingleTheta",
            single_theta=True,
            lambda_cost=0.15,
            lambda_covred=0.0,
            lambda_struct=0.0,
            lambda_size=0.0,
            insertion_reorder=False,
            local_swap=False,
        ),
        VariantConfig(
            name="A2_MultiThreshold",
            single_theta=False,
            lambda_cost=0.15,
            lambda_covred=0.0,
            lambda_struct=0.0,
            lambda_size=0.0,
            insertion_reorder=False,
            local_swap=False,
        ),
        VariantConfig(
            name="A3_MultiThresholdPrefix",
            single_theta=False,
            lambda_cost=0.20,
            lambda_covred=0.0,
            lambda_struct=0.0,
            lambda_size=0.0,
            insertion_reorder=True,
            local_swap=False,
        ),
        VariantConfig(
            name="A4_MultiThresholdPrefixCovRedSwap",
            single_theta=False,
            lambda_cost=0.20,
            lambda_covred=0.02,
            lambda_struct=0.01,
            lambda_size=0.005,
            insertion_reorder=True,
            local_swap=True,
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _quantile_label(quantile: float) -> str:
    percent = float(quantile) * 100.0
    if math.isclose(percent, round(percent), abs_tol=1e-12):
        return f"q{int(round(percent)):02d}"
    rendered = f"{float(quantile):.12g}".replace(".", "p")
    return f"q{rendered}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(value, ensure_ascii=False)
                            if isinstance(value, (list, tuple, dict))
                            else ("" if value is None else value)
                        )
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(payload)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    if resolved.is_file():
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest() if resolved.is_file() else None,
    }


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _basename_has_test_token(path: Path) -> bool:
    name = path.name.lower()
    normalized = name.replace("-", "_").replace(".", "_")
    return "test" in {token for token in normalized.split("_") if token}


def _manifest_input_paths(payload: Any, key_hint: str = "") -> list[Path]:
    paths: list[Path] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            paths.extend(_manifest_input_paths(value, str(key).lower()))
    elif isinstance(payload, list):
        for value in payload:
            paths.extend(_manifest_input_paths(value, key_hint))
    elif isinstance(payload, str) and any(
        token in key_hint for token in ("path", "csv", "file", "dir")
    ):
        paths.append(Path(payload))
    return paths


def _validate_calibration_only(
    matrix_run_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if _basename_has_test_token(matrix_run_dir):
        raise ValueError(f"Test matrix path is forbidden: {matrix_run_dir}")
    if summary.get("test_loaded") is not False:
        raise ValueError("Matrix summary does not prove test_loaded=false.")
    if manifest.get("test_loaded") is not False:
        raise ValueError("Matrix manifest does not prove test_loaded=false.")
    inputs = manifest.get("inputs") or {}
    cohort_name = str(inputs.get("cohort_name") or "").strip().lower()
    if cohort_name != "calibration":
        raise ValueError(
            f"Matrix cohort must be calibration, found {cohort_name!r}."
        )
    for path in _manifest_input_paths(inputs):
        if _basename_has_test_token(path):
            raise ValueError(f"Matrix manifest references test input: {path}")


def load_calibration_matrix(
    matrix_run_dir: str | Path,
    *,
    parent_limit: int = 0,
    candidate_limit: int = 0,
    forbid_test: bool = True,
) -> MatrixData:
    """Load a stable calibration subset while retaining full distances for thresholds."""

    root = Path(matrix_run_dir).expanduser().resolve()
    required = (
        root / "pair_matrix.jsonl",
        root / "selected_candidate_universe.jsonl",
        root / "summary.json",
        root / "run_manifest.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required matrix artifact is missing: {path}")
    if int(parent_limit) < 0 or int(candidate_limit) < 0:
        raise ValueError("parent_limit and candidate_limit must be non-negative.")

    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "run_manifest.json")
    _validate_calibration_only(root, summary, manifest)
    if forbid_test:
        _validate_calibration_only(root, summary, manifest)

    all_candidates = _read_jsonl(root / "selected_candidate_universe.jsonl")
    all_candidate_ids = [str(row.get("candidate_id") or "") for row in all_candidates]
    if any(not candidate_id for candidate_id in all_candidate_ids):
        raise ValueError("Candidate universe contains an empty candidate_id.")
    if len(all_candidate_ids) != len(set(all_candidate_ids)):
        raise ValueError("Candidate universe contains duplicate candidate IDs.")
    full_candidate_set = set(all_candidate_ids)

    pair_path = root / "pair_matrix.jsonl"
    parent_order: list[str] = []
    seen_parents: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    full_finite_distances: list[float] = []
    full_pair_count = 0
    full_strict_count = 0
    with pair_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            parent_id = str(row.get("parent_id") or "").strip()
            candidate_id = str(row.get("candidate_id") or "").strip()
            if not parent_id or candidate_id not in full_candidate_set:
                raise ValueError(f"Invalid pair identity at line {line_number}.")
            key = (parent_id, candidate_id)
            if key in seen_pairs:
                raise ValueError(f"Duplicate matrix pair: {key}")
            seen_pairs.add(key)
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_order.append(parent_id)
            full_pair_count += 1
            if _bool_value(row.get("pair_strict_flip")):
                distance = _finite_float(row.get("wnode_distance"))
                if distance is None or distance < 0.0:
                    raise ValueError(
                        f"Strict-flip pair lacks finite non-negative WNode: {key}"
                    )
                if not _bool_value(row.get("applicable")):
                    raise ValueError(f"Strict-flip pair is not applicable: {key}")
                full_strict_count += 1
                full_finite_distances.append(distance)

    expected_full_pairs = len(parent_order) * len(all_candidates)
    if full_pair_count != expected_full_pairs:
        raise ValueError(
            "Matrix is not a complete parent-candidate Cartesian product: "
            f"rows={full_pair_count}, expected={expected_full_pairs}."
        )
    if int(summary.get("parent_count") or 0) != len(parent_order):
        raise ValueError("Matrix summary parent_count does not match pair_matrix.")
    if int(summary.get("selected_candidate_count") or 0) != len(all_candidates):
        raise ValueError(
            "Matrix summary selected_candidate_count does not match candidate universe."
        )
    if int(summary.get("strict_flip_pair_count") or 0) != full_strict_count:
        raise ValueError("Matrix summary strict_flip_pair_count does not match.")

    selected_parent_ids = (
        parent_order[: int(parent_limit)] if int(parent_limit) else parent_order
    )
    selected_candidates = (
        all_candidates[: int(candidate_limit)]
        if int(candidate_limit)
        else all_candidates
    )
    if not selected_parent_ids or not selected_candidates:
        raise ValueError("Selector requires at least one parent and one candidate.")
    parent_index = {
        parent_id: index for index, parent_id in enumerate(selected_parent_ids)
    }
    candidate_index = {
        str(row["candidate_id"]): index
        for index, row in enumerate(selected_candidates)
    }
    distances = np.full(
        (len(selected_parent_ids), len(selected_candidates)),
        np.inf,
        dtype=np.float64,
    )
    cf_drops = np.full_like(distances, np.nan)
    applicable = np.zeros_like(distances, dtype=bool)
    selected_pair_count = 0
    with pair_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            parent_id = str(row.get("parent_id") or "").strip()
            candidate_id = str(row.get("candidate_id") or "").strip()
            if parent_id not in parent_index or candidate_id not in candidate_index:
                continue
            parent_position = parent_index[parent_id]
            candidate_position = candidate_index[candidate_id]
            selected_pair_count += 1
            applicable[parent_position, candidate_position] = _bool_value(
                row.get("applicable")
            )
            if not _bool_value(row.get("pair_strict_flip")):
                continue
            distance = _finite_float(row.get("wnode_distance"))
            cf_drop = _finite_float(row.get("cf_drop"))
            if distance is None or distance < 0.0 or cf_drop is None:
                raise ValueError(
                    "Strict-flip matrix entry requires finite WNode and CFDrop: "
                    f"{parent_id}/{candidate_id}"
                )
            distances[parent_position, candidate_position] = distance
            cf_drops[parent_position, candidate_position] = cf_drop
    expected_selected_pairs = len(selected_parent_ids) * len(selected_candidates)
    if selected_pair_count != expected_selected_pairs:
        raise ValueError(
            "Selected matrix subset is incomplete: "
            f"rows={selected_pair_count}, expected={expected_selected_pairs}."
        )

    return MatrixData(
        matrix_run_dir=root,
        parent_ids=tuple(selected_parent_ids),
        candidate_rows=tuple(dict(row) for row in selected_candidates),
        distances=distances,
        cf_drops=cf_drops,
        applicable=applicable,
        full_finite_distances=np.asarray(full_finite_distances, dtype=np.float64),
        full_parent_count=len(parent_order),
        full_candidate_count=len(all_candidates),
        full_pair_count=full_pair_count,
        full_strict_flip_pair_count=full_strict_count,
        summary=summary,
        manifest=manifest,
        full_candidate_rows=tuple(dict(row) for row in all_candidates),
    )


def derive_thresholds(
    finite_distances: Sequence[float] | np.ndarray,
    *,
    quantiles: Sequence[float] = DEFAULT_THRESHOLD_QUANTILES,
    weights: Sequence[float] = DEFAULT_THRESHOLD_WEIGHTS,
    theta_star_quantile: float = DEFAULT_THETA_STAR_QUANTILE,
    cost_cap_quantile: float = DEFAULT_COST_CAP_QUANTILE,
) -> ThresholdBundle:
    values = np.asarray(finite_distances, dtype=np.float64)
    if values.ndim != 1 or values.size <= 0:
        raise ValueError("Threshold calibration requires finite strict-flip distances.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Threshold calibration distances must be finite and non-negative.")
    parsed_quantiles = tuple(float(value) for value in quantiles)
    parsed_weights = tuple(float(value) for value in weights)
    if len(parsed_quantiles) != len(parsed_weights) or not parsed_quantiles:
        raise ValueError("threshold quantiles and weights must have equal non-zero length.")
    if any(value < 0.0 or value > 1.0 for value in parsed_quantiles):
        raise ValueError("Threshold quantiles must be in [0, 1].")
    if any(value < 0.0 or not math.isfinite(value) for value in parsed_weights):
        raise ValueError("Threshold weights must be finite and non-negative.")
    if sum(parsed_weights) <= 0.0:
        raise ValueError("At least one threshold weight must be positive.")
    for value, name in (
        (theta_star_quantile, "theta_star_quantile"),
        (cost_cap_quantile, "cost_cap_quantile"),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    raw = np.quantile(
        values,
        np.asarray(parsed_quantiles, dtype=np.float64),
        method="linear",
    ).astype(np.float64)
    labels = tuple(_quantile_label(value) for value in parsed_quantiles)
    grouped: dict[float, dict[str, Any]] = {}
    order: list[float] = []
    for quantile, label, threshold, weight in zip(
        parsed_quantiles,
        labels,
        raw.tolist(),
        parsed_weights,
    ):
        key = float(threshold)
        if key not in grouped:
            grouped[key] = {
                "weight": 0.0,
                "quantiles": [],
                "labels": [],
            }
            order.append(key)
        grouped[key]["weight"] += float(weight)
        grouped[key]["quantiles"].append(float(quantile))
        grouped[key]["labels"].append(label)
    levels = tuple(
        ThresholdLevel(
            threshold_id=str(grouped[value]["labels"][0]),
            threshold=float(value),
            weight=float(grouped[value]["weight"]),
            quantiles=tuple(grouped[value]["quantiles"]),
            quantile_labels=tuple(grouped[value]["labels"]),
        )
        for value in order
    )
    theta_star = float(
        np.quantile(
            values,
            np.float64(theta_star_quantile),
            method="linear",
        )
    )
    cost_cap = float(
        np.quantile(
            values,
            np.float64(cost_cap_quantile),
            method="linear",
        )
    )
    return ThresholdBundle(
        finite_distance_count=int(values.size),
        requested_quantiles=parsed_quantiles,
        requested_weights=parsed_weights,
        raw_thresholds=tuple(float(value) for value in raw.tolist()),
        quantile_labels=labels,
        levels=levels,
        theta_star_quantile=float(theta_star_quantile),
        theta_star=theta_star,
        cost_cap_quantile=float(cost_cap_quantile),
        cost_cap=cost_cap,
    )


def single_threshold_coverage(
    best_distances: np.ndarray,
    threshold: float,
) -> float:
    values = np.asarray(best_distances, dtype=np.float64)
    return float(np.count_nonzero(values <= float(threshold)) / values.size)


def weighted_multi_threshold_utility(
    best_distances: np.ndarray,
    levels: Sequence[ThresholdLevel],
) -> float:
    total_weight = float(sum(level.weight for level in levels))
    if total_weight <= 0.0:
        raise ValueError("Threshold levels require positive total weight.")
    return float(
        sum(
            level.weight
            * single_threshold_coverage(best_distances, level.threshold)
            for level in levels
        )
        / total_weight
    )


def fixed_denominator_capped_cost(
    best_distances: np.ndarray,
    cost_cap: float,
) -> tuple[float, float, np.ndarray]:
    cap = float(cost_cap)
    if cap < 0.0 or not math.isfinite(cap):
        raise ValueError("cost_cap must be finite and non-negative.")
    values = np.asarray(best_distances, dtype=np.float64)
    capped = np.minimum(values, cap)
    capped[~np.isfinite(capped)] = cap
    return float(np.mean(capped)), float(np.median(capped)), capped


def capped_cost_utility(capped_mean_cost: float, cost_cap: float) -> float:
    cap = float(cost_cap)
    if cap <= 0.0:
        return 1.0 if float(capped_mean_cost) <= 0.0 else 0.0
    return float(1.0 - float(capped_mean_cost) / cap)


def weighted_coverage_jaccard(
    left_distances: np.ndarray,
    right_distances: np.ndarray,
    levels: Sequence[ThresholdLevel],
) -> float:
    left = np.asarray(left_distances, dtype=np.float64)
    right = np.asarray(right_distances, dtype=np.float64)
    total_weight = float(sum(level.weight for level in levels))
    if total_weight <= 0.0:
        raise ValueError("Coverage Jaccard requires positive threshold weight.")
    result = 0.0
    for level in levels:
        left_covered = left <= level.threshold
        right_covered = right <= level.threshold
        union = int(np.count_nonzero(left_covered | right_covered))
        jaccard = (
            float(np.count_nonzero(left_covered & right_covered) / union)
            if union
            else 0.0
        )
        result += level.weight * jaccard
    return float(result / total_weight)


def build_coverage_redundancy_matrix(
    distances: np.ndarray,
    levels: Sequence[ThresholdLevel],
) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    candidate_count = int(values.shape[1])
    result = np.zeros((candidate_count, candidate_count), dtype=np.float64)
    total_weight = float(sum(level.weight for level in levels))
    if total_weight <= 0.0:
        raise ValueError("Coverage redundancy requires positive threshold weight.")
    for level in levels:
        covered = (values <= level.threshold).astype(np.int32)
        counts = covered.sum(axis=0, dtype=np.int64)
        intersections = covered.T @ covered
        unions = counts[:, None] + counts[None, :] - intersections
        jaccard = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=np.float64),
            where=unions > 0,
        )
        result += float(level.weight) * jaccard
    result /= total_weight
    return result


def morgan_tanimoto(
    left_smiles: str,
    right_smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    if Chem is None or DataStructs is None or rdFingerprintGenerator is None:
        raise RuntimeError("RDKit is required for Morgan/Tanimoto redundancy.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )
    molecules = [
        Chem.MolFromSmiles(str(smiles or "").strip())
        for smiles in (left_smiles, right_smiles)
    ]
    if any(molecule is None for molecule in molecules):
        raise ValueError(
            f"Invalid fragment for Morgan fingerprint: {left_smiles!r}, "
            f"{right_smiles!r}"
        )
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    return float(DataStructs.TanimotoSimilarity(*fingerprints))


def build_candidate_chemistry(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    size_normalization_rows: Sequence[dict[str, Any]] | None = None,
    radius: int = 2,
    n_bits: int = 2048,
) -> ChemistryData:
    native_flags = [isinstance(row.get("selector_chemistry"), dict) for row in candidate_rows]
    if any(native_flags) and not all(native_flags):
        raise ValueError(
            "Candidate universe mixes native-rule and fragment chemistry schemas."
        )
    if native_flags and all(native_flags):
        def parse_native(row: dict[str, Any]) -> tuple[set[int], int, int]:
            payload = dict(row["selector_chemistry"])
            if (
                payload.get("schema_version")
                != "globalgce_native_rule_selector_chemistry_v1"
                or payload.get("role")
                != "native_lhs_rhs_rule_redundancy_only"
                or payload.get("fingerprint_kind")
                != "hashed_aligned_label_transition_bits"
                or payload.get("canonical_fragment_applicable") is not False
            ):
                raise ValueError("Invalid native GlobalGCE selector chemistry contract.")
            width = int(payload.get("fingerprint_n_bits") or 0)
            if width < 128:
                raise ValueError("Native rule selector fingerprint width is invalid.")
            raw_bits = payload.get("fingerprint_bits")
            if not isinstance(raw_bits, list) or not raw_bits:
                raise ValueError("Native rule selector fingerprint is empty.")
            bits = {int(value) for value in raw_bits}
            if len(bits) != len(raw_bits) or min(bits) < 0 or max(bits) >= width:
                raise ValueError("Native rule selector fingerprint bits are invalid.")
            heavy = int(payload.get("heavy_atom_count") or 0)
            if heavy <= 0:
                raise ValueError("Native rule selector heavy-atom count is invalid.")
            return bits, heavy, width

        parsed = [parse_native(dict(row)) for row in candidate_rows]
        widths = {width for _, _, width in parsed}
        if len(widths) != 1:
            raise ValueError("Native rule selector fingerprint widths differ.")
        normalization_rows = (
            size_normalization_rows
            if size_normalization_rows is not None
            else candidate_rows
        )
        if not normalization_rows or not all(
            isinstance(row.get("selector_chemistry"), dict)
            for row in normalization_rows
        ):
            raise ValueError(
                "Native rule size normalization requires the same chemistry schema."
            )
        normalization = [parse_native(dict(row)) for row in normalization_rows]
        if {width for _, _, width in normalization} != widths:
            raise ValueError("Native rule normalization fingerprint width changed.")
        maximum = max(heavy for _, heavy, _ in normalization)
        fingerprints = [bits for bits, _, _ in parsed]
        similarity = np.eye(len(fingerprints), dtype=np.float64)
        for left_index, left in enumerate(fingerprints):
            for right_index in range(left_index + 1, len(fingerprints)):
                right = fingerprints[right_index]
                union = left | right
                value = float(len(left & right) / len(union)) if union else 0.0
                similarity[left_index, right_index] = value
                similarity[right_index, left_index] = value
        heavy_array = np.asarray(
            [heavy for _, heavy, _ in parsed], dtype=np.int64
        )
        return ChemistryData(
            heavy_atom_counts=heavy_array,
            normalized_sizes=heavy_array.astype(np.float64) / float(maximum),
            structural_similarity=similarity,
        )
    if Chem is None or DataStructs is None or rdFingerprintGenerator is None:
        raise RuntimeError("RDKit is required for selector chemistry metrics.")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )
    fingerprints: list[Any] = []
    heavy_atoms: list[int] = []
    for row in candidate_rows:
        fragment = str(row.get("canonical_fragment") or "").strip()
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            raise ValueError(
                f"Invalid candidate fragment cannot be fingerprinted: {fragment!r}"
            )
        try:
            Chem.SanitizeMol(molecule)
        except Exception as exc:
            raise ValueError(
                f"Candidate fragment failed RDKit sanitization: {fragment!r}"
            ) from exc
        heavy_atoms.append(int(molecule.GetNumHeavyAtoms()))
        fingerprints.append(generator.GetFingerprint(molecule))
    normalization_rows = (
        size_normalization_rows
        if size_normalization_rows is not None
        else candidate_rows
    )
    normalization_heavy_atoms: list[int] = []
    for row in normalization_rows:
        fragment = str(row.get("canonical_fragment") or "").strip()
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            raise ValueError(
                f"Invalid size-normalization fragment: {fragment!r}"
            )
        normalization_heavy_atoms.append(int(molecule.GetNumHeavyAtoms()))
    maximum = max(normalization_heavy_atoms) if normalization_heavy_atoms else 0
    if maximum <= 0:
        raise ValueError("Candidate universe has no heavy atoms.")
    similarity = np.eye(len(fingerprints), dtype=np.float64)
    for left_index, left in enumerate(fingerprints):
        for right_index in range(left_index + 1, len(fingerprints)):
            value = float(
                DataStructs.TanimotoSimilarity(left, fingerprints[right_index])
            )
            similarity[left_index, right_index] = value
            similarity[right_index, left_index] = value
    heavy_array = np.asarray(heavy_atoms, dtype=np.int64)
    return ChemistryData(
        heavy_atom_counts=heavy_array,
        normalized_sizes=heavy_array.astype(np.float64) / float(maximum),
        structural_similarity=similarity,
    )


def _pairwise_mean(matrix: np.ndarray, sequence: Sequence[int]) -> float:
    if len(sequence) < 2:
        return 0.0
    indices = np.asarray(sequence, dtype=np.int64)
    selected = matrix[np.ix_(indices, indices)]
    upper = selected[np.triu_indices(len(sequence), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _active_levels(
    variant: VariantConfig,
    thresholds: ThresholdBundle,
) -> tuple[ThresholdLevel, ...]:
    if not variant.single_theta:
        return thresholds.levels
    return (
        ThresholdLevel(
            threshold_id="theta_star",
            threshold=thresholds.theta_star,
            weight=1.0,
            quantiles=(thresholds.theta_star_quantile,),
            quantile_labels=(_quantile_label(thresholds.theta_star_quantile),),
        ),
    )


def _update_best(
    best_distances: np.ndarray,
    best_cf_drops: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_cf_drops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    updated_distances = np.minimum(best_distances, candidate_distances)
    improved = candidate_distances < best_distances
    tied_better_cf = (
        np.isfinite(candidate_distances)
        & np.isclose(candidate_distances, best_distances, atol=0.0, rtol=0.0)
        & (
            np.nan_to_num(candidate_cf_drops, nan=-np.inf)
            > np.nan_to_num(best_cf_drops, nan=-np.inf)
        )
    )
    choose = improved | tied_better_cf
    updated_cf_drops = np.where(choose, candidate_cf_drops, best_cf_drops)
    return updated_distances, updated_cf_drops


def sequence_objective_components(
    sequence: Sequence[int],
    *,
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    prefix_weights: Sequence[float],
    variant: VariantConfig,
    coverage_redundancy_matrix: np.ndarray,
    structural_similarity_matrix: np.ndarray,
    normalized_sizes: np.ndarray,
) -> dict[str, float]:
    if not sequence:
        return {
            "prefix_weighted_multi_threshold_utility": 0.0,
            "prefix_weighted_capped_cost_utility": 0.0,
            "coverage_redundancy_penalty_value": 0.0,
            "structural_redundancy_penalty_value": 0.0,
            "size_penalty_value": 0.0,
            "objective": 0.0,
        }
    if len(prefix_weights) < len(sequence):
        raise ValueError("prefix_weights is shorter than the selected sequence.")
    levels = _active_levels(variant, thresholds)
    best = np.full(len(matrix.parent_ids), np.inf, dtype=np.float64)
    weighted_multi_sum = 0.0
    weighted_cost_sum = 0.0
    rho_sum = 0.0
    for rank, candidate_index in enumerate(sequence):
        best = np.minimum(best, matrix.distances[:, int(candidate_index)])
        rho = float(prefix_weights[rank])
        if rho < 0.0 or not math.isfinite(rho):
            raise ValueError("prefix weights must be finite and non-negative.")
        capped_mean, _, _ = fixed_denominator_capped_cost(best, thresholds.cost_cap)
        weighted_multi_sum += rho * weighted_multi_threshold_utility(best, levels)
        weighted_cost_sum += rho * capped_cost_utility(
            capped_mean,
            thresholds.cost_cap,
        )
        rho_sum += rho
    if rho_sum <= 0.0:
        raise ValueError("Selected prefix weights must have positive sum.")
    prefix_multi = float(weighted_multi_sum / rho_sum)
    prefix_cost = float(weighted_cost_sum / rho_sum)
    covred = _pairwise_mean(coverage_redundancy_matrix, sequence)
    structural = _pairwise_mean(structural_similarity_matrix, sequence)
    size = float(np.mean(normalized_sizes[np.asarray(sequence, dtype=np.int64)]))
    objective = (
        prefix_multi
        + float(variant.lambda_cost) * prefix_cost
        - float(variant.lambda_covred) * covred
        - float(variant.lambda_struct) * structural
        - float(variant.lambda_size) * size
    )
    return {
        "prefix_weighted_multi_threshold_utility": prefix_multi,
        "prefix_weighted_capped_cost_utility": prefix_cost,
        "coverage_redundancy_penalty_value": covred,
        "structural_redundancy_penalty_value": structural,
        "size_penalty_value": size,
        "objective": float(objective),
    }


def compute_prefix_metrics(
    sequence: Sequence[int],
    *,
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    coverage_redundancy_matrix: np.ndarray,
    structural_similarity_matrix: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    best = np.full(len(matrix.parent_ids), np.inf, dtype=np.float64)
    best_cf_drop = np.full(len(matrix.parent_ids), np.nan, dtype=np.float64)
    best_candidate = np.full(len(matrix.parent_ids), -1, dtype=np.int64)
    applicable = np.zeros(len(matrix.parent_ids), dtype=bool)
    metrics: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    selected: list[int] = []
    for rank, candidate_index in enumerate(sequence, start=1):
        candidate_index = int(candidate_index)
        candidate_distances = matrix.distances[:, candidate_index]
        candidate_cf_drops = matrix.cf_drops[:, candidate_index]
        previous_best = best.copy()
        previous_cf_drop = best_cf_drop.copy()
        improved = candidate_distances < previous_best
        tied_better_cf = (
            np.isfinite(candidate_distances)
            & np.isclose(candidate_distances, previous_best, atol=0.0, rtol=0.0)
            & (
                np.nan_to_num(candidate_cf_drops, nan=-np.inf)
                > np.nan_to_num(previous_cf_drop, nan=-np.inf)
            )
        )
        best, best_cf_drop = _update_best(
            best,
            best_cf_drop,
            candidate_distances,
            candidate_cf_drops,
        )
        best_candidate[improved | tied_better_cf] = candidate_index
        applicable |= matrix.applicable[:, candidate_index]
        selected.append(candidate_index)
        finite = np.isfinite(best)
        theta_covered = best <= thresholds.theta_star
        capped_mean, capped_median, capped = fixed_denominator_capped_cost(
            best,
            thresholds.cost_cap,
        )
        conditional = best[finite]
        theta_costs = best[theta_covered]
        row: dict[str, Any] = {
            "k": rank,
            "ccrcov_theta_star": float(np.mean(theta_covered)),
            "weighted_multi_threshold_utility": weighted_multi_threshold_utility(
                best,
                thresholds.levels,
            ),
            "applicable_rate": float(np.mean(applicable)),
            "strict_flip_parent_count": int(np.count_nonzero(finite)),
            "num_theta_star_covered": int(np.count_nonzero(theta_covered)),
            "fixed_capped_mean_cost": capped_mean,
            "fixed_capped_median_cost": capped_median,
            "conditional_mean_cost": (
                float(np.mean(conditional)) if conditional.size else None
            ),
            "conditional_median_cost": (
                float(np.median(conditional)) if conditional.size else None
            ),
            "theta_star_conditional_mean_cost": (
                float(np.mean(theta_costs)) if theta_costs.size else None
            ),
            "theta_star_conditional_median_cost": (
                float(np.median(theta_costs)) if theta_costs.size else None
            ),
            "mean_cf_drop": (
                float(np.mean(best_cf_drop[theta_covered]))
                if np.any(theta_covered)
                else None
            ),
            "coverage_redundancy": _pairwise_mean(
                coverage_redundancy_matrix,
                selected,
            ),
            "structural_redundancy": _pairwise_mean(
                structural_similarity_matrix,
                selected,
            ),
        }
        for label, threshold in zip(
            thresholds.quantile_labels,
            thresholds.raw_thresholds,
        ):
            row[f"ccrcov_theta_{label}"] = single_threshold_coverage(
                best,
                threshold,
            )
        metrics.append(row)
        for parent_position, parent_id in enumerate(matrix.parent_ids):
            chosen_index = int(best_candidate[parent_position])
            parent_rows.append(
                {
                    "k": rank,
                    "parent_id": parent_id,
                    "best_distance": (
                        float(best[parent_position])
                        if math.isfinite(float(best[parent_position]))
                        else None
                    ),
                    "capped_distance": float(capped[parent_position]),
                    "best_candidate_id": (
                        matrix.candidate_ids[chosen_index]
                        if chosen_index >= 0
                        else None
                    ),
                    "cf_drop": (
                        float(best_cf_drop[parent_position])
                        if math.isfinite(float(best_cf_drop[parent_position]))
                        else None
                    ),
                    "strict_recourse_available": bool(finite[parent_position]),
                    "theta_star_covered": bool(theta_covered[parent_position]),
                    "applicable": bool(applicable[parent_position]),
                }
            )
    return metrics, parent_rows


def _objective_callable(
    *,
    matrix: MatrixData,
    thresholds: ThresholdBundle,
    prefix_weights: Sequence[float],
    variant: VariantConfig,
    coverage_redundancy_matrix: np.ndarray,
    structural_similarity_matrix: np.ndarray,
    normalized_sizes: np.ndarray,
) -> Callable[[Sequence[int]], float]:
    def evaluate(sequence: Sequence[int]) -> float:
        return sequence_objective_components(
            sequence,
            matrix=matrix,
            thresholds=thresholds,
            prefix_weights=prefix_weights,
            variant=variant,
            coverage_redundancy_matrix=coverage_redundancy_matrix,
            structural_similarity_matrix=structural_similarity_matrix,
            normalized_sizes=normalized_sizes,
        )["objective"]

    return evaluate


def greedy_select(
    candidate_indices: Sequence[int],
    *,
    top_k: int,
    objective_fn: Callable[[Sequence[int]], float],
    candidate_ids: Sequence[str],
) -> tuple[list[int], list[dict[str, Any]]]:
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")
    if len(candidate_indices) < int(top_k):
        raise ValueError(
            f"Candidate universe has {len(candidate_indices)} rows, below top_k={top_k}."
        )
    selected: list[int] = []
    remaining = set(int(index) for index in candidate_indices)
    trace: list[dict[str, Any]] = []
    previous_objective = 0.0
    for rank in range(1, int(top_k) + 1):
        scored: list[tuple[float, str, int]] = []
        for candidate_index in remaining:
            score = float(objective_fn([*selected, candidate_index]))
            scored.append((score, str(candidate_ids[candidate_index]), candidate_index))
        best_score = max(row[0] for row in scored)
        tied = [
            row
            for row in scored
            if math.isclose(row[0], best_score, abs_tol=FLOAT_TOLERANCE, rel_tol=0.0)
        ]
        _, candidate_id, chosen = min(tied, key=lambda row: row[1])
        selected.append(chosen)
        remaining.remove(chosen)
        trace.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "objective_before": previous_objective,
                "objective_after": best_score,
                "marginal_objective_gain": best_score - previous_objective,
                "tie_count": len(tied),
                "tie_break": "candidate_id_lexicographic",
            }
        )
        previous_objective = best_score
    return selected, trace


def optimize_insertion_order(
    sequence: Sequence[int],
    *,
    objective_fn: Callable[[Sequence[int]], float],
    candidate_ids: Sequence[str],
    tolerance: float = FLOAT_TOLERANCE,
) -> tuple[list[int], list[dict[str, Any]]]:
    current = [int(value) for value in sequence]
    current_objective = float(objective_fn(current))
    trace: list[dict[str, Any]] = [
        {
            "operation": "insertion_start",
            "objective": current_objective,
            "candidate_ids": [candidate_ids[index] for index in current],
        }
    ]
    maximum_iterations = max(1, len(current) * len(current))
    for iteration in range(1, maximum_iterations + 1):
        proposals: list[tuple[float, tuple[str, ...], int, int, list[int]]] = []
        for source_position in range(len(current)):
            for target_position in range(len(current)):
                if source_position == target_position:
                    continue
                proposal = list(current)
                moved = proposal.pop(source_position)
                proposal.insert(target_position, moved)
                score = float(objective_fn(proposal))
                proposals.append(
                    (
                        score,
                        tuple(candidate_ids[index] for index in proposal),
                        source_position,
                        target_position,
                        proposal,
                    )
                )
        if not proposals:
            break
        best_score = max(row[0] for row in proposals)
        if best_score <= current_objective + float(tolerance):
            break
        tied = [
            row
            for row in proposals
            if math.isclose(row[0], best_score, abs_tol=tolerance, rel_tol=0.0)
        ]
        score, _, source_position, target_position, proposal = min(
            tied,
            key=lambda row: row[1],
        )
        trace.append(
            {
                "operation": "insertion_move",
                "iteration": iteration,
                "source_position": source_position,
                "target_position": target_position,
                "objective_before": current_objective,
                "objective_after": score,
                "candidate_ids": [candidate_ids[index] for index in proposal],
            }
        )
        current = proposal
        current_objective = score
    trace.append(
        {
            "operation": "insertion_end",
            "objective": current_objective,
            "candidate_ids": [candidate_ids[index] for index in current],
        }
    )
    return current, trace


def local_swap_search(
    sequence: Sequence[int],
    *,
    all_candidate_indices: Sequence[int],
    objective_fn: Callable[[Sequence[int]], float],
    candidate_ids: Sequence[str],
    max_passes: int = 2,
    tolerance: float = FLOAT_TOLERANCE,
) -> tuple[list[int], list[dict[str, Any]]]:
    if int(max_passes) < 0:
        raise ValueError("max_passes must be non-negative.")
    current = [int(value) for value in sequence]
    current_objective = float(objective_fn(current))
    trace: list[dict[str, Any]] = [
        {
            "operation": "swap_start",
            "objective": current_objective,
            "candidate_ids": [candidate_ids[index] for index in current],
        }
    ]
    universe = set(int(value) for value in all_candidate_indices)
    for pass_index in range(1, int(max_passes) + 1):
        unselected = sorted(
            universe - set(current),
            key=lambda index: candidate_ids[index],
        )
        proposals: list[
            tuple[float, str, str, int, list[int]]
        ] = []
        for selected_position, outgoing in enumerate(current):
            for incoming in unselected:
                proposal = list(current)
                proposal[selected_position] = incoming
                score = float(objective_fn(proposal))
                proposals.append(
                    (
                        score,
                        str(candidate_ids[incoming]),
                        str(candidate_ids[outgoing]),
                        selected_position,
                        proposal,
                    )
                )
        if not proposals:
            break
        best_score = max(row[0] for row in proposals)
        if best_score <= current_objective + float(tolerance):
            trace.append(
                {
                    "operation": "swap_no_improvement",
                    "pass": pass_index,
                    "objective": current_objective,
                }
            )
            break
        tied = [
            row
            for row in proposals
            if math.isclose(row[0], best_score, abs_tol=tolerance, rel_tol=0.0)
        ]
        score, incoming_id, outgoing_id, position, proposal = min(
            tied,
            key=lambda row: (row[1], row[2], row[3]),
        )
        trace.append(
            {
                "operation": "swap_accept",
                "pass": pass_index,
                "position": position,
                "incoming_candidate_id": incoming_id,
                "outgoing_candidate_id": outgoing_id,
                "objective_before": current_objective,
                "objective_after": score,
                "strict_improvement": True,
            }
        )
        current = proposal
        current_objective = score
    trace.append(
        {
            "operation": "swap_end",
            "objective": current_objective,
            "candidate_ids": [candidate_ids[index] for index in current],
        }
    )
    return current, trace


def _prefix_weighted_metric(
    metrics: Sequence[dict[str, Any]],
    prefix_weights: Sequence[float],
    field: str,
) -> float:
    weights = np.asarray(prefix_weights[: len(metrics)], dtype=np.float64)
    values = np.asarray([float(row[field]) for row in metrics], dtype=np.float64)
    return float(np.sum(weights * values) / np.sum(weights))


def build_variant_comparison_row(
    variant: VariantConfig,
    metrics: Sequence[dict[str, Any]],
    *,
    table_k: int,
    top_k: int,
    prefix_weights: Sequence[float],
    final_objective: float,
) -> dict[str, Any]:
    by_k = {int(row["k"]): row for row in metrics}
    table = by_k[int(table_k)]
    final = by_k[int(top_k)]
    return {
        "variant": variant.name,
        "prefix_weighted_multi_threshold_utility": _prefix_weighted_metric(
            metrics,
            prefix_weights,
            "weighted_multi_threshold_utility",
        ),
        "k10_ccrcov_theta_star": float(table["ccrcov_theta_star"]),
        "k10_fixed_capped_mean_cost": float(table["fixed_capped_mean_cost"]),
        "k20_weighted_multi_threshold_utility": float(
            final["weighted_multi_threshold_utility"]
        ),
        "k20_coverage_redundancy": float(final["coverage_redundancy"]),
        "k20_structural_redundancy": float(final["structural_redundancy"]),
        "final_variant_objective": float(final_objective),
        "table_k": int(table_k),
        "top_k": int(top_k),
    }


def variant_decision_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["prefix_weighted_multi_threshold_utility"]),
        -float(row["k10_ccrcov_theta_star"]),
        float(row["k10_fixed_capped_mean_cost"]),
        -float(row["k20_weighted_multi_threshold_utility"]),
        float(row["k20_coverage_redundancy"]),
        float(row["k20_structural_redundancy"]),
        str(row["variant"]),
    )


def choose_variant(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if {str(row.get("variant")) for row in rows} != set(VARIANT_NAMES):
        raise ValueError("Variant decision requires exactly preregistered A1-A4.")
    return min((dict(row) for row in rows), key=variant_decision_sort_key)


def _selected_sequence_rows(
    sequence: Sequence[int],
    *,
    matrix: MatrixData,
    chemistry: ChemistryData,
    thresholds: ThresholdBundle,
    metrics: Sequence[dict[str, Any]],
    objective_fn: Callable[[Sequence[int]], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_multi = 0.0
    for rank, candidate_index in enumerate(sequence, start=1):
        candidate = matrix.candidate_rows[int(candidate_index)]
        distances = matrix.distances[:, int(candidate_index)]
        current_multi = float(
            metrics[rank - 1]["weighted_multi_threshold_utility"]
        )
        selected = dict(candidate)
        selected.update(
            {
                "rank": rank,
                "candidate_id": str(candidate["candidate_id"]),
                "canonical_fragment": str(
                    candidate.get("canonical_fragment") or "N/A"
                ),
                "source_parent_count": int(candidate.get("source_parent_count") or 0),
                "source_cf_drop_mean": _finite_float(
                    candidate.get("source_cf_drop_mean")
                ),
                "source_reward_mean": _finite_float(
                    candidate.get("source_reward_mean")
                ),
                "heavy_atom_count": int(
                    chemistry.heavy_atom_counts[int(candidate_index)]
                ),
                "standalone_strict_parent_count": int(
                    np.count_nonzero(np.isfinite(distances))
                ),
                "standalone_theta_star_coverage": single_threshold_coverage(
                    distances,
                    thresholds.theta_star,
                ),
                "marginal_multi_threshold_gain": current_multi - previous_multi,
                "objective_after_selection": float(
                    objective_fn(sequence[:rank])
                ),
            }
        )
        rows.append(selected)
        previous_multi = current_multi
    return rows


def _variant_config_payload(
    variants: Sequence[VariantConfig],
    *,
    top_k: int,
    table_k: int,
    prefix_weights: Sequence[float],
    local_swap_passes: int,
) -> dict[str, Any]:
    return {
        "top_k": int(top_k),
        "table_k": int(table_k),
        "prefix_weights": [float(value) for value in prefix_weights],
        "coverage_redundancy": "weighted_threshold_parent_jaccard_mean",
        "structural_redundancy": "morgan_radius2_2048_tanimoto_mean",
        "size_penalty": "mean_heavy_atom_count_normalized_by_universe_max",
        "objective_formula": (
            "weighted_prefix_mean(F_multi + lambda_cost * "
            "(1 - capped_mean_cost / cost_cap)) - "
            "lambda_covred * CovRed - lambda_struct * StructRed - "
            "lambda_size * normalized_size"
        ),
        "deterministic_tie_break": "candidate_id_lexicographic",
        "local_swap_passes": int(local_swap_passes),
        "variants": {variant.name: asdict(variant) for variant in variants},
    }


def _run_one_variant(
    variant: VariantConfig,
    *,
    matrix: MatrixData,
    chemistry: ChemistryData,
    thresholds: ThresholdBundle,
    prefix_weights: Sequence[float],
    top_k: int,
    table_k: int,
    local_swap_passes: int,
    coverage_redundancy_matrix: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], list[int]]:
    objective_fn = _objective_callable(
        matrix=matrix,
        thresholds=thresholds,
        prefix_weights=prefix_weights,
        variant=variant,
        coverage_redundancy_matrix=coverage_redundancy_matrix,
        structural_similarity_matrix=chemistry.structural_similarity,
        normalized_sizes=chemistry.normalized_sizes,
    )
    all_indices = list(range(len(matrix.candidate_rows)))
    greedy_sequence, objective_trace = greedy_select(
        all_indices,
        top_k=top_k,
        objective_fn=objective_fn,
        candidate_ids=matrix.candidate_ids,
    )
    sequence = list(greedy_sequence)
    insertion_trace: list[dict[str, Any]] = []
    if variant.insertion_reorder:
        sequence, insertion_trace = optimize_insertion_order(
            sequence,
            objective_fn=objective_fn,
            candidate_ids=matrix.candidate_ids,
        )
    pre_swap_objective = float(objective_fn(sequence))
    swap_trace: list[dict[str, Any]] = []
    if variant.local_swap:
        sequence, swap_trace = local_swap_search(
            sequence,
            all_candidate_indices=all_indices,
            objective_fn=objective_fn,
            candidate_ids=matrix.candidate_ids,
            max_passes=local_swap_passes,
        )
    post_swap_objective = float(objective_fn(sequence))
    if post_swap_objective + FLOAT_TOLERANCE < pre_swap_objective:
        raise RuntimeError(f"{variant.name} local search decreased J_prefix.")

    metrics, parent_rows = compute_prefix_metrics(
        sequence,
        matrix=matrix,
        thresholds=thresholds,
        coverage_redundancy_matrix=coverage_redundancy_matrix,
        structural_similarity_matrix=chemistry.structural_similarity,
    )
    selected_rows = _selected_sequence_rows(
        sequence,
        matrix=matrix,
        chemistry=chemistry,
        thresholds=thresholds,
        metrics=metrics,
        objective_fn=objective_fn,
    )
    variant_dir = output_dir / "variants" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(variant_dir / "selected_sequence.jsonl", selected_rows)
    _write_csv(variant_dir / "prefix_metrics.csv", metrics)
    _write_json(
        variant_dir / "prefix_metrics.json",
        {
            "variant": variant.name,
            "metrics": metrics,
            "theta_star": thresholds.theta_star,
            "cost_cap": thresholds.cost_cap,
        },
    )
    _write_csv(variant_dir / "parent_best_distances.csv", parent_rows)
    _write_json(
        variant_dir / "selected_top10.json",
        {
            "variant": variant.name,
            "top_k": int(table_k),
            "candidate_ids": [
                str(row["candidate_id"]) for row in selected_rows[:table_k]
            ],
            "candidates": selected_rows[:table_k],
        },
    )
    _write_json(
        variant_dir / "selected_top20.json",
        {
            "variant": variant.name,
            "top_k": int(top_k),
            "candidate_ids": [str(row["candidate_id"]) for row in selected_rows],
            "candidates": selected_rows,
        },
    )
    _write_json(
        variant_dir / "objective_trace.json",
        {
            "variant": variant.name,
            "greedy_trace": objective_trace,
            "greedy_final_objective": float(objective_fn(greedy_sequence)),
            "final_objective": post_swap_objective,
        },
    )
    _write_json(
        variant_dir / "local_search_trace.json",
        {
            "variant": variant.name,
            "insertion_trace": insertion_trace,
            "swap_trace": swap_trace,
            "pre_swap_objective": pre_swap_objective,
            "post_swap_objective": post_swap_objective,
            "objective_non_decreasing": (
                post_swap_objective + FLOAT_TOLERANCE >= pre_swap_objective
            ),
        },
    )
    comparison = build_variant_comparison_row(
        variant,
        metrics,
        table_k=table_k,
        top_k=top_k,
        prefix_weights=prefix_weights,
        final_objective=post_swap_objective,
    )
    return comparison, sequence


def run_mutagenicity_wnode_selector(
    *,
    matrix_run_dir: str | Path,
    output_dir: str | Path,
    top_k: int = 20,
    table_k: int = 10,
    threshold_quantiles: Sequence[float] = DEFAULT_THRESHOLD_QUANTILES,
    threshold_weights: Sequence[float] = DEFAULT_THRESHOLD_WEIGHTS,
    theta_star_quantile: float = DEFAULT_THETA_STAR_QUANTILE,
    cost_cap_quantile: float = DEFAULT_COST_CAP_QUANTILE,
    prefix_weights: Sequence[float] = DEFAULT_PREFIX_WEIGHTS,
    parent_limit: int = 0,
    candidate_limit: int = 0,
    local_swap_passes: int = 2,
    seed: int = 13,
    forbid_test: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {destination}")
    if int(top_k) <= 0 or int(table_k) <= 0 or int(table_k) > int(top_k):
        raise ValueError("Require top_k > 0 and 0 < table_k <= top_k.")
    parsed_prefix_weights = tuple(float(value) for value in prefix_weights)
    if len(parsed_prefix_weights) != int(top_k):
        raise ValueError("prefix_weights length must equal top_k.")
    destination.mkdir(parents=True, exist_ok=True)

    matrix = load_calibration_matrix(
        matrix_run_dir,
        parent_limit=parent_limit,
        candidate_limit=candidate_limit,
        forbid_test=forbid_test,
    )
    if len(matrix.candidate_rows) < int(top_k):
        raise ValueError("Selected candidate subset is smaller than top_k.")
    thresholds = derive_thresholds(
        matrix.full_finite_distances,
        quantiles=threshold_quantiles,
        weights=threshold_weights,
        theta_star_quantile=theta_star_quantile,
        cost_cap_quantile=cost_cap_quantile,
    )
    chemistry = build_candidate_chemistry(
        matrix.candidate_rows,
        size_normalization_rows=matrix.full_candidate_rows,
    )
    coverage_redundancy_matrix = build_coverage_redundancy_matrix(
        matrix.distances,
        thresholds.levels,
    )
    variants = preregistered_variant_configs()
    _write_json(destination / "thresholds.json", thresholds.to_dict())
    variant_payload = _variant_config_payload(
        variants,
        top_k=top_k,
        table_k=table_k,
        prefix_weights=parsed_prefix_weights,
        local_swap_passes=local_swap_passes,
    )
    _write_json(destination / "variant_configs.json", variant_payload)

    comparison_rows: list[dict[str, Any]] = []
    selected_sequences: dict[str, list[str]] = {}
    for variant in variants:
        comparison, sequence = _run_one_variant(
            variant,
            matrix=matrix,
            chemistry=chemistry,
            thresholds=thresholds,
            prefix_weights=parsed_prefix_weights,
            top_k=int(top_k),
            table_k=int(table_k),
            local_swap_passes=int(local_swap_passes),
            coverage_redundancy_matrix=coverage_redundancy_matrix,
            output_dir=destination,
        )
        comparison_rows.append(comparison)
        selected_sequences[variant.name] = [
            matrix.candidate_ids[index] for index in sequence
        ]
    _write_csv(destination / "variant_comparison.csv", comparison_rows)
    chosen = choose_variant(comparison_rows)
    decision = {
        "selected_variant": str(chosen["variant"]),
        "decision_rule": [
            "max prefix_weighted_multi_threshold_utility",
            "max K=10 CCRCov@theta_star",
            "min K=10 fixed-denominator capped mean cost",
            "max K=20 weighted multi-threshold utility",
            "min K=20 coverage redundancy",
            "min K=20 structural redundancy",
            "variant name lexicographic",
        ],
        "selected_metrics": chosen,
        "ordered_variants": [
            str(row["variant"])
            for row in sorted(comparison_rows, key=variant_decision_sort_key)
        ],
        "test_used": False,
    }
    _write_json(destination / "calibration_decision.json", decision)

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "created_at": _utc_now(),
        "git_commit": _git_commit(repo_root),
        "matrix_run_dir": str(matrix.matrix_run_dir),
        "matrix_identity": {
            name: _file_identity(matrix.matrix_run_dir / name)
            for name in (
                "pair_matrix.jsonl",
                "selected_candidate_universe.jsonl",
                "summary.json",
                "run_manifest.json",
            )
        },
        "cohort": "calibration",
        "test_loaded": False,
        "config": {
            "top_k": int(top_k),
            "table_k": int(table_k),
            "threshold_quantiles": list(thresholds.requested_quantiles),
            "threshold_weights": list(thresholds.requested_weights),
            "theta_star_quantile": thresholds.theta_star_quantile,
            "cost_cap_quantile": thresholds.cost_cap_quantile,
            "prefix_weights": list(parsed_prefix_weights),
            "parent_limit": int(parent_limit),
            "candidate_limit": int(candidate_limit),
            "local_swap_passes": int(local_swap_passes),
            "seed": int(seed),
            "forbid_test": bool(forbid_test),
        },
        "run_complete": False,
    }
    _write_json(destination / "run_manifest.json", manifest)
    summary = {
        "matrix_parent_count_full": matrix.full_parent_count,
        "matrix_candidate_count_full": matrix.full_candidate_count,
        "matrix_pair_count_full": matrix.full_pair_count,
        "matrix_strict_flip_pair_count_full": matrix.full_strict_flip_pair_count,
        "selector_parent_count": len(matrix.parent_ids),
        "selector_candidate_count": len(matrix.candidate_rows),
        "top_k": int(top_k),
        "table_k": int(table_k),
        "theta_star": thresholds.theta_star,
        "cost_cap": thresholds.cost_cap,
        "finite_threshold_distance_count": thresholds.finite_distance_count,
        "selected_variant": decision["selected_variant"],
        "selected_sequences": selected_sequences,
        "test_loaded": False,
        "run_complete": True,
    }
    _write_json(destination / "summary.json", summary)
    manifest["run_complete"] = True
    manifest["completed_at"] = _utc_now()
    _write_json(destination / "run_manifest.json", manifest)
    _write_json(
        destination / "_RUN_COMPLETE.json",
        {
            "run_complete": True,
            "completed_at": _utc_now(),
            "selected_variant": decision["selected_variant"],
            "test_loaded": False,
        },
    )
    return summary


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    field: str,
    tolerance: float = 1e-10,
) -> None:
    if actual in (None, "") and expected is None:
        return
    actual_float = _finite_float(actual)
    expected_float = _finite_float(expected)
    if actual_float is None or expected_float is None:
        if actual != expected:
            raise AssertionError(f"{field} mismatch: {actual!r} != {expected!r}")
        return
    if not math.isclose(
        actual_float,
        expected_float,
        abs_tol=tolerance,
        rel_tol=tolerance,
    ):
        raise AssertionError(f"{field} mismatch: {actual_float} != {expected_float}")


def audit_mutagenicity_wnode_selector(
    *,
    run_dir: str | Path,
    matrix_run_dir: str | Path,
    expected_parent_count: int = 0,
    expected_candidate_count: int = 0,
    expected_top_k: int = 20,
    expected_table_k: int = 10,
    require_all_variants: bool = False,
    require_nested_prefix: bool = False,
    require_monotonic_coverage: bool = False,
    require_nonincreasing_capped_cost: bool = False,
    forbid_test: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    required = (
        "thresholds.json",
        "variant_configs.json",
        "variant_comparison.csv",
        "calibration_decision.json",
        "summary.json",
        "run_manifest.json",
        "_RUN_COMPLETE.json",
    )
    for filename in required:
        path = root / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise AssertionError(f"Missing selector artifact: {path}")
    manifest = _read_json(root / "run_manifest.json")
    summary = _read_json(root / "summary.json")
    complete = _read_json(root / "_RUN_COMPLETE.json")
    config = manifest.get("config") or {}
    if Path(str(manifest.get("matrix_run_dir"))).resolve() != Path(
        matrix_run_dir
    ).expanduser().resolve():
        raise AssertionError("Selector manifest matrix_run_dir mismatch.")
    if forbid_test:
        if (
            manifest.get("test_loaded") is not False
            or summary.get("test_loaded") is not False
            or complete.get("test_loaded") is not False
        ):
            raise AssertionError("Selector run does not prove test_loaded=false.")
        if _basename_has_test_token(Path(matrix_run_dir)):
            raise AssertionError("Selector references a forbidden test matrix path.")

    matrix = load_calibration_matrix(
        matrix_run_dir,
        parent_limit=int(config.get("parent_limit") or 0),
        candidate_limit=int(config.get("candidate_limit") or 0),
        forbid_test=forbid_test,
    )
    if expected_parent_count and len(matrix.parent_ids) != int(expected_parent_count):
        raise AssertionError("Selector parent count mismatch.")
    if expected_candidate_count and len(matrix.candidate_rows) != int(
        expected_candidate_count
    ):
        raise AssertionError("Selector candidate count mismatch.")
    top_k = int(config["top_k"])
    table_k = int(config["table_k"])
    if top_k != int(expected_top_k) or table_k != int(expected_table_k):
        raise AssertionError("Selector top_k/table_k mismatch.")
    prefix_weights = tuple(float(value) for value in config["prefix_weights"])
    thresholds = derive_thresholds(
        matrix.full_finite_distances,
        quantiles=tuple(float(value) for value in config["threshold_quantiles"]),
        weights=tuple(float(value) for value in config["threshold_weights"]),
        theta_star_quantile=float(config["theta_star_quantile"]),
        cost_cap_quantile=float(config["cost_cap_quantile"]),
    )
    persisted_thresholds = _read_json(root / "thresholds.json")
    expected_thresholds = thresholds.to_dict()
    if persisted_thresholds != expected_thresholds:
        raise AssertionError("thresholds.json is not derived from all calibration distances.")
    if not math.isclose(
        thresholds.theta_star,
        float(
            np.quantile(
                matrix.full_finite_distances,
                np.float64(0.30),
                method="linear",
            )
        ),
        abs_tol=FLOAT_TOLERANCE,
        rel_tol=0.0,
    ):
        raise AssertionError("theta_star is not exactly calibration q30.")

    chemistry = build_candidate_chemistry(
        matrix.candidate_rows,
        size_normalization_rows=matrix.full_candidate_rows,
    )
    coverage_redundancy_matrix = build_coverage_redundancy_matrix(
        matrix.distances,
        thresholds.levels,
    )
    variant_payload = _read_json(root / "variant_configs.json")
    variants = {
        name: VariantConfig(**payload)
        for name, payload in (variant_payload.get("variants") or {}).items()
    }
    if require_all_variants and set(variants) != set(VARIANT_NAMES):
        raise AssertionError("Selector run does not contain all preregistered variants.")

    comparison_rows = _read_csv(root / "variant_comparison.csv")
    recomputed_comparison: list[dict[str, Any]] = []
    candidate_index = matrix.candidate_index
    audited_variants: list[str] = []
    for variant_name in VARIANT_NAMES:
        if variant_name not in variants:
            continue
        variant = variants[variant_name]
        variant_dir = root / "variants" / variant_name
        selected_rows = _read_jsonl(variant_dir / "selected_sequence.jsonl")
        selected_ids = [str(row.get("candidate_id")) for row in selected_rows]
        if len(selected_ids) != top_k or len(set(selected_ids)) != top_k:
            raise AssertionError(f"{variant_name} is not exactly {top_k} unique candidates.")
        if any(candidate_id not in candidate_index for candidate_id in selected_ids):
            raise AssertionError(f"{variant_name} selected an unknown candidate.")
        sequence = [candidate_index[candidate_id] for candidate_id in selected_ids]
        top10 = _read_json(variant_dir / "selected_top10.json")
        top20 = _read_json(variant_dir / "selected_top20.json")
        if require_nested_prefix:
            if top10.get("candidate_ids") != selected_ids[:table_k]:
                raise AssertionError(f"{variant_name} top10 is not the top20 prefix.")
            if top20.get("candidate_ids") != selected_ids:
                raise AssertionError(f"{variant_name} top20 order mismatch.")
        persisted_metrics = _read_csv(variant_dir / "prefix_metrics.csv")
        if [int(row["k"]) for row in persisted_metrics] != list(
            range(1, top_k + 1)
        ):
            raise AssertionError(f"{variant_name} prefix K grid is incomplete.")
        recomputed_metrics, _ = compute_prefix_metrics(
            sequence,
            matrix=matrix,
            thresholds=thresholds,
            coverage_redundancy_matrix=coverage_redundancy_matrix,
            structural_similarity_matrix=chemistry.structural_similarity,
        )
        fields = tuple(recomputed_metrics[0])
        for actual, expected in zip(persisted_metrics, recomputed_metrics):
            for field in fields:
                _assert_close(
                    actual.get(field),
                    expected.get(field),
                    field=f"{variant_name}.{field}",
                )
        coverages = [
            float(row["ccrcov_theta_star"]) for row in recomputed_metrics
        ]
        capped_costs = [
            float(row["fixed_capped_mean_cost"]) for row in recomputed_metrics
        ]
        if require_monotonic_coverage and any(
            right + FLOAT_TOLERANCE < left
            for left, right in zip(coverages, coverages[1:])
        ):
            raise AssertionError(f"{variant_name} coverage decreases with K.")
        if require_nonincreasing_capped_cost and any(
            right > left + FLOAT_TOLERANCE
            for left, right in zip(capped_costs, capped_costs[1:])
        ):
            raise AssertionError(f"{variant_name} capped cost increases with K.")
        objective_fn = _objective_callable(
            matrix=matrix,
            thresholds=thresholds,
            prefix_weights=prefix_weights,
            variant=variant,
            coverage_redundancy_matrix=coverage_redundancy_matrix,
            structural_similarity_matrix=chemistry.structural_similarity,
            normalized_sizes=chemistry.normalized_sizes,
        )
        local_trace = _read_json(variant_dir / "local_search_trace.json")
        swap_trace = list(local_trace.get("swap_trace") or [])
        if swap_trace:
            start_ids = list(swap_trace[0].get("candidate_ids") or [])
            end_ids = list(swap_trace[-1].get("candidate_ids") or [])
            if any(candidate_id not in candidate_index for candidate_id in start_ids):
                raise AssertionError(f"{variant_name} swap start has unknown candidate.")
            if start_ids:
                start_objective = float(
                    objective_fn([candidate_index[value] for value in start_ids])
                )
                _assert_close(
                    local_trace.get("pre_swap_objective"),
                    start_objective,
                    field=f"{variant_name}.pre_swap_objective",
                )
            if end_ids != selected_ids:
                raise AssertionError(f"{variant_name} swap end sequence mismatch.")
            for event in swap_trace:
                if event.get("operation") == "swap_accept" and not (
                    float(event["objective_after"])
                    > float(event["objective_before"]) + FLOAT_TOLERANCE
                ):
                    raise AssertionError(
                        f"{variant_name} accepted a non-improving swap."
                    )
        if float(local_trace["post_swap_objective"]) + FLOAT_TOLERANCE < float(
            local_trace["pre_swap_objective"]
        ):
            raise AssertionError(f"{variant_name} local swap lowered objective.")
        comparison = build_variant_comparison_row(
            variant,
            recomputed_metrics,
            table_k=table_k,
            top_k=top_k,
            prefix_weights=prefix_weights,
            final_objective=float(objective_fn(sequence)),
        )
        recomputed_comparison.append(comparison)
        audited_variants.append(variant_name)

    comparison_by_name = {
        str(row["variant"]): row for row in comparison_rows
    }
    for expected in recomputed_comparison:
        actual = comparison_by_name.get(str(expected["variant"]))
        if actual is None:
            raise AssertionError(f"Missing comparison row: {expected['variant']}")
        for field, value in expected.items():
            _assert_close(
                actual.get(field),
                value,
                field=f"variant_comparison.{expected['variant']}.{field}",
            )
    expected_decision = choose_variant(recomputed_comparison)
    decision = _read_json(root / "calibration_decision.json")
    if decision.get("selected_variant") != expected_decision.get("variant"):
        raise AssertionError("Calibration decision violates preregistered ordering.")
    if (
        summary.get("run_complete") is not True
        or manifest.get("run_complete") is not True
        or complete.get("run_complete") is not True
    ):
        raise AssertionError("Selector run is not complete.")
    return {
        "audit_passed": True,
        "matrix_run_dir": str(matrix.matrix_run_dir),
        "parent_count": len(matrix.parent_ids),
        "candidate_count": len(matrix.candidate_rows),
        "top_k": top_k,
        "table_k": table_k,
        "audited_variants": audited_variants,
        "selected_variant": decision["selected_variant"],
        "theta_star": thresholds.theta_star,
        "cost_cap": thresholds.cost_cap,
        "finite_threshold_distance_count": thresholds.finite_distance_count,
        "test_loaded": False,
        "run_complete": True,
    }


__all__ = [
    "ChemistryData",
    "DEFAULT_COST_CAP_QUANTILE",
    "DEFAULT_PREFIX_WEIGHTS",
    "DEFAULT_THETA_STAR_QUANTILE",
    "DEFAULT_THRESHOLD_QUANTILES",
    "DEFAULT_THRESHOLD_WEIGHTS",
    "FLOAT_TOLERANCE",
    "MatrixData",
    "ThresholdBundle",
    "ThresholdLevel",
    "VARIANT_NAMES",
    "VariantConfig",
    "audit_mutagenicity_wnode_selector",
    "build_candidate_chemistry",
    "build_coverage_redundancy_matrix",
    "build_variant_comparison_row",
    "capped_cost_utility",
    "choose_variant",
    "compute_prefix_metrics",
    "derive_thresholds",
    "fixed_denominator_capped_cost",
    "greedy_select",
    "load_calibration_matrix",
    "local_swap_search",
    "morgan_tanimoto",
    "optimize_insertion_order",
    "preregistered_variant_configs",
    "run_mutagenicity_wnode_selector",
    "sequence_objective_components",
    "single_threshold_coverage",
    "variant_decision_sort_key",
    "weighted_coverage_jaccard",
    "weighted_multi_threshold_utility",
]
