"""Read-only audit registry for the four-method by four-dataset paper matrix.

This module intentionally does not run a model, select a candidate, fit a
threshold, or render a paper result.  It inventories already materialized
artifacts and promotes a cell only when its own frozen evidence proves the
shared evaluation contract.  Directory names are hints for inventory only;
they are never sufficient evidence for adoption.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASETS = ("AIDS", "Mutagenicity", "BACE", "TasteMolNet")
METHODS = ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC")
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
CF_MODE = "strict_flip"
K_PREFIXES = tuple(range(1, 21))
TABLE2_K = 10
SCHEMA_VERSION = "four_methods_four_datasets_registry_v1"
THRESHOLD_ARTIFACT_NAMES = {
    "AIDS": "threshold_contracts/aids.json",
    "Mutagenicity": "threshold_contracts/mutagenicity.json",
    "BACE": "threshold_contracts/bace.json",
    "TasteMolNet": "threshold_contracts/tastemolnet.json",
}


class CellStatus(str, Enum):
    """The only states accepted by the paper-matrix registry."""

    FROZEN_PASS = "FROZEN_PASS"
    ADOPTABLE_PASS = "ADOPTABLE_PASS"
    RUNNING = "RUNNING"
    READY = "READY"
    MISSING = "MISSING"
    STALE_ORACLE = "STALE_ORACLE"
    STALE_DATASET = "STALE_DATASET"
    STALE_SPLIT = "STALE_SPLIT"
    STALE_METRIC = "STALE_METRIC"
    INCOMPLETE = "INCOMPLETE"
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    BLOCKED_CODE = "BLOCKED_CODE"
    FAILED = "FAILED"


PASS_STATUSES = {CellStatus.FROZEN_PASS, CellStatus.ADOPTABLE_PASS}


METHOD_ALIASES = {
    "ours": "Ours",
    "our": "Ours",
    "gcfexplainer": "GCFExplainer",
    "gcfexplainerhivcsv": "GCFExplainer",
    "globalgce": "GlobalGCE",
    "comrecgc": "ComRecGC",
    "comrecgcadapteddeterministicchemrepair": "ComRecGC",
}
DATASET_ALIASES = {
    "aids": "AIDS",
    "aidshiv": "AIDS",
    "hiv": "AIDS",
    "hivquick": "AIDS",
    "mut": "Mutagenicity",
    "mutagenicity": "Mutagenicity",
    "bace": "BACE",
    "taste": "TasteMolNet",
    "tastemolnet": "TasteMolNet",
}


REQUIRED_ADOPTION_FILES = (
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "summary.json",
    "run_manifest.json",
    "final_artifact_audit.json",
)
FINAL_STANDARDIZED_FILES = (
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
    "destination_distribution.csv",
    "summary.json",
    "run_manifest.json",
    "oracle_manifest.json",
    "evaluation_manifest.json",
    "final_artifact_audit.json",
)
ANCHOR_NAMES = frozenset(
    REQUIRED_ADOPTION_FILES
    + FINAL_STANDARDIZED_FILES
    + (
        "pair_details.csv",
        "pair_matrix.jsonl",
        "candidate_pool.jsonl",
        "selected_subgraphs.csv",
        "selected_top20_for_eval.csv",
        "per_candidate_eval.jsonl",
        "combined_manifest.json",
        "combined_audit.json",
        "final_gate.json",
        "freeze_manifest.json",
        "_RUN_COMPLETE.json",
        "_FINALIZED.json",
        "PASS",
        "PASSED",
        "FAILED",
        "FAILED.json",
        "FAIL.json",
    )
)
RAW_EVIDENCE_NAMES = (
    "pair_details.csv",
    "pair_matrix.jsonl",
    "candidate_pool.jsonl",
    "selected_subgraphs.csv",
    "selected_top20_for_eval.csv",
    "per_candidate_eval.jsonl",
)
SKIP_SCAN_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        "node_embeddings",
        "wandb",
        ".cache",
        "cache",
    }
)


DEFAULT_ORACLES: dict[str, dict[str, Any]] = {
    "AIDS": {
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "num_classes": 2,
        "source_label": 1,
        "rf_oracle_used": True,
    },
    "Mutagenicity": {
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "num_classes": 2,
        "source_label": 1,
        "rf_oracle_used": True,
    },
    "BACE": {
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "num_classes": 2,
        "source_label": 1,
        "rf_oracle_used": False,
        "oracle_checkpoint": (
            "/autodl-fs/data/counterfactual-subgraph-runtime/outputs/"
            "gnn_oracles/bace/gine/seed7/"
            "calibrated-20260821T181039Z-97689"
        ),
    },
    "TasteMolNet": {
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "num_classes": 3,
        "source_label": 1,
        "source_label_name": "Sweet",
        "counterfactual_mode": "untargeted_strict_flip",
        "destination_labels": [0, 2],
        "rf_oracle_used": False,
    },
}


MATRIX_FIELDS = (
    "dataset",
    "method",
    "raw_output_root",
    "standardized_output_root",
    "generation_adoption_candidate",
    "generation_adoption_reason",
    "oracle_backend",
    "oracle_checkpoint",
    "oracle_hash",
    "dataset_hash",
    "split_hash",
    "distance_line",
    "molclr_checkpoint_hash",
    "cf_mode",
    "k_max",
    "table2_k",
    "threshold_config_hash",
    "status",
    "adoption_reason",
    "rerun_reason",
)
INVENTORY_FIELDS = (
    "scan_root",
    "candidate_root",
    "relative_path",
    "file_name",
    "size_bytes",
    "sha256",
    "hash_status",
    "dataset_hint",
    "method_hint",
)
STALE_FIELDS = (
    "candidate_root",
    "dataset",
    "method",
    "status",
    "reason_codes",
    "selected_for_cell",
)


@dataclass(frozen=True)
class AuditConfig:
    scan_roots: tuple[Path, ...]
    output_root: Path
    expectations: Mapping[str, Any] = field(default_factory=dict)
    explicit_cells: Mapping[str, Any] = field(default_factory=dict)
    taste_license_gate: Mapping[str, Any] | None = None
    max_hash_bytes: int = 64 * 1024 * 1024


@dataclass
class CandidateAudit:
    root: Path
    dataset: str | None
    method: str | None
    status: CellStatus
    reason_codes: list[str]
    row: dict[str, Any]
    artifact_hashes: dict[str, str]


@dataclass(frozen=True)
class ArtifactLayout:
    """Physical layout of one cell without inferring its scientific identity."""

    container_root: Path
    standardized_root: Path
    nested_standardized: bool


@dataclass(frozen=True)
class RegistryResult:
    matrix_rows: tuple[dict[str, Any], ...]
    inventory_rows: tuple[dict[str, Any], ...]
    stale_rows: tuple[dict[str, Any], ...]
    oracle_registry: dict[str, Any]
    evaluation_contract: dict[str, Any]
    threshold_contracts: dict[str, dict[str, Any]]
    matrix_complete_cells: int
    matrix_total_cells: int = 16


def _normalized_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def canonical_dataset(value: Any) -> str | None:
    return DATASET_ALIASES.get(_normalized_token(value))


def canonical_method(value: Any) -> str | None:
    return METHOD_ALIASES.get(_normalized_token(value))


def _canonical_oracle_backend(value: Any) -> str:
    token = _normalized_token(value)
    if token in {"rf", "randomforest"}:
        return "rf"
    if token in {"gnn", "graphneuralnetwork"}:
        return "gnn"
    return str(value or "").strip().lower()


def _canonical_classifier_family(value: Any) -> str:
    token = _normalized_token(value)
    if token in {"rf", "randomforest", "randomforestclassifier"}:
        return "random_forest"
    if token in {"gine", "gineconv"}:
        return "gine"
    return str(value or "").strip().lower()


def _layout(root: Path) -> ArtifactLayout:
    nested = root / "standardized"
    if nested.is_dir() and any(
        (nested / name).is_file()
        for name in ("run_manifest.json", "final_artifact_audit.json", "freeze_manifest.json")
    ):
        return ArtifactLayout(
            container_root=root,
            standardized_root=nested,
            nested_standardized=True,
        )
    return ArtifactLayout(
        container_root=root,
        standardized_root=root,
        nested_standardized=False,
    )


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _walk_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)


def _find_value(payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    key_set = set(keys)
    for payload in payloads:
        for key, value in _walk_values(payload):
            if key in key_set and value not in (None, "", [], {}):
                return value
    return None


def _extract_path(payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    value = _find_value(payloads, keys)
    if isinstance(value, Mapping):
        return str(value.get("path") or value.get("root") or "").strip()
    return str(value or "").strip()


def _extract_hash(payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    value = _find_value(payloads, keys)
    if isinstance(value, Mapping):
        value = value.get("sha256") or value.get("hash") or value.get("identity_sha256")
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def _collect_bool_values(
    payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> set[bool]:
    key_set = set(keys)
    values: set[bool] = set()
    for payload in payloads:
        for key, raw in _walk_values(payload):
            if key not in key_set:
                continue
            if isinstance(raw, bool):
                values.add(raw)
            elif isinstance(raw, (int, float)):
                values.add(bool(raw))
            else:
                token = str(raw or "").strip().lower()
                if token in {"true", "1", "yes", "pass", "passed"}:
                    values.add(True)
                elif token in {"false", "0", "no", "fail", "failed"}:
                    values.add(False)
    return values


def _collect_scalar_values(
    payloads: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    canonicalizer: Any | None = None,
) -> set[str]:
    key_set = set(keys)
    values: set[str] = set()
    for payload in payloads:
        for key, raw in _walk_values(payload):
            if key not in key_set or raw in (None, "", [], {}):
                continue
            normalized = canonicalizer(raw) if canonicalizer else str(raw).strip()
            if normalized not in (None, ""):
                values.add(str(normalized))
    return values


def _collect_hash_values(
    payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> set[str]:
    values = _collect_scalar_values(payloads, keys)
    return {
        value.lower()
        for value in values
        if re.fullmatch(r"[0-9a-fA-F]{64}", value)
    }


def _extract_int(payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> int | None:
    value = _find_value(payloads, keys)
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _candidate_identity(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None, bool]:
    dataset_values = {
        value
        for key, raw in (item for payload in payloads for item in _walk_values(payload))
        if key in {"dataset", "dataset_name", "dataset_key"}
        and (value := canonical_dataset(raw)) is not None
    }
    method_values = {
        value
        for key, raw in (item for payload in payloads for item in _walk_values(payload))
        if key in {"method", "method_name", "display_method", "source_method"}
        and (value := canonical_method(raw)) is not None
    }
    dataset = next(iter(dataset_values)) if len(dataset_values) == 1 else None
    method = next(iter(method_values)) if len(method_values) == 1 else None
    return dataset, method, bool(dataset is not None and method is not None)


def _table2_path(root: Path, method: str | None) -> Path | None:
    if method is not None:
        slug = _normalized_token(method)
        exact = root / f"table2_{slug}_k10.csv"
        if exact.is_file():
            return exact
        if method == "ComRecGC":
            legacy = root / "table2_comrecgc_k10.csv"
            if legacy.is_file():
                return legacy
    generic = root / "table2.csv"
    if generic.is_file():
        return generic
    candidates = sorted(root.glob("table2_*_k10.csv"))
    return candidates[0] if len(candidates) == 1 else None


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _validate_rate(raw: Any, *, field_name: str, path: Path) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field_name}={raw!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{path}: {field_name} must be finite in [0,1]")
    return value


def _first_field(fields: Sequence[str], aliases: Sequence[str]) -> str | None:
    return next((name for name in aliases if name in fields), None)


def _validate_finite(raw: Any, *, field_name: str, path: Path) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {field_name}={raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {field_name} must be finite")
    return value


def _validate_standardized_csvs(
    root: Path, method: str | None
) -> tuple[dict[str, str], list[str], int | None, int | None, str]:
    reasons: list[str] = []
    hashes: dict[str, str] = {}
    figure3 = root / "figure3_coverage_vs_k.csv"
    figure4 = root / "figure4_coverage_vs_threshold.csv"
    table2 = _table2_path(root, method)
    if table2 is None:
        reasons.append("MISSING_TABLE2_K10")
        return hashes, reasons, None, None, ""
    k_max: int | None = None
    table2_k: int | None = None
    threshold_hash = ""
    try:
        fields3, rows3 = _csv_rows(figure3)
        coverage_field = _first_field(
            fields3, ("coverage", "ccrcov", "close_cf_coverage")
        )
        cost_field = _first_field(
            fields3,
            (
                "cost",
                "conditional_median_cost",
                "conditional_mean_cost",
                "fixed_capped_mean_cost",
            ),
        )
        if not {"method", "k"}.issubset(fields3) or not coverage_field or not cost_field:
            reasons.append("FIGURE3_SCHEMA_MISMATCH")
        else:
            ks = [int(row["k"]) for row in rows3]
            if ks != list(K_PREFIXES):
                reasons.append("FIGURE3_K_GRID_NOT_1_TO_20")
            else:
                k_max = max(ks)
            coverages = [
                _validate_rate(
                    row[coverage_field], field_name=coverage_field, path=figure3
                )
                for row in rows3
            ]
            for row in rows3:
                _validate_finite(row[cost_field], field_name=cost_field, path=figure3)
            if any(right + 1e-12 < left for left, right in zip(coverages, coverages[1:])):
                reasons.append("FIGURE3_COVERAGE_NOT_MONOTONE")
            if method is not None and any(
                canonical_method(row["method"]) != method for row in rows3
            ):
                reasons.append("FIGURE3_METHOD_IDENTITY_MISMATCH")
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        reasons.append(f"FIGURE3_INVALID:{type(exc).__name__}")
    try:
        fields4, rows4 = _csv_rows(figure4)
        coverage_field = _first_field(
            fields4, ("coverage", "ccrcov", "close_cf_coverage")
        )
        if not {"method", "threshold"}.issubset(fields4) or not coverage_field:
            reasons.append("FIGURE4_SCHEMA_MISMATCH")
        else:
            thresholds = [float(row["threshold"]) for row in rows4]
            if not thresholds or any(not math.isfinite(value) for value in thresholds):
                reasons.append("FIGURE4_THRESHOLD_GRID_INVALID")
            elif any(right < left for left, right in zip(thresholds, thresholds[1:])):
                reasons.append("FIGURE4_THRESHOLD_GRID_NOT_MONOTONE")
            else:
                threshold_hash = stable_json_sha256(thresholds)
            coverages = [
                _validate_rate(
                    row[coverage_field], field_name=coverage_field, path=figure4
                )
                for row in rows4
            ]
            if any(right + 1e-12 < left for left, right in zip(coverages, coverages[1:])):
                reasons.append("FIGURE4_COVERAGE_NOT_MONOTONE")
            if method is not None and any(
                canonical_method(row["method"]) != method for row in rows4
            ):
                reasons.append("FIGURE4_METHOD_IDENTITY_MISMATCH")
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        reasons.append(f"FIGURE4_INVALID:{type(exc).__name__}")
    try:
        fields2, rows2 = _csv_rows(table2)
        coverage_field = _first_field(
            fields2, ("coverage", "ccrcov", "close_cf_coverage")
        )
        cost_field = _first_field(
            fields2,
            (
                "cost",
                "conditional_median_cost",
                "conditional_mean_cost",
                "fixed_capped_mean_cost",
            ),
        )
        if not {"method", "k"}.issubset(fields2) or not coverage_field or not cost_field:
            reasons.append("TABLE2_SCHEMA_MISMATCH")
        elif len(rows2) != 1:
            reasons.append("TABLE2_ROW_COUNT_NOT_ONE")
        else:
            table2_k = int(rows2[0]["k"])
            if table2_k != TABLE2_K:
                reasons.append("TABLE2_K_NOT_10")
            _validate_rate(
                rows2[0][coverage_field], field_name=coverage_field, path=table2
            )
            _validate_finite(rows2[0][cost_field], field_name=cost_field, path=table2)
            if method is not None and canonical_method(rows2[0]["method"]) != method:
                reasons.append("TABLE2_METHOD_IDENTITY_MISMATCH")
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        reasons.append(f"TABLE2_INVALID:{type(exc).__name__}")
    for path in (figure3, figure4, table2):
        if path.is_file():
            hashes[path.name] = sha256_file(path)
    return hashes, reasons, k_max, table2_k, threshold_hash


def _path_exists(path_text: str, *, relative_to: Path) -> bool:
    if not path_text:
        return False
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    try:
        return path.exists() and (path.is_file() or any(path.iterdir()))
    except OSError:
        return False


def _raw_root(payloads: Sequence[Mapping[str, Any]], root: Path) -> str:
    recorded = _extract_path(
        payloads,
        (
            "raw_output_root",
            "raw_root",
            "source_run_dir",
            "source_output_root",
            "source_generation_root",
            "evaluation_run_dir",
            "run_dir",
        ),
    )
    if recorded and _path_exists(recorded, relative_to=root):
        return recorded
    if any((root / name).is_file() for name in RAW_EVIDENCE_NAMES):
        return str(root)
    raw = root / "raw"
    if raw.is_dir() and any(raw.iterdir()):
        return str(raw)
    return recorded


def _process_is_live(payloads: Sequence[Mapping[str, Any]]) -> bool:
    states = {
        value.upper()
        for value in _collect_scalar_values(
            payloads, ("state", "status", "run_state")
        )
    }
    if states.intersection({"RUNNING", "STARTING"}):
        return True
    pid = _extract_int(payloads, ("worker_pid", "pid", "process_id"))
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError):
        return False
    expected_ticks = str(
        _find_value(payloads, ("process_start_ticks", "pid_start_ticks", "start_ticks"))
        or ""
    ).strip()
    stat_path = Path("/proc") / str(pid) / "stat"
    if expected_ticks and stat_path.is_file():
        try:
            current_ticks = stat_path.read_text(encoding="utf-8").split()[21]
        except (OSError, IndexError):
            return False
        return current_ticks == expected_ticks
    return True


def _license_pass(gate: Mapping[str, Any] | None) -> bool:
    if not gate:
        return False
    status = str(gate.get("status") or gate.get("state") or "").strip().upper()
    passed = gate.get("passed") is True and status in {"PASS", "LICENSE_PASS"}
    basis = str(
        gate.get("license_basis")
        or gate.get("reuse_basis")
        or gate.get("approval_file")
        or gate.get("evidence_file")
        or ""
    ).strip()
    return bool(passed and basis)


def build_threshold_contracts(
    expectations: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Materialize thresholds from calibration or an explicit frozen protocol."""

    dataset_expectations = dict((expectations or {}).get("datasets") or {})
    contracts: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        source = dict(dataset_expectations.get(dataset) or {})
        base = {
            "schema_version": "four_by_four_frozen_threshold_contract_v1",
            "dataset": dataset,
            "distance_line": DISTANCE_LINE,
            "cf_mode": CF_MODE,
            "test_used_for_selection": False,
        }
        values = source.get("thresholds")
        theta_star = source.get("theta_star")
        cost_cap = source.get("cost_cap")
        threshold_source = str(source.get("threshold_source") or "").strip()
        selection_split = str(
            source.get("threshold_source_split")
            or source.get("selection_split")
            or ""
        ).strip().lower()
        source_sha = str(source.get("threshold_config_hash") or "").strip().lower()
        test_used = source.get("test_used_for_selection")
        if values in (None, "") and theta_star in (None, "") and cost_cap in (None, ""):
            contracts[dataset] = {
                **base,
                "status": "MISSING_NOT_INFERRED",
                "reason": (
                    "no explicit calibration-frozen threshold contract was supplied; "
                    "paper/test outputs are never mined for thresholds"
                ),
            }
            continue
        errors: list[str] = []
        parsed: list[float] = []
        if not isinstance(values, list) or not values:
            errors.append("thresholds_must_be_nonempty_list")
        else:
            try:
                parsed = [float(value) for value in values]
            except (TypeError, ValueError):
                errors.append("thresholds_must_be_numeric")
            if parsed and (
                any(not math.isfinite(value) or value < 0.0 for value in parsed)
                or any(right <= left for left, right in zip(parsed, parsed[1:]))
            ):
                errors.append("thresholds_must_be_finite_nonnegative_strictly_increasing")
        try:
            parsed_theta = float(theta_star)
            parsed_cap = float(cost_cap)
        except (TypeError, ValueError):
            parsed_theta = math.nan
            parsed_cap = math.nan
            errors.append("theta_star_and_cost_cap_must_be_numeric")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (parsed_theta, parsed_cap)
        ):
            errors.append("theta_star_and_cost_cap_must_be_finite_nonnegative")
        elif parsed and (parsed_theta > parsed[-1] or parsed_cap < parsed_theta):
            errors.append("theta_star_or_cost_cap_outside_preregistered_contract")
        if selection_split not in {
            "calibration",
            "frozen_calibration",
            "legacy_frozen_calibration",
            "frozen_protocol",
            "existing_frozen_protocol",
            "legacy_frozen_protocol",
        }:
            errors.append("threshold_source_split_must_be_calibration_or_frozen_protocol")
        if test_used is not False:
            errors.append("test_selection_exclusion_not_explicit_false")
        if not threshold_source:
            errors.append("threshold_source_missing")
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            errors.append("threshold_config_hash_missing_or_invalid")
        if errors:
            contracts[dataset] = {
                **base,
                "status": "INVALID_FAIL_CLOSED",
                "errors": sorted(set(errors)),
            }
            continue
        contracts[dataset] = {
            **base,
            "status": "PASS",
            "thresholds": parsed,
            "theta_star": parsed_theta,
            "cost_cap": parsed_cap,
            "threshold_source": threshold_source,
            "threshold_source_split": selection_split,
            "threshold_config_hash": source_sha,
        }
    return contracts


def build_evaluation_contract(expectations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    datasets = dict((expectations or {}).get("datasets") or {})
    return {
        "schema_version": "four_methods_four_datasets_evaluation_contract_v1",
        "distance_line": DISTANCE_LINE,
        "distance_definition": "MolCLR-Node-Wasserstein / WNode",
        "cf_mode": CF_MODE,
        "strict_flip_definition": "pred_before == source_label and pred_after != source_label",
        "k_prefixes": list(K_PREFIXES),
        "table2_k": TABLE2_K,
        "figure3": "coverage/cost vs K",
        "figure4": "coverage vs threshold; empirical points only; no smoothing",
        "threshold_policy": (
            "one calibration-frozen threshold grid and theta_star per dataset, "
            "shared by all four methods"
        ),
        "threshold_artifacts": dict(THRESHOLD_ARTIFACT_NAMES),
        "threshold_artifact_policy": (
            "per-dataset evaluator-ready JSON is emitted only from explicit "
            "calibration-frozen expectations; missing values are not inferred from test"
        ),
        "selection_policy": "candidate/rule order is frozen before held-out test access",
        "methods": {
            "Ours": {"native_action": "hard_delete_selected_subgraph"},
            "GCFExplainer": {"native_action": "nearest_full_counterfactual_graph"},
            "GlobalGCE": {"native_action": "apply_lhs_to_rhs_transformation_rule"},
            "ComRecGC": {"native_action": "validated_lineage_transition_or_recourse"},
        },
        "required_adoption_files": [
            *REQUIRED_ADOPTION_FILES,
            "table2_<method>_k10.csv",
        ],
        "final_standardized_files": [
            *FINAL_STANDARDIZED_FILES,
            "table2_<method>_k10.csv",
        ],
        "required_metrics": [
            "SuppCov",
            "CCRCov",
            "cost",
            "CFDrop",
            "FlipRate",
            "StructRed",
            "CovRed",
            "ValidRate",
            "AvgSize",
            "applicable_rate",
        ],
        "taste_additional_fields": [
            "destination_label",
            "Sweet_to_Bitter_count",
            "Sweet_to_Bitter_rate",
            "Sweet_to_Tasteless_count",
            "Sweet_to_Tasteless_rate",
            "per_rule_destination_distribution",
        ],
        "dataset_contracts": datasets,
        "no_test_selection": True,
        "no_numeric_imputation": True,
        "final_export_gate": {
            "marker": "matrix_status.json",
            "required_field": "all_cells_complete",
            "required_value": True,
            "required_complete_cells": 16,
            "blocked_or_missing_cells_rendered_as_zero": False,
        },
    }


def _expected_oracle(dataset: str, expectations: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(DEFAULT_ORACLES[dataset])
    dataset_expectation = dict((expectations.get("datasets") or {}).get(dataset) or {})
    for key in (
        "oracle_backend",
        "classifier_family",
        "oracle_checkpoint",
        "oracle_hash",
        "dataset_hash",
        "split_hash",
        "molclr_checkpoint_hash",
        "threshold_config_hash",
        "num_classes",
        "source_label",
        "rf_oracle_used",
    ):
        if dataset_expectation.get(key) not in (None, ""):
            result[key] = dataset_expectation[key]
    return result


def _load_layout_payloads(
    layout: ArtifactLayout,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    relative_names = [
        "summary.json",
        "run_manifest.json",
        "final_artifact_audit.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "freeze_manifest.json",
        "_FINALIZED.json",
    ]
    paths: list[tuple[str, Path, bool]] = [
        (f"standardized/{name}" if layout.nested_standardized else name,
         layout.standardized_root / name,
         name in {"summary.json", "run_manifest.json", "final_artifact_audit.json"})
        for name in relative_names
    ]
    if layout.nested_standardized:
        paths.extend(
            (
                name,
                layout.container_root / name,
                name in {"run_manifest.json", "final_gate.json", "_RUN_COMPLETE.json"},
            )
            for name in (
                "run_manifest.json",
                "final_gate.json",
                "_RUN_COMPLETE.json",
                "generation_adoption_manifest.json",
                "stage_state.json",
            )
        )
    for label, path, required in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            if required:
                reasons.append(f"MISSING_OR_EMPTY:{label}")
            continue
        try:
            payload = _read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            reasons.append(f"INVALID_JSON:{label}:{type(exc).__name__}")
            continue
        payloads.append(payload)
        payload_by_name[label] = payload
    return payloads, payload_by_name, reasons


def _freeze_inventory_reasons(
    layout: ArtifactLayout,
    payload_by_name: Mapping[str, Mapping[str, Any]],
    *,
    max_hash_bytes: int,
) -> list[str]:
    if not layout.nested_standardized:
        return []
    freeze = payload_by_name.get("standardized/freeze_manifest.json") or {}
    files = freeze.get("files")
    if not isinstance(files, Mapping) or not files:
        return ["FROZEN_FILE_INVENTORY_MISSING"]
    reasons: list[str] = []
    required_names = set(REQUIRED_ADOPTION_FILES)
    table2 = _table2_path(layout.standardized_root, "ComRecGC")
    if table2 is not None:
        required_names.add(table2.name)
    for name in sorted(required_names):
        metadata = files.get(name)
        target = layout.standardized_root / name
        if not isinstance(metadata, Mapping):
            reasons.append(f"FROZEN_FILE_NOT_DECLARED:{name}")
            continue
        if not target.is_file():
            reasons.append(f"FROZEN_FILE_MISSING:{name}")
            continue
        try:
            actual_bytes = target.stat().st_size
        except OSError:
            reasons.append(f"FROZEN_FILE_STAT_FAILED:{name}")
            continue
        try:
            expected_bytes = int(metadata.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
        if actual_bytes != expected_bytes:
            reasons.append(f"FROZEN_FILE_SIZE_MISMATCH:{name}")
        claimed_sha = str(metadata.get("sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", claimed_sha):
            reasons.append(f"FROZEN_FILE_SHA256_MISSING:{name}")
        elif actual_bytes <= max_hash_bytes:
            try:
                if sha256_file(target) != claimed_sha:
                    reasons.append(f"FROZEN_FILE_SHA256_MISMATCH:{name}")
            except OSError:
                reasons.append(f"FROZEN_FILE_HASH_FAILED:{name}")
    return reasons


def _nested_gate_reasons(
    layout: ArtifactLayout,
    payload_by_name: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not layout.nested_standardized:
        return []
    reasons: list[str] = []
    final_gate = payload_by_name.get("final_gate.json") or {}
    complete = payload_by_name.get("_RUN_COMPLETE.json") or {}
    finalized = payload_by_name.get("standardized/_FINALIZED.json") or {}
    if str(final_gate.get("status") or "").upper() != "PASS":
        reasons.append("CONTAINER_FINAL_GATE_NOT_PASS")
    if complete.get("run_complete") is not True or str(
        complete.get("status") or ""
    ).upper() != "PASS":
        reasons.append("CONTAINER_RUN_COMPLETE_NOT_TRUE")
    if finalized.get("finalized") is not True or finalized.get("gate_passed") is not True:
        reasons.append("STANDARDIZED_FREEZE_NOT_FINALIZED")
    marker = layout.container_root / "PASS"
    try:
        marker_ok = marker.is_file() and marker.read_text(encoding="utf-8").strip() == "PASS"
    except OSError:
        marker_ok = False
    if not marker_ok:
        reasons.append("CONTAINER_PASS_MARKER_MISSING_OR_INVALID")
    recorded_root = str(final_gate.get("standardized_output_root") or "").strip()
    try:
        recorded_root_ok = (
            Path(recorded_root).expanduser().resolve() == layout.standardized_root.resolve()
        )
    except OSError:
        recorded_root_ok = False
    if not recorded_root_ok:
        reasons.append("CONTAINER_STANDARDIZED_ROOT_MISMATCH")
    for field_name, relative_path in (
        ("standardized_run_manifest_sha256", "run_manifest.json"),
        ("freeze_manifest_sha256", "freeze_manifest.json"),
    ):
        claimed = str(final_gate.get(field_name) or "").strip().lower()
        target = layout.standardized_root / relative_path
        if not re.fullmatch(r"[0-9a-f]{64}", claimed) or not target.is_file():
            reasons.append(f"CONTAINER_{field_name.upper()}_MISSING")
        else:
            try:
                if sha256_file(target) != claimed:
                    reasons.append(f"CONTAINER_{field_name.upper()}_MISMATCH")
            except OSError:
                reasons.append(f"CONTAINER_{field_name.upper()}_HASH_FAILED")
    return reasons


def _audit_candidate(
    root: Path,
    *,
    expectations: Mapping[str, Any],
    max_hash_bytes: int,
    explicit_dataset: str | None = None,
    explicit_method: str | None = None,
) -> CandidateAudit:
    layout = _layout(root)
    payloads, payload_by_name, reasons = _load_layout_payloads(layout)
    inferred_dataset, inferred_method, manifest_identity = _candidate_identity(payloads)
    dataset = explicit_dataset or inferred_dataset
    method = explicit_method or inferred_method
    explicit_identity = bool(explicit_dataset and explicit_method)
    if explicit_dataset and inferred_dataset and inferred_dataset != explicit_dataset:
        reasons.append("EXPLICIT_DATASET_DISAGREES_WITH_ARTIFACT")
    if explicit_method and inferred_method and inferred_method != explicit_method:
        reasons.append("EXPLICIT_METHOD_DISAGREES_WITH_ARTIFACT")
    if not manifest_identity:
        reasons.append("ARTIFACT_IDENTITY_NOT_UNIQUE")
    standardized_root = layout.standardized_root
    for name in ("figure3_coverage_vs_k.csv", "figure4_coverage_vs_threshold.csv"):
        path = standardized_root / name
        if not path.is_file() or path.stat().st_size <= 0:
            reasons.append(f"MISSING_OR_EMPTY:{name}")
    artifact_hashes, csv_reasons, k_max, table2_k, derived_threshold_hash = (
        _validate_standardized_csvs(standardized_root, method)
        if not any(reason.startswith("MISSING_OR_EMPTY:figure") for reason in reasons)
        else ({}, [], None, None, "")
    )
    reasons.extend(csv_reasons)
    for name in ("summary.json", "run_manifest.json", "final_artifact_audit.json"):
        path = standardized_root / name
        if path.is_file() and path.stat().st_size > 0:
            artifact_hashes[name] = sha256_file(path)
    reasons.extend(
        _freeze_inventory_reasons(
            layout,
            payload_by_name,
            max_hash_bytes=max_hash_bytes,
        )
    )
    reasons.extend(_nested_gate_reasons(layout, payload_by_name))
    raw_root = _raw_root(payloads, standardized_root)
    raw_root_exists = bool(raw_root and _path_exists(raw_root, relative_to=standardized_root))
    raw_complete_values = _collect_bool_values(
        payloads,
        (
            "raw_output_complete",
            "raw_artifacts_complete",
            "source_artifacts_complete",
            "generation_adopted",
        ),
    )
    raw_complete_proven = raw_complete_values == {True}
    generation_adoption_candidate = bool(
        raw_root_exists
        and (manifest_identity or explicit_identity)
        and not _process_is_live(payloads)
        and not (layout.container_root / "FAILED").exists()
        and not (layout.container_root / "FAILED.json").exists()
        and not (layout.container_root / "FAIL.json").exists()
    )
    if not raw_root_exists:
        reasons.append("RAW_OUTPUT_NOT_PROVEN_COMPLETE")
    if not raw_complete_proven:
        reasons.append("RAW_OUTPUT_COMPLETENESS_GATE_NOT_TRUE")
    audit_label = (
        "standardized/final_artifact_audit.json"
        if layout.nested_standardized
        else "final_artifact_audit.json"
    )
    audit = payload_by_name.get(audit_label, {})
    audit_pass_values = {
        value
        for key in ("passed", "audit_passed")
        if (value := audit.get(key)) is not None
    }
    if audit_pass_values != {True}:
        reasons.append("FINAL_ARTIFACT_AUDIT_NOT_PASS")
    if any(
        (layout.container_root / name).exists()
        for name in ("FAILED", "FAILED.json", "FAIL.json")
    ):
        reasons.append("FAILED_SENTINEL_PRESENT")
    if _process_is_live(payloads):
        status = CellStatus.RUNNING
    else:
        status = CellStatus.INCOMPLETE

    oracle_backend_keys = ("oracle_backend", "teacher_backend")
    classifier_family_keys = ("classifier_family", "classifier_type")
    oracle_hash_keys = (
        "oracle_hash",
        "oracle_checkpoint_hash",
        "gnn_checkpoint_hash",
        "teacher_sha256",
        "teacher_hash",
    )
    dataset_hash_keys = (
        "dataset_hash",
        "dataset_sha256",
        "processed_dataset_sha256",
        "dataset_csv_sha256",
    )
    test_cohort_hash_keys = (
        "test_parent_ids_sha256",
        "test_cohort_hash",
        "parent_ids_sha256",
    )
    split_manifest_hash_keys = ("test_split_hash", "split_hash")
    molclr_hash_keys = (
        "molclr_checkpoint_hash",
        "molclr_checkpoint_sha256",
        "distance_encoder_hash",
    )
    threshold_hash_keys = (
        "threshold_config_hash",
        "thresholds_json_sha256",
        "thresholds_sha256",
    )
    oracle_backend = _canonical_oracle_backend(
        _find_value(payloads, oracle_backend_keys)
    )
    classifier_family = _canonical_classifier_family(
        _find_value(payloads, classifier_family_keys)
    )
    oracle_checkpoint = _extract_path(
        payloads,
        ("oracle_checkpoint", "gnn_checkpoint", "teacher_path", "oracle_path"),
    )
    oracle_hash = _extract_hash(payloads, oracle_hash_keys)
    dataset_hash = _extract_hash(payloads, dataset_hash_keys)
    split_hash = _extract_hash(payloads, test_cohort_hash_keys) or _extract_hash(
        payloads, split_manifest_hash_keys
    )
    distance_line = str(
        _find_value(payloads, ("distance_line", "distance_label")) or ""
    ).strip()
    molclr_hash = _extract_hash(payloads, molclr_hash_keys)
    cf_mode = str(_find_value(payloads, ("cf_mode", "CF_MODE")) or "").strip().lower()
    recorded_threshold_hash = _extract_hash(payloads, threshold_hash_keys)
    threshold_hash = recorded_threshold_hash or derived_threshold_hash
    rf_used_values = _collect_bool_values(
        payloads, ("rf_oracle_used", "RF_ORACLE_USED")
    )
    test_selection_values = _collect_bool_values(
        payloads, ("test_used_for_selection", "selection_used_test")
    )
    threshold_test_values = _collect_bool_values(
        payloads, ("threshold_fitted_on_test", "threshold_fit_on_test")
    )
    frozen_values = _collect_bool_values(
        payloads, ("frozen", "artifacts_frozen", "finalized")
    )
    if len(frozen_values) > 1:
        reasons.append("FROZEN_EVIDENCE_CONFLICT")
    observed_test_cohort_hashes = _collect_hash_values(
        payloads, test_cohort_hash_keys
    )

    evidence_conflicts = {
        "ORACLE_BACKEND_EVIDENCE_CONFLICT": _collect_scalar_values(
            payloads, oracle_backend_keys, canonicalizer=_canonical_oracle_backend
        ),
        "CLASSIFIER_FAMILY_EVIDENCE_CONFLICT": _collect_scalar_values(
            payloads,
            classifier_family_keys,
            canonicalizer=_canonical_classifier_family,
        ),
        "ORACLE_HASH_EVIDENCE_CONFLICT": _collect_hash_values(
            payloads, oracle_hash_keys
        ),
        "DATASET_HASH_EVIDENCE_CONFLICT": _collect_hash_values(
            payloads, dataset_hash_keys
        ),
        "SPLIT_HASH_EVIDENCE_CONFLICT": _collect_hash_values(
            payloads,
            test_cohort_hash_keys
            if observed_test_cohort_hashes
            else split_manifest_hash_keys,
        ),
        "MOLCLR_HASH_EVIDENCE_CONFLICT": _collect_hash_values(
            payloads, molclr_hash_keys
        ),
        "THRESHOLD_HASH_EVIDENCE_CONFLICT": _collect_hash_values(
            payloads, threshold_hash_keys
        ),
        "DISTANCE_LINE_EVIDENCE_CONFLICT": _collect_scalar_values(
            payloads, ("distance_line", "distance_label")
        ),
        "CF_MODE_EVIDENCE_CONFLICT": _collect_scalar_values(
            payloads, ("cf_mode", "CF_MODE"), canonicalizer=_normalized_token
        ),
    }
    reasons.extend(
        reason for reason, values in evidence_conflicts.items() if len(values) > 1
    )

    if not dataset_hash:
        reasons.append("MISSING_DATASET_HASH")
    if not split_hash:
        reasons.append("MISSING_TEST_SPLIT_HASH")
    if not oracle_backend:
        reasons.append("MISSING_ORACLE_BACKEND")
    if not classifier_family:
        reasons.append("MISSING_CLASSIFIER_FAMILY")
    if not oracle_checkpoint:
        reasons.append("MISSING_ORACLE_CHECKPOINT")
    if not oracle_hash:
        reasons.append("MISSING_ORACLE_HASH")
    if not molclr_hash:
        reasons.append("MISSING_MOLCLR_CHECKPOINT_HASH")
    if not threshold_hash:
        reasons.append("MISSING_THRESHOLD_CONFIG_HASH")
    if test_selection_values != {False}:
        reasons.append("TEST_SELECTION_EXCLUSION_NOT_PROVEN")
    if threshold_test_values != {False}:
        reasons.append("TEST_THRESHOLD_EXCLUSION_NOT_PROVEN")
    if not distance_line:
        reasons.append("MISSING_DISTANCE_LINE")
    elif distance_line != DISTANCE_LINE:
        reasons.append("DISTANCE_LINE_MISMATCH")
    if not cf_mode:
        reasons.append("MISSING_CF_MODE")
    elif cf_mode != CF_MODE:
        reasons.append("CF_MODE_MISMATCH")
    if k_max is None:
        reasons.append("MISSING_K_PREFIX_GRID")
    elif k_max != 20:
        reasons.append("K_MAX_NOT_20")
    if table2_k is None:
        reasons.append("MISSING_TABLE2_K")
    elif table2_k != 10:
        reasons.append("TABLE2_K_NOT_10")

    expected = _expected_oracle(dataset, expectations) if dataset in DATASETS else {}
    if dataset in {"AIDS", "Mutagenicity"}:
        if (oracle_backend and oracle_backend != "rf") or (
            classifier_family
            and classifier_family not in {"random_forest", "randomforest", "rf"}
        ):
            reasons.append("RF_ORACLE_CONTRACT_MISMATCH")
    elif dataset in {"BACE", "TasteMolNet"}:
        if (oracle_backend and oracle_backend != "gnn") or (
            classifier_family and classifier_family != "gine"
        ) or True in rf_used_values:
            reasons.append("GNN_ORACLE_CONTRACT_MISMATCH")
        if rf_used_values != {False}:
            reasons.append("MISSING_RF_ORACLE_USED_GUARD")
    for field_name, actual in (
        ("oracle_backend", oracle_backend),
        ("classifier_family", classifier_family),
        ("oracle_checkpoint", oracle_checkpoint),
        ("oracle_hash", oracle_hash),
        ("dataset_hash", dataset_hash),
        ("split_hash", split_hash),
        ("molclr_checkpoint_hash", molclr_hash),
        ("threshold_config_hash", threshold_hash),
    ):
        expected_value = expected.get(field_name)
        if field_name == "oracle_backend":
            expected_value = _canonical_oracle_backend(expected_value)
        elif field_name == "classifier_family":
            expected_value = _canonical_classifier_family(expected_value)
        if (
            expected_value not in (None, "")
            and actual not in (None, "")
            and str(actual) != str(expected_value)
        ):
            reasons.append(f"EXPECTED_{field_name.upper()}_MISMATCH")

    if status is not CellStatus.RUNNING:
        if "FAILED_SENTINEL_PRESENT" in reasons:
            status = CellStatus.FAILED
        elif any(
            reason in {"RF_ORACLE_CONTRACT_MISMATCH", "GNN_ORACLE_CONTRACT_MISMATCH"}
            or reason.startswith("EXPECTED_ORACLE_")
            or reason
            in {
                "ORACLE_BACKEND_EVIDENCE_CONFLICT",
                "CLASSIFIER_FAMILY_EVIDENCE_CONFLICT",
                "ORACLE_HASH_EVIDENCE_CONFLICT",
            }
            for reason in reasons
        ):
            status = CellStatus.STALE_ORACLE
        elif any(
            reason == "EXPLICIT_DATASET_DISAGREES_WITH_ARTIFACT"
            or reason == "EXPECTED_DATASET_HASH_MISMATCH"
            or reason == "DATASET_HASH_EVIDENCE_CONFLICT"
            for reason in reasons
        ):
            status = CellStatus.STALE_DATASET
        elif any(
            reason == "EXPECTED_SPLIT_HASH_MISMATCH"
            or reason.startswith("CROSS_METHOD_SPLIT_")
            or reason == "SPLIT_HASH_EVIDENCE_CONFLICT"
            for reason in reasons
        ):
            status = CellStatus.STALE_SPLIT
        elif any(
            token in reason
            for reason in reasons
            for token in (
                "_MISMATCH",
                "FIGURE3_SCHEMA",
                "FIGURE4_SCHEMA",
                "FIGURE3_K_GRID",
                "TABLE2_K_NOT",
                "COVERAGE_NOT_MONOTONE",
                "THRESHOLD_GRID_INVALID",
                "THRESHOLD_GRID_NOT_MONOTONE",
            )
        ):
            status = CellStatus.STALE_METRIC
        elif reasons:
            status = CellStatus.INCOMPLETE
        else:
            stage = str(_find_value(payloads, ("stage", "final_stage")) or "").upper()
            status = (
                CellStatus.FROZEN_PASS
                if frozen_values == {True}
                or stage in {"B14", "B14_FROZEN", "FREEZE"}
                else CellStatus.ADOPTABLE_PASS
            )
    row = {
        "dataset": dataset or "",
        "method": method or "",
        "raw_output_root": raw_root,
        "standardized_output_root": str(standardized_root),
        "generation_adoption_candidate": generation_adoption_candidate,
        "generation_adoption_reason": (
            "raw evidence may be reused only after deterministic unified re-evaluation"
            if generation_adoption_candidate
            else ""
        ),
        "oracle_backend": oracle_backend,
        "oracle_checkpoint": oracle_checkpoint,
        "oracle_hash": oracle_hash,
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "distance_line": distance_line,
        "molclr_checkpoint_hash": molclr_hash,
        "cf_mode": cf_mode,
        "k_max": k_max if k_max is not None else "",
        "table2_k": table2_k if table2_k is not None else "",
        "threshold_config_hash": threshold_hash,
        "status": status.value,
        "adoption_reason": (
            "all required frozen evidence and protocol checks passed"
            if status in PASS_STATUSES
            else ""
        ),
        "rerun_reason": ";".join(sorted(set(reasons))),
    }
    return CandidateAudit(
        root=root,
        dataset=dataset,
        method=method,
        status=status,
        reason_codes=sorted(set(reasons)),
        row=row,
        artifact_hashes=artifact_hashes,
    )


def _scan_roots(
    roots: Sequence[Path], *, max_hash_bytes: int
) -> tuple[set[Path], list[dict[str, Any]]]:
    candidate_roots: set[Path] = set()
    inventory: list[dict[str, Any]] = []
    for scan_root in roots:
        root = scan_root.expanduser().resolve()
        if not root.is_dir():
            continue
        for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
            directory_names[:] = [
                name
                for name in directory_names
                if name not in SKIP_SCAN_DIR_NAMES
                and not name.startswith("checkpoint-")
            ]
            current_path = Path(current)
            anchors = sorted(
                name
                for name in file_names
                if name in ANCHOR_NAMES or name.startswith("table2_")
            )
            if not anchors:
                continue
            candidate_root = current_path.resolve()
            if (
                current_path.name == "standardized"
                and (current_path.parent / "final_gate.json").is_file()
                and (current_path.parent / "run_manifest.json").is_file()
            ):
                candidate_root = current_path.parent.resolve()
            candidate_roots.add(candidate_root)
            dataset_hint = canonical_dataset(current_path.name) or ""
            method_hint = canonical_method(current_path.name) or ""
            for name in anchors:
                path = current_path / name
                try:
                    size = path.stat().st_size
                except OSError:
                    size = -1
                if 0 <= size <= max_hash_bytes and path.is_file():
                    try:
                        digest = sha256_file(path)
                        hash_status = "HASHED"
                    except OSError:
                        digest = ""
                        hash_status = "HASH_FAILED"
                else:
                    digest = ""
                    hash_status = "SKIPPED_SIZE_LIMIT" if size > max_hash_bytes else "STAT_FAILED"
                inventory.append(
                    {
                        "scan_root": str(root),
                        "candidate_root": str(candidate_root),
                        "relative_path": path.relative_to(root).as_posix(),
                        "file_name": name,
                        "size_bytes": size,
                        "sha256": digest,
                        "hash_status": hash_status,
                        "dataset_hint": dataset_hint,
                        "method_hint": method_hint,
                    }
                )
    return candidate_roots, inventory


def _explicit_cell_roots(explicit_cells: Mapping[str, Any]) -> dict[Path, tuple[str, str]]:
    result: dict[Path, tuple[str, str]] = {}
    for key, value in explicit_cells.items():
        if "/" not in key:
            raise ValueError(f"Explicit cell key must be '<dataset>/<method>': {key}")
        raw_dataset, raw_method = key.split("/", 1)
        dataset = canonical_dataset(raw_dataset)
        method = canonical_method(raw_method)
        if dataset is None or method is None:
            raise ValueError(f"Unsupported explicit cell identity: {key}")
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, Mapping):
                item = item.get("standardized_output_root") or item.get("root")
            if not item:
                continue
            result[Path(str(item)).expanduser().resolve()] = (dataset, method)
    return result


def _empty_cell(dataset: str, method: str, status: CellStatus) -> dict[str, Any]:
    reason = (
        "TasteMolNet license gate has no explicit PASS reuse basis"
        if status is CellStatus.BLOCKED_LICENSE
        else "no artifact root with self-identifying evidence was found"
    )
    return {
        "dataset": dataset,
        "method": method,
        "raw_output_root": "",
        "standardized_output_root": "",
        "generation_adoption_candidate": False,
        "generation_adoption_reason": "",
        "oracle_backend": DEFAULT_ORACLES[dataset]["oracle_backend"],
        "oracle_checkpoint": DEFAULT_ORACLES[dataset].get("oracle_checkpoint", ""),
        "oracle_hash": "",
        "dataset_hash": "",
        "split_hash": "",
        "distance_line": DISTANCE_LINE,
        "molclr_checkpoint_hash": "",
        "cf_mode": CF_MODE,
        "k_max": 20,
        "table2_k": 10,
        "threshold_config_hash": "",
        "status": status.value,
        "adoption_reason": "",
        "rerun_reason": reason,
    }


def _downgrade_cross_cell_conflicts(rows: list[dict[str, Any]]) -> None:
    conflict_fields = {
        "oracle_hash": CellStatus.STALE_ORACLE,
        "dataset_hash": CellStatus.STALE_DATASET,
        "split_hash": CellStatus.STALE_SPLIT,
        "distance_line": CellStatus.STALE_METRIC,
        "molclr_checkpoint_hash": CellStatus.STALE_METRIC,
        "cf_mode": CellStatus.STALE_METRIC,
        "threshold_config_hash": CellStatus.STALE_METRIC,
    }
    for dataset in DATASETS:
        passing = [
            row
            for row in rows
            if row["dataset"] == dataset and CellStatus(row["status"]) in PASS_STATUSES
        ]
        for field_name, stale_status in conflict_fields.items():
            values = {str(row.get(field_name) or "") for row in passing}
            if len(values) <= 1:
                continue
            for row in passing:
                row["status"] = stale_status.value
                row["adoption_reason"] = ""
                reason = f"CROSS_METHOD_{field_name.upper()}_CONFLICT"
                current = str(row.get("rerun_reason") or "")
                row["rerun_reason"] = ";".join(filter(None, (current, reason)))


def _status_priority(status: CellStatus) -> int:
    return {
        CellStatus.FROZEN_PASS: 130,
        CellStatus.ADOPTABLE_PASS: 120,
        CellStatus.RUNNING: 110,
        CellStatus.READY: 100,
        CellStatus.INCOMPLETE: 80,
        CellStatus.STALE_METRIC: 70,
        CellStatus.STALE_SPLIT: 60,
        CellStatus.STALE_DATASET: 50,
        CellStatus.STALE_ORACLE: 40,
        CellStatus.BLOCKED_CODE: 30,
        CellStatus.FAILED: 20,
        CellStatus.MISSING: 10,
        CellStatus.BLOCKED_LICENSE: 0,
    }[status]


def _build_oracle_registry(
    rows: Sequence[Mapping[str, Any]], expectations: Mapping[str, Any]
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        expected = _expected_oracle(dataset, expectations)
        observed = [
            {
                "method": row["method"],
                "status": row["status"],
                "oracle_backend": row["oracle_backend"],
                "oracle_checkpoint": row["oracle_checkpoint"],
                "oracle_hash": row["oracle_hash"],
            }
            for row in rows
            if row["dataset"] == dataset and row["standardized_output_root"]
        ]
        passing = [row for row in observed if CellStatus(row["status"]) in PASS_STATUSES]
        hashes = {row["oracle_hash"] for row in passing if row["oracle_hash"]}
        backends = {row["oracle_backend"] for row in passing if row["oracle_backend"]}
        datasets[dataset] = {
            **expected,
            "observed_cells": observed,
            "passing_oracle_hashes": sorted(hashes),
            "passing_oracle_backends": sorted(backends),
            "same_oracle_across_passing_cells": len(hashes) <= 1 and len(backends) <= 1,
        }
    return {
        "schema_version": "four_methods_four_datasets_oracle_registry_v1",
        "datasets": datasets,
        "within_dataset_same_classifier_required": True,
        "cross_dataset_classifier_performance_compared": False,
    }


def audit_registry(config: AuditConfig) -> RegistryResult:
    if config.max_hash_bytes <= 0:
        raise ValueError("max_hash_bytes must be positive")
    explicit = _explicit_cell_roots(config.explicit_cells)
    discovered, inventory = _scan_roots(
        config.scan_roots, max_hash_bytes=config.max_hash_bytes
    )
    candidates: list[CandidateAudit] = []
    for root in sorted(discovered | set(explicit), key=str):
        identity = explicit.get(root)
        candidates.append(
            _audit_candidate(
                root,
                expectations=config.expectations,
                max_hash_bytes=config.max_hash_bytes,
                explicit_dataset=identity[0] if identity else None,
                explicit_method=identity[1] if identity else None,
            )
        )
    taste_allowed = _license_pass(config.taste_license_gate)
    matrix: list[dict[str, Any]] = []
    selected_roots: set[Path] = set()
    for dataset in DATASETS:
        for method in METHODS:
            matching = [
                candidate
                for candidate in candidates
                if candidate.dataset == dataset and candidate.method == method
            ]
            if dataset == "TasteMolNet" and not taste_allowed:
                row = _empty_cell(dataset, method, CellStatus.BLOCKED_LICENSE)
                if matching:
                    row["rerun_reason"] += (
                        ";existing artifacts cannot be promoted while license is blocked"
                    )
                matrix.append(row)
                continue
            if not matching:
                matrix.append(_empty_cell(dataset, method, CellStatus.MISSING))
                continue
            matching.sort(key=lambda item: (-_status_priority(item.status), str(item.root)))
            best = matching[0]
            passing = [item for item in matching if item.status in PASS_STATUSES]
            if len(passing) > 1:
                identities = {
                    (
                        item.row.get("dataset_hash"),
                        item.row.get("split_hash"),
                        item.row.get("oracle_hash"),
                        item.row.get("threshold_config_hash"),
                        tuple(sorted(item.artifact_hashes.items())),
                    )
                    for item in passing
                }
                if len(identities) > 1:
                    row = dict(best.row)
                    row["status"] = CellStatus.INCOMPLETE.value
                    row["adoption_reason"] = ""
                    row["rerun_reason"] = "AMBIGUOUS_MULTIPLE_PASSING_ARTIFACT_ROOTS"
                    matrix.append(row)
                    continue
            selected_roots.add(best.root)
            matrix.append(dict(best.row))
    _downgrade_cross_cell_conflicts(matrix)
    matrix_by_standardized_root = {
        Path(str(row["standardized_output_root"])).expanduser().resolve(): row
        for row in matrix
        if row["standardized_output_root"]
    }
    stale: list[dict[str, Any]] = []
    for candidate in candidates:
        standardized = Path(
            str(candidate.row["standardized_output_root"])
        ).expanduser().resolve()
        selected_row = matrix_by_standardized_root.get(standardized)
        selected = candidate.root in selected_roots
        effective_status = (
            CellStatus(str(selected_row["status"]))
            if selected and selected_row is not None
            else candidate.status
        )
        if selected and effective_status in PASS_STATUSES:
            continue
        reason_codes = (
            str(selected_row.get("rerun_reason") or "")
            if selected and selected_row is not None
            else ";".join(candidate.reason_codes)
        )
        stale.append(
            {
                "candidate_root": str(candidate.root),
                "dataset": candidate.dataset or "",
                "method": candidate.method or "",
                "status": effective_status.value,
                "reason_codes": reason_codes,
                "selected_for_cell": selected,
            }
        )
    matrix.sort(key=lambda row: (DATASETS.index(row["dataset"]), METHODS.index(row["method"])))
    complete = sum(CellStatus(row["status"]) in PASS_STATUSES for row in matrix)
    return RegistryResult(
        matrix_rows=tuple(matrix),
        inventory_rows=tuple(inventory),
        stale_rows=tuple(stale),
        oracle_registry=_build_oracle_registry(matrix, config.expectations),
        evaluation_contract=build_evaluation_contract(config.expectations),
        threshold_contracts=build_threshold_contracts(config.expectations),
        matrix_complete_cells=complete,
    )


def _csv_text(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for source in rows:
        row = dict(source)
        writer.writerow(
            {
                field: json.dumps(value, sort_keys=True, ensure_ascii=True)
                if isinstance((value := row.get(field)), (dict, list, tuple))
                else ""
                if value is None
                else value
                for field in fields
            }
        )
    return buffer.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _report(result: RegistryResult) -> str:
    lines = [
        "# Four methods × four datasets adoption report",
        "",
        "This report is read-only with respect to scanned experiment artifacts. "
        "No missing metric, license evidence, hash, or numeric result is synthesized.",
        "",
        f"- Complete/adoptable cells: {result.matrix_complete_cells}/16",
        f"- Distance line: `{DISTANCE_LINE}`",
        f"- Counterfactual mode: `{CF_MODE}`",
        f"- K prefixes: `1..20`; Table 2 K: `{TABLE2_K}`",
        "",
        "| Dataset | Method | Status | Artifact root | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in result.matrix_rows:
        reason = row["adoption_reason"] or row["rerun_reason"] or "-"
        reason = str(reason).replace("|", "\\|")
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['status']} | "
            f"{row['standardized_output_root'] or '-'} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Frozen threshold contracts",
            "",
            *[
                f"- {dataset}: `{result.threshold_contracts[dataset]['status']}` "
                f"(`{THRESHOLD_ARTIFACT_NAMES[dataset]}`)"
                for dataset in DATASETS
            ],
            "",
            "## Promotion rule",
            "",
            "A cell is paper-eligible only when its status is `FROZEN_PASS` or "
            "`ADOPTABLE_PASS`, all four cells in that dataset share the same dataset, "
            "test-split, oracle, distance encoder, CF-mode, and threshold identities, "
            "and the final combined gate is run separately. `MISSING`, `BLOCKED`, "
            "`INCOMPLETE`, `FAILED`, and every `STALE_*` state must never be rendered "
            "as zero or as a plausible paper value.",
            "",
        ]
    )
    return "\n".join(lines)


def write_registry_outputs(result: RegistryResult, output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Registry output root must be fresh and empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    matrix_payload = {
        "schema_version": SCHEMA_VERSION,
        "audit_complete": True,
        "matrix_complete_cells": result.matrix_complete_cells,
        "matrix_total_cells": result.matrix_total_cells,
        "all_cells_complete": result.matrix_complete_cells == result.matrix_total_cells,
        "cells": list(result.matrix_rows),
        "pass_statuses": sorted(status.value for status in PASS_STATUSES),
        "allowed_statuses": [status.value for status in CellStatus],
        "no_numeric_imputation": True,
    }
    outputs = {
        "matrix_status.csv": _csv_text(result.matrix_rows, MATRIX_FIELDS).encode("utf-8"),
        "matrix_status.json": (
            json.dumps(matrix_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8"),
        "oracle_registry.json": (
            json.dumps(result.oracle_registry, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8"),
        "evaluation_contract.json": (
            json.dumps(result.evaluation_contract, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        ).encode("utf-8"),
        "artifact_inventory.csv": _csv_text(result.inventory_rows, INVENTORY_FIELDS).encode(
            "utf-8"
        ),
        "stale_artifacts.csv": _csv_text(result.stale_rows, STALE_FIELDS).encode("utf-8"),
        "adoption_report.md": (_report(result) + "\n").encode("utf-8"),
    }
    for dataset, relative_path in THRESHOLD_ARTIFACT_NAMES.items():
        outputs[relative_path] = (
            json.dumps(
                result.threshold_contracts[dataset],
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
    marker_payload = outputs.pop("matrix_status.json")
    for name, payload in outputs.items():
        _atomic_write(root / name, payload)
    # The controller marker is published last so a visible audit_complete=true
    # also proves that every required sibling artifact was already materialized.
    _atomic_write(root / "matrix_status.json", marker_payload)
    return root


__all__ = [
    "AuditConfig",
    "CellStatus",
    "DATASETS",
    "DISTANCE_LINE",
    "K_PREFIXES",
    "METHODS",
    "MATRIX_FIELDS",
    "RegistryResult",
    "TABLE2_K",
    "THRESHOLD_ARTIFACT_NAMES",
    "audit_registry",
    "build_evaluation_contract",
    "build_threshold_contracts",
    "canonical_dataset",
    "canonical_method",
    "write_registry_outputs",
]
