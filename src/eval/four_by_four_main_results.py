"""Fail-closed aggregation and rendering for the four-by-four main results.

The registry is the only source of cell locations.  This module never computes
scientific metrics: it validates frozen standardized artifacts, copies their
reported values into combined tables, and renders those values without
interpolation or smoothing.  An incomplete or inconsistent matrix produces a
small staging audit only; it never produces a plausible-looking final table or
figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.eval.four_by_four_registry import (
    CF_MODE,
    DATASETS,
    DISTANCE_LINE,
    K_PREFIXES,
    METHODS,
    PASS_STATUSES,
    SCHEMA_VERSION as REGISTRY_SCHEMA_VERSION,
    TABLE2_K,
    canonical_dataset,
    canonical_method,
    sha256_file,
    stable_json_sha256,
)
from src.eval.user_approved_frozen_v4 import (
    APPROVAL_ID as FROZEN_V4_APPROVAL_ID,
    EXCEPTION_SCHEMA as FROZEN_V4_EXCEPTION_SCHEMA,
    validate_adopted_cell as validate_frozen_v4_adopted_cell,
)


EXPORT_SCHEMA_VERSION = "four_methods_four_datasets_main_results_v1"
METHOD_ORDER = ("Ours", "GCFExplainer", "GlobalGCE", "ComRecGC")
DATASET_ORDER = DATASETS
METHOD_SLUGS = {
    "Ours": "ours",
    "GCFExplainer": "gcfexplainer",
    "GlobalGCE": "globalgce",
    "ComRecGC": "comrecgc",
}
DATASET_SLUGS = {
    "AIDS": "aids",
    "Mutagenicity": "mutagenicity",
    "BACE": "bace",
    "TasteMolNet": "tastemolnet",
}
PASS_STATUS_NAMES = frozenset(status.value for status in PASS_STATUSES)
REQUIRED_CELL_FILES = (
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
IDENTITY_FIELDS = (
    "oracle_hash",
    "dataset_hash",
    "split_hash",
    "distance_line",
    "molclr_checkpoint_hash",
    "cf_mode",
    "threshold_config_hash",
)
METHOD_STYLES = {
    "Ours": {"color": "black", "marker": "s"},
    "GCFExplainer": {"color": "#B02BC7", "marker": "^"},
    "GlobalGCE": {"color": "#E53935", "marker": "x"},
    "ComRecGC": {"color": "#2E7D32", "marker": "*"},
}
FIGURE3_MARKER_INDICES = (0, 2, 4, 9, 14, 19)
FIGURE4_THRESHOLD_START = 0.0
FIGURE4_THRESHOLD_STOP = 0.0535
FIGURE4_THRESHOLD_POINTS = 601
TABLE2_THETA = 0.05
NA_COST_VALUES = frozenset({"", "n/a", "na"})


class MainResultsError(ValueError):
    """The final presentation contract cannot be satisfied."""


@dataclass(frozen=True)
class CellArtifacts:
    dataset: str
    method: str
    root: Path
    row: Mapping[str, Any]
    figure3: tuple[dict[str, str], ...]
    figure4: tuple[dict[str, str], ...]
    table2: tuple[dict[str, str], ...]
    destination: tuple[dict[str, str], ...]
    destination_fields: tuple[str, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class ExportResult:
    output_root: Path
    complete: bool
    matrix_complete_cells: int
    blocked_reasons: tuple[str, ...]
    generated_files: tuple[str, ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MainResultsError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MainResultsError(f"Expected one JSON object: {path}")
    return dict(payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _values(payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[Any]:
    wanted = set(keys)
    return [value for payload in payloads for key, value in _walk(payload) if key in wanted]


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _canonical_dataset_strict(value: Any) -> str:
    dataset = canonical_dataset(value)
    if dataset is None:
        raise MainResultsError(f"Unsupported dataset identity: {value!r}")
    return dataset


def _canonical_method_strict(value: Any) -> str:
    if _normalized(value) == "clear":
        raise MainResultsError("CLEAR is not ComRecGC and is forbidden in the 4x4 matrix")
    method = canonical_method(value)
    if method is None or method not in METHOD_ORDER:
        raise MainResultsError(f"Unsupported method identity: {value!r}")
    return method


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if not fields or len(fields) != len(set(fields)):
                raise MainResultsError(f"Invalid or duplicate CSV header: {path}")
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as exc:
        raise MainResultsError(f"Invalid CSV artifact {path}: {exc}") from exc
    if not rows:
        raise MainResultsError(f"CSV artifact has no data rows: {path}")
    return fields, rows


def _field(fields: Sequence[str], aliases: Sequence[str], *, path: Path) -> str:
    lookup = {_normalized(name): name for name in fields}
    for alias in aliases:
        value = lookup.get(_normalized(alias))
        if value is not None:
            return value
    raise MainResultsError(f"{path} lacks required field from {tuple(aliases)}")


def _finite(raw: Any, *, field: str, path: Path, rate: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise MainResultsError(f"{path}: {field} is not numeric: {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0 or (rate and value > 1.0):
        interval = "[0,1]" if rate else "nonnegative finite"
        raise MainResultsError(f"{path}: {field} must be {interval}: {raw!r}")
    return value


def _conditional_cost(
    raw: Any, *, coverage: float, field: str, path: Path
) -> float | None:
    """Validate a conditional cost without converting an empty cohort to zero."""

    if str(raw).strip().lower() in NA_COST_VALUES:
        if coverage != 0.0:
            raise MainResultsError(
                f"{path}: {field} may be N/A only when strict-flip coverage is zero"
            )
        return None
    return _finite(raw, field=field, path=path)


def _method_rows(
    path: Path,
    *,
    expected_method: str,
    kind: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    fields, rows = _read_csv(path)
    method_field = _field(fields, ("method", "Method"), path=path)
    for row in rows:
        observed = _canonical_method_strict(row.get(method_field))
        if observed != expected_method:
            raise MainResultsError(
                f"{path}: expected method {expected_method}, observed {observed}"
            )
    if kind == "figure3":
        k_field = _field(fields, ("k", "K"), path=path)
        coverage_field = _field(
            fields, ("coverage", "ccrcov", "close_cf_coverage"), path=path
        )
        cost_field = _field(
            fields,
            (
                "cost",
                "conditional_median_cost",
                "conditional_mean_cost",
                "fixed_capped_mean_cost",
            ),
            path=path,
        )
        try:
            ordered = sorted(rows, key=lambda row: int(row[k_field]))
        except (TypeError, ValueError) as exc:
            raise MainResultsError(f"{path}: invalid K value") from exc
        ks = [int(row[k_field]) for row in ordered]
        if ks != list(K_PREFIXES):
            raise MainResultsError(f"{path}: Figure 3 K grid must be exactly 1..20")
        coverage = [
            _finite(row[coverage_field], field=coverage_field, path=path, rate=True)
            for row in ordered
        ]
        for row, row_coverage in zip(ordered, coverage):
            _conditional_cost(
                row[cost_field],
                coverage=row_coverage,
                field=cost_field,
                path=path,
            )
        if any(right + 1e-12 < left for left, right in zip(coverage, coverage[1:])):
            raise MainResultsError(f"{path}: Figure 3 coverage is not monotone")
        return ordered, {
            "method": method_field,
            "k": k_field,
            "coverage": coverage_field,
            "cost": cost_field,
        }
    if kind == "figure4":
        threshold_field = _field(fields, ("threshold", "Threshold"), path=path)
        coverage_field = _field(
            fields, ("coverage", "ccrcov", "close_cf_coverage"), path=path
        )
        thresholds = [
            _finite(row[threshold_field], field=threshold_field, path=path)
            for row in rows
        ]
        expected_thresholds = [
            FIGURE4_THRESHOLD_START
            + (FIGURE4_THRESHOLD_STOP - FIGURE4_THRESHOLD_START)
            * index
            / (FIGURE4_THRESHOLD_POINTS - 1)
            for index in range(FIGURE4_THRESHOLD_POINTS)
        ]
        if len(thresholds) != FIGURE4_THRESHOLD_POINTS or any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
            for observed, expected in zip(thresholds, expected_thresholds)
        ):
            raise MainResultsError(
                f"{path}: Figure 4 grid must be the frozen "
                f"{FIGURE4_THRESHOLD_POINTS}-point "
                f"{FIGURE4_THRESHOLD_START}..{FIGURE4_THRESHOLD_STOP} grid"
            )
        if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
            raise MainResultsError(
                f"{path}: Figure 4 thresholds must be raw, strictly increasing points"
            )
        coverage = [
            _finite(row[coverage_field], field=coverage_field, path=path, rate=True)
            for row in rows
        ]
        if any(right + 1e-12 < left for left, right in zip(coverage, coverage[1:])):
            raise MainResultsError(f"{path}: Figure 4 coverage is not monotone")
        return rows, {
            "method": method_field,
            "threshold": threshold_field,
            "coverage": coverage_field,
        }
    if kind == "table2":
        if len(rows) != 1:
            raise MainResultsError(f"{path}: Table 2 cell must contain exactly one row")
        k_field = _field(fields, ("k", "K"), path=path)
        try:
            k = int(rows[0][k_field])
        except (TypeError, ValueError) as exc:
            raise MainResultsError(f"{path}: invalid Table 2 K") from exc
        if k != TABLE2_K:
            raise MainResultsError(f"{path}: Table 2 K must be {TABLE2_K}")
        coverage_field = _field(
            fields, ("coverage", "ccrcov", "close_cf_coverage"), path=path
        )
        cost_field = _field(
            fields,
            (
                "cost",
                "conditional_median_cost",
                "conditional_mean_cost",
                "fixed_capped_mean_cost",
            ),
            path=path,
        )
        coverage = _finite(
            rows[0][coverage_field], field=coverage_field, path=path, rate=True
        )
        _conditional_cost(
            rows[0][cost_field],
            coverage=coverage,
            field=cost_field,
            path=path,
        )
        return rows, {
            "method": method_field,
            "k": k_field,
            "coverage": coverage_field,
            "cost": cost_field,
        }
    raise AssertionError(kind)


def _declared_hashes(payloads: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for payload in payloads:
        for key, value in _walk(payload):
            if key not in {"file_sha256", "files", "output_files", "artifacts"}:
                continue
            if not isinstance(value, Mapping):
                continue
            for name, metadata in value.items():
                if isinstance(metadata, Mapping):
                    digest = metadata.get("sha256") or metadata.get("hash")
                else:
                    digest = metadata
                text = str(digest or "").lower()
                if re.fullmatch(r"[0-9a-f]{64}", text):
                    result.setdefault(Path(str(name)).name, set()).add(text)
    return result


def _identity_hash_values(
    payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> set[str]:
    return {
        str(value).lower()
        for value in _values(payloads, keys)
        if re.fullmatch(r"[0-9a-fA-F]{64}", str(value))
    }


def _explicit_false(payloads: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> bool:
    observed = _values(payloads, keys)
    if not observed:
        return False
    normalized: set[bool] = set()
    for value in observed:
        if isinstance(value, bool):
            normalized.add(value)
        elif str(value).strip().lower() in {"false", "0", "no"}:
            normalized.add(False)
        elif str(value).strip().lower() in {"true", "1", "yes"}:
            normalized.add(True)
    return normalized == {False}


def _explicit_true_or_calibration(
    payloads: Sequence[Mapping[str, Any]], *, true_keys: Sequence[str], split_keys: Sequence[str]
) -> bool:
    if True in _values(payloads, true_keys):
        return True
    return any(
        str(value).strip().lower() in {"calibration", "frozen_calibration"}
        for value in _values(payloads, split_keys)
    )


def _validate_manifest_identity(
    payloads: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    dataset: str,
    method: str,
    *,
    root: Path,
) -> None:
    dataset_values = {
        canonical_dataset(value)
        for value in _values(payloads, ("dataset", "dataset_name", "dataset_key"))
        if canonical_dataset(value) is not None
    }
    method_values: set[str] = set()
    for value in _values(payloads, ("method", "method_name", "display_method")):
        token = _normalized(value)
        if token == "clear":
            raise MainResultsError(f"{dataset}/{method}: CLEAR provenance is forbidden")
        canonical = canonical_method(value)
        if canonical is not None:
            method_values.add(canonical)
    if dataset_values != {dataset}:
        raise MainResultsError(
            f"{dataset}/{method}: manifest dataset identity is not uniquely {dataset}"
        )
    if method_values != {method}:
        raise MainResultsError(
            f"{dataset}/{method}: manifest method identity is not uniquely {method}"
        )
    exception_payloads = [
        payload
        for payload in payloads
        if payload.get("schema_version") == FROZEN_V4_EXCEPTION_SCHEMA
    ]
    frozen_v4_exception = False
    if exception_payloads:
        if len(exception_payloads) != 1:
            raise MainResultsError(
                f"{dataset}/{method}: duplicate frozen-v4 registry exceptions"
            )
        valid, reasons, _ = validate_frozen_v4_adopted_cell(root)
        exception_path = root / "registry_exception.json"
        if not valid:
            raise MainResultsError(
                f"{dataset}/{method}: invalid frozen-v4 registry exception: {list(reasons)}"
            )
        if (
            row.get("registry_exception") != FROZEN_V4_APPROVAL_ID
            or str(row.get("registry_exception_hash") or "").lower()
            != sha256_file(exception_path)
            or row.get("identity_evidence_status")
            != "USER_APPROVED_LEGACY_IDENTITIES_NOT_EMBEDDED"
        ):
            raise MainResultsError(
                f"{dataset}/{method}: matrix row is not bound to the validated frozen-v4 exception"
            )
        frozen_v4_exception = True
    hash_keys = {
        "oracle_hash": (
            "oracle_hash",
            "oracle_checkpoint_hash",
            "gnn_checkpoint_hash",
            "teacher_sha256",
            "teacher_hash",
        ),
        "dataset_hash": (
            "dataset_hash",
            "dataset_sha256",
            "processed_dataset_sha256",
            "dataset_csv_sha256",
        ),
        "molclr_checkpoint_hash": (
            "molclr_checkpoint_hash",
            "molclr_checkpoint_sha256",
            "distance_encoder_hash",
        ),
        "threshold_config_hash": (
            "threshold_config_hash",
            "thresholds_json_sha256",
            "thresholds_sha256",
        ),
    }
    for row_key, manifest_keys in hash_keys.items():
        expected = str(row.get(row_key) or "").lower()
        observed = _identity_hash_values(payloads, manifest_keys)
        if frozen_v4_exception and row_key in {
            "oracle_hash",
            "dataset_hash",
            "molclr_checkpoint_hash",
        }:
            if expected or observed:
                raise MainResultsError(
                    f"{dataset}/{method}: waived {row_key} must remain explicitly unavailable"
                )
            continue
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise MainResultsError(f"{dataset}/{method}: matrix lacks {row_key}")
        if observed != {expected}:
            raise MainResultsError(
                f"{dataset}/{method}: {row_key} closure mismatch: {sorted(observed)}"
            )
    expected_split = str(row.get("split_hash") or "").lower()
    if not frozen_v4_exception and re.fullmatch(r"[0-9a-f]{64}", expected_split) is None:
        raise MainResultsError(f"{dataset}/{method}: matrix lacks split_hash")
    cohort_hashes = _identity_hash_values(
        payloads,
        ("test_parent_ids_sha256", "test_cohort_hash", "parent_ids_sha256"),
    )
    observed_split = cohort_hashes or _identity_hash_values(
        payloads, ("test_split_hash", "split_hash")
    )
    if frozen_v4_exception:
        if expected_split or observed_split:
            raise MainResultsError(
                f"{dataset}/{method}: waived split_hash must remain explicitly unavailable"
            )
    elif observed_split != {expected_split}:
        raise MainResultsError(
            f"{dataset}/{method}: split_hash closure mismatch: {sorted(observed_split)}"
        )
    backend = str(row.get("oracle_backend") or "").lower()
    expected_backend = "rf" if dataset in {"AIDS", "Mutagenicity"} else "gnn"
    if backend != expected_backend:
        raise MainResultsError(
            f"{dataset}/{method}: oracle backend must be {expected_backend}"
        )
    if str(row.get("distance_line") or "") != DISTANCE_LINE:
        raise MainResultsError(f"{dataset}/{method}: distance line mismatch")
    if str(row.get("cf_mode") or "").lower() != CF_MODE:
        raise MainResultsError(f"{dataset}/{method}: counterfactual mode mismatch")
    backend_values = {
        str(value).strip().lower()
        for value in _values(payloads, ("oracle_backend", "teacher_backend"))
    }
    if backend_values != {expected_backend}:
        raise MainResultsError(
            f"{dataset}/{method}: manifest oracle backend closure mismatch"
        )
    if dataset in {"BACE", "TasteMolNet"}:
        if not _explicit_false(payloads, ("rf_oracle_used", "RF_ORACLE_USED")):
            raise MainResultsError(f"{dataset}/{method}: RF exclusion is not proven")
    if frozen_v4_exception:
        return
    if not _explicit_false(
        payloads, ("test_used_for_selection", "selection_used_test")
    ):
        raise MainResultsError(f"{dataset}/{method}: test selection exclusion is not proven")
    if not _explicit_false(
        payloads, ("threshold_fitted_on_test", "threshold_fit_on_test")
    ):
        raise MainResultsError(f"{dataset}/{method}: test threshold exclusion is not proven")
    if not _explicit_true_or_calibration(
        payloads,
        true_keys=("selector_fitted_on_calibration",),
        split_keys=("selection_split", "selector_split", "threshold_source_split"),
    ):
        raise MainResultsError(
            f"{dataset}/{method}: calibration-only selector provenance is missing"
        )
    if True not in _values(
        payloads,
        (
            "test_loaded_only_after_freeze",
            "test_used_only_after_freeze",
            "selector_frozen_before_test",
            "selection_frozen_before_test",
        ),
    ):
        raise MainResultsError(
            f"{dataset}/{method}: test-after-selector-freeze provenance is missing"
        )


def _table2_path(root: Path, method: str) -> Path:
    return root / f"table2_{METHOD_SLUGS[method]}_k10.csv"


def audit_cell(row: Mapping[str, Any]) -> CellArtifacts:
    dataset = _canonical_dataset_strict(row.get("dataset"))
    method = _canonical_method_strict(row.get("method"))
    status = str(row.get("status") or "")
    if status not in PASS_STATUS_NAMES:
        raise MainResultsError(f"{dataset}/{method}: status is not paper PASS: {status}")
    raw_root = str(row.get("standardized_output_root") or "")
    if not raw_root or not Path(raw_root).expanduser().is_absolute():
        raise MainResultsError(f"{dataset}/{method}: standardized root must be absolute")
    unresolved = Path(raw_root).expanduser()
    if unresolved.is_symlink():
        raise MainResultsError(f"{dataset}/{method}: standardized root may not be a symlink")
    try:
        root = unresolved.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MainResultsError(f"{dataset}/{method}: standardized root is missing") from exc
    if not root.is_dir():
        raise MainResultsError(f"{dataset}/{method}: standardized root is not a directory")
    paths = {name: root / name for name in REQUIRED_CELL_FILES}
    paths["table2"] = _table2_path(root, method)
    missing = [name for name, path in paths.items() if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise MainResultsError(f"{dataset}/{method}: incomplete standardized closure: {missing}")
    json_names = (
        "prefix_metrics.json",
        "summary.json",
        "run_manifest.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
        "final_artifact_audit.json",
    )
    payload_by_name = {name: _read_json_object(paths[name]) for name in json_names}
    payloads = list(payload_by_name.values())
    for optional_name in (
        "freeze_manifest.json",
        "registry_exception.json",
        "_FINALIZED.json",
    ):
        optional_path = root / optional_name
        if optional_path.is_file() and optional_path.stat().st_size > 0:
            payloads.append(_read_json_object(optional_path))
    final_audit = payload_by_name["final_artifact_audit.json"]
    if final_audit.get("passed") is not True and final_audit.get("audit_passed") is not True:
        raise MainResultsError(f"{dataset}/{method}: final artifact audit is not PASS")
    _validate_manifest_identity(payloads, row, dataset, method, root=root)

    figure3, _ = _method_rows(paths["figure3_coverage_vs_k.csv"], expected_method=method, kind="figure3")
    figure4, _ = _method_rows(paths["figure4_coverage_vs_threshold.csv"], expected_method=method, kind="figure4")
    table2, _ = _method_rows(paths["table2"], expected_method=method, kind="table2")
    destination_fields, destination = _read_csv(paths["destination_distribution.csv"])
    destination_method_field = next(
        (name for name in destination_fields if _normalized(name) == "method"), None
    )
    destination_dataset_field = next(
        (name for name in destination_fields if _normalized(name) == "dataset"), None
    )
    for destination_row in destination:
        if destination_method_field is not None and _canonical_method_strict(
            destination_row.get(destination_method_field)
        ) != method:
            raise MainResultsError(
                f"{dataset}/{method}: destination method identity mismatch"
            )
        if destination_dataset_field is not None and _canonical_dataset_strict(
            destination_row.get(destination_dataset_field)
        ) != dataset:
            raise MainResultsError(
                f"{dataset}/{method}: destination dataset identity mismatch"
            )
    if dataset == "TasteMolNet":
        normalized_fields = {_normalized(name) for name in destination_fields}
        required_destination = {
            "destinationlabel",
            "sweettobittercount",
            "sweettobitterrate",
            "sweettotastelesscount",
            "sweettotastelessrate",
            "perruledestinationdistribution",
        }
        if not normalized_fields.intersection(required_destination):
            raise MainResultsError(
                f"{dataset}/{method}: multiclass destination distribution is missing"
            )
        label_field = next(
            (name for name in destination_fields if _normalized(name) == "destinationlabel"),
            None,
        )
        if label_field is not None:
            observed_labels = {
                int(row[label_field])
                for row in destination
                if str(row.get(label_field) or "").strip() not in {"", "N/A", "NA"}
            }
            if not observed_labels.issubset({0, 2}):
                raise MainResultsError(
                    f"{dataset}/{method}: destination labels must be a subset of {{0,2}}"
                )

    source_paths = tuple(dict.fromkeys(paths.values()))
    source_hashes = {path.name: sha256_file(path) for path in source_paths}
    declarations = _declared_hashes(payloads)
    closure_names = {
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        paths["table2"].name,
        "prefix_metrics.csv",
        "prefix_metrics.json",
        "parent_best_distances.csv",
        "destination_distribution.csv",
        "summary.json",
        "oracle_manifest.json",
        "evaluation_manifest.json",
    }
    for name in closure_names:
        expected = source_hashes.get(name)
        declared = declarations.get(name, set())
        if expected is None or declared != {expected}:
            raise MainResultsError(
                f"{dataset}/{method}: frozen file hash closure mismatch for {name}"
            )
    return CellArtifacts(
        dataset=dataset,
        method=method,
        root=root,
        row=dict(row),
        figure3=tuple(figure3),
        figure4=tuple(figure4),
        table2=tuple(table2),
        destination=tuple(destination),
        destination_fields=tuple(destination_fields),
        source_hashes=source_hashes,
    )


def _validate_matrix(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise MainResultsError("Unsupported matrix_status.json schema")
    if payload.get("audit_complete") is not True:
        raise MainResultsError("Matrix registry audit is not complete")
    if payload.get("no_numeric_imputation") is not True:
        raise MainResultsError("Matrix registry does not forbid numeric imputation")
    rows = payload.get("cells")
    if not isinstance(rows, list) or len(rows) != 16 or not all(isinstance(row, dict) for row in rows):
        raise MainResultsError("matrix_status.json must contain exactly 16 cell objects")
    expected = {(dataset, method) for dataset in DATASET_ORDER for method in METHOD_ORDER}
    observed: set[tuple[str, str]] = set()
    normalized_rows: list[dict[str, Any]] = []
    for source in rows:
        dataset = _canonical_dataset_strict(source.get("dataset"))
        method = _canonical_method_strict(source.get("method"))
        key = (dataset, method)
        if key in observed:
            raise MainResultsError(f"Duplicate matrix cell: {dataset}/{method}")
        observed.add(key)
        normalized_rows.append({**source, "dataset": dataset, "method": method})
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise MainResultsError(f"Matrix cell set mismatch; missing={missing}, extra={extra}")
    try:
        complete_cells = int(payload.get("matrix_complete_cells"))
        total_cells = int(payload.get("matrix_total_cells"))
    except (TypeError, ValueError) as exc:
        raise MainResultsError("Matrix completion counts are invalid") from exc
    if total_cells != 16:
        raise MainResultsError("matrix_total_cells must be 16")
    all_complete = payload.get("all_cells_complete") is True
    normalized_rows.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]), METHOD_ORDER.index(row["method"])
        )
    )
    return normalized_rows, complete_cells, all_complete


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _union_fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> list[str]:
    observed = {key for row in rows for key in row}
    result = [key for key in preferred if key in observed]
    result.extend(sorted(observed - set(result)))
    return result


def _canonical_combined_rows(
    cells: Sequence[CellArtifacts], *, kind: str
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for cell in sorted(cells, key=lambda item: METHOD_ORDER.index(item.method)):
        source_rows = getattr(cell, kind)
        path = (
            cell.root / "figure3_coverage_vs_k.csv"
            if kind == "figure3"
            else cell.root / "figure4_coverage_vs_threshold.csv"
            if kind == "figure4"
            else _table2_path(cell.root, cell.method)
        )
        fields = tuple(source_rows[0].keys())
        method_field = _field(fields, ("method", "Method"), path=path)
        dataset_field = next(
            (name for name in fields if _normalized(name) == "dataset"), None
        )
        k_field = _field(fields, ("k", "K"), path=path) if kind != "figure4" else None
        threshold_field = _field(fields, ("threshold", "Threshold"), path=path) if kind == "figure4" else None
        coverage_field = _field(fields, ("coverage", "ccrcov", "close_cf_coverage"), path=path)
        cost_field = (
            _field(
                fields,
                ("cost", "conditional_median_cost", "conditional_mean_cost", "fixed_capped_mean_cost"),
                path=path,
            )
            if kind != "figure4"
            else None
        )
        for source in source_rows:
            row = dict(source)
            row["dataset"] = cell.dataset
            row["method"] = cell.method
            if dataset_field is not None and dataset_field != "dataset":
                row.pop(dataset_field, None)
            if method_field != "method":
                row.pop(method_field, None)
            if k_field is not None:
                row["k"] = source[k_field]
                if k_field != "k":
                    row.pop(k_field, None)
            if threshold_field is not None:
                row["threshold"] = source[threshold_field]
                if threshold_field != "threshold":
                    row.pop(threshold_field, None)
            row["coverage"] = source[coverage_field]
            if coverage_field != "coverage":
                row.pop(coverage_field, None)
            if cost_field is not None:
                row["cost"] = source[cost_field]
                if cost_field != "cost":
                    row.pop(cost_field, None)
            result.append(row)
    return result


def _destination_rows(cells: Sequence[CellArtifacts]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for cell in sorted(cells, key=lambda item: METHOD_ORDER.index(item.method)):
        for source in cell.destination:
            row = {
                key: value
                for key, value in source.items()
                if _normalized(key) not in {"dataset", "method"}
            }
            result.append({**row, "dataset": cell.dataset, "method": cell.method})
    return result


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _display_rate(value: Any) -> str:
    return f"{100.0 * float(value):.2f}\\%"


def _display_cost(value: Any) -> str:
    if str(value).strip().lower() in NA_COST_VALUES:
        return "N/A"
    return f"{float(value):.4f}"


def _write_dataset_table(root: Path, dataset: str, rows: Sequence[Mapping[str, str]]) -> None:
    fields = _union_fields(rows, ("dataset", "method", "k", "coverage", "cost", "flip_rate", "cf_drop"))
    _write_csv(root / "table2.csv", rows, fields)
    markdown = [
        f"| Method | {dataset} CCRCOV | {dataset} Cost |",
        "|---|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        (
            f"Method & {_latex_escape(dataset)} CCRCOV & "
            f"{_latex_escape(dataset)} Cost" + r" \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {100.0 * float(row['coverage']):.2f}% | "
            f"{_display_cost(row['cost'])} |"
        )
        latex.append(
            f"{_latex_escape(row['method'])} & {_display_rate(row['coverage'])} & {_display_cost(row['cost'])} \\\\"
        )
    latex.extend((r"\bottomrule", r"\end{tabular}"))
    (root / "table2.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (root / "table2.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def _write_four_dataset_table(path: Path, by_dataset: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    lines = [
        r"\begin{tabular}{l" + "rr" * len(DATASET_ORDER) + "}",
        r"\toprule",
        "Method & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{_latex_escape(dataset)}}}"
            for dataset in DATASET_ORDER
        ) + r" \\",
        " & " + " & ".join("CCRCOV & Cost" for _ in DATASET_ORDER) + r" \\",
        r"\midrule",
    ]
    indexed = {
        dataset: {str(row["method"]): row for row in rows}
        for dataset, rows in by_dataset.items()
    }
    for method in METHOD_ORDER:
        cells: list[str] = []
        for dataset in DATASET_ORDER:
            row = indexed[dataset][method]
            cells.extend((_display_rate(row["coverage"]), _display_cost(row["cost"])))
        lines.append(f"{_latex_escape(method)} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _configure_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - production image dependency
        raise MainResultsError("matplotlib is required for final figure rendering") from exc
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
        }
    )
    return plt


def _plot_lines(axis: Any, rows: Sequence[Mapping[str, str]], *, x: str, y: str, marker_every: Any) -> None:
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        style = METHOD_STYLES[method]
        axis.plot(
            [float(row[x]) for row in selected],
            [
                math.nan
                if str(row[y]).strip().lower() in NA_COST_VALUES
                else float(row[y])
                for row in selected
            ],
            color=style["color"],
            marker=style["marker"],
            markevery=marker_every,
            markersize=5.5,
            markeredgewidth=0.9,
            label=method,
        )
    axis.grid(alpha=0.42, linewidth=0.7)


def render_outputs(
    root: Path,
    figure3_by_dataset: Mapping[str, Sequence[Mapping[str, str]]],
    figure4_by_dataset: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Render exact empirical rows; no interpolation, spline, or smoothing."""

    plt = _configure_matplotlib()
    for dataset in DATASET_ORDER:
        combined = root / DATASET_SLUGS[dataset] / "combined"
        fig3, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
        _plot_lines(axes[0], figure3_by_dataset[dataset], x="k", y="coverage", marker_every=list(FIGURE3_MARKER_INDICES))
        _plot_lines(axes[1], figure3_by_dataset[dataset], x="k", y="cost", marker_every=list(FIGURE3_MARKER_INDICES))
        axes[0].set_title(dataset)
        axes[0].set_ylabel("Strict-flip CCRCOV")
        axes[1].set_ylabel("Conditional cost")
        axes[1].set_xlabel("Number of global actions K")
        axes[1].set_xticks([1, 5, 10, 15, 20])
        handles, labels = axes[0].get_legend_handles_labels()
        fig3.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
        fig3.tight_layout(rect=(0, 0.07, 1, 1))
        fig3.savefig(combined / "figure3_coverage_vs_k.png", dpi=300, bbox_inches="tight")
        fig3.savefig(combined / "figure3_coverage_vs_k.pdf", bbox_inches="tight")
        plt.close(fig3)

        fig4, axis = plt.subplots(1, 1, figsize=(6.4, 3.8))
        _plot_lines(axis, figure4_by_dataset[dataset], x="threshold", y="coverage", marker_every=max(1, len(figure4_by_dataset[dataset]) // 24))
        axis.set_title(dataset)
        axis.set_xlabel("WNode threshold")
        axis.set_ylabel("Strict-flip CCRCOV")
        handles, labels = axis.get_legend_handles_labels()
        fig4.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
        fig4.tight_layout(rect=(0, 0.12, 1, 1))
        fig4.savefig(combined / "figure4_coverage_vs_threshold.png", dpi=300, bbox_inches="tight")
        fig4.savefig(combined / "figure4_coverage_vs_threshold.pdf", bbox_inches="tight")
        plt.close(fig4)

    fig3, axes = plt.subplots(2, 4, figsize=(16.0, 6.3), sharex="col")
    for column, dataset in enumerate(DATASET_ORDER):
        _plot_lines(axes[0, column], figure3_by_dataset[dataset], x="k", y="coverage", marker_every=list(FIGURE3_MARKER_INDICES))
        _plot_lines(axes[1, column], figure3_by_dataset[dataset], x="k", y="cost", marker_every=list(FIGURE3_MARKER_INDICES))
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("K")
        axes[1, column].set_xticks([1, 5, 10, 15, 20])
        if column == 0:
            axes[0, column].set_ylabel("Strict-flip CCRCOV")
            axes[1, column].set_ylabel("Conditional cost")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig3.tight_layout(rect=(0, 0.07, 1, 1))
    fig3.savefig(root / "paper_figure3_four_datasets.pdf", bbox_inches="tight")
    plt.close(fig3)

    fig4, axes4 = plt.subplots(1, 4, figsize=(16.0, 3.8))
    for column, dataset in enumerate(DATASET_ORDER):
        _plot_lines(axes4[column], figure4_by_dataset[dataset], x="threshold", y="coverage", marker_every=max(1, len(figure4_by_dataset[dataset]) // 24))
        axes4[column].set_title(dataset)
        axes4[column].set_xlabel("WNode threshold")
        if column == 0:
            axes4[column].set_ylabel("Strict-flip CCRCOV")
    handles, labels = axes4[0].get_legend_handles_labels()
    fig4.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig4.tight_layout(rect=(0, 0.12, 1, 1))
    fig4.savefig(root / "paper_figure4_four_datasets.pdf", bbox_inches="tight")
    plt.close(fig4)


def _output_inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    ignored = set(exclude)
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in ignored
    }


def _partial(
    output_root: Path,
    *,
    matrix_path: Path,
    matrix_hash: str,
    complete_cells: int,
    rows: Sequence[Mapping[str, Any]],
    reasons: Sequence[str],
) -> ExportResult:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=False)
    incomplete = [
        {
            "dataset": row["dataset"],
            "method": row["method"],
            "status": row.get("status"),
            "reason": row.get("rerun_reason") or row.get("adoption_reason") or "",
        }
        for row in rows
        if row.get("status") not in PASS_STATUS_NAMES
    ]
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "status": "BLOCKED_INCOMPLETE_MATRIX",
        "matrix_status_path": str(matrix_path),
        "matrix_status_sha256": matrix_hash,
        "matrix_complete_cells": complete_cells,
        "matrix_total_cells": 16,
        "incomplete_cells": incomplete,
        "closure_failures": list(reasons),
        "numeric_outputs_generated": False,
        "figures_generated": False,
        "tables_generated": False,
        "zero_fill_used": False,
        "paper_directory_written": False,
    }
    _atomic_json(output_root / "partial_staging_audit.json", payload)
    (output_root / "BLOCKED_INCOMPLETE_MATRIX").write_text(
        "BLOCKED_INCOMPLETE_MATRIX\n", encoding="utf-8"
    )
    return ExportResult(
        output_root=output_root,
        complete=False,
        matrix_complete_cells=complete_cells,
        blocked_reasons=tuple(reasons),
        generated_files=("partial_staging_audit.json", "BLOCKED_INCOMPLETE_MATRIX"),
    )


def export_main_results(
    *,
    matrix_status: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    renderer: Callable[[Path, Mapping[str, Sequence[Mapping[str, str]]], Mapping[str, Sequence[Mapping[str, str]]]], None] = render_outputs,
) -> ExportResult:
    matrix_path = Path(matrix_status).expanduser().resolve(strict=True)
    if not matrix_path.is_file() or matrix_path.is_symlink():
        raise MainResultsError("matrix_status.json must be a physical file")
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root must be fresh: {destination}")
    project = Path(project_root).expanduser().resolve(strict=True)
    paper = (project / "paper").resolve(strict=False)
    try:
        destination.relative_to(paper)
    except ValueError:
        pass
    else:
        raise MainResultsError("Final exporter may not write into paper/")
    payload = _read_json_object(matrix_path)
    rows, complete_cells, all_complete = _validate_matrix(payload)
    matrix_hash = sha256_file(matrix_path)
    statuses_complete = all(row.get("status") in PASS_STATUS_NAMES for row in rows)
    if not (all_complete and complete_cells == 16 and statuses_complete):
        return _partial(
            destination,
            matrix_path=matrix_path,
            matrix_hash=matrix_hash,
            complete_cells=complete_cells,
            rows=rows,
            reasons=("matrix_status does not prove 16/16 paper-pass cells",),
        )

    artifacts: list[CellArtifacts] = []
    closure_failures: list[str] = []
    for row in rows:
        try:
            artifacts.append(audit_cell(row))
        except (MainResultsError, OSError, ValueError) as exc:
            closure_failures.append(f"{row['dataset']}/{row['method']}: {exc}")
    if closure_failures:
        return _partial(
            destination,
            matrix_path=matrix_path,
            matrix_hash=matrix_hash,
            complete_cells=complete_cells,
            rows=rows,
            reasons=closure_failures,
        )

    by_dataset = {
        dataset: [item for item in artifacts if item.dataset == dataset]
        for dataset in DATASET_ORDER
    }
    for dataset, cells in by_dataset.items():
        for field in IDENTITY_FIELDS:
            values = {str(cell.row.get(field) or "") for cell in cells}
            if len(values) != 1 or "" in values:
                closure_failures.append(
                    f"{dataset}: cross-method {field} is not one nonempty identity"
                )
        grids = [
            tuple(row["threshold"] for row in _canonical_combined_rows([cell], kind="figure4"))
            for cell in cells
        ]
        if len(set(grids)) != 1:
            closure_failures.append(f"{dataset}: methods do not share one raw threshold grid")
    if closure_failures:
        return _partial(
            destination,
            matrix_path=matrix_path,
            matrix_hash=matrix_hash,
            complete_cells=complete_cells,
            rows=rows,
            reasons=closure_failures,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        figure3_by_dataset: dict[str, list[dict[str, str]]] = {}
        figure4_by_dataset: dict[str, list[dict[str, str]]] = {}
        table2_by_dataset: dict[str, list[dict[str, str]]] = {}
        for dataset in DATASET_ORDER:
            cells = by_dataset[dataset]
            combined = temporary / DATASET_SLUGS[dataset] / "combined"
            combined.mkdir(parents=True, exist_ok=True)
            figure3 = _canonical_combined_rows(cells, kind="figure3")
            figure4 = _canonical_combined_rows(cells, kind="figure4")
            table2 = _canonical_combined_rows(cells, kind="table2")
            destinations = _destination_rows(cells)
            figure3_by_dataset[dataset] = figure3
            figure4_by_dataset[dataset] = figure4
            table2_by_dataset[dataset] = table2
            _write_csv(combined / "figure3_coverage_vs_k.csv", figure3, _union_fields(figure3, ("dataset", "method", "k", "coverage", "cost")))
            _write_csv(combined / "figure4_coverage_vs_threshold.csv", figure4, _union_fields(figure4, ("dataset", "method", "threshold", "coverage")))
            _write_dataset_table(combined, dataset, table2)
            _write_csv(combined / "destination_distribution.csv", destinations, _union_fields(destinations, ("dataset", "method", "destination_label")))
            input_cells = {
                cell.method: {
                    "standardized_output_root": str(cell.root),
                    "source_hashes": dict(sorted(cell.source_hashes.items())),
                    "matrix_identity": {field: cell.row.get(field) for field in IDENTITY_FIELDS},
                }
                for cell in cells
            }
            combined_manifest = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "status": "PASS",
                "dataset": dataset,
                "methods": list(METHOD_ORDER),
                "distance_line": DISTANCE_LINE,
                "cf_mode": CF_MODE,
                "k_prefixes": list(K_PREFIXES),
                "table2_k": TABLE2_K,
                "table2_theta": TABLE2_THETA,
                "figure4_threshold_start": FIGURE4_THRESHOLD_START,
                "figure4_threshold_stop": FIGURE4_THRESHOLD_STOP,
                "figure4_threshold_points": FIGURE4_THRESHOLD_POINTS,
                "figure4_rendering": "raw_empirical_points_no_spline_no_smoothing",
                "selection_performed_in_export": False,
                "metric_recomputation_performed": False,
                "numeric_imputation_used": False,
                "source_matrix_sha256": matrix_hash,
                "input_cells": input_cells,
            }
            _atomic_json(combined / "combined_manifest.json", combined_manifest)
        renderer(temporary, figure3_by_dataset, figure4_by_dataset)
        for dataset in DATASET_ORDER:
            combined = temporary / DATASET_SLUGS[dataset] / "combined"
            combined_manifest_path = combined / "combined_manifest.json"
            combined_manifest = _read_json_object(combined_manifest_path)
            combined_manifest["outputs"] = _output_inventory(
                combined,
                exclude=("combined_manifest.json", "combined_audit.json"),
            )
            _atomic_json(combined_manifest_path, combined_manifest)
            audit = {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "status": "PASS",
                "passed": True,
                "dataset": dataset,
                "methods": list(METHOD_ORDER),
                "figure3_row_count": 4 * 20,
                "figure4_raw_threshold_grid_sha256": stable_json_sha256(
                    [row["threshold"] for row in figure4_by_dataset[dataset] if row["method"] == "Ours"]
                ),
                "table2_row_count": 4,
                "clear_present": False,
                "zero_fill_used": False,
                "smoothing_used": False,
                "paper_directory_written": False,
                "combined_manifest_sha256": sha256_file(combined_manifest_path),
                "files": _output_inventory(combined, exclude=("combined_audit.json",)),
            }
            _atomic_json(combined / "combined_audit.json", audit)
        _write_four_dataset_table(temporary / "paper_table2_four_datasets.tex", table2_by_dataset)
        final_manifest = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "status": "PASS",
            "matrix_complete_cells": 16,
            "matrix_total_cells": 16,
            "all_cells_complete": True,
            "matrix_status_path": str(matrix_path),
            "matrix_status_sha256": matrix_hash,
            "datasets": list(DATASET_ORDER),
            "methods": list(METHOD_ORDER),
            "distance_line": DISTANCE_LINE,
            "cf_mode": CF_MODE,
            "table2_k": TABLE2_K,
            "table2_theta": TABLE2_THETA,
            "figure4_threshold_start": FIGURE4_THRESHOLD_START,
            "figure4_threshold_stop": FIGURE4_THRESHOLD_STOP,
            "figure4_threshold_points": FIGURE4_THRESHOLD_POINTS,
            "scientific_metrics_recomputed": False,
            "thresholds_selected_in_export": False,
            "smoothing_used": False,
            "numeric_imputation_used": False,
            "paper_directory_written": False,
            "outputs": _output_inventory(temporary),
        }
        _atomic_json(temporary / "final_export_manifest.json", final_manifest)
        final_audit = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "status": "PASS",
            "passed": True,
            "all_16_cells_verified": True,
            "same_oracle_split_distance_threshold_within_dataset": True,
            "strict_flip": True,
            "clear_rejected": True,
            "taste_destination_fields_preserved": True,
            "zero_fill_used": False,
            "undefined_conditional_cost_preserved_as_na": True,
            "paper_directory_written": False,
            "final_export_manifest_sha256": sha256_file(temporary / "final_export_manifest.json"),
            "outputs": _output_inventory(temporary, exclude=("final_export_audit.json", "FINAL_EXPORT_PASS.json", "PASS")),
        }
        _atomic_json(temporary / "final_export_audit.json", final_audit)
        _atomic_json(
            temporary / "FINAL_EXPORT_PASS.json",
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "status": "PASS",
                "passed": True,
                "final_export_audit_sha256": sha256_file(temporary / "final_export_audit.json"),
            },
        )
        (temporary / "PASS").write_text("PASS\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    files = tuple(sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()))
    return ExportResult(
        output_root=destination,
        complete=True,
        matrix_complete_cells=16,
        blocked_reasons=(),
        generated_files=files,
    )


def expected_final_output_files() -> list[str]:
    files: list[str] = []
    for dataset in DATASET_ORDER:
        prefix = f"{DATASET_SLUGS[dataset]}/combined"
        files.extend(
            f"{prefix}/{name}"
            for name in (
                "figure3_coverage_vs_k.csv",
                "figure3_coverage_vs_k.png",
                "figure3_coverage_vs_k.pdf",
                "figure4_coverage_vs_threshold.csv",
                "figure4_coverage_vs_threshold.png",
                "figure4_coverage_vs_threshold.pdf",
                "table2.csv",
                "table2.tex",
                "table2.md",
                "destination_distribution.csv",
                "combined_manifest.json",
                "combined_audit.json",
            )
        )
    files.extend(
        (
            "paper_figure3_four_datasets.pdf",
            "paper_figure4_four_datasets.pdf",
            "paper_table2_four_datasets.tex",
            "final_export_manifest.json",
            "final_export_audit.json",
            "FINAL_EXPORT_PASS.json",
            "PASS",
        )
    )
    return files


__all__ = [
    "DATASET_ORDER",
    "EXPORT_SCHEMA_VERSION",
    "ExportResult",
    "METHOD_ORDER",
    "MainResultsError",
    "audit_cell",
    "expected_final_output_files",
    "export_main_results",
    "render_outputs",
]
