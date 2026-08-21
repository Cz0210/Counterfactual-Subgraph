#!/usr/bin/env python3
"""Fail-closed audit for BACE/TasteMolNet frozen-GNN provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import (  # noqa: E402
    assert_oracle_backend_allowed,
    normalize_dataset_id,
)


_RF_VALUE_PATTERN = re.compile(
    r"(?:^|[/_.:\-])(rf|random_forest|randomforest)(?:$|[/_.:\-])",
    re.IGNORECASE,
)
_KNOWN_JSON_FILES = (
    "model_card.json",
    "oracle_provenance.json",
    "training_metrics.json",
    "environment.json",
    "git_state.json",
    "label_map.json",
    "split_manifest.json",
    "temperature_scaling.json",
)


class OracleProvenanceError(ValueError):
    """Raised when formal oracle provenance fails the task-specific guard."""


def _walk(payload: Any, prefix: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            yield path, key.lower(), value
            yield from _walk(value, path)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            path = f"{prefix}[{index}]"
            yield from _walk(value, path)


def _values_for_key(payload: Mapping[str, Any], key: str) -> list[tuple[str, Any]]:
    normalized = key.lower()
    return [(path, value) for path, observed, value in _walk(payload) if observed == normalized]


def _false_value(value: Any) -> bool:
    if value is False or value == 0:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "0", "no", "none"}:
        return True
    return False


def _nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _extract_dataset(payload: Mapping[str, Any]) -> str | None:
    observed = _values_for_key(payload, "dataset")
    normalized: list[str] = []
    for _path, value in observed:
        try:
            dataset_id = normalize_dataset_id(str(value))
        except ValueError:
            continue
        if dataset_id not in normalized:
            normalized.append(dataset_id)
    if len(normalized) == 1:
        return normalized[0]
    return None


def audit_oracle_provenance(
    payload: Mapping[str, Any],
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Return an auditable PASS/FAIL report without mutating the artifact."""

    errors: list[str] = []
    warnings: list[str] = []
    requested_dataset: str | None = None
    if dataset is not None:
        requested_dataset = normalize_dataset_id(dataset)
    inferred_dataset = _extract_dataset(payload)
    resolved_dataset = requested_dataset or inferred_dataset
    if resolved_dataset is None:
        errors.append("dataset_missing_or_ambiguous")
    if requested_dataset is not None and inferred_dataset not in {None, requested_dataset}:
        errors.append(
            f"dataset_mismatch:requested={requested_dataset}:artifact={inferred_dataset}"
        )

    backend_values = _values_for_key(payload, "oracle_backend")
    if not backend_values:
        errors.append("oracle_backend_missing")
    for path, value in backend_values:
        backend = str(value).strip().lower()
        if resolved_dataset is not None:
            try:
                assert_oracle_backend_allowed(resolved_dataset, backend)
            except ValueError as exc:
                errors.append(f"{path}:{exc}")
        if resolved_dataset in {"bace", "tastemolnet"} and backend != "gnn":
            errors.append(f"{path}:expected_gnn:observed={value!r}")

    classifier_values = _values_for_key(payload, "classifier_type")
    if not classifier_values:
        errors.append("classifier_type_missing")
    for path, value in classifier_values:
        if resolved_dataset in {"bace", "tastemolnet"} and str(value).strip().lower() != "gnn":
            errors.append(f"{path}:expected_gnn:observed={value!r}")

    rf_used_values = _values_for_key(payload, "rf_oracle_used")
    if not rf_used_values:
        errors.append("rf_oracle_used_missing")
    for path, value in rf_used_values:
        if resolved_dataset in {"bace", "tastemolnet"} and not _false_value(value):
            errors.append(f"{path}:expected_false:observed={value!r}")

    forbidden_references: list[dict[str, Any]] = []
    for path, key, value in _walk(payload):
        if key == "rf_oracle_used":
            continue
        if not _nonempty(value) or isinstance(value, (Mapping, list, tuple)):
            continue
        text = str(value).strip()
        key_mentions_rf = bool(_RF_VALUE_PATTERN.search(key))
        value_mentions_rf = bool(_RF_VALUE_PATTERN.search(text)) or text.lower().endswith(
            (".pkl", ".pickle")
        )
        if key_mentions_rf or value_mentions_rf:
            forbidden_references.append({"path": path, "value": text})
    if resolved_dataset in {"bace", "tastemolnet"} and forbidden_references:
        errors.append("forbidden_rf_provenance_reference")

    for required_key in ("num_classes", "source_label", "backbone"):
        if not _values_for_key(payload, required_key):
            warnings.append(f"recommended_field_missing:{required_key}")

    unique_errors = list(dict.fromkeys(errors))
    unique_warnings = list(dict.fromkeys(warnings))
    return {
        "schema_version": 1,
        "status": "PASS" if not unique_errors else "FAIL",
        "passed": not unique_errors,
        "dataset": resolved_dataset,
        "required_contract": {
            "oracle_backend": "gnn" if resolved_dataset in {"bace", "tastemolnet"} else None,
            "classifier_type": "gnn" if resolved_dataset in {"bace", "tastemolnet"} else None,
            "rf_oracle_used": False if resolved_dataset in {"bace", "tastemolnet"} else None,
        },
        "errors": unique_errors,
        "warnings": unique_warnings,
        "forbidden_rf_references": forbidden_references,
    }


def assert_oracle_provenance(
    payload: Mapping[str, Any],
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    report = audit_oracle_provenance(payload, dataset=dataset)
    if not report["passed"]:
        raise OracleProvenanceError(
            "Oracle provenance failed: " + "; ".join(report["errors"])
        )
    return report


def load_provenance_artifact(path: str | Path) -> dict[str, Any]:
    """Load one JSON file or the known metadata files in a checkpoint bundle."""

    artifact = Path(path).expanduser().resolve()
    if artifact.is_file():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Oracle provenance JSON must be an object: {artifact}")
        return payload
    if not artifact.is_dir():
        raise FileNotFoundError(f"Oracle provenance artifact does not exist: {artifact}")
    bundle: dict[str, Any] = {}
    for filename in _KNOWN_JSON_FILES:
        candidate = artifact / filename
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint JSON must be an object: {candidate}")
        bundle[filename] = payload
    if not bundle:
        raise FileNotFoundError(
            f"No recognized oracle provenance JSON exists under: {artifact}"
        )
    bundle["artifact_inventory"] = {
        "model_pt_present": (artifact / "model.pt").is_file(),
        "checkpoint_root": str(artifact),
    }
    return bundle


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_provenance_artifact(args.artifact)
    report = audit_oracle_provenance(payload, dataset=args.dataset)
    report["artifact"] = str(Path(args.artifact).expanduser().resolve())
    if args.output_json:
        _atomic_write_json(Path(args.output_json).expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["passed"]:
        print("[ORACLE_PROVENANCE_AUDIT_PASS]", flush=True)
        return 0
    print("[ORACLE_PROVENANCE_AUDIT_FAIL]", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
