#!/usr/bin/env python3
"""Audit whether molecular WNode thresholds follow one preregistered protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _quantile_rows(payload: dict[str, Any]) -> list[tuple[float, float]]:
    raw = payload.get("raw_quantile_thresholds")
    if isinstance(raw, list):
        result = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            result.append((float(row["quantile"]), float(row["threshold"])))
        return result
    quantiles = payload.get("quantiles")
    thresholds = payload.get("thresholds")
    if isinstance(quantiles, list) and isinstance(thresholds, list):
        if len(quantiles) != len(thresholds):
            raise ValueError("Threshold quantile/value lengths differ.")
        return [(float(q), float(value)) for q, value in zip(quantiles, thresholds, strict=True)]
    return []


def _record(dataset: str, name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _quantile_rows(payload)
    source = str(payload.get("threshold_source") or "")
    protocol = str(payload.get("threshold_protocol_version") or "")
    method_specific = any(
        token in f"{source} {path}".lower()
        for token in ("ours", "gcf", "globalgce", "comrecgc", "selector")
    )
    fitted_on_test = payload.get("threshold_fitted_on_test")
    selection_used_test = payload.get("selection_used_test")
    test_independent_proven = fitted_on_test is False and selection_used_test is False
    return {
        "dataset": dataset,
        "name": name,
        "path": str(path),
        "sha256": _sha256(path),
        "threshold_source": source,
        "threshold_protocol_version": protocol or None,
        "theta_star_quantile": payload.get("theta_star_quantile"),
        "theta_star": payload.get("theta_star"),
        "shared_across_methods_declared": payload.get("shared_across_methods"),
        "method_specific_source": method_specific,
        "threshold_fitted_on_test": fitted_on_test,
        "selection_used_test": selection_used_test,
        "test_independent_proven": test_independent_proven,
        "quantile_rows": rows,
    }


def audit_threshold_protocol(
    *,
    aids_thresholds: Sequence[Path],
    mut_thresholds: Sequence[Path],
    bace_old_threshold: Path,
    bace_connected_threshold: Path,
    output_dir: Path,
) -> dict[str, Any]:
    records = [
        *(_record("AIDS", f"aids_{index}", path) for index, path in enumerate(aids_thresholds)),
        *(
            _record("Mutagenicity", f"mut_{index}", path)
            for index, path in enumerate(mut_thresholds)
        ),
        _record("BACE", "bace_legacy", bace_old_threshold),
        _record("BACE", "bace_connected_v3", bace_connected_threshold),
    ]
    by_dataset = {
        dataset: [row for row in records if row["dataset"] == dataset]
        for dataset in ("AIDS", "Mutagenicity", "BACE")
    }
    q30_used_aids = any(row.get("theta_star_quantile") == 0.3 for row in by_dataset["AIDS"])
    q30_used_mut = any(
        row.get("theta_star_quantile") == 0.3 for row in by_dataset["Mutagenicity"]
    )
    protocol_versions = {
        str(row["threshold_protocol_version"])
        for row in records
        if row.get("threshold_protocol_version")
    }
    q30_preregistered = bool(
        q30_used_aids
        and q30_used_mut
        and all(by_dataset.values())
        and len(protocol_versions) == 1
        and all(row.get("theta_star_quantile") == 0.3 for row in records)
    )
    method_independent = bool(
        records
        and all(
            row.get("shared_across_methods_declared") is True
            and row.get("method_specific_source") is False
            for row in records
        )
    )
    test_independent = bool(records and all(row["test_independent_proven"] for row in records))
    bace_rule_matches = bool(q30_preregistered and method_independent)
    stable_common_rule = q30_preregistered and method_independent and test_independent

    comparison_rows = [
        {key: value for key, value in row.items() if key != "quantile_rows"}
        for row in records
    ]
    quantile_rows = [
        {
            "dataset": row["dataset"],
            "name": row["name"],
            "quantile": quantile,
            "threshold": threshold,
            "source": row["threshold_source"],
        }
        for row in records
        for quantile, threshold in row["quantile_rows"]
    ]
    connected = next(row for row in records if row["name"] == "bace_connected_v3")
    payload = {
        "status": "PASS_AUDIT_REQUIRES_NEW_COMMON_THRESHOLD"
        if not stable_common_rule
        else "PASS_EXISTING_COMMON_THRESHOLD",
        "Q30_PRE_REGISTERED_ACROSS_DATASETS": q30_preregistered,
        "Q30_USED_FOR_AIDS": q30_used_aids,
        "Q30_USED_FOR_MUT": q30_used_mut,
        "BACE_RULE_MATCHES_AIDS_MUT": bace_rule_matches,
        "THRESHOLD_METHOD_INDEPENDENT": method_independent,
        "THRESHOLD_TEST_INDEPENDENT": test_independent,
        "STABLE_COMMON_THRESHOLD_RULE_FOUND": stable_common_rule,
        "OLD_BACE_CONNECTED_THETA": connected.get("theta_star"),
        "OLD_BACE_CONNECTED_SOURCE": connected.get("threshold_source"),
        "required_protocol": None
        if stable_common_rule
        else {
            "strict_primary": "pooled_method_independent_calibration_q30",
            "standard_primary": "pooled_method_independent_calibration_q50",
            "same_parent_cohort": True,
            "test_used": False,
            "freeze_before_final_test": True,
        },
        "final_test_allowed": stable_common_rule,
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "threshold_protocol_audit.json", payload)
    _write_csv(output_dir / "threshold_protocol_comparison.csv", comparison_rows)
    _write_csv(output_dir / "calibration_distance_quantiles.csv", quantile_rows)
    report = [
        "Common threshold protocol audit",
        "===============================",
        f"Q30_PRE_REGISTERED_ACROSS_DATASETS={str(q30_preregistered).lower()}",
        f"Q30_USED_FOR_AIDS={str(q30_used_aids).lower()}",
        f"Q30_USED_FOR_MUT={str(q30_used_mut).lower()}",
        f"BACE_RULE_MATCHES_AIDS_MUT={str(bace_rule_matches).lower()}",
        f"THRESHOLD_METHOD_INDEPENDENT={str(method_independent).lower()}",
        f"THRESHOLD_TEST_INDEPENDENT={str(test_independent).lower()}",
        "DECISION="
        + (
            "retain_existing_preregistered_rule"
            if stable_common_rule
            else "freeze_pooled_calibration_q30_and_q50_before_final_test"
        ),
    ]
    _atomic_text(output_dir / "threshold_protocol_report.txt", "\n".join(report) + "\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--aids-threshold", action="append", default=[])
    parser.add_argument("--mut-threshold", action="append", default=[])
    parser.add_argument("--bace-old-threshold", required=True)
    parser.add_argument("--bace-connected-threshold", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_threshold_protocol(
        aids_thresholds=[Path(path).expanduser().resolve() for path in args.aids_threshold],
        mut_thresholds=[Path(path).expanduser().resolve() for path in args.mut_threshold],
        bace_old_threshold=Path(args.bace_old_threshold).expanduser().resolve(),
        bace_connected_threshold=Path(args.bace_connected_threshold).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, sort_keys=True))
    print("[COMMON_THRESHOLD_PROTOCOL_AUDIT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
