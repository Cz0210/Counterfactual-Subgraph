"""BBBP paper artifacts computed from the existing GCF-style run contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.eval.gcf_style_recourse_report import (
    best_recourse_by_parent,
    compute_k_curve,
    compute_prefix_metrics,
    load_method_run,
)


DISTANCE_LINE = "MolCLR-Node-Wasserstein"
DISTANCE_TYPE = "node_wasserstein"
CF_MODE = "strict_flip"
QUANTILES = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
FIGURE3_FIELDS = ("method", "k", "coverage", "cost")
FIGURE4_FIELDS = ("method", "threshold", "coverage")
TABLE2_FIELDS = ("method", "k", "coverage", "cost", "flip_rate", "cf_drop")
METHODS = {
    "ours": {
        "display": "Ours",
        "candidate_kind": "ours",
        "selection_method": "chemllm_stable300_external_selector",
    },
    "globalgce": {
        "display": "GlobalGCE",
        "candidate_kind": "fullgraph",
        "selection_method": "official_frequency_order",
    },
    "gcfexplainer": {
        "display": "GCFExplainer",
        "candidate_kind": "fullgraph",
        "selection_method": "native_gcf_summary_rank_filtered_by_validity",
    },
    "comrecgc": {
        "display": "COMRECGC",
        "candidate_kind": "fullgraph",
        "selection_method": "official_common_recourse_greedy_rank",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for source in rows:
                row = dict(source)
                writer.writerow(
                    {
                        field: ""
                        if row.get(field) is None
                        or (
                            isinstance(row.get(field), float)
                            and not math.isfinite(float(row[field]))
                        )
                        else row.get(field)
                        for field in fields
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def freeze_bbbp_thresholds(
    *,
    calibration_run_dir: str | Path,
    output_path: str | Path,
    calibration_parent_csv: str | Path,
) -> dict[str, Any]:
    run_root = Path(calibration_run_dir).expanduser().resolve()
    source = run_root / "distance_quantiles.csv"
    if not source.is_file():
        raise FileNotFoundError(f"BBBP calibration quantiles are missing: {source}")
    rows = _read_csv(source)
    parsed: list[tuple[float, float]] = []
    for row in rows:
        try:
            quantile = float(row["quantile"])
            threshold = float(row["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid BBBP distance quantile row: {row}") from exc
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"BBBP threshold must be finite and non-negative: {row}")
        parsed.append((quantile, threshold))
    if len(parsed) != len(QUANTILES):
        raise ValueError(f"BBBP calibration must yield {len(QUANTILES)} quantiles.")
    for (actual, _threshold), expected in zip(parsed, QUANTILES, strict=True):
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"BBBP quantile grid differs from the frozen protocol: {actual} != {expected}"
            )
    thresholds = [value for _quantile, value in parsed]
    if any(right < left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("BBBP calibration thresholds are not monotone.")
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    if str(run_config.get("threshold_source")) != "auto_quantile":
        raise ValueError("BBBP thresholds must originate from Ours auto-quantile calibration.")
    payload = {
        "schema_version": "bbbp_wnode_thresholds_v1",
        "dataset": "BBBP",
        "distance_line": DISTANCE_LINE,
        "distance_type": DISTANCE_TYPE,
        "cf_mode": CF_MODE,
        "threshold_source": "frozen_ours_calibration",
        "quantiles": list(QUANTILES),
        "thresholds": thresholds,
        "theta_star_quantile": 0.30,
        "theta_star": thresholds[3],
        "cost_cap_quantile": 0.90,
        "cost_cap": thresholds[-1],
        "calibration_run_dir": str(run_root),
        "calibration_distance_quantiles_sha256": sha256_file(source),
        "calibration_parent_csv": str(Path(calibration_parent_csv).expanduser().resolve()),
        "calibration_parent_csv_sha256": sha256_file(calibration_parent_csv),
        "selection_used_test": False,
        "threshold_fitted_on_test": False,
    }
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Frozen BBBP threshold contract already differs: {target}")
        return existing
    _write_json(target, payload)
    return payload


def load_bbbp_thresholds(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "bbbp_wnode_thresholds_v1":
        raise ValueError(f"Unsupported BBBP threshold schema: {source}")
    if payload.get("distance_line") != DISTANCE_LINE or payload.get("cf_mode") != CF_MODE:
        raise ValueError("BBBP threshold distance/strict-flip semantics changed.")
    if payload.get("threshold_source") != "frozen_ours_calibration":
        raise ValueError("BBBP thresholds must be frozen from calibration.")
    if payload.get("selection_used_test") is not False:
        raise ValueError("BBBP threshold selection must not use test data.")
    if payload.get("threshold_fitted_on_test") is not False:
        raise ValueError("BBBP thresholds must not be fitted on test data.")
    quantiles = [float(value) for value in payload.get("quantiles", [])]
    thresholds = [float(value) for value in payload.get("thresholds", [])]
    if len(quantiles) != len(QUANTILES) or len(thresholds) != len(QUANTILES):
        raise ValueError("BBBP frozen threshold vector must have seven values.")
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(quantiles, QUANTILES, strict=True)
    ):
        raise ValueError("BBBP frozen quantile protocol changed.")
    if not math.isclose(
        float(payload["theta_star"]), thresholds[3], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("BBBP theta_star is not q30.")
    if not math.isclose(
        float(payload["cost_cap"]), thresholds[-1], rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("BBBP cost_cap is not q90.")
    return payload


def export_bbbp_method_artifacts(
    *,
    method: str,
    test_run_dir: str | Path,
    thresholds_json: str | Path,
    output_dir: str | Path,
    expected_parent_count: int,
    expected_top_k: int = 20,
    protocol_manifest: str | Path,
    split_manifest: str | Path,
    split_leakage_audit: str | Path,
    candidate_lineage_audit: str | Path,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unsupported BBBP method: {method}")
    spec = METHODS[method]
    contract_paths = {
        "protocol_manifest.json": Path(protocol_manifest).expanduser().resolve(),
        "split_manifest.json": Path(split_manifest).expanduser().resolve(),
        "split_leakage_audit.json": Path(split_leakage_audit).expanduser().resolve(),
        "candidate_lineage_audit.json": Path(candidate_lineage_audit).expanduser().resolve(),
    }
    missing_contracts = [str(path) for path in contract_paths.values() if not path.is_file()]
    if missing_contracts:
        raise FileNotFoundError(f"BBBP artifact contracts are missing: {missing_contracts}")
    leakage_payload = json.loads(
        contract_paths["split_leakage_audit.json"].read_text(encoding="utf-8")
    )
    lineage_payload = json.loads(
        contract_paths["candidate_lineage_audit.json"].read_text(encoding="utf-8")
    )
    if leakage_payload.get("passed") is not True or lineage_payload.get("passed") is not True:
        raise ValueError("BBBP paper export requires passing leakage and lineage audits.")
    thresholds = load_bbbp_thresholds(thresholds_json)
    run = load_method_run(
        str(spec["display"]),
        test_run_dir,
        expected_top_k=int(expected_top_k),
        expected_num_parents=int(expected_parent_count),
    )
    theta_star = float(thresholds["theta_star"])
    prefix = compute_k_curve(run, threshold=theta_star, max_k=int(expected_top_k))
    figure3 = [
        {
            "method": spec["display"],
            "k": int(row["k"]),
            "coverage": float(row["coverage"]),
            "cost": row["conditional_median_cost"],
        }
        for row in prefix
    ]
    distances, _drops = best_recourse_by_parent(run, k=10)
    ordered_distances = [float(distances[parent_id]) for parent_id in run.parent_ids]
    figure4 = [
        {
            "method": spec["display"],
            "threshold": float(threshold),
            "coverage": sum(value <= float(threshold) for value in ordered_distances)
            / len(ordered_distances),
        }
        for threshold in thresholds["thresholds"]
    ]
    k10 = compute_prefix_metrics(run, k=10, threshold=theta_star)
    table2 = {
        "method": spec["display"],
        "k": 10,
        "coverage": float(k10["coverage"]),
        "cost": k10["conditional_median_cost"],
        "flip_rate": k10["flip_rate_among_covered"],
        "cf_drop": k10["mean_cf_drop_among_covered"],
    }
    if [int(row["k"]) for row in figure3] != list(range(1, int(expected_top_k) + 1)):
        raise AssertionError("BBBP Figure 3 K grid changed.")
    if any(
        right["coverage"] + 1e-12 < left["coverage"]
        for left, right in zip(figure3, figure3[1:])
    ):
        raise AssertionError("BBBP Figure 3 coverage is not monotone.")
    if any(
        right["coverage"] + 1e-12 < left["coverage"]
        for left, right in zip(figure4, figure4[1:])
    ):
        raise AssertionError("BBBP Figure 4 coverage is not monotone.")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"BBBP paper artifact root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        _write_csv(temporary / "figure3_coverage_vs_k.csv", figure3, FIGURE3_FIELDS)
        _write_csv(
            temporary / "figure4_coverage_vs_threshold.csv", figure4, FIGURE4_FIELDS
        )
        for name, source in contract_paths.items():
            shutil.copyfile(source, temporary / name)
        _write_csv(
            temporary / f"table2_{method}_k10.csv", [table2], TABLE2_FIELDS
        )
        candidate_ids = [candidate.candidate_id for candidate in run.candidates]
        parent_ids_sha256 = stable_json_sha256(list(run.parent_ids))
        summary = {
            "schema_version": "bbbp_paper_method_summary_v1",
            "dataset": "BBBP",
            "method": spec["display"],
            "source_label": 1,
            "target_label": 0,
            "distance_line": DISTANCE_LINE,
            "distance_type": DISTANCE_TYPE,
            "cf_mode": CF_MODE,
            "test_parent_count": len(run.parent_ids),
            "test_parent_ids_sha256": parent_ids_sha256,
            "candidate_count": len(run.candidates),
            "pair_count": int(run.num_unique_parent_candidate_pairs),
            "theta_star": theta_star,
            "cost_cap": float(thresholds["cost_cap"]),
            "thresholds": list(thresholds["thresholds"]),
            "k10": table2,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
            "protocol_manifest_sha256": sha256_file(temporary / "protocol_manifest.json"),
            "split_manifest_sha256": sha256_file(temporary / "split_manifest.json"),
            "split_leakage_audit_sha256": sha256_file(
                temporary / "split_leakage_audit.json"
            ),
            "candidate_lineage_audit_sha256": sha256_file(
                temporary / "candidate_lineage_audit.json"
            ),
            "candidate_order_sha256": stable_json_sha256(candidate_ids),
            "run_complete": True,
        }
        _write_json(temporary / "summary.json", summary)
        run_config = run.config
        manifest = {
            "schema_version": "bbbp_paper_method_manifest_v1",
            "dataset": "BBBP",
            "method": spec["display"],
            "source_evaluation_run": str(run.run_dir),
            "source_run_config_sha256": sha256_file(run.run_dir / "run_config.json"),
            "source_pair_details_sha256": sha256_file(
                run.run_dir / "details" / "pair_details.csv"
            ),
            "candidate_path": str(run.candidate_path),
            "candidate_path_sha256": (
                sha256_file(run.candidate_path)
                if run.candidate_path.is_file()
                else None
            ),
            "candidate_order_sha256": summary["candidate_order_sha256"],
            "test_parent_ids_sha256": parent_ids_sha256,
            "selection_method": run.selection_method,
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "thresholds_json": str(Path(thresholds_json).expanduser().resolve()),
            "thresholds_json_sha256": sha256_file(thresholds_json),
            "teacher_path": run_config.get("teacher_path"),
            "molclr_checkpoint": run_config.get("molclr_checkpoint"),
            "distance_line": DISTANCE_LINE,
            "cf_mode": CF_MODE,
            "test_used_for_selection": False,
            "threshold_fitted_on_test": False,
        }
        _write_json(temporary / "run_manifest.json", manifest)
        audit = {
            "schema_version": "bbbp_paper_method_audit_v1",
            "passed": True,
            "dataset": "BBBP",
            "method": spec["display"],
            "figure3_schema": list(FIGURE3_FIELDS),
            "figure3_rows": len(figure3),
            "figure3_k": [int(row["k"]) for row in figure3],
            "figure4_schema": list(FIGURE4_FIELDS),
            "figure4_rows": len(figure4),
            "figure4_thresholds": [float(row["threshold"]) for row in figure4],
            "table2_schema": list(TABLE2_FIELDS),
            "table2_rows": 1,
            "strict_flip": True,
            "candidate_order_preserved": True,
            "selection_performed_in_eval": False,
            "coverage_vs_k_monotone": True,
            "coverage_vs_threshold_monotone": True,
        }
        _write_json(temporary / "final_artifact_audit.json", audit)
        file_inventory = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        _write_json(
            temporary / "artifact_manifest.json",
            {"schema_version": "bbbp_paper_files_v1", "files": file_inventory},
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


__all__ = [
    "CF_MODE",
    "DISTANCE_LINE",
    "DISTANCE_TYPE",
    "FIGURE3_FIELDS",
    "FIGURE4_FIELDS",
    "METHODS",
    "QUANTILES",
    "TABLE2_FIELDS",
    "export_bbbp_method_artifacts",
    "freeze_bbbp_thresholds",
    "load_bbbp_thresholds",
    "sha256_file",
]
