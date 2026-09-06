"""Deterministic publication-only K10 projection of the sealed Mut fbef result.

No model, dataset, pair store, selector, or distance implementation is imported.
The original K20 export remains authoritative source evidence, not a K10 curve.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "mut_registry_k10_projection_v1"
SCIENCE_COMMIT = "fbefa4caff172453d42afd90f8518bc7e8bddf47"
REFERENCE_GRID_SHA = "817968eb0260902205f9faedd634b7b6872fef8b12c46772d84423c9336102ae"
PARENT_COUNT = 217
COPIED = ("figure3_coverage_vs_k.csv", "table2_comrecgc_k10.csv",
          "prefix_metrics.csv", "prefix_metrics.json", "parent_best_distances.csv")
FLAGS = {"aggregate_reexport": True, "figure_table_recomputed": True,
         "inference_rerun": False, "ot_rerun": False, "selector_rerun": False,
         "test_dataset_rerun": False, "numeric_imputation": False,
         "source_files_modified": False}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("MUT_K10_PROJECTION_REJECTED:" + message)


def _path(value: str | Path, *, directory: bool = False) -> Path:
    p = Path(value)
    _require(p.is_absolute() and not p.is_symlink(), "physical absolute path required")
    _require(p.is_dir() if directory else p.is_file(), f"missing source:{p}")
    _require(p.resolve() == p, f"symlink ancestor:{p}")
    return p


def _sha(p: Path) -> str:
    return hashlib.sha256(_path(p).read_bytes()).hexdigest()


def _json(p: Path) -> dict[str, Any]:
    value = json.loads(_path(p).read_text())
    _require(isinstance(value, dict), f"object required:{p}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _rows(p: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_path(p).read_text())))


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _grid_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _driver_identity() -> dict[str, str]:
    checkout = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    return {"scientific_implementation_commit": SCIENCE_COMMIT,
            "publication_driver_commit": commit,
            "publication_driver_checkout": str(checkout),
            "publication_driver_module": str(Path(__file__).resolve()),
            "publication_driver_module_sha256": _sha(Path(__file__).resolve())}


def _validate_driver(receipt: Mapping[str, Any]) -> None:
    # Reopen the preserved producer, not the current reader's unrelated HEAD.
    checkout = _path(receipt["publication_driver_checkout"], directory=True)
    module = _path(receipt["publication_driver_module"])
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    _require(receipt.get("scientific_implementation_commit") == SCIENCE_COMMIT
             and receipt.get("publication_driver_commit") == commit
             and module == checkout / "src/eval/mut_registry_k10_projection.py"
             and receipt.get("publication_driver_module_sha256") == _sha(module),
             "actual publication driver/science identity")


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _source_inputs(source: Path, reference: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen only small sealed summaries and the independently frozen protocol."""
    source = _path(source, directory=True)
    standard = source / "standardized"
    native = _json(standard / "run_manifest.json")
    outer = _json(source / "run_manifest.json")
    _require(native.get("project_commit") == outer.get("project_commit") == SCIENCE_COMMIT,
             "original scientific commit changed")
    _require(native.get("source_label") == 1 and native.get("target_label") == 0,
             "sealed source/target labels")
    for key, expected in {"dataset": "mutagenicity", "method": "COMRECGC",
                          "oracle_backend": "rf", "classifier_family": "random_forest",
                          "rf_oracle_used": True, "generation_adopted": True,
                          "independent_scientific_adoption_authorized": True,
                          "status": "PASS"}.items():
        _require(outer.get(key) == expected, f"outer scientific proof:{key}")
    _require(outer == _json(source / "final_gate.json") and
             _json(source / "_RUN_COMPLETE.json") == {**outer, "run_complete": True},
             "outer terminal closure changed")
    freeze = _json(standard / "freeze_manifest.json")
    _require(outer.get("standardized_run_manifest_sha256") == _sha(standard / "run_manifest.json")
             and outer.get("freeze_manifest_sha256") == _sha(standard / "freeze_manifest.json"),
             "outer-to-native seal changed")
    evidence = {}
    for name in (*COPIED, "figure4_coverage_vs_threshold.csv", "run_manifest.json",
                 "final_artifact_audit.json", "summary.json"):
        p = standard / name
        digest = _sha(p)
        sealed = freeze.get("files", {}).get(name, {})
        _require(sealed.get("sha256") == digest and sealed.get("bytes") == p.stat().st_size,
                 f"sealed native bytes changed:{name}")
        evidence[str(p)] = digest
    for name in ("run_manifest.json", "final_gate.json", "_RUN_COMPLETE.json", "FAILED.json"):
        evidence[str(source / name)] = _sha(source / name)
    evidence[str(standard / "freeze_manifest.json")] = _sha(standard / "freeze_manifest.json")
    contract_path = _path(native["thresholds_path"])
    contract = _json(contract_path)
    _require(_sha(contract_path) == native.get("thresholds_sha256"), "threshold file binding")
    expected_contract = {"schema_version": "four_by_four_frozen_threshold_contract_v1",
        "dataset": "Mutagenicity", "status": "PASS", "cf_mode": "strict_flip",
        "distance_line": "MolCLR-Node-Wasserstein", "theta_star": .05, "cost_cap": .0535,
        "shared_across_methods": True, "test_used_for_selection": False,
        "selection_used_test": False, "threshold_fitted_on_test": False,
        "threshold_source_split": "existing_frozen_protocol"}
    for key, value in expected_contract.items():
        _require(contract.get(key) == value, f"predeclared threshold contract:{key}")
    upstream_path = _path(contract["source_contract"])
    upstream = _json(upstream_path)
    _require(_sha(upstream_path) == contract.get("source_contract_sha256"),
             "predeclared threshold source hash")
    for key in ("thresholds", "theta_star", "cost_cap", "dataset", "cf_mode", "distance_line"):
        _require(contract.get(key) == upstream.get(key), f"threshold source identity:{key}")
    evidence.update({str(contract_path): _sha(contract_path), str(upstream_path): _sha(upstream_path)})

    from src.eval.user_approved_frozen_v4 import validate_adopted_cell
    valid, reasons, details = validate_adopted_cell(reference)
    _require(valid and details.get("dataset") == "Mutagenicity" and details.get("method") == "Ours",
             "approved original Mut/Ours reference:" + ";".join(reasons))
    ref_eval = _json(reference / "evaluation_manifest.json")
    ref_rows = _rows(reference / "figure4_coverage_vs_threshold.csv")
    grid = [row["threshold"] for row in ref_rows]
    _require(ref_eval.get("figure4_k") == 10 and ref_eval.get("figure4_points") == 601
             and len(grid) == 601 and {int(row["k"]) for row in ref_rows} == {10},
             "approved Figure4 K10 protocol")
    _require(_grid_hash(grid) == ref_eval.get("threshold_config_hash") == REFERENCE_GRID_SHA,
             "exact preregistered reference threshold strings")
    native_grid = contract["thresholds"]
    _require(native_grid == native.get("threshold_grid") and len(native_grid) == 601,
             "native frozen threshold vector")
    quantum = Decimal("1e-15")
    for i, (old, new) in enumerate(zip(native_grid, grid)):
        nominal = (Decimal(i) * Decimal("0.0535") / 600).quantize(quantum)
        _require(Decimal(str(old)).quantize(quantum) == Decimal(new).quantize(quantum) == nominal,
                 "non-preregistered grid point")
    for name in ("run_manifest.json", "evaluation_manifest.json", "figure4_coverage_vs_threshold.csv",
                 "registry_exception.json"):
        evidence[str(reference / name)] = _sha(reference / name)
    return {"native": native, "grid": grid, "native_grid": native_grid,
            "threshold_legacy_label_unchanged": contract.get("threshold_config_hash")}, evidence


def _aggregate(source: Path, inputs: Mapping[str, Any]) -> tuple[bytes, list[int]]:
    standard = source / "standardized"
    native = inputs["native"]
    parents = _rows(standard / "parent_best_distances.csv")
    groups = {k: [row for row in parents if int(row["k"]) == k] for k in range(1, 21)}
    ids = [row["parent_id"] for row in groups[1]]
    # The shared evaluator serializes its parent identifiers as strings.
    cohort_hash = hashlib.sha256(json.dumps(ids, sort_keys=True, ensure_ascii=True,
                                 separators=(",", ":")).encode()).hexdigest()
    _require(len(parents) == 20 * PARENT_COUNT and len(ids) == len(set(ids)) == PARENT_COUNT
             and native.get("parent_count") == PARENT_COUNT
             and native.get("parent_ids_sha256") == cohort_hash, "frozen parent cohort")
    prefixes = _rows(standard / "figure3_coverage_vs_k.csv")
    _require([int(row["k"]) for row in prefixes] == list(range(1, 21)), "prefix budget grid")
    distances: dict[int, list[float]] = {}
    null_prefixes = []
    for k, rows in groups.items():
        _require([row["parent_id"] for row in rows] == ids, f"cohort/order at K{k}")
        finite = []
        for row in rows:
            _require(int(row["requested_k"]) == k, "requested budget changed")
            raw = row["best_distance"]
            available = row["strict_recourse_available"]
            _require(available in ("True", "False") and (available == "True") == (raw != ""),
                     "strict recourse/missing-cost contradiction")
            if raw:
                distance = float(raw)
                _require(math.isfinite(distance) and distance >= 0, "invalid sealed distance")
                _require(row["theta_star_covered"] == str(distance <= .05), "parent coverage contradiction")
                finite.append(distance)
            else:
                _require(row["theta_star_covered"] == "False", "missing cost claims coverage")
        distances[k] = finite
        prefix = prefixes[k - 1]
        covered = sum(value <= .05 for value in finite)
        _require(int(prefix["num_parents"]) == PARENT_COUNT
                 and int(prefix["num_any_strict_flip_parents"]) == len(finite)
                 and int(prefix["num_close_cf_covered"]) == covered
                 and float(prefix["close_cf_coverage"]) == covered / PARENT_COUNT,
                 f"prefix/parent aggregate mismatch K{k}")
        if not finite:
            _require(prefix["conditional_median_cost"] == prefix["conditional_mean_cost"] == "",
                     "undefined conditional costs must remain missing")
            null_prefixes.append(k)
        else:
            _require(float(prefix["conditional_median_cost"]) == statistics.median(finite)
                     and math.isfinite(float(prefix["conditional_mean_cost"])),
                     "conditional median changed or finite cost missing")
    table = _rows(standard / "table2_comrecgc_k10.csv")
    _require(len(table) == 1 and int(table[0]["k"]) == 10
             and float(table[0]["coverage"]) == float(prefixes[9]["close_cf_coverage"])
             and table[0]["conditional_median_cost"] == prefixes[9]["conditional_median_cost"],
             "frozen Table2 K10 identity")
    old_curve = _rows(standard / "figure4_coverage_vs_threshold.csv")
    _require(len(old_curve) == 601 and {int(row["k"]) for row in old_curve} == {20}
             and [float(row["threshold"]) for row in old_curve] == inputs["native_grid"],
             "original Figure4 is not the sealed K20 export")
    result = []
    for i, text in enumerate(inputs["grid"]):
        old, new = float(inputs["native_grid"][i]), float(text)
        # Per-parent decisions, not equal aggregate counts or a numeric tolerance.
        _require(all((distance <= old) == (distance <= new) for distance in distances[10]),
                 f"K10 threshold crossing at point {i}")
        old_count = sum(distance <= old for distance in distances[20])
        _require(int(old_curve[i]["num_close_cf_covered"]) == old_count
                 and float(old_curve[i]["close_cf_coverage"]) == old_count / PARENT_COUNT,
                 "original K20 aggregate differs from frozen records")
        count = sum(distance <= new for distance in distances[10])
        result.append({"dataset": "Mutagenicity", "method": "ComRecGC", "k": 10,
                       "threshold": text, "coverage": count / PARENT_COUNT,
                       "num_close_cf_covered": count, "num_parents": PARENT_COUNT})
    return _csv_bytes(result), null_prefixes


def _artifacts(source: Path, reference: Path, projection: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    inputs, evidence = _source_inputs(source, reference)
    curve, null_prefixes = _aggregate(source, inputs)
    native = inputs["native"]
    common = {"schema_version": SCHEMA, "dataset": "Mutagenicity", "method": "ComRecGC",
        "status": "PASS", "frozen": True, "finalized": True, "run_complete": True,
        "raw_output_root": str(source), "raw_output_complete": True,
        "oracle_backend": "rf", "classifier_family": "random_forest", "rf_oracle_used": True,
        "oracle_checkpoint": native["teacher_path"], "oracle_hash": native["teacher_sha256"],
        "dataset_hash": native["dataset_csv_sha256"], "split_hash": native["parent_ids_sha256"],
        "molclr_checkpoint_hash": native["molclr_checkpoint_sha256"],
        "threshold_config_hash": REFERENCE_GRID_SHA,
        "threshold_identity_basis": "exact ordered raw strings from approved Mut/Ours Figure4 K10",
        "project_commit": SCIENCE_COMMIT, "distance_line": "MolCLR-Node-Wasserstein",
        "cf_mode": "strict_flip", "k_max": 20, "table2_k": 10, "figure4_k": 10,
        "figure4_points": 601, "theta_star": .05, "cost_cap": .0535,
        "source_label": native["source_label"], "test_used_for_selection": False,
        "threshold_fitted_on_test": False, "projection_receipt": str(projection / "projection.json"),
        "outer_rf_and_completeness_proof": str(source / "final_gate.json"),
        "outer_rf_and_completeness_proof_sha256": _sha(source / "final_gate.json"), **FLAGS}
    artifacts = {name: (source / "standardized" / name).read_bytes() for name in COPIED}
    artifacts["figure4_coverage_vs_threshold.csv"] = curve
    artifacts["destination_distribution.csv"] = _csv_bytes([{
        "dataset": "Mutagenicity", "method": "ComRecGC", "destination_label": "",
        "count": "", "rate": "", "status": "NOT_RECORDED_IN_NATIVE_EXPORT"}])
    for name in ("run_manifest.json", "summary.json", "oracle_manifest.json", "evaluation_manifest.json"):
        artifacts[name] = _json_bytes(common)
    artifacts["final_artifact_audit.json"] = _json_bytes({**common, "passed": True,
        "audit_passed": True, "undefined_conditional_cost_prefixes": null_prefixes,
        "null_cost_basis": "zero authoritative strict-flip parents and zero coverage; no imputation"})
    return artifacts, {"source_files": evidence, "undefined_conditional_cost_prefixes": null_prefixes,
        "native_figure4_k": 20, "projected_figure4_k": 10, "threshold_count": 601,
        "parent_count": PARENT_COUNT, "no_threshold_crossing": True,
        "threshold_legacy_label_unchanged": inputs["threshold_legacy_label_unchanged"]}


def _validate_contents(root: Path) -> dict[str, Any]:
    root = _path(root, directory=True)
    receipt = _json(root / "projection.json")
    _require(receipt.get("schema_version") == SCHEMA and receipt.get("status") == "PASS",
             "typed projection receipt")
    _validate_driver(receipt)
    _require({p.name for p in root.iterdir()} == {"projection.json", "standardized"},
             "unexpected projection root diagnostic or artifact")
    source = _path(receipt["terminal_root"], directory=True)
    reference = _path(receipt["reference_standardized_root"], directory=True)
    _require(receipt.get("standardized_root") == str(root / "standardized"), "projection locator")
    artifacts, proof = _artifacts(source, reference, root)
    _require(all(receipt.get(k) == v for k, v in {**FLAGS, **proof}.items()), "projection proof changed")
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()}
    _require(receipt.get("projected_files") == expected, "projection inventory changed")
    _require({p.name for p in (root / "standardized").iterdir()} == set(artifacts),
             "unexpected or missing projected files")
    for name, data in artifacts.items():
        _require(_path(root / "standardized" / name).read_bytes() == data,
                 f"non-deterministic projected artifact:{name}")
    return receipt


def validated_undefined_cost_prefixes(standardized_root: Path) -> set[int]:
    """Only this typed, fully replayed projection permits missing conditional costs."""
    run = _json(standardized_root / "run_manifest.json")
    if run.get("schema_version") != SCHEMA:
        return set()
    receipt = _validate_contents(standardized_root.parent)
    return set(receipt["undefined_conditional_cost_prefixes"])


def validate_registry_projection(projection_root: str | Path, *, terminal: Mapping[str, Any],
        reference: Mapping[str, Any], startup_repair_receipt: str | Path | None,
        proc_root: str | Path = "/proc") -> dict[str, Any]:
    """Called only after the original independent full terminal validation succeeds."""
    receipt = _validate_contents(Path(projection_root))
    _require(terminal.get("terminal_kind") == "MUT_FAST_ACCURATE_STANDARDIZATION_FINAL"
             and terminal.get("root") == receipt["terminal_root"]
             and terminal.get("startup_failure_supersession") is not None,
             "full original terminal validation required")
    for key, name in (("run_manifest_sha256", "run_manifest.json"),
                      ("final_gate_sha256", "final_gate.json"),
                      ("run_complete_sha256", "_RUN_COMPLETE.json")):
        _require(terminal.get(key) == receipt["source_files"].get(str(Path(receipt["terminal_root"]) / name)),
                 "strict terminal evidence binding:" + key)
    _require(startup_repair_receipt is not None
             and str(_path(startup_repair_receipt)) == receipt["startup_repair_receipt"]
             and _sha(Path(startup_repair_receipt)) == receipt["startup_repair_receipt_sha256"],
             "explicit startup repair binding")
    _require(reference.get("dataset") == "Mutagenicity" and reference.get("method") == "Ours"
             and reference.get("standardized_output_root") == receipt["reference_standardized_root"]
             and reference.get("threshold_config_hash") == REFERENCE_GRID_SHA,
             "matrix authoritative reference binding")
    from src.eval.non_taste_matrix_append import _writer_audit
    writer = _writer_audit(Path(projection_root), proc_root=proc_root, required=True)
    return {**receipt, "projection_receipt_sha256": _sha(Path(projection_root) / "projection.json"),
            "writer_audit": writer}


def create_registry_projection(*, terminal_root: str | Path, reference_standardized_root: str | Path,
        startup_repair_receipt: str | Path, output_root: str | Path,
        proc_root: str | Path = "/proc") -> dict[str, Any]:
    source = _path(terminal_root, directory=True)
    reference = _path(reference_standardized_root, directory=True)
    output = Path(output_root)
    _require(output.is_absolute() and not output.exists() and not output.is_symlink(), "fresh absolute output required")
    _require(output.resolve() == output and not any(output == p or p in output.parents or output in p.parents
             for p in (source, reference)), "output overlaps original evidence")
    from src.eval.non_taste_matrix_append import _validate_mut_fast_accurate_terminal
    terminal = _validate_mut_fast_accurate_terminal(source, proc_root=proc_root,
        require_writer_audit=True, startup_repair_receipt=startup_repair_receipt)
    artifacts, proof = _artifacts(source, reference, output)
    receipt = {"schema_version": SCHEMA, "status": "PASS", "terminal_root": str(source),
        "reference_standardized_root": str(reference), "standardized_root": str(output / "standardized"),
        "startup_repair_receipt": str(_path(startup_repair_receipt)),
        "startup_repair_receipt_sha256": _sha(Path(startup_repair_receipt)),
        **_driver_identity(),
        "projected_files": {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
        **FLAGS, **proof}
    (output / "standardized").mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        _write_new(output / "standardized" / name, data)
    _sync_directory(output / "standardized")
    _write_new(output / "projection.json", _json_bytes(receipt))
    _sync_directory(output)
    _sync_directory(output.parent)
    result = validate_registry_projection(output, terminal=terminal,
        reference={"dataset": "Mutagenicity", "method": "Ours", "standardized_output_root": str(reference),
                   "threshold_config_hash": REFERENCE_GRID_SHA},
        startup_repair_receipt=startup_repair_receipt, proc_root=proc_root)
    from src.eval.four_by_four_registry import audit_explicit_candidate, CellStatus
    audit = audit_explicit_candidate(output / "standardized", dataset="Mutagenicity", method="ComRecGC",
        expectations={"datasets": {"Mutagenicity": {"threshold_config_hash": REFERENCE_GRID_SHA}}})
    if audit.status != CellStatus.FROZEN_PASS:
        _write_new(output / "projection_creation_failed.json", _json_bytes({
            "status": "FAILED", "ordinary_registry_status": audit.status.value,
            "reason_codes": audit.reason_codes, "original_sources_modified": False}))
        _sync_directory(output)
        _require(False, "ordinary registry audit:" + ";".join(audit.reason_codes))
    return {**result, "ordinary_registry_status": audit.status.value}
