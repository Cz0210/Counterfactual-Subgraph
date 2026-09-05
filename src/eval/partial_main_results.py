"""Presentation-only AIDS/BACE staging from the unique published authority.

This does not recertify science, update the registry, or relax the final 16-cell
exporter. Existing authority adoptions (including legacy and valid-zero cells)
are disclosed, while the actually plotted CSV bytes must match their source
audit declarations. Missing or unbound cells stay PENDING, never numeric zero.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

from src.eval.four_by_four_main_results import (
    CellArtifacts, MainResultsError, PASS_STATUS_NAMES, REQUIRED_CELL_FILES,
    _atomic_json, _canonical_combined_rows, _canonical_dataset_strict,
    _canonical_method_strict, _configure_matplotlib, _declared_hashes, _field,
    _finite, _method_rows, _output_inventory, _plot_lines, _read_csv,
    _read_json_object, _table2_path, _union_fields, _validate_matrix, _write_csv,
    _write_dataset_table,
)
from src.eval.four_by_four_registry import sha256_file, stable_json_sha256


SCHEMA_VERSION = "aids_bace_partial_presentation_v1"
DATASETS = ("AIDS", "BACE")


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _figure4(path: Path, dataset: str, method: str) -> tuple[dict[str, str], ...]:
    """Validate recorded points, without imposing another dataset's grid."""
    fields, rows = _read_csv(path)
    method_field = _field(fields, ("method",), path=path)
    threshold = _field(fields, ("threshold",), path=path)
    coverage = _field(fields, ("coverage", "ccrcov", "close_cf_coverage"), path=path)
    xs, ys = [], []
    for row in rows:
        if _canonical_method_strict(row[method_field]) != method:
            raise MainResultsError(f"{path}: method identity mismatch")
        if "dataset" in row and _canonical_dataset_strict(row["dataset"]) != dataset:
            raise MainResultsError(f"{path}: dataset identity mismatch")
        xs.append(_finite(row[threshold], field=threshold, path=path))
        ys.append(_finite(row[coverage], field=coverage, path=path, rate=True))
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise MainResultsError(f"{path}: threshold order/duplicate mismatch")
    if any(b + 1e-12 < a for a, b in zip(ys, ys[1:])):
        raise MainResultsError(f"{path}: nonmonotone recorded coverage")
    return tuple(rows)


def _source_cell(row: Mapping[str, Any]) -> tuple[CellArtifacts, dict[str, Any]]:
    if row.get("status") not in PASS_STATUS_NAMES:
        raise MainResultsError("NOT_PUBLISHED_PASS_IN_AUTHORITY")
    root = Path(str(row.get("standardized_output_root") or ""))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise MainResultsError("MISSING_OR_INVALID_STANDARDIZED_ROOT")
    root = root.resolve()
    dataset, method = row["dataset"], row["method"]
    audit_path, run_path = root / "final_artifact_audit.json", root / "run_manifest.json"
    audit, run = _read_json_object(audit_path), _read_json_object(run_path)
    if audit.get("passed") is not True and audit.get("audit_passed") is not True:
        raise MainResultsError("SOURCE_FINAL_AUDIT_NOT_PASS")
    for payload in (audit, run):
        if payload.get("dataset") and _canonical_dataset_strict(payload["dataset"]) != dataset:
            raise MainResultsError("SOURCE_DATASET_IDENTITY_MISMATCH")
        if payload.get("method") and _canonical_method_strict(payload["method"]) != method:
            raise MainResultsError("SOURCE_METHOD_IDENTITY_MISMATCH")
    paths = {
        "figure3": root / "figure3_coverage_vs_k.csv",
        "figure4": root / "figure4_coverage_vs_threshold.csv",
        "table2": _table2_path(root, method),
    }
    # Frozen-v4 calls this existing hash map audited_files. This is schema
    # translation only; it creates no waiver or new scientific assertion.
    declarations = _declared_hashes([
        audit, run, {"files": audit.get("audited_files", {})},
    ])
    hashes = {audit_path.name: sha256_file(audit_path), run_path.name: sha256_file(run_path)}
    for path in paths.values():
        actual = sha256_file(path)
        if declarations.get(path.name, set()) != {actual}:
            raise MainResultsError(f"SOURCE_HASH_CLOSURE_MISMATCH:{path.name}")
        hashes[path.name] = actual
    f3, _ = _method_rows(paths["figure3"], expected_method=method, kind="figure3")
    f4 = _figure4(paths["figure4"], dataset, method)
    table, _ = _method_rows(paths["table2"], expected_method=method, kind="table2")
    for records in (f3, table):
        for record in records:
            if "dataset" in record and _canonical_dataset_strict(record["dataset"]) != dataset:
                raise MainResultsError("CSV_DATASET_IDENTITY_MISMATCH")
    summary = root / "summary.json"
    if summary.is_file():
        hashes[summary.name] = sha256_file(summary)
    cell = CellArtifacts(dataset, method, root, row, tuple(f3), f4, tuple(table), (), (), hashes)
    evidence = {
        "dataset": dataset, "method": method, "state": "ADOPTED_FROM_AUTHORITY",
        "source_root": str(root), "authority_cell_sha256": stable_json_sha256(row),
        "source_sha256": hashes, "source_audit_schema": audit.get("schema_version"),
        "schema_status": "SCHEMA_VERSION_DIFFERENCE",
        "schema_note": "Presentation-only source CSV binding; no new waiver or science recertification.",
        "absent_newer_schema_files": sorted(n for n in REQUIRED_CELL_FILES if not (root / n).is_file()),
        "existing_registry_exception": row.get("registry_exception"),
        "existing_registry_exception_hash": row.get("registry_exception_hash"),
        "existing_registry_exception_waivers": row.get("registry_exception_waivers"),
        "adoption_reason_from_authority": row.get("adoption_reason"),
        "threshold_point_count": len(f4),
        "threshold_grid_changed": False, "numeric_imputation": False,
    }
    return cell, evidence


def render_partial(root: Path, by_dataset: Mapping[str, list[CellArtifacts]]) -> None:
    plt = _configure_matplotlib()
    for dataset, cells in by_dataset.items():
        target = root / dataset.lower()
        f3 = _canonical_combined_rows(cells, kind="figure3")
        f4 = _canonical_combined_rows(cells, kind="figure4")
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
        for axis, metric, ylabel in zip(axes, ("coverage", "cost"), ("Strict-flip CCRCOV", "Reported cost (source schema)")):
            _plot_lines(axis, f3, x="k", y=metric, marker_every=[0, 4, 9, 14, 19])
            for line in axis.lines:
                if len(line.get_xdata()) == 0:
                    line.set_label(line.get_label() + " [PENDING]")
            axis.set_ylabel(ylabel)
        axes[0].set_title(f"PARTIAL — {dataset} / published-source evidence")
        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Number of global actions K")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(target / f"figure3_PARTIAL.{suffix}", dpi=180)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(6.5, 3.8))
        _plot_lines(axis, f4, x="threshold", y="coverage", marker_every=max(1, len(f4) // 24))
        for line in axis.lines:
            if len(line.get_xdata()) == 0:
                line.set_label(line.get_label() + " [PENDING]")
        axis.set(title=f"PARTIAL — {dataset} / original threshold grid", xlabel="WNode threshold", ylabel="Strict-flip CCRCOV")
        axis.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(target / f"figure4_PARTIAL.{suffix}", dpi=180)
        plt.close(fig)


def export_partial_results(
    *, matrix_authority_state: Path, output_root: Path, project_root: Path,
    renderer: Callable[[Path, Mapping[str, list[CellArtifacts]]], None] = render_partial,
) -> dict[str, Any]:
    """Copy/render only current AIDS/BACE published evidence to a fresh root."""
    state_path = matrix_authority_state.resolve(strict=True)
    state_sha = sha256_file(state_path)
    state = _read_json_object(state_path)
    authority = Path(str(state.get("latest_authority_root") or ""))
    if not authority.is_absolute():
        raise MainResultsError("AUTHORITY_POINTER_MISSING")
    matrix_path = authority.resolve(strict=True) / "matrix_status.json"
    matrix_sha = sha256_file(matrix_path)
    matrix = _read_json_object(matrix_path)
    rows, complete, _ = _validate_matrix(matrix)
    if complete != sum(row.get("status") in PASS_STATUS_NAMES for row in rows):
        raise MainResultsError("AUTHORITY_COUNT_MISMATCH")
    root = output_root.resolve(strict=False)
    protected = [state_path.parent, matrix_path.parent, project_root.resolve() / "paper"]
    for row in rows:
        for key in ("standardized_output_root", "raw_output_root"):
            path = Path(str(row.get(key) or ""))
            if path.is_absolute():
                protected.append(path.resolve(strict=False))
    if not output_root.is_absolute() or root.exists() or any(_within(root, path) for path in protected):
        raise MainResultsError("OUTPUT_MUST_BE_FRESH_OUTSIDE_PAPER_AUTHORITY_AND_SCIENCE_ROOTS")
    by_dataset: dict[str, list[CellArtifacts]] = {dataset: [] for dataset in DATASETS}
    evidence, status_rows = [], []
    for row in rows:
        base = {"dataset": row["dataset"], "method": row["method"], "authority_status": row["status"]}
        if row["dataset"] not in DATASETS:
            status_rows.append({**base, "partial_state": "PENDING" if row["status"] not in PASS_STATUS_NAMES else "NOT_INCLUDED_PARTIAL_SCOPE", "reason": "NOT_IN_AIDS_BACE_PARTIAL_SCOPE"})
            continue
        try:
            cell, receipt = _source_cell(row)
        except (MainResultsError, OSError, ValueError) as exc:
            receipt = {**base, "state": "PENDING_WITH_PROVENANCE_REASON", "reason": str(exc), "source_root": row.get("standardized_output_root")}
        else:
            by_dataset[row["dataset"]].append(cell)
        evidence.append(receipt)
        status_rows.append({**base, "partial_state": receipt["state"], "reason": receipt.get("reason", "")})
    root.mkdir(parents=True, exist_ok=False)
    for dataset, cells in by_dataset.items():
        target = root / dataset.lower()
        target.mkdir()
        for kind, filename in (("figure3", "figure3_coverage_vs_k.csv"), ("figure4", "figure4_coverage_vs_threshold.csv"), ("table2", "table2_k10.csv")):
            combined = _canonical_combined_rows(cells, kind=kind)
            _write_csv(target / filename, combined, _union_fields(combined, ("dataset", "method", "k", "threshold", "coverage", "cost")) or ("dataset", "method"))
        _write_dataset_table(target, f"PARTIAL — {dataset}", _canonical_combined_rows(cells, kind="table2"))
        for cell in cells:
            copied = target / "source_csv" / cell.method.lower()
            copied.mkdir(parents=True)
            for name, digest in cell.source_hashes.items():
                source = cell.root / name
                if sha256_file(source) != digest:
                    raise MainResultsError("SOURCE_CHANGED_DURING_EXPORT")
                shutil.copyfile(source, copied / name)
                if sha256_file(copied / name) != digest:
                    raise MainResultsError("SOURCE_COPY_HASH_MISMATCH")
    renderer(root, by_dataset)
    _write_csv(root / "cell_status_PARTIAL.csv", status_rows, ("dataset", "method", "authority_status", "partial_state", "reason"))
    if sha256_file(matrix_path) != matrix_sha or sha256_file(state_path) != state_sha:
        raise MainResultsError("AUTHORITY_CHANGED_DURING_EXPORT_RERUN_FROM_FRESH_ROOT")
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "PARTIAL", "title": "PARTIAL — AIDS and BACE published main-table evidence",
        "matrix_complete_cells": complete, "matrix_total_cells": 16,
        "matrix_authority_state": str(state_path), "matrix_authority_state_sha256": state_sha,
        "matrix_status_path": str(matrix_path), "matrix_status_sha256": matrix_sha,
        "included_datasets": list(DATASETS), "rendered_cells": sum(map(len, by_dataset.values())),
        "source_cells": evidence, "cell_status": status_rows,
        "numeric_imputation": False, "science_recomputed": False, "new_waivers_created": False,
        "matrix_modified": False, "paper_modified": False, "final_16_of_16_claimed": False,
        "output_files": _output_inventory(root, exclude={"partial_manifest.json"}),
    }
    _atomic_json(root / "partial_manifest.json", manifest)
    return manifest
