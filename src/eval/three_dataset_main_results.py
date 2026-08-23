"""Staging-only exporter for the complete AIDS/Mutagenicity/BACE matrix.

This module deliberately does not weaken the final four-dataset exporter.  It
accepts the canonical 16-cell registry only when the twelve non-TasteMolNet
cells are paper-pass artifacts and the four TasteMolNet cells remain blocked
by the explicit licence gate.  Reported CSV values are copied and rendered;
no scientific metric, threshold, ordering, interpolation, or imputation is
performed here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.eval.four_by_four_main_results import (
    CF_MODE,
    DATASET_SLUGS,
    DISTANCE_LINE,
    FIGURE3_MARKER_INDICES,
    IDENTITY_FIELDS,
    K_PREFIXES,
    METHOD_ORDER,
    PASS_STATUS_NAMES,
    TABLE2_K,
    CellArtifacts,
    MainResultsError,
    _atomic_json,
    _canonical_combined_rows,
    _configure_matplotlib,
    _destination_rows,
    _display_cost,
    _display_rate,
    _latex_escape,
    _output_inventory,
    _plot_lines,
    _read_json_object,
    _union_fields,
    _validate_matrix,
    _write_csv,
    _write_dataset_table,
    audit_cell,
)
from src.eval.four_by_four_registry import sha256_file, stable_json_sha256


SCHEMA_VERSION = "three_datasets_four_methods_staging_v1"
DATASET_ORDER = ("AIDS", "Mutagenicity", "BACE")
TASTE_DATASET = "TasteMolNet"
TASTE_BLOCKED_STATUS = "BLOCKED_LICENSE"
TASTE_BLOCKED_REASON = "BLOCKED_LICENSE_REVIEW"


@dataclass(frozen=True)
class ThreeDatasetExportResult:
    output_root: Path
    paper_staging_root: Path | None
    complete: bool
    matrix_complete_cells: int
    blocked_reasons: tuple[str, ...]
    generated_files: tuple[str, ...]


def _taste_is_license_blocked(row: Mapping[str, Any]) -> bool:
    if row.get("status") != TASTE_BLOCKED_STATUS:
        return False
    evidence = " ".join(
        str(row.get(field) or "")
        for field in ("rerun_reason", "blocked_reason", "reason", "adoption_reason")
    )
    return (
        TASTE_BLOCKED_REASON in evidence
        or "license" in evidence.lower()
    )


def _validate_three_dataset_boundary(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, complete_cells, all_complete = _validate_matrix(payload)
    selected = [row for row in rows if row["dataset"] in DATASET_ORDER]
    taste = [row for row in rows if row["dataset"] == TASTE_DATASET]
    actual_pass = sum(row.get("status") in PASS_STATUS_NAMES for row in rows)
    if complete_cells != actual_pass:
        raise MainResultsError(
            "matrix_complete_cells disagrees with the 16 canonical cell statuses"
        )
    if all_complete:
        raise MainResultsError(
            "Three-dataset staging may not masquerade as the final 16-cell export"
        )
    if complete_cells != 12:
        raise MainResultsError(
            f"Three-dataset staging requires exactly 12/16 paper-pass cells; got {complete_cells}/16"
        )
    if len(selected) != 12 or not all(
        row.get("status") in PASS_STATUS_NAMES for row in selected
    ):
        raise MainResultsError(
            "Every AIDS, Mutagenicity, and BACE cell must be paper-pass"
        )
    if len(taste) != 4 or not all(_taste_is_license_blocked(row) for row in taste):
        raise MainResultsError(
            "All four TasteMolNet cells must remain explicitly BLOCKED_LICENSE_REVIEW"
        )
    return selected, taste


def _validate_cross_method_identity(
    by_dataset: Mapping[str, Sequence[CellArtifacts]],
) -> None:
    failures: list[str] = []
    for dataset, cells in by_dataset.items():
        if len(cells) != 4 or {cell.method for cell in cells} != set(METHOD_ORDER):
            failures.append(f"{dataset}: exact four-method cell set is missing")
            continue
        for field in IDENTITY_FIELDS:
            values = {str(cell.row.get(field) or "") for cell in cells}
            if len(values) != 1 or "" in values:
                failures.append(
                    f"{dataset}: cross-method {field} is not one nonempty identity"
                )
        grids = [
            tuple(
                row["threshold"]
                for row in _canonical_combined_rows([cell], kind="figure4")
            )
            for cell in cells
        ]
        if len(set(grids)) != 1:
            failures.append(
                f"{dataset}: methods do not share one raw empirical threshold grid"
            )
    if failures:
        raise MainResultsError("; ".join(failures))


def _write_three_dataset_table(
    path: Path,
    by_dataset: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    lines = [
        r"\begin{tabular}{l" + "rr" * len(DATASET_ORDER) + "}",
        r"\toprule",
        "Method & "
        + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{_latex_escape(dataset)}}}"
            for dataset in DATASET_ORDER
        )
        + r" \\",
        " & " + " & ".join("CCRCOV & Cost" for _ in DATASET_ORDER) + r" \\",
        r"\midrule",
    ]
    indexed = {
        dataset: {str(row["method"]): row for row in rows}
        for dataset, rows in by_dataset.items()
    }
    for method in METHOD_ORDER:
        values: list[str] = []
        for dataset in DATASET_ORDER:
            row = indexed[dataset][method]
            values.extend((_display_rate(row["coverage"]), _display_cost(row["cost"])))
        lines.append(f"{_latex_escape(method)} & " + " & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_three_dataset_outputs(
    root: Path,
    figure3_by_dataset: Mapping[str, Sequence[Mapping[str, str]]],
    figure4_by_dataset: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Render only exact empirical rows with the established paper styling."""

    plt = _configure_matplotlib()
    for dataset in DATASET_ORDER:
        combined = root / DATASET_SLUGS[dataset] / "combined"
        fig3, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
        _plot_lines(
            axes[0], figure3_by_dataset[dataset], x="k", y="coverage",
            marker_every=list(FIGURE3_MARKER_INDICES),
        )
        _plot_lines(
            axes[1], figure3_by_dataset[dataset], x="k", y="cost",
            marker_every=list(FIGURE3_MARKER_INDICES),
        )
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
        _plot_lines(
            axis, figure4_by_dataset[dataset], x="threshold", y="coverage",
            marker_every=max(1, len(figure4_by_dataset[dataset]) // 24),
        )
        axis.set_title(dataset)
        axis.set_xlabel("WNode threshold")
        axis.set_ylabel("Strict-flip CCRCOV")
        handles, labels = axis.get_legend_handles_labels()
        fig4.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
        fig4.tight_layout(rect=(0, 0.12, 1, 1))
        fig4.savefig(combined / "figure4_coverage_vs_threshold.png", dpi=300, bbox_inches="tight")
        fig4.savefig(combined / "figure4_coverage_vs_threshold.pdf", bbox_inches="tight")
        plt.close(fig4)

    fig3, axes = plt.subplots(2, 3, figsize=(12.4, 6.3), sharex="col")
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
    fig3.savefig(root / "paper_figure3_three_datasets.pdf", bbox_inches="tight")
    plt.close(fig3)

    fig4, axes4 = plt.subplots(1, 3, figsize=(12.4, 3.8))
    for column, dataset in enumerate(DATASET_ORDER):
        _plot_lines(
            axes4[column], figure4_by_dataset[dataset], x="threshold", y="coverage",
            marker_every=max(1, len(figure4_by_dataset[dataset]) // 24),
        )
        axes4[column].set_title(dataset)
        axes4[column].set_xlabel("WNode threshold")
        if column == 0:
            axes4[column].set_ylabel("Strict-flip CCRCOV")
    handles, labels = axes4[0].get_legend_handles_labels()
    fig4.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig4.tight_layout(rect=(0, 0.12, 1, 1))
    fig4.savefig(root / "paper_figure4_three_datasets.pdf", bbox_inches="tight")
    plt.close(fig4)


def _copy_staging_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Paper staging root must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, copy_function=shutil.copy2)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def export_three_dataset_results(
    *,
    matrix_status: str | Path,
    output_root: str | Path,
    project_root: str | Path,
    paper_staging_root: str | Path | None = None,
    renderer: Callable[[Path, Mapping[str, Sequence[Mapping[str, str]]], Mapping[str, Sequence[Mapping[str, str]]]], None] = render_three_dataset_outputs,
) -> ThreeDatasetExportResult:
    matrix_path = Path(matrix_status).expanduser().resolve(strict=True)
    if not matrix_path.is_file() or matrix_path.is_symlink():
        raise MainResultsError("matrix_status.json must be a physical file")
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root must be fresh: {destination}")
    project = Path(project_root).expanduser().resolve(strict=True)
    paper = (project / "paper").resolve(strict=False)
    staging = (
        Path(paper_staging_root).expanduser().resolve(strict=False)
        if paper_staging_root is not None else None
    )
    for candidate in (destination, staging):
        if candidate is None:
            continue
        try:
            candidate.relative_to(paper)
        except ValueError:
            pass
        else:
            raise MainResultsError("Three-dataset staging may not write into paper/")
    if staging is not None and staging.exists():
        raise FileExistsError(f"Paper staging root must be fresh: {staging}")

    payload = _read_json_object(matrix_path)
    rows, taste_rows = _validate_three_dataset_boundary(payload)
    matrix_hash = sha256_file(matrix_path)
    artifacts = [audit_cell(row) for row in rows]
    by_dataset = {
        dataset: [item for item in artifacts if item.dataset == dataset]
        for dataset in DATASET_ORDER
    }
    _validate_cross_method_identity(by_dataset)

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
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "dataset": dataset,
                "methods": list(METHOD_ORDER),
                "distance_line": DISTANCE_LINE,
                "cf_mode": CF_MODE,
                "k_prefixes": list(K_PREFIXES),
                "table2_k": TABLE2_K,
                "figure4_rendering": "raw_empirical_points_no_spline_no_smoothing",
                "selection_performed_in_export": False,
                "metric_recomputation_performed": False,
                "numeric_imputation_used": False,
                "source_matrix_sha256": matrix_hash,
                "input_cells": {
                    cell.method: {
                        "standardized_output_root": str(cell.root),
                        "source_hashes": dict(sorted(cell.source_hashes.items())),
                        "matrix_identity": {field: cell.row.get(field) for field in IDENTITY_FIELDS},
                    }
                    for cell in cells
                },
            }
            _atomic_json(combined / "combined_manifest.json", manifest)

        renderer(temporary, figure3_by_dataset, figure4_by_dataset)
        for dataset in DATASET_ORDER:
            combined = temporary / DATASET_SLUGS[dataset] / "combined"
            manifest_path = combined / "combined_manifest.json"
            manifest = _read_json_object(manifest_path)
            manifest["outputs"] = _output_inventory(
                combined, exclude=("combined_manifest.json", "combined_audit.json")
            )
            _atomic_json(manifest_path, manifest)
            audit = {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "passed": True,
                "dataset": dataset,
                "methods": list(METHOD_ORDER),
                "figure3_row_count": 80,
                "figure4_raw_threshold_grid_sha256": stable_json_sha256(
                    [row["threshold"] for row in figure4_by_dataset[dataset] if row["method"] == "Ours"]
                ),
                "table2_row_count": 4,
                "clear_present": False,
                "zero_fill_used": False,
                "smoothing_used": False,
                "paper_directory_written": False,
                "combined_manifest_sha256": sha256_file(manifest_path),
                "files": _output_inventory(combined, exclude=("combined_audit.json",)),
            }
            _atomic_json(combined / "combined_audit.json", audit)

        _write_three_dataset_table(
            temporary / "paper_table2_three_datasets.tex", table2_by_dataset
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "matrix_complete_cells": 12,
            "matrix_total_cells": 16,
            "all_three_dataset_cells_complete": True,
            "final_four_dataset_export": False,
            "paper_status": "PAPER_FROZEN_PARTIAL",
            "matrix_status_path": str(matrix_path),
            "matrix_status_sha256": matrix_hash,
            "datasets": list(DATASET_ORDER),
            "methods": list(METHOD_ORDER),
            "taste_cells": taste_rows,
            "scientific_metrics_recomputed": False,
            "thresholds_selected_in_export": False,
            "smoothing_used": False,
            "numeric_imputation_used": False,
            "paper_directory_written": False,
            "outputs": _output_inventory(temporary),
        }
        _atomic_json(temporary / "three_dataset_export_manifest.json", manifest)
        audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "passed": True,
            "all_12_three_dataset_cells_verified": True,
            "taste_license_block_preserved": True,
            "final_16_cell_result_claimed": False,
            "same_oracle_split_distance_threshold_within_dataset": True,
            "strict_flip": True,
            "zero_fill_used": False,
            "paper_directory_written": False,
            "manifest_sha256": sha256_file(temporary / "three_dataset_export_manifest.json"),
            "outputs": _output_inventory(temporary, exclude=("three_dataset_export_audit.json", "THREE_DATASET_EXPORT_PASS.json", "PASS")),
        }
        _atomic_json(temporary / "three_dataset_export_audit.json", audit)
        _atomic_json(
            temporary / "THREE_DATASET_EXPORT_PASS.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "passed": True,
                "audit_sha256": sha256_file(temporary / "three_dataset_export_audit.json"),
            },
        )
        (temporary / "PASS").write_text("PASS\n", encoding="utf-8")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    if staging is not None:
        _copy_staging_tree(destination, staging)
    files = tuple(
        sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        )
    )
    return ThreeDatasetExportResult(
        output_root=destination,
        paper_staging_root=staging,
        complete=True,
        matrix_complete_cells=12,
        blocked_reasons=(),
        generated_files=files,
    )

