#!/usr/bin/env python3
"""Re-render the combined AIDS/MUT figures from frozen presentation CSVs only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.paper import plot_aids_mut_gcf_style as renderer  # noqa: E402


FIGURE3_FILENAME = f"{renderer.FIGURE3_OUTPUT_STEM}_data.csv"
FIGURE4_FILENAME = f"{renderer.FIGURE4_OUTPUT_STEM}_data.csv"
TABLE2_FILENAME = f"{renderer.TABLE2_OUTPUT_STEM}.csv"
SOURCE_FILENAMES = (FIGURE3_FILENAME, FIGURE4_FILENAME, TABLE2_FILENAME)
EXPECTED_DATASETS = ("AIDS", "Mutagenicity")
EXPECTED_FIGURE3_ROWS = len(EXPECTED_DATASETS) * len(renderer.METHODS) * 20
EXPECTED_FIGURE4_ROWS = (
    len(EXPECTED_DATASETS)
    * len(renderer.METHODS)
    * renderer.AIDS_FIGURE4_POINTS_PER_METHOD
)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str], *, source: Path) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"{source}: {column} contains NaN or infinity.")
        frame[column] = converted


def load_figure3(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = ["Dataset", "Method", "K", "Theta", "Coverage", "Cost"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{path}: expected columns {expected_columns}, got {list(frame.columns)}")
    if len(frame) != EXPECTED_FIGURE3_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_FIGURE3_ROWS} rows, got {len(frame)}")
    _finite_numeric(frame, ("K", "Theta", "Coverage", "Cost"), source=path)
    if set(frame["Dataset"]) != set(EXPECTED_DATASETS):
        raise ValueError(f"{path}: datasets must be {EXPECTED_DATASETS}")
    if frame.duplicated(["Dataset", "Method", "K"]).any():
        raise ValueError(f"{path}: duplicate Dataset/Method/K rows")
    for dataset in EXPECTED_DATASETS:
        for method in renderer.METHODS:
            rows = frame.loc[(frame["Dataset"] == dataset) & (frame["Method"] == method)]
            if rows["K"].astype(int).tolist() != list(range(1, 21)):
                raise ValueError(f"{path}: {dataset}/{method} K order must be exactly 1..20")
    if not np.allclose(frame["Theta"], renderer.AIDS_FIGURE3_THETA, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path}: every Figure 3 theta must equal 0.05")
    if not frame["Coverage"].between(0.0, 1.0).all():
        raise ValueError(f"{path}: coverage must be in [0, 1]")
    if (frame["Cost"] < 0.0).any():
        raise ValueError(f"{path}: cost must be non-negative")
    return frame


def load_figure4(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected_columns = ["Dataset", "Method", "K", "Threshold", "Coverage"]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{path}: expected columns {expected_columns}, got {list(frame.columns)}")
    if len(frame) != EXPECTED_FIGURE4_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_FIGURE4_ROWS} rows, got {len(frame)}")
    _finite_numeric(frame, ("K", "Threshold", "Coverage"), source=path)
    if set(frame["Dataset"]) != set(EXPECTED_DATASETS):
        raise ValueError(f"{path}: datasets must be {EXPECTED_DATASETS}")
    if frame.duplicated(["Dataset", "Method", "Threshold"]).any():
        raise ValueError(f"{path}: duplicate Dataset/Method/Threshold rows")
    if not np.allclose(frame["K"], renderer.AIDS_FIGURE4_K, rtol=0.0, atol=0.0):
        raise ValueError(f"{path}: every Figure 4 K must equal 10")
    if not frame["Coverage"].between(0.0, 1.0).all():
        raise ValueError(f"{path}: coverage must be in [0, 1]")

    reference_grid: np.ndarray | None = None
    for dataset in EXPECTED_DATASETS:
        for method in renderer.METHODS:
            rows = frame.loc[(frame["Dataset"] == dataset) & (frame["Method"] == method)]
            grid = rows["Threshold"].to_numpy(dtype=float)
            if len(grid) != renderer.AIDS_FIGURE4_POINTS_PER_METHOD:
                raise ValueError(f"{path}: {dataset}/{method} must contain 601 thresholds")
            if not np.isclose(grid[0], renderer.AIDS_FIGURE4_THRESHOLD_MIN, atol=1e-15):
                raise ValueError(f"{path}: {dataset}/{method} threshold minimum is not 0.0")
            if not np.isclose(grid[-1], renderer.AIDS_FIGURE4_THRESHOLD_MAX, atol=1e-15):
                raise ValueError(f"{path}: {dataset}/{method} threshold maximum is not 0.0535")
            if np.any(np.diff(grid) <= 0):
                raise ValueError(f"{path}: {dataset}/{method} thresholds are not increasing")
            if reference_grid is None:
                reference_grid = grid
            elif not np.allclose(grid, reference_grid, rtol=0.0, atol=1e-15):
                raise ValueError(f"{path}: threshold grids differ across datasets or methods")
    return frame


def load_table2(path: Path, figure3: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.read_csv(path)
    expected_columns = ["Method"] + [
        f"{dataset} {metric}"
        for dataset in renderer.DATASET_ORDER
        for metric in ("Coverage", "Cost")
    ]
    if list(matrix.columns) != expected_columns:
        raise ValueError(f"{path}: expected columns {expected_columns}, got {list(matrix.columns)}")
    if matrix["Method"].tolist() != list(renderer.METHODS):
        raise ValueError(f"{path}: method order must be {renderer.METHODS}")
    for dataset in ("NCI1", "Proteins"):
        if matrix[[f"{dataset} Coverage", f"{dataset} Cost"]].notna().any().any():
            raise ValueError(f"{path}: {dataset} cells must remain empty")

    rows: list[dict[str, Any]] = []
    for _, row in matrix.iterrows():
        method = str(row["Method"])
        for dataset in EXPECTED_DATASETS:
            coverage = float(row[f"{dataset} Coverage"])
            cost = float(row[f"{dataset} Cost"])
            if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
                raise ValueError(f"{path}: invalid {dataset}/{method} coverage")
            if not np.isfinite(cost) or cost < 0.0:
                raise ValueError(f"{path}: invalid {dataset}/{method} cost")
            k10 = figure3.loc[
                (figure3["Dataset"] == dataset)
                & (figure3["Method"] == method)
                & (figure3["K"] == renderer.AIDS_TABLE2_K)
            ]
            if len(k10) != 1:
                raise ValueError(f"{path}: missing unique Figure 3 K=10 row for {dataset}/{method}")
            if not np.isclose(coverage, float(k10.iloc[0]["Coverage"]), rtol=0.0, atol=1e-12):
                raise ValueError(f"{path}: {dataset}/{method} coverage differs from Figure 3 K=10")
            if not np.isclose(cost, float(k10.iloc[0]["Cost"]), rtol=0.0, atol=1e-12):
                raise ValueError(f"{path}: {dataset}/{method} cost differs from Figure 3 K=10")
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "K": renderer.AIDS_TABLE2_K,
                    "Theta": renderer.AIDS_TABLE2_THETA,
                    "Coverage": coverage,
                    "Cost": cost,
                }
            )
    return pd.DataFrame(rows)


def load_frozen_csvs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    paths = [input_dir / name for name in SOURCE_FILENAMES]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing non-empty frozen plotting CSV: {path}")
    figure3 = load_figure3(paths[0])
    figure4 = load_figure4(paths[1])
    table = load_table2(paths[2], figure3)
    return figure3, figure4, table


def _source_inventory(input_dir: Path) -> dict[str, Any]:
    declared: dict[str, Any] = {}
    source_manifest = input_dir / "combined_manifest.json"
    if source_manifest.is_file():
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("outputs"), dict):
            declared = payload["outputs"]
    inventory: dict[str, Any] = {}
    for filename in SOURCE_FILENAMES:
        path = input_dir / filename
        actual_sha = renderer.sha256(path)
        entry = declared.get(filename, {}) if isinstance(declared, dict) else {}
        declared_sha = entry.get("sha256") if isinstance(entry, dict) else None
        inventory[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "source_manifest_declared_sha256": declared_sha,
            "source_manifest_sha256_matches": declared_sha == actual_sha,
        }
    return inventory


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-render AIDS/MUT GCF-style plots from frozen combined CSV values only."
    )
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure3-coverage-ymax", type=float, default=90.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    input_dir = _resolve(project_root, args.input_dir)
    output_dir = _resolve(project_root, args.output_dir)
    if input_dir == output_dir:
        raise ValueError("Input and output directories must differ.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not np.isclose(args.figure3_coverage_ymax, 90.0, rtol=0.0, atol=0.0):
        raise ValueError("The V3 replay contract requires --figure3-coverage-ymax 90.")

    source_inventory_before = _source_inventory(input_dir)
    figure3, figure4, table = load_frozen_csvs(input_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent)))
    try:
        renderer.render_figure3(
            figure3,
            temp_dir,
            mut_matches_aids=True,
            coverage_ymax=args.figure3_coverage_ymax,
        )
        renderer.render_figure4(figure4, temp_dir, mut_matches_aids=True)
        renderer.write_table_outputs(table, temp_dir)

        for filename in SOURCE_FILENAMES:
            shutil.copyfile(input_dir / filename, temp_dir / filename)

        source_inventory_after = _source_inventory(input_dir)
        if source_inventory_before != source_inventory_after:
            raise RuntimeError("Frozen source CSVs changed during rendering.")

        audit_lines = [
            "AIDS + Mutagenicity GCF-style CSV replay audit",
            f"source_root={input_dir}",
            f"distance_line={renderer.DISTANCE_LABEL}",
            f"distance_type={renderer.DISTANCE_TYPE}",
            f"cf_mode={renderer.CF_MODE}",
            f"figure3_rows={len(figure3)}",
            f"figure3_theta={renderer.AIDS_FIGURE3_THETA}",
            f"figure3_coverage_ymax={args.figure3_coverage_ymax:g}",
            f"figure4_rows={len(figure4)}",
            f"figure4_points_per_method={renderer.AIDS_FIGURE4_POINTS_PER_METHOD}",
            f"figure4_threshold_min={renderer.AIDS_FIGURE4_THRESHOLD_MIN}",
            f"figure4_threshold_max={renderer.AIDS_FIGURE4_THRESHOLD_MAX}",
            f"table2_rows={len(table)}",
            "distance_recomputed=false",
            "teacher_recomputed=false",
            "candidate_ranking_recomputed=false",
            "source_csv_values_changed=false",
            "[AIDS_MUT_WNODE_GCF_STYLE_CSV_REPLAY_V3_OK]",
        ]
        (temp_dir / "combined_audit_report.txt").write_text(
            "\n".join(audit_lines) + "\n", encoding="utf-8"
        )
        manifest: dict[str, Any] = {
            "schema_version": "aids_mut_gcf_style_csv_replay_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(input_dir),
            "source_csv_inventory": source_inventory_before,
            "source_manifest_status": "advisory_not_used_as_numeric_source",
            "render_only": True,
            "figure3_coverage_ymax": args.figure3_coverage_ymax,
            "distance_line": renderer.DISTANCE_LABEL,
            "distance_type": renderer.DISTANCE_TYPE,
            "cf_mode": renderer.CF_MODE,
            "distance_recomputed": False,
            "teacher_recomputed": False,
            "candidate_ranking_recomputed": False,
            "candidate_order_changed": False,
            "selection_performed_in_plot": False,
            "outputs": {},
        }
        for path in sorted(temp_dir.iterdir()):
            if path.is_file() and path.name not in {"combined_manifest.json", "_RUN_COMPLETE.json"}:
                manifest["outputs"][path.name] = {
                    "bytes": path.stat().st_size,
                    "sha256": renderer.sha256(path),
                }
        _write_json(temp_dir / "combined_manifest.json", manifest)
        _write_json(
            temp_dir / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "render_only": True,
                "distance_line": renderer.DISTANCE_LABEL,
                "distance_type": renderer.DISTANCE_TYPE,
                "cf_mode": renderer.CF_MODE,
                "source_root": str(input_dir),
                "manifest_sha256": renderer.sha256(temp_dir / "combined_manifest.json"),
            },
        )
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"output_dir={output_dir}")
    print("[AIDS_MUT_WNODE_GCF_STYLE_CSV_REPLAY_V3_OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
