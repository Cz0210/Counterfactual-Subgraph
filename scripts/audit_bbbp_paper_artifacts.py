#!/usr/bin/env python3
"""Audit the four standardized BBBP paper artifact roots as one frozen set."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bbbp_paper_artifacts import (  # noqa: E402
    CF_MODE,
    DISTANCE_LINE,
    FIGURE3_FIELDS,
    FIGURE4_FIELDS,
    TABLE2_FIELDS,
    load_bbbp_thresholds,
    sha256_file,
)


METHODS = (
    ("ours", "Ours"),
    ("globalgce", "GlobalGCE"),
    ("gcfexplainer", "GCFExplainer"),
    ("comrecgc", "COMRECGC"),
)
REQUIRED_JSON = (
    "summary.json",
    "run_manifest.json",
    "protocol_manifest.json",
    "split_manifest.json",
    "split_leakage_audit.json",
    "candidate_lineage_audit.json",
    "final_artifact_audit.json",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _finite_rate(value: Any, *, field: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {path}: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite within [0, 1] in {path}: {number}")
    return number


def _optional_nonnegative(value: Any, *, field: str, path: Path) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} in {path}: {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative in {path}: {number}")
    return number


def _same_grid(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15)
        for a, b in zip(left, right, strict=True)
    )


def _write_frozen_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Existing combined BBBP artifact differs: {path}")
        return
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


def audit_bbbp_artifacts(
    root: str | Path,
    *,
    methods: Sequence[tuple[str, str]] = METHODS,
    thresholds_path: str | Path | None = None,
    write_combined: bool = True,
) -> dict[str, Any]:
    paper_root = Path(root).expanduser().resolve()
    selected_methods = tuple(methods)
    if not selected_methods:
        raise ValueError("At least one BBBP method is required for artifact audit.")
    known = dict(METHODS)
    for slug, display in selected_methods:
        if known.get(slug) != display:
            raise ValueError(f"Unsupported BBBP method identity: {slug}/{display}")
    threshold_path = (
        Path(thresholds_path).expanduser().resolve()
        if thresholds_path is not None
        else paper_root / "thresholds.json"
    )
    threshold_contract = load_bbbp_thresholds(threshold_path)
    expected_thresholds = [float(value) for value in threshold_contract["thresholds"]]
    method_audits: list[dict[str, Any]] = []
    reference_parent_count: int | None = None
    reference_parent_ids_sha256: str | None = None
    reference_teacher_path: str | None = None
    reference_molclr_checkpoint: str | None = None
    combined_figure3: list[dict[str, str]] = []
    combined_figure4: list[dict[str, str]] = []
    combined_table2: list[dict[str, str]] = []
    for slug, display in selected_methods:
        direct_single = (
            len(selected_methods) == 1
            and (paper_root / "figure3_coverage_vs_k.csv").is_file()
        )
        method_root = paper_root if direct_single else paper_root / slug
        required = (
            method_root / "figure3_coverage_vs_k.csv",
            method_root / "figure4_coverage_vs_threshold.csv",
            method_root / f"table2_{slug}_k10.csv",
            *(method_root / name for name in REQUIRED_JSON),
        )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"BBBP {display} artifacts are missing/empty: {missing}")
        figure3_path, figure4_path, table_path = required[:3]
        headers3, rows3 = _read_csv(figure3_path)
        headers4, rows4 = _read_csv(figure4_path)
        headers2, rows2 = _read_csv(table_path)
        if tuple(headers3) != FIGURE3_FIELDS:
            raise ValueError(f"BBBP Figure 3 schema mismatch for {display}: {headers3}")
        if tuple(headers4) != FIGURE4_FIELDS:
            raise ValueError(f"BBBP Figure 4 schema mismatch for {display}: {headers4}")
        if tuple(headers2) != TABLE2_FIELDS:
            raise ValueError(f"BBBP Table 2 schema mismatch for {display}: {headers2}")
        if len(rows3) != 20 or [int(row["k"]) for row in rows3] != list(range(1, 21)):
            raise ValueError(f"BBBP Figure 3 K grid is not exactly 1..20 for {display}")
        if any(row["method"] != display for row in rows3 + rows4 + rows2):
            raise ValueError(f"BBBP method label changed within {method_root}")
        coverages3 = [
            _finite_rate(row["coverage"], field="coverage", path=figure3_path)
            for row in rows3
        ]
        if any(b + 1e-12 < a for a, b in zip(coverages3, coverages3[1:])):
            raise ValueError(f"BBBP Figure 3 coverage is not monotone for {display}")
        for row in rows3:
            _optional_nonnegative(row["cost"], field="cost", path=figure3_path)
        grid = [float(row["threshold"]) for row in rows4]
        if not _same_grid(grid, expected_thresholds):
            raise ValueError(f"BBBP Figure 4 threshold protocol differs for {display}")
        coverages4 = [
            _finite_rate(row["coverage"], field="coverage", path=figure4_path)
            for row in rows4
        ]
        if any(b + 1e-12 < a for a, b in zip(coverages4, coverages4[1:])):
            raise ValueError(f"BBBP Figure 4 coverage is not monotone for {display}")
        if len(rows2) != 1 or int(rows2[0]["k"]) != 10:
            raise ValueError(f"BBBP Table 2 must contain one K=10 row for {display}")
        table_coverage = _finite_rate(
            rows2[0]["coverage"], field="coverage", path=table_path
        )
        table_cost = _optional_nonnegative(
            rows2[0]["cost"], field="cost", path=table_path
        )
        figure3_k10 = rows3[9]
        if not math.isclose(
            table_coverage,
            float(figure3_k10["coverage"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"BBBP Table 2/Figure 3 K=10 coverage differs for {display}")
        figure3_k10_cost = _optional_nonnegative(
            figure3_k10["cost"], field="cost", path=figure3_path
        )
        if (table_cost is None) != (figure3_k10_cost is None) or (
            table_cost is not None
            and figure3_k10_cost is not None
            and not math.isclose(
                table_cost, figure3_k10_cost, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(f"BBBP Table 2/Figure 3 K=10 cost differs for {display}")
        summary = _read_json(method_root / "summary.json")
        manifest = _read_json(method_root / "run_manifest.json")
        final_audit = _read_json(method_root / "final_artifact_audit.json")
        if summary.get("dataset") != "BBBP" or summary.get("method") != display:
            raise ValueError(f"BBBP summary identity mismatch for {display}")
        if summary.get("distance_line") != DISTANCE_LINE or summary.get("cf_mode") != CF_MODE:
            raise ValueError(f"BBBP distance/strict-flip semantics changed for {display}")
        if manifest.get("candidate_set_preselected") is not True:
            raise ValueError(f"BBBP candidates are not marked preselected for {display}")
        if manifest.get("selection_performed_in_eval") is not False:
            raise ValueError(f"BBBP evaluator-side selection detected for {display}")
        if final_audit.get("passed") is not True:
            raise ValueError(f"BBBP final artifact audit is not passing for {display}")
        parent_count = int(summary["test_parent_count"])
        parent_ids_sha256 = str(summary.get("test_parent_ids_sha256") or "")
        if len(parent_ids_sha256) != 64:
            raise ValueError(f"BBBP parent ID hash is missing for {display}")
        if reference_parent_count is None:
            reference_parent_count = parent_count
            reference_parent_ids_sha256 = parent_ids_sha256
        elif parent_count != reference_parent_count:
            raise ValueError(
                f"BBBP parent universe differs: {display}={parent_count}, "
                f"reference={reference_parent_count}"
            )
        elif parent_ids_sha256 != reference_parent_ids_sha256:
            raise ValueError(f"BBBP parent ID universe/order differs for {display}")
        teacher_path = str(manifest.get("teacher_path") or "")
        molclr_checkpoint = str(manifest.get("molclr_checkpoint") or "")
        if not teacher_path or not molclr_checkpoint:
            raise ValueError(f"BBBP teacher/MolCLR lineage is missing for {display}")
        if reference_teacher_path is None:
            reference_teacher_path = teacher_path
            reference_molclr_checkpoint = molclr_checkpoint
        elif (
            teacher_path != reference_teacher_path
            or molclr_checkpoint != reference_molclr_checkpoint
        ):
            raise ValueError(f"BBBP teacher/MolCLR lineage differs for {display}")
        combined_figure3.extend(rows3)
        combined_figure4.extend(rows4)
        combined_table2.extend(rows2)
        method_audits.append(
            {
                "method": display,
                "root": str(method_root),
                "test_parent_count": parent_count,
                "candidate_count": int(summary["candidate_count"]),
                "test_parent_ids_sha256": parent_ids_sha256,
                "figure3_rows": len(rows3),
                "figure4_rows": len(rows4),
                "table2_k": 10,
                "files": {
                    path.name: sha256_file(path)
                    for path in required
                },
            }
        )
    combined_paths = {
        "figure3": paper_root / "figure3_coverage_vs_k.csv",
        "figure4": paper_root / "figure4_coverage_vs_threshold.csv",
        "table2": paper_root / "table2_bbbp_k10.csv",
    }
    if write_combined:
        _write_frozen_csv(combined_paths["figure3"], FIGURE3_FIELDS, combined_figure3)
        _write_frozen_csv(combined_paths["figure4"], FIGURE4_FIELDS, combined_figure4)
        _write_frozen_csv(combined_paths["table2"], TABLE2_FIELDS, combined_table2)
    return {
        "schema_version": "bbbp_common4_audit_v1",
        "passed": True,
        "dataset": "BBBP",
        "methods": [display for _slug, display in selected_methods],
        "distance_line": DISTANCE_LINE,
        "cf_mode": CF_MODE,
        "test_parent_count": reference_parent_count,
        "test_parent_ids_sha256": reference_parent_ids_sha256,
        "teacher_path": reference_teacher_path,
        "molclr_checkpoint": reference_molclr_checkpoint,
        "thresholds_json": str(threshold_path),
        "thresholds_json_sha256": sha256_file(threshold_path),
        "theta_star": float(threshold_contract["theta_star"]),
        "thresholds": expected_thresholds,
        "method_audits": method_audits,
        "figure3_schema": list(FIGURE3_FIELDS),
        "figure4_schema": list(FIGURE4_FIELDS),
        "table2_schema": list(TABLE2_FIELDS),
        "plotting_adapter_required": False,
        "combined_artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path) if write_combined else None,
                "written": bool(write_combined),
            }
            for name, path in combined_paths.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--root",
        default="outputs/hpc/eval/paper/bbbp_common3_standardized_v1",
    )
    parser.add_argument("--output-json")
    parser.add_argument(
        "--methods",
        default=",".join(slug for slug, _display in METHODS),
        help="Comma-separated method slugs to audit.",
    )
    parser.add_argument("--thresholds-json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    known = dict(METHODS)
    slugs = tuple(value.strip().lower() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(slugs) - set(known))
    if unknown:
        raise ValueError(f"Unknown BBBP method slugs: {unknown}")
    audit = audit_bbbp_artifacts(
        args.root,
        methods=tuple((slug, known[slug]) for slug in slugs),
        thresholds_path=args.thresholds_json,
        write_combined=not (args.validate_only or args.dry_run),
    )
    if args.validate_only or args.dry_run:
        print(
            json.dumps(
                {
                    **audit,
                    "status": "VALIDATED_NOT_RUN",
                    "formal_output_written": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else Path(args.root).expanduser().resolve() / "bbbp_paper_artifact_audit.json"
    )
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True), flush=True)
    print("[BBBP_PAPER_ARTIFACT_AUDIT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
