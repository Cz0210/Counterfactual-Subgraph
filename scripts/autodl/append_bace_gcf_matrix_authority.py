#!/usr/bin/env python3
"""Strictly append one frozen BACE/GCFExplainer cell to a 7/16 authority."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.am_legacy_standardization import scan_live_writers  # noqa: E402
from src.eval.four_by_four_registry import (  # noqa: E402
    AuditConfig,
    CellStatus,
    DATASETS,
    METHODS,
    PASS_STATUSES,
    SCHEMA_VERSION,
    audit_registry,
    stable_json_sha256,
    write_registry_outputs,
)


APPEND_SCHEMA = "bace_gcf_matrix_authority_append_v1"
SUPERSESSION_SCHEMA = "four_by_four_matrix_supersession_v1"
TARGET_DATASET = "BACE"
TARGET_METHOD = "GCFExplainer"
EXPECTED_PRIOR_COMPLETE = 7
EXPECTED_NEW_COMPLETE = 8
PASS_MARKER = "[BACE_GCFEXPLAINER_PASS]"
MATRIX_MARKER = "[MATRIX_8_OF_16_PASS]"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MatrixAppendError(RuntimeError):
    """The proposed authority is not an exact append of the frozen predecessor."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MatrixAppendError(f"Required physical JSON file is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixAppendError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise MatrixAppendError(f"JSON object required: {path}")
    return dict(value)


def _physical_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise MatrixAppendError(f"{label} may not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise MatrixAppendError(f"{label} is absent: {path}") from exc
    if not resolved.is_dir():
        raise MatrixAppendError(f"{label} is not a directory: {resolved}")
    return resolved


def _matrix_rows(matrix: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cells = matrix.get("cells")
    if not isinstance(cells, list) or len(cells) != len(DATASETS) * len(METHODS):
        raise MatrixAppendError("Matrix authority must contain exactly 16 cells")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in cells:
        if not isinstance(raw, dict):
            raise MatrixAppendError("Matrix cell must be a JSON object")
        row = dict(raw)
        key = (str(row.get("dataset") or ""), str(row.get("method") or ""))
        if key not in {(dataset, method) for dataset in DATASETS for method in METHODS}:
            raise MatrixAppendError(f"Unsupported matrix cell identity: {key}")
        if key in rows:
            raise MatrixAppendError(f"Duplicate matrix cell identity: {key}")
        rows[key] = row
    return rows


def _passing_count(rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> int:
    values = {status.value for status in PASS_STATUSES}
    return sum(str(row.get("status") or "") in values for row in rows.values())


def _verify_authority(
    root_like: str | Path,
    *,
    expected_complete: int | None = None,
) -> dict[str, Any]:
    root = _physical_directory(Path(root_like).expanduser(), label="matrix authority")
    matrix_path = root / "matrix_status.json"
    combined_path = root / "combined_audit.json"
    matrix = _read_json(matrix_path)
    combined = _read_json(combined_path)
    if (
        matrix.get("schema_version") != SCHEMA_VERSION
        or matrix.get("audit_complete") is not True
        or matrix.get("matrix_total_cells") != 16
        or matrix.get("no_numeric_imputation") is not True
    ):
        raise MatrixAppendError(f"Matrix terminal contract changed: {matrix_path}")
    if (
        combined.get("schema_version")
        != "four_methods_four_datasets_combined_audit_v1"
        or combined.get("status") != "PASS"
        or combined.get("audit_complete") is not True
        or combined.get("matrix_total_cells") != 16
        or combined.get("source_artifacts_read_only") is not True
        or combined.get("scientific_metrics_recomputed") is not False
        or combined.get("numeric_imputation_used") is not False
    ):
        raise MatrixAppendError(f"Combined authority closure changed: {combined_path}")
    rows = _matrix_rows(matrix)
    observed_complete = _passing_count(rows)
    if matrix.get("matrix_complete_cells") != observed_complete:
        raise MatrixAppendError("Matrix complete count does not match its cell rows")
    if combined.get("matrix_complete_cells") != observed_complete:
        raise MatrixAppendError("Combined audit complete count does not match matrix")
    if expected_complete is not None and observed_complete != expected_complete:
        raise MatrixAppendError(
            f"Expected {expected_complete}/16 authority, observed {observed_complete}/16"
        )
    files = combined.get("files")
    if not isinstance(files, dict) or not files:
        raise MatrixAppendError("Combined authority has no file hash closure")
    for name, raw_identity in files.items():
        relative = Path(str(name))
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise MatrixAppendError(f"Unsafe combined-audit member: {name!r}")
        if not isinstance(raw_identity, dict):
            raise MatrixAppendError(f"Malformed combined-audit identity: {name}")
        path = root / relative
        expected_sha = str(raw_identity.get("sha256") or "")
        expected_bytes = raw_identity.get("bytes")
        if (
            path.is_symlink()
            or not path.is_file()
            or not _SHA256_RE.fullmatch(expected_sha)
            or not isinstance(expected_bytes, int)
            or path.stat().st_size != expected_bytes
            or _sha256_file(path) != expected_sha
        ):
            raise MatrixAppendError(f"Combined-audit member drifted: {name}")
    matrix_sha = _sha256_file(matrix_path)
    if combined.get("matrix_status_sha256") != matrix_sha:
        raise MatrixAppendError("Combined audit does not bind matrix_status.json")
    return {
        "root": root,
        "matrix": matrix,
        "rows": rows,
        "complete": observed_complete,
        "matrix_sha256": matrix_sha,
        "combined_sha256": _sha256_file(combined_path),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat(follow_symlinks=False)
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "sha256": _sha256_file(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "link_count": stat.st_nlink,
    }


def _verify_superseded_snapshot(root_like: str | Path) -> dict[str, Any]:
    """Bind an incomplete snapshot without promoting it to an authority.

    Historical top-level matrix_status.json predates combined-audit closure.
    It may still be truthfully marked superseded by hashing that physical file;
    the missing closure is recorded rather than invented.
    """

    root = _physical_directory(Path(root_like).expanduser(), label="superseded snapshot")
    combined_path = root / "combined_audit.json"
    if combined_path.is_file():
        closed = _verify_authority(root)
        return {
            "root": closed["root"],
            "complete": closed["complete"],
            "matrix_sha256": closed["matrix_sha256"],
            "combined_sha256": closed["combined_sha256"],
            "closure_status": "HASH_CLOSED_COMBINED_AUDIT",
            "matrix_status_identity": _file_identity(root / "matrix_status.json"),
        }
    matrix_path = root / "matrix_status.json"
    matrix = _read_json(matrix_path)
    if (
        matrix.get("schema_version") != SCHEMA_VERSION
        or matrix.get("audit_complete") is not True
        or matrix.get("matrix_total_cells") != 16
        or matrix.get("no_numeric_imputation") is not True
    ):
        raise MatrixAppendError(f"Superseded matrix snapshot contract changed: {matrix_path}")
    rows = _matrix_rows(matrix)
    complete = _passing_count(rows)
    if matrix.get("matrix_complete_cells") != complete:
        raise MatrixAppendError("Superseded snapshot count does not match its rows")
    identity = _file_identity(matrix_path)
    return {
        "root": root,
        "complete": complete,
        "matrix_sha256": identity["sha256"],
        "combined_sha256": None,
        "closure_status": "LEGACY_MATRIX_STATUS_ONLY_COMBINED_AUDIT_ABSENT",
        "matrix_status_identity": identity,
    }


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    before = root.stat(follow_symlinks=False)
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MatrixAppendError(f"Cell artifact contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        first = path.stat(follow_symlinks=False)
        digest = _sha256_file(path)
        second = path.stat(follow_symlinks=False)
        if (
            first.st_dev,
            first.st_ino,
            first.st_size,
            first.st_mtime_ns,
            first.st_ctime_ns,
        ) != (
            second.st_dev,
            second.st_ino,
            second.st_size,
            second.st_mtime_ns,
            second.st_ctime_ns,
        ):
            raise MatrixAppendError(f"Cell artifact drifted while hashing: {path}")
        result[relative] = {
            "bytes": first.st_size,
            "sha256": digest,
            "device": first.st_dev,
            "inode": first.st_ino,
            "mtime_ns": first.st_mtime_ns,
            "ctime_ns": first.st_ctime_ns,
            "link_count": first.st_nlink,
        }
    after = root.stat(follow_symlinks=False)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise MatrixAppendError("Cell artifact root drifted while hashing")
    if not result or result.get("PASS", {}).get("sha256") != hashlib.sha256(b"PASS\n").hexdigest():
        raise MatrixAppendError("BACE GCF standardized PASS marker is absent or changed")
    return result


def _git_identity(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if not re.fullmatch(r"[0-9a-f]{40}", head) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise MatrixAppendError("Execution Git identity is malformed")
    if dirty:
        raise MatrixAppendError("Matrix append requires a clean committed worktree")
    return {"commit": head, "tree": tree}


def append_bace_gcf_authority(
    *,
    prior_authority_root: str | Path,
    bace_gcf_standardized_root: str | Path,
    output_root: str | Path,
    superseded_audit_roots: Sequence[str | Path] = (),
    proc_root: str | Path = "/proc",
    require_writer_audit: bool = True,
    git_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    prior = _verify_authority(
        prior_authority_root, expected_complete=EXPECTED_PRIOR_COMPLETE
    )
    prior_rows = prior["rows"]
    target_key = (TARGET_DATASET, TARGET_METHOD)
    if str(prior_rows[target_key].get("status") or "") in {
        status.value for status in PASS_STATUSES
    }:
        raise MatrixAppendError("Prior authority already contains a passing BACE/GCFExplainer cell")

    cell_root = _physical_directory(
        Path(bace_gcf_standardized_root).expanduser(),
        label="BACE GCF standardized root",
    )
    cell_inventory = _inventory(cell_root)
    writer_audit = (
        scan_live_writers(cell_root, proc_root=proc_root)
        if require_writer_audit
        else {
            "procfs_verified": False,
            "scanned_process_count": 0,
            "writable_fd_count": 0,
            "writers": [],
        }
    )

    explicit_cells = {
        f"{dataset}/{method}": str(Path(str(row["standardized_output_root"])).resolve(strict=True))
        for (dataset, method), row in prior_rows.items()
        if str(row.get("status") or "") in {status.value for status in PASS_STATUSES}
    }
    explicit_cells[f"{TARGET_DATASET}/{TARGET_METHOD}"] = str(cell_root)
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise MatrixAppendError(f"Append authority output root must be absent: {destination}")
    result = audit_registry(
        AuditConfig(
            scan_roots=(),
            output_root=destination,
            explicit_cells=explicit_cells,
        )
    )
    current_rows = {
        (str(row["dataset"]), str(row["method"])): dict(row)
        for row in result.matrix_rows
    }
    for key, prior_row in prior_rows.items():
        if key == target_key:
            continue
        if current_rows.get(key) != prior_row:
            raise MatrixAppendError(f"Non-target matrix row drifted during append: {key}")
    if result.matrix_complete_cells != EXPECTED_NEW_COMPLETE:
        raise MatrixAppendError(
            f"Strict append must produce 8/16, observed {result.matrix_complete_cells}/16"
        )
    target = current_rows[target_key]
    if (
        target.get("status") != CellStatus.FROZEN_PASS.value
        or Path(str(target.get("standardized_output_root") or "")).resolve(strict=True)
        != cell_root
        or target.get("k_max") != 20
        or target.get("table2_k") != 10
    ):
        raise MatrixAppendError(
            "BACE GCF standardized artifact did not pass the ordinary frozen-cell gate"
        )
    ours = current_rows[(TARGET_DATASET, "Ours")]
    shared_fields = (
        "dataset_hash",
        "split_hash",
        "oracle_backend",
        "oracle_checkpoint",
        "oracle_hash",
        "molclr_checkpoint_hash",
        "distance_line",
        "cf_mode",
        "threshold_config_hash",
    )
    if any(target.get(field) != ours.get(field) for field in shared_fields):
        raise MatrixAppendError("BACE GCF is not identity-compatible with frozen BACE Ours")

    superseded: list[dict[str, Any]] = []
    superseded_evidence: list[dict[str, Any]] = []
    for root_like in superseded_audit_roots:
        stale = _verify_superseded_snapshot(root_like)
        if stale["root"] == prior["root"]:
            raise MatrixAppendError("Prior authority cannot be marked superseded as stale")
        if stale["complete"] >= EXPECTED_NEW_COMPLETE:
            raise MatrixAppendError("A complete/equal authority cannot be marked stale")
        superseded.append(
            {
                "root": str(stale["root"]),
                "observed_matrix_complete_cells": stale["complete"],
                "matrix_status_sha256": stale["matrix_sha256"],
                "combined_audit_sha256": stale["combined_sha256"],
                "closure_status": stale["closure_status"],
                "matrix_status_identity": stale["matrix_status_identity"],
                "state": "SUPERSEDED_BY_STRICT_APPEND_AUTHORITY",
                "reason": "snapshot_did_not_append_the_frozen_7of16_predecessor",
                "historical_evidence_preserved": True,
                "historical_root_modified": False,
            }
        )
        superseded_evidence.append(stale)

    execution = dict(git_identity or _git_identity())
    if set(execution) != {"commit", "tree"} or any(
        not re.fullmatch(r"[0-9a-f]{40}", str(execution[field]))
        for field in ("commit", "tree")
    ):
        raise MatrixAppendError("Execution Git identity is incomplete")
    created_at = _utc_now()
    append_manifest = {
        "schema_version": APPEND_SCHEMA,
        "status": "PASS",
        "created_at": created_at,
        "execution": execution,
        "prior_authority_root": str(prior["root"]),
        "prior_matrix_complete_cells": prior["complete"],
        "prior_matrix_status_sha256": prior["matrix_sha256"],
        "prior_combined_audit_sha256": prior["combined_sha256"],
        "prior_rows_sha256": stable_json_sha256(list(prior["matrix"]["cells"])),
        "appended_cell": {
            "dataset": TARGET_DATASET,
            "method": TARGET_METHOD,
            "standardized_output_root": str(cell_root),
            "status": target["status"],
            "registry_row": target,
            "source_inventory": cell_inventory,
            "source_inventory_sha256": stable_json_sha256(cell_inventory),
            "writer_audit": writer_audit,
        },
        "unchanged_non_target_rows": True,
        "unchanged_prior_passing_cells": EXPECTED_PRIOR_COMPLETE,
        "new_matrix_complete_cells": result.matrix_complete_cells,
        "new_matrix_total_cells": result.matrix_total_cells,
        "new_authority_root": str(destination),
        "shared_bace_identity_fields": list(shared_fields),
        "scientific_metrics_recomputed": False,
        "candidate_order_changed": False,
        "raw_test_opened": False,
        "numeric_imputation_used": False,
        "marker": MATRIX_MARKER,
    }
    supersession_manifest = {
        "schema_version": SUPERSESSION_SCHEMA,
        "status": "PASS",
        "created_at": created_at,
        "superseded_snapshots": superseded,
        "superseded_snapshot_count": len(superseded),
        "new_authority_root": str(destination),
        "new_matrix_complete_cells": EXPECTED_NEW_COMPLETE,
        "historical_roots_modified": False,
    }
    write_registry_outputs(
        result,
        destination,
        supplemental_outputs={
            "append_authority.json": _json_bytes(append_manifest),
            "superseded_snapshots.json": _json_bytes(supersession_manifest),
        },
    )
    reopened = _verify_authority(destination, expected_complete=EXPECTED_NEW_COMPLETE)
    if reopened["rows"] != current_rows:
        raise MatrixAppendError("Published matrix rows changed on independent reopen")
    published_append = _read_json(destination / "append_authority.json")
    published_supersession = _read_json(destination / "superseded_snapshots.json")
    if published_append != append_manifest or published_supersession != supersession_manifest:
        raise MatrixAppendError("Published append/supersession evidence changed")
    for before in superseded_evidence:
        after = _verify_superseded_snapshot(before["root"])
        if (
            after["complete"] != before["complete"]
            or after["matrix_sha256"] != before["matrix_sha256"]
            or after["combined_sha256"] != before["combined_sha256"]
            or after["closure_status"] != before["closure_status"]
            or after["matrix_status_identity"] != before["matrix_status_identity"]
        ):
            raise MatrixAppendError("Superseded historical snapshot changed during append")
    return {
        "status": "PASS",
        "output_root": str(destination),
        "matrix_status_path": str(destination / "matrix_status.json"),
        "matrix_status_sha256": reopened["matrix_sha256"],
        "combined_audit_sha256": reopened["combined_sha256"],
        "matrix_complete_cells": reopened["complete"],
        "matrix_total_cells": 16,
        "appended_cell": f"{TARGET_DATASET}/{TARGET_METHOD}",
        "appended_standardized_root": str(cell_root),
        "appended_source_inventory_sha256": stable_json_sha256(cell_inventory),
        "superseded_snapshot_count": len(superseded),
        "execution": execution,
    }


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"Absolute path required: {value}")
    return path.resolve(strict=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--prior-authority-root", type=_absolute_path, required=True)
    parser.add_argument("--bace-gcf-standardized-root", type=_absolute_path, required=True)
    parser.add_argument("--output-root", type=_absolute_path, required=True)
    parser.add_argument(
        "--superseded-audit-root",
        action="append",
        default=[],
        type=_absolute_path,
        help="An incomplete historical audit to reference as superseded without modifying it.",
    )
    parser.add_argument("--proc-root", type=_absolute_path, default=Path("/proc"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = append_bace_gcf_authority(
        prior_authority_root=args.prior_authority_root,
        bace_gcf_standardized_root=args.bace_gcf_standardized_root,
        output_root=args.output_root,
        superseded_audit_roots=args.superseded_audit_root,
        proc_root=args.proc_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(PASS_MARKER, flush=True)
    print(MATRIX_MARKER, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
