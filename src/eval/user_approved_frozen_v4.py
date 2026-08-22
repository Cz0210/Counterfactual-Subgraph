"""Byte-faithful adoption of the user-approved AIDS/Mutagenicity v4 tables.

This is a deliberately narrow exception path.  It does not reconstruct raw
method outputs, recompute a distance, change a candidate order, fit a selector,
or render a figure.  It accepts exactly one checksum-pinned legacy bundle and
projects only six explicitly approved dataset/method row sets into fresh,
self-auditing standardized directories.  CLEAR is always excluded and is never
treated as ComRecGC.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


APPROVAL_ID = "USER_APPROVED_FROZEN_V4"
POLICY_SCHEMA = "user_approved_frozen_v4_policy_v1"
EXCEPTION_SCHEMA = "user_approved_frozen_v4_registry_exception_v1"
SOURCE_ROOT_BASENAME = "aids_mutagenicity_wnode_gcf_style_matched_aids_v4"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
CF_MODE = "strict_flip"
SOURCE_FILES = (
    "_RUN_COMPLETE.json",
    "combined_manifest.json",
    "figure3_gcf_style_aids_mut_data.csv",
    "figure4_gcf_style_aids_mut_data.csv",
    "table2_gcf_style_aids_mut.csv",
)
NUMERIC_SOURCE_FILES = SOURCE_FILES[2:]
APPROVED_CELLS = (
    ("AIDS", "Ours"),
    ("AIDS", "GCFExplainer"),
    ("AIDS", "GlobalGCE"),
    ("Mutagenicity", "Ours"),
    ("Mutagenicity", "GCFExplainer"),
    ("Mutagenicity", "GlobalGCE"),
)
APPROVED_METHODS = ("Ours", "GCFExplainer", "GlobalGCE")
SOURCE_METHODS = ("Ours", "GlobalGCE", "CLEAR", "GCFExplainer")
METHOD_SLUGS = {
    "Ours": "ours",
    "GCFExplainer": "gcfexplainer",
    "GlobalGCE": "globalgce",
}
DATASET_SLUGS = {"AIDS": "aids", "Mutagenicity": "mutagenicity"}
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/autodl/user_approved_frozen_v4_adoption_v1.json"
)

# These are the only generic-registry failures that the explicit user
# authorization may waive.  Schema, values, methods, source hashes, output
# closure, RF backend, WNode, strict-flip, K and threshold-grid checks are never
# waived.
WAIVABLE_REGISTRY_REASONS = frozenset(
    {
        "RAW_OUTPUT_NOT_PROVEN_COMPLETE",
        "RAW_OUTPUT_COMPLETENESS_GATE_NOT_TRUE",
        "MISSING_DATASET_HASH",
        "MISSING_TEST_SPLIT_HASH",
        "MISSING_ORACLE_CHECKPOINT",
        "MISSING_ORACLE_HASH",
        "MISSING_MOLCLR_CHECKPOINT_HASH",
        "TEST_SELECTION_EXCLUSION_NOT_PROVEN",
        "TEST_THRESHOLD_EXCLUSION_NOT_PROVEN",
    }
)


class FrozenV4AdoptionError(ValueError):
    """The checksum-pinned adoption contract was not satisfied."""


@dataclass(frozen=True)
class ApprovalPolicy:
    path: Path
    sha256: str
    source_root_basename: str
    source_hashes: Mapping[str, str]
    approval_scope: tuple[str, ...]
    waivers: tuple[str, ...]


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    payloads: Mapping[str, bytes]
    inventory: Mapping[str, Mapping[str, Any]]
    figure3_rows: tuple[Mapping[str, str], ...]
    figure4_rows: tuple[Mapping[str, str], ...]
    table2_rows: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class AdoptionResult:
    output_root: Path
    cell_roots: Mapping[str, str]
    source_inventory: Mapping[str, Mapping[str, Any]]
    approval_policy_sha256: str


_SOURCE_CACHE: dict[tuple[Any, ...], SourceBundle] = {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(fields), lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _read_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenV4AdoptionError(f"invalid JSON source {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenV4AdoptionError(f"source {name} must contain one JSON object")
    return dict(value)


def _read_csv_bytes(
    data: bytes, *, name: str, expected_fields: Sequence[str]
) -> tuple[dict[str, str], ...]:
    try:
        handle = io.StringIO(data.decode("utf-8-sig"), newline="")
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise FrozenV4AdoptionError(f"invalid CSV source {name}: {exc}") from exc
    if fields != tuple(expected_fields):
        raise FrozenV4AdoptionError(
            f"{name} header mismatch: observed={fields}, expected={tuple(expected_fields)}"
        )
    if not rows:
        raise FrozenV4AdoptionError(f"{name} contains no rows")
    return rows


def _finite(raw: str, *, field: str, rate: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise FrozenV4AdoptionError(f"invalid {field}={raw!r}") from exc
    if not math.isfinite(value) or value < 0.0 or (rate and value > 1.0):
        raise FrozenV4AdoptionError(f"out-of-contract {field}={raw!r}")
    return value


def _stat_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_mode,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_approval_policy(path: str | Path | None = None) -> ApprovalPolicy:
    source = Path(path or DEFAULT_POLICY_PATH).expanduser().resolve(strict=True)
    data = source.read_bytes()
    payload = _read_json_bytes(data, name=str(source))
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise FrozenV4AdoptionError("unsupported v4 adoption policy schema")
    if payload.get("approval_id") != APPROVAL_ID:
        raise FrozenV4AdoptionError("v4 adoption policy approval_id mismatch")
    if payload.get("source_root_basename") != SOURCE_ROOT_BASENAME:
        raise FrozenV4AdoptionError("v4 adoption policy source basename mismatch")
    scope = tuple(str(item) for item in payload.get("approval_scope") or ())
    expected_scope = tuple(f"{dataset}/{method}" for dataset, method in APPROVED_CELLS)
    if scope != expected_scope:
        raise FrozenV4AdoptionError("v4 adoption policy scope is not the exact six cells")
    excluded = tuple(str(item) for item in payload.get("excluded_methods") or ())
    if set(excluded) != {"CLEAR", "ComRecGC"}:
        raise FrozenV4AdoptionError("v4 adoption policy must exclude CLEAR and ComRecGC")
    declared = payload.get("source_files")
    if not isinstance(declared, Mapping) or set(declared) != set(SOURCE_FILES):
        raise FrozenV4AdoptionError("v4 adoption policy source-file set mismatch")
    hashes: dict[str, str] = {}
    for name in SOURCE_FILES:
        metadata = declared.get(name)
        digest = str(metadata.get("sha256") if isinstance(metadata, Mapping) else "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FrozenV4AdoptionError(f"invalid policy SHA-256 for {name}")
        hashes[name] = digest
    waivers = tuple(str(item) for item in payload.get("user_approved_waivers") or ())
    if not waivers:
        raise FrozenV4AdoptionError("v4 adoption policy must enumerate its waivers")
    return ApprovalPolicy(
        path=source,
        sha256=_sha(data),
        source_root_basename=SOURCE_ROOT_BASENAME,
        source_hashes=hashes,
        approval_scope=scope,
        waivers=waivers,
    )


def _writable_source_fds(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Return Linux processes with a writable FD to an adopted source file."""

    proc = Path("/proc")
    if platform.system() != "Linux" or not proc.is_dir():
        return []
    targets = {path.resolve(): path for path in paths}
    result: list[dict[str, Any]] = []
    for pid_root in proc.iterdir():
        if not pid_root.name.isdigit():
            continue
        fd_root = pid_root / "fd"
        try:
            entries = list(fd_root.iterdir())
        except OSError:
            continue
        for fd in entries:
            try:
                target = fd.resolve(strict=True)
            except OSError:
                continue
            if target not in targets:
                continue
            try:
                info = (pid_root / "fdinfo" / fd.name).read_text(encoding="utf-8")
                flags_line = next(line for line in info.splitlines() if line.startswith("flags:"))
                flags = int(flags_line.split(":", 1)[1].strip(), 8)
            except (OSError, StopIteration, ValueError):
                continue
            if (flags & os.O_ACCMODE) in {os.O_WRONLY, os.O_RDWR}:
                result.append(
                    {"pid": int(pid_root.name), "fd": int(fd.name), "path": str(target)}
                )
    return result


def _validate_source_semantics(
    payloads: Mapping[str, bytes], policy: ApprovalPolicy
) -> tuple[
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    complete = _read_json_bytes(payloads["_RUN_COMPLETE.json"], name="_RUN_COMPLETE.json")
    combined = _read_json_bytes(payloads["combined_manifest.json"], name="combined_manifest.json")
    if complete.get("run_complete") is not True or complete.get("render_only") is not True:
        raise FrozenV4AdoptionError("v4 _RUN_COMPLETE contract is not complete/render-only")
    if complete.get("cf_mode") != CF_MODE or complete.get("distance_line") != DISTANCE_LINE:
        raise FrozenV4AdoptionError("v4 _RUN_COMPLETE scientific contract mismatch")
    if complete.get("manifest_sha256") != policy.source_hashes["combined_manifest.json"]:
        raise FrozenV4AdoptionError("v4 combined manifest binding mismatch")
    required_combined = {
        "schema_version": "aids_mut_gcf_style_csv_replay_v1",
        "render_only": True,
        "candidate_order_changed": False,
        "candidate_ranking_recomputed": False,
        "distance_recomputed": False,
        "teacher_recomputed": False,
        "selection_performed_in_plot": False,
        "cf_mode": CF_MODE,
        "distance_line": DISTANCE_LINE,
    }
    for key, expected in required_combined.items():
        if combined.get(key) != expected:
            raise FrozenV4AdoptionError(
                f"v4 combined manifest mismatch for {key}: {combined.get(key)!r}"
            )
    outputs = combined.get("outputs")
    inventory = combined.get("source_csv_inventory")
    if not isinstance(outputs, Mapping) or not isinstance(inventory, Mapping):
        raise FrozenV4AdoptionError("v4 combined manifest lacks output/source inventory")
    for name in NUMERIC_SOURCE_FILES:
        for label, mapping in (("outputs", outputs), ("source_csv_inventory", inventory)):
            metadata = mapping.get(name)
            if not isinstance(metadata, Mapping):
                raise FrozenV4AdoptionError(f"{label} lacks {name}")
            if metadata.get("sha256") != policy.source_hashes[name]:
                raise FrozenV4AdoptionError(f"{label} SHA mismatch for {name}")
    if combined.get("source_manifest_status") != "advisory_not_used_as_numeric_source":
        raise FrozenV4AdoptionError("legacy advisory-hash exception is not explicit")

    figure3 = _read_csv_bytes(
        payloads[NUMERIC_SOURCE_FILES[0]],
        name=NUMERIC_SOURCE_FILES[0],
        expected_fields=("Dataset", "Method", "K", "Theta", "Coverage", "Cost"),
    )
    figure4 = _read_csv_bytes(
        payloads[NUMERIC_SOURCE_FILES[1]],
        name=NUMERIC_SOURCE_FILES[1],
        expected_fields=("Dataset", "Method", "K", "Threshold", "Coverage"),
    )
    table2 = _read_csv_bytes(
        payloads[NUMERIC_SOURCE_FILES[2]],
        name=NUMERIC_SOURCE_FILES[2],
        expected_fields=(
            "Method",
            "AIDS Coverage",
            "AIDS Cost",
            "NCI1 Coverage",
            "NCI1 Cost",
            "Mutagenicity Coverage",
            "Mutagenicity Cost",
            "Proteins Coverage",
            "Proteins Cost",
        ),
    )
    expected_pairs = {
        (dataset, method)
        for dataset in ("AIDS", "Mutagenicity")
        for method in SOURCE_METHODS
    }
    observed3 = {(row["Dataset"], row["Method"]) for row in figure3}
    observed4 = {(row["Dataset"], row["Method"]) for row in figure4}
    if observed3 != expected_pairs or observed4 != expected_pairs:
        raise FrozenV4AdoptionError("v4 dataset/method source matrix mismatch")
    if len(figure3) != 160 or len(figure4) != 4808 or len(table2) != 4:
        raise FrozenV4AdoptionError("v4 source row-count contract mismatch")
    table_by_method = {row["Method"]: row for row in table2}
    if set(table_by_method) != set(SOURCE_METHODS):
        raise FrozenV4AdoptionError("v4 Table 2 method set mismatch")
    threshold_reference: dict[str, tuple[str, ...]] = {}
    for dataset, method in sorted(expected_pairs):
        rows3 = [row for row in figure3 if row["Dataset"] == dataset and row["Method"] == method]
        rows4 = [row for row in figure4 if row["Dataset"] == dataset and row["Method"] == method]
        if [row["K"] for row in rows3] != [str(k) for k in range(1, 21)]:
            raise FrozenV4AdoptionError(f"v4 Figure 3 K grid mismatch for {dataset}/{method}")
        if len(rows4) != 601 or {row["K"] for row in rows4} != {"10"}:
            raise FrozenV4AdoptionError(f"v4 Figure 4 K/row grid mismatch for {dataset}/{method}")
        if {row["Theta"] for row in rows3} != {"0.05"}:
            raise FrozenV4AdoptionError(f"v4 Figure 3 theta mismatch for {dataset}/{method}")
        thresholds = tuple(row["Threshold"] for row in rows4)
        threshold_values = [_finite(value, field="Threshold") for value in thresholds]
        if any(right <= left for left, right in zip(threshold_values, threshold_values[1:])):
            raise FrozenV4AdoptionError(f"v4 threshold grid is not increasing for {dataset}/{method}")
        if dataset not in threshold_reference:
            threshold_reference[dataset] = thresholds
        elif thresholds != threshold_reference[dataset]:
            raise FrozenV4AdoptionError(
                f"v4 threshold grid differs across methods within {dataset}"
            )
        coverage3 = [_finite(row["Coverage"], field="Coverage", rate=True) for row in rows3]
        coverage4 = [_finite(row["Coverage"], field="Coverage", rate=True) for row in rows4]
        if any(right < left for left, right in zip(coverage3, coverage3[1:])):
            raise FrozenV4AdoptionError(f"v4 Figure 3 coverage decreases for {dataset}/{method}")
        if any(right < left for left, right in zip(coverage4, coverage4[1:])):
            raise FrozenV4AdoptionError(f"v4 Figure 4 coverage decreases for {dataset}/{method}")
        for row in rows3:
            _finite(row["Cost"], field="Cost")
        table_row = table_by_method[method]
        k10 = next(row for row in rows3 if row["K"] == "10")
        if (
            k10["Coverage"] != table_row[f"{dataset} Coverage"]
            or k10["Cost"] != table_row[f"{dataset} Cost"]
        ):
            raise FrozenV4AdoptionError(
                f"v4 Table 2 is not an exact string match to Figure 3 K=10 for {dataset}/{method}"
            )
    return figure3, figure4, table2


def _load_source_once(
    root: Path,
    policy: ApprovalPolicy,
    *,
    require_original_basename: bool,
    require_proc_writer_audit: bool,
) -> SourceBundle:
    root = root.expanduser()
    if root.is_symlink():
        raise FrozenV4AdoptionError(f"source root may not be a symlink: {root}")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise FrozenV4AdoptionError(f"source root is not a directory: {root}")
    if require_original_basename and root.name != policy.source_root_basename:
        raise FrozenV4AdoptionError(
            f"source basename mismatch: observed={root.name}, expected={policy.source_root_basename}"
        )
    paths = [root / name for name in SOURCE_FILES]
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise FrozenV4AdoptionError(f"source file missing, empty, or symlinked: {path}")
    if require_proc_writer_audit:
        if platform.system() != "Linux" or not Path("/proc").is_dir():
            raise FrozenV4AdoptionError("Linux procfs is required for source writer audit")
        writers = _writable_source_fds(paths)
        if writers:
            raise FrozenV4AdoptionError(f"source bundle has writable file descriptors: {writers}")
    before = {path.name: _stat_identity(path) for path in paths}
    payloads: dict[str, bytes] = {}
    inventory: dict[str, dict[str, Any]] = {}
    for path in paths:
        # Exactly one content read per adopted source file.  The bytes are reused
        # for hashing, parsing, validation, copying, and every manifest.
        data = path.read_bytes()
        digest = _sha(data)
        expected = policy.source_hashes[path.name]
        if digest != expected:
            raise FrozenV4AdoptionError(
                f"source SHA mismatch for {path.name}: observed={digest}, expected={expected}"
            )
        payloads[path.name] = data
        inventory[path.name] = {
            "bytes": len(data),
            "sha256": digest,
            "content_read_count": 1,
            "hash_scan_count": 1,
        }
    after = {path.name: _stat_identity(path) for path in paths}
    if before != after:
        raise FrozenV4AdoptionError("source files changed during the one-pass adoption scan")
    figure3, figure4, table2 = _validate_source_semantics(payloads, policy)
    return SourceBundle(
        root=root,
        payloads=payloads,
        inventory=inventory,
        figure3_rows=figure3,
        figure4_rows=figure4,
        table2_rows=table2,
    )


def _cell_numeric_payloads(
    bundle: SourceBundle, *, dataset: str, method: str
) -> tuple[dict[str, bytes], dict[str, Any]]:
    rows3_source = [
        row
        for row in bundle.figure3_rows
        if row["Dataset"] == dataset and row["Method"] == method
    ]
    rows4_source = [
        row
        for row in bundle.figure4_rows
        if row["Dataset"] == dataset and row["Method"] == method
    ]
    table_source = next(row for row in bundle.table2_rows if row["Method"] == method)
    rows3 = [
        {
            "dataset": row["Dataset"],
            "method": row["Method"],
            "k": row["K"],
            "theta": row["Theta"],
            "coverage": row["Coverage"],
            "cost": row["Cost"],
        }
        for row in rows3_source
    ]
    rows4 = [
        {
            "dataset": row["Dataset"],
            "method": row["Method"],
            "k": row["K"],
            "threshold": row["Threshold"],
            "coverage": row["Coverage"],
        }
        for row in rows4_source
    ]
    table = [
        {
            "dataset": dataset,
            "method": method,
            "k": "10",
            "coverage": table_source[f"{dataset} Coverage"],
            "cost": table_source[f"{dataset} Cost"],
            "flip_rate": "N/A",
            "cf_drop": "N/A",
            "not_available_reason": "not_embedded_in_user_approved_frozen_v4",
        }
    ]
    threshold_strings = [row["Threshold"] for row in rows4_source]
    threshold_hash = _sha(("\n".join(threshold_strings) + "\n").encode("utf-8"))
    table_name = f"table2_{METHOD_SLUGS[method]}_k10.csv"
    figure3_bytes = _csv_bytes(
        ("dataset", "method", "k", "theta", "coverage", "cost"), rows3
    )
    payloads = {
        "figure3_coverage_vs_k.csv": figure3_bytes,
        "figure4_coverage_vs_threshold.csv": _csv_bytes(
            ("dataset", "method", "k", "threshold", "coverage"), rows4
        ),
        table_name: _csv_bytes(
            (
                "dataset",
                "method",
                "k",
                "coverage",
                "cost",
                "flip_rate",
                "cf_drop",
                "not_available_reason",
            ),
            table,
        ),
        "prefix_metrics.csv": figure3_bytes,
        "parent_best_distances.csv": _csv_bytes(
            ("dataset", "method", "parent_id", "best_distance", "status", "reason"),
            [
                {
                    "dataset": dataset,
                    "method": method,
                    "parent_id": "N/A",
                    "best_distance": "N/A",
                    "status": "N/A",
                    "reason": "parent-level rows not embedded in user-approved frozen v4",
                }
            ],
        ),
        "destination_distribution.csv": _csv_bytes(
            ("dataset", "method", "destination_label", "count", "rate", "status", "reason"),
            [
                {
                    "dataset": dataset,
                    "method": method,
                    "destination_label": "N/A",
                    "count": "N/A",
                    "rate": "N/A",
                    "status": "N/A",
                    "reason": "destination distribution not embedded in user-approved frozen v4",
                }
            ],
        ),
    }
    proof = {
        "dataset": dataset,
        "method": method,
        "figure3_source_rows": len(rows3_source),
        "figure4_source_rows": len(rows4_source),
        "table2_source_rows": 1,
        "threshold_raw_string_sha256": threshold_hash,
        "figure3_projected_sha256": _sha(payloads["figure3_coverage_vs_k.csv"]),
        "figure4_projected_sha256": _sha(payloads["figure4_coverage_vs_threshold.csv"]),
        "table2_projected_sha256": _sha(payloads[table_name]),
        "numeric_values_recomputed": False,
        "numeric_strings_changed": False,
    }
    return payloads, proof


def _write_payload(
    root: Path,
    name: str,
    data: bytes,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if metadata is not None:
        metadata[name] = {"bytes": len(data), "sha256": _sha(data)}


def _fsync_directories(root: Path) -> None:
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ) + [root]:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_cell(
    root: Path,
    *,
    final_root: Path,
    source_bundle_root: Path,
    bundle: SourceBundle,
    policy: ApprovalPolicy,
    dataset: str,
    method: str,
    created_at: str,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    files: dict[str, Any] = {}
    numeric, proof = _cell_numeric_payloads(bundle, dataset=dataset, method=method)
    for name, data in numeric.items():
        _write_payload(root, name, data, files)
    common = {
        "schema_version": "user_approved_frozen_v4_cell_v1",
        "approval_id": APPROVAL_ID,
        "approval_policy_sha256": policy.sha256,
        "dataset": dataset,
        "method": method,
        "oracle_backend": "rf",
        "classifier_family": "random_forest",
        "rf_oracle_used": True,
        "oracle_identity_status": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "dataset_identity_status": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "test_split_identity_status": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "molclr_identity_status": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "distance_line": DISTANCE_LINE,
        "cf_mode": CF_MODE,
        "k_max": 20,
        "table2_k": 10,
        "threshold_config_hash": proof["threshold_raw_string_sha256"],
        "threshold_identity_basis": "exact ordered raw strings from frozen Figure 4 CSV",
        "raw_output_root": str(source_bundle_root),
        "raw_output_complete": False,
        "frozen_numeric_bundle_complete": True,
        "scientific_recomputation_performed": False,
        "numeric_values_recomputed": False,
        "numeric_strings_changed": False,
        "selection_performed_in_adoption": False,
        "rendering_performed_in_adoption": False,
        "created_at_utc": created_at,
    }
    prefix_json = {
        **common,
        "schema_version": "user_approved_frozen_v4_prefix_reference_v1",
        "source_projection": proof,
        "status": "SOURCE_ROWS_COPIED_WITHOUT_NUMERIC_CHANGE",
    }
    _write_payload(root, "prefix_metrics.json", _json_bytes(prefix_json), files)
    summary = {
        **common,
        "schema_version": "user_approved_frozen_v4_summary_v1",
        "status": "ADOPTABLE_PASS",
        "coverage_k10": next(
            row[f"{dataset} Coverage"]
            for row in bundle.table2_rows
            if row["Method"] == method
        ),
        "cost_k10": next(
            row[f"{dataset} Cost"]
            for row in bundle.table2_rows
            if row["Method"] == method
        ),
        "unavailable_fields": {
            "parent_best_distances": "not embedded in source bundle",
            "destination_distribution": "not embedded in source bundle",
            "flip_rate": "not embedded in source bundle",
            "cf_drop": "not embedded in source bundle",
        },
    }
    _write_payload(root, "summary.json", _json_bytes(summary), files)
    oracle = {
        **common,
        "schema_version": "user_approved_frozen_v4_oracle_manifest_v1",
        "oracle_checkpoint": "",
        "oracle_hash": "",
        "exception_reason": "legacy v4 records RF family but not checkpoint identity",
    }
    _write_payload(root, "oracle_manifest.json", _json_bytes(oracle), files)
    evaluation = {
        **common,
        "schema_version": "user_approved_frozen_v4_evaluation_manifest_v1",
        "figure3_k": list(range(1, 21)),
        "figure4_k": 10,
        "figure4_points": 601,
        "theta_star": "0.05",
        "test_used_for_selection_evidence": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "threshold_fitted_on_test_evidence": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "selector_fitted_on_calibration_evidence": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
        "test_after_freeze_evidence": "NOT_EMBEDDED_IN_FROZEN_V4_SOURCE",
    }
    _write_payload(root, "evaluation_manifest.json", _json_bytes(evaluation), files)
    run_manifest = {
        **common,
        "schema_version": "user_approved_frozen_v4_run_manifest_v1",
        "stage": "USER_APPROVED_FROZEN_V4_ADOPTION",
        "status": "ADOPTABLE_PASS",
        "standardized_output_root": str(final_root),
        "source_bundle_root": str(source_bundle_root),
        "source_files": bundle.inventory,
        "source_projection": proof,
        "scientific_files": dict(files),
    }
    _write_payload(root, "run_manifest.json", _json_bytes(run_manifest), files)
    artifact_manifest = {
        **common,
        "schema_version": "user_approved_frozen_v4_artifact_manifest_v1",
        "files": dict(files),
    }
    _write_payload(root, "artifact_manifest.json", _json_bytes(artifact_manifest), files)
    final_audit = {
        **common,
        "schema_version": "user_approved_frozen_v4_final_artifact_audit_v1",
        "passed": True,
        "audit_passed": True,
        "passed_under_registry_exception": True,
        "generic_raw_provenance_audit_passed": False,
        "numeric_copy_audit_passed": True,
        "approval_exception": APPROVAL_ID,
        "source_projection": proof,
        "audited_files": dict(files),
        "clear_excluded": True,
        "comrecgc_not_substituted": True,
    }
    _write_payload(root, "final_artifact_audit.json", _json_bytes(final_audit), files)
    freeze_manifest = {
        **common,
        "schema_version": "user_approved_frozen_v4_freeze_manifest_v1",
        "frozen": True,
        "files": dict(files),
    }
    _write_payload(root, "freeze_manifest.json", _json_bytes(freeze_manifest), files)
    exception = {
        **common,
        "schema_version": EXCEPTION_SCHEMA,
        "exception_kind": APPROVAL_ID,
        "registry_status": "ADOPTABLE_PASS",
        "approval_scope": list(policy.approval_scope),
        "user_approved_waivers": list(policy.waivers),
        "waivable_registry_reasons": sorted(WAIVABLE_REGISTRY_REASONS),
        "source_root_basename": policy.source_root_basename,
        "adopted_source_bundle_root": str(source_bundle_root),
        "adopted_source_files": bundle.inventory,
        "source_projection": proof,
        "standardized_output_root": str(final_root),
        "output_files": dict(files),
        "clear_excluded": True,
        "comrecgc_not_substituted": True,
        "values_changed": False,
        "scientific_recomputation_performed": False,
    }
    _write_payload(root, "registry_exception.json", _json_bytes(exception), files)
    finalized = {
        "schema_version": "user_approved_frozen_v4_finalized_v1",
        "approval_id": APPROVAL_ID,
        "dataset": dataset,
        "method": method,
        "finalized": True,
        "gate_passed": True,
        "registry_exception_sha256": files["registry_exception.json"]["sha256"],
        "freeze_manifest_sha256": files["freeze_manifest.json"]["sha256"],
        "final_artifact_audit_sha256": files["final_artifact_audit.json"]["sha256"],
    }
    _write_payload(root, "_FINALIZED.json", _json_bytes(finalized), files)
    _write_payload(root, "PASS", b"ADOPTABLE_PASS\n", files)
    return {
        "dataset": dataset,
        "method": method,
        "standardized_output_root": str(final_root),
        "registry_status": "ADOPTABLE_PASS",
        "registry_exception_sha256": files["registry_exception.json"]["sha256"],
        "source_projection": proof,
    }


def adopt_user_approved_frozen_v4(
    *,
    source_root: str | Path,
    runtime_root: str | Path,
    output_root: str | Path,
    policy: ApprovalPolicy | None = None,
    require_proc_writer_audit: bool = True,
) -> AdoptionResult:
    policy = policy or load_approval_policy()
    runtime = Path(runtime_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser()
    if destination.exists() or destination.is_symlink():
        raise FrozenV4AdoptionError(f"output root must be fresh and absent: {destination}")
    destination = destination.resolve()
    if not _is_relative_to(destination, runtime):
        raise FrozenV4AdoptionError("output root must be below the persistent runtime root")
    if "paper" in destination.parts:
        raise FrozenV4AdoptionError("the user-approved adoption may not write under paper/")
    source = Path(source_root).expanduser().resolve(strict=True)
    if _is_relative_to(destination, source) or _is_relative_to(source, destination):
        raise FrozenV4AdoptionError("source and fresh output roots must be disjoint")
    bundle = _load_source_once(
        source,
        policy,
        require_original_basename=True,
        require_proc_writer_audit=require_proc_writer_audit,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
    )
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        copied_source = temporary / "source_bundle"
        copied_source.mkdir()
        for name in SOURCE_FILES:
            _write_payload(copied_source, name, bundle.payloads[name])
        copied_inventory = {
            name: {
                **metadata,
                "copied_byte_for_byte": True,
                "source_path": str(bundle.root / name),
                "adopted_path": str(destination / "source_bundle" / name),
            }
            for name, metadata in bundle.inventory.items()
        }
        _write_payload(
            temporary,
            "adopted_source_inventory.json",
            _json_bytes(
                {
                    "schema_version": "user_approved_frozen_v4_source_inventory_v1",
                    "approval_id": APPROVAL_ID,
                    "approval_policy_sha256": policy.sha256,
                    "source_root": str(bundle.root),
                    "source_root_basename": bundle.root.name,
                    "files": copied_inventory,
                    "all_content_read_once": True,
                    "images_or_pdfs_read": False,
                }
            ),
        )
        cell_records: list[dict[str, Any]] = []
        explicit_cells: dict[str, str] = {}
        for dataset, method in APPROVED_CELLS:
            relative = Path("cells") / DATASET_SLUGS[dataset] / METHOD_SLUGS[method] / "standardized"
            temporary_cell = temporary / relative
            final_cell = destination / relative
            record = _write_cell(
                temporary_cell,
                final_root=final_cell,
                source_bundle_root=destination / "source_bundle",
                bundle=bundle,
                policy=policy,
                dataset=dataset,
                method=method,
                created_at=created_at,
            )
            cell_records.append(record)
            explicit_cells[f"{dataset}/{method}"] = str(final_cell)
        aggregate = {
            "schema_version": "user_approved_frozen_v4_adoption_manifest_v1",
            "approval_id": APPROVAL_ID,
            "approval_policy_path": str(policy.path),
            "approval_policy_sha256": policy.sha256,
            "created_at_utc": created_at,
            "source_root": str(bundle.root),
            "source_root_basename": bundle.root.name,
            "adopted_source_bundle_root": str(destination / "source_bundle"),
            "source_files": copied_inventory,
            "cells": cell_records,
            "cell_count": len(cell_records),
            "registry_status": "ADOPTABLE_PASS",
            "clear_excluded": True,
            "comrecgc_not_substituted": True,
            "scientific_recomputation_performed": False,
            "numeric_values_recomputed": False,
            "numeric_strings_changed": False,
            "images_or_pdfs_read": False,
        }
        _write_payload(temporary, "adoption_manifest.json", _json_bytes(aggregate))
        _write_payload(
            temporary,
            "explicit_cells.json",
            _json_bytes(
                {
                    "schema_version": "four_by_four_explicit_cells_v1",
                    "approval_id": APPROVAL_ID,
                    "cells": explicit_cells,
                }
            ),
        )
        _write_payload(
            temporary,
            "registry_exception_schema.json",
            _json_bytes(
                {
                    "schema_version": EXCEPTION_SCHEMA,
                    "exception_kind": APPROVAL_ID,
                    "allowed_cells": list(policy.approval_scope),
                    "allowed_status": "ADOPTABLE_PASS",
                    "generic_requirements_waived": list(policy.waivers),
                    "requirements_never_waived": [
                        "exact pinned source hashes",
                        "exact projected numeric strings",
                        "RF backend",
                        DISTANCE_LINE,
                        CF_MODE,
                        "Figure 3 K=1..20",
                        "Figure 4 exact 601-point grid shared within each dataset",
                        "Table 2 exact Figure 3 K=10 equality",
                        "CLEAR exclusion",
                        "ComRecGC non-substitution",
                    ],
                }
            ),
        )
        _write_payload(
            temporary,
            "supersession_manifest.json",
            _json_bytes(
                {
                    "schema_version": "user_approved_frozen_v4_supersession_v1",
                    "approval_id": APPROVAL_ID,
                    "cells": list(policy.approval_scope),
                    "future_controller_action": "OMIT_NOT_STARTED_REPAIR_FOR_APPROVED_CELL",
                    "running_tasks_may_be_stopped": False,
                    "running_tasks_affected_by_this_adoption": False,
                    "existing_controller_state_mutated": False,
                    "reason": "user authorized exact frozen v4 cell values; duplicate generation/evaluation is unnecessary",
                }
            ),
        )
        _write_payload(
            temporary,
            "_RUN_COMPLETE.json",
            _json_bytes(
                {
                    "schema_version": "user_approved_frozen_v4_run_complete_v1",
                    "approval_id": APPROVAL_ID,
                    "status": "PASS",
                    "run_complete": True,
                    "cell_count": 6,
                    "adoption_manifest_sha256": _sha(_json_bytes(aggregate)),
                }
            ),
        )
        _write_payload(temporary, "PASS", b"PASS\n")
        _fsync_directories(temporary)
        os.replace(temporary, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return AdoptionResult(
        output_root=destination,
        cell_roots=explicit_cells,
        source_inventory=copied_inventory,
        approval_policy_sha256=policy.sha256,
    )


def _source_cache_key(root: Path, policy: ApprovalPolicy) -> tuple[Any, ...]:
    identities = tuple((name, *_stat_identity(root / name)) for name in SOURCE_FILES)
    return (str(root), policy.sha256, identities)


def validate_adopted_cell(
    standardized_root: str | Path,
    *,
    policy: ApprovalPolicy | None = None,
) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
    """Validate one exception cell without trusting its self-declared values."""

    policy = policy or load_approval_policy()
    root = Path(standardized_root).expanduser()
    reasons: list[str] = []
    details: dict[str, Any] = {}
    try:
        if root.is_symlink():
            raise FrozenV4AdoptionError("standardized root may not be a symlink")
        root = root.resolve(strict=True)
        exception = _read_json_bytes(
            (root / "registry_exception.json").read_bytes(),
            name=str(root / "registry_exception.json"),
        )
        dataset = str(exception.get("dataset") or "")
        method = str(exception.get("method") or "")
        cell = f"{dataset}/{method}"
        details.update({"dataset": dataset, "method": method, "cell": cell})
        required = {
            "schema_version": EXCEPTION_SCHEMA,
            "exception_kind": APPROVAL_ID,
            "approval_id": APPROVAL_ID,
            "approval_policy_sha256": policy.sha256,
            "registry_status": "ADOPTABLE_PASS",
            "source_root_basename": policy.source_root_basename,
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "rf_oracle_used": True,
            "distance_line": DISTANCE_LINE,
            "cf_mode": CF_MODE,
            "clear_excluded": True,
            "comrecgc_not_substituted": True,
            "values_changed": False,
            "scientific_recomputation_performed": False,
        }
        for key, expected in required.items():
            if exception.get(key) != expected:
                reasons.append(f"EXCEPTION_FIELD_MISMATCH:{key}")
        if cell not in policy.approval_scope:
            reasons.append("CELL_OUTSIDE_USER_APPROVED_SCOPE")
        if tuple(exception.get("approval_scope") or ()) != policy.approval_scope:
            reasons.append("APPROVAL_SCOPE_MISMATCH")
        if set(exception.get("waivable_registry_reasons") or ()) != set(
            WAIVABLE_REGISTRY_REASONS
        ):
            reasons.append("WAIVABLE_REASON_SET_MISMATCH")
        declared_sources = exception.get("adopted_source_files")
        if not isinstance(declared_sources, Mapping):
            reasons.append("ADOPTED_SOURCE_INVENTORY_MISSING")
        else:
            for name, expected_sha in policy.source_hashes.items():
                metadata = declared_sources.get(name)
                if not isinstance(metadata, Mapping) or metadata.get("sha256") != expected_sha:
                    reasons.append(f"ADOPTED_SOURCE_DECLARATION_MISMATCH:{name}")
        bundle_root_text = str(exception.get("adopted_source_bundle_root") or "")
        bundle_root = Path(bundle_root_text).expanduser()
        if not bundle_root_text or bundle_root.is_symlink():
            reasons.append("ADOPTED_SOURCE_BUNDLE_ROOT_INVALID")
        else:
            bundle_root = bundle_root.resolve(strict=True)
            key = _source_cache_key(bundle_root, policy)
            bundle = _SOURCE_CACHE.get(key)
            if bundle is None:
                bundle = _load_source_once(
                    bundle_root,
                    policy,
                    require_original_basename=False,
                    require_proc_writer_audit=False,
                )
                if len(_SOURCE_CACHE) >= 8:
                    _SOURCE_CACHE.clear()
                _SOURCE_CACHE[key] = bundle
            expected_numeric, proof = _cell_numeric_payloads(
                bundle, dataset=dataset, method=method
            )
            for name, expected_bytes in expected_numeric.items():
                target = root / name
                if not target.is_file() or target.is_symlink():
                    reasons.append(f"STANDARDIZED_FILE_MISSING_OR_SYMLINK:{name}")
                elif target.read_bytes() != expected_bytes:
                    reasons.append(f"STANDARDIZED_SOURCE_PROJECTION_MISMATCH:{name}")
            if exception.get("source_projection") != proof:
                reasons.append("SOURCE_PROJECTION_PROOF_MISMATCH")
        freeze = _read_json_bytes(
            (root / "freeze_manifest.json").read_bytes(),
            name=str(root / "freeze_manifest.json"),
        )
        declared_outputs = freeze.get("files")
        if not isinstance(declared_outputs, Mapping) or not declared_outputs:
            reasons.append("OUTPUT_FILE_CLOSURE_MISSING")
        else:
            for name, metadata in declared_outputs.items():
                target = root / str(name)
                if not isinstance(metadata, Mapping) or not target.is_file() or target.is_symlink():
                    reasons.append(f"OUTPUT_FILE_CLOSURE_INVALID:{name}")
                    continue
                data = target.read_bytes()
                if metadata.get("bytes") != len(data) or metadata.get("sha256") != _sha(data):
                    reasons.append(f"OUTPUT_FILE_CLOSURE_MISMATCH:{name}")
        final_audit = _read_json_bytes(
            (root / "final_artifact_audit.json").read_bytes(),
            name=str(root / "final_artifact_audit.json"),
        )
        if (
            final_audit.get("passed") is not True
            or final_audit.get("passed_under_registry_exception") is not True
            or final_audit.get("numeric_copy_audit_passed") is not True
        ):
            reasons.append("FINAL_EXCEPTION_AUDIT_NOT_PASS")
        if (root / "PASS").read_text(encoding="utf-8").strip() != "ADOPTABLE_PASS":
            reasons.append("ADOPTABLE_PASS_MARKER_INVALID")
    except (OSError, KeyError, ValueError, FrozenV4AdoptionError) as exc:
        reasons.append(f"EXCEPTION_VALIDATION_ERROR:{type(exc).__name__}:{exc}")
    unique = tuple(sorted(set(reasons)))
    return not unique, unique, details


__all__ = [
    "APPROVAL_ID",
    "APPROVED_CELLS",
    "ApprovalPolicy",
    "AdoptionResult",
    "DEFAULT_POLICY_PATH",
    "EXCEPTION_SCHEMA",
    "FrozenV4AdoptionError",
    "SOURCE_FILES",
    "WAIVABLE_REGISTRY_REASONS",
    "adopt_user_approved_frozen_v4",
    "load_approval_policy",
    "validate_adopted_cell",
]
