"""Fail-closed adoption and inventory for legacy AIDS/Mutagenicity cells.

The module is deliberately post-processing only.  It never generates a
candidate, loads an oracle model, fits a threshold, or changes a frozen order.
The one-time pre-controller Ours adopter opens the immutable test CSV only to
reconstruct the already-frozen matrix contract; the later controller verifier
is genuinely manifest-only.  Other legacy roots are inventoried as evidence
and stay explicitly incomplete, missing, or code-blocked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from src.eval import mutagenicity_wnode_frozen_test as frozen_test
from src.eval.fullgraph_wnode_artifacts import (
    load_ranked_candidates,
    stable_json_sha256,
    validate_frozen_candidate_contract,
)
from src.eval.molclr_node_embeddings import canonicalize_smiles


MUTAGENICITY_RF_SHA256 = (
    "af213aa766626decaf99876b43ede725412a355adf37f1aa0d56233d8653e204"
)
MOLCLR_SHA256 = (
    "93bc4f02ea8847cd44fa21ec3f65600ff2f4a7ae6d3a85e8519a5bcc56afc20a"
)
MUT_GCF_SELECTED_CSV_SHA256 = (
    "e968fa140a4e34ad6abc6430b17f539e5d568b319d4856d74763496e9181f341"
)
MUT_GCF_CANDIDATE_ORDER_SHA256 = (
    "e8758517a1c81fa150298497a8a799806abbe5a8c17ba048790636aacf4a1a46"
)
MUT_GCF_NATIVE_RANKS = (
    112,
    120,
    177,
    179,
    195,
    217,
    388,
    442,
    605,
    701,
    794,
    810,
    1034,
    1095,
    1417,
    1786,
    1788,
    1975,
    3815,
    4198,
)
MUT_GCF_SELECTION_METHOD = "native_gcf_summary_rank_filtered_by_validity"
ALLOWED_STATUSES = {
    "FROZEN_PASS",
    "ADOPTABLE_PASS",
    "RUNNING",
    "READY",
    "MISSING",
    "STALE_ORACLE",
    "STALE_DATASET",
    "STALE_SPLIT",
    "STALE_METRIC",
    "INCOMPLETE",
    "BLOCKED_LICENSE",
    "BLOCKED_CODE",
    "FAILED",
}
STANDARD_FILES = (
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_ours_k10.csv",
    "table2_ours_k20.csv",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
)
MUT_OURS_REQUIRED_STANDARDIZED_FILES = (
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_ours_k10.csv",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
    "destination_distribution.csv",
    "oracle_manifest.json",
    "evaluation_manifest.json",
    "summary.json",
    "run_manifest.json",
)
MUT_OURS_REQUIRED_SOURCE_FILES = (
    "pair_matrix.jsonl",
    "match_instances.jsonl",
    "selected_sequence.jsonl",
    "thresholds.json",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_ours_k10.csv",
    "summary.json",
    "run_manifest.json",
    "_RUN_COMPLETE.json",
    "_FINALIZED.json",
    "manual_final_test_audit_v2.json",
    "threshold_freeze_semantic_audit_v2.json",
)
MUT_OURS_SCIENTIFIC_FILES = (
    "pair_matrix.jsonl",
    "match_instances.jsonl",
    "selected_sequence.jsonl",
    "thresholds.json",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_ours_k10.csv",
    "table2_ours_k20.csv",
    "summary.json",
    "run_manifest.json",
    "_RUN_COMPLETE.json",
)


class LegacyStandardizationError(ValueError):
    """Legacy evidence cannot satisfy the frozen-cell contract."""


@dataclass(slots=True)
class HashCache:
    """Hash each physical file at most once during one adoption invocation."""

    values: dict[Path, str]
    calls: dict[Path, int]

    def __init__(self) -> None:
        self.values = {}
        self.calls = {}

    def sha256(self, path_like: str | Path) -> str:
        path = Path(path_like).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in self.values:
            return self.values[path]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self.values[path] = value
        self.calls[path] = self.calls.get(path, 0) + 1
        return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_object(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like).expanduser().resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message normalization
        raise LegacyStandardizationError(f"Invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise LegacyStandardizationError(f"Expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    if not fields:
        raise LegacyStandardizationError(f"Refusing to write empty CSV: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: (
                            json.dumps(value, ensure_ascii=False, sort_keys=True)
                            if isinstance(value, (dict, list, tuple))
                            else ("" if value is None else value)
                        )
                        for field, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise LegacyStandardizationError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _deep_values(payload: Any, names: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if str(key) in names and value not in (None, ""):
                values.append(value)
            values.extend(_deep_values(value, names))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_deep_values(value, names))
    return values


def _first_deep(payload: Mapping[str, Any], names: set[str]) -> Any:
    values = _deep_values(payload, names)
    return values[0] if values else None


def _identity_path(value: Any) -> str:
    if isinstance(value, Mapping):
        for name in ("path", "file", "root"):
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate:
                return candidate
    if isinstance(value, str) and value:
        return value
    raise LegacyStandardizationError(f"Identity does not contain a path: {value!r}")


def _identity_sha(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for name in ("sha256", "file_sha256", "hash"):
            candidate = value.get(name)
            if isinstance(candidate, str) and len(candidate.strip()) == 64:
                return candidate.strip().lower()
    return None


def _identity_size(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    for name in ("size", "size_bytes", "bytes"):
        candidate = value.get(name)
        if candidate is None:
            continue
        if isinstance(candidate, bool):
            raise LegacyStandardizationError(f"Invalid identity size: {value!r}")
        try:
            size = int(candidate)
        except (TypeError, ValueError) as exc:
            raise LegacyStandardizationError(
                f"Invalid identity size: {value!r}"
            ) from exc
        if size < 0:
            raise LegacyStandardizationError(f"Invalid identity size: {value!r}")
        return size
    return None


def _normalize_hash(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise LegacyStandardizationError(f"{label} is not a valid SHA256: {value!r}")
    return digest


def _safe_relative(value: str) -> str:
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise LegacyStandardizationError(
            f"Manifest artifact path must be normalized and relative: {value!r}"
        )
    return relative.as_posix()


def _manifest_entries(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize common final-result checksum-manifest layouts."""

    normalized: dict[str, dict[str, Any]] = {}
    for container_name in ("file_inventory", "artifacts", "files", "file_sha256"):
        container = payload.get(container_name)
        rows: list[tuple[str, Any]] = []
        if isinstance(container, Mapping):
            rows = [(str(path), metadata) for path, metadata in container.items()]
        elif isinstance(container, list):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                path = item.get("path") or item.get("relative_path") or item.get("file")
                if isinstance(path, str):
                    rows.append((path, item))
        for raw_path, metadata in rows:
            relative = _safe_relative(raw_path)
            if isinstance(metadata, str):
                digest = metadata
                size = None
            elif isinstance(metadata, Mapping):
                digest = (
                    metadata.get("sha256")
                    or metadata.get("file_sha256")
                    or metadata.get("hash")
                )
                size = metadata.get("size")
                if size is None:
                    size = metadata.get("size_bytes")
                if size is None:
                    size = metadata.get("bytes")
            else:
                continue
            if digest in (None, ""):
                continue
            entry = {
                "sha256": _normalize_hash(
                    digest, label=f"{container_name}[{relative!r}].sha256"
                ),
                "size": size,
                "source_container": container_name,
            }
            previous = normalized.get(relative)
            if previous is not None and previous["sha256"] != entry["sha256"]:
                raise LegacyStandardizationError(
                    f"Conflicting checksum entries for {relative}"
                )
            normalized[relative] = entry
    if not normalized:
        raise LegacyStandardizationError(
            "final_result_manifest.json contains no supported checksum entries"
        )
    return normalized


def verify_manifest_closure(
    source_root: str | Path,
    manifest_path: str | Path,
    *,
    required_relative_files: Sequence[str],
    hash_cache: HashCache,
) -> dict[str, Any]:
    root = Path(source_root).expanduser().resolve(strict=True)
    manifest = _load_object(manifest_path)
    entries = _manifest_entries(manifest)
    missing_from_manifest = [
        relative for relative in required_relative_files if relative not in entries
    ]
    if missing_from_manifest:
        raise LegacyStandardizationError(
            "Required source artifacts lack final-result checksum closure: "
            f"{missing_from_manifest}"
        )
    checked: list[dict[str, Any]] = []
    for relative, entry in sorted(entries.items()):
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise LegacyStandardizationError(
                f"Manifest artifact escapes source root: {relative}"
            ) from exc
        if not path.is_file():
            raise LegacyStandardizationError(f"Manifest artifact is not a file: {path}")
        expected_size = entry.get("size")
        if expected_size is not None:
            if isinstance(expected_size, bool) or int(expected_size) != path.stat().st_size:
                raise LegacyStandardizationError(
                    f"Manifest size mismatch: {relative}"
                )
        actual = hash_cache.sha256(path)
        if actual != entry["sha256"]:
            raise LegacyStandardizationError(
                f"Manifest SHA256 mismatch: {relative}"
            )
        checked.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": actual,
            }
        )
    return {
        "closure_verified": True,
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": hash_cache.sha256(manifest_path),
        "artifact_count": len(checked),
        "artifacts": checked,
    }


def resolve_recorded_path(
    recorded: Any,
    *,
    remap_roots: Sequence[str | Path],
) -> Path:
    raw = _identity_path(recorded)
    original = Path(raw).expanduser()
    candidates: list[Path] = [original]
    markers = ("counterfactual-subgraph/", "/payload/project/")
    suffixes: list[Path] = []
    normalized = original.as_posix()
    for marker in markers:
        if marker in normalized:
            suffixes.append(Path(normalized.split(marker, 1)[1]))
    if not original.is_absolute():
        suffixes.append(original)
    for root_like in remap_roots:
        root = Path(root_like).expanduser()
        candidates.extend(root / suffix for suffix in suffixes)
    existing = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved not in existing:
            existing.append(resolved)
    if not existing:
        raise FileNotFoundError(
            f"Recorded path is unavailable after explicit remapping: {raw}"
        )
    # Remap roots are an explicit precedence list.  A Step0 payload and its
    # static-project snapshot can legitimately contain byte-identical copies
    # with different inodes; the downstream selector/test/oracle gates verify
    # the chosen bytes independently.  Selecting the first explicit match
    # avoids another large-checkpoint hash pass merely to resolve a path.
    return existing[0]


def scan_live_writers(
    source_root: str | Path,
    *,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Prove that no Linux process holds a writable FD under ``source_root``."""

    root = Path(source_root).expanduser().resolve(strict=True)
    proc = Path(proc_root)
    if not proc.is_dir():
        raise LegacyStandardizationError(
            "Writable-FD audit requires Linux procfs; adoption remains blocked"
        )
    writers: list[dict[str, Any]] = []
    scanned = 0
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit() or not pid_dir.is_dir():
            continue
        fd_dir = pid_dir / "fd"
        fdinfo_dir = pid_dir / "fdinfo"
        try:
            descriptors = list(fd_dir.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        scanned += 1
        for descriptor in descriptors:
            try:
                target = descriptor.resolve(strict=True)
                target.relative_to(root)
                lines = (fdinfo_dir / descriptor.name).read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except (FileNotFoundError, PermissionError, ValueError, OSError):
                continue
            flags_text = next(
                (line.split(":", 1)[1].strip() for line in lines if line.startswith("flags:")),
                None,
            )
            if flags_text is None:
                continue
            try:
                flags = int(flags_text, 8)
            except ValueError:
                continue
            if (flags & os.O_ACCMODE) in {os.O_WRONLY, os.O_RDWR}:
                writers.append(
                    {
                        "pid": int(pid_dir.name),
                        "fd": int(descriptor.name),
                        "path": str(target),
                        "flags_octal": flags_text,
                    }
                )
    if writers:
        raise LegacyStandardizationError(
            f"Legacy source has live writable file descriptors: {writers[:8]}"
        )
    return {
        "procfs_verified": True,
        "scanned_process_count": scanned,
        "writable_fd_count": 0,
        "writers": [],
    }


def _assert_bool(payload: Mapping[str, Any], field: str, expected: bool) -> None:
    if payload.get(field) is not expected:
        raise LegacyStandardizationError(
            f"{field}={payload.get(field)!r}; expected {expected!r}"
        )


def _assert_mut_ours_csvs(source: Path) -> dict[str, Any]:
    figure3 = _read_csv(source / "figure3_coverage_vs_k.csv")
    figure4 = _read_csv(source / "figure4_coverage_vs_threshold.csv")
    table2 = _read_csv(source / "table2_ours_k10.csv")
    prefix = _read_csv(source / "prefix_metrics.csv")
    k_values = [int(float(row.get("k") or 0)) for row in figure3]
    if k_values != list(range(1, 21)):
        raise LegacyStandardizationError("Mut Ours Figure 3 is not frozen K=1..20")
    coverages = []
    for row in figure3:
        value = row.get("coverage") or row.get("ccrcov_theta_star") or row.get("ccrcov")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise LegacyStandardizationError("Figure 3 coverage is non-numeric") from exc
        if not math.isfinite(number):
            raise LegacyStandardizationError("Figure 3 coverage is non-finite")
        coverages.append(number)
    if any(right + 1e-12 < left for left, right in zip(coverages, coverages[1:])):
        raise LegacyStandardizationError("Mut Ours Figure 3 coverage decreases with K")
    if not figure4:
        raise LegacyStandardizationError("Mut Ours Figure 4 is empty")
    if len(table2) != 1 or int(float(table2[0].get("k") or 0)) != 10:
        raise LegacyStandardizationError("Mut Ours Table 2 is not exactly K=10")
    if len(prefix) != 20:
        raise LegacyStandardizationError("Mut Ours prefix_metrics is not K=1..20")
    return {
        "figure3_rows": len(figure3),
        "figure4_rows": len(figure4),
        "table2_rows": len(table2),
        "prefix_rows": len(prefix),
        "coverage_monotonic": True,
    }


def _validate_finalized_audit_evidence(
    source: Path,
    *,
    expected_candidate_count: int,
    expected_pair_count: int,
) -> dict[str, Any]:
    finalized = _load_object(source / "_FINALIZED.json")
    manual = _load_object(source / "manual_final_test_audit_v2.json")
    threshold = _load_object(source / "threshold_freeze_semantic_audit_v2.json")
    if finalized.get("finalized") is not True:
        raise LegacyStandardizationError("_FINALIZED.json does not assert finalized=true")
    expected_manual = {
        "audit_marker_present": True,
        "can_finalize_ours_results": True,
        "experiment_success": True,
        "official_audit_passed": True,
    }
    for field, expected in expected_manual.items():
        if manual.get(field) is not expected:
            raise LegacyStandardizationError(
                f"manual_final_test_audit_v2.{field} is not true"
            )
    if manual.get("failed_hard_checks") != []:
        raise LegacyStandardizationError(
            "manual_final_test_audit_v2 has failed hard checks"
        )
    checks = manual.get("checks")
    if not isinstance(checks, Mapping):
        raise LegacyStandardizationError(
            "manual_final_test_audit_v2.checks is missing"
        )
    false_checks = [name for name, value in checks.items() if value is False]
    if false_checks != ["thresholds_exactly_frozen"]:
        raise LegacyStandardizationError(
            f"Unexpected false manual audit checks: {false_checks}"
        )
    if any(value is not True for name, value in checks.items() if name not in false_checks):
        raise LegacyStandardizationError(
            "manual_final_test_audit_v2 contains a non-passing check"
        )
    correction = manual.get("audit_correction")
    if not isinstance(correction, Mapping) or correction.get("replacement_check") != (
        "threshold_values_and_weights_exactly_frozen_with_provenance_metadata_allowed"
    ):
        raise LegacyStandardizationError(
            "Manual threshold audit correction is absent or unexpected"
        )
    if int(manual.get("candidate_count", -1)) != expected_candidate_count or int(
        manual.get("pair_rows", -1)
    ) != expected_pair_count:
        raise LegacyStandardizationError(
            "Manual final audit candidate/pair count contract differs"
        )
    expected_threshold = {
        "audit_passed": True,
        "scientific_core_equal": True,
    }
    for field, expected in expected_threshold.items():
        if threshold.get(field) is not expected:
            raise LegacyStandardizationError(
                f"threshold_freeze_semantic_audit_v2.{field} is not true"
            )
    for field, expected in (
        ("scientific_mismatches", {}),
        ("unexpected_difference_keys", []),
        ("missing_scientific_keys", []),
    ):
        if threshold.get(field) != expected:
            raise LegacyStandardizationError(
                f"threshold_freeze_semantic_audit_v2.{field} is not empty"
            )
    provenance = threshold.get("provenance_checks")
    if (
        not isinstance(provenance, Mapping)
        or len(provenance) != 4
        or any(value is not True for value in provenance.values())
    ):
        raise LegacyStandardizationError(
            "Threshold semantic provenance checks are not exactly four PASS values"
        )
    return {
        "finalized": True,
        "manual_final_test_audit_passed": True,
        "threshold_freeze_semantic_audit_passed": True,
        "allowed_legacy_false_check": "thresholds_exactly_frozen",
        "replacement_check": correction["replacement_check"],
    }


def _audit_self_contained_finalized_bundle(
    source: Path,
    *,
    test_csv: Path,
    expected_parent_count: int,
    expected_candidate_count: int,
    expected_pair_count: int,
) -> dict[str, Any]:
    """Recompute the frozen-test matrix contract without missing source dirs."""

    parents = frozen_test.load_test_parents(
        test_csv, expected_parent_count=expected_parent_count
    )
    selected = frozen_test._read_jsonl(source / "selected_sequence.jsonl")
    selected_ids: list[str] = []
    for rank, row in enumerate(selected, start=1):
        if int(row.get("rank") or 0) != rank:
            raise LegacyStandardizationError(
                "Finalized selected_sequence rank is incomplete or reordered"
            )
        candidate_id = str(row.get("candidate_id") or "").strip()
        fragment = str(row.get("canonical_fragment") or "").strip()
        if not candidate_id or canonicalize_smiles(fragment) is None:
            raise LegacyStandardizationError(
                "Finalized selected_sequence contains an invalid candidate"
            )
        selected_ids.append(candidate_id)
    if (
        len(selected_ids) != expected_candidate_count
        or len(set(selected_ids)) != expected_candidate_count
    ):
        raise LegacyStandardizationError(
            "Finalized selected_sequence is not 20 unique candidates"
        )
    threshold_payload = _load_object(source / "thresholds.json")
    thresholds = frozen_test._load_threshold_bundle(threshold_payload)
    pair_rows = frozen_test._read_jsonl(source / "pair_matrix.jsonl")
    match_rows = frozen_test._read_jsonl(source / "match_instances.jsonl")
    parent_ids = {parent.parent_id for parent in parents}
    expected_keys = {
        f"{parent_id}\0{candidate_id}"
        for parent_id in parent_ids
        for candidate_id in selected_ids
    }
    actual_keys = [
        f"{row.get('parent_id')}\0{row.get('candidate_id')}" for row in pair_rows
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise LegacyStandardizationError("Finalized pair matrix has duplicate keys")
    if len(pair_rows) != expected_pair_count or set(actual_keys) != expected_keys:
        raise LegacyStandardizationError(
            "Finalized pair matrix is not the complete parent x candidate product"
        )
    for row in pair_rows:
        strict = frozen_test._bool_value(row.get("pair_strict_flip"))
        distance = frozen_test._finite_float(row.get("wnode_distance"))
        if strict:
            if (
                not frozen_test._bool_value(row.get("applicable"))
                or int(row.get("pred_before")) != 1
                or int(row.get("pred_after")) != 0
                or distance is None
                or distance < 0.0
            ):
                raise LegacyStandardizationError(
                    "Finalized strict-flip row violates the 1 -> 0 WNode contract"
                )
        elif distance is not None:
            raise LegacyStandardizationError(
                "Finalized non-flip pair has a finite WNode distance"
            )
    frozen_test._audit_match_aggregation(pair_rows, match_rows)
    recomputed, _ = frozen_test.compute_frozen_prefix_metrics(
        pair_rows, parents, selected, thresholds
    )
    recorded = _read_csv(source / "prefix_metrics.csv")
    if len(recorded) != 20 or len(recomputed) != 20:
        raise LegacyStandardizationError("Finalized prefix metrics are not K=1..20")
    previous_coverage = -math.inf
    previous_cost = math.inf
    for rank, (actual, expected) in enumerate(zip(recorded, recomputed), start=1):
        if int(actual.get("k") or 0) != rank:
            raise LegacyStandardizationError("Finalized prefix K order changed")
        for field, value in expected.items():
            if not frozen_test._values_equal(actual.get(field), value):
                raise LegacyStandardizationError(
                    f"Finalized prefix metric mismatch at K={rank}, field={field}"
                )
        coverage = float(expected["ccrcov_theta_star"])
        cost = float(expected["fixed_capped_mean_cost"])
        if coverage + 1e-12 < previous_coverage or cost > previous_cost + 1e-12:
            raise LegacyStandardizationError(
                "Finalized prefix monotonicity contract failed"
            )
        previous_coverage, previous_cost = coverage, cost
    summary = _load_object(source / "summary.json")
    if summary.get("test_cohort_hash") != frozen_test.test_cohort_hash(parents):
        raise LegacyStandardizationError("Finalized test cohort hash mismatch")
    expected_summary = {
        "selected_variant": "A2_MultiThreshold",
        "test_parent_count": expected_parent_count,
        "candidate_count": expected_candidate_count,
        "actual_pair_rows": expected_pair_count,
        "complete_cartesian": True,
        "run_complete": True,
        "wnode_self_test_passed": True,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise LegacyStandardizationError(
                f"Finalized summary.{field}={summary.get(field)!r}; expected {expected!r}"
            )
    return {
        "audit_passed": True,
        "audit_mode": "self_contained_finalized_bundle",
        "selector_frozen": True,
        "frozen_selector_sha256_verified": False,
        "frozen_selector_source_available": False,
        "test_parent_count": len(parents),
        "candidate_count": len(selected_ids),
        "pair_count": len(pair_rows),
        "complete_cartesian": True,
        "candidate_order_recomputed": True,
        "threshold_semantics_recomputed": True,
        "prefix_metrics_recomputed": True,
        "match_aggregation_recomputed": True,
        "test_threshold_fitting": False,
        "test_candidate_selection": False,
        "test_variant_selection": False,
        "run_complete": True,
    }


def _normalize_csv_method(source: Path, destination: Path) -> None:
    rows = _read_csv(source)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        copied: dict[str, Any] = dict(row)
        copied["method"] = "Ours"
        normalized.append(copied)
    _atomic_csv(destination, normalized)


def _snapshot(paths: Iterable[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    }


def adopt_mutagenicity_ours(
    *,
    source_root: str | Path,
    output_root: str | Path,
    remap_roots: Sequence[str | Path],
    expected_teacher_sha256: str = MUTAGENICITY_RF_SHA256,
    expected_molclr_sha256: str = MOLCLR_SHA256,
    expected_parent_count: int = 217,
    expected_candidate_count: int = 20,
    expected_pair_count: int = 4340,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Adopt the exact frozen Mutagenicity Ours result into a fresh cell."""

    source = Path(source_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root must be fresh: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    live_writer_audit = scan_live_writers(source, proc_root=proc_root)
    for relative in MUT_OURS_REQUIRED_SOURCE_FILES:
        path = source / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise LegacyStandardizationError(
                f"Mut Ours frozen source lacks required file: {relative}"
            )
    final_manifest_path = source / "final_result_manifest.json"
    if not final_manifest_path.is_file():
        raise LegacyStandardizationError(
            "Mut Ours source lacks final_result_manifest.json checksum closure"
        )

    cache = HashCache()
    closure = verify_manifest_closure(
        source,
        final_manifest_path,
        required_relative_files=MUT_OURS_REQUIRED_SOURCE_FILES,
        hash_cache=cache,
    )
    final_manifest = _load_object(final_manifest_path)
    run_manifest = _load_object(source / "run_manifest.json")
    summary = _load_object(source / "summary.json")
    complete = _load_object(source / "_RUN_COMPLETE.json")
    _assert_bool(complete, "run_complete", True)
    _assert_bool(run_manifest, "run_complete", True)
    _assert_bool(summary, "run_complete", True)
    if final_manifest.get("finalized") is not True:
        raise LegacyStandardizationError(
            "final_result_manifest.finalized is not true"
        )
    expected_final_identity = {
        "dataset": "Mutagenicity",
        "method": "Ours-ChemLLM-PPO-WNode-A2",
        "source_label": 1,
        "target_label": 0,
        "selected_variant": "A2_MultiThreshold",
    }
    for field, expected_value in expected_final_identity.items():
        if final_manifest.get(field) != expected_value:
            raise LegacyStandardizationError(
                f"final_result_manifest.{field}={final_manifest.get(field)!r}; "
                f"expected {expected_value!r}"
            )
    for payload, label in ((run_manifest, "run_manifest"), (summary, "summary")):
        for field in (
            "test_used_for_selection",
            "test_threshold_fitting",
            "test_candidate_selection",
            "test_variant_selection",
        ):
            if payload.get(field) is not False:
                raise LegacyStandardizationError(
                    f"{label}.{field} does not prove calibration/test isolation"
                )
    if final_manifest.get("test_used_for_selection") is not False:
        raise LegacyStandardizationError(
            "final_result_manifest.test_used_for_selection is not false"
        )
    for field, expected_value in {
        "test_parent_count": int(expected_parent_count),
        "top_k": 20,
        "table_k": 10,
    }.items():
        if int(final_manifest.get(field, -1)) != expected_value:
            raise LegacyStandardizationError(
                f"final_result_manifest.{field}={final_manifest.get(field)!r}; "
                f"expected {expected_value}"
            )
    test_run_identity = final_manifest.get("test_run_root")
    if test_run_identity is None:
        raise LegacyStandardizationError(
            "final_result_manifest.test_run_root is missing"
        )
    try:
        resolved_test_run: Path | None = resolve_recorded_path(
            test_run_identity, remap_roots=remap_roots
        )
    except FileNotFoundError:
        resolved_test_run = None
    test_run_live_writer_audit = (
        scan_live_writers(resolved_test_run, proc_root=proc_root)
        if resolved_test_run is not None
        else {
            "source_available": False,
            "reason": "FINALIZED_TEST_RUN_NOT_PRESENT_IN_STEP0_PAYLOAD",
        }
    )
    if str(run_manifest.get("dataset") or "").lower() != "mutagenicity":
        raise LegacyStandardizationError("Mut Ours dataset identity mismatch")
    if str(run_manifest.get("distance_line") or "") != "MolCLR-Node-Wasserstein":
        raise LegacyStandardizationError("Mut Ours distance line is not WNode")
    _assert_bool(run_manifest, "selector_frozen", True)
    _assert_bool(run_manifest, "selector_frozen_before_test", True)
    if (
        str(run_manifest.get("strict_flip_definition") or "")
        != "pred_before == 1 and pred_after == 0"
    ):
        raise LegacyStandardizationError(
            "Mut Ours strict-flip definition is not the frozen 1 -> 0 contract"
        )
    if int(run_manifest.get("source_label", -1)) != 1 or int(
        run_manifest.get("target_label", -1)
    ) != 0:
        raise LegacyStandardizationError("Mut Ours label direction is not 1 -> 0")

    inputs = run_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LegacyStandardizationError("Mut Ours run_manifest.inputs is missing")
    frozen_identity = inputs.get("frozen_selector_root") or final_manifest.get(
        "frozen_selector_root"
    )
    test_identity = inputs.get("test_csv")
    teacher_identity = inputs.get("teacher_path")
    molclr_identity = inputs.get("molclr_checkpoint")
    if any(value is None for value in (frozen_identity, test_identity, teacher_identity, molclr_identity)):
        raise LegacyStandardizationError(
            "Mut Ours run manifest lacks frozen selector/test/RF/MolCLR identities"
        )
    try:
        frozen_root: Path | None = resolve_recorded_path(
            frozen_identity, remap_roots=remap_roots
        )
    except FileNotFoundError:
        frozen_root = None
    test_csv = resolve_recorded_path(test_identity, remap_roots=remap_roots)
    teacher = resolve_recorded_path(teacher_identity, remap_roots=remap_roots)
    molclr = resolve_recorded_path(molclr_identity, remap_roots=remap_roots)
    expected_teacher = _normalize_hash(
        expected_teacher_sha256, label="expected Mutagenicity RF SHA256"
    )
    expected_molclr = _normalize_hash(
        expected_molclr_sha256, label="expected MolCLR SHA256"
    )
    actual_teacher = cache.sha256(teacher)
    actual_molclr = cache.sha256(molclr)
    actual_test_csv = cache.sha256(test_csv)
    if actual_teacher != expected_teacher:
        raise LegacyStandardizationError(
            f"Mut Ours RF oracle SHA mismatch: {actual_teacher}"
        )
    if actual_molclr != expected_molclr:
        raise LegacyStandardizationError(
            f"Mut Ours MolCLR SHA mismatch: {actual_molclr}"
        )
    for identity, actual, label in (
        (teacher_identity, actual_teacher, "teacher"),
        (molclr_identity, actual_molclr, "MolCLR"),
        (test_identity, actual_test_csv, "test CSV"),
    ):
        claimed = _identity_sha(identity)
        if claimed is not None and claimed != actual:
            raise LegacyStandardizationError(
                f"Mut Ours {label} recorded SHA does not match resolved bytes"
            )
        claimed_size = _identity_size(identity)
        resolved_path = {
            "teacher": teacher,
            "MolCLR": molclr,
            "test CSV": test_csv,
        }[label]
        if claimed_size is not None and claimed_size != resolved_path.stat().st_size:
            raise LegacyStandardizationError(
                f"Mut Ours {label} recorded size does not match resolved bytes"
            )

    finalized_audit_evidence = _validate_finalized_audit_evidence(
        source,
        expected_candidate_count=int(expected_candidate_count),
        expected_pair_count=int(expected_pair_count),
    )
    upstream_sources_available = (
        resolved_test_run is not None and frozen_root is not None
    )
    if upstream_sources_available:
        frozen_audit = frozen_test.audit_frozen_test_run(
            resolved_test_run,
            frozen_selector_root=frozen_root,
            test_csv=test_csv,
            expected_parent_count=int(expected_parent_count),
            expected_candidate_count=int(expected_candidate_count),
            expected_pair_count=int(expected_pair_count),
            expected_top_k=20,
            expected_table_k=10,
            require_complete_cartesian=True,
            require_frozen_thresholds=True,
            require_frozen_candidate_order=True,
            require_monotonic_coverage=True,
            require_nonincreasing_capped_cost=True,
        )
    else:
        frozen_audit = _audit_self_contained_finalized_bundle(
            source,
            test_csv=test_csv,
            expected_parent_count=int(expected_parent_count),
            expected_candidate_count=int(expected_candidate_count),
            expected_pair_count=int(expected_pair_count),
        )
    csv_audit = _assert_mut_ours_csvs(source)
    if frozen_audit.get("audit_passed") is not True:
        raise LegacyStandardizationError("Original frozen-test audit did not pass")
    raw_evaluation_hashes: dict[str, str] = {}
    if upstream_sources_available:
        assert resolved_test_run is not None
        for relative in MUT_OURS_SCIENTIFIC_FILES:
            final_path = source / relative
            raw_path = resolved_test_run / relative
            if not raw_path.is_file():
                raise LegacyStandardizationError(
                    f"Underlying frozen test run lacks {relative}"
                )
            final_hash = cache.sha256(final_path)
            raw_hash = cache.sha256(raw_path)
            if final_hash != raw_hash:
                raise LegacyStandardizationError(
                    f"Finalized bundle differs from frozen test run: {relative}"
                )
            raw_evaluation_hashes[relative] = raw_hash
    frozen_thresholds = _load_object(source / "thresholds.json")
    if frozen_thresholds.get("threshold_source") != "frozen_calibration_selector":
        raise LegacyStandardizationError(
            "Mut Ours thresholds do not name the frozen calibration selector"
        )
    for field in (
        "test_threshold_fitting",
        "test_candidate_selection",
        "test_variant_selection",
    ):
        if frozen_thresholds.get(field) is not False:
            raise LegacyStandardizationError(
                f"Mut Ours thresholds do not prove {field}=false"
            )
    split_hash = _normalize_hash(
        summary.get("test_cohort_hash"), label="Mut Ours test cohort hash"
    )
    selector_hash = _normalize_hash(
        summary.get("frozen_selector_hash"), label="Mut Ours frozen selector hash"
    )
    threshold_hash = cache.sha256(source / "thresholds.json")
    selector_threshold_hash = (
        cache.sha256(frozen_root / "thresholds.json")
        if frozen_root is not None
        else _normalize_hash(
            inputs.get("thresholds_sha256"), label="inputs.thresholds_sha256"
        )
    )
    for input_field, actual, label in (
        ("test_cohort_hash", split_hash, "test cohort hash"),
        ("frozen_selector_hash", selector_hash, "frozen selector hash"),
        ("thresholds_sha256", selector_threshold_hash, "selector threshold hash"),
    ):
        recorded = _normalize_hash(inputs.get(input_field), label=f"inputs.{input_field}")
        if recorded != actual:
            raise LegacyStandardizationError(
                f"Mut Ours {label} differs between frozen inputs and audited output"
            )
    selected_ids: list[str] = []
    for line_number, line in enumerate(
        (source / "selected_sequence.jsonl").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            candidate_id = str(row["candidate_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise LegacyStandardizationError(
                f"Invalid selected_sequence.jsonl row {line_number}"
            ) from exc
        selected_ids.append(candidate_id)
    if inputs.get("candidate_ids_in_order") != selected_ids:
        raise LegacyStandardizationError(
            "Mut Ours candidate order differs from run_manifest.inputs"
        )

    protected_paths = [
        source / relative for relative in MUT_OURS_REQUIRED_SOURCE_FILES
    ] + [final_manifest_path, teacher, molclr, test_csv]
    if upstream_sources_available:
        assert resolved_test_run is not None
        protected_paths.extend(
            resolved_test_run / relative
            for relative in MUT_OURS_SCIENTIFIC_FILES
        )
    before = _snapshot(protected_paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    raw = temporary / "raw"
    standardized = temporary / "standardized"
    raw.mkdir(parents=True)
    standardized.mkdir(parents=True)
    try:
        source_hashes = {
            relative: cache.sha256(source / relative)
            for relative in MUT_OURS_REQUIRED_SOURCE_FILES
        }
        adoption_manifest = {
            "schema_version": "am_legacy_adoption_v1",
            "dataset": "Mutagenicity",
            "method": "Ours",
            "status": "ADOPTABLE_PASS",
            "source_root": str(source),
            "source_final_result_manifest": str(final_manifest_path),
            "source_final_result_manifest_sha256": cache.sha256(final_manifest_path),
            "source_artifact_sha256": source_hashes,
            "source_checksum_closure": closure,
            "source_test_run_recorded": _identity_path(test_run_identity),
            "source_test_run_root": (
                str(resolved_test_run) if resolved_test_run is not None else None
            ),
            "source_test_run_artifact_sha256": raw_evaluation_hashes,
            "upstream_test_run_and_selector_available": upstream_sources_available,
            "finalized_audit_evidence": finalized_audit_evidence,
            "generation_adopted": True,
            "ordering_adopted": True,
            "evaluation_adopted": True,
            "generation_rerun": False,
            "selector_rerun": False,
            "evaluation_rerun": False,
            "source_read_only": True,
            "live_writer_audit": {
                "finalized_bundle": live_writer_audit,
                "frozen_test_run": test_run_live_writer_audit,
            },
            "adopted_at": utc_now(),
        }
        _atomic_json(raw / "adoption_manifest.json", adoption_manifest)
        _atomic_json(raw / "source_frozen_test_audit.json", frozen_audit)
        _atomic_json(raw / "source_final_result_manifest.json", final_manifest)
        shutil.copy2(
            source / "_FINALIZED.json", raw / "source_FINALIZED.json"
        )
        shutil.copy2(
            source / "manual_final_test_audit_v2.json",
            raw / "source_manual_final_test_audit_v2.json",
        )
        shutil.copy2(
            source / "threshold_freeze_semantic_audit_v2.json",
            raw / "source_threshold_freeze_semantic_audit_v2.json",
        )
        official_reaudit = source / "official_final_test_reaudit.txt"
        if official_reaudit.is_file():
            shutil.copy2(
                official_reaudit, raw / "source_official_final_test_reaudit.txt"
            )

        for name in (
            "figure3_coverage_vs_k.csv",
            "figure4_coverage_vs_threshold.csv",
            "table2_ours_k10.csv",
            "prefix_metrics.csv",
            "parent_best_distances.csv",
        ):
            _normalize_csv_method(source / name, standardized / name)
        shutil.copy2(source / "prefix_metrics.json", standardized / "prefix_metrics.json")
        _atomic_csv(
            standardized / "destination_distribution.csv",
            [
                {
                    "dataset": "Mutagenicity",
                    "method": "Ours",
                    "destination_label": "N/A",
                    "count": "N/A",
                    "rate": "N/A",
                    "reason": "binary_1_to_0_task_destination_distribution_not_applicable",
                }
            ],
        )
        dataset_hash = actual_test_csv
        for field in ("theta_star", "cost_cap"):
            try:
                summary_value = float(summary[field])
                final_value = float(final_manifest[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise LegacyStandardizationError(
                    f"Mut Ours {field} provenance is missing or non-numeric"
                ) from exc
            if not math.isclose(
                summary_value, final_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise LegacyStandardizationError(
                    f"Mut Ours {field} differs between summary and final manifest"
                )
        oracle_manifest = {
            "schema_version": 1,
            "dataset": "Mutagenicity",
            "oracle_backend": "rf",
            "classifier_family": "random_forest",
            "oracle_checkpoint": str(teacher),
            "oracle_hash": actual_teacher,
            "rf_preserved": True,
            "retrained": False,
        }
        evaluation_manifest = {
            "schema_version": 1,
            "dataset": "Mutagenicity",
            "method": "Ours",
            "distance_line": "MolCLR-Node-Wasserstein",
            "distance_encoder": str(molclr),
            "distance_encoder_hash": actual_molclr,
            "cf_mode": "strict_flip",
            "source_label": 1,
            "target_label": 0,
            "k_max": 20,
            "table2_k": 10,
            "theta_star": summary.get("theta_star"),
            "threshold_config_hash": threshold_hash,
            "frozen_selector_threshold_source_hash": selector_threshold_hash,
            "frozen_selector_source_available": frozen_root is not None,
            "dataset_hash": dataset_hash,
            "dataset_hash_scope": "frozen_test_csv",
            "split_hash": split_hash,
            "selector_hash": selector_hash,
            "selector_fitted_on_calibration": True,
            "selector_frozen_before_test": True,
            "test_used_only_after_freeze": True,
            "test_used_for_selection": False,
            "test_threshold_fitting": False,
            "complete_cartesian": True,
        }
        standardized_summary = dict(summary)
        standardized_summary.update(
            {
                "method": "Ours",
                "raw_method_name": summary.get("method"),
                "dataset": "Mutagenicity",
                "oracle_backend": "rf",
                "classifier_family": "random_forest",
                "oracle_hash": actual_teacher,
                "molclr_checkpoint_hash": actual_molclr,
                "dataset_hash": dataset_hash,
                "split_hash": split_hash,
                "status": "ADOPTABLE_PASS",
                "generation_adopted": True,
                "ordering_adopted": True,
                "evaluation_adopted": True,
            }
        )
        standardized_manifest = {
            "schema_version": "four_by_four_standardized_cell_v1",
            "dataset": "Mutagenicity",
            "method": "Ours",
            "status": "ADOPTABLE_PASS",
            "raw_output_root": str(source),
            "standardized_output_root": str(destination / "standardized"),
            "oracle_backend": "rf",
            "oracle_checkpoint": str(teacher),
            "oracle_hash": actual_teacher,
            "dataset_hash": dataset_hash,
            "split_hash": split_hash,
            "distance_line": "MolCLR-Node-Wasserstein",
            "molclr_checkpoint_hash": actual_molclr,
            "cf_mode": "strict_flip",
            "k_max": 20,
            "table2_k": 10,
            "threshold_config_hash": threshold_hash,
            "frozen_selector_threshold_source_hash": selector_threshold_hash,
            "frozen_selector_source_available": frozen_root is not None,
            "generation_adopted": True,
            "ordering_adopted": True,
            "evaluation_adopted": True,
            "adoption_reason": (
                "exact final-result checksum closure plus independent frozen-test "
                "reconstruction audit"
            ),
            "rerun_reason": "N/A",
            "paper_written": False,
        }
        _atomic_json(standardized / "oracle_manifest.json", oracle_manifest)
        _atomic_json(standardized / "evaluation_manifest.json", evaluation_manifest)
        _atomic_json(standardized / "summary.json", standardized_summary)
        _atomic_json(standardized / "run_manifest.json", standardized_manifest)

        artifact_hashes = {
            path.relative_to(standardized).as_posix(): cache.sha256(path)
            for path in sorted(standardized.iterdir())
            if path.is_file()
        }
        artifact_manifest = {
            "schema_version": 1,
            "files": artifact_hashes,
            "file_count": len(artifact_hashes),
            "self_excluded": "artifact_manifest.json",
            "audit_excluded": "final_artifact_audit.json",
        }
        _atomic_json(standardized / "artifact_manifest.json", artifact_manifest)
        final_audit = {
            "schema_version": "four_by_four_final_artifact_audit_v1",
            "final_artifact_audit_passed": True,
            "audit_passed": True,
            "passed": True,
            "dataset": "Mutagenicity",
            "method": "Ours",
            "status": "ADOPTABLE_PASS",
            "source_checksum_closure_passed": True,
            "source_frozen_test_audit_passed": True,
            "source_test_run_recorded": _identity_path(test_run_identity),
            "source_test_run_root": (
                str(resolved_test_run) if resolved_test_run is not None else None
            ),
            "finalized_bundle_matches_test_run": (
                True if upstream_sources_available else "NOT_AVAILABLE_IN_STEP0"
            ),
            "audit_mode": frozen_audit.get("audit_mode", "full_upstream_replay"),
            "finalized_audit_evidence": finalized_audit_evidence,
            "source_unchanged": True,
            "source_read_only": True,
            "live_writer_audit_passed": True,
            "oracle_backend": "rf",
            "oracle_hash": actual_teacher,
            "distance_line": "MolCLR-Node-Wasserstein",
            "molclr_checkpoint_hash": actual_molclr,
            "cf_mode": "strict_flip",
            "dataset_hash": dataset_hash,
            "split_hash": split_hash,
            "selector_fitted_on_calibration": True,
            "test_used_only_after_freeze": True,
            "complete_cartesian": True,
            "parent_count": int(expected_parent_count),
            "candidate_count": int(expected_candidate_count),
            "pair_count": int(expected_pair_count),
            "csv_audit": csv_audit,
            "artifact_manifest_sha256": cache.sha256(
                standardized / "artifact_manifest.json"
            ),
            "hash_cache_unique_file_count": len(cache.values),
            "hash_cache_duplicate_hash_calls": sum(
                max(0, count - 1) for count in cache.calls.values()
            ),
            "paper_written": False,
            "audited_at": utc_now(),
        }
        _atomic_json(standardized / "final_artifact_audit.json", final_audit)
        after = _snapshot(protected_paths)
        if before != after:
            raise LegacyStandardizationError("Legacy source changed during adoption")
        scan_live_writers(source, proc_root=proc_root)
        if resolved_test_run is not None:
            scan_live_writers(resolved_test_run, proc_root=proc_root)
        _atomic_json(temporary / "run_manifest.json", standardized_manifest)
        _atomic_json(
            temporary / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "audit_passed": True,
                "status": "ADOPTABLE_PASS",
                "standardized_root": str(destination / "standardized"),
                "paper_written": False,
            },
        )
        _atomic_text(temporary / "PASS", "MUT_OURS_LEGACY_ADOPTION_PASS\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "ADOPTABLE_PASS",
        "dataset": "Mutagenicity",
        "method": "Ours",
        "output_root": str(destination),
        "standardized_output_root": str(destination / "standardized"),
        "final_artifact_audit": str(
            destination / "standardized" / "final_artifact_audit.json"
        ),
        "source_root": str(source),
    }


def freeze_mutagenicity_gcf_candidates(
    *,
    source_export_root: str | Path,
    output_root: str | Path,
    expected_csv_sha256: str = MUT_GCF_SELECTED_CSV_SHA256,
    expected_order_sha256: str = MUT_GCF_CANDIDATE_ORDER_SHA256,
    expected_native_ranks: Sequence[int] = MUT_GCF_NATIVE_RANKS,
    expected_selection_method: str = MUT_GCF_SELECTION_METHOD,
    expected_teacher_sha256: str = MUTAGENICITY_RF_SHA256,
    expected_candidate_count: int = 20,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    """Freeze the already-exported GCF Top20 without running generation.

    The legacy exporter produced a complete, preselected CSV but no standalone
    frozen-candidate package.  This function verifies the export's own audit
    closure and exact scientific identity, then copies the CSV into a fresh,
    immutable package understood by the deterministic calibration/test code.
    """

    source = Path(source_export_root).expanduser().resolve(strict=True)
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root must be fresh: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    required = (
        "selected_top20.csv",
        "run_manifest.json",
        "_RUN_COMPLETE.json",
        "filter_summary.json",
        "candidate_filter_audit.jsonl",
    )
    paths = {name: source / name for name in required}
    missing = [
        name
        for name, path in paths.items()
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0
    ]
    if missing:
        raise LegacyStandardizationError(
            f"Mut GCF legacy export lacks complete audit evidence: {missing}"
        )
    live_writer_audit = scan_live_writers(source, proc_root=proc_root)
    cache = HashCache()
    csv_sha = cache.sha256(paths["selected_top20.csv"])
    if csv_sha != _normalize_hash(
        expected_csv_sha256, label="Mut GCF expected selected CSV SHA256"
    ):
        raise LegacyStandardizationError(
            f"Mut GCF selected CSV SHA256 mismatch: actual={csv_sha}"
        )
    try:
        candidates, fields = load_ranked_candidates(
            paths["selected_top20.csv"], expected_count=expected_candidate_count
        )
    except ValueError as exc:
        raise LegacyStandardizationError(
            f"Mut GCF selected CSV contract failed: {exc}"
        ) from exc
    if "native_rank" not in fields:
        raise LegacyStandardizationError("Mut GCF selected CSV lacks native_rank")
    native_ranks = [int(row["native_rank"]) for row in candidates]
    expected_ranks = [int(value) for value in expected_native_ranks]
    if native_ranks != expected_ranks:
        raise LegacyStandardizationError(
            f"Mut GCF native ranks differ: actual={native_ranks}"
        )
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    order_sha = stable_json_sha256(candidate_ids)
    if order_sha != _normalize_hash(
        expected_order_sha256, label="Mut GCF expected candidate order SHA256"
    ):
        raise LegacyStandardizationError(
            f"Mut GCF selected order SHA256 mismatch: actual={order_sha}"
        )

    run_manifest = _load_object(paths["run_manifest.json"])
    complete = _load_object(paths["_RUN_COMPLETE.json"])
    filter_summary = _load_object(paths["filter_summary.json"])
    if complete != run_manifest:
        raise LegacyStandardizationError(
            "Mut GCF _RUN_COMPLETE does not exactly reproduce run_manifest"
        )
    expected_run_fields: dict[str, Any] = {
        "dataset": "Mutagenicity",
        "profile": "full",
        "parent_limit": 1448,
        "selected_count": int(expected_candidate_count),
        "selected_top20_rows": int(expected_candidate_count),
        "candidate_yield_gate_passed": True,
        "full_result_ready": True,
        "candidate_set_preselected": True,
        "selection_performed_in_eval": False,
        "rf_reranking_performed": False,
        "wnode_reranking_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "teacher_used_only_for_target_validation": True,
        "selection_method": str(expected_selection_method),
        "selected_candidate_order_sha256": order_sha,
        "run_complete": True,
    }
    for field, expected in expected_run_fields.items():
        if run_manifest.get(field) != expected:
            raise LegacyStandardizationError(
                f"Mut GCF run_manifest.{field}={run_manifest.get(field)!r}; "
                f"expected {expected!r}"
            )
    teacher_sha = _normalize_hash(
        run_manifest.get("teacher_sha256"), label="Mut GCF teacher_sha256"
    )
    if teacher_sha != _normalize_hash(
        expected_teacher_sha256, label="Mut GCF expected teacher SHA256"
    ):
        raise LegacyStandardizationError("Mut GCF export used a different RF oracle")
    expected_filter_fields: dict[str, Any] = {
        "selected_count": int(expected_candidate_count),
        "audit_complete": True,
        "all_candidates_terminal": True,
        "native_order_preserved": True,
        "candidate_yield_gate_passed": True,
        "rf_reranking_performed": False,
        "wnode_reranking_performed": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "selected_native_ranks": expected_ranks,
    }
    for field, expected in expected_filter_fields.items():
        if filter_summary.get(field) != expected:
            raise LegacyStandardizationError(
                f"Mut GCF filter_summary.{field}={filter_summary.get(field)!r}; "
                f"expected {expected!r}"
            )
    audit_rows = frozen_test._read_jsonl(paths["candidate_filter_audit.jsonl"])
    expected_audit_rows = int(run_manifest.get("candidate_filter_audit_rows", -1))
    if expected_audit_rows <= 0 or len(audit_rows) != expected_audit_rows:
        raise LegacyStandardizationError(
            "Mut GCF candidate filter audit row count is incomplete"
        )
    if any(not str(row.get("rejection_stage") or "").strip() for row in audit_rows):
        raise LegacyStandardizationError(
            "Mut GCF candidate filter audit contains a non-terminal row"
        )
    selected_audit_ranks = [
        int(row["native_rank"])
        for row in audit_rows
        if frozen_test._bool_value(row.get("selected"))
    ]
    if selected_audit_ranks != expected_ranks:
        raise LegacyStandardizationError(
            "Mut GCF filter audit selected ranks differ from frozen CSV"
        )

    protected = list(paths.values())
    before = _snapshot(protected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        export = temporary / "export"
        export.mkdir(parents=True)
        copied_csv = export / "selected_top20.csv"
        shutil.copy2(paths["selected_top20.csv"], copied_csv)
        frozen_manifest = {
            "schema_version": "mut_gcf_frozen_top20_v1",
            "dataset": "Mutagenicity",
            "source_label": 1,
            "target_label": 0,
            "candidate_count": int(expected_candidate_count),
            "candidate_set_preselected": True,
            "selection_performed_in_eval": False,
            "selection_method": str(expected_selection_method),
            "rf_reranking_performed": False,
            "wnode_reranking_performed": False,
            "selected_candidate_order_sha256": order_sha,
            "selected_native_ranks": expected_ranks,
            "file_inventory": {
                "export/selected_top20.csv": {
                    "bytes": copied_csv.stat().st_size,
                    "sha256": cache.sha256(copied_csv),
                }
            },
            "source_export_root": str(source),
            "source_run_manifest_sha256": cache.sha256(paths["run_manifest.json"]),
            "source_filter_summary_sha256": cache.sha256(paths["filter_summary.json"]),
            "source_candidate_filter_audit_sha256": cache.sha256(
                paths["candidate_filter_audit.jsonl"]
            ),
            "generation_rerun": False,
            "selection_rerun": False,
            "frozen_at": utc_now(),
        }
        _atomic_json(temporary / "frozen_candidate_manifest.json", frozen_manifest)
        try:
            validation = validate_frozen_candidate_contract(
                candidates_csv=copied_csv,
                frozen_manifest_path=temporary / "frozen_candidate_manifest.json",
                expected_count=int(expected_candidate_count),
                expected_csv_sha256=csv_sha,
                expected_order_sha256=order_sha,
                expected_native_ranks=expected_ranks,
                expected_selection_method=str(expected_selection_method),
            )
        except ValueError as exc:
            raise LegacyStandardizationError(
                f"Mut GCF frozen package validation failed: {exc}"
            ) from exc
        audit = {
            "schema_version": "mut_gcf_legacy_freeze_audit_v1",
            "audit_passed": True,
            "dataset": "Mutagenicity",
            "method": "GCFExplainer",
            "status": "PASS",
            "source_export_root": str(source),
            "source_live_writer_audit": live_writer_audit,
            "source_file_sha256": {
                name: cache.sha256(path) for name, path in paths.items()
            },
            "frozen_candidate_contract": validation,
            "generation_rerun": False,
            "selection_rerun": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "audited_at": utc_now(),
        }
        _atomic_json(temporary / "audit.json", audit)
        _atomic_json(
            temporary / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "audit_passed": True,
                "candidate_count": int(expected_candidate_count),
                "selected_candidate_order_sha256": order_sha,
                "generation_rerun": False,
            },
        )
        _atomic_text(temporary / "PASS", "MUT_GCF_LEGACY_FREEZE_PASS\n")
        after = _snapshot(protected)
        if before != after:
            raise LegacyStandardizationError(
                "Mut GCF legacy export changed during freezing"
            )
        scan_live_writers(source, proc_root=proc_root)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "dataset": "Mutagenicity",
        "method": "GCFExplainer",
        "output_root": str(destination),
        "frozen_manifest": str(destination / "frozen_candidate_manifest.json"),
        "selected_candidates": str(destination / "export/selected_top20.csv"),
        "generation_rerun": False,
    }


def _bounded_named_evidence(
    root: Path,
    names: Sequence[str],
    *,
    max_entries: int = 50_000,
    max_depth: int = 8,
) -> tuple[dict[str, list[str]], bool]:
    wanted = set(names)
    found: dict[str, list[str]] = {name: [] for name in names}
    scanned = 0
    truncated = False
    root_depth = len(root.parts)
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        if depth >= max_depth:
            directories[:] = []
        directories[:] = [name for name in directories if name not in {".git", "cache", "__pycache__"}]
        scanned += len(directories) + len(files)
        if scanned > max_entries:
            truncated = True
            break
        for name in files:
            if name in wanted and len(found[name]) < 16:
                found[name].append((current_path / name).relative_to(root).as_posix())
    return found, truncated


def _validate_adopted_mut_ours(root_like: str | Path) -> Path:
    root = Path(root_like).expanduser().resolve(strict=True)
    required = (
        "PASS",
        "_RUN_COMPLETE.json",
        "standardized/figure3_coverage_vs_k.csv",
        "standardized/figure4_coverage_vs_threshold.csv",
        "standardized/table2_ours_k10.csv",
        "standardized/summary.json",
        "standardized/run_manifest.json",
        "standardized/oracle_manifest.json",
        "standardized/evaluation_manifest.json",
        "standardized/artifact_manifest.json",
        "standardized/final_artifact_audit.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise LegacyStandardizationError(
            f"Mut Ours adopted dependency is incomplete: {missing}"
        )
    complete = _load_object(root / "_RUN_COMPLETE.json")
    audit = _load_object(root / "standardized/final_artifact_audit.json")
    manifest = _load_object(root / "standardized/run_manifest.json")
    oracle = _load_object(root / "standardized/oracle_manifest.json")
    standardized = root / "standardized"
    artifact_manifest_path = standardized / "artifact_manifest.json"
    artifact_manifest = _load_object(artifact_manifest_path)
    artifact_files = artifact_manifest.get("files")
    if not isinstance(artifact_files, Mapping):
        raise LegacyStandardizationError(
            "Mut Ours artifact_manifest.files is not a mapping"
        )
    expected_names = set(MUT_OURS_REQUIRED_STANDARDIZED_FILES)
    actual_names = set(str(name) for name in artifact_files)
    if actual_names != expected_names:
        raise LegacyStandardizationError(
            "Mut Ours standardized artifact closure differs: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    if int(artifact_manifest.get("file_count", -1)) != len(expected_names):
        raise LegacyStandardizationError(
            "Mut Ours standardized artifact file_count differs"
        )
    for relative, expected_digest in sorted(artifact_files.items()):
        safe = _safe_relative(str(relative))
        if safe != str(relative):
            raise LegacyStandardizationError(
                f"Mut Ours standardized artifact path is not canonical: {relative!r}"
            )
        path = standardized / safe
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise LegacyStandardizationError(
                f"Mut Ours standardized artifact is absent, empty, or a symlink: {safe}"
            )
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != _normalize_hash(
            expected_digest, label=f"artifact_manifest.files[{safe!r}]"
        ):
            raise LegacyStandardizationError(
                f"Mut Ours standardized artifact SHA256 mismatch: {safe}"
            )
    expected_artifact_manifest_hash = str(
        audit.get("artifact_manifest_sha256") or ""
    ).lower()
    if (
        complete.get("run_complete") is not True
        or audit.get("final_artifact_audit_passed") is not True
        or manifest.get("status") != "ADOPTABLE_PASS"
        or str(manifest.get("dataset") or "").lower() != "mutagenicity"
        or manifest.get("method") != "Ours"
        or oracle.get("oracle_backend") != "rf"
        or audit.get("selector_fitted_on_calibration") is not True
        or audit.get("test_used_only_after_freeze") is not True
        or expected_artifact_manifest_hash
        != hashlib.sha256(artifact_manifest_path.read_bytes()).hexdigest()
    ):
        raise LegacyStandardizationError(
            "Mut Ours adopted dependency fails its standardized closure"
        )
    return root


def verify_adopted_mut_ours(
    *,
    adopted_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Verify a pre-controller adoption without reopening raw held-out inputs."""

    adopted = _validate_adopted_mut_ours(adopted_root)
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        run_manifest = adopted / "standardized/run_manifest.json"
        final_audit = adopted / "standardized/final_artifact_audit.json"
        payload = {
            "schema_version": "mut_ours_precontroller_adoption_verification_v1",
            "status": "PASS",
            "adopted_root": str(adopted),
            "adopted_run_manifest": str(run_manifest),
            "adopted_run_manifest_sha256": hashlib.sha256(
                run_manifest.read_bytes()
            ).hexdigest(),
            "final_artifact_audit": str(final_audit),
            "final_artifact_audit_sha256": hashlib.sha256(
                final_audit.read_bytes()
            ).hexdigest(),
            "raw_heldout_input_opened": False,
            "manifest_only": True,
            "verified_at": utc_now(),
        }
        _atomic_json(temporary / "adoption_verification.json", payload)
        _atomic_json(
            temporary / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "manifest_only": True,
                "raw_heldout_input_opened": False,
            },
        )
        _atomic_text(temporary / "PASS", "MUT_OURS_ADOPTION_VERIFY_PASS\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "output_root": str(destination),
        "adopted_root": str(adopted),
        "manifest_only": True,
    }


def audit_legacy_inventory(
    *,
    source_spec: str | Path,
    output_root: str | Path,
    adopted_mut_ours_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inventory old raw evidence without promoting incomplete cells."""

    spec_path = Path(source_spec).expanduser().resolve(strict=True)
    spec = _load_object(spec_path)
    if spec.get("schema_version") != "am_legacy_sources_v1":
        raise LegacyStandardizationError("Unsupported A/M legacy source spec")
    destination = Path(output_root).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Output root must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cells = spec.get("cells")
    if not isinstance(cells, list) or not cells:
        raise LegacyStandardizationError("A/M legacy source spec has no cells")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    try:
        for raw in cells:
            if not isinstance(raw, Mapping):
                raise LegacyStandardizationError("Legacy cell spec must be an object")
            dataset = str(raw.get("dataset") or "").strip()
            method = str(raw.get("method") or "").strip()
            if method.lower() == "clear":
                raise LegacyStandardizationError(
                    "CLEAR is excluded from the four-method matrix and cannot be probed"
                )
            status = str(raw.get("status") or "").strip()
            if status not in ALLOWED_STATUSES:
                raise LegacyStandardizationError(
                    f"Invalid legacy inventory status for {dataset}/{method}: {status}"
                )
            source_roots = raw.get("source_roots")
            if not isinstance(source_roots, list) or not all(
                isinstance(value, str) and Path(value).expanduser().is_absolute()
                for value in source_roots
            ):
                raise LegacyStandardizationError(
                    f"{dataset}/{method} requires absolute source_roots"
                )
            existing = [
                Path(value).expanduser().resolve()
                for value in source_roots
                if Path(value).expanduser().is_dir()
            ]
            if len(existing) > 1:
                raise LegacyStandardizationError(
                    f"Ambiguous legacy roots for {dataset}/{method}: {existing}"
                )
            source = existing[0] if existing else None
            probe_names = [str(value) for value in raw.get("probe_basenames", [])]
            evidence: dict[str, list[str]] = {name: [] for name in probe_names}
            truncated = False
            if source is not None and probe_names:
                evidence, truncated = _bounded_named_evidence(source, probe_names)
            effective_status = status if source is not None else "MISSING"
            if dataset.lower() == "mutagenicity" and method == "Ours":
                if source is None:
                    effective_status = "MISSING"
                elif adopted_mut_ours_root is None:
                    effective_status = "INCOMPLETE"
                else:
                    adopted = _validate_adopted_mut_ours(adopted_mut_ours_root)
                    effective_status = "ADOPTABLE_PASS"
            reason = str(raw.get("reason") or "").strip()
            row = {
                "dataset": dataset,
                "method": method,
                "status": effective_status,
                "raw_output_root": str(source) if source is not None else "",
                "standardized_output_root": (
                    str(Path(adopted_mut_ours_root).resolve() / "standardized")
                    if effective_status == "ADOPTABLE_PASS"
                    and dataset.lower() == "mutagenicity"
                    and method == "Ours"
                    and adopted_mut_ours_root is not None
                    else ""
                ),
                "generation_adopted": bool(raw.get("generation_adopted", False)),
                "ordering_adopted": bool(raw.get("ordering_adopted", False)),
                "evaluation_adopted": bool(raw.get("evaluation_adopted", False)),
                "reason": reason if source is not None else "SOURCE_ROOT_NOT_FOUND",
                "rerun_generation": False,
                "clear_excluded": True,
            }
            if effective_status == "ADOPTABLE_PASS":
                row["generation_adopted"] = True
                row["ordering_adopted"] = True
                row["evaluation_adopted"] = True
                row["reason"] = "STRICT_FROZEN_CELL_ADOPTION_PASS"
            rows.append(row)
            details.append(
                {
                    **row,
                    "configured_source_roots": list(source_roots),
                    "probe_basenames": probe_names,
                    "evidence_paths": evidence,
                    "inventory_truncated": truncated,
                    "missing_capabilities": list(raw.get("missing_capabilities", [])),
                    "recommended_action": raw.get("recommended_action"),
                    "native_action_semantics": raw.get("native_action_semantics"),
                    "notes": raw.get("notes"),
                }
            )
        _atomic_csv(temporary / "matrix_patch.csv", rows)
        _atomic_json(
            temporary / "matrix_patch.json",
            {
                "schema_version": "four_by_four_matrix_patch_v1",
                "cells": rows,
            },
        )
        _atomic_json(
            temporary / "legacy_artifact_inventory.json",
            {
                "schema_version": "am_legacy_inventory_v1",
                "source_spec": str(spec_path),
                "source_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                "cells": details,
                "clear_excluded": True,
                "generation_rerun": False,
                "paper_written": False,
                "audited_at": utc_now(),
            },
        )
        _atomic_json(
            temporary / "_RUN_COMPLETE.json",
            {
                "run_complete": True,
                "audit_completed": True,
                "cell_count": len(rows),
                "adoptable_pass_count": sum(
                    row["status"] == "ADOPTABLE_PASS" for row in rows
                ),
                "paper_written": False,
            },
        )
        _atomic_text(temporary / "PASS", "AM_LEGACY_INVENTORY_PASS\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "PASS",
        "output_root": str(destination),
        "cell_count": len(rows),
        "adoptable_pass_count": sum(
            row["status"] == "ADOPTABLE_PASS" for row in rows
        ),
        "matrix_patch": str(destination / "matrix_patch.json"),
    }


def load_source_spec(path_like: str | Path) -> dict[str, Any]:
    payload = _load_object(path_like)
    if payload.get("schema_version") != "am_legacy_sources_v1":
        raise LegacyStandardizationError("Unsupported A/M legacy source spec")
    return payload


def mut_ours_contract_from_spec(
    source_spec: str | Path,
) -> dict[str, Any]:
    spec = load_source_spec(source_spec)
    matches = [
        row
        for row in spec.get("cells", [])
        if isinstance(row, Mapping)
        and str(row.get("dataset") or "").lower() == "mutagenicity"
        and str(row.get("method") or "") == "Ours"
    ]
    if len(matches) != 1:
        raise LegacyStandardizationError(
            "Source spec must contain exactly one Mutagenicity/Ours cell"
        )
    row = dict(matches[0])
    roots = [
        Path(value).expanduser().resolve()
        for value in row.get("source_roots", [])
        if Path(value).expanduser().is_dir()
    ]
    if len(roots) != 1:
        raise LegacyStandardizationError(
            f"Expected exactly one existing Mutagenicity/Ours source root: {roots}"
        )
    remap_roots = spec.get("path_remap_roots") or []
    if not isinstance(remap_roots, list) or not remap_roots:
        raise LegacyStandardizationError("Source spec lacks path_remap_roots")
    return {
        "source_root": roots[0],
        "remap_roots": [str(value) for value in remap_roots],
        "expected_teacher_sha256": str(
            row.get("expected_teacher_sha256") or MUTAGENICITY_RF_SHA256
        ),
        "expected_molclr_sha256": str(
            row.get("expected_molclr_sha256") or MOLCLR_SHA256
        ),
        "expected_parent_count": int(row.get("expected_parent_count", 217)),
        "expected_candidate_count": int(row.get("expected_candidate_count", 20)),
        "expected_pair_count": int(row.get("expected_pair_count", 4340)),
    }


def mut_gcf_contract_from_spec(source_spec: str | Path) -> dict[str, Any]:
    """Load the exact legacy GCF export-freeze contract from the source spec."""

    spec = load_source_spec(source_spec)
    matches = [
        row
        for row in spec.get("cells", [])
        if isinstance(row, Mapping)
        and str(row.get("dataset") or "").lower() == "mutagenicity"
        and str(row.get("method") or "") == "GCFExplainer"
    ]
    if len(matches) != 1:
        raise LegacyStandardizationError(
            "Source spec must contain exactly one Mutagenicity/GCFExplainer cell"
        )
    row = dict(matches[0])
    roots = [
        Path(value).expanduser().resolve()
        for value in row.get("source_roots", [])
        if Path(value).expanduser().is_dir()
    ]
    if len(roots) != 1:
        raise LegacyStandardizationError(
            f"Expected exactly one existing Mutagenicity/GCF export root: {roots}"
        )
    ranks = row.get("expected_native_ranks", MUT_GCF_NATIVE_RANKS)
    if not isinstance(ranks, list) and not isinstance(ranks, tuple):
        raise LegacyStandardizationError(
            "Mutagenicity/GCF expected_native_ranks must be a list"
        )
    return {
        "source_export_root": roots[0],
        "expected_csv_sha256": str(
            row.get("expected_csv_sha256") or MUT_GCF_SELECTED_CSV_SHA256
        ),
        "expected_order_sha256": str(
            row.get("expected_order_sha256") or MUT_GCF_CANDIDATE_ORDER_SHA256
        ),
        "expected_native_ranks": [int(value) for value in ranks],
        "expected_selection_method": str(
            row.get("expected_selection_method") or MUT_GCF_SELECTION_METHOD
        ),
        "expected_teacher_sha256": str(
            row.get("expected_teacher_sha256") or MUTAGENICITY_RF_SHA256
        ),
        "expected_candidate_count": int(row.get("expected_candidate_count", 20)),
    }


__all__ = [
    "ALLOWED_STATUSES",
    "HashCache",
    "LegacyStandardizationError",
    "MOLCLR_SHA256",
    "MUT_GCF_CANDIDATE_ORDER_SHA256",
    "MUT_GCF_NATIVE_RANKS",
    "MUT_GCF_SELECTED_CSV_SHA256",
    "MUT_GCF_SELECTION_METHOD",
    "MUTAGENICITY_RF_SHA256",
    "MUT_OURS_REQUIRED_SOURCE_FILES",
    "MUT_OURS_REQUIRED_STANDARDIZED_FILES",
    "adopt_mutagenicity_ours",
    "audit_legacy_inventory",
    "freeze_mutagenicity_gcf_candidates",
    "load_source_spec",
    "mut_ours_contract_from_spec",
    "mut_gcf_contract_from_spec",
    "resolve_recorded_path",
    "scan_live_writers",
    "verify_manifest_closure",
    "verify_adopted_mut_ours",
]
