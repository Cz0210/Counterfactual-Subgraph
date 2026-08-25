"""Fail-closed public-artifact audit for private TasteMolNet research.

The audit consumes a separately constructed, manifest-closed public root.  It
never sanitizes or copies private outputs.  Dataset rows, graph caches,
per-molecule predictions, model checkpoints, archives, opaque binaries, and
unregistered files are rejected even when renamed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

from src.utils.tastemolnet_research_policy import (
    SOURCE_CSV_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_TERMS_STATUS,
    TasteResearchPolicy,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    sha256_file,
    stable_json_sha256,
)


PUBLIC_MANIFEST_SCHEMA = "tastemolnet_public_release_manifest_v1"
AUDIT_SCHEMA = "tastemolnet_public_artifact_audit_v1"
AUDIT_STATUS = "PUBLIC_REPORT_NO_DATA_REDISTRIBUTION_VERIFIED"
AUDIT_MARKER = "TASTEMOLNET_PUBLIC_ARTIFACT_NO_DATA_REDISTRIBUTION_AUDIT"
PREPARED_OUTPUT_MANIFEST_SCHEMA = 1
GRAPH_CACHE_MANIFEST_SCHEMA = "molecular_graph_cache_manifest_v1"
MAX_STRUCTURED_BYTES = 16 * 1024 * 1024
MAX_SVG_BYTES = 32 * 1024 * 1024
MAX_TABLE_ROWS = 1000

_HEX = re.compile(r"[0-9a-f]{64}")
_SAFE_RELATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*")
_TASTE_ID = re.compile(r"\bTASTE_[0-9A-F]{8,}\b")
_PRIVATE_PATH = re.compile(
    r"(?i)(?:/autodl-fs/|/root/autodl-fs/|/data/).*tastemolnet"
)
_MOLECULE_ROW_HEADER = re.compile(
    r"(?i)(?:^|[,\t|])(?:raw_smiles|canonical_smiles|model_smiles|smiles|"
    r"molecule_id|compound_id|source_row_id)(?:[,\t|]|$)"
)
_FORBIDDEN_KEY_TOKENS = {
    "smiles",
    "raw_smiles",
    "canonical_smiles",
    "model_smiles",
    "molecule_id",
    "molecule_ids",
    "compound_id",
    "compound_ids",
    "source_row_id",
    "source_row_ids",
    "graph_features",
    "node_features",
    "edge_index",
    "edge_attr",
    "embedding",
    "embeddings",
    "validation_predictions",
    "per_example_predictions",
    "candidate_rows",
    "counterfactual_rows",
}
_ALLOWED_ROLES = {
    "aggregate_metrics",
    "aggregate_confusion_matrix",
    "aggregate_split_counts",
    "aggregate_table",
    "aggregate_figure_svg",
    "method_configuration",
    "provenance_hashes",
}
_AGGREGATE_TABLE_COLUMNS = {
    "dataset",
    "method",
    "metric",
    "split",
    "class_label",
    "class_name",
    "value",
    "mean",
    "std",
    "count",
    "ci_lower",
    "ci_upper",
    "rank",
    "seed",
    "k",
    "threshold",
}


class TastePublicArtifactError(RuntimeError):
    """A proposed public artifact could redistribute private Taste data."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _physical_bytes(path: Path, *, maximum: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TastePublicArtifactError(
                f"public artifact must be one physical, unlinked regular file: {path}"
            )
        if maximum is not None and before.st_size > maximum:
            raise TastePublicArtifactError(f"public artifact is too large: {path}")
        parts: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            parts.append(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        projection = lambda value: (  # noqa: E731
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
            value.st_nlink,
        )
        if projection(before) != projection(after) or projection(after) != projection(current):
            raise TastePublicArtifactError(f"public artifact changed while read: {path}")
        return b"".join(parts)
    finally:
        os.close(descriptor)


def _json_object(data: bytes, *, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TastePublicArtifactError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise TastePublicArtifactError(f"JSON artifact must be one object: {path}")
    return value


def _safe_relative(value: Any) -> str:
    raw = str(value or "")
    if not raw or _SAFE_RELATIVE.fullmatch(raw) is None:
        raise TastePublicArtifactError(f"unsafe public artifact path: {raw!r}")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise TastePublicArtifactError(f"unsafe public artifact path: {raw!r}")
    return pure.as_posix()


def _scan_strings(value: Any, *, key: str | None = None) -> None:
    if key is not None and key.strip().lower() in _FORBIDDEN_KEY_TOKENS:
        raise TastePublicArtifactError(f"forbidden molecule-level field: {key}")
    if isinstance(value, Mapping):
        lowered = {str(item).strip().lower() for item in value}
        if "predicted_label" in lowered and (
            "probabilities" in lowered or "logits" in lowered or "label" in lowered
        ):
            raise TastePublicArtifactError("per-example prediction object is forbidden")
        for child_key, child in value.items():
            _scan_strings(child, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        if len(value) > MAX_TABLE_ROWS:
            raise TastePublicArtifactError("public structured array exceeds aggregate bound")
        for child in value:
            _scan_strings(child)
    elif isinstance(value, str):
        if _TASTE_ID.search(value) or _PRIVATE_PATH.search(value):
            raise TastePublicArtifactError("molecule identifier or private path is forbidden")
        if _MOLECULE_ROW_HEADER.search(value):
            raise TastePublicArtifactError("embedded molecule-row table is forbidden")


def _numeric_tree(value: Any, *, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise TastePublicArtifactError(f"{field} contains an invalid key")
            _numeric_tree(child, field=f"{field}.{key}")
    elif isinstance(value, list):
        if len(value) > MAX_TABLE_ROWS:
            raise TastePublicArtifactError(f"{field} exceeds aggregate bound")
        for index, child in enumerate(value):
            _numeric_tree(child, field=f"{field}[{index}]")
    elif not isinstance(value, (int, float, bool)) and value is not None:
        raise TastePublicArtifactError(f"{field} must contain aggregate scalars only")


def _validate_json_role(role: str, payload: Mapping[str, Any]) -> None:
    _scan_strings(payload)
    if payload.get("dataset") != "tastemolnet":
        raise TastePublicArtifactError(f"{role} must declare dataset=tastemolnet")
    if role == "aggregate_metrics":
        if set(payload) != {"schema_version", "dataset", "num_classes", "source_label", "metrics"}:
            raise TastePublicArtifactError("aggregate_metrics schema changed")
        if payload.get("schema_version") != "tastemolnet_public_aggregate_metrics_v1":
            raise TastePublicArtifactError("aggregate_metrics schema changed")
        if payload.get("num_classes") != 3 or payload.get("source_label") != 1:
            raise TastePublicArtifactError("aggregate_metrics class semantics changed")
        _numeric_tree(payload.get("metrics"), field="metrics")
    elif role == "aggregate_confusion_matrix":
        if set(payload) != {"schema_version", "dataset", "split", "labels", "matrix"}:
            raise TastePublicArtifactError("aggregate_confusion_matrix schema changed")
        if payload.get("schema_version") != "tastemolnet_public_confusion_matrix_v1":
            raise TastePublicArtifactError("aggregate_confusion_matrix schema changed")
        if payload.get("labels") != [0, 1, 2] or payload.get("split") not in {
            "validation",
            "calibration",
            "test",
        }:
            raise TastePublicArtifactError("confusion-matrix semantics changed")
        matrix = payload.get("matrix")
        if (
            not isinstance(matrix, list)
            or len(matrix) != 3
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row)
                for row in matrix
            )
        ):
            raise TastePublicArtifactError("confusion matrix must be 3x3 nonnegative integers")
    elif role == "aggregate_split_counts":
        if set(payload) != {"schema_version", "dataset", "counts"}:
            raise TastePublicArtifactError("aggregate_split_counts schema changed")
        if payload.get("schema_version") != "tastemolnet_public_split_counts_v1":
            raise TastePublicArtifactError("aggregate_split_counts schema changed")
        counts = payload.get("counts")
        if not isinstance(counts, Mapping) or set(counts) != {
            "train",
            "validation",
            "calibration",
            "test",
        }:
            raise TastePublicArtifactError("split-count keys changed")
        for split, values in counts.items():
            if (
                not isinstance(values, Mapping)
                or set(values) != {"0", "1", "2"}
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values())
            ):
                raise TastePublicArtifactError(f"invalid aggregate counts for {split}")
    elif role == "method_configuration":
        if set(payload) != {
            "schema_version",
            "dataset",
            "oracle_backend",
            "classifier_family",
            "rf_oracle_used",
            "num_classes",
            "source_label",
            "source_label_name",
            "counterfactual_mode",
            "hyperparameters",
        }:
            raise TastePublicArtifactError("method_configuration schema changed")
        if payload.get("schema_version") != "tastemolnet_public_method_configuration_v1":
            raise TastePublicArtifactError("method_configuration schema changed")
        if payload.get("num_classes") != 3 or payload.get("source_label") != 1:
            raise TastePublicArtifactError("method_configuration class semantics changed")
        if payload.get("oracle_backend") != "gnn" or payload.get("classifier_family") != "gine":
            raise TastePublicArtifactError("method_configuration must use GINE")
        if payload.get("rf_oracle_used") is not False:
            raise TastePublicArtifactError("RF provenance is forbidden")
        if (
            payload.get("source_label_name") != "Sweet"
            or payload.get("counterfactual_mode") != "untargeted_strict_flip"
        ):
            raise TastePublicArtifactError("method_configuration task semantics changed")
        if not isinstance(payload.get("hyperparameters"), Mapping):
            raise TastePublicArtifactError("method_configuration hyperparameters changed")
        _numeric_tree(payload["hyperparameters"], field="hyperparameters")
    elif role == "provenance_hashes":
        if set(payload) != {
            "schema_version",
            "dataset",
            "upstream_commit",
            "upstream_terms_status",
            "policy_file_sha256",
            "policy_canonical_sha256",
            "prepared_output_manifest_sha256",
            "split_manifest_sha256",
            "graph_cache_manifest_sha256",
            "classifier_checkpoint_sha256",
            "feature_schema_sha256",
        }:
            raise TastePublicArtifactError("provenance_hashes schema changed")
        if payload.get("schema_version") != "tastemolnet_public_provenance_hashes_v1":
            raise TastePublicArtifactError("provenance_hashes schema changed")
        for key, value in payload.items():
            if key in {"schema_version", "dataset", "upstream_commit", "upstream_terms_status"}:
                continue
            if not isinstance(value, str) or _HEX.fullmatch(value) is None:
                raise TastePublicArtifactError(f"provenance field {key} is not a SHA-256")
        if payload.get("upstream_commit") != UPSTREAM_COMMIT:
            raise TastePublicArtifactError("upstream commit changed")
        if payload.get("upstream_terms_status") != UPSTREAM_TERMS_STATUS:
            raise TastePublicArtifactError("upstream terms status changed")
    else:  # pragma: no cover - caller dispatches exact role set.
        raise TastePublicArtifactError(f"unsupported JSON role: {role}")


def _validate_table(data: bytes) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TastePublicArtifactError("aggregate table must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)) or not set(fields) <= _AGGREGATE_TABLE_COLUMNS:
        raise TastePublicArtifactError("aggregate table has unsafe columns")
    if "dataset" not in fields or "metric" not in fields:
        raise TastePublicArtifactError("aggregate table lacks dataset/metric columns")
    count = 0
    for row in reader:
        count += 1
        if count > MAX_TABLE_ROWS:
            raise TastePublicArtifactError("aggregate table has too many rows")
        if row.get("dataset") != "tastemolnet":
            raise TastePublicArtifactError("aggregate table contains another dataset")
        _scan_strings(row)


def _validate_svg(data: bytes) -> None:
    try:
        root = ET.fromstring(data.decode("utf-8"))
    except (UnicodeDecodeError, ET.ParseError) as exc:
        raise TastePublicArtifactError("aggregate SVG is invalid") from exc
    allowed = {
        "svg",
        "g",
        "defs",
        "clipPath",
        "path",
        "rect",
        "circle",
        "line",
        "polyline",
        "polygon",
        "text",
        "tspan",
        "style",
        "title",
        "desc",
    }
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in allowed:
            raise TastePublicArtifactError(f"aggregate SVG tag is forbidden: {tag}")
        for key, value in element.attrib.items():
            name = key.rsplit("}", 1)[-1].lower()
            if name in {"href", "src", "onclick", "onload"} or "url(" in value.lower():
                raise TastePublicArtifactError("aggregate SVG external/embedded content is forbidden")
        if element.text:
            _scan_strings(element.text)


def _manifest_file_set(root: Path) -> set[str]:
    result: set[str] = set()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            child = base / name
            info = os.lstat(child)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TastePublicArtifactError(f"public root contains a nonphysical directory: {child}")
        for name in files:
            child = base / name
            info = os.lstat(child)
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise TastePublicArtifactError(f"public root contains symlink/special file: {child}")
            result.add(child.relative_to(root).as_posix())
    return result


def _load_protected_hashes(prepared_root: Path, cache_root: Path) -> tuple[set[str], dict[str, str]]:
    prepared_manifest_path = prepared_root / "output_manifest.json"
    cache_manifest_path = cache_root / "manifest.json"
    prepared_manifest = _json_object(
        _physical_bytes(prepared_manifest_path, maximum=MAX_STRUCTURED_BYTES),
        path=prepared_manifest_path,
    )
    if prepared_manifest.get("schema_version") != PREPARED_OUTPUT_MANIFEST_SCHEMA:
        raise TastePublicArtifactError("prepared output manifest schema changed")
    files = prepared_manifest.get("files")
    if not isinstance(files, Mapping) or stable_json_sha256(files) != prepared_manifest.get("manifest_digest"):
        raise TastePublicArtifactError("prepared output manifest digest changed")
    protected: set[str] = {SOURCE_CSV_SHA256}
    for relative, identity in files.items():
        safe = _safe_relative(relative)
        if not isinstance(identity, Mapping) or _HEX.fullmatch(str(identity.get("sha256") or "")) is None:
            raise TastePublicArtifactError("prepared output manifest file identity changed")
        source = prepared_root / safe
        if sha256_file(source) != identity["sha256"]:
            raise TastePublicArtifactError(f"prepared file hash changed: {safe}")
        protected.add(str(identity["sha256"]))

    cache_manifest = _json_object(
        _physical_bytes(cache_manifest_path, maximum=MAX_STRUCTURED_BYTES),
        path=cache_manifest_path,
    )
    if (
        cache_manifest.get("schema_version") != GRAPH_CACHE_MANIFEST_SCHEMA
        or cache_manifest.get("dataset") != "tastemolnet"
        or cache_manifest.get("num_classes") != 3
        or cache_manifest.get("split_order") != ["train", "validation", "calibration", "test"]
    ):
        raise TastePublicArtifactError("graph-cache manifest contract changed")
    splits = cache_manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation", "calibration", "test"}:
        raise TastePublicArtifactError("graph-cache split inventory changed")
    for split, identity in splits.items():
        if not isinstance(identity, Mapping):
            raise TastePublicArtifactError(f"graph-cache {split} identity changed")
        filename = _safe_relative(identity.get("cache_file"))
        expected = str(identity.get("cache_sha256") or "")
        if _HEX.fullmatch(expected) is None or sha256_file(cache_root / filename) != expected:
            raise TastePublicArtifactError(f"graph-cache {split} hash changed")
        protected.add(expected)
    protected.add(sha256_file(prepared_manifest_path))
    protected.add(sha256_file(cache_manifest_path))
    identities = {
        "prepared_output_manifest_sha256": sha256_file(prepared_manifest_path),
        "graph_cache_manifest_sha256": sha256_file(cache_manifest_path),
    }
    return protected, identities


def audit_tastemolnet_public_artifacts(
    *,
    public_root: str | Path,
    policy_path: str | Path,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
    expected_policy_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit one exact public tree without copying or modifying it."""

    policy = load_tastemolnet_research_policy(
        policy_path, expected_file_sha256=expected_policy_sha256
    )
    policy.require_active()
    root = Path(public_root).expanduser().resolve(strict=True)
    prepared = Path(prepared_root).expanduser().resolve(strict=True)
    cache = Path(graph_cache_root).expanduser().resolve(strict=True)
    for value, label in ((root, "public"), (prepared, "prepared"), (cache, "cache")):
        info = os.lstat(value)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise TastePublicArtifactError(f"{label} root must be one physical directory")
    if root == prepared or root == cache or root in prepared.parents or root in cache.parents:
        raise TastePublicArtifactError("public root may not contain a private data root")
    if prepared in root.parents or cache in root.parents:
        raise TastePublicArtifactError("public root may not live under private data/cache")

    protected_hashes, private_identities = _load_protected_hashes(prepared, cache)
    manifest_path = root / "public_release_manifest.json"
    manifest_data = _physical_bytes(manifest_path, maximum=MAX_STRUCTURED_BYTES)
    manifest = _json_object(manifest_data, path=manifest_path)
    if set(manifest) != {
        "schema_version",
        "dataset",
        "policy_file_sha256",
        "policy_canonical_sha256",
        "upstream_terms_status",
        "dataset_redistribution_allowed",
        "artifacts",
    }:
        raise TastePublicArtifactError("public release manifest schema changed")
    if (
        manifest.get("schema_version") != PUBLIC_MANIFEST_SCHEMA
        or manifest.get("dataset") != "tastemolnet"
        or manifest.get("policy_file_sha256") != policy.file_sha256
        or manifest.get("policy_canonical_sha256") != policy.canonical_sha256
        or manifest.get("upstream_terms_status") != UPSTREAM_TERMS_STATUS
        or manifest.get("dataset_redistribution_allowed") is not False
    ):
        raise TastePublicArtifactError("public release policy binding changed")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise TastePublicArtifactError("public release manifest requires artifacts")
    expected_files = {"public_release_manifest.json"}
    seen_casefold: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "role",
            "sha256",
            "contains_molecule_level_content",
        }:
            raise TastePublicArtifactError("public artifact row schema changed")
        relative = _safe_relative(row.get("path"))
        folded = relative.casefold()
        if relative in expected_files or folded in seen_casefold:
            raise TastePublicArtifactError("duplicate/case-colliding public artifact path")
        seen_casefold.add(folded)
        expected_files.add(relative)
        role = str(row.get("role") or "")
        if role not in _ALLOWED_ROLES:
            raise TastePublicArtifactError(f"public artifact role is forbidden: {role}")
        if row.get("contains_molecule_level_content") is not False:
            raise TastePublicArtifactError("molecule-level content attestation must be false")
        path = root / relative
        maximum = MAX_SVG_BYTES if role == "aggregate_figure_svg" else MAX_STRUCTURED_BYTES
        data = _physical_bytes(path, maximum=maximum)
        observed = _sha256_bytes(data)
        if row.get("sha256") != observed:
            raise TastePublicArtifactError(f"public artifact hash changed: {relative}")
        if observed in protected_hashes:
            raise TastePublicArtifactError(f"public artifact copies protected dataset/cache bytes: {relative}")
        suffix = path.suffix.lower()
        if role == "aggregate_table":
            if suffix != ".csv":
                raise TastePublicArtifactError("aggregate table must be CSV")
            _validate_table(data)
        elif role == "aggregate_figure_svg":
            if suffix != ".svg":
                raise TastePublicArtifactError("aggregate figure must be inspected SVG")
            _validate_svg(data)
        else:
            if suffix != ".json":
                raise TastePublicArtifactError(f"{role} must be JSON")
            _validate_json_role(role, _json_object(data, path=path))
        artifacts.append({"path": relative, "role": role, "sha256": observed, "bytes": len(data)})
    current_files = _manifest_file_set(root)
    if current_files != expected_files:
        raise TastePublicArtifactError(
            "public artifact inventory changed: "
            f"missing={sorted(expected_files-current_files)}, "
            f"extra={sorted(current_files-expected_files)}"
        )
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": AUDIT_STATUS,
        "dataset": "tastemolnet",
        "upstream_terms_status": UPSTREAM_TERMS_STATUS,
        "dataset_redistribution_allowed": False,
        "policy": policy.evidence(),
        "public_root": str(root),
        "public_release_manifest_sha256": _sha256_bytes(manifest_data),
        "private_authority": private_identities,
        "protected_source_hash_count": len(protected_hashes),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "dataset_payloads_copied": False,
        "audit_marker": AUDIT_MARKER,
    }


__all__ = [
    "AUDIT_MARKER",
    "AUDIT_SCHEMA",
    "AUDIT_STATUS",
    "PUBLIC_MANIFEST_SCHEMA",
    "TastePublicArtifactError",
    "audit_tastemolnet_public_artifacts",
]
