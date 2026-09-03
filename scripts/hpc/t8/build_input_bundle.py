#!/usr/bin/env python3
"""Build the minimal immutable TasteMolNet T8 CPU-mining input bundle.

The builder intentionally accepts only six named, small, immutable inputs.
It never walks a source directory.  This prevents a transfer from accidentally
capturing model weights, live SQLite journals, or transient checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_VERSION = 1
ROUTE_KIND = "T8_T13_GRADE_GLOBALGCE_EXACT_CPU_OFFLOAD"
EXPECTED_TARGETS = (0, 2)
EXPECTED_SCIENCE_CONFIG = {
    "source_label": 1,
    "seed": 7,
    "epochs": 100,
    "min_support": 2,
    "min_vertices": 3,
    "max_vertices": 20,
    "top_k": 20,
    "root_count": 50,
}
ROLE_TO_NAME = {
    "graph_jsonl": "production_graphs.jsonl",
    "source_input_manifest": "source_input_manifest.json",
    "split_manifest": "split_manifest.json",
    "train_cohort_manifest": "train_cohort_manifest.json",
    "feature_schema": "feature_schema.json",
    "data_use_authorization": "data_use_authorization.json",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".db",
    ".h5",
    ".hdf5",
    ".joblib",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
}
FORBIDDEN_NAME_PARTS = (
    "checkpoint.tmp",
    "-journal",
    "-shm",
    "-wal",
    ".journal",
    ".tmp",
    ".wal",
)


class BundleContractError(RuntimeError):
    """Raised when a source cannot enter the immutable transfer bundle."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_hex_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise BundleContractError(f"{field} must be a lowercase-or-uppercase SHA-256")
    return normalized


def _require_hex_commit(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 40 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise BundleContractError(f"{field} must be an exact 40-hex commit")
    return normalized


def _atomic_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(payload))


def _stat_projection(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
    }


def _parse_role_values(values: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        role, separator, item = value.partition("=")
        if not separator or role not in ROLE_TO_NAME or not item:
            raise BundleContractError(
                f"{option} requires one ROLE=VALUE for each of {sorted(ROLE_TO_NAME)}"
            )
        if role in parsed:
            raise BundleContractError(f"duplicate {option} role: {role}")
        parsed[role] = item
    missing = sorted(set(ROLE_TO_NAME).difference(parsed))
    extra = sorted(set(parsed).difference(ROLE_TO_NAME))
    if missing or extra:
        raise BundleContractError(f"{option} role mismatch: missing={missing}, extra={extra}")
    return parsed


def _within(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _reject_forbidden_name(path: Path) -> None:
    lower = path.name.lower()
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or any(part in lower for part in FORBIDDEN_NAME_PARTS):
        raise BundleContractError(f"forbidden mutable/weight payload: {path}")
    if any(token in lower for token in ("model", "weight")) and path.suffix.lower() not in {".json"}:
        raise BundleContractError(f"possible model-weight payload is not allowed: {path}")


@dataclass(frozen=True)
class FrozenCopy:
    role: str
    source_path: str
    archive_path: str
    sha256: str
    size_bytes: int
    stat_before: dict[str, int]
    stat_after: dict[str, int]
    source_sha256_before: str
    source_sha256_after: str
    copied_sha256: str


def _copy_frozen_regular_file(
    *,
    role: str,
    source: Path,
    destination: Path,
    expected_sha256: str,
    allowed_roots: Sequence[Path],
    stability_window_seconds: float,
) -> FrozenCopy:
    if source.is_symlink():
        raise BundleContractError(f"symlink source is forbidden for {role}: {source}")
    try:
        before_lstat = source.lstat()
    except FileNotFoundError as exc:
        raise BundleContractError(f"missing source for {role}: {source}") from exc
    if not stat.S_ISREG(before_lstat.st_mode):
        raise BundleContractError(f"source is not a regular file for {role}: {source}")
    resolved = source.resolve(strict=True)
    if not _within(resolved, allowed_roots):
        raise BundleContractError(f"source for {role} is outside all allowed roots: {resolved}")
    _reject_forbidden_name(resolved)

    normalized_expected = _require_hex_sha256(expected_sha256, field=f"expected SHA-256 for {role}")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(source, flags)
    try:
        first_fd_stat = os.fstat(fd)
        if _stat_projection(first_fd_stat) != _stat_projection(before_lstat):
            raise BundleContractError(f"source identity changed while opening {role}")
        digest = hashlib.sha256()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        first_sha = digest.hexdigest()
        if first_sha != normalized_expected:
            raise BundleContractError(
                f"source SHA-256 mismatch for {role}: expected {normalized_expected}, got {first_sha}"
            )
        if stability_window_seconds:
            time.sleep(stability_window_seconds)
        os.lseek(fd, 0, os.SEEK_SET)
        second_digest = hashlib.sha256()
        for block in iter(lambda: os.read(fd, 1024 * 1024), b""):
            second_digest.update(block)
        second_sha = second_digest.hexdigest()
        after_fd_stat = os.fstat(fd)
    finally:
        os.close(fd)

    after_lstat = source.lstat()
    before_projection = _stat_projection(before_lstat)
    after_projection = _stat_projection(after_lstat)
    if before_projection != after_projection or _stat_projection(after_fd_stat) != after_projection:
        raise BundleContractError(f"source stat changed while freezing {role}")
    if first_sha != second_sha:
        raise BundleContractError(f"source content changed while freezing {role}")
    copied_sha = _sha256_file(destination)
    if copied_sha != first_sha:
        raise BundleContractError(f"copied payload hash mismatch for {role}")
    os.chmod(destination, 0o444)
    return FrozenCopy(
        role=role,
        source_path=str(resolved),
        archive_path=f"payload/{ROLE_TO_NAME[role]}",
        sha256=first_sha,
        size_bytes=int(before_lstat.st_size),
        stat_before=before_projection,
        stat_after=after_projection,
        source_sha256_before=first_sha,
        source_sha256_after=second_sha,
        copied_sha256=copied_sha,
    )


def _read_json(path: Path, *, role: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleContractError(f"invalid JSON payload for {role}: {path}") from exc


def _validate_graph_jsonl(path: Path, *, expected_graph_count: int) -> dict[str, Any]:
    rows = 0
    graph_ids: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise BundleContractError(f"blank graph JSONL row at line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleContractError(f"invalid graph JSONL at line {line_number}") from exc
            if type(payload) is not dict or set(payload) != {"graph_id", "nodes", "edges"}:
                raise BundleContractError(
                    f"graph JSONL row {line_number} must contain exactly graph_id,nodes,edges"
                )
            graph_id = payload["graph_id"]
            expected_graph_id = line_number - 1
            if type(graph_id) is not int or graph_id != expected_graph_id:
                raise BundleContractError(
                    "graph_id must equal its zero-based JSONL insertion index: "
                    f"line={line_number} expected={expected_graph_id} observed={graph_id!r}"
                )
            graph_ids.append(graph_id)
            nodes = payload["nodes"]
            edges = payload["edges"]
            if type(nodes) is not list or not nodes or type(edges) is not list:
                raise BundleContractError(f"invalid nodes/edges at line {line_number}")
            node_ids: list[int] = []
            for node in nodes:
                if type(node) is not dict or set(node) != {"id", "label"}:
                    raise BundleContractError(f"invalid node schema at line {line_number}")
                if (
                    type(node["id"]) is not int
                    or node["id"] < 0
                    or type(node["label"]) is not int
                ):
                    raise BundleContractError(f"invalid node value at line {line_number}")
                node_ids.append(node["id"])
            if node_ids != list(range(len(node_ids))):
                raise BundleContractError(f"node order is not consecutive at line {line_number}")
            seen_edges: set[tuple[int, int]] = set()
            adjacency = {node_id: set() for node_id in node_ids}
            for edge in edges:
                if type(edge) is not dict or set(edge) != {"source", "target", "label"}:
                    raise BundleContractError(f"invalid edge schema at line {line_number}")
                left, right = edge["source"], edge["target"]
                if (
                    type(left) is not int
                    or type(right) is not int
                    or left not in node_ids
                    or right not in node_ids
                    or left == right
                    or type(edge["label"]) is not int
                ):
                    raise BundleContractError(f"invalid edge value at line {line_number}")
                identity = (min(left, right), max(left, right))
                if identity in seen_edges:
                    raise BundleContractError(f"duplicate edge at line {line_number}")
                seen_edges.add(identity)
                adjacency[left].add(right)
                adjacency[right].add(left)
            visited = {node_ids[0]}
            frontier = [node_ids[0]]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current].difference(visited):
                    visited.add(neighbor)
                    frontier.append(neighbor)
            if len(visited) != len(node_ids):
                raise BundleContractError(f"graph is disconnected at line {line_number}")
            rows += 1
    if rows != expected_graph_count:
        raise BundleContractError(f"graph count mismatch: expected {expected_graph_count}, got {rows}")
    return {
        "row_count": rows,
        "ordered_graph_ids_sha256": _canonical_sha256(graph_ids),
        "schema": "graph_id_nodes_edges_v1",
        "train_scope_proven_by_source_manifest_and_authorization": True,
        "non_train_payload_rows": 0,
    }


def _validate_no_loaded_heldout(payload: Any, *, role: str) -> None:
    if not isinstance(payload, Mapping):
        raise BundleContractError(f"{role} must be a JSON object")
    for key, value in payload.items():
        lower = str(key).lower()
        if any(token in lower for token in ("calibration_loaded", "test_loaded")) and value is not False:
            raise BundleContractError(f"{role} does not prove {key}=false")
        if isinstance(value, (Mapping, list)):
            _validate_no_loaded_heldout_nested(value, role=role)


def _validate_no_loaded_heldout_nested(payload: Any, *, role: str) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            lower = str(key).lower()
            if any(token in lower for token in ("calibration_loaded", "test_loaded")) and value is not False:
                raise BundleContractError(f"{role} does not prove {key}=false")
            _validate_no_loaded_heldout_nested(value, role=role)
    elif isinstance(payload, list):
        for value in payload:
            _validate_no_loaded_heldout_nested(value, role=role)


def _add_deterministic_tar_member(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o444
    with path.open("rb") as handle:
        tar.addfile(info, handle)


def _build_tar(staging: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="x", format=tarfile.PAX_FORMAT) as archive:
        for relative in sorted(path.relative_to(staging) for path in staging.rglob("*") if path.is_file()):
            _add_deterministic_tar_member(archive, staging / relative, relative.as_posix())
    with destination.open("rb") as handle:
        os.fsync(handle.fileno())


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    sources_raw = _parse_role_values(args.source, option="--source")
    expected_hashes = _parse_role_values(args.expected_sha256, option="--expected-sha256")
    targets = tuple(sorted(set(args.target)))
    if targets != EXPECTED_TARGETS or len(args.target) != len(EXPECTED_TARGETS):
        raise BundleContractError(f"targets must be exactly {list(EXPECTED_TARGETS)} once each")
    if args.route_kind != ROUTE_KIND:
        raise BundleContractError(f"route kind must be {ROUTE_KIND}")
    source_commit = _require_hex_commit(args.source_commit, field="--source-commit")
    official_commit = _require_hex_commit(
        args.official_globalgce_commit, field="--official-globalgce-commit"
    )
    cohort_sha = _require_hex_sha256(args.source_cohort_sha256, field="--source-cohort-sha256")
    production_fingerprint = _require_hex_sha256(
        args.production_fingerprint, field="--production-fingerprint"
    )
    native_train_csv_sha = _require_hex_sha256(
        args.native_train_csv_sha256, field="--native-train-csv-sha256"
    )

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise BundleContractError(f"fresh output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    allowed_roots = tuple(root.resolve(strict=True) for root in args.allowed_source_root)
    if not allowed_roots:
        raise BundleContractError("at least one --allowed-source-root is required")

    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        payload_root = temporary_root / "payload"
        copies: list[FrozenCopy] = []
        for role in sorted(ROLE_TO_NAME):
            copies.append(
                _copy_frozen_regular_file(
                    role=role,
                    source=Path(sources_raw[role]),
                    destination=payload_root / ROLE_TO_NAME[role],
                    expected_sha256=expected_hashes[role],
                    allowed_roots=allowed_roots,
                    stability_window_seconds=args.stability_window_seconds,
                )
            )

        graph_audit = _validate_graph_jsonl(
            payload_root / ROLE_TO_NAME["graph_jsonl"], expected_graph_count=args.graph_count
        )
        source_manifest = _read_json(
            payload_root / ROLE_TO_NAME["source_input_manifest"], role="source_input_manifest"
        )
        split_manifest = _read_json(
            payload_root / ROLE_TO_NAME["split_manifest"], role="split_manifest"
        )
        train_cohort_manifest = _read_json(
            payload_root / ROLE_TO_NAME["train_cohort_manifest"],
            role="train_cohort_manifest",
        )
        authorization = _read_json(
            payload_root / ROLE_TO_NAME["data_use_authorization"], role="data_use_authorization"
        )
        _validate_no_loaded_heldout(source_manifest, role="source_input_manifest")
        _validate_no_loaded_heldout(split_manifest, role="split_manifest")
        _validate_no_loaded_heldout(
            train_cohort_manifest, role="train_cohort_manifest"
        )
        _validate_no_loaded_heldout(authorization, role="data_use_authorization")
        authorization_without_hash = dict(authorization)
        declared_authorization_sha = authorization_without_hash.pop("authorization_sha256", None)
        if not isinstance(declared_authorization_sha, str):
            raise BundleContractError("data-use authorization is missing authorization_sha256")
        if _canonical_sha256(authorization_without_hash) != declared_authorization_sha.lower():
            raise BundleContractError("data-use authorization self-hash changed")
        if authorization.get("allow_hpc_train_only_derived_t8_input_transfer") is not True:
            raise BundleContractError("data-use authorization does not allow this narrow HPC transfer")
        if authorization.get("calibration_payload_included") is not False:
            raise BundleContractError("authorization must declare calibration_payload_included=false")
        if authorization.get("test_payload_included") is not False:
            raise BundleContractError("authorization must declare test_payload_included=false")
        if authorization.get("matrix_publication_allowed_from_hpc") is not False:
            raise BundleContractError("authorization must forbid HPC matrix publication")
        if authorization.get("no_redistribution") is not True:
            raise BundleContractError("authorization must declare no_redistribution=true")

        graph_sha = expected_hashes["graph_jsonl"].lower()
        if source_manifest.get("train_only") is not True:
            raise BundleContractError("source input manifest must declare train_only=true")
        if source_manifest.get("calibration_loaded") is not False or source_manifest.get("test_loaded") is not False:
            raise BundleContractError("source input manifest must prove calibration/test were not loaded")
        if str(source_manifest.get("graph_jsonl_sha256", "")).lower() != graph_sha:
            raise BundleContractError("source input manifest graph SHA-256 mismatch")
        if source_manifest.get("gspan_graph_count") != args.graph_count:
            raise BundleContractError("source input manifest graph count mismatch")
        if source_manifest.get("selected_parent_count") != args.selected_parent_count:
            raise BundleContractError("source input manifest selected-parent count mismatch")
        if source_manifest.get("seed") != 7 or source_manifest.get("source_label") != 1:
            raise BundleContractError("source input manifest seed/source label mismatch")
        if str(source_manifest.get("selected_parent_cohort_sha256", "")).lower() != cohort_sha:
            raise BundleContractError("source input manifest cohort SHA-256 mismatch")
        if source_manifest.get("target_label") != 0:
            raise BundleContractError("shared transaction source must be the audited target-0 export")
        if str(source_manifest.get("native_train_csv_sha256", "")).lower() != native_train_csv_sha:
            raise BundleContractError("source input manifest train CSV SHA-256 mismatch")
        if authorization.get("route_kind") != ROUTE_KIND:
            raise BundleContractError("authorization route_kind mismatch")
        if (
            authorization.get("dataset") != "tastemolnet"
            or str(authorization.get("method", "")).lower() != "globalgce"
            or authorization.get("split_scope") != "train_only"
        ):
            raise BundleContractError("authorization must bind tastemolnet/train_only")
        if authorization.get("allowed_targets") != list(EXPECTED_TARGETS):
            raise BundleContractError("authorization must bind allowed_targets=[0,2]")
        if authorization.get("source_label") != 1:
            raise BundleContractError("authorization must bind source_label=1")
        if authorization.get("shared_transaction_database_authorized") is not True:
            raise BundleContractError("authorization must explicitly allow the shared transaction database")
        if authorization.get("target_label_affects_graph_export") is not False:
            raise BundleContractError("authorization must prove target label does not affect graph export")
        if str(authorization.get("source_graph_jsonl_sha256", "")).lower() != graph_sha:
            raise BundleContractError("authorization source graph SHA-256 mismatch")
        if authorization.get("source_graph_count") != args.graph_count:
            raise BundleContractError("authorization source graph count mismatch")
        if str(authorization.get("native_train_csv_sha256", "")).lower() != native_train_csv_sha:
            raise BundleContractError("authorization train CSV SHA-256 mismatch")
        for field in ("model_weights_included", "active_sqlite_or_wal_included", "checkpoint_tmp_included"):
            if authorization.get(field) is not False:
                raise BundleContractError(f"authorization must declare {field}=false")
        if (
            train_cohort_manifest.get("train_only") is not True
            or train_cohort_manifest.get("calibration_loaded") is not False
            or train_cohort_manifest.get("test_loaded") is not False
            or train_cohort_manifest.get("selected_count") != args.selected_parent_count
            or str(train_cohort_manifest.get("selected_cohort_sha256", "")).lower()
            != cohort_sha
        ):
            raise BundleContractError("train cohort manifest binding mismatch")
        split_train = (
            split_manifest.get("files", {}).get("train", {})
            if isinstance(split_manifest.get("files"), Mapping)
            else {}
        )
        if (
            split_manifest.get("dataset") != "tastemolnet"
            or split_manifest.get("calibration_loaded_for_training") is not False
            or split_manifest.get("test_loaded_for_training") is not False
            or split_manifest.get("test_evaluated_during_training") is not False
            or split_manifest.get("test_used_for_checkpoint_selection") is not False
            or str(split_train.get("sha256", "")).lower() != native_train_csv_sha
        ):
            raise BundleContractError("split manifest train-only provenance mismatch")
        supplied_science_config = {
            "source_label": args.source_label,
            "seed": args.seed,
            "epochs": args.epochs,
            "min_support": args.min_support,
            "min_vertices": args.min_vertices,
            "max_vertices": args.max_vertices,
            "top_k": args.top_k,
            "root_count": args.root_count,
        }
        if supplied_science_config != EXPECTED_SCIENCE_CONFIG:
            raise BundleContractError(
                "T8/T13-grade science config mismatch: "
                f"expected={EXPECTED_SCIENCE_CONFIG}, got={supplied_science_config}"
            )
        config = {
            **supplied_science_config,
            "min_support": args.min_support,
            "min_vertices": args.min_vertices,
            "max_vertices": args.max_vertices,
            "top_k": args.top_k,
            "root_count": args.root_count,
            "exact": True,
            "approximate_pruning": False,
            "production_fingerprint": production_fingerprint,
        }
        transaction_binding = {
            "shared_transaction_database": True,
            "target_labels": list(targets),
            "target_to_graph_jsonl_sha256": {str(target): graph_sha for target in targets},
            "target_semantics_do_not_modify_transaction_database": True,
            "graph_jsonl_sha256": graph_sha,
            "graph_count": args.graph_count,
            "selected_parent_count": args.selected_parent_count,
            "source_cohort_sha256": cohort_sha,
        }
        manifest: dict[str, Any] = {
            "bundle_version": BUNDLE_VERSION,
            "state": "PASS",
            "dataset": "tastemolnet",
            "method": "globalgce",
            "stage": "EXACT_GSPAN_CPU_INPUT",
            "route_kind": ROUTE_KIND,
            "split_scope": "train_only",
            "calibration_payload_included": False,
            "test_payload_included": False,
            "model_weights_included": False,
            "active_database_or_checkpoint_included": False,
            "matrix_publication_allowed_from_hpc": False,
            "source_commit": source_commit,
            "official_globalgce_commit": official_commit,
            "hpc_runtime_config": {
                "name": args.config.name,
                "sha256": _sha256_file(args.config.resolve(strict=True)),
            },
            "files": [asdict(item) for item in copies],
            "file_role_allowlist": sorted(ROLE_TO_NAME),
            "graph_payload_audit": graph_audit,
            "native_train_csv_provenance": {
                "payload_included": False,
                "sha256": native_train_csv_sha,
                "source_input_manifest_bound": True,
                "authorization_bound": True,
                "split_manifest_sha256": expected_hashes["split_manifest"].lower(),
            },
            "mining_config": config,
            "mining_config_sha256": _canonical_sha256(config),
            "transaction_binding": transaction_binding,
            "transfer_policy": {
                "direction": "AUTODL_TO_MAC_TO_HPC",
                "purpose": "T8_T13_GLOBALGCE_EXACT_CPU_MINING_ONLY",
                "no_redistribution": True,
                "source_data_is_train_only_derived": True,
                "hpc_may_modify_autodl_matrix": False,
            },
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        manifest_path = temporary_root / "input_bundle_manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest_file_sha = _sha256_file(manifest_path)
        _atomic_bytes(
            temporary_root / "input_bundle_manifest.json.sha256",
            f"{manifest_file_sha}  input_bundle_manifest.json\n".encode("ascii"),
        )

        tar_path = temporary_root / "t8_hpc_input_bundle.tar"
        # The tar cannot contain itself; include only payload and its closed manifest.
        tar_sources = temporary_root / ".tar-sources"
        tar_sources.mkdir()
        for name in ("payload", "input_bundle_manifest.json", "input_bundle_manifest.json.sha256"):
            source = temporary_root / name
            target = tar_sources / name
            if source.is_dir():
                shutil.copytree(source, target, copy_function=shutil.copy2)
            else:
                shutil.copy2(source, target)
        _build_tar(tar_sources, tar_path)
        shutil.rmtree(tar_sources)
        tar_sha = _sha256_file(tar_path)
        _atomic_bytes(
            temporary_root / "t8_hpc_input_bundle.tar.sha256",
            f"{tar_sha}  t8_hpc_input_bundle.tar\n".encode("ascii"),
        )
        completion = {
            "state": "PASS",
            "manifest_path": "input_bundle_manifest.json",
            "manifest_file_sha256": manifest_file_sha,
            "bundle_path": "t8_hpc_input_bundle.tar",
            "bundle_sha256": tar_sha,
            "bundle_size_bytes": tar_path.stat().st_size,
        }
        completion["receipt_sha256"] = _canonical_sha256(completion)
        _atomic_json(temporary_root / "build_receipt.json", completion)
        os.replace(temporary_root, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return completion
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allowed-source-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help=f"repeat exactly once for each role: {','.join(sorted(ROLE_TO_NAME))}",
    )
    parser.add_argument(
        "--expected-sha256",
        action="append",
        required=True,
        metavar="ROLE=SHA256",
    )
    parser.add_argument("--route-kind", required=True, choices=[ROUTE_KIND])
    parser.add_argument("--target", action="append", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--official-globalgce-commit", required=True)
    parser.add_argument("--graph-count", type=int, required=True)
    parser.add_argument("--selected-parent-count", type=int, required=True)
    parser.add_argument("--source-cohort-sha256", required=True)
    parser.add_argument("--production-fingerprint", required=True)
    parser.add_argument("--native-train-csv-sha256", required=True)
    parser.add_argument("--source-label", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--min-support", type=int, required=True)
    parser.add_argument("--min-vertices", type=int, required=True)
    parser.add_argument("--max-vertices", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--root-count", type=int, required=True)
    parser.add_argument("--stability-window-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.graph_count < 1 or args.selected_parent_count < args.graph_count:
        parser.error("graph counts must be positive and selected-parent-count >= graph-count")
    if any(value < 1 for value in (args.min_support, args.min_vertices, args.max_vertices, args.top_k)):
        parser.error("mining integer parameters must be positive")
    if args.min_vertices > args.max_vertices:
        parser.error("min-vertices cannot exceed max-vertices")
    if args.root_count < 1:
        parser.error("root-count must be positive")
    if args.stability_window_seconds < 0:
        parser.error("stability window cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_bundle(args)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
