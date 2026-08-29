"""Immutable TasteMolNet T3 adoption and adaptive T4 oracle-smoke science.

T3 validates and adopts the temperature already fitted on T2 validation
logits.  It never invokes a fitter and never copies or mutates the classifier
bundle.  T4 opens only the frozen graph-cache manifest and ``calibration.pt``;
train, validation, test, and every CSV payload remain unopened.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.chem.hard_deletion import (
    HardDeletionOutcome,
    apply_hard_deletion_match,
    enumerate_connected_hard_deletions,
)
from src.data.molecular_graph_dataset import (
    MolecularGraphData,
    MolecularGraphDataset,
    load_molecular_graph_cache,
)
from src.data.molecular_graph_featurizer import (
    MolecularFeatureSchema,
    MolecularGraphFeaturizer,
)
from src.eval.counterfactual_semantics import (
    CounterfactualRecord,
    compute_counterfactual_semantics,
    destination_distribution,
)
from src.models.molecular_gnn import MolecularGNN, MolecularGNNConfig
from src.oracles.gnn_oracle import (
    CHECKPOINT_BUNDLE_VERSION,
    GNNOracle,
    REQUIRED_CHECKPOINT_FILES,
    TASTE_REQUIRED_CHECKPOINT_FILES,
    classification_metrics,
)
from src.utils.tastemolnet_downstream_policy import (
    EXECUTION_SOURCE_ROOT,
    TasteDownstreamPolicy,
    load_tastemolnet_downstream_policy,
)
from src.utils.tastemolnet_gine_pass_adoption_v1 import (
    ADOPTION_MARKER,
    DOWNSTREAM_BINDING_KEYS,
    DOWNSTREAM_BINDING_SCHEMA,
    HeldT2PassAdoption,
    T2PassAdoptionError,
    hold_t2_gine_pass_adoption,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required by the AutoDL runtime gate.
    Chem = None


T3_STAGE = "T3_GINE_CALIBRATED"
T4_STAGE = "T4_ORACLE_SMOKE"
T3_MARKER = "TASTE_GINE_CALIBRATION_PASS"
T4_MARKER = "TASTE_MULTICLASS_ORACLE_PASS"
LABEL_MAP = {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}
INT_LABEL_MAP = {0: "Bitter", 1: "Sweet", 2: "Tasteless"}
NUM_CLASSES = 3
SOURCE_LABEL = 1
T4_ADAPTIVE_SEARCH_SCHEDULE = ((16, 8), (64, 16), (128, 32))
T4_MIN_STRICT_FLIPS = 16
T4_MIN_FLIPPED_PARENTS = 8
_HEX = frozenset("0123456789abcdef")
T4_CHECKPOINT_PAYLOAD_FILES = (
    "model.pt",
    "config.yaml",
    "model_card.json",
    "feature_schema.json",
    "label_map.json",
    "split_manifest.json",
    "test_evaluation_status.json",
    "temperature_scaling.json",
    "data_use_policy_binding.json",
    "graph_cache_usage.json",
    "oracle_manifest.json",
    "last.pt",
    "last_checkpoint.json",
    "checkpoint_reload.json",
)
T6_FROZEN_GINE_PAYLOAD_FILES = frozenset(
    {
        "model.pt",
        "config.yaml",
        "model_card.json",
        "feature_schema.json",
        "label_map.json",
        "split_manifest.json",
        "test_evaluation_status.json",
        "temperature_scaling.json",
    }
)
HELD_STAGE_EVIDENCE_KEYS = frozenset(
    {
        "stage",
        "gate_sha256",
        "root_inventory_sha256",
        "checkpoint_dir",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "t2_adoption_gate_sha256",
        "t2_adoption_receipt_sha256",
        "t2_adoption_binding_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_t2_adoption_binding_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TasteGNNStageError("fresh T2 adoption binding is not one mapping")
    if set(value) != DOWNSTREAM_BINDING_KEYS:
        raise TasteGNNStageError("fresh T2 adoption binding schema changed")
    if (
        value.get("schema_version") != DOWNSTREAM_BINDING_SCHEMA
        or value.get("stage") != "T2_GINE_FULL"
        or value.get("status") != "PASS"
        or value.get("state") != ADOPTION_MARKER
    ):
        raise TasteGNNStageError("fresh T2 adoption is not PASS")
    for key in (
        "adoption_root_inventory_sha256",
        "gate_sha256",
        "receipt_sha256",
        "source_evidence_sha256",
        "formal_bundle_inventory_sha256",
        "formal_bundle_model_sha256",
        "formal_bundle_sha256s_sha256",
    ):
        _hex(value.get(key), field=f"t2_adoption.{key}")
    for key in (
        "adoption_root",
        "gate_path",
        "receipt_path",
        "formal_bundle_root",
    ):
        item = value.get(key)
        if (
            type(item) is not str
            or not Path(item).is_absolute()
            or Path(os.path.abspath(item)) != Path(item)
        ):
            raise TasteGNNStageError(
                f"fresh T2 adoption {key} is not exact absolute"
            )
    if Path(value["gate_path"]) != Path(value["adoption_root"]) / "gate.json":
        raise TasteGNNStageError("fresh T2 adoption gate path changed")
    if Path(value["receipt_path"]) != Path(value["adoption_root"]) / "manifest.json":
        raise TasteGNNStageError("fresh T2 adoption receipt path changed")
    inventory = value.get("formal_bundle_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise TasteGNNStageError("fresh T2 formal inventory is absent")
    if _canonical_sha256(inventory) != value["formal_bundle_inventory_sha256"]:
        raise TasteGNNStageError("fresh T2 formal inventory digest changed")
    return json.loads(json.dumps(value))


def _validated_t2_adoption_binding(
    authority: HeldT2PassAdoption,
) -> dict[str, Any]:
    try:
        value = authority.revalidate()
    except (T2PassAdoptionError, OSError, ValueError) as exc:
        raise TasteGNNStageError(
            f"fresh T2 GINE PASS adoption authority failed: {exc}"
        ) from exc
    return _validate_t2_adoption_binding_value(value)


def _t2_binding_stage_evidence(gate: Mapping[str, Any]) -> dict[str, str]:
    binding = _validate_t2_adoption_binding_value(
        gate.get("t2_adoption_binding")
    )
    return {
        "t2_adoption_gate_sha256": str(binding["gate_sha256"]),
        "t2_adoption_receipt_sha256": str(binding["receipt_sha256"]),
        "t2_adoption_binding_sha256": _canonical_sha256(binding),
    }


def _formal_bundle_inventory(root: PhysicalDirectory) -> dict[str, Any]:
    """Rebuild the adoption inventory without reopening split-evidence payloads.

    The fresh receipt records both every file's physical identity and the
    already-verified bundle hash manifest.  Reconstructing from those two held
    authorities preserves T4's calibration-cache-only data-access boundary;
    T3 separately performs the full byte verification before publication.
    """

    root.verify(label="T2 formal bundle before adoption inventory")
    hashes = _checkpoint_sha_inventory(root)
    hashes["sha256sums.txt"] = _sha256_at(
        root, "sha256sums.txt", label="T2 adoption checkpoint hash inventory"
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root.descriptor)):
        if Path(name).name != name:
            raise TasteGNNStageError("T2 formal bundle contains an unsafe name")
        info = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise TasteGNNStageError(
                f"T2 formal bundle contains a non-file entry: {name}"
            )
        rows.append(
            {
                "path": name,
                "kind": "file",
                "identity": {
                    "device": int(info.st_dev),
                    "inode": int(info.st_ino),
                    "mode": int(info.st_mode),
                    "uid": int(info.st_uid),
                    "nlink": int(info.st_nlink),
                    "size": int(info.st_size),
                    "mtime_ns": int(info.st_mtime_ns),
                    "ctime_ns": int(info.st_ctime_ns),
                },
                "sha256": hashes.get(name),
            }
        )
    if any(type(row["sha256"]) is not str for row in rows):
        raise TasteGNNStageError("T2 adoption hash inventory does not close the bundle")
    rows.sort(key=lambda row: (str(row["path"]), str(row["kind"])))
    root.verify(label="T2 formal bundle after adoption inventory")
    return {"inventory": rows, "inventory_sha256": _canonical_sha256(rows)}


def _bind_t2_adoption_to_checkpoint(
    authority: HeldT2PassAdoption,
    checkpoint: PhysicalDirectory,
) -> dict[str, Any]:
    binding = _validated_t2_adoption_binding(authority)
    if str(checkpoint.path) != binding["formal_bundle_root"]:
        raise TasteGNNStageError(
            "checkpoint path differs from fresh T2 adoption formal bundle"
        )
    observed = _formal_bundle_inventory(checkpoint)
    if (
        observed["inventory"] != binding["formal_bundle_inventory"]
        or observed["inventory_sha256"]
        != binding["formal_bundle_inventory_sha256"]
        or _sha256_at(checkpoint, "model.pt", label="T2 adopted model.pt")
        != binding["formal_bundle_model_sha256"]
        or _sha256_at(
            checkpoint,
            "sha256sums.txt",
            label="T2 adopted checkpoint hash inventory",
        )
        != binding["formal_bundle_sha256s_sha256"]
    ):
        raise TasteGNNStageError(
            "checkpoint physical inventory differs from fresh T2 adoption receipt"
        )
    return binding


class TasteGNNStageError(RuntimeError):
    """A Taste T3/T4 scientific, authority, or immutable-input gate failed."""


def _native_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise TasteGNNStageError(f"{field} must be one native JSON integer")
    return value


def _hex(value: Any, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _HEX for c in value):
        raise TasteGNNStageError(f"{field} must be one lowercase SHA-256")
    return value


def _finite(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TasteGNNStageError(f"{field} must be one finite JSON number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise TasteGNNStageError(f"{field} is non-finite or outside its domain")
    return result


def _require_gpu1_environment(*, gpu_uuid: str) -> None:
    if os.environ.get("AUTODL_PHYSICAL_GPU_INDEX") != "1":
        raise TasteGNNStageError("T4 process lacks physical GPU1 index binding")
    if os.environ.get("AUTODL_PHYSICAL_GPU_UUID") != gpu_uuid:
        raise TasteGNNStageError("T4 process GPU UUID differs from exp_run binding")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise TasteGNNStageError("T4 process CUDA visibility is not physical GPU1")


def _stat_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


@dataclass(slots=True)
class _HeldDirectory:
    path: Path
    name: str | None
    descriptor: int
    evidence: dict[str, int]


@dataclass(slots=True)
class PhysicalDirectory:
    path: Path
    directories: list[_HeldDirectory]

    @property
    def descriptor(self) -> int:
        return self.directories[-1].descriptor

    @property
    def evidence(self) -> Mapping[str, int]:
        return self.directories[-1].evidence

    def identity_paths(self) -> dict[tuple[int, int], set[Path]]:
        result: dict[tuple[int, int], set[Path]] = {}
        for directory in self.directories:
            identity = (directory.evidence["device"], directory.evidence["inode"])
            result.setdefault(identity, set()).add(directory.path)
        return result

    def verify(self, *, label: str) -> None:
        try:
            for index, directory in enumerate(self.directories):
                held = os.fstat(directory.descriptor)
                current_identity = _stat_identity(held)
                keys = (
                    tuple(directory.evidence)
                    if index >= max(0, len(self.directories) - 2)
                    else ("device", "inode", "mode")
                )
                if (
                    not stat.S_ISDIR(held.st_mode)
                    or any(
                        current_identity[key] != directory.evidence[key]
                        for key in keys
                    )
                ):
                    raise TasteGNNStageError(
                        f"{label} physical root identity drifted"
                    )
                if index:
                    parent = self.directories[index - 1]
                    named = os.stat(
                        str(directory.name),
                        dir_fd=parent.descriptor,
                        follow_symlinks=False,
                    )
                    if _stat_identity(named) != _stat_identity(held):
                        raise TasteGNNStageError(
                            f"{label} physical root identity drifted"
                        )
        except (FileNotFoundError, OSError) as exc:
            raise TasteGNNStageError(f"{label} disappeared") from exc

    def refresh_leaf_after_authorized_mutation(self, *, label: str) -> None:
        """Accept only leaf metadata drift after checking every retained ancestor."""

        if len(self.directories) > 1:
            original = self.directories.pop()
            try:
                self.verify(label=f"{label} ancestors")
            finally:
                self.directories.append(original)
        leaf = self.directories[-1]
        held = os.fstat(leaf.descriptor)
        if not stat.S_ISDIR(held.st_mode):
            raise TasteGNNStageError(f"{label} leaf is no longer a directory")
        if len(self.directories) > 1:
            parent = self.directories[-2]
            named = os.stat(
                str(leaf.name),
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
            expected_core = (
                leaf.evidence["device"],
                leaf.evidence["inode"],
                leaf.evidence["mode"],
            )
            if (
                (int(held.st_dev), int(held.st_ino), int(held.st_mode))
                != expected_core
                or _stat_identity(named) != _stat_identity(held)
            ):
                raise TasteGNNStageError(f"{label} leaf identity drifted")
        leaf.evidence = _stat_identity(held)
        self.verify(label=label)

    def open_child_directory(
        self,
        name: str,
        *,
        label: str,
        create: bool,
        require_fresh: bool = False,
        forbidden_identity_paths: Mapping[tuple[int, int], set[Path]] | None = None,
    ) -> None:
        child = _safe_child_name(name, label=label)
        self.verify(label=f"{label} parent before open")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        created = False
        if create:
            try:
                os.mkdir(child, mode=0o700, dir_fd=self.descriptor)
                created = True
            except FileExistsError:
                if require_fresh:
                    raise FileExistsError(f"{label} must be fresh and absent")
            if created:
                self.refresh_leaf_after_authorized_mutation(
                    label=f"{label} parent after mkdir"
                )
        descriptor = os.open(child, flags, dir_fd=self.descriptor)
        try:
            held = os.fstat(descriptor)
            named = os.stat(child, dir_fd=self.descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(held.st_mode)
                or _stat_identity(held) != _stat_identity(named)
            ):
                raise TasteGNNStageError(f"{label} is not one physical directory")
            path = self.path / child
            identity = (int(held.st_dev), int(held.st_ino))
            aliases = (forbidden_identity_paths or {}).get(identity, set())
            if any(alias != path for alias in aliases):
                raise TasteGNNStageError(
                    f"{label} aliases a protected physical directory"
                )
            self.directories.append(
                _HeldDirectory(
                    path=path,
                    name=child,
                    descriptor=descriptor,
                    evidence=_stat_identity(held),
                )
            )
            self.path = path
            descriptor = -1
            self.verify(label=label)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def close(self) -> None:
        for directory in reversed(self.directories):
            if directory.descriptor >= 0:
                try:
                    os.close(directory.descriptor)
                except OSError:
                    pass
                directory.descriptor = -1

    def __del__(self) -> None:  # pragma: no cover - process-exit failure cleanup.
        self.close()


def _physical_directory(
    path: str | Path,
    *,
    field: str,
    mutable_contents: bool = False,
) -> PhysicalDirectory:
    del mutable_contents  # writes explicitly refresh only the held leaf metadata.
    root = Path(path).expanduser()
    if not root.is_absolute():
        raise TasteGNNStageError(f"{field} must be absolute")
    root = Path(os.path.abspath(root))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directories: list[_HeldDirectory] = []
    pending_descriptor = -1
    try:
        pending_descriptor = os.open(root.anchor, flags)
        info = os.fstat(pending_descriptor)
        directories.append(
            _HeldDirectory(
                path=Path(root.anchor),
                name=None,
                descriptor=pending_descriptor,
                evidence=_stat_identity(info),
            )
        )
        pending_descriptor = -1
        current = Path(root.anchor)
        for part in root.parts[1:]:
            pending_descriptor = os.open(
                part, flags, dir_fd=directories[-1].descriptor
            )
            held = os.fstat(pending_descriptor)
            named = os.stat(
                part,
                dir_fd=directories[-1].descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(held.st_mode)
                or _stat_identity(held) != _stat_identity(named)
            ):
                raise TasteGNNStageError(f"{field} must be one physical directory")
            current = current / part
            directories.append(
                _HeldDirectory(
                    path=current,
                    name=part,
                    descriptor=pending_descriptor,
                    evidence=_stat_identity(held),
                )
            )
            pending_descriptor = -1
        physical = PhysicalDirectory(path=root, directories=directories)
        physical.verify(label=field)
        return physical
    except Exception:
        if pending_descriptor >= 0:
            try:
                os.close(pending_descriptor)
            except OSError:
                pass
        for directory in reversed(directories):
            try:
                os.close(directory.descriptor)
            except OSError:
                pass
        raise


def _safe_child_name(name: str, *, label: str) -> str:
    if type(name) is not str or not name or Path(name).name != name:
        raise TasteGNNStageError(f"{label} has an unsafe child name")
    return name


def _read_bytes_at(
    directory: PhysicalDirectory,
    name: str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    child = _safe_child_name(name, label=label)
    directory.verify(label=f"{label} parent before open")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(child, flags, dir_fd=directory.descriptor)
    except (FileNotFoundError, OSError) as exc:
        raise TasteGNNStageError(f"{label} is absent or cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteGNNStageError(f"{label} must be one regular single-link file")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(child, dir_fd=directory.descriptor, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named)
        ):
            raise TasteGNNStageError(f"{label} changed while read")
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise TasteGNNStageError(f"{label} SHA-256 differs from authority")
        data = b"".join(chunks)
        evidence = {
            "path": str(directory.path / child),
            "sha256": observed,
            **_stat_identity(after),
        }
    finally:
        os.close(descriptor)
    directory.verify(label=f"{label} parent after read")
    return data, evidence


def _read_text_at(
    directory: PhysicalDirectory,
    name: str,
    *,
    label: str,
) -> str:
    data, _evidence = _read_bytes_at(directory, name, label=label)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TasteGNNStageError(f"{label} is not UTF-8") from exc


def _read_json_at(
    directory: PhysicalDirectory,
    name: str,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(_read_text_at(directory, name, label=label))
    except json.JSONDecodeError as exc:
        raise TasteGNNStageError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise TasteGNNStageError(f"{label} must be one JSON object")
    return value


def _sha256_at(directory: PhysicalDirectory, name: str, *, label: str) -> str:
    _data, evidence = _read_bytes_at(directory, name, label=label)
    return str(evidence["sha256"])


def _write_bytes_at(
    directory: PhysicalDirectory,
    name: str,
    data: bytes,
    *,
    label: str,
) -> dict[str, int]:
    child = _safe_child_name(name, label=label)
    directory.verify(label=f"{label} parent before write")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(child, flags, 0o600, dir_fd=directory.descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TasteGNNStageError(f"{label} is not one regular single-link file")
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named = os.stat(child, dir_fd=directory.descriptor, follow_symlinks=False)
        if _stat_identity(after) != _stat_identity(named):
            raise TasteGNNStageError(f"{label} changed while publishing")
        evidence = _stat_identity(after)
    finally:
        os.close(descriptor)
    directory.refresh_leaf_after_authorized_mutation(
        label=f"{label} parent after write"
    )
    return evidence


def _checkpoint_snapshot(root: PhysicalDirectory) -> dict[str, Any]:
    root.verify(label="immutable checkpoint before full snapshot")
    root_info = os.fstat(root.descriptor)
    if not stat.S_ISDIR(root_info.st_mode):
        raise TasteGNNStageError("immutable checkpoint root is not a physical directory")
    files: dict[str, Any] = {}
    for name in sorted(os.listdir(root.descriptor)):
        _data, evidence = _read_bytes_at(root, name, label=f"checkpoint {name}")
        files[name] = {
            "sha256": evidence["sha256"],
            **{
                key: evidence[key]
                for key in _stat_identity(os.fstat(root.descriptor))
            },
        }
    root.verify(label="immutable checkpoint after full snapshot")
    return {
        "root_identity": _stat_identity(root_info),
        "files": files,
        "inventory_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _checkpoint_stat_snapshot(root: PhysicalDirectory) -> dict[str, Any]:
    """Inventory checkpoint identities without opening any checkpoint payload."""

    root.verify(label="immutable checkpoint before stat snapshot")
    root_info = os.fstat(root.descriptor)
    if not stat.S_ISDIR(root_info.st_mode):
        raise TasteGNNStageError("immutable checkpoint root is not a physical directory")
    files: dict[str, Any] = {}
    for name in sorted(os.listdir(root.descriptor)):
        info = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TasteGNNStageError(
                f"immutable checkpoint contains non-regular or linked entry: {name}"
            )
        files[name] = _stat_identity(info)
    payload = {"root_identity": _stat_identity(root_info), "files": files}
    result = {
        **payload,
        "inventory_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    root.verify(label="immutable checkpoint after stat snapshot")
    return result


def _checkpoint_sha_inventory(root: PhysicalDirectory) -> dict[str, str]:
    """Read the bundle's hash manifest without opening split-evidence payloads."""

    result: dict[str, str] = {}
    for line in _read_text_at(
        root, "sha256sums.txt", label="checkpoint SHA inventory"
    ).splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or Path(name).name != name
            or name in result
        ):
            raise TasteGNNStageError("checkpoint SHA inventory is malformed")
        result[name] = _hex(digest, field=f"checkpoint.sha256s.{name}")
    actual = {name for name in os.listdir(root.descriptor) if name != "sha256sums.txt"}
    if set(result) != actual:
        raise TasteGNNStageError("checkpoint SHA inventory does not close its root")
    return result


def _checkpoint_manifest_snapshot(root: PhysicalDirectory) -> dict[str, Any]:
    """Rebuild the full inventory digest without opening split CSV payloads."""

    hashes = _checkpoint_sha_inventory(root)
    hashes["sha256sums.txt"] = _sha256_at(
        root, "sha256sums.txt", label="checkpoint SHA inventory"
    )
    stats = _checkpoint_stat_snapshot(root)
    if set(hashes) != set(stats["files"]):
        raise TasteGNNStageError("checkpoint hash/stat inventories differ")
    files = {
        name: {"sha256": hashes[name], **stats["files"][name]}
        for name in sorted(hashes)
    }
    return {
        "root_identity": stats["root_identity"],
        "files": files,
        "inventory_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "stat_inventory_sha256": stats["inventory_sha256"],
    }


def _selected_file_snapshot(
    root: PhysicalDirectory,
    names: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    root.verify(label=f"{label} before snapshot")
    files: dict[str, Any] = {}
    for name in sorted(names):
        _data, evidence = _read_bytes_at(root, name, label=f"{label} {name}")
        files[name] = {
            key: evidence[key]
            for key in (
                "sha256",
                "device",
                "inode",
                "mode",
                "nlink",
                "size",
                "mtime_ns",
                "ctime_ns",
            )
        }
    root.verify(label=f"{label} after snapshot")
    return {
        "root_identity": dict(root.evidence),
        "files": files,
        "inventory_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _complete_directory_snapshot(
    root: PhysicalDirectory,
    *,
    label: str,
) -> dict[str, Any]:
    names = sorted(os.listdir(root.descriptor))
    return _selected_file_snapshot(root, names, label=label)


def _paths_related(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _merge_identity_paths(
    *values: Mapping[tuple[int, int], set[Path]],
) -> dict[tuple[int, int], set[Path]]:
    result: dict[tuple[int, int], set[Path]] = {}
    for value in values:
        for identity, paths in value.items():
            result.setdefault(identity, set()).update(paths)
    return result


@dataclass(slots=True)
class StageOutputPlan:
    target: Path
    basename: str
    directory: PhysicalDirectory
    remaining_parts: tuple[str, ...]
    forbidden_identity_paths: Mapping[tuple[int, int], set[Path]]
    created: bool = False

    def create(self) -> PhysicalDirectory:
        if self.created:
            raise TasteGNNStageError("stage output plan was already consumed")
        for index, part in enumerate(self.remaining_parts):
            self.directory.open_child_directory(
                part,
                label=(
                    "stage output"
                    if index == len(self.remaining_parts) - 1
                    else "stage output parent"
                ),
                create=True,
                require_fresh=index == len(self.remaining_parts) - 1,
                forbidden_identity_paths=self.forbidden_identity_paths,
            )
        if self.directory.path != self.target:
            raise TasteGNNStageError("stage output physical path differs from formula")
        self.created = True
        return self.directory

    def close(self) -> None:
        self.directory.close()

    def __del__(self) -> None:  # pragma: no cover - process-exit cleanup.
        self.close()


def _plan_stage_output(
    *,
    artifact_root: str | Path,
    output_dir: str | Path,
    basename_prefix: str,
    forbidden_paths: Sequence[Path],
    forbidden_identity_paths: Mapping[tuple[int, int], set[Path]],
) -> StageOutputPlan:
    artifact = Path(artifact_root).expanduser()
    output = Path(output_dir).expanduser()
    if not artifact.is_absolute() or not output.is_absolute():
        raise TasteGNNStageError("artifact root and stage output must be absolute")
    artifact = Path(os.path.abspath(artifact))
    output = Path(os.path.abspath(output))
    basename = output.name
    if (
        not basename.startswith(basename_prefix)
        or len(basename) == len(basename_prefix)
        or basename in {".", ".."}
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in basename
        )
    ):
        raise TasteGNNStageError(
            f"stage output basename must match {basename_prefix}*"
        )
    fixed_parent = artifact / "gnn_oracles" / "tastemolnet" / "gine" / "seed7"
    expected = fixed_parent / basename
    if output != expected or output.parent != fixed_parent:
        raise TasteGNNStageError(
            "stage output must be a direct child of the exact Taste GINE seed7 artifact root"
        )
    for protected in forbidden_paths:
        if _paths_related(expected, protected):
            raise TasteGNNStageError(
                f"stage output overlaps protected authority: {protected}"
            )
    held_artifact = _physical_directory(artifact, field="AutoDL artifact root")
    try:
        for identity, paths in held_artifact.identity_paths().items():
            aliases = forbidden_identity_paths.get(identity, set())
            if any(alias not in paths for alias in aliases):
                raise TasteGNNStageError(
                    "AutoDL artifact root aliases a protected physical authority"
                )
        return StageOutputPlan(
            target=expected,
            basename=basename,
            directory=held_artifact,
            remaining_parts=(
                "gnn_oracles",
                "tastemolnet",
                "gine",
                "seed7",
                basename,
            ),
            forbidden_identity_paths=forbidden_identity_paths,
        )
    except Exception:
        held_artifact.close()
        raise


def _verify_taste_checkpoint_bundle(root: PhysicalDirectory) -> dict[str, Any]:
    """Verify the full Taste T2 closure using only descriptor-relative reads."""

    snapshot = _checkpoint_snapshot(root)
    hashes = _checkpoint_sha_inventory(root)
    required = (
        set(REQUIRED_CHECKPOINT_FILES)
        | set(TASTE_REQUIRED_CHECKPOINT_FILES)
    ) - {"sha256sums.txt"}
    if not required.issubset(hashes):
        raise TasteGNNStageError(
            f"Taste checkpoint closure is missing: {sorted(required - set(hashes))}"
        )
    for name, expected in hashes.items():
        if snapshot["files"][name]["sha256"] != expected:
            raise TasteGNNStageError(f"Taste checkpoint SHA mismatch: {name}")

    model_card = _read_json_at(root, "model_card.json", label="model card")
    validate_taste_model_card(model_card)
    binding = _read_json_at(
        root, "data_use_policy_binding.json", label="T2 policy binding"
    )
    cache_usage = _read_json_at(
        root, "graph_cache_usage.json", label="T2 graph-cache usage"
    )
    oracle_manifest = _read_json_at(
        root, "oracle_manifest.json", label="T2 oracle manifest"
    )
    last_checkpoint = _read_json_at(
        root, "last_checkpoint.json", label="T2 last-checkpoint manifest"
    )
    checkpoint_reload = _read_json_at(
        root, "checkpoint_reload.json", label="T2 checkpoint reload"
    )
    if (
        binding.get("schema_version") != "tastemolnet_training_policy_binding_v1"
        or binding.get("dataset") != "tastemolnet"
        or binding.get("status") != "NOT_EXPLICITLY_STATED"
        or binding.get("authorization_status")
        != "RESEARCH_REPORTING_ALLOWED_NO_REDISTRIBUTION"
        or binding.get("paper_result_reporting_allowed") is not True
        or binding.get("dataset_redistributed") is not False
        or binding.get("data_redistribution_allowed") is not False
        or binding.get("upstream_license_not_explicit") is not True
        or binding.get("upstream_license_status") != "NOT_EXPLICITLY_STATED"
        or binding.get("upstream_license_claimed_resolved") is not False
        or binding.get("license_pass_claimed") is not False
        or binding.get("hpc_execution_authorized") is not False
    ):
        raise TasteGNNStageError("Taste T2 policy binding changed")
    binding_policy = binding.get("policy")
    binding_receipt = binding.get("policy_receipt")
    if (
        not isinstance(binding_policy, Mapping)
        or not isinstance(binding_receipt, Mapping)
        or binding_policy.get("policy_file_sha256")
        != model_card.get("data_use_policy_file_sha256")
        or binding_policy.get("policy_canonical_sha256")
        != model_card.get("data_use_policy_canonical_sha256")
        or binding_receipt.get("sha256")
        != model_card.get("data_use_policy_receipt_sha256")
    ):
        raise TasteGNNStageError("Taste T2 policy hashes conflict with model card")
    if (
        cache_usage.get("schema_version") != "tastemolnet_graph_cache_usage_v1"
        or cache_usage.get("dataset") != "tastemolnet"
        or cache_usage.get("mode") != "read_only_existing_cache"
        or cache_usage.get("graph_cache_used") is not True
        or cache_usage.get("loaded_splits") != ["train", "validation"]
        or cache_usage.get("calibration_loaded") is not False
        or cache_usage.get("test_loaded") is not False
        or cache_usage.get("graph_cache_rebuilt") is not False
        or cache_usage.get("data_reprepared") is not False
        or cache_usage.get("graph_cache_manifest_sha256")
        != model_card.get("graph_cache_manifest_sha256")
    ):
        raise TasteGNNStageError("Taste T2 graph-cache closure changed")
    if (
        oracle_manifest.get("schema_version")
        != "tastemolnet_three_class_gine_oracle_manifest_v1"
        or oracle_manifest.get("dataset") != "tastemolnet"
        or oracle_manifest.get("status") != "PASS"
        or oracle_manifest.get("checkpoint_id") != model_card.get("checkpoint_id")
        or oracle_manifest.get("oracle_backend") != "gnn"
        or oracle_manifest.get("classifier_family") != "gine"
        or oracle_manifest.get("rf_oracle_used") is not False
        or type(oracle_manifest.get("num_classes")) is not int
        or oracle_manifest.get("num_classes") != NUM_CLASSES
        or type(oracle_manifest.get("source_label")) is not int
        or oracle_manifest.get("source_label") != SOURCE_LABEL
        or oracle_manifest.get("test_loaded") is not False
        or oracle_manifest.get("test_evaluated") is not False
        or oracle_manifest.get("paper_result_reporting_allowed") is not True
        or oracle_manifest.get("dataset_redistributed") is not False
        or oracle_manifest.get("upstream_license_not_explicit") is not True
        or oracle_manifest.get("health_gate", {}).get("status") != "PASS"
    ):
        raise TasteGNNStageError("Taste T2 oracle manifest changed")
    completed_epoch = _native_int(
        last_checkpoint.get("completed_epoch"), field="last_checkpoint.completed_epoch"
    )
    if (
        last_checkpoint.get("schema_version")
        != "tastemolnet_last_training_checkpoint_v1"
        or last_checkpoint.get("checkpoint_file") != "last.pt"
        or last_checkpoint.get("same_bytes_as_latest_epoch_checkpoint") is not True
        or completed_epoch < 1
        or last_checkpoint.get("checkpoint_sha256")
        != snapshot["files"]["last.pt"]["sha256"]
        or last_checkpoint.get("source_checkpoint_sha256")
        != last_checkpoint.get("checkpoint_sha256")
    ):
        raise TasteGNNStageError("Taste T2 last-checkpoint closure changed")
    if (
        checkpoint_reload.get("schema_version")
        != "tastemolnet_gine_checkpoint_reload_v1"
        or checkpoint_reload.get("status") != "PASS"
        or checkpoint_reload.get("checkpoint_reload_pass") is not True
        or checkpoint_reload.get("batch_single_probability_equivalence") is not True
        or checkpoint_reload.get("all_probabilities_finite") is not True
        or type(checkpoint_reload.get("num_classes")) is not int
        or checkpoint_reload.get("num_classes") != NUM_CLASSES
        or type(checkpoint_reload.get("source_label")) is not int
        or checkpoint_reload.get("source_label") != SOURCE_LABEL
        or checkpoint_reload.get("checkpoint_id")
        != snapshot["files"]["model.pt"]["sha256"]
        or checkpoint_reload.get("last_checkpoint") != last_checkpoint
    ):
        raise TasteGNNStageError("Taste T2 checkpoint-reload closure changed")
    test_status = _read_json_at(
        root, "test_evaluation_status.json", label="test status"
    )
    if (
        test_status.get("status") != "NOT_EVALUATED"
        or test_status.get("test_loaded") is not False
        or not str(test_status.get("reason") or "").strip()
    ):
        raise TasteGNNStageError("Taste T2 held-out test status changed")
    test_path = str(test_status.get("path") or "").strip()
    test_sha256 = str(test_status.get("sha256") or "").strip().lower()
    _hex(test_sha256, field="test_status.sha256")
    if not test_path:
        raise TasteGNNStageError("Taste T2 test status lacks its private path")
    split_manifest = _read_json_at(
        root, "split_manifest.json", label="split manifest"
    )
    manifest_test = split_manifest.get("files", {}).get("test", {})
    if manifest_test and (
        str(manifest_test.get("path")) != test_path
        or str(manifest_test.get("sha256", "")).lower() != test_sha256
    ):
        raise TasteGNNStageError("Taste T2 test status conflicts with split manifest")
    serialized = json.dumps(
        [binding, cache_usage, oracle_manifest, last_checkpoint, checkpoint_reload, model_card],
        sort_keys=True,
    )
    if "TASTE_LICENSE_PASS" in serialized or "LICENSE_PASS" in serialized:
        raise TasteGNNStageError("Taste T2 bundle claims an upstream license PASS")
    return {"model_card": model_card, "snapshot": snapshot, "hashes": hashes}


def validate_taste_model_card(model_card: Mapping[str, Any]) -> None:
    exact = {
        "dataset": "tastemolnet",
        "backbone": "gine",
        "oracle_backend": "gnn",
        "classifier_type": "gnn",
        "rf_oracle_used": False,
        "profile": "full",
        "selection_split": "validation",
        "temperature_calibration_split": "validation",
        "temperature_calibration_fit_on_validation": True,
        "calibration_used_for_model_fit_or_selection": False,
        "test_used_for_model_fit_or_selection": False,
        "test_loaded_during_training": False,
        "test_evaluated_during_training": False,
        "graph_cache_used": True,
    }
    failures = [
        f"{key}={model_card.get(key)!r}"
        for key, expected in exact.items()
        if type(model_card.get(key)) is not type(expected)
        or model_card.get(key) != expected
    ]
    for key, expected in (("num_classes", NUM_CLASSES), ("source_label", SOURCE_LABEL)):
        if type(model_card.get(key)) is not int or model_card.get(key) != expected:
            failures.append(f"{key}={model_card.get(key)!r}")
    health = model_card.get("health_gate")
    if not isinstance(health, Mapping) or health.get("status") != "PASS":
        failures.append("health_gate.status!=PASS")
    if failures:
        raise TasteGNNStageError(
            "Taste three-class GINE model-card contract failed: " + ", ".join(failures)
        )


def _softmax(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    selected = probabilities[np.arange(labels.size), labels]
    return float(-np.log(np.clip(selected, 1e-300, 1.0)).mean())


def _validate_existing_temperature_fit_anchored(
    checkpoint: PhysicalDirectory,
) -> dict[str, Any]:
    """Recompute T2 validation calibration without fitting or split-file access."""

    temperature = _read_json_at(
        checkpoint, "temperature_scaling.json", label="temperature scaling"
    )
    required = {
        "schema_version": "temperature_scaling_v1",
        "status": "fit",
        "selection_split": "validation",
        "test_used_for_fit": False,
        "argmax_invariant": True,
    }
    failures = [
        f"{key}={temperature.get(key)!r}"
        for key, expected in required.items()
        if type(temperature.get(key)) is not type(expected)
        or temperature.get(key) != expected
    ]
    if failures:
        raise TasteGNNStageError("existing temperature fit is unsafe: " + ", ".join(failures))
    num_examples = _native_int(temperature.get("num_examples"), field="temperature.num_examples")
    if _native_int(temperature.get("num_classes"), field="temperature.num_classes") != NUM_CLASSES:
        raise TasteGNNStageError("temperature fit is not three-class")
    scalar = _finite(temperature.get("temperature"), field="temperature", positive=True)

    rows: list[dict[str, str]] = []
    with io.StringIO(
        _read_text_at(
            checkpoint,
            "validation_predictions.csv",
            label="validation predictions",
        ),
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "molecule_id", "smiles", "split", "label", "predicted_label",
            "logits", "probabilities", "source_graph_hash",
        }
        if not required_columns.issubset(reader.fieldnames or ()):
            raise TasteGNNStageError("validation prediction schema is incomplete")
        rows = [dict(row) for row in reader]
    if len(rows) != num_examples or not rows:
        raise TasteGNNStageError("validation prediction count differs from temperature fit")
    molecule_ids = [str(row["molecule_id"]) for row in rows]
    if any(not value for value in molecule_ids) or len(set(molecule_ids)) != len(rows):
        raise TasteGNNStageError("validation prediction molecule IDs are empty or duplicated")

    labels: list[int] = []
    logits: list[list[float]] = []
    stored_probabilities: list[list[float]] = []
    for index, row in enumerate(rows):
        if row["split"].strip().lower() not in {"val", "validation"}:
            raise TasteGNNStageError(f"validation row {index} has a non-validation split")
        try:
            label = int(row["label"])
            predicted = int(row["predicted_label"])
            row_logits = json.loads(row["logits"])
            row_probabilities = json.loads(row["probabilities"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TasteGNNStageError(f"validation row {index} is malformed") from exc
        if str(label) != row["label"].strip() or str(predicted) != row["predicted_label"].strip():
            raise TasteGNNStageError(f"validation row {index} has non-integral labels")
        if label not in range(NUM_CLASSES) or predicted not in range(NUM_CLASSES):
            raise TasteGNNStageError(f"validation row {index} class index is out of range")
        if not isinstance(row_logits, list) or not isinstance(row_probabilities, list):
            raise TasteGNNStageError(f"validation row {index} vectors are not lists")
        if len(row_logits) != NUM_CLASSES or len(row_probabilities) != NUM_CLASSES:
            raise TasteGNNStageError(f"validation row {index} vectors are not three-wide")
        parsed_logits = [
            _finite(value, field=f"validation[{index}].logits") for value in row_logits
        ]
        parsed_probabilities = [
            _finite(value, field=f"validation[{index}].probabilities")
            for value in row_probabilities
        ]
        if predicted != int(np.argmax(parsed_probabilities)):
            raise TasteGNNStageError(f"validation row {index} predicted_label changed")
        labels.append(label)
        logits.append(parsed_logits)
        stored_probabilities.append(parsed_probabilities)

    label_array = np.asarray(labels, dtype=np.int64)
    logits_array = np.asarray(logits, dtype=np.float64)
    raw_probabilities = _softmax(logits_array, temperature=1.0)
    if not np.allclose(
        raw_probabilities,
        np.asarray(stored_probabilities, dtype=np.float64),
        rtol=0.0,
        # T2 stores float32 torch.softmax values but serializes logits after a
        # float64 cast. Recomputing from those logits in float64 may differ by
        # one float32 rounding unit while remaining the exact same prediction.
        atol=1e-7,
    ):
        raise TasteGNNStageError("stored validation probabilities differ from raw logits")
    calibrated_probabilities = _softmax(logits_array, temperature=scalar)
    if not np.array_equal(
        raw_probabilities.argmax(axis=1), calibrated_probabilities.argmax(axis=1)
    ):
        raise TasteGNNStageError("existing positive temperature changed argmax")
    before = classification_metrics(label_array, raw_probabilities, num_classes=NUM_CLASSES)
    after = classification_metrics(
        label_array, calibrated_probabilities, num_classes=NUM_CLASSES
    )
    recomputed = {
        "nll_before": _nll(raw_probabilities, label_array),
        "nll_after": _nll(calibrated_probabilities, label_array),
        "ece_before": float(before["ece"]),
        "ece_after": float(after["ece"]),
        "brier_before": float(before["brier_score"]),
        "brier_after": float(after["brier_score"]),
    }
    for key, value in recomputed.items():
        recorded = _finite(temperature.get(key), field=f"temperature.{key}")
        if not math.isclose(recorded, value, rel_tol=1e-8, abs_tol=1e-10):
            raise TasteGNNStageError(f"recorded {key} differs from validation logits")

    split_manifest = _read_json_at(
        checkpoint, "split_manifest.json", label="split manifest"
    )
    validation_manifest = split_manifest.get("validation_manifest")
    if not isinstance(validation_manifest, Mapping):
        raise TasteGNNStageError("split manifest lacks validation manifest")
    if (
        _native_int(
            validation_manifest.get("num_records"), field="validation.num_records"
        )
        != num_examples
    ):
        raise TasteGNNStageError("validation manifest count differs from temperature fit")
    if (
        split_manifest.get("roles", {}).get("validation")
        != "checkpoint_selection_and_temperature_calibration"
        or split_manifest.get("calibration_loaded_for_training") is not False
        or split_manifest.get("test_loaded_for_training") is not False
        or split_manifest.get("test_evaluated_during_training") is not False
        or split_manifest.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteGNNStageError("split manifest violates train/validation-only fitting")
    return {
        "status": "PASS",
        "existing_fit_adopted": True,
        "temperature_refit_performed": False,
        "temperature": scalar,
        "num_examples": num_examples,
        "num_classes": NUM_CLASSES,
        "selection_split": "validation",
        "test_used_for_fit": False,
        "argmax_invariant": True,
        "recomputed_metrics": recomputed,
    }


def validate_existing_temperature_fit(
    checkpoint: str | Path | PhysicalDirectory,
) -> dict[str, Any]:
    """Public descriptor-anchored verifier for an existing validation fit."""

    if isinstance(checkpoint, PhysicalDirectory):
        return _validate_existing_temperature_fit_anchored(checkpoint)
    root = _physical_directory(checkpoint, field="T2 checkpoint")
    try:
        return _validate_existing_temperature_fit_anchored(root)
    finally:
        root.close()


def _stage_required_files(stage: Any) -> set[str] | None:
    required_by_stage = {
        T3_STAGE: {
            "calibration_adoption.json",
            "oracle_reference.json",
            "policy_binding.json",
            "gate.json",
            T3_MARKER,
        },
        T4_STAGE: {
            "oracle_smoke.json",
            "oracle_provenance.json",
            "data_access_manifest.json",
            "policy_binding.json",
            "gate.json",
            T4_MARKER,
        },
    }
    return required_by_stage.get(stage)


def _read_stage_hash_inventory(output: PhysicalDirectory) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in _read_text_at(
        output, "sha256sums.txt", label="stage SHA inventory"
    ).splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or Path(name).name != name or name in expected:
            raise TasteGNNStageError("stage SHA inventory is malformed")
        expected[name] = _hex(digest, field=f"sha256sums.{name}")
    return expected


def _stage_publication_authority(
    output: PhysicalDirectory,
    *,
    document_names: set[str],
) -> tuple[dict[str, Any], str]:
    files: dict[str, dict[str, int]] = {}
    for name in sorted(document_names | {"sha256sums.txt"}):
        info = os.stat(name, dir_fd=output.descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TasteGNNStageError(
                f"stage publication authority is not one regular file: {name}"
            )
        identity = _stat_identity(info)
        if name == "sha256sums.txt":
            identity = {
                key: identity[key]
                for key in ("device", "inode", "mode", "nlink", "size")
            }
        files[name] = identity
    payload = {
        "schema_version": "tastemolnet_stage_publication_authority_v1",
        "files": files,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload, digest


def _stage_marker_payload(marker: str, authority_sha256: str) -> bytes:
    return (marker + "\n" + _hex(
        authority_sha256,
        field="stage_publication_authority_sha256",
    ) + "\n").encode("utf-8")


@dataclass(slots=True)
class _HeldPreparedStageFile:
    name: str
    descriptor: int
    evidence: dict[str, int]
    sha256: str


@dataclass(slots=True)
class PreparedStageOutput:
    """Descriptor-retained non-terminal stage evidence."""

    output: PhysicalDirectory
    marker: str
    files: dict[str, _HeldPreparedStageFile]
    sha_identity: dict[str, int]
    marker_payload: bytes

    def _verify_files(self) -> None:
        self.output.verify(label="retained prepared stage output")
        actual = set(os.listdir(self.output.descriptor))
        if actual != set(self.files):
            raise TasteGNNStageError("retained prepared stage file set changed")
        for name, retained in self.files.items():
            held = os.fstat(retained.descriptor)
            named = os.stat(
                name,
                dir_fd=self.output.descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(held.st_mode)
                or held.st_nlink != 1
                or _stat_identity(held) != retained.evidence
                or _stat_identity(named) != retained.evidence
            ):
                raise TasteGNNStageError(
                    f"retained prepared stage file changed: {name}"
                )
            digest = hashlib.sha256()
            offset = 0
            while True:
                chunk = os.pread(retained.descriptor, 1024 * 1024, offset)
                if not chunk:
                    break
                digest.update(chunk)
                offset += len(chunk)
            if digest.hexdigest() != retained.sha256:
                raise TasteGNNStageError(
                    f"retained prepared stage bytes changed: {name}"
                )
        self.output.verify(label="retained prepared stage output after files")

    def revalidate(self) -> dict[str, Any]:
        self._verify_files()
        verified = _verify_prepared_stage_output(
            self.output,
            marker=self.marker,
            retained=None,
        )
        self._verify_files()
        return verified

    def close(self) -> None:
        for retained in self.files.values():
            if retained.descriptor >= 0:
                try:
                    os.close(retained.descriptor)
                except OSError:
                    pass
                retained.descriptor = -1

    def __enter__(self) -> PreparedStageOutput:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _verify_prepared_stage_output(
    output: PhysicalDirectory,
    *,
    marker: str,
    retained: PreparedStageOutput | None = None,
) -> dict[str, Any]:
    """Verify a complete stage whose sole missing file is its PASS marker."""

    if Path(marker).name != marker or marker in {"", "sha256sums.txt"}:
        raise TasteGNNStageError("prepared stage marker is unsafe")
    output.verify(label="prepared stage output")
    expected = _read_stage_hash_inventory(output)
    actual = {
        name for name in os.listdir(output.descriptor) if name != "sha256sums.txt"
    }
    if marker in actual:
        raise TasteGNNStageError("prepared stage already contains its PASS marker")
    if set(expected) != actual | {marker}:
        raise TasteGNNStageError(
            "prepared stage SHA inventory does not close the future output root"
        )
    _authority, authority_sha256 = _stage_publication_authority(
        output,
        document_names=actual,
    )
    marker_payload = _stage_marker_payload(marker, authority_sha256)
    expected_marker_sha256 = hashlib.sha256(marker_payload).hexdigest()
    if expected.get(marker) != expected_marker_sha256:
        raise TasteGNNStageError("prepared stage marker digest changed")
    for name in actual:
        if (
            _sha256_at(output, name, label=f"prepared stage output {name}")
            != expected[name]
        ):
            raise TasteGNNStageError(f"prepared stage output hash mismatch: {name}")
    gate = _read_json_at(output, "gate.json", label="prepared stage gate")
    if type(gate.get("status")) is not str or gate.get("status") != "PASS":
        raise TasteGNNStageError("prepared stage gate is not PASS")
    if gate.get("marker") != marker:
        raise TasteGNNStageError("prepared stage marker differs from gate")
    required = _stage_required_files(gate.get("stage"))
    if required is not None and set(expected) != required:
        raise TasteGNNStageError("prepared stage file set differs from frozen contract")
    output.verify(label="prepared stage output after verification")
    result = {
        "root": str(output.path),
        "gate": gate,
        "hashes_verified": len(actual),
    }
    if retained is not None:
        if retained.output is not output or retained.marker != marker:
            raise TasteGNNStageError("prepared stage retained authority changed")
        if retained.marker_payload != marker_payload:
            raise TasteGNNStageError("prepared stage publication authority changed")
        retained._verify_files()
    return result


def _prepare_stage_output(
    output: PhysicalDirectory,
    *,
    documents: Mapping[str, Mapping[str, Any]],
    marker: str,
) -> PreparedStageOutput:
    """Publish all stage evidence except the final PASS marker."""

    directory = output
    directory.verify(label="stage output before preparation")
    if os.listdir(directory.descriptor):
        raise TasteGNNStageError("fresh stage output is not empty")
    if "gate.json" not in documents:
        raise TasteGNNStageError("stage publication lacks gate.json")
    if Path(marker).name != marker or marker in {"", "sha256sums.txt"}:
        raise TasteGNNStageError("stage marker is unsafe")
    if marker in documents or "sha256sums.txt" in documents:
        raise TasteGNNStageError("stage documents collide with terminal files")
    for name, payload in documents.items():
        if Path(name).name != name or not name.endswith(".json"):
            raise TasteGNNStageError(f"unsafe stage document name: {name}")
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8")
        _write_bytes_at(directory, name, encoded, label=f"stage document {name}")
        if not set(os.listdir(directory.descriptor)).issubset(documents):
            raise TasteGNNStageError("unexpected file appeared during stage preparation")
    if set(os.listdir(directory.descriptor)) != set(documents):
        raise TasteGNNStageError("stage document preparation is incomplete")
    hashes = {
        name: _sha256_at(directory, name, label=f"stage output {name}")
        for name in documents
    }
    placeholder_hashes = {**hashes, marker: "0" * 64}
    placeholder_lines = [
        f"{placeholder_hashes[name]}  {name}" for name in sorted(placeholder_hashes)
    ]
    expected_sha_size = len(("\n".join(placeholder_lines) + "\n").encode("utf-8"))
    sha_descriptor = os.open(
        "sha256sums.txt",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory.descriptor,
    )
    try:
        initial_sha = os.fstat(sha_descriptor)
        if not stat.S_ISREG(initial_sha.st_mode) or initial_sha.st_nlink != 1:
            raise TasteGNNStageError(
                "stage SHA inventory is not one regular single-link file"
            )
        document_names = set(documents)
        authority_files: dict[str, dict[str, int]] = {}
        for name in sorted(document_names):
            info = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise TasteGNNStageError(
                    f"stage document is not one regular single-link file: {name}"
                )
            authority_files[name] = _stat_identity(info)
        sha_initial_identity = _stat_identity(initial_sha)
        authority_files["sha256sums.txt"] = {
            **{
                key: sha_initial_identity[key]
                for key in ("device", "inode", "mode", "nlink")
            },
            "size": expected_sha_size,
        }
        authority_payload = {
            "schema_version": "tastemolnet_stage_publication_authority_v1",
            "files": authority_files,
        }
        authority_sha256 = hashlib.sha256(
            json.dumps(
                authority_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        marker_payload = _stage_marker_payload(marker, authority_sha256)
        hashes[marker] = hashlib.sha256(marker_payload).hexdigest()
        lines = [f"{hashes[name]}  {name}" for name in sorted(hashes)]
        sha_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        if len(sha_bytes) != expected_sha_size:
            raise TasteGNNStageError("stage SHA inventory size formula changed")
        offset = 0
        while offset < len(sha_bytes):
            offset += os.write(sha_descriptor, sha_bytes[offset:])
        os.fsync(sha_descriptor)
        final_sha = os.fstat(sha_descriptor)
        named_sha = os.stat(
            "sha256sums.txt",
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        if (
            _stat_identity(final_sha) != _stat_identity(named_sha)
            or any(
                _stat_identity(final_sha)[key]
                != authority_files["sha256sums.txt"][key]
                for key in ("device", "inode", "mode", "nlink", "size")
            )
        ):
            raise TasteGNNStageError("stage SHA inventory changed while publishing")
        sha_identity = _stat_identity(final_sha)
    finally:
        os.close(sha_descriptor)
    directory.refresh_leaf_after_authorized_mutation(
        label="stage output after SHA inventory"
    )
    os.fsync(directory.descriptor)
    retained_files: dict[str, _HeldPreparedStageFile] = {}
    try:
        for name in sorted(os.listdir(directory.descriptor)):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory.descriptor,
            )
            try:
                held = os.fstat(descriptor)
                named = os.stat(
                    name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(held.st_mode)
                    or held.st_nlink != 1
                    or _stat_identity(held) != _stat_identity(named)
                ):
                    raise TasteGNNStageError(
                        f"prepared stage file is not one physical file: {name}"
                    )
                _data, read_evidence = _read_bytes_at(
                    directory,
                    name,
                    label=f"prepared stage retained file {name}",
                )
                retained_files[name] = _HeldPreparedStageFile(
                    name=name,
                    descriptor=descriptor,
                    evidence=_stat_identity(held),
                    sha256=str(read_evidence["sha256"]),
                )
                descriptor = -1
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        prepared = PreparedStageOutput(
            output=directory,
            marker=marker,
            files=retained_files,
            sha_identity=sha_identity,
            marker_payload=marker_payload,
        )
        prepared.revalidate()
        return prepared
    except Exception:
        for retained_file in retained_files.values():
            if retained_file.descriptor >= 0:
                os.close(retained_file.descriptor)
        raise


def _publish_prepared_stage_marker(
    prepared: PreparedStageOutput,
    *,
    retained_input_closure: Callable[[], None],
) -> dict[str, int]:
    """Commit a prepared stage after its retained inputs close one last time.

    The callback deliberately runs inside this helper, after the prepared
    output has been revalidated and immediately before the O_EXCL marker
    creation.  Keeping that boundary here prevents a caller from accidentally
    leaving an unchecked input-mutation window between its last snapshot and
    the terminal publication.
    """

    prepared.revalidate()
    retained_input_closure()
    return _write_bytes_at(
        prepared.output,
        prepared.marker,
        prepared.marker_payload,
        label="stage PASS marker",
    )


def _write_stage_output(
    output: PhysicalDirectory,
    *,
    documents: Mapping[str, Mapping[str, Any]],
    marker: str,
) -> dict[str, dict[str, int]]:
    """Convenience publisher; production routes split prepare from commit."""

    prepared = _prepare_stage_output(output, documents=documents, marker=marker)
    try:
        marker_identity = _publish_prepared_stage_marker(
            prepared,
            retained_input_closure=lambda: None,
        )
        return {
            "sha256sums.txt": prepared.sha_identity,
            marker: marker_identity,
        }
    finally:
        prepared.close()


def verify_stage_output(
    root: str | Path | PhysicalDirectory,
) -> dict[str, Any]:
    owned = not isinstance(root, PhysicalDirectory)
    output = _physical_directory(root, field="stage output") if owned else root
    assert isinstance(output, PhysicalDirectory)
    try:
        expected = _read_stage_hash_inventory(output)
        actual = {
            name
            for name in os.listdir(output.descriptor)
            if name != "sha256sums.txt"
        }
        if set(expected) != actual:
            raise TasteGNNStageError("stage SHA inventory does not close the output root")
        for name, digest in expected.items():
            if _sha256_at(output, name, label=f"stage output {name}") != digest:
                raise TasteGNNStageError(f"stage output hash mismatch: {name}")
        gate = _read_json_at(output, "gate.json", label="stage gate")
        if type(gate.get("status")) is not str or gate.get("status") != "PASS":
            raise TasteGNNStageError("stage gate is not PASS")
        marker = gate.get("marker")
        if type(marker) is not str or Path(marker).name != marker:
            raise TasteGNNStageError("stage marker is unsafe")
        document_names = actual - {marker}
        _authority, authority_sha256 = _stage_publication_authority(
            output,
            document_names=document_names,
        )
        marker_bytes, _marker_evidence = _read_bytes_at(
            output,
            marker,
            label="stage marker",
        )
        if marker_bytes != _stage_marker_payload(marker, authority_sha256):
            raise TasteGNNStageError(
                "stage marker is absent, malformed, or physically unbound"
            )
        required = _stage_required_files(gate.get("stage"))
        if required is not None and actual != required:
            raise TasteGNNStageError("stage output file set differs from frozen contract")
        return {
            "root": str(output.path),
            "gate": gate,
            "hashes_verified": len(expected),
        }
    finally:
        if owned:
            output.close()


@dataclass(slots=True)
class HeldTasteStageOutput:
    """Retained T3/T4 authority for downstream consumers with no reopen window."""

    directory: PhysicalDirectory
    snapshot: Mapping[str, Any]
    evidence: Mapping[str, str]

    @property
    def root(self) -> Path:
        return self.directory.path

    def revalidate(self) -> dict[str, str]:
        verified = verify_stage_output(self.directory)
        gate = verified["gate"]
        current_snapshot = _complete_directory_snapshot(
            self.directory, label="held Taste stage output"
        )
        checkpoint_dir = gate.get("checkpoint_dir")
        if (
            type(checkpoint_dir) is not str
            or not Path(checkpoint_dir).is_absolute()
            or Path(os.path.abspath(checkpoint_dir)) != Path(checkpoint_dir)
        ):
            raise TasteGNNStageError("held stage checkpoint_dir is not exact absolute")
        current = {
            "stage": str(gate.get("stage") or ""),
            "gate_sha256": _sha256_at(
                self.directory, "gate.json", label="held Taste stage gate"
            ),
            "root_inventory_sha256": str(current_snapshot["inventory_sha256"]),
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_id": _hex(
                gate.get("checkpoint_id"), field="held_stage.checkpoint_id"
            ),
            "checkpoint_inventory_sha256": _hex(
                gate.get("checkpoint_inventory_sha256"),
                field="held_stage.checkpoint_inventory_sha256",
            ),
            "checkpoint_stat_inventory_sha256": _hex(
                gate.get("checkpoint_stat_inventory_sha256"),
                field="held_stage.checkpoint_stat_inventory_sha256",
            ),
            "checkpoint_sha256s_sha256": _hex(
                gate.get("checkpoint_sha256s_sha256"),
                field="held_stage.checkpoint_sha256s_sha256",
            ),
            **_t2_binding_stage_evidence(gate),
        }
        if current_snapshot != self.snapshot or current != dict(self.evidence):
            raise TasteGNNStageError("held Taste stage output authority changed")
        return current

    def close(self) -> None:
        self.directory.close()

    def __enter__(self) -> HeldTasteStageOutput:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - process-exit cleanup.
        self.close()


def hold_taste_stage_output(root: str | Path) -> HeldTasteStageOutput:
    """Open one complete T3/T4 output once and retain its physical authority."""

    directory = _physical_directory(root, field="Taste stage output")
    try:
        verified = verify_stage_output(directory)
        gate = verified["gate"]
        stage = gate.get("stage")
        if stage not in {T3_STAGE, T4_STAGE}:
            raise TasteGNNStageError("held Taste output is not a T3/T4 stage")
        snapshot = _complete_directory_snapshot(
            directory, label="held Taste stage output"
        )
        checkpoint_dir = gate.get("checkpoint_dir")
        if (
            type(checkpoint_dir) is not str
            or not Path(checkpoint_dir).is_absolute()
            or Path(os.path.abspath(checkpoint_dir)) != Path(checkpoint_dir)
        ):
            raise TasteGNNStageError("held stage checkpoint_dir is not exact absolute")
        evidence = {
            "stage": str(stage),
            "gate_sha256": _sha256_at(
                directory, "gate.json", label="held Taste stage gate"
            ),
            "root_inventory_sha256": str(snapshot["inventory_sha256"]),
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_id": _hex(
                gate.get("checkpoint_id"), field="held_stage.checkpoint_id"
            ),
            "checkpoint_inventory_sha256": _hex(
                gate.get("checkpoint_inventory_sha256"),
                field="held_stage.checkpoint_inventory_sha256",
            ),
            "checkpoint_stat_inventory_sha256": _hex(
                gate.get("checkpoint_stat_inventory_sha256"),
                field="held_stage.checkpoint_stat_inventory_sha256",
            ),
            "checkpoint_sha256s_sha256": _hex(
                gate.get("checkpoint_sha256s_sha256"),
                field="held_stage.checkpoint_sha256s_sha256",
            ),
            **_t2_binding_stage_evidence(gate),
        }
        held = HeldTasteStageOutput(
            directory=directory,
            snapshot=snapshot,
            evidence=evidence,
        )
        held.revalidate()
        return held
    except Exception:
        directory.close()
        raise


@dataclass(slots=True)
class HeldTasteCheckpointBundle:
    """Retained exact T2 root for frozen-GINE downstream model/reward loading."""

    directory: PhysicalDirectory
    snapshot: Mapping[str, Any]
    evidence: Mapping[str, str]

    @property
    def checkpoint_dir(self) -> Path:
        return self.directory.path

    def revalidate(self) -> dict[str, str]:
        self.directory.verify(label="held Taste T2 checkpoint")
        current = _checkpoint_manifest_snapshot(self.directory)
        if current != self.snapshot:
            raise TasteGNNStageError("held Taste T2 checkpoint authority changed")
        if (
            str(self.directory.path) != self.evidence["checkpoint_dir"]
            or current["inventory_sha256"]
            != self.evidence["checkpoint_inventory_sha256"]
            or current["stat_inventory_sha256"]
            != self.evidence["checkpoint_stat_inventory_sha256"]
            or _sha256_at(
                self.directory,
                "sha256sums.txt",
                label="held checkpoint SHA inventory",
            )
            != self.evidence["checkpoint_sha256s_sha256"]
            or _sha256_at(
                self.directory,
                "model.pt",
                label="held selected model.pt",
            )
            != self.evidence["checkpoint_id"]
        ):
            raise TasteGNNStageError("held Taste T2 checkpoint gate binding changed")
        return dict(self.evidence)

    def read_frozen_gine_payload(self, name: str) -> bytes:
        """Descriptor-relative read for exact frozen-GINE downstream payloads."""

        child = _safe_child_name(name, label="frozen GINE payload")
        if child not in T6_FROZEN_GINE_PAYLOAD_FILES:
            raise TasteGNNStageError(
                "downstream stages may not open checkpoint payload through "
                f"this API: {child}"
            )
        self.revalidate()
        expected = self.snapshot["files"][child]["sha256"]
        data, _evidence = _read_bytes_at(
            self.directory,
            child,
            label=f"held frozen GINE {child}",
            expected_sha256=expected,
        )
        self.revalidate()
        return data

    def close(self) -> None:
        self.directory.close()

    def __enter__(self) -> HeldTasteCheckpointBundle:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - process-exit cleanup.
        self.close()


def hold_taste_checkpoint_bundle(
    path: str | Path,
    *,
    expected_stage_evidence: Mapping[str, Any],
) -> HeldTasteCheckpointBundle:
    """Hold exact T2 authority bound by a retained T3/T4 stage output.

    This verifier uses the hash manifest plus full stat inventory and therefore
    does not open validation/calibration/test payloads.  Consumers must retain
    this object and call ``revalidate`` after any path-based model loader, or
    use ``read_frozen_gine_payload`` for descriptor-relative bytes.
    """

    if set(expected_stage_evidence) != HELD_STAGE_EVIDENCE_KEYS:
        raise TasteGNNStageError("checkpoint stage evidence keys changed")
    evidence: dict[str, str] = {}
    for key in HELD_STAGE_EVIDENCE_KEYS:
        value = expected_stage_evidence.get(key)
        if type(value) is not str:
            raise TasteGNNStageError(f"checkpoint stage evidence {key} must be a string")
        evidence[key] = value
    if evidence["stage"] not in {T3_STAGE, T4_STAGE}:
        raise TasteGNNStageError("checkpoint authority requires T3/T4 evidence")
    for key in (
        "gate_sha256",
        "root_inventory_sha256",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "t2_adoption_gate_sha256",
        "t2_adoption_receipt_sha256",
        "t2_adoption_binding_sha256",
    ):
        _hex(evidence[key], field=f"checkpoint_stage_evidence.{key}")
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        raise TasteGNNStageError("Taste checkpoint path must be absolute")
    requested = Path(os.path.abspath(requested))
    if str(requested) != evidence["checkpoint_dir"]:
        raise TasteGNNStageError(
            "Taste checkpoint path differs from exact stage authority"
        )
    directory = _physical_directory(requested, field="held Taste T2 checkpoint")
    try:
        snapshot = _checkpoint_manifest_snapshot(directory)
        held = HeldTasteCheckpointBundle(
            directory=directory,
            snapshot=snapshot,
            evidence=evidence,
        )
        held.revalidate()
        return held
    except Exception:
        directory.close()
        raise


def _run_t3_existing_fit_adoption(
    *,
    t2_adoption: HeldT2PassAdoption,
    checkpoint_dir: str | Path,
    graph_cache_root: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    downstream_policy_path: str | Path,
    base_policy_path: str | Path,
) -> dict[str, Any]:
    policy = load_tastemolnet_downstream_policy(
        downstream_policy_path, base_policy_path=base_policy_path
    )
    checkpoint_root: PhysicalDirectory | None = None
    cache_root: PhysicalDirectory | None = None
    execution_root: PhysicalDirectory | None = None
    output_plan: StageOutputPlan | None = None
    output_root: PhysicalDirectory | None = None
    prepared_output: PreparedStageOutput | None = None
    try:
        contract = policy.stage(T3_STAGE)
        if (
            contract.get("mode") != "adopt_existing_validation_fit"
            or contract.get("device") != "cpu"
            or contract.get("physical_gpu_index") is not None
        ):
            raise TasteGNNStageError("T3 authority does not permit existing-fit adoption")
        checkpoint_root = _physical_directory(checkpoint_dir, field="T2 checkpoint")
        t2_binding = _bind_t2_adoption_to_checkpoint(
            t2_adoption, checkpoint_root
        )
        cache_root = _physical_directory(
            graph_cache_root, field="graph-cache exclusion authority"
        )
        execution_root = _physical_directory(
            EXECUTION_SOURCE_ROOT, field="loaded execution source root"
        )
        checkpoint = checkpoint_root.path
        forbidden_identities = _merge_identity_paths(
            checkpoint_root.identity_paths(),
            cache_root.identity_paths(),
            execution_root.identity_paths(),
            policy.directory_identity_paths(),
        )
        output_plan = _plan_stage_output(
            artifact_root=artifact_root,
            output_dir=output_dir,
            basename_prefix="calibrated-",
            forbidden_paths=(
                checkpoint,
                cache_root.path,
                execution_root.path,
                t2_adoption.root,
                *policy.protected_paths(),
            ),
            forbidden_identity_paths=forbidden_identities,
        )
        checkpoint_root.verify(label="T2 checkpoint before snapshot")
        before = _checkpoint_snapshot(checkpoint_root)
        stat_before = _checkpoint_stat_snapshot(checkpoint_root)
        audit = _verify_taste_checkpoint_bundle(checkpoint_root)
        checkpoint_root.verify(label="T2 checkpoint after bundle verification")
        model_card = audit["model_card"]
        validate_taste_model_card(model_card)
        if _read_json_at(checkpoint_root, "label_map.json", label="label map") != LABEL_MAP:
            raise TasteGNNStageError("Taste label map changed")
        test_status = _read_json_at(
            checkpoint_root, "test_evaluation_status.json", label="test status"
        )
        if (
            test_status.get("status") != "NOT_EVALUATED"
            or test_status.get("test_loaded") is not False
        ):
            raise TasteGNNStageError("T3 source bundle has unsafe test status")
        calibration = validate_existing_temperature_fit(checkpoint_root)
        checkpoint_root.verify(label="T2 checkpoint after calibration adoption")
        after = _checkpoint_snapshot(checkpoint_root)
        if before != after:
            raise TasteGNNStageError("T3 immutable source bundle changed during adoption")
        checkpoint_id = str(model_card.get("checkpoint_id") or "")
        if checkpoint_id != _sha256_at(checkpoint_root, "model.pt", label="model.pt"):
            raise TasteGNNStageError("T3 checkpoint ID differs from model.pt")
        policy_binding = policy.evidence(stage=T3_STAGE)
        adoption = {
            "schema_version": "tastemolnet_t3_existing_fit_adoption_v1",
            "status": "PASS",
            "stage": T3_STAGE,
            "dataset": "tastemolnet",
            "checkpoint_dir": str(checkpoint),
            "checkpoint_id": checkpoint_id,
            "checkpoint_inventory_sha256": before["inventory_sha256"],
            "checkpoint_stat_inventory_sha256": stat_before["inventory_sha256"],
            "checkpoint_sha256s_sha256": _sha256_at(
                checkpoint_root, "sha256sums.txt", label="checkpoint SHA inventory"
            ),
            "source_bundle_unchanged": True,
            "checkpoint_copied": False,
            "temperature_refit_performed": False,
            "bundle_evidence_files_opened": ["validation_predictions.csv"],
            "external_split_payload_files_opened": [],
            "validation_predictions_source": "immutable_t2_bundle",
            "calibration": calibration,
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "strict_flip": "pred_before == 1 and pred_after != 1",
            "test_loaded": False,
            "data_redistribution_allowed": False,
            "t2_adoption_binding": t2_binding,
        }
        oracle_reference = {
            "schema_version": "tastemolnet_t3_oracle_reference_v1",
            "dataset": "tastemolnet",
            "checkpoint_id": checkpoint_id,
            "selected_inference_asset": "model.pt",
            "model_sha256": _sha256_at(
                checkpoint_root, "model.pt", label="model.pt"
            ),
            "last_checkpoint_terminal_only": True,
            "last_sha256": _sha256_at(
                checkpoint_root, "last.pt", label="last.pt"
            ),
            "temperature_scaling_sha256": _sha256_at(
                checkpoint_root,
                "temperature_scaling.json",
                label="temperature scaling",
            ),
            "config_sha256": _sha256_at(
                checkpoint_root, "config.yaml", label="config"
            ),
            "feature_schema_sha256": _sha256_at(
                checkpoint_root, "feature_schema.json", label="feature schema"
            ),
            "label_map_sha256": _sha256_at(
                checkpoint_root, "label_map.json", label="label map"
            ),
            "num_classes": NUM_CLASSES,
            "source_label": SOURCE_LABEL,
            "rf_oracle_used": False,
            "t2_adoption_binding": t2_binding,
        }
        gate = {
            "schema_version": "tastemolnet_main_stage_gate_v1",
            "stage": T3_STAGE,
            "status": "PASS",
            "marker": T3_MARKER,
            "depends_on": [ADOPTION_MARKER],
            "t2_science_bundle_verified": True,
            "t2_adoption_binding": t2_binding,
            "checkpoint_dir": str(checkpoint),
            "checkpoint_id": checkpoint_id,
            "checkpoint_inventory_sha256": before["inventory_sha256"],
            "checkpoint_stat_inventory_sha256": stat_before["inventory_sha256"],
            "checkpoint_sha256s_sha256": _sha256_at(
                checkpoint_root, "sha256sums.txt", label="checkpoint SHA inventory"
            ),
            "downstream_policy_sha256": policy.file_sha256,
            "existing_fit_adopted": True,
            "temperature_refit_performed": False,
            "test_loaded": False,
        }
        checkpoint_root.verify(label="T2 checkpoint before T3 publication")
        _validated_t2_adoption_binding(t2_adoption)
        policy.verify_authorities()
        execution_root.verify(label="loaded execution source before T3 publication")
        cache_root.verify(label="graph-cache exclusion before T3 publication")
        if before != _checkpoint_snapshot(checkpoint_root):
            raise TasteGNNStageError("T3 immutable source bundle drifted before publication")
        output_root = output_plan.create()
        prepared_output = _prepare_stage_output(
            output_root,
            documents={
                "calibration_adoption.json": adoption,
                "oracle_reference.json": oracle_reference,
                "policy_binding.json": policy_binding,
                "gate.json": gate,
            },
            marker=T3_MARKER,
        )

        # Prepared evidence is not a PASS boundary.  Close every retained input
        # authority before the marker is created as the final commit operation.
        checkpoint_root.verify(label="T2 checkpoint after T3 preparation")
        _validated_t2_adoption_binding(t2_adoption)
        if before != _checkpoint_snapshot(checkpoint_root):
            raise TasteGNNStageError("T3 checkpoint bytes drifted during preparation")
        if stat_before != _checkpoint_stat_snapshot(checkpoint_root):
            raise TasteGNNStageError("T3 checkpoint stat inventory drifted during preparation")
        policy.verify_authorities()
        execution_root.verify(label="loaded execution source after T3 preparation")
        cache_root.verify(label="graph-cache exclusion after T3 preparation")
        prepared_output.revalidate()
        def retained_t3_input_closure() -> None:
            if _validated_t2_adoption_binding(t2_adoption) != t2_binding:
                raise TasteGNNStageError(
                    "fresh T2 adoption binding drifted at T3 PASS boundary"
                )
            checkpoint_root.verify(label="T2 checkpoint at T3 PASS boundary")
            if before != _checkpoint_snapshot(checkpoint_root):
                raise TasteGNNStageError(
                    "T3 checkpoint bytes drifted at PASS boundary"
                )
            if stat_before != _checkpoint_stat_snapshot(checkpoint_root):
                raise TasteGNNStageError(
                    "T3 checkpoint stat inventory drifted at PASS boundary"
                )
            policy.verify_authorities()
            execution_root.verify(label="loaded execution source at T3 PASS boundary")
            cache_root.verify(label="graph-cache exclusion at T3 PASS boundary")

        _publish_prepared_stage_marker(
            prepared_output,
            retained_input_closure=retained_t3_input_closure,
        )
        return adoption
    finally:
        if prepared_output is not None:
            prepared_output.close()
        if output_plan is not None:
            output_plan.close()
        if checkpoint_root is not None:
            checkpoint_root.close()
        if cache_root is not None:
            cache_root.close()
        if execution_root is not None:
            execution_root.close()
        policy.close()


def run_t3_existing_fit_adoption(
    *,
    t2_adoption_root: str | Path,
    t2_adoption_gate_sha256: str,
    t2_adoption_receipt_sha256: str,
    t2_source_evidence_sha256: str,
    checkpoint_dir: str | Path,
    graph_cache_root: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    downstream_policy_path: str | Path,
    base_policy_path: str | Path,
) -> dict[str, Any]:
    """Run T3 only while the fresh T2 PASS adoption remains held."""

    try:
        authority = hold_t2_gine_pass_adoption(
            t2_adoption_root,
            expected_gate_sha256=t2_adoption_gate_sha256,
            expected_receipt_sha256=t2_adoption_receipt_sha256,
            expected_source_evidence_sha256=t2_source_evidence_sha256,
        )
    except (T2PassAdoptionError, OSError, ValueError) as exc:
        raise TasteGNNStageError(
            f"T3 requires the fresh T2 GINE PASS adoption: {exc}"
        ) from exc
    try:
        return _run_t3_existing_fit_adoption(
            t2_adoption=authority,
            checkpoint_dir=checkpoint_dir,
            graph_cache_root=graph_cache_root,
            artifact_root=artifact_root,
            output_dir=output_dir,
            downstream_policy_path=downstream_policy_path,
            base_policy_path=base_policy_path,
        )
    finally:
        authority.close()


def _real_connected_deletions(
    parent_smiles: str,
    *,
    parent_id: str,
    maximum: int,
) -> list[tuple[str, HardDeletionOutcome]]:
    if Chem is None:
        raise TasteGNNStageError("RDKit is required for Taste T4 oracle smoke")
    parent = Chem.MolFromSmiles(parent_smiles, sanitize=True)
    if parent is None:
        return []
    atom_sets: list[tuple[int, ...]] = [(int(atom.GetIdx()),) for atom in parent.GetAtoms()]
    atom_sets.extend(
        sorted(
            {
                tuple(sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))))
                for bond in parent.GetBonds()
            }
        )
    )
    retained: list[tuple[str, HardDeletionOutcome]] = []
    seen: set[tuple[tuple[int, ...], str]] = set()
    for attempt, atom_indices in enumerate(atom_sets):
        outcome = apply_hard_deletion_match(
            parent,
            atom_indices,
            parent_id=parent_id,
            candidate_id=f"taste-t4-{attempt}",
            match_id=attempt,
        )
        if not outcome.valid or not outcome.residual_smiles:
            continue
        fragment = Chem.MolFragmentToSmiles(
            parent,
            atomsToUse=list(atom_indices),
            canonical=True,
            isomericSmiles=True,
        )
        if not fragment or Chem.MolFromSmiles(fragment, sanitize=True) is None:
            continue
        identity = (outcome.match_atom_indices, outcome.residual_smiles)
        if identity in seen:
            continue
        seen.add(identity)
        retained.append((fragment, outcome))
        if len(retained) == maximum:
            break
    return retained


def _graph_from_smiles(
    featurizer: MolecularGraphFeaturizer,
    smiles: str,
    molecule_id: str,
) -> MolecularGraphData:
    features = featurizer.featurize(smiles)
    return MolecularGraphData(
        x=features.node_features,
        edge_index=features.edge_index,
        edge_attr=features.edge_features,
        y=-1,
        molecule_id=molecule_id,
        smiles=features.canonical_smiles,
        split="oracle_smoke",
        graph_sha256=features.graph_sha256,
    )


def _validate_probability_record(record: Mapping[str, Any], *, checkpoint_id: str) -> None:
    if record.get("checkpoint_id") != checkpoint_id:
        raise TasteGNNStageError("oracle record checkpoint identity changed")
    if type(record.get("num_classes")) is not int or record.get("num_classes") != NUM_CLASSES:
        raise TasteGNNStageError("oracle record is not three-class")
    if record.get("source_label") != SOURCE_LABEL:
        raise TasteGNNStageError("oracle record source label changed")
    predicted = record.get("predicted_label")
    if type(predicted) is not int or predicted < 0 or predicted >= NUM_CLASSES:
        raise TasteGNNStageError("oracle predicted label is outside the three classes")
    values = record.get("probabilities")
    if not isinstance(values, list) or len(values) != NUM_CLASSES:
        raise TasteGNNStageError("oracle probability vector is not three-wide")
    probabilities = np.asarray(
        [_finite(value, field="oracle.probability") for value in values], dtype=np.float64
    )
    if (probabilities < 0.0).any() or not math.isclose(
        float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise TasteGNNStageError("oracle probabilities are outside the simplex")
    logits = record.get("logits")
    if not isinstance(logits, list) or len(logits) != NUM_CLASSES:
        raise TasteGNNStageError("oracle logits vector is not three-wide")
    normalized_logits = np.asarray(
        [_finite(value, field="oracle.logit") for value in logits], dtype=np.float64
    )
    if not np.isfinite(normalized_logits).all():
        raise TasteGNNStageError("oracle logits contain a non-finite value")
    temperature = _finite(record.get("temperature"), field="oracle.temperature", positive=True)
    shifted = normalized_logits / temperature
    shifted -= float(np.max(shifted))
    exponentials = np.exp(shifted)
    probabilities_from_logits = exponentials / float(exponentials.sum())
    if int(np.argmax(probabilities)) != predicted:
        raise TasteGNNStageError("oracle predicted label differs from probability argmax")
    if int(np.argmax(normalized_logits)) != predicted:
        raise TasteGNNStageError("oracle predicted label differs from logits argmax")
    if not np.allclose(
        probabilities_from_logits, probabilities, rtol=0.0, atol=1e-7
    ):
        raise TasteGNNStageError("oracle probabilities differ from calibrated logits")
    source_probability = _finite(
        record.get("source_probability"), field="oracle.source_probability"
    )
    confidence = _finite(record.get("confidence"), field="oracle.confidence")
    if (
        not math.isclose(
            source_probability,
            float(probabilities[SOURCE_LABEL]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or not math.isclose(
            confidence,
            float(probabilities[predicted]),
            rel_tol=0.0,
            abs_tol=1e-7,
        )
    ):
        raise TasteGNNStageError("oracle confidence fields differ from probabilities")


def _load_calibration_cache(
    *,
    graph_cache_root: str | Path | PhysicalDirectory,
    expected_manifest_sha256: str,
    feature_schema: MolecularFeatureSchema,
) -> tuple[MolecularGraphDataset, dict[str, Any]]:
    owned = not isinstance(graph_cache_root, PhysicalDirectory)
    root = (
        _physical_directory(graph_cache_root, field="graph-cache root")
        if owned
        else graph_cache_root
    )
    assert isinstance(root, PhysicalDirectory)
    try:
        manifest_bytes, manifest_evidence = _read_bytes_at(
            root,
            "manifest.json",
            label="Taste graph-cache manifest",
            expected_sha256=expected_manifest_sha256,
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TasteGNNStageError("graph-cache manifest is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise TasteGNNStageError("graph-cache manifest must be one JSON object")
        splits = manifest.get("splits")
        if (
            manifest.get("schema_version") != "molecular_graph_cache_manifest_v1"
            or manifest.get("dataset") != "tastemolnet"
            or type(manifest.get("num_classes")) is not int
            or manifest.get("num_classes") != NUM_CLASSES
            or manifest.get("split_order") != ["train", "validation", "calibration", "test"]
            or not isinstance(splits, Mapping)
            or set(splits) != {"train", "validation", "calibration", "test"}
        ):
            raise TasteGNNStageError("Taste graph-cache manifest contract changed")
        entry = splits["calibration"]
        if not isinstance(entry, Mapping):
            raise TasteGNNStageError("calibration cache entry is not a mapping")
        if (
            entry.get("cache_file") != "calibration.pt"
            or type(entry.get("num_classes")) is not int
            or entry.get("num_classes") != NUM_CLASSES
            or entry.get("safe_load_verified") is not True
        ):
            raise TasteGNNStageError("calibration cache entry contract changed")
        graph_count = _native_int(entry.get("graph_count"), field="calibration.graph_count")
        if graph_count <= 0:
            raise TasteGNNStageError("calibration graph count must be positive")
        cache_sha256 = _hex(entry.get("cache_sha256"), field="calibration.cache_sha256")
        source_sha256 = _hex(
            entry.get("source_csv_sha256"), field="calibration.source_csv_sha256"
        )
        cache_bytes, cache_evidence = _read_bytes_at(
            root,
            "calibration.pt",
            expected_sha256=cache_sha256,
            label="Taste calibration graph cache",
        )
        dataset = load_molecular_graph_cache(
            io.BytesIO(cache_bytes),
            expected_num_classes=NUM_CLASSES,
            expected_source_sha256=source_sha256,
            expected_feature_schema=feature_schema,
        )
        root.verify(label="graph-cache root after calibration decode")
        if len(dataset) != graph_count:
            raise TasteGNNStageError("calibration cache count differs from manifest")
        if any(str(dataset[index].split) != "calibration" for index in range(len(dataset))):
            raise TasteGNNStageError("calibration cache contains a non-calibration row")
        return dataset, {
            "schema_version": "tastemolnet_t4_data_access_v1",
            "graph_cache_manifest": manifest_evidence,
            "opened_payload_files": [
                {
                    "name": "calibration.pt",
                    "sha256": cache_evidence["sha256"],
                    "graph_count": graph_count,
                    "source_csv_sha256": source_sha256,
                }
            ],
            "opened_payload_splits": ["calibration"],
            "train_payload_opened": False,
            "validation_payload_opened": False,
            "test_payload_opened": False,
            "test_metadata_hash_only": True,
            "csv_payload_opened": False,
            "graph_cache_rebuilt": False,
            "data_reprepared": False,
        }
    finally:
        if owned:
            root.close()


def _cohort_digest(selected: Sequence[tuple[int, MolecularGraphData, Any]]) -> str:
    payload = [
        {"source_index": index, "graph_sha256": graph.graph_sha256}
        for index, graph, _actions in selected
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_gnn_oracle_anchored(
    checkpoint: PhysicalDirectory,
    *,
    feature_schema: MolecularFeatureSchema,
    device: str,
    batch_size: int,
) -> GNNOracle:
    """Load the selected model once from bytes anchored under the held T2 dirfd."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - AutoDL gate owns torch.
        raise TasteGNNStageError("PyTorch is required for Taste T4") from exc
    model_bytes, model_evidence = _read_bytes_at(
        checkpoint, "model.pt", label="selected model.pt"
    )
    try:
        payload = torch.load(
            io.BytesIO(model_bytes), map_location=device, weights_only=True
        )
    except Exception as exc:
        raise TasteGNNStageError("selected model.pt is not safely loadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("bundle_version") != CHECKPOINT_BUNDLE_VERSION
    ):
        raise TasteGNNStageError("selected model.pt bundle version changed")
    schema_payload = feature_schema.to_dict()
    if payload.get("feature_schema_sha256") != schema_payload["schema_sha256"]:
        raise TasteGNNStageError("selected model and feature schema differ")
    config = MolecularGNNConfig.from_mapping(payload.get("model_config", {}))
    model = MolecularGNN(
        config,
        node_cardinalities=feature_schema.node_cardinalities,
        edge_cardinalities=feature_schema.edge_cardinalities,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TasteGNNStageError("selected model lacks a state_dict")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model_card = _read_json_at(checkpoint, "model_card.json", label="model card")
    temperature = _read_json_at(
        checkpoint, "temperature_scaling.json", label="temperature scaling"
    )
    return GNNOracle(
        model,
        device=device,
        checkpoint_id=str(model_evidence["sha256"]),
        backbone=str(model_card.get("backbone")),
        num_classes=_native_int(
            model_card.get("num_classes"), field="model_card.num_classes"
        ),
        source_label=_native_int(
            model_card.get("source_label"), field="model_card.source_label"
        ),
        temperature=_finite(
            temperature.get("temperature"), field="temperature", positive=True
        ),
        edge_feature_dim=len(feature_schema.edge_fields),
        default_batch_size=batch_size,
        checkpoint_dir=None,
    )


def run_bounded_oracle_smoke(
    *,
    dataset: MolecularGraphDataset,
    oracle: GNNOracle,
    feature_schema: MolecularFeatureSchema,
    batch_size: int,
    source_count: int,
    max_deletions_per_parent: int,
) -> dict[str, Any]:
    """Exercise one loaded multiclass oracle on a deterministic 16-parent cohort."""

    if type(source_count) is not int or source_count != 16:
        raise TasteGNNStageError("Taste T4 source cohort is frozen at exactly 16")
    if type(max_deletions_per_parent) is not int or max_deletions_per_parent != 4:
        raise TasteGNNStageError("Taste T4 deletion cap is frozen at exactly four")
    if type(batch_size) is not int or batch_size <= 0:
        raise TasteGNNStageError("Taste T4 batch size must be a native positive integer")
    graphs = [dataset[index] for index in range(len(dataset))]
    records = oracle.predict_records(graphs, batch_size=batch_size)
    if len(records) != len(graphs):
        raise TasteGNNStageError("oracle prediction count differs from calibration cache")
    for record in records:
        _validate_probability_record(record, checkpoint_id=oracle.checkpoint_id)

    selected: list[tuple[int, MolecularGraphData, list[tuple[str, HardDeletionOutcome]]]] = []
    correctly_predicted_source_count = 0
    for index, (graph, record) in enumerate(zip(graphs, records, strict=True)):
        if int(graph.y) != SOURCE_LABEL or int(record["predicted_label"]) != SOURCE_LABEL:
            continue
        correctly_predicted_source_count += 1
        actions = _real_connected_deletions(
            graph.smiles,
            parent_id=graph.molecule_id,
            maximum=max_deletions_per_parent,
        )
        if len(actions) != max_deletions_per_parent:
            continue
        selected.append((index, graph, actions))
        if len(selected) == source_count:
            break
    if len(selected) != source_count:
        raise TasteGNNStageError(
            f"Taste T4 requires 16 eligible source parents; found {len(selected)}"
        )
    selected_graphs = [row[1] for row in selected]
    batched = oracle.predict_proba(selected_graphs, batch_size=batch_size)
    singles = np.vstack([oracle.predict_proba([graph], batch_size=1) for graph in selected_graphs])
    if not np.isfinite(batched).all() or not np.allclose(
        batched, singles, rtol=0.0, atol=1e-7
    ):
        raise TasteGNNStageError("Taste T4 batch/single probabilities differ")

    full_parent = enumerate_connected_hard_deletions("CC", "CC")
    invalid = enumerate_connected_hard_deletions("CC", "not-a-smiles")
    if not full_parent or any(outcome.valid for outcome in full_parent):
        raise TasteGNNStageError("Taste T4 full-parent deletion did not fail closed")
    if invalid:
        raise TasteGNNStageError("Taste T4 invalid deletion did not fail closed")

    featurizer = MolecularGraphFeaturizer(feature_schema)
    pair_graphs: list[MolecularGraphData] = []
    pair_index: list[tuple[int, str, HardDeletionOutcome]] = []
    deletion_counts: list[int] = []
    for cohort_position, (_source_index, graph, actions) in enumerate(selected):
        deletion_counts.append(len(actions))
        for action_index, (_fragment, outcome) in enumerate(actions):
            pair_graphs.extend(
                (
                    graph,
                    _graph_from_smiles(
                        featurizer,
                        str(outcome.residual_smiles),
                        f"t4-residual-{cohort_position}-{action_index}",
                    ),
                )
            )
            pair_index.append((cohort_position, _fragment, outcome))
    pair_records = oracle.predict_records(pair_graphs, batch_size=batch_size)
    if len(pair_records) != 2 * len(pair_index):
        raise TasteGNNStageError("Taste T4 pair prediction count changed")
    pair_batched = oracle.predict_proba(pair_graphs, batch_size=batch_size)
    pair_singles = np.vstack(
        [oracle.predict_proba([graph], batch_size=1) for graph in pair_graphs]
    )
    if (
        pair_batched.shape != (len(pair_graphs), NUM_CLASSES)
        or pair_singles.shape != pair_batched.shape
        or not np.isfinite(pair_batched).all()
        or not np.isfinite(pair_singles).all()
        or not np.allclose(pair_batched, pair_singles, rtol=0.0, atol=1e-7)
    ):
        raise TasteGNNStageError(
            "Taste T4 deletion-pair batch/single probabilities differ"
        )
    semantics: list[CounterfactualRecord] = []
    for offset, (_cohort_position, _fragment, _outcome) in enumerate(pair_index):
        before = pair_records[2 * offset]
        after = pair_records[2 * offset + 1]
        _validate_probability_record(before, checkpoint_id=oracle.checkpoint_id)
        _validate_probability_record(after, checkpoint_id=oracle.checkpoint_id)
        if int(before["predicted_label"]) != SOURCE_LABEL:
            raise TasteGNNStageError("Taste T4 deletion pair escaped the source cohort")
        semantics.append(
            compute_counterfactual_semantics(
                source_label=SOURCE_LABEL,
                pred_before=before["predicted_label"],
                pred_after=after["predicted_label"],
                probabilities_before=before["probabilities"],
                probabilities_after=after["probabilities"],
            )
        )
    distribution = destination_distribution(
        semantics,
        source_label=SOURCE_LABEL,
        num_classes=NUM_CLASSES,
        label_map=INT_LABEL_MAP,
    )
    transitions = distribution["overall"]["transitions"]
    if (
        transitions["1->0"]["count"] <= 0
        or transitions["1->2"]["count"] <= 0
    ):
        raise TasteGNNStageError(
            "Taste T4 requires observed strict flips from Sweet to both Bitter and Tasteless"
        )
    drops = np.asarray([record.cf_drop for record in semantics], dtype=np.float64)
    return {
        "schema_version": "tastemolnet_t4_bounded_oracle_smoke_metrics_v1",
        "status": "PASS",
        "selected_count": len(selected),
        "selection_rule": (
            "calibration_source_order_first_16_true_source_and_predicted_source_"
            "with_exactly_four_connected_one_or_two_atom_deletions"
        ),
        "selected_cohort_digest": _cohort_digest(selected),
        "calibration_graph_count": len(dataset),
        "correctly_predicted_source_scanned": correctly_predicted_source_count,
        "all_selected_true_source": True,
        "all_selected_predicted_source": True,
        "all_selected_have_four_connected_deletions": all(
            count == max_deletions_per_parent for count in deletion_counts
        ),
        "parent_deletion_counts_by_position": deletion_counts,
        "valid_deletion_count": len(semantics),
        "batch_examples": len(selected),
        "batch_single_max_abs_difference": float(
            max(
                np.max(np.abs(batched - singles)),
                np.max(np.abs(pair_batched - pair_singles)),
            )
        ),
        "batch_single_graph_count": len(selected_graphs) + len(pair_graphs),
        "all_three_probabilities_validated": True,
        "checkpoint_load_count": 1,
        "checkpoint_id": oracle.checkpoint_id,
        "temperature": float(oracle.temperature),
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "strict_flip": "pred_before == 1 and pred_after != 1",
        "destination_distribution": distribution,
        "strict_flip_to_bitter_observed": True,
        "strict_flip_to_tasteless_observed": True,
        "cf_drop": {
            "mean": float(drops.mean()),
            "minimum": float(drops.min()),
            "maximum": float(drops.max()),
        },
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
        "per_example_predictions_written": False,
        "smiles_written": False,
        "molecule_identifiers_written": False,
    }


def run_adaptive_oracle_smoke(
    *,
    dataset: MolecularGraphDataset,
    oracle: GNNOracle,
    feature_schema: MolecularFeatureSchema,
    batch_size: int,
) -> dict[str, Any]:
    """Run the authorized calibration-only adaptive T4 search.

    The search keeps calibration source order fixed and expands only the number
    of eligible source parents and the per-parent connected-deletion cap.  It
    never opens another split and never uses destination diversity as a PASS
    requirement.
    """

    if type(batch_size) is not int or batch_size <= 0:
        raise TasteGNNStageError("Taste T4 batch size must be a native positive integer")
    graphs = [dataset[index] for index in range(len(dataset))]
    records = oracle.predict_records(graphs, batch_size=batch_size)
    if len(records) != len(graphs):
        raise TasteGNNStageError("oracle prediction count differs from calibration cache")
    for record in records:
        _validate_probability_record(record, checkpoint_id=oracle.checkpoint_id)

    eligible = [
        (index, graph, record)
        for index, (graph, record) in enumerate(zip(graphs, records, strict=True))
        if int(graph.y) == SOURCE_LABEL
        and int(record["predicted_label"]) == SOURCE_LABEL
    ]
    if not eligible:
        raise TasteGNNStageError(
            "Taste T4 found no calibration parent with true_label=1 and pred_before=1"
        )

    full_parent = enumerate_connected_hard_deletions("CC", "CC")
    invalid = enumerate_connected_hard_deletions("CC", "not-a-smiles")
    if not full_parent or any(outcome.valid for outcome in full_parent):
        raise TasteGNNStageError("Taste T4 full-parent deletion did not fail closed")
    if invalid:
        raise TasteGNNStageError("Taste T4 invalid deletion did not fail closed")

    featurizer = MolecularGraphFeaturizer(feature_schema)
    round_summaries: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    terminal_selected: list[
        tuple[int, MolecularGraphData, list[tuple[str, HardDeletionOutcome]]]
    ] = []
    terminal_semantics: list[CounterfactualRecord] = []
    terminal_distribution: dict[str, Any] | None = None
    terminal_deletion_counts: list[int] = []
    terminal_batch_single_max = 0.0
    terminal_batch_single_graph_count = 0

    for round_index, (parent_limit, deletion_cap) in enumerate(
        T4_ADAPTIVE_SEARCH_SCHEDULE, start=1
    ):
        selected_source = eligible[:parent_limit]
        selected: list[
            tuple[int, MolecularGraphData, list[tuple[str, HardDeletionOutcome]]]
        ] = []
        for source_index, graph, _record in selected_source:
            actions = _real_connected_deletions(
                graph.smiles,
                parent_id=graph.molecule_id,
                maximum=deletion_cap,
            )
            selected.append((source_index, graph, actions))

        selected_graphs = [row[1] for row in selected]
        parent_batched = np.asarray(
            [row[2]["probabilities"] for row in selected_source], dtype=np.float64
        )
        parent_singles = np.vstack(
            [oracle.predict_proba([graph], batch_size=1) for graph in selected_graphs]
        )
        if (
            parent_batched.shape != (len(selected_graphs), NUM_CLASSES)
            or parent_singles.shape != parent_batched.shape
            or not np.isfinite(parent_batched).all()
            or not np.isfinite(parent_singles).all()
            or not np.allclose(parent_batched, parent_singles, rtol=0.0, atol=1e-7)
        ):
            raise TasteGNNStageError("Taste T4 batch/single probabilities differ")

        residual_graphs: list[MolecularGraphData] = []
        residual_parent_positions: list[int] = []
        deletion_counts: list[int] = []
        for cohort_position, (_source_index, _graph, actions) in enumerate(selected):
            deletion_counts.append(len(actions))
            for action_index, (_fragment, outcome) in enumerate(actions):
                residual_graphs.append(
                    _graph_from_smiles(
                        featurizer,
                        str(outcome.residual_smiles),
                        (
                            f"t4-adaptive-r{round_index}-p{cohort_position}-"
                            f"a{action_index}"
                        ),
                    )
                )
                residual_parent_positions.append(cohort_position)

        # An early prefix may contain no usable connected deletions even when a
        # later, larger prefix does.  Record the empty round and continue the
        # authorized schedule instead of asking the oracle to predict an empty
        # graph sequence (which is intentionally rejected by GNNOracle).
        residual_records = (
            oracle.predict_records(residual_graphs, batch_size=batch_size)
            if residual_graphs
            else []
        )
        if len(residual_records) != len(residual_graphs):
            raise TasteGNNStageError("Taste T4 residual prediction count changed")
        for record in residual_records:
            _validate_probability_record(record, checkpoint_id=oracle.checkpoint_id)

        residual_sample_count = len(residual_graphs)
        residual_batch_single_max = 0.0
        if residual_sample_count:
            residual_sample = residual_graphs[:residual_sample_count]
            residual_batched = np.asarray(
                [
                    residual_records[index]["probabilities"]
                    for index in range(residual_sample_count)
                ],
                dtype=np.float64,
            )
            residual_singles = np.vstack(
                [
                    oracle.predict_proba([graph], batch_size=1)
                    for graph in residual_sample
                ]
            )
            if (
                residual_batched.shape != (residual_sample_count, NUM_CLASSES)
                or residual_singles.shape != residual_batched.shape
                or not np.isfinite(residual_batched).all()
                or not np.isfinite(residual_singles).all()
                or not np.allclose(
                    residual_batched, residual_singles, rtol=0.0, atol=1e-7
                )
            ):
                raise TasteGNNStageError(
                    "Taste T4 deletion batch/single probabilities differ"
                )
            residual_batch_single_max = float(
                np.max(np.abs(residual_batched - residual_singles))
            )

        semantics: list[CounterfactualRecord] = []
        flipped_parent_positions: set[int] = set()
        for residual_position, after in enumerate(residual_records):
            cohort_position = residual_parent_positions[residual_position]
            before = selected_source[cohort_position][2]
            if int(before["predicted_label"]) != SOURCE_LABEL:
                raise TasteGNNStageError(
                    "Taste T4 deletion pair escaped the source cohort"
                )
            semantic = compute_counterfactual_semantics(
                source_label=SOURCE_LABEL,
                pred_before=before["predicted_label"],
                pred_after=after["predicted_label"],
                probabilities_before=before["probabilities"],
                probabilities_after=after["probabilities"],
            )
            semantics.append(semantic)
            if semantic.cf_flip:
                flipped_parent_positions.add(cohort_position)

        distribution = destination_distribution(
            semantics,
            source_label=SOURCE_LABEL,
            num_classes=NUM_CLASSES,
            label_map=INT_LABEL_MAP,
        )
        transitions = distribution["overall"]["transitions"]
        strict_flip_count = int(distribution["overall"]["total_strict_flips"])
        distinct_flipped_parent_count = len(flipped_parent_positions)
        round_summary = {
            "round": round_index,
            "parent_limit": parent_limit,
            "deletion_cap_per_parent": deletion_cap,
            "selected_count": len(selected),
            "valid_deletion_count": len(semantics),
            "strict_flip_count": strict_flip_count,
            "distinct_flipped_parent_count": distinct_flipped_parent_count,
            "destination_0_count": int(transitions["1->0"]["count"]),
            "destination_2_count": int(transitions["1->2"]["count"]),
            "gate_pass": (
                strict_flip_count >= T4_MIN_STRICT_FLIPS
                and distinct_flipped_parent_count >= T4_MIN_FLIPPED_PARENTS
            ),
        }
        round_summaries.append(round_summary)
        if round_summary["gate_pass"]:
            terminal = round_summary
            terminal_selected = selected
            terminal_semantics = semantics
            terminal_distribution = distribution
            terminal_deletion_counts = deletion_counts
            terminal_batch_single_max = float(
                max(
                    np.max(np.abs(parent_batched - parent_singles)),
                    residual_batch_single_max,
                )
            )
            terminal_batch_single_graph_count = (
                len(selected_graphs) + residual_sample_count
            )
            break

    if terminal is None or terminal_distribution is None:
        last = round_summaries[-1]
        raise TasteGNNStageError(
            "Taste T4 adaptive calibration search did not reach "
            f"strict_flips>={T4_MIN_STRICT_FLIPS} and "
            f"distinct_flipped_parents>={T4_MIN_FLIPPED_PARENTS}; "
            f"observed {last['strict_flip_count']} flips from "
            f"{last['distinct_flipped_parent_count']} parents"
        )

    transitions = terminal_distribution["overall"]["transitions"]
    destination_0_count = int(transitions["1->0"]["count"])
    destination_2_count = int(transitions["1->2"]["count"])
    observed_destinations = [
        destination
        for destination, count in ((0, destination_0_count), (2, destination_2_count))
        if count > 0
    ]
    diversity_status = (
        "DESTINATION_DIVERSITY_PASS"
        if len(observed_destinations) == 2
        else "DESTINATION_DIVERSITY_SINGLE_CLASS_WARNING"
    )
    drops = np.asarray(
        [record.cf_drop for record in terminal_semantics], dtype=np.float64
    )
    return {
        "schema_version": "tastemolnet_t4_adaptive_oracle_smoke_metrics_v1",
        "status": "PASS",
        "adaptive_calibration_search": True,
        "search_schedule": [
            {
                "round": index,
                "parent_limit": parent_limit,
                "deletion_cap_per_parent": deletion_cap,
            }
            for index, (parent_limit, deletion_cap) in enumerate(
                T4_ADAPTIVE_SEARCH_SCHEDULE, start=1
            )
        ],
        "rounds_executed": round_summaries,
        "terminal_round": terminal["round"],
        "selected_count": len(terminal_selected),
        "selection_rule": (
            "calibration_source_order_true_label_1_and_pred_before_1_with_"
            "adaptive_connected_deletion_caps"
        ),
        "selected_cohort_digest": _cohort_digest(terminal_selected),
        "calibration_graph_count": len(dataset),
        "correctly_predicted_source_scanned": len(eligible),
        "all_selected_true_source": True,
        "all_selected_predicted_source": True,
        "all_selected_have_connected_deletions": all(
            count > 0 for count in terminal_deletion_counts
        ),
        "at_least_one_connected_deletion": bool(terminal_semantics),
        "parent_deletion_counts_by_position": terminal_deletion_counts,
        "deletion_cap_per_parent": terminal["deletion_cap_per_parent"],
        "valid_deletion_count": len(terminal_semantics),
        "batch_examples": len(terminal_selected),
        "batch_single_max_abs_difference": terminal_batch_single_max,
        "batch_single_graph_count": terminal_batch_single_graph_count,
        "all_three_probabilities_validated": True,
        "all_three_logits_validated": True,
        "three_class_api_validated": True,
        "checkpoint_load_count": 1,
        "checkpoint_id": oracle.checkpoint_id,
        "temperature": float(oracle.temperature),
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "strict_flip": "pred_before == 1 and pred_after != 1",
        "strict_flip_count": terminal["strict_flip_count"],
        "minimum_strict_flip_count": T4_MIN_STRICT_FLIPS,
        "distinct_flipped_parent_count": terminal[
            "distinct_flipped_parent_count"
        ],
        "minimum_distinct_flipped_parent_count": T4_MIN_FLIPPED_PARENTS,
        "strict_flip_gate_pass": True,
        "destination_distribution": terminal_distribution,
        "destination_0_count": destination_0_count,
        "destination_2_count": destination_2_count,
        "observed_destination_labels": observed_destinations,
        "destination_diversity_status": diversity_status,
        "destination_diversity_single_class_warning": (
            diversity_status == "DESTINATION_DIVERSITY_SINGLE_CLASS_WARNING"
        ),
        "strict_flip_to_bitter_observed": destination_0_count > 0,
        "strict_flip_to_tasteless_observed": destination_2_count > 0,
        "cf_drop": {
            "mean": float(drops.mean()),
            "minimum": float(drops.min()),
            "maximum": float(drops.max()),
        },
        "empty_deletion_failed_closed": True,
        "invalid_deletion_failed_closed": True,
        "calibration_payload_loaded": True,
        "train_payload_loaded": False,
        "validation_payload_loaded": False,
        "test_payload_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "per_example_predictions_written": False,
        "smiles_written": False,
        "molecule_identifiers_written": False,
    }


def _run_t4_calibration_cache_smoke(
    *,
    t2_adoption: HeldT2PassAdoption,
    checkpoint_dir: str | Path,
    t3_gate_path: str | Path,
    graph_cache_root: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    downstream_policy_path: str | Path,
    base_policy_path: str | Path,
    gpu_uuid: str,
    physical_gpu_index: int = 1,
    device: str = "cuda:0",
    batch_size: int = 32,
    source_count: int = 16,
    max_deletions_per_parent: int = 4,
    oracle_factory: Callable[..., GNNOracle] | None = None,
) -> dict[str, Any]:
    if type(physical_gpu_index) is not int or physical_gpu_index != 1:
        raise TasteGNNStageError("T4 is frozen to physical GPU index 1")
    if type(gpu_uuid) is not str or not gpu_uuid.startswith("GPU-"):
        raise TasteGNNStageError("T4 requires the physical GPU1 UUID")
    if device != "cuda:0":
        raise TasteGNNStageError("T4 visible-device mapping must use cuda:0")
    _require_gpu1_environment(gpu_uuid=gpu_uuid)
    policy = load_tastemolnet_downstream_policy(
        downstream_policy_path, base_policy_path=base_policy_path
    )
    contract = policy.stage(T4_STAGE)
    if contract.get("mode") != "calibration_cache_only_bounded_oracle_smoke":
        raise TasteGNNStageError("T4 authority does not permit calibration-cache smoke")
    if type(batch_size) is not int or batch_size <= 0:
        raise TasteGNNStageError("T4 batch size must be a native positive integer")
    if (
        type(source_count) is not int
        or type(max_deletions_per_parent) is not int
        or source_count != contract["source_count"]
        or max_deletions_per_parent != contract["max_deletions_per_parent"]
        or max_deletions_per_parent != contract["minimum_deletions_per_parent"]
    ):
        raise TasteGNNStageError("T4 bounded cohort/deletion parameters changed")

    checkpoint_root = _physical_directory(checkpoint_dir, field="T2 checkpoint")
    t2_binding = _bind_t2_adoption_to_checkpoint(
        t2_adoption, checkpoint_root
    )
    cache_directory = _physical_directory(graph_cache_root, field="graph-cache root")
    execution_root = _physical_directory(
        EXECUTION_SOURCE_ROOT, field="loaded execution source root"
    )
    checkpoint = checkpoint_root.path
    unresolved_gate = Path(os.path.abspath(Path(t3_gate_path).expanduser()))
    if not unresolved_gate.is_absolute() or unresolved_gate.name != "gate.json":
        raise TasteGNNStageError("T4 predecessor must be the T3 gate.json")
    t3_output = _physical_directory(unresolved_gate.parent, field="T3 output planning authority")
    gate_path = t3_output.path / "gate.json"
    if gate_path != unresolved_gate:
        raise TasteGNNStageError("T4 predecessor gate path changed identity")
    forbidden_identities = _merge_identity_paths(
        checkpoint_root.identity_paths(),
        cache_directory.identity_paths(),
        t3_output.identity_paths(),
        execution_root.identity_paths(),
        policy.directory_identity_paths(),
    )
    output_plan = _plan_stage_output(
        artifact_root=artifact_root,
        output_dir=output_dir,
        basename_prefix="t4-oracle-smoke-",
        forbidden_paths=(
            checkpoint_root.path,
            cache_directory.path,
            t3_output.path,
            execution_root.path,
            t2_adoption.root,
            *policy.protected_paths(),
        ),
        forbidden_identity_paths=forbidden_identities,
    )
    # Create and retain the fresh empty leaf before taking the long-lived T3
    # authority.  The T4 leaf is a sibling of T3 under seed7; creating it later
    # would itself change T3's retained parent metadata and mask swap history.
    output_root = output_plan.create()
    t3_output.close()
    t3_output = _physical_directory(unresolved_gate.parent, field="T3 output")
    t3 = verify_stage_output(t3_output)
    t3_before = _complete_directory_snapshot(t3_output, label="complete T3 output")
    t3_gate = t3["gate"]
    if t3_gate.get("stage") != T3_STAGE or t3_gate.get("marker") != T3_MARKER:
        raise TasteGNNStageError("T4 predecessor is not a T3 calibration PASS")
    required_t3_semantics = {
        "t2_science_bundle_verified": True,
        "existing_fit_adopted": True,
        "temperature_refit_performed": False,
        "test_loaded": False,
    }
    if any(
        type(t3_gate.get(key)) is not type(expected)
        or t3_gate.get(key) != expected
        for key, expected in required_t3_semantics.items()
    ):
        raise TasteGNNStageError("T4 predecessor T3 science semantics changed")
    if t3_gate.get("downstream_policy_sha256") != policy.file_sha256:
        raise TasteGNNStageError("T3 and T4 downstream-policy authorities differ")
    if t3_gate.get("t2_adoption_binding") != t2_binding:
        raise TasteGNNStageError(
            "T3 and T4 fresh T2 adoption authorities differ"
        )

    checkpoint_root.verify(label="T2 checkpoint before T4 verification")
    before = _checkpoint_stat_snapshot(checkpoint_root)
    expected_stat_inventory = _hex(
        t3_gate.get("checkpoint_stat_inventory_sha256"),
        field="t3_gate.checkpoint_stat_inventory_sha256",
    )
    if before["inventory_sha256"] != expected_stat_inventory:
        raise TasteGNNStageError("T4 checkpoint stat inventory differs from T3")
    expected_full_inventory = _hex(
        t3_gate.get("checkpoint_inventory_sha256"),
        field="t3_gate.checkpoint_inventory_sha256",
    )
    expected_sha256s_sha256 = _hex(
        t3_gate.get("checkpoint_sha256s_sha256"),
        field="t3_gate.checkpoint_sha256s_sha256",
    )
    if _sha256_at(
        checkpoint_root, "sha256sums.txt", label="checkpoint SHA inventory"
    ) != expected_sha256s_sha256:
        raise TasteGNNStageError("T4 checkpoint SHA inventory differs from T3")
    checkpoint_hashes = _checkpoint_sha_inventory(checkpoint_root)
    for name in T4_CHECKPOINT_PAYLOAD_FILES:
        if name not in checkpoint_hashes or _sha256_at(
            checkpoint_root, name, label=f"T4 checkpoint payload {name}"
        ) != checkpoint_hashes[name]:
            raise TasteGNNStageError(f"T4 checkpoint payload hash mismatch: {name}")
    checkpoint_manifest_before = _checkpoint_manifest_snapshot(checkpoint_root)
    if checkpoint_manifest_before["inventory_sha256"] != expected_full_inventory:
        raise TasteGNNStageError("T4 full checkpoint inventory differs from T3")
    checkpoint_selected_before = _selected_file_snapshot(
        checkpoint_root,
        T4_CHECKPOINT_PAYLOAD_FILES,
        label="T4 selected checkpoint payloads",
    )
    checkpoint_root.verify(label="T2 checkpoint after T4 selective verification")
    model_card = _read_json_at(checkpoint_root, "model_card.json", label="model card")
    validate_taste_model_card(model_card)
    checkpoint_id = str(model_card.get("checkpoint_id") or "")
    if (
        t3_gate.get("checkpoint_dir") != str(checkpoint)
        or checkpoint_id != t3_gate.get("checkpoint_id")
        or checkpoint_id
        != _sha256_at(checkpoint_root, "model.pt", label="selected model.pt")
    ):
        raise TasteGNNStageError("T4 checkpoint differs from adopted T3 oracle")
    feature_schema = MolecularFeatureSchema.from_dict(
        _read_json_at(checkpoint_root, "feature_schema.json", label="feature schema")
    )
    manifest_sha256 = _hex(
        model_card.get("graph_cache_manifest_sha256"),
        field="model_card.graph_cache_manifest_sha256",
    )
    cache_before = _selected_file_snapshot(
        cache_directory,
        ("manifest.json", "calibration.pt"),
        label="T4 calibration-cache authority",
    )
    dataset, data_access = _load_calibration_cache(
        graph_cache_root=cache_directory,
        expected_manifest_sha256=manifest_sha256,
        feature_schema=feature_schema,
    )
    cache_directory.verify(label="graph-cache root after calibration load")
    checkpoint_root.verify(label="T2 checkpoint before oracle load")
    expected_temperature = _finite(
        _read_json_at(
            checkpoint_root,
            "temperature_scaling.json",
            label="temperature scaling",
        ).get("temperature"),
        field="temperature",
        positive=True,
    )
    if oracle_factory is None:
        oracle = _load_gnn_oracle_anchored(
            checkpoint_root,
            feature_schema=feature_schema,
            device=device,
            batch_size=batch_size,
        )
    else:
        oracle = oracle_factory(
            checkpoint_root,
            device=device,
            batch_size=batch_size,
            verify_hashes=False,
            require_taste_closure=True,
        )
    checkpoint_root.verify(label="T2 checkpoint after oracle load")
    if (
        oracle.checkpoint_id != checkpoint_id
        or oracle.backbone != "gine"
        or type(oracle.num_classes) is not int
        or oracle.num_classes != NUM_CLASSES
        or type(oracle.source_label) is not int
        or oracle.source_label != SOURCE_LABEL
        or not math.isfinite(float(oracle.temperature))
        or float(oracle.temperature) <= 0.0
        or not math.isclose(
            float(oracle.temperature), expected_temperature, rel_tol=0.0, abs_tol=0.0
        )
    ):
        raise TasteGNNStageError("loaded T4 oracle contract differs from T3")
    smoke = run_bounded_oracle_smoke(
        dataset=dataset,
        oracle=oracle,
        feature_schema=feature_schema,
        batch_size=batch_size,
        source_count=source_count,
        max_deletions_per_parent=max_deletions_per_parent,
    )
    cache_directory.verify(label="graph-cache root after T4 smoke")
    checkpoint_root.verify(label="T2 checkpoint after T4 smoke")
    after = _checkpoint_stat_snapshot(checkpoint_root)
    if before != after:
        raise TasteGNNStageError("T4 immutable checkpoint changed during smoke")
    policy_binding = policy.evidence(stage=T4_STAGE)
    provenance = {
        "schema_version": "tastemolnet_t4_oracle_provenance_v1",
        "dataset": "tastemolnet",
        "checkpoint_dir": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "checkpoint_inventory_sha256": expected_full_inventory,
        "checkpoint_stat_inventory_sha256": before["inventory_sha256"],
        "checkpoint_sha256s_sha256": expected_sha256s_sha256,
        "checkpoint_payload_files_opened": list(T4_CHECKPOINT_PAYLOAD_FILES),
        "checkpoint_csv_payload_opened": False,
        "selected_inference_asset": "model.pt",
        "model_sha256": _sha256_at(
            checkpoint_root, "model.pt", label="selected model.pt"
        ),
        "temperature_scaling_sha256": _sha256_at(
            checkpoint_root,
            "temperature_scaling.json",
            label="temperature scaling",
        ),
        "config_sha256": _sha256_at(
            checkpoint_root, "config.yaml", label="config"
        ),
        "feature_schema_sha256": _sha256_at(
            checkpoint_root, "feature_schema.json", label="feature schema"
        ),
        "physical_gpu_index": physical_gpu_index,
        "gpu_uuid": gpu_uuid,
        "visible_device": device,
        "cuda_visible_devices": "1",
        "checkpoint_load_count": 1,
        "rf_oracle_used": False,
        "test_loaded": False,
        "t2_adoption_binding": t2_binding,
    }
    gate = {
        "schema_version": "tastemolnet_main_stage_gate_v1",
        "stage": T4_STAGE,
        "status": "PASS",
        "marker": T4_MARKER,
        "depends_on": [T3_STAGE],
        "t3_gate_sha256": _sha256_at(t3_output, "gate.json", label="T3 gate"),
        "t2_adoption_binding": t2_binding,
        "checkpoint_dir": str(checkpoint),
        "checkpoint_id": checkpoint_id,
        "checkpoint_inventory_sha256": expected_full_inventory,
        "checkpoint_stat_inventory_sha256": before["inventory_sha256"],
        "checkpoint_sha256s_sha256": expected_sha256s_sha256,
        "physical_gpu_index": physical_gpu_index,
        "gpu_uuid": gpu_uuid,
        "visible_device": device,
        "cuda_visible_devices": "1",
        "downstream_policy_sha256": policy.file_sha256,
        "selected_count": smoke["selected_count"],
        "calibration_payload_loaded": True,
        "test_loaded": False,
        "per_example_predictions_written": False,
    }
    checkpoint_root.verify(label="T2 checkpoint before T4 publication")
    _validated_t2_adoption_binding(t2_adoption)
    cache_directory.verify(label="graph-cache root before T4 publication")
    t3_output.verify(label="T3 output before T4 publication")
    policy.verify_authorities()
    execution_root.verify(label="loaded execution source before T4 publication")
    if before != _checkpoint_stat_snapshot(checkpoint_root):
        raise TasteGNNStageError("T4 immutable checkpoint drifted before publication")
    if checkpoint_manifest_before != _checkpoint_manifest_snapshot(checkpoint_root):
        raise TasteGNNStageError("T4 full checkpoint inventory drifted before publication")
    if checkpoint_selected_before != _selected_file_snapshot(
        checkpoint_root,
        T4_CHECKPOINT_PAYLOAD_FILES,
        label="T4 selected checkpoint payloads before publication",
    ):
        raise TasteGNNStageError("T4 selected checkpoint payloads drifted before publication")
    if cache_before != _selected_file_snapshot(
        cache_directory,
        ("manifest.json", "calibration.pt"),
        label="T4 calibration-cache before publication",
    ):
        raise TasteGNNStageError("T4 calibration-cache authority drifted before publication")
    if t3_before != _complete_directory_snapshot(
        t3_output, label="complete T3 output before publication"
    ):
        raise TasteGNNStageError("complete T3 output drifted before publication")
    prepared_output = _prepare_stage_output(
        output_root,
        documents={
            "oracle_smoke.json": smoke,
            "oracle_provenance.json": provenance,
            "data_access_manifest.json": data_access,
            "policy_binding.json": policy_binding,
            "gate.json": gate,
        },
        marker=T4_MARKER,
    )
    # Complete the retained closure while the output is still non-terminal.
    checkpoint_root.verify(label="T2 checkpoint after T4 preparation")
    _validated_t2_adoption_binding(t2_adoption)
    if before != _checkpoint_stat_snapshot(checkpoint_root):
        raise TasteGNNStageError(
            "T4 checkpoint stat inventory drifted during preparation"
        )
    if checkpoint_manifest_before != _checkpoint_manifest_snapshot(checkpoint_root):
        raise TasteGNNStageError(
            "T4 full checkpoint inventory drifted during preparation"
        )
    if checkpoint_selected_before != _selected_file_snapshot(
        checkpoint_root,
        T4_CHECKPOINT_PAYLOAD_FILES,
        label="T4 selected checkpoint payloads after preparation",
    ):
        raise TasteGNNStageError(
            "T4 selected checkpoint payloads drifted during preparation"
        )
    if cache_before != _selected_file_snapshot(
        cache_directory,
        ("manifest.json", "calibration.pt"),
        label="T4 calibration-cache after preparation",
    ):
        raise TasteGNNStageError(
            "T4 calibration-cache authority drifted during preparation"
        )
    verify_stage_output(t3_output)
    if t3_before != _complete_directory_snapshot(
        t3_output, label="complete T3 output after preparation"
    ):
        raise TasteGNNStageError("complete T3 output drifted during preparation")
    policy.verify_authorities()
    execution_root.verify(label="loaded execution source after T4 preparation")
    try:
        def retained_t4_input_closure() -> None:
            if _validated_t2_adoption_binding(t2_adoption) != t2_binding:
                raise TasteGNNStageError(
                    "fresh T2 adoption binding drifted at T4 PASS boundary"
                )
            checkpoint_root.verify(label="T2 checkpoint at T4 PASS boundary")
            if before != _checkpoint_stat_snapshot(checkpoint_root):
                raise TasteGNNStageError(
                    "T4 checkpoint stat inventory drifted at PASS boundary"
                )
            if checkpoint_manifest_before != _checkpoint_manifest_snapshot(
                checkpoint_root
            ):
                raise TasteGNNStageError(
                    "T4 full checkpoint inventory drifted at PASS boundary"
                )
            if checkpoint_selected_before != _selected_file_snapshot(
                checkpoint_root,
                T4_CHECKPOINT_PAYLOAD_FILES,
                label="T4 selected checkpoint payloads at PASS boundary",
            ):
                raise TasteGNNStageError(
                    "T4 selected checkpoint payloads drifted at PASS boundary"
                )
            cache_directory.verify(label="graph-cache root at T4 PASS boundary")
            if cache_before != _selected_file_snapshot(
                cache_directory,
                ("manifest.json", "calibration.pt"),
                label="T4 calibration-cache at PASS boundary",
            ):
                raise TasteGNNStageError(
                    "T4 calibration-cache authority drifted at PASS boundary"
                )
            verify_stage_output(t3_output)
            if t3_before != _complete_directory_snapshot(
                t3_output, label="complete T3 output at PASS boundary"
            ):
                raise TasteGNNStageError(
                    "complete T3 output drifted at PASS boundary"
                )
            policy.verify_authorities()
            execution_root.verify(label="loaded execution source at T4 PASS boundary")

        _publish_prepared_stage_marker(
            prepared_output,
            retained_input_closure=retained_t4_input_closure,
        )
    finally:
        prepared_output.close()
    output_plan.close()
    checkpoint_root.close()
    cache_directory.close()
    t3_output.close()
    execution_root.close()
    policy.close()
    return smoke


def run_t4_calibration_cache_smoke(
    *,
    t2_adoption_root: str | Path,
    t2_adoption_gate_sha256: str,
    t2_adoption_receipt_sha256: str,
    t2_source_evidence_sha256: str,
    checkpoint_dir: str | Path,
    t3_gate_path: str | Path,
    graph_cache_root: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    downstream_policy_path: str | Path,
    base_policy_path: str | Path,
    gpu_uuid: str,
    physical_gpu_index: int = 1,
    device: str = "cuda:0",
    batch_size: int = 32,
    source_count: int = 16,
    max_deletions_per_parent: int = 4,
    oracle_factory: Callable[..., GNNOracle] | None = None,
) -> dict[str, Any]:
    """Run T4 only while the same fresh T2 adoption as T3 remains held."""

    # Keep the cheap direct-child GPU binding ahead of the expensive historical
    # source audit, but never create/open a stage output before T2 is held.
    if type(physical_gpu_index) is not int or physical_gpu_index != 1:
        raise TasteGNNStageError("Taste T4 is frozen to physical GPU index 1")
    if type(gpu_uuid) is not str or not gpu_uuid.startswith("GPU-"):
        raise TasteGNNStageError("Taste T4 requires the physical GPU1 UUID")
    if device != "cuda:0":
        raise TasteGNNStageError("Taste T4 visible-device mapping must use cuda:0")
    _require_gpu1_environment(gpu_uuid=gpu_uuid)
    try:
        authority = hold_t2_gine_pass_adoption(
            t2_adoption_root,
            expected_gate_sha256=t2_adoption_gate_sha256,
            expected_receipt_sha256=t2_adoption_receipt_sha256,
            expected_source_evidence_sha256=t2_source_evidence_sha256,
        )
    except (T2PassAdoptionError, OSError, ValueError) as exc:
        raise TasteGNNStageError(
            f"T4 requires the fresh T2 GINE PASS adoption: {exc}"
        ) from exc
    try:
        return _run_t4_calibration_cache_smoke(
            t2_adoption=authority,
            checkpoint_dir=checkpoint_dir,
            t3_gate_path=t3_gate_path,
            graph_cache_root=graph_cache_root,
            artifact_root=artifact_root,
            output_dir=output_dir,
            downstream_policy_path=downstream_policy_path,
            base_policy_path=base_policy_path,
            gpu_uuid=gpu_uuid,
            physical_gpu_index=physical_gpu_index,
            device=device,
            batch_size=batch_size,
            source_count=source_count,
            max_deletions_per_parent=max_deletions_per_parent,
            oracle_factory=oracle_factory,
        )
    finally:
        authority.close()


__all__ = [
    "LABEL_MAP",
    "NUM_CLASSES",
    "SOURCE_LABEL",
    "T3_MARKER",
    "T3_STAGE",
    "T4_MARKER",
    "T4_STAGE",
    "T4_ADAPTIVE_SEARCH_SCHEDULE",
    "T4_MIN_FLIPPED_PARENTS",
    "T4_MIN_STRICT_FLIPS",
    "HELD_STAGE_EVIDENCE_KEYS",
    "HeldTasteCheckpointBundle",
    "HeldTasteStageOutput",
    "TasteGNNStageError",
    "hold_taste_stage_output",
    "hold_taste_checkpoint_bundle",
    "run_adaptive_oracle_smoke",
    "run_bounded_oracle_smoke",
    "run_t3_existing_fit_adoption",
    "run_t4_calibration_cache_smoke",
    "validate_existing_temperature_fit",
    "validate_taste_model_card",
    "verify_stage_output",
]
