"""Release gate and retained-input closure for TasteMolNet T8 GlobalGCE.

This module is an execution boundary, never a scheduler.  The managed AutoDL
controller owns process generation, storage reservation, and the exclusive
physical-GPU2 lease.  T8 only consumes the injected held child authority and
keeps every scientific input open until the final PASS commit.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Protocol

from src.baselines.globalgce_bace_native_rules import (
    OFFICIAL_GLOBALGCE_COMMIT,
    validate_official_globalgce_root,
)
from src.baselines.tastemolnet_globalgce_smoke import (
    DATASET,
    GINE_PAYLOAD_FILES,
    MANAGED_TASK_ID,
    MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA,
    MANAGED_V2_PROTOCOL,
    MANAGED_V2_SOURCE_COMMIT,
    NUM_CLASSES,
    PHYSICAL_GPU_INDEX,
    SOURCE_LABEL,
    STAGE,
    TasteGlobalGCESmokeError,
)
from src.utils.retained_readonly_file import hold_readonly_file
from src.utils.tastemolnet_downstream_policy import (
    BASE_POLICY_FILE_SHA256,
    DOWNSTREAM_POLICY_FILE_SHA256,
)


REPO_ROOT = Path(__file__).resolve(strict=True).parents[2]
RELEASE_CONFIG_PATH = (
    REPO_ROOT
    / "configs/autodl/tastemolnet_t8_globalgce_smoke_release_v1.json"
)
RELEASE_SCHEMA = "tastemolnet_t8_globalgce_smoke_release_v1"
DISABLED_RELEASE_STATE = (
    "RELEASE_DISABLED_PENDING_MANAGED_V2_AUTHORITY_AND_TARGET_FS_PREFLIGHT"
)
RELEASED_STATE = "RELEASED_BY_MANAGED_T8_EXECUTION_AUTHORITY"


class TasteT8ExternalManagedV2Authority(Protocol):
    """Narrow held boundary supplied by a separately reviewed controller adapter.

    This repository deliberately provides no implementation.  In particular,
    the worker cannot construct this authority from its attempt manifest or raw
    evidence and cannot substitute a managed-v1 receipt holder.
    """

    def revalidate_t8_managed_v2_authority(self) -> Mapping[str, Any]:
        ...

    def revalidate_t8_official_startup_authority(self) -> Mapping[str, Any]:
        """Return API/import expectations captured outside the worker."""

        ...

RELEASE_KEYS = frozenset(
    {
        "schema_version",
        "release_enabled",
        "release_state",
        "implementation_commit",
        "implementation_tree",
        "critical_blobs_sha256",
        "t2_adoption_root",
        "t2_gate_sha256",
        "t2_receipt_sha256",
        "t2_source_evidence_sha256",
        "t3_output_root",
        "t3_gate_sha256",
        "t3_root_inventory_sha256",
        "t4_output_root",
        "t4_gate_sha256",
        "t4_root_inventory_sha256",
        "checkpoint_dir",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "official_root",
        "official_commit",
        "official_source_inventory_sha256",
        "downstream_policy_sha256",
        "base_policy_sha256",
        "state_parent",
        "output_parent",
        "gpu_index",
    }
)
RELEASE_PIN_FIELDS = RELEASE_KEYS - {
    "schema_version",
    "release_enabled",
    "release_state",
    "gpu_index",
}
CRITICAL_BLOBS = frozenset(
    {
        "scripts/run_tastemolnet_globalgce_smoke.py",
        "scripts/slurm/run_tastemolnet_globalgce_smoke.sh",
        "src/baselines/globalgce_frozen_gine_bridge.py",
        "src/baselines/globalgce_mutagenicity_adapter.py",
        "src/baselines/globalgce_resumable.py",
        "src/baselines/tastemolnet_globalgce_smoke.py",
        "src/baselines/tastemolnet_multiclass_adapters.py",
        "src/eval/tastemolnet_gnn_stages.py",
        "src/oracles/gnn_oracle.py",
        "src/utils/managed_execution_v2.py",
        "src/utils/process_identity_v2.py",
        "src/utils/retained_output_directory.py",
        "src/utils/retained_readonly_file.py",
        "src/utils/terminal_publisher_v2.py",
        "src/utils/tastemolnet_downstream_policy.py",
        "src/utils/tastemolnet_gine_pass_adoption_v1.py",
        "src/utils/tastemolnet_t8_globalgce_release.py",
        "src/utils/tastemolnet_t8_managed_v2.py",
    }
)
OFFICIAL_RUNTIME_FILES = (
    "src/main.py",
    "src/models/GTGNN.py",
    "src/models/GlobalGCE.py",
    "src/models/models_utils.py",
    "src/models/fsg.py",
    "src/data/data_preprocess.py",
    "src/data/dataset.py",
)

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class TasteT8ReleaseDisabled(TasteGlobalGCESmokeError):
    """The checked-in implementation is intentionally not executable yet."""


def _json(data: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 {field} is malformed JSON") from exc
    if type(value) is not dict:
        raise TasteGlobalGCESmokeError(f"T8 {field} must be one JSON object")
    return value


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, *, field: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise TasteGlobalGCESmokeError(f"T8 {field} must be one lowercase SHA-256")
    return value


def _git_oid(value: Any, *, field: str) -> str:
    if type(value) is not str or _HEX40.fullmatch(value) is None:
        raise TasteGlobalGCESmokeError(f"T8 {field} must be one full Git SHA-1")
    return value


def _absolute(value: Any, *, field: str, exists: bool = True) -> Path:
    if type(value) is not str or not value or "\0" in value:
        raise TasteGlobalGCESmokeError(f"T8 {field} is not one exact path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGlobalGCESmokeError(
            f"T8 {field} must be normalized and absolute"
        )
    try:
        resolved = path.resolve(strict=exists)
    except OSError as exc:
        raise TasteGlobalGCESmokeError(f"T8 {field} cannot be resolved") from exc
    if resolved != path:
        raise TasteGlobalGCESmokeError(f"T8 {field} contains a symlink or alias")
    return path


def _lexical_absolute(value: Any, *, field: str) -> Path:
    """Validate a manifest path without touching the named filesystem object."""

    if type(value) is not str or not value or "\0" in value:
        raise TasteGlobalGCESmokeError(f"T8 {field} is not one exact path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteGlobalGCESmokeError(
            f"T8 {field} must be lexically normalized and absolute"
        )
    return path


def _native_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TasteGlobalGCESmokeError(f"T8 {field} is not a native integer")
    return value


def assert_execution_released() -> dict[str, Any]:
    """Reject the checked-in candidate before any caller-selected science input."""

    release = _json(RELEASE_CONFIG_PATH.read_bytes(), field="release config")
    if set(release) != RELEASE_KEYS:
        raise TasteGlobalGCESmokeError("TASTE_T8_RELEASE_CONFIG_KEYS_CHANGED")
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or type(release.get("release_enabled")) is not bool
        or type(release.get("release_state")) is not str
        or type(release.get("gpu_index")) is not int
        or release.get("gpu_index") != PHYSICAL_GPU_INDEX
    ):
        raise TasteGlobalGCESmokeError("TASTE_T8_RELEASE_CONFIG_INVALID")
    if release["release_enabled"] is not True:
        if release["release_state"] != DISABLED_RELEASE_STATE or any(
            release.get(field) is not None for field in RELEASE_PIN_FIELDS
        ):
            raise TasteGlobalGCESmokeError(
                "TASTE_T8_DISABLED_RELEASE_CONFIG_DRIFTED"
            )
        raise TasteT8ReleaseDisabled("TASTE_T8_GLOBALGCE_EXECUTION_NOT_RELEASED")
    if release["release_state"] != RELEASED_STATE:
        raise TasteGlobalGCESmokeError("TASTE_T8_RELEASE_STATE_INVALID")
    _git_oid(release.get("implementation_commit"), field="implementation commit")
    _git_oid(release.get("implementation_tree"), field="implementation tree")
    for field in (
        "t2_gate_sha256",
        "t2_receipt_sha256",
        "t2_source_evidence_sha256",
        "t3_gate_sha256",
        "t3_root_inventory_sha256",
        "t4_gate_sha256",
        "t4_root_inventory_sha256",
        "checkpoint_id",
        "checkpoint_inventory_sha256",
        "checkpoint_stat_inventory_sha256",
        "checkpoint_sha256s_sha256",
        "official_source_inventory_sha256",
        "downstream_policy_sha256",
        "base_policy_sha256",
    ):
        _sha(release.get(field), field=field)
    for field in (
        "t2_adoption_root",
        "t3_output_root",
        "t4_output_root",
        "checkpoint_dir",
        "official_root",
        "state_parent",
        "output_parent",
    ):
        _absolute(release.get(field), field=field)
    if release.get("official_commit") != OFFICIAL_GLOBALGCE_COMMIT:
        raise TasteGlobalGCESmokeError("T8 official GlobalGCE commit pin changed")
    if (
        release.get("downstream_policy_sha256") != DOWNSTREAM_POLICY_FILE_SHA256
        or release.get("base_policy_sha256") != BASE_POLICY_FILE_SHA256
    ):
        raise TasteGlobalGCESmokeError("T8 policy release pins changed")
    blobs = release.get("critical_blobs_sha256")
    if type(blobs) is not dict or set(blobs) != CRITICAL_BLOBS:
        raise TasteGlobalGCESmokeError("T8 critical blob inventory changed")
    for relative, digest in blobs.items():
        _sha(digest, field=f"critical blob {relative}")
    return release


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(root),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CEILING_DIRECTORIES": str(root.parent),
        },
    )
    return completed.stdout.strip()


def _reject_hidden_index_flags(root: Path) -> None:
    records = _git_output(root, "ls-files", "-v", "-z")
    for record in records.split("\0"):
        if not record:
            continue
        if len(record) < 3 or record[1] != " ":
            raise TasteGlobalGCESmokeError("T8 Git index inventory is malformed")
        if record[0] == "S" or record[0].islower():
            raise TasteGlobalGCESmokeError(
                "T8 checkout has skip-worktree/assume-unchanged entries"
            )


def verify_execution_checkout(release: Mapping[str, Any]) -> dict[str, str]:
    _reject_hidden_index_flags(REPO_ROOT)
    if _git_output(
        REPO_ROOT,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored=matching",
    ):
        raise TasteGlobalGCESmokeError("T8 execution checkout is not fully clean")
    lineage = _git_output(REPO_ROOT, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if len(lineage) != 2 or lineage[1] != release["implementation_commit"]:
        raise TasteGlobalGCESmokeError(
            "T8 release is not a one-parent implementation successor"
        )
    parent_tree = _git_output(REPO_ROOT, "rev-parse", f"{lineage[1]}^{{tree}}")
    if parent_tree != release["implementation_tree"]:
        raise TasteGlobalGCESmokeError("T8 implementation tree pin changed")
    changed = set(
        filter(
            None,
            _git_output(
                REPO_ROOT,
                "diff",
                "--name-only",
                "--no-renames",
                lineage[1],
                "HEAD",
            ).splitlines(),
        )
    )
    if changed != {
        "configs/autodl/tastemolnet_t8_globalgce_smoke_release_v1.json",
        "scripts/autodl/run_tastemolnet_globalgce_smoke.sh",
    }:
        raise TasteGlobalGCESmokeError("T8 release commit changed non-release files")
    for relative, expected in release["critical_blobs_sha256"].items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise TasteGlobalGCESmokeError(f"T8 critical blob changed: {relative}")
    wrapper = REPO_ROOT / "scripts/autodl/run_tastemolnet_globalgce_smoke.sh"
    assignments = [
        line.strip()
        for line in wrapper.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("TASTE_T8_GLOBALGCE_WRAPPER_RELEASED=")
    ]
    if assignments != ["TASTE_T8_GLOBALGCE_WRAPPER_RELEASED=1"]:
        raise TasteGlobalGCESmokeError("T8 AutoDL wrapper is not released")
    return {
        "commit": lineage[0],
        "tree": _git_output(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
    }


def _official_git_snapshot(root: Path) -> dict[str, str]:
    if _git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise TasteGlobalGCESmokeError("T8 official GlobalGCE checkout is not clean")
    runtime_paths = tuple(
        path
        for path in _git_output(
            root,
            "ls-files",
            "-z",
            "--",
            "src",
        ).split("\0")
        if path
    )
    if any(
        "__pycache__" in Path(path).parts
        or Path(path).suffix.lower() in {".pyc", ".pyo", ".so", ".dylib"}
        for path in runtime_paths
    ):
        raise TasteGlobalGCESmokeError(
            "T8 official GlobalGCE tracks forbidden runtime code artifacts"
        )
    ignored_candidates = tuple(
        path
        for path in _git_output(
            root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
            "--",
            "src",
        ).split("\0")
        if path
    )
    ignored_runtime = tuple(
        path
        for path in ignored_candidates
        if "__pycache__" in Path(path).parts
        or Path(path).suffix.lower()
        in {".py", ".pyc", ".pyo", ".so", ".dylib"}
    )
    if ignored_runtime:
        raise TasteGlobalGCESmokeError(
            "T8 official GlobalGCE source closure contains ignored runtime files"
        )
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked = _git_output(root, "ls-files", "-s", "-z").encode("utf-8")
    return {
        "commit": commit,
        "tracked_tree_sha256": _sha256_bytes(tracked),
    }


@dataclass(slots=True)
class HeldOfficialGlobalGCE:
    root: Path
    relative_files: tuple[str, ...]
    files: tuple[Any, ...]
    snapshot: Mapping[str, str]
    source_inventory_sha256: str

    @property
    def runtime_root(self) -> Path:
        """Descriptor-backed checkout root used by the official Python imports."""

        self.revalidate()
        if not sys.platform.startswith("linux"):
            return self.root
        first = self.files[0]
        root_index = len(self.root.parts) - 1
        descriptor = first.directory_fds[root_index]
        held = os.fstat(descriptor)
        named = self.root.stat()
        if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
            raise TasteGlobalGCESmokeError("T8 official root descriptor changed")
        return Path(f"/proc/self/fd/{descriptor}")

    def revalidate(self) -> dict[str, Any]:
        for held in self.files:
            held.revalidate()
        current = _official_git_snapshot(self.root)
        if current != dict(self.snapshot):
            raise TasteGlobalGCESmokeError("T8 official GlobalGCE checkout changed")
        inventory = {
            "schema_version": "tastemolnet_t8_official_globalgce_sources_v1",
            "files": [
                {
                    "relative": relative,
                    "bytes": held.file_identity.size,
                    "sha256": held.sha256,
                }
                for relative, held in zip(self.relative_files, self.files, strict=True)
            ],
        }
        if _canonical_sha256(inventory) != self.source_inventory_sha256:
            raise TasteGlobalGCESmokeError("T8 official source inventory changed")
        return {
            **current,
            "source_inventory_sha256": self.source_inventory_sha256,
            "clean": True,
        }

    def import_authority(self) -> dict[str, dict[str, Any]]:
        """Return held source-relative identities for secure Python imports."""

        self.revalidate()
        authority: dict[str, dict[str, Any]] = {}
        for relative, held in zip(self.relative_files, self.files, strict=True):
            source_relative = Path(relative).relative_to("src").as_posix()
            evidence = held.revalidate()
            authority[source_relative] = {
                "device": evidence["device"],
                "inode": evidence["inode"],
                "bytes": evidence["bytes"],
                "sha256": evidence["sha256"],
            }
        return authority

    def close(self) -> None:
        for held in reversed(self.files):
            held.close()


def hold_official_globalgce(
    stack: ExitStack,
    *,
    root: Path,
    release: Mapping[str, Any],
) -> HeldOfficialGlobalGCE:
    root = _absolute(str(root), field="official GlobalGCE root")
    if root != _absolute(release["official_root"], field="released official root"):
        raise TasteGlobalGCESmokeError("T8 official root differs from release")
    audited = validate_official_globalgce_root(
        root, expected_commit=str(release["official_commit"])
    )
    if Path(audited["official_root"]) != root:
        raise TasteGlobalGCESmokeError("T8 official root audit changed")
    snapshot = _official_git_snapshot(root)
    if snapshot["commit"] != release["official_commit"]:
        raise TasteGlobalGCESmokeError("T8 official commit changed")
    tracked_source = tuple(
        sorted(
            relative
            for relative in _git_output(
                root, "ls-files", "-z", "--", "src"
            ).split("\0")
            if relative.endswith(".py")
        )
    )
    if not set(OFFICIAL_RUNTIME_FILES).issubset(tracked_source):
        raise TasteGlobalGCESmokeError(
            "T8 official tracked Python source inventory is incomplete"
        )
    held_files = tuple(
        stack.enter_context(hold_readonly_file(root / relative))
        for relative in tracked_source
    )
    inventory = {
        "schema_version": "tastemolnet_t8_official_globalgce_sources_v1",
        "files": [
            {
                "relative": relative,
                "bytes": held.file_identity.size,
                "sha256": held.sha256,
            }
            for relative, held in zip(tracked_source, held_files, strict=True)
        ],
    }
    digest = _canonical_sha256(inventory)
    if digest != release["official_source_inventory_sha256"]:
        raise TasteGlobalGCESmokeError("T8 official source inventory differs from release")
    held = HeldOfficialGlobalGCE(root, tracked_source, held_files, snapshot, digest)
    held.revalidate()
    return held


def _checkpoint_train_contract(
    *,
    payloads: Mapping[str, bytes],
    checkpoint_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    if set(payloads) != set(GINE_PAYLOAD_FILES):
        raise TasteGlobalGCESmokeError("T8 frozen GINE payload set changed")
    card = _json(payloads["model_card.json"], field="model card")
    expected_card = {
        "dataset": DATASET,
        "oracle_backend": "gnn",
        "rf_oracle_used": False,
        "backbone": "gine",
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "profile": "full",
    }
    if any(
        type(card.get(key)) is not type(expected) or card.get(key) != expected
        for key, expected in expected_card.items()
    ):
        raise TasteGlobalGCESmokeError("T8 frozen GINE model-card contract changed")
    checkpoint_id = _sha256_bytes(payloads["model.pt"])
    if (
        card.get("checkpoint_id") != checkpoint_id
        or checkpoint_evidence.get("checkpoint_id") != checkpoint_id
    ):
        raise TasteGlobalGCESmokeError("T8 frozen GINE model identity changed")
    label_map = _json(payloads["label_map.json"], field="label map")
    if label_map != {"0": "Bitter", "1": "Sweet", "2": "Tasteless"}:
        raise TasteGlobalGCESmokeError("T8 frozen class order changed")
    temperature = _json(
        payloads["temperature_scaling.json"], field="temperature scaling"
    )
    if (
        type(temperature.get("temperature")) is not float
        or not math.isfinite(temperature["temperature"])
        or temperature["temperature"] <= 0.0
        or temperature.get("selection_split") != "validation"
        or temperature.get("test_used_for_fit") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 frozen calibration contract changed")
    split = _json(payloads["split_manifest.json"], field="split manifest")
    roles = split.get("roles")
    files = split.get("files")
    if (
        set(split)
        != {
            "schema_version",
            "dataset",
            "roles",
            "files",
            "train_manifest",
            "validation_manifest",
            "calibration_loaded_for_training",
            "test_loaded_for_training",
            "test_evaluated_during_training",
            "test_used_for_checkpoint_selection",
        }
        or split.get("schema_version") != "molecular_gnn_split_manifest_v1"
        or split.get("dataset") != DATASET
        or roles
        != {
            "train": "model_fitting",
            "validation": "checkpoint_selection_and_temperature_calibration",
            "calibration": "reserved_for_threshold_and_selector_only",
            "test": "frozen_model_final_quality_evaluation",
        }
        or type(files) is not dict
        or set(files) != {"train", "validation", "calibration", "test"}
        or split.get("calibration_loaded_for_training") is not False
        or split.get("test_loaded_for_training") is not False
        or split.get("test_evaluated_during_training") is not False
        or split.get("test_used_for_checkpoint_selection") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 split-role authority changed")
    normalized: dict[str, tuple[Path, str]] = {}
    for role in ("train", "validation", "calibration", "test"):
        row = files[role]
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise TasteGlobalGCESmokeError(f"T8 {role} split binding changed")
        normalized[role] = (
            _lexical_absolute(row["path"], field=f"{role} split"),
            _sha(row["sha256"], field=f"{role} split SHA"),
        )
    manifest = split.get("train_manifest")
    if type(manifest) is not dict:
        raise TasteGlobalGCESmokeError("T8 train manifest is absent")
    expected_manifest_keys = {
        "schema_version",
        "num_records",
        "num_classes",
        "label_counts",
        "split_counts",
        "source_path",
        "source_sha256",
        "dataset_fingerprint",
        "feature_schema_sha256",
    }
    counts = manifest.get("label_counts")
    records = manifest.get("num_records")
    train_path, train_sha = normalized["train"]
    feature = _json(payloads["feature_schema.json"], field="feature schema")
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != "molecular_graph_dataset_v1"
        or type(records) is not int
        or records <= 0
        or manifest.get("num_classes") != NUM_CLASSES
        or type(counts) is not dict
        or set(counts) != {"0", "1", "2"}
        or any(type(value) is not int or value <= 0 for value in counts.values())
        or sum(counts.values()) != records
        or manifest.get("split_counts") != {"train": records}
        or manifest.get("source_path") != str(train_path)
        or manifest.get("source_sha256") != train_sha
        or _HEX64.fullmatch(str(manifest.get("dataset_fingerprint"))) is None
        or _HEX64.fullmatch(str(manifest.get("feature_schema_sha256"))) is None
        or manifest.get("feature_schema_sha256") != feature.get("schema_sha256")
    ):
        raise TasteGlobalGCESmokeError("T8 exact train manifest changed")
    test_status = _json(
        payloads["test_evaluation_status.json"], field="test status"
    )
    if (
        set(test_status)
        != {"schema_version", "status", "test_loaded", "reason", "path", "sha256"}
        or test_status.get("schema_version")
        != "molecular_gnn_test_evaluation_status_v1"
        or test_status.get("status") != "NOT_EVALUATED"
        or test_status.get("test_loaded") is not False
        or test_status.get("reason") != "held_out_until_frozen_final_evaluation"
        or test_status.get("path") != str(normalized["test"][0])
        or test_status.get("sha256") != normalized["test"][1]
    ):
        raise TasteGlobalGCESmokeError("T8 held-out test contract changed")
    return (
        {
            "sha256": train_sha,
            "row_count": records,
            "label_counts": dict(counts),
            "split": "train",
            "checkpoint_id": checkpoint_id,
            "feature_schema_sha256": _sha256_bytes(payloads["feature_schema.json"]),
            "temperature_scaling_sha256": _sha256_bytes(
                payloads["temperature_scaling.json"]
            ),
        },
        train_path,
    )


def _exact_stage_policy(policy: Any) -> dict[str, Any]:
    observed = dict(policy.stage(STAGE))
    expected = {
        "allowed_input_files": [
            "immutable_t2_bundle",
            "immutable_t3_stage_output",
            "immutable_t4_stage_output",
            "frozen_train_csv",
            "pinned_official_globalgce_checkout",
        ],
        "checkpoint_resume_required": True,
        "device": "cuda:0",
        "fresh_output_required": True,
        "fresh_state_required": True,
        "generation_chunk_size": 8,
        "gspan_flush_every": 64,
        "gspan_max_in_memory_candidates": 64,
        "gpu_uuid_binding_required": True,
        "learning_rate_hex": "0x1.999999999999ap-4",
        "merge_canonical_dedup_required": True,
        "min_freq": 2,
        "minimum_strict_flips_per_branch": 1,
        "mode": "train_only_two_target_native_globalgce_smoke",
        "native_action_space": "lhs_rhs_graph_rewrite",
        "num_classes": 3,
        "num_epochs": 5,
        "oracle_batch_size": 256,
        "physical_gpu_index": 2,
        "rf_oracle_used": False,
        "run": 1,
        "same_frozen_gine_required": True,
        "seed": 7,
        "source_count": 16,
        "source_label": 1,
        "source_scan_limit": 64,
        "split_payload_access": {
            "calibration": False,
            "test": False,
            "train": True,
            "validation": False,
        },
        "target_branches": [0, 2],
        "top_k_native": 20,
        "dropout_hex": "0x1.0000000000000p-1",
        "untargeted_strict_flip_required": True,
    }
    if observed != expected:
        raise TasteGlobalGCESmokeError("T8 downstream stage authority changed")
    return observed


def _managed_binding(
    held: TasteT8ExternalManagedV2Authority,
    *,
    expected_closure: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(held, Mapping):
        raise TasteGlobalGCESmokeError(
            "T8 managed-v2 authority must be a held external object"
        )
    revalidate = getattr(held, "revalidate_t8_managed_v2_authority", None)
    startup_revalidate = getattr(
        held,
        "revalidate_t8_official_startup_authority",
        None,
    )
    if not callable(revalidate) or not callable(startup_revalidate):
        raise TasteGlobalGCESmokeError(
            "T8 managed-v2 external authority adapter is absent"
        )
    evidence = revalidate()
    expected_keys = {
        "schema_version",
        "status",
        "protocol",
        "protocol_source_commit",
        "task_id",
        "run_id",
        "stage",
        "authority_record_sha256",
        "active_generation_sha256",
        "child_identity_sha256",
        "process_lineage_sha256",
        "expected_closure_sha256",
        "gpu_index",
        "gpu_uuid",
        "gpu_lock_mode",
        "auto_terminate_uncontrolled_children",
    }
    if (
        type(evidence) is not dict
        or set(evidence) != expected_keys
        or evidence.get("schema_version")
        != MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA
        or evidence.get("status") != "HELD_ACTIVE_VALID"
        or evidence.get("protocol") != MANAGED_V2_PROTOCOL
        or evidence.get("protocol_source_commit") != MANAGED_V2_SOURCE_COMMIT
        or type(evidence.get("task_id")) is not str
        or evidence.get("task_id") != MANAGED_TASK_ID
        or type(evidence.get("run_id")) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", evidence["run_id"])
        is None
        or evidence.get("stage") != STAGE
        or any(
            _HEX64.fullmatch(str(evidence.get(field))) is None
            for field in (
                "authority_record_sha256",
                "active_generation_sha256",
                "child_identity_sha256",
                "process_lineage_sha256",
                "expected_closure_sha256",
            )
        )
        or evidence.get("expected_closure_sha256")
        != _canonical_sha256(expected_closure)
        or type(evidence.get("gpu_index")) is not int
        or evidence.get("gpu_index") != PHYSICAL_GPU_INDEX
        or type(evidence.get("gpu_uuid")) is not str
        or _GPU_UUID.fullmatch(evidence["gpu_uuid"]) is None
        or evidence.get("gpu_lock_mode") != "exclusive"
        or evidence.get("auto_terminate_uncontrolled_children") is not False
    ):
        raise TasteGlobalGCESmokeError(
            "T8 external managed-v2 GPU/ACTIVE authority changed"
        )
    return {
        "external_authority_schema": MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA,
        "protocol": MANAGED_V2_PROTOCOL,
        "protocol_source_commit": MANAGED_V2_SOURCE_COMMIT,
        "task_id": MANAGED_TASK_ID,
        "run_id": evidence["run_id"],
        "stage": STAGE,
        "authority_record_sha256": evidence["authority_record_sha256"],
        "active_generation_sha256": evidence["active_generation_sha256"],
        "child_identity_sha256": evidence["child_identity_sha256"],
        "process_lineage_sha256": evidence["process_lineage_sha256"],
        "expected_closure_sha256": evidence["expected_closure_sha256"],
        "gpu_index": PHYSICAL_GPU_INDEX,
        "gpu_uuid": evidence["gpu_uuid"],
        "gpu_lock_mode": "exclusive",
        "auto_terminate_uncontrolled_children": False,
        "same_child_revalidated_at_terminal": True,
    }


def _predecessor_binding(
    *,
    t2: Mapping[str, Any],
    t3: Mapping[str, Any],
    t4: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "t2_gate_sha256": t2["gate_sha256"],
        "t2_receipt_sha256": t2["receipt_sha256"],
        "t2_source_evidence_sha256": t2["source_evidence_sha256"],
        "t2_binding_sha256": _canonical_sha256(t2),
        "t3_gate_sha256": t3["gate_sha256"],
        "t3_root_inventory_sha256": t3["root_inventory_sha256"],
        "t4_gate_sha256": t4["gate_sha256"],
        "t4_root_inventory_sha256": t4["root_inventory_sha256"],
        "t3_t4_same_t2_binding": True,
        "t3_t4_same_checkpoint": True,
    }


def _managed_expected_closure(
    *,
    execution: Mapping[str, Any],
    output_root: Path,
    state_root: Path,
    predecessors: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    train: Mapping[str, Any],
    official: Mapping[str, Any],
    base_policy_sha256: str,
    downstream_policy_sha256: str,
) -> dict[str, Any]:
    """Exact values a controller adapter must bind independently of the worker."""

    return {
        "schema_version": "tastemolnet_t8_managed_v2_expected_closure_v1",
        "task_id": MANAGED_TASK_ID,
        "stage": STAGE,
        "execution": {
            "commit": execution["commit"],
            "tree": execution["tree"],
        },
        "fresh_roots": {
            "output_root": str(output_root),
            "state_root": str(state_root),
        },
        "predecessors": dict(predecessors),
        "frozen_gine": {
            field: checkpoint[field]
            for field in (
                "checkpoint_id",
                "checkpoint_inventory_sha256",
                "checkpoint_stat_inventory_sha256",
                "checkpoint_sha256s_sha256",
            )
        },
        "train": {
            "sha256": train["sha256"],
            "row_count": train["row_count"],
            "label_counts": dict(train["label_counts"]),
            "feature_schema_sha256": train["feature_schema_sha256"],
            "temperature_scaling_sha256": train[
                "temperature_scaling_sha256"
            ],
        },
        "official_globalgce": {
            "commit": official["commit"],
            "tracked_tree_sha256": official["tracked_tree_sha256"],
            "source_inventory_sha256": official[
                "source_inventory_sha256"
            ],
        },
        "policy": {
            "base_policy_sha256": base_policy_sha256,
            "downstream_policy_sha256": downstream_policy_sha256,
            "data_redistribution_allowed": False,
        },
    }


@dataclass(slots=True)
class HeldTasteT8Inputs:
    stack: ExitStack
    release: Mapping[str, Any]
    execution: Mapping[str, str]
    release_file: Any
    managed: Any
    managed_binding: Mapping[str, Any]
    t2: Any
    t2_evidence: Mapping[str, Any]
    t3: Any
    t3_evidence: Mapping[str, Any]
    t4: Any
    t4_evidence: Mapping[str, Any]
    checkpoint: Any
    checkpoint_payloads: Mapping[str, bytes]
    checkpoint_evidence: Mapping[str, Any]
    train: Any
    train_bytes: bytes
    train_contract: Mapping[str, Any]
    official: HeldOfficialGlobalGCE
    policy: Any
    predecessor_binding: Mapping[str, Any]
    output_root: Path
    state_root: Path
    _detached: bool = False

    def revalidate(self) -> None:
        self.release_file.revalidate()
        if verify_execution_checkout(self.release) != dict(self.execution):
            raise TasteGlobalGCESmokeError("T8 execution identity changed")
        current_t2 = self.t2.revalidate()
        current_t3_raw = self.t3.revalidate()
        current_t4_raw = self.t4.revalidate()
        current_t3 = {**current_t3_raw, "output_root": str(self.t3.root)}
        current_t4 = {**current_t4_raw, "output_root": str(self.t4.root)}
        if (
            current_t2 != dict(self.t2_evidence)
            or current_t3 != dict(self.t3_evidence)
            or current_t4 != dict(self.t4_evidence)
        ):
            raise TasteGlobalGCESmokeError("T8 retained predecessor changed")
        if self.checkpoint.revalidate() != dict(self.checkpoint_evidence):
            raise TasteGlobalGCESmokeError("T8 frozen checkpoint changed")
        self.train.revalidate()
        current_official = self.official.revalidate()
        self.policy.revalidate(stage=STAGE)
        current_predecessors = _predecessor_binding(
            t2=current_t2,
            t3=current_t3,
            t4=current_t4,
        )
        if current_predecessors != dict(self.predecessor_binding):
            raise TasteGlobalGCESmokeError("T8 managed predecessor closure changed")
        current_expected_closure = _managed_expected_closure(
            execution=self.execution,
            output_root=self.output_root,
            state_root=self.state_root,
            predecessors=current_predecessors,
            checkpoint=self.checkpoint_evidence,
            train=self.train_contract,
            official=current_official,
            base_policy_sha256=self.policy.base_policy.file_sha256,
            downstream_policy_sha256=self.policy.file_sha256,
        )
        current_managed = _managed_binding(
            self.managed,
            expected_closure=current_expected_closure,
        )
        if current_managed != dict(self.managed_binding):
            raise TasteGlobalGCESmokeError("T8 managed-v2 child changed")

    def terminal_authority(self) -> dict[str, Any]:
        self.revalidate()
        official = self.official.revalidate()
        return {
            "execution": {
                **dict(self.execution),
                "release_config_sha256": self.release_file.sha256,
                "python_entrypoint_sha256": _sha256_file(
                    REPO_ROOT / "scripts/run_tastemolnet_globalgce_smoke.py"
                ),
                "autodl_wrapper_sha256": _sha256_file(
                    REPO_ROOT / "scripts/autodl/run_tastemolnet_globalgce_smoke.sh"
                ),
                "slurm_wrapper_sha256": _sha256_file(
                    REPO_ROOT / "scripts/slurm/run_tastemolnet_globalgce_smoke.sh"
                ),
            },
            "managed_execution": dict(self.managed_binding),
            "predecessors": dict(self.predecessor_binding),
            "frozen_gine": {
                "checkpoint_id": self.checkpoint_evidence["checkpoint_id"],
                "checkpoint_inventory_sha256": self.checkpoint_evidence[
                    "checkpoint_inventory_sha256"
                ],
                "checkpoint_stat_inventory_sha256": self.checkpoint_evidence[
                    "checkpoint_stat_inventory_sha256"
                ],
                "checkpoint_sha256s_sha256": self.checkpoint_evidence[
                    "checkpoint_sha256s_sha256"
                ],
                "feature_schema_sha256": self.train_contract[
                    "feature_schema_sha256"
                ],
                "temperature_scaling_sha256": self.train_contract[
                    "temperature_scaling_sha256"
                ],
                "num_classes": NUM_CLASSES,
                "source_label": SOURCE_LABEL,
                "oracle_backend": "gnn",
                "classifier_family": "gine",
                "rf_oracle_used": False,
            },
            "train_split": {
                "sha256": self.train_contract["sha256"],
                "bytes": len(self.train_bytes),
                "row_count": self.train_contract["row_count"],
                "label_counts": dict(self.train_contract["label_counts"]),
                "split": "train",
            },
            "official_globalgce": official,
            "policy": {
                "base_policy_sha256": self.policy.base_policy.file_sha256,
                "downstream_policy_sha256": self.policy.file_sha256,
                "research_compute_allowed": True,
                "aggregate_reporting_allowed": True,
                "data_redistribution_allowed": False,
                "hpc_execution_allowed": False,
                "train_loaded": True,
                "external_validation_loaded": False,
                "calibration_loaded": False,
                "test_loaded": False,
            },
        }

    def revalidate_t8_terminal_authority(self) -> Mapping[str, Any]:
        """Adapt the live held input closure to the public consumer protocol."""

        return self.terminal_authority()

    def revalidate_t8_official_startup_authority(self) -> Mapping[str, Any]:
        """Delegate only to the independently retained controller authority."""

        revalidate = getattr(
            self.managed,
            "revalidate_t8_official_startup_authority",
            None,
        )
        if not callable(revalidate):
            raise TasteGlobalGCESmokeError(
                "T8 official startup expectation authority is absent"
            )
        observed = revalidate()
        if type(observed) is not dict:
            raise TasteGlobalGCESmokeError(
                "T8 official startup expectation authority changed"
            )
        return json.loads(json.dumps(observed))

    def detach_for_process_exit(self) -> ExitStack:
        if self._detached:
            raise TasteGlobalGCESmokeError("T8 retained input stack already detached")
        self._detached = True
        return self.stack.pop_all()

    def close(self) -> None:
        if not self._detached:
            self.stack.close()


def _compare_release_pins(
    release: Mapping[str, Any],
    *,
    t2: Mapping[str, Any],
    t3: Mapping[str, Any],
    t4: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    observed = {
        "t2_adoption_root": t2["adoption_root"],
        "t2_gate_sha256": t2["gate_sha256"],
        "t2_receipt_sha256": t2["receipt_sha256"],
        "t2_source_evidence_sha256": t2["source_evidence_sha256"],
        "t3_output_root": t3["output_root"],
        "t3_gate_sha256": t3["gate_sha256"],
        "t3_root_inventory_sha256": t3["root_inventory_sha256"],
        "t4_output_root": t4["output_root"],
        "t4_gate_sha256": t4["gate_sha256"],
        "t4_root_inventory_sha256": t4["root_inventory_sha256"],
        "checkpoint_dir": checkpoint["checkpoint_dir"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_inventory_sha256": checkpoint["checkpoint_inventory_sha256"],
        "checkpoint_stat_inventory_sha256": checkpoint[
            "checkpoint_stat_inventory_sha256"
        ],
        "checkpoint_sha256s_sha256": checkpoint["checkpoint_sha256s_sha256"],
    }
    if any(release.get(key) != value for key, value in observed.items()):
        raise TasteGlobalGCESmokeError("T8 retained input differs from release pins")


def hold_tastemolnet_t8_inputs(
    *,
    output_dir: str | Path,
    state_dir: str | Path,
    config_path: str | Path,
    t2_adoption: str | Path,
    t3_output: str | Path,
    t4_output: str | Path,
    checkpoint_dir: str | Path,
    train_csv: str | Path,
    official_root: str | Path,
    downstream_policy: str | Path,
    base_policy: str | Path,
    managed_v2_authority: TasteT8ExternalManagedV2Authority | None = None,
) -> HeldTasteT8Inputs:
    """Open every T8 authority once and retain it across terminal publication."""

    release = assert_execution_released()
    if Path(config_path) != REPO_ROOT / "configs/hpc.yaml":
        raise TasteGlobalGCESmokeError("T8 requires the exact integrated HPC config")
    output = _absolute(str(output_dir), field="output root", exists=False)
    state = _absolute(str(state_dir), field="state root", exists=False)
    if (
        output == state
        or output in state.parents
        or state in output.parents
    ):
        raise TasteGlobalGCESmokeError("T8 state and terminal output must be disjoint")
    if output.parent != _absolute(release["output_parent"], field="output parent"):
        raise TasteGlobalGCESmokeError("T8 output is outside the released parent")
    if state.parent != _absolute(release["state_parent"], field="state parent"):
        raise TasteGlobalGCESmokeError("T8 state is outside the released parent")
    immutable_paths = tuple(
        _absolute(release[field], field=f"released {field}")
        for field in (
            "t2_adoption_root",
            "t3_output_root",
            "t4_output_root",
            "checkpoint_dir",
            "official_root",
        )
    ) + (REPO_ROOT,)
    for destination in (output, state):
        if any(
            destination == source
            or destination in source.parents
            or source in destination.parents
            for source in immutable_paths
        ):
            raise TasteGlobalGCESmokeError(
                "T8 fresh state/output overlaps an immutable input"
            )
    stack = ExitStack()
    try:
        release_file = stack.enter_context(
            hold_readonly_file(
                RELEASE_CONFIG_PATH,
                expected_sha256=_sha256_file(RELEASE_CONFIG_PATH),
            )
        )
        execution = verify_execution_checkout(release)
        if managed_v2_authority is None:
            raise TasteGlobalGCESmokeError(
                "T8 requires a reviewed managed-v2 external GPU/ACTIVE "
                "authority adapter; the blocked legacy holder is forbidden"
            )
        if isinstance(managed_v2_authority, Mapping):
            raise TasteGlobalGCESmokeError(
                "T8 managed-v2 authority cannot be a worker-supplied mapping"
            )

        from src.eval.tastemolnet_gnn_stages import (
            hold_taste_checkpoint_bundle,
            hold_taste_stage_output,
        )
        from src.utils.tastemolnet_downstream_policy import (
            load_tastemolnet_downstream_policy,
        )
        from src.utils.tastemolnet_gine_pass_adoption_v1 import (
            hold_t2_gine_pass_adoption,
        )

        t2 = stack.enter_context(
            hold_t2_gine_pass_adoption(
                t2_adoption,
                expected_gate_sha256=release["t2_gate_sha256"],
                expected_receipt_sha256=release["t2_receipt_sha256"],
                expected_source_evidence_sha256=release[
                    "t2_source_evidence_sha256"
                ],
            )
        )
        t2_evidence = t2.revalidate()
        t3 = stack.enter_context(hold_taste_stage_output(t3_output))
        t4 = stack.enter_context(hold_taste_stage_output(t4_output))
        t3_raw = t3.revalidate()
        t4_raw = t4.revalidate()
        t3_evidence = {**t3_raw, "output_root": str(t3.root)}
        t4_evidence = {**t4_raw, "output_root": str(t4.root)}
        if (
            t3_evidence["stage"] != "T3_GINE_CALIBRATED"
            or t4_evidence["stage"] != "T4_ORACLE_SMOKE"
        ):
            raise TasteGlobalGCESmokeError("T8 T3/T4 stage order changed")
        t2_binding_sha = _canonical_sha256(t2_evidence)
        for evidence in (t3_evidence, t4_evidence):
            if (
                evidence["t2_adoption_gate_sha256"] != t2_evidence["gate_sha256"]
                or evidence["t2_adoption_receipt_sha256"]
                != t2_evidence["receipt_sha256"]
                or evidence["t2_adoption_binding_sha256"] != t2_binding_sha
            ):
                raise TasteGlobalGCESmokeError("T8 T2/T3/T4 binding changed")
        checkpoint_fields = (
            "checkpoint_dir",
            "checkpoint_id",
            "checkpoint_inventory_sha256",
            "checkpoint_stat_inventory_sha256",
            "checkpoint_sha256s_sha256",
        )
        if any(
            t3_evidence[field] != t4_evidence[field] for field in checkpoint_fields
        ):
            raise TasteGlobalGCESmokeError("T8 T3/T4 bind different GINE checkpoints")
        checkpoint_path = _absolute(str(checkpoint_dir), field="checkpoint")
        if checkpoint_path != Path(t3_evidence["checkpoint_dir"]):
            raise TasteGlobalGCESmokeError("T8 checkpoint argument differs from T3/T4")
        checkpoint = stack.enter_context(
            hold_taste_checkpoint_bundle(
                checkpoint_path,
                expected_stage_evidence=t3_raw,
            )
        )
        checkpoint_evidence = checkpoint.revalidate()
        payloads = {
            name: checkpoint.read_frozen_gine_payload(name)
            for name in GINE_PAYLOAD_FILES
        }
        checkpoint.revalidate()
        train_contract, expected_train = _checkpoint_train_contract(
            payloads=payloads,
            checkpoint_evidence=checkpoint_evidence,
        )
        requested_train = _absolute(str(train_csv), field="train CSV")
        if requested_train != expected_train:
            raise TasteGlobalGCESmokeError("T8 train argument differs from checkpoint")
        train = stack.enter_context(
            hold_readonly_file(
                requested_train, expected_sha256=train_contract["sha256"]
            )
        )
        train_bytes = train.read_bytes()
        train.revalidate()
        if len(train_bytes) <= 0:
            raise TasteGlobalGCESmokeError("T8 train payload is empty")
        official = hold_official_globalgce(
            stack,
            root=_absolute(str(official_root), field="official root"),
            release=release,
        )
        policy = stack.enter_context(
            load_tastemolnet_downstream_policy(
                downstream_policy,
                base_policy_path=base_policy,
            )
        )
        _exact_stage_policy(policy)
        if (
            policy.file_sha256 != release["downstream_policy_sha256"]
            or policy.base_policy.file_sha256 != release["base_policy_sha256"]
        ):
            raise TasteGlobalGCESmokeError("T8 policy differs from release")
        _compare_release_pins(
            release,
            t2=t2_evidence,
            t3=t3_evidence,
            t4=t4_evidence,
            checkpoint=checkpoint_evidence,
        )
        predecessor_binding = _predecessor_binding(
            t2=t2_evidence,
            t3=t3_evidence,
            t4=t4_evidence,
        )
        official_evidence = official.revalidate()
        managed_expected_closure = _managed_expected_closure(
            execution=execution,
            output_root=output,
            state_root=state,
            predecessors=predecessor_binding,
            checkpoint=checkpoint_evidence,
            train=train_contract,
            official=official_evidence,
            base_policy_sha256=policy.base_policy.file_sha256,
            downstream_policy_sha256=policy.file_sha256,
        )
        managed_binding = _managed_binding(
            managed_v2_authority,
            expected_closure=managed_expected_closure,
        )
        result = HeldTasteT8Inputs(
            stack=stack,
            release=release,
            execution=execution,
            release_file=release_file,
            managed=managed_v2_authority,
            managed_binding=managed_binding,
            t2=t2,
            t2_evidence=t2_evidence,
            t3=t3,
            t3_evidence=t3_evidence,
            t4=t4,
            t4_evidence=t4_evidence,
            checkpoint=checkpoint,
            checkpoint_payloads=payloads,
            checkpoint_evidence=checkpoint_evidence,
            train=train,
            train_bytes=train_bytes,
            train_contract=train_contract,
            official=official,
            policy=policy,
            predecessor_binding=predecessor_binding,
            output_root=output,
            state_root=state,
        )
        result.revalidate()
        return result
    except BaseException:
        stack.close()
        raise


__all__ = [
    "CRITICAL_BLOBS",
    "DISABLED_RELEASE_STATE",
    "HeldOfficialGlobalGCE",
    "HeldTasteT8Inputs",
    "RELEASE_CONFIG_PATH",
    "RELEASE_KEYS",
    "RELEASE_SCHEMA",
    "TasteT8ExternalManagedV2Authority",
    "TasteT8ReleaseDisabled",
    "assert_execution_released",
    "hold_official_globalgce",
    "hold_tastemolnet_t8_inputs",
    "verify_execution_checkout",
]
