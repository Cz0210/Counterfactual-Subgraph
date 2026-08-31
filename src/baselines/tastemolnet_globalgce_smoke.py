"""Bounded native GlobalGCE smoke for the three-class TasteMolNet route.

The smoke preserves GlobalGCE's attachment-aware LHS-to-RHS action.  It runs
two target-specific optimization branches (Sweet -> Bitter and Sweet ->
Tasteless) against one descriptor-authorized calibrated GINE, deliberately
interrupts each branch after its first durable epoch checkpoint, resumes the
same identity-bound state, merges rules by canonical action content, and then
uses the original three-class class order for final untargeted strict-flip
validation.

Only aggregate evidence is eligible for the terminal output.  Train rows,
SMILES, molecule identifiers, rule tensors, and per-example predictions remain
inside the private process/state boundary and are never serialized there.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from src.baselines.globalgce_bace_native_rules import (
    ACTION_ENGINE_VERSION,
    OFFICIAL_GLOBALGCE_COMMIT,
)
from src.baselines.globalgce_mutagenicity_adapter import (
    FROZEN_GINE_IN_MEMORY_FILES,
    NativeGenerationResult,
    OFFICIAL_API_SIGNATURE_FILE,
    OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA,
    OFFICIAL_GLOBALGCE_MODULE_PROVENANCE_SCHEMA,
    OfficialGlobalGCEMutagenicityGenerator,
    PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES,
    PYTHON_MODULE_PROVENANCE_FILE,
    TrainParent,
)
from src.baselines.globalgce_resumable import (
    GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2,
    GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION,
)
from src.baselines.tastemolnet_multiclass_adapters import (
    merge_globalgce_target_branches,
)
from src.utils.retained_output_directory import (
    FreshOutputDirectory,
    HeldPublishedTerminalOutput,
    PreparedTerminalOutput,
    RetainedOutputTree,
    prepare_terminal_output,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - required on AutoDL.
    Chem = None


STAGE = "T8_GLOBALGCE_SMOKE"
DATASET = "tastemolnet"
METHOD = "GlobalGCE"
NUM_CLASSES = 3
SOURCE_LABEL = 1
TARGET_BRANCHES = (0, 2)
DESTINATION_LABELS = (0, 2)
PHYSICAL_GPU_INDEX = 2
VISIBLE_DEVICE = "cuda:0"
SEED = 7
PASS_MARKER = "[TASTE_T8_GLOBALGCE_SMOKE_PASS]"
MANAGED_TASK_ID = "tastemolnet_t8_globalgce_smoke"
MANAGED_V2_PROTOCOL = "managed_execution_v2"
MANAGED_V2_SOURCE_COMMIT = "3405ae1d24fdaeb7a4af40b14823b36051966a35"
MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA = (
    "tastemolnet_t8_external_managed_v2_authority_v1"
)
OUTPUT_PAYLOAD_FILES = (
    "input_hashes.json",
    "state.json",
    "manifest.json",
    "gate.json",
)
TERMINAL_FILES = frozenset((*OUTPUT_PAYLOAD_FILES, "output_hashes.json", "PASS"))
GINE_PAYLOAD_FILES = tuple(sorted(FROZEN_GINE_IN_MEMORY_FILES))

INPUT_SCHEMA = "tastemolnet_t8_globalgce_smoke_inputs_v1"
STATE_SCHEMA = "tastemolnet_t8_globalgce_smoke_state_v1"
MANIFEST_SCHEMA = "tastemolnet_t8_globalgce_smoke_manifest_v1"
GATE_SCHEMA = "tastemolnet_t8_globalgce_smoke_gate_v1"
BRANCH_SCHEMA = "tastemolnet_t8_globalgce_branch_resume_v2"
CHECKPOINT_SEAL_SCHEMA = "tastemolnet_t8_post_callback_checkpoint_seal_v1"
SCIENCE_SCHEMA = "tastemolnet_t8_globalgce_science_v1"
_SAFE_MANAGED_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]+$")
_MUTABLE_CHECKPOINT_DIRECTORY = "globalgce_training_checkpoints"
_SEALED_CHECKPOINT_DIRECTORY = "sealed-planned-checkpoint"
_POST_CALLBACK_SETTLE_ATTEMPTS = 16
_POST_CALLBACK_SETTLE_SECONDS = 0.025
_POST_CALLBACK_STABLE_SAMPLES = 3


class TasteGlobalGCESmokeError(RuntimeError):
    """The bounded T8 science or its terminal closure failed closed."""


class PlannedGlobalGCECheckpointStop(RuntimeError):
    """Internal control-flow signal emitted only after a durable checkpoint."""


@dataclass(frozen=True, slots=True)
class TasteGlobalGCESmokeConfig:
    source_parent_count: int = 16
    source_scan_limit: int = 64
    epochs: int = 5
    top_k_native: int = 20
    min_freq: int = 2
    learning_rate: float = 0.1
    dropout: float = 0.5
    generation_chunk_size: int = 8
    oracle_batch_size: int = 256
    gspan_flush_every: int = 64
    gspan_max_in_memory_candidates: int = 64
    minimum_strict_flips_per_branch: int = 1
    seed: int = SEED

    def validate(self) -> None:
        exact_positive = {
            "source_parent_count": self.source_parent_count,
            "source_scan_limit": self.source_scan_limit,
            "epochs": self.epochs,
            "top_k_native": self.top_k_native,
            "min_freq": self.min_freq,
            "generation_chunk_size": self.generation_chunk_size,
            "oracle_batch_size": self.oracle_batch_size,
            "gspan_flush_every": self.gspan_flush_every,
            "gspan_max_in_memory_candidates": self.gspan_max_in_memory_candidates,
            "minimum_strict_flips_per_branch": self.minimum_strict_flips_per_branch,
        }
        for name, value in exact_positive.items():
            if type(value) is not int or value <= 0:
                raise TasteGlobalGCESmokeError(
                    f"T8 {name} must be one native positive integer"
                )
        if self.source_scan_limit < self.source_parent_count:
            raise TasteGlobalGCESmokeError(
                "T8 source scan limit cannot be smaller than its bounded cohort"
            )
        if self.epochs < 5:
            raise TasteGlobalGCESmokeError(
                "T8 must include the first official validation checkpoint at epoch 0 "
                "and at least five subsequent bounded updates"
            )
        if self.top_k_native < 20:
            raise TasteGlobalGCESmokeError(
                "T8 must retain at least the official Top20 native rule surface"
            )
        if self.min_freq < 2:
            raise TasteGlobalGCESmokeError("T8 GlobalGCE min_freq must be at least two")
        for name, value in {
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
        }.items():
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise TasteGlobalGCESmokeError(f"T8 {name} must be finite and positive")
        if not 0.0 < self.dropout < 1.0:
            raise TasteGlobalGCESmokeError("T8 dropout must be strictly inside (0,1)")
        if type(self.seed) is not int or self.seed != SEED:
            raise TasteGlobalGCESmokeError("T8 is frozen to seed 7")
        if (
            self.source_parent_count,
            self.source_scan_limit,
            self.epochs,
            self.top_k_native,
            self.min_freq,
            self.learning_rate.hex(),
            self.dropout.hex(),
            self.generation_chunk_size,
            self.oracle_batch_size,
            self.gspan_flush_every,
            self.gspan_max_in_memory_candidates,
            self.minimum_strict_flips_per_branch,
        ) != (
            16,
            64,
            5,
            20,
            2,
            "0x1.999999999999ap-4",
            "0x1.0000000000000p-1",
            8,
            256,
            64,
            64,
            1,
        ):
            raise TasteGlobalGCESmokeError(
                "T8 bounded science configuration differs from the frozen smoke"
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "source_parent_count": self.source_parent_count,
            "source_scan_limit": self.source_scan_limit,
            "epochs": self.epochs,
            "top_k_native": self.top_k_native,
            "min_freq": self.min_freq,
            "learning_rate_hex": self.learning_rate.hex(),
            "dropout_hex": self.dropout.hex(),
            "generation_chunk_size": self.generation_chunk_size,
            "oracle_batch_size": self.oracle_batch_size,
            "gspan_flush_every": self.gspan_flush_every,
            "gspan_max_in_memory_candidates": self.gspan_max_in_memory_candidates,
            "minimum_strict_flips_per_branch": self.minimum_strict_flips_per_branch,
            "seed": self.seed,
        }


class TastePredictionScorer(Protocol):
    checkpoint_id: str
    num_classes: int
    source_label: int
    temperature: float

    def score_smiles(self, values: Sequence[str]) -> list[dict[str, Any]]:
        ...


class TasteBranchGenerator(Protocol):
    target_label: int

    def generate(
        self,
        parents: Sequence[TrainParent],
        *,
        output_dir: Path,
        seed: int,
        epochs: int,
        top_k_native: int,
        learning_rate: float,
        dropout: float,
        device: str,
        resume: bool,
        generation_chunk_size: int = 32,
        generation_num_workers: int = 0,
        memory_log_every_chunks: int = 1,
        gspan_flush_every: int = 256,
        gspan_max_in_memory_candidates: int = 256,
        gspan_exact_top_k_pruning: bool = False,
        gspan_adoption_proof: str | Path | None = None,
        start_parent_offset: int = 0,
        on_training_ready: Callable[[dict[str, Any]], None] | None = None,
        on_chunk: (
            Callable[[int, int, int, list[dict[str, Any]]], None] | None
        ) = None,
        rules_only: bool = False,
        expected_resume_checkpoint: Mapping[str, Any] | None = None,
        on_resume_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        after_epoch_checkpoint: Callable[[dict[str, Any]], None] | None = None,
        on_generation_complete: Callable[[], None] | None = None,
    ) -> NativeGenerationResult:
        ...


_REQUIRED_PARAMETER = object()
_BRANCH_GENERATOR_PARAMETER_CONTRACT = (
    ("parents", inspect.Parameter.POSITIONAL_OR_KEYWORD, _REQUIRED_PARAMETER),
    ("output_dir", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("seed", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("epochs", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("top_k_native", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("learning_rate", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("dropout", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("device", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("resume", inspect.Parameter.KEYWORD_ONLY, _REQUIRED_PARAMETER),
    ("generation_chunk_size", inspect.Parameter.KEYWORD_ONLY, 32),
    ("generation_num_workers", inspect.Parameter.KEYWORD_ONLY, 0),
    ("memory_log_every_chunks", inspect.Parameter.KEYWORD_ONLY, 1),
    ("gspan_flush_every", inspect.Parameter.KEYWORD_ONLY, 256),
    ("gspan_max_in_memory_candidates", inspect.Parameter.KEYWORD_ONLY, 256),
    ("gspan_exact_top_k_pruning", inspect.Parameter.KEYWORD_ONLY, False),
    ("gspan_adoption_proof", inspect.Parameter.KEYWORD_ONLY, None),
    ("start_parent_offset", inspect.Parameter.KEYWORD_ONLY, 0),
    ("on_training_ready", inspect.Parameter.KEYWORD_ONLY, None),
    ("on_chunk", inspect.Parameter.KEYWORD_ONLY, None),
    ("rules_only", inspect.Parameter.KEYWORD_ONLY, False),
    ("expected_resume_checkpoint", inspect.Parameter.KEYWORD_ONLY, None),
    ("on_resume_checkpoint", inspect.Parameter.KEYWORD_ONLY, None),
    ("after_epoch_checkpoint", inspect.Parameter.KEYWORD_ONLY, None),
    ("on_generation_complete", inspect.Parameter.KEYWORD_ONLY, None),
)


def _require_exact_branch_generator_signature(
    generator: TasteBranchGenerator,
) -> None:
    try:
        signature = inspect.signature(generator.generate)
    except (TypeError, ValueError) as exc:
        raise TasteGlobalGCESmokeError(
            "T8 generator API signature is not inspectable"
        ) from exc
    parameters = tuple(signature.parameters.values())
    if len(parameters) != len(_BRANCH_GENERATOR_PARAMETER_CONTRACT):
        raise TasteGlobalGCESmokeError("T8 generator API signature changed")
    for observed, (name, kind, default) in zip(
        parameters,
        _BRANCH_GENERATOR_PARAMETER_CONTRACT,
        strict=True,
    ):
        if observed.name != name or observed.kind is not kind:
            raise TasteGlobalGCESmokeError("T8 generator API signature changed")
        if default is _REQUIRED_PARAMETER:
            if observed.default is not inspect.Parameter.empty:
                raise TasteGlobalGCESmokeError(
                    "T8 generator API signature changed"
                )
        elif (
            type(observed.default) is not type(default)
            or observed.default != default
        ):
            raise TasteGlobalGCESmokeError("T8 generator API signature changed")


class TasteGlobalGCETerminalAuthority(Protocol):
    """Independent held authority required by every public T8 consumer.

    A managed COMPLETION/ACTIVE holder can implement this one narrow method.
    The returned mapping is the full input authority frozen before terminal
    publication; it must never be reconstructed from the terminal root.
    """

    def revalidate_t8_terminal_authority(self) -> Mapping[str, Any]:
        ...

    def revalidate_t8_official_startup_authority(self) -> Mapping[str, Any]:
        """Return pre-worker expected API/import evidence for the verifier."""

        ...


_RETAINED_FILE_EVIDENCE_KEYS = {
    "device",
    "inode",
    "mode",
    "uid",
    "gid",
    "link_count",
    "bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
}


def _hash_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not block:
            raise TasteGlobalGCESmokeError("T8 retained branch leaf ended early")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        raise TasteGlobalGCESmokeError("T8 retained branch leaf grew")
    return digest.hexdigest()


def _regular_fd_evidence(descriptor: int) -> dict[str, Any]:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise TasteGlobalGCESmokeError(
            "T8 retained branch leaf is not a single-link regular file"
        )
    return {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "uid": int(observed.st_uid),
        "gid": int(observed.st_gid),
        "link_count": int(observed.st_nlink),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
        "sha256": _hash_fd(descriptor, int(observed.st_size)),
    }


def _held_replaced_leaf_is_unchanged(
    descriptor: int,
    expected: Mapping[str, Any],
) -> bool:
    """Check an epoch-0 leaf after the trainer atomically replaced its name.

    A successful ``os.replace`` legitimately drops the retained old inode's
    link count to zero and changes its ctime.  All content-bearing identity
    fields must remain exact; while the inode is still named, even ctime and
    link count remain part of the comparison.
    """

    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink not in {0, 1}:
        return False
    current = {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": int(observed.st_mode),
        "uid": int(observed.st_uid),
        "gid": int(observed.st_gid),
        "link_count": int(observed.st_nlink),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
        "sha256": _hash_fd(descriptor, int(observed.st_size)),
    }
    stable_keys = {
        "device",
        "inode",
        "mode",
        "uid",
        "gid",
        "bytes",
        "mtime_ns",
        "sha256",
    }
    if any(current[key] != expected[key] for key in stable_keys):
        return False
    if current["link_count"] == 1:
        return current == dict(expected)
    return expected.get("link_count") == 1


def _normalize_retained_file_evidence(
    value: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _RETAINED_FILE_EVIDENCE_KEYS:
        raise TasteGlobalGCESmokeError(f"T8 {field} file-evidence schema changed")
    if (
        any(
            type(value.get(key)) is not int or value[key] < minimum
            for key, minimum in (
                ("device", 0),
                ("inode", 1),
                ("mode", 1),
                ("uid", 0),
                ("gid", 0),
                ("link_count", 1),
                ("bytes", 1),
                ("mtime_ns", 0),
                ("ctime_ns", 0),
            )
        )
        or value["link_count"] != 1
        or not _is_sha256(value.get("sha256"))
    ):
        raise TasteGlobalGCESmokeError(f"T8 {field} file identity changed")
    return dict(value)


@dataclass(frozen=True, slots=True)
class _BranchDirectoryIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_BranchDirectoryIdentity":
        if not stat.S_ISDIR(value.st_mode):
            raise TasteGlobalGCESmokeError("T8 retained branch is not a directory")
        return cls(
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_uid),
            int(value.st_gid),
        )


@dataclass(slots=True)
class _HeldBranchDirectory:
    parent: FreshOutputDirectory
    name: str
    path: Path
    descriptor: int
    identity: _BranchDirectoryIdentity

    @classmethod
    def create(
        cls, parent: FreshOutputDirectory, *, target_label: int
    ) -> "_HeldBranchDirectory":
        name = f"target-{target_label}"
        if target_label not in TARGET_BRANCHES:
            raise TasteGlobalGCESmokeError("T8 branch target directory changed")
        parent.revalidate()
        try:
            os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise TasteGlobalGCESmokeError("T8 branch root must be fresh")
        os.mkdir(name, 0o700, dir_fd=parent.descriptor)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, dir_fd=parent.descriptor)
        try:
            identity = _BranchDirectoryIdentity.from_stat(os.fstat(descriptor))
            if identity != _BranchDirectoryIdentity.from_stat(
                os.stat(name, dir_fd=parent.descriptor, follow_symlinks=False)
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 branch directory changed while retaining it"
                )
            os.fsync(parent.descriptor)
            held = cls(parent, name, parent.path / name, descriptor, identity)
            held.revalidate()
            return held
        except BaseException:
            os.close(descriptor)
            raise

    @property
    def runtime_path(self) -> Path:
        if sys.platform.startswith("linux"):
            return Path(f"/proc/self/fd/{self.descriptor}")
        return self.path

    def revalidate(self) -> None:
        self.parent.revalidate()
        if (
            self.descriptor < 0
            or _BranchDirectoryIdentity.from_stat(os.fstat(self.descriptor))
            != self.identity
            or _BranchDirectoryIdentity.from_stat(
                os.stat(
                    self.name,
                    dir_fd=self.parent.descriptor,
                    follow_symlinks=False,
                )
            )
            != self.identity
            or stat.S_IMODE(self.identity.mode) != 0o700
        ):
            raise TasteGlobalGCESmokeError("T8 retained branch directory changed")

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor, self.descriptor = self.descriptor, -1
            os.close(descriptor)


@dataclass(slots=True)
class _HeldPlannedCheckpoint:
    branch: _HeldBranchDirectory
    directory_fd: int
    directory_identity: _BranchDirectoryIdentity
    checkpoint_fd: int
    heartbeat_fd: int
    checkpoint_evidence: Mapping[str, Any]
    heartbeat_evidence: Mapping[str, Any]

    @staticmethod
    def _open_leaf(parent_fd: int, name: str) -> tuple[int, dict[str, Any]]:
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            evidence = _regular_fd_evidence(descriptor)
            if any(
                int(getattr(named, attribute)) != evidence[key]
                for key, attribute in (
                    ("device", "st_dev"),
                    ("inode", "st_ino"),
                    ("mode", "st_mode"),
                    ("uid", "st_uid"),
                    ("gid", "st_gid"),
                    ("link_count", "st_nlink"),
                    ("bytes", "st_size"),
                    ("mtime_ns", "st_mtime_ns"),
                    ("ctime_ns", "st_ctime_ns"),
                )
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 planned checkpoint leaf changed while opening"
                )
            return descriptor, evidence
        except BaseException:
            os.close(descriptor)
            raise

    @classmethod
    def capture(
        cls,
        branch: _HeldBranchDirectory,
        *,
        checkpoint_evidence: Mapping[str, Any],
        heartbeat_evidence: Mapping[str, Any],
    ) -> "_HeldPlannedCheckpoint":
        branch.revalidate()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        named_directory = os.stat(
            "globalgce_training_checkpoints",
            dir_fd=branch.descriptor,
            follow_symlinks=False,
        )
        directory_fd = os.open(
            "globalgce_training_checkpoints", flags, dir_fd=branch.descriptor
        )
        checkpoint_fd = heartbeat_fd = -1
        try:
            directory_identity = _BranchDirectoryIdentity.from_stat(
                os.fstat(directory_fd)
            )
            if directory_identity != _BranchDirectoryIdentity.from_stat(
                named_directory
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 planned checkpoint directory changed while opening"
                )
            checkpoint_fd, observed_checkpoint = cls._open_leaf(
                directory_fd, "training_checkpoint.pt"
            )
            heartbeat_fd, observed_heartbeat = cls._open_leaf(
                directory_fd, "training_heartbeat.json"
            )
            if observed_checkpoint != _normalize_retained_file_evidence(
                checkpoint_evidence, field="planned checkpoint"
            ) or observed_heartbeat != _normalize_retained_file_evidence(
                heartbeat_evidence, field="planned heartbeat"
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 planned checkpoint callback differs from retained leaves"
                )
            held = cls(
                branch,
                directory_fd,
                directory_identity,
                checkpoint_fd,
                heartbeat_fd,
                observed_checkpoint,
                observed_heartbeat,
            )
            held.require_named_for_resume()
            return held
        except BaseException:
            for descriptor in (checkpoint_fd, heartbeat_fd, directory_fd):
                if descriptor >= 0:
                    os.close(descriptor)
            raise

    def _revalidate_directory(self) -> None:
        self.branch.revalidate()
        if (
            _BranchDirectoryIdentity.from_stat(os.fstat(self.directory_fd))
            != self.directory_identity
            or _BranchDirectoryIdentity.from_stat(
                os.stat(
                    "globalgce_training_checkpoints",
                    dir_fd=self.branch.descriptor,
                    follow_symlinks=False,
                )
            )
            != self.directory_identity
        ):
            raise TasteGlobalGCESmokeError(
                "T8 planned checkpoint directory was replaced"
            )

    def require_named_for_resume(self) -> None:
        self._revalidate_directory()
        for name, descriptor, expected in (
            ("training_checkpoint.pt", self.checkpoint_fd, self.checkpoint_evidence),
            ("training_heartbeat.json", self.heartbeat_fd, self.heartbeat_evidence),
        ):
            try:
                held_evidence = _regular_fd_evidence(descriptor)
            except TasteGlobalGCESmokeError as exc:
                raise TasteGlobalGCESmokeError(
                    "T8 named checkpoint leaf differs before resume"
                ) from exc
            if held_evidence != dict(expected):
                raise TasteGlobalGCESmokeError(
                    "T8 planned checkpoint bytes changed"
                )
            named = os.stat(name, dir_fd=self.directory_fd, follow_symlinks=False)
            if any(
                int(getattr(named, attribute)) != expected[key]
                for key, attribute in (
                    ("device", "st_dev"),
                    ("inode", "st_ino"),
                    ("mode", "st_mode"),
                    ("uid", "st_uid"),
                    ("gid", "st_gid"),
                    ("link_count", "st_nlink"),
                    ("bytes", "st_size"),
                    ("mtime_ns", "st_mtime_ns"),
                    ("ctime_ns", "st_ctime_ns"),
                )
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 named checkpoint leaf differs before resume"
                )

    def revalidate_held(self) -> None:
        self._revalidate_directory()
        if (
            not _held_replaced_leaf_is_unchanged(
                self.checkpoint_fd, self.checkpoint_evidence
            )
            or not _held_replaced_leaf_is_unchanged(
                self.heartbeat_fd, self.heartbeat_evidence
            )
        ):
            raise TasteGlobalGCESmokeError(
                "T8 retained epoch-0 checkpoint was mutated in place"
            )

    def close(self) -> None:
        for field in ("heartbeat_fd", "checkpoint_fd", "directory_fd"):
            descriptor = getattr(self, field)
            setattr(self, field, -1)
            if descriptor >= 0:
                os.close(descriptor)


_POST_CALLBACK_STABLE_FIELDS = frozenset(
    _RETAINED_FILE_EVIDENCE_KEYS - {"ctime_ns"}
)


def _post_callback_settled_leaf(
    parent_fd: int,
    name: str,
    *,
    callback_evidence: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    """Freeze one leaf only after the checkpoint callback has fully unwound.

    AutoFS can publish the final ctime after a completed writer returns through
    its Python callback.  Content and every other physical field remain exact;
    only a monotone ctime transition is accepted during this bounded quiet
    period.  The returned evidence is therefore created after all writer-owned
    objects have left the callback stack.
    """

    expected = (
        _normalize_retained_file_evidence(
            callback_evidence,
            field=f"post-callback {name}",
        )
        if callback_evidence is not None
        else None
    )
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    previous: dict[str, Any] | None = None
    stable_samples = 0
    ctime_settled = False
    try:
        for attempt in range(_POST_CALLBACK_SETTLE_ATTEMPTS):
            observed = _regular_fd_evidence(descriptor)
            named_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            named = {
                key: int(getattr(named_stat, attribute))
                for key, attribute in (
                    ("device", "st_dev"),
                    ("inode", "st_ino"),
                    ("mode", "st_mode"),
                    ("uid", "st_uid"),
                    ("gid", "st_gid"),
                    ("link_count", "st_nlink"),
                    ("bytes", "st_size"),
                    ("mtime_ns", "st_mtime_ns"),
                    ("ctime_ns", "st_ctime_ns"),
                )
            }
            if any(named[key] != observed[key] for key in named):
                stable_samples = 0
                previous = None
            else:
                if expected is not None:
                    if any(
                        observed[key] != expected[key]
                        for key in _POST_CALLBACK_STABLE_FIELDS
                    ):
                        raise TasteGlobalGCESmokeError(
                            f"T8 {name} bytes or physical identity changed "
                            "after callback unwind"
                        )
                    if observed["ctime_ns"] < expected["ctime_ns"]:
                        raise TasteGlobalGCESmokeError(
                            f"T8 {name} ctime moved backwards after callback unwind"
                        )
                    ctime_settled = (
                        ctime_settled
                        or observed["ctime_ns"] != expected["ctime_ns"]
                    )
                if observed == previous:
                    stable_samples += 1
                else:
                    previous = observed
                    stable_samples = 1
                if stable_samples >= _POST_CALLBACK_STABLE_SAMPLES:
                    return observed, ctime_settled
            if attempt + 1 < _POST_CALLBACK_SETTLE_ATTEMPTS:
                time.sleep(_POST_CALLBACK_SETTLE_SECONDS)
        raise TasteGlobalGCESmokeError(
            f"T8 {name} did not settle after callback unwind"
        )
    finally:
        os.close(descriptor)


def _checkpoint_content_manifest(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    directories = inventory.get("directories")
    files = inventory.get("files")
    if type(directories) is not dict or type(files) is not dict:
        raise TasteGlobalGCESmokeError(
            "T8 sealed checkpoint inventory schema changed"
        )
    return {
        "directories": sorted(directories),
        "files": {
            relative: {
                "bytes": entry.get("bytes"),
                "sha256": entry.get("sha256"),
            }
            for relative, entry in sorted(files.items())
        },
    }


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise TasteGlobalGCESmokeError(
                "T8 sealed checkpoint copy ended early"
            )
        offset += written


def _copy_retained_checkpoint_tree(
    source: RetainedOutputTree,
    *,
    parent_fd: int,
    name: str,
) -> int:
    """Create one descriptor-anchored, no-symlink resume copy."""

    source.revalidate()
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = os.open(name, flags, dir_fd=parent_fd)
    directory_fds: dict[str, int] = {"": root_fd}
    try:
        directories = source.inventory["directories"]
        for relative in sorted(
            directories,
            key=lambda value: (len(Path(value).parts), value),
        ):
            path = Path(relative)
            parent = str(path.parent) if str(path.parent) != "." else ""
            mode = stat.S_IMODE(int(directories[relative]["mode"]))
            os.mkdir(path.name, mode, dir_fd=directory_fds[parent])
            child = os.open(path.name, flags, dir_fd=directory_fds[parent])
            directory_fds[relative] = child
        files = source.inventory["files"]
        for relative in sorted(files):
            path = Path(relative)
            parent = str(path.parent) if str(path.parent) != "." else ""
            evidence = files[relative]
            descriptor = os.open(
                path.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                stat.S_IMODE(int(evidence["mode"])),
                dir_fd=directory_fds[parent],
            )
            try:
                _write_all(descriptor, source.read_bytes(relative))
                os.fchmod(descriptor, stat.S_IMODE(int(evidence["mode"])))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for relative in sorted(
            (value for value in directory_fds if value),
            key=lambda value: (len(Path(value).parts), value),
            reverse=True,
        ):
            os.fsync(directory_fds[relative])
        os.fsync(root_fd)
        source.revalidate()
        return root_fd
    except BaseException:
        os.close(root_fd)
        raise
    finally:
        for relative, descriptor in list(directory_fds.items())[::-1]:
            if relative:
                os.close(descriptor)


@dataclass(slots=True)
class _HeldSealedCheckpointTree:
    branch: _HeldBranchDirectory
    name: str
    descriptor: int
    identity: _BranchDirectoryIdentity
    tree: RetainedOutputTree

    def revalidate(self) -> None:
        self.branch.revalidate()
        if (
            _BranchDirectoryIdentity.from_stat(os.fstat(self.descriptor))
            != self.identity
            or _BranchDirectoryIdentity.from_stat(
                os.stat(
                    self.name,
                    dir_fd=self.branch.descriptor,
                    follow_symlinks=False,
                )
            )
            != self.identity
        ):
            raise TasteGlobalGCESmokeError(
                "T8 sealed planned-checkpoint directory changed"
            )
        self.tree.revalidate()

    def close(self) -> None:
        self.tree.close()
        if self.descriptor >= 0:
            descriptor, self.descriptor = self.descriptor, -1
            os.close(descriptor)


def _independently_verify_resume_checkpoint(
    branch: _HeldBranchDirectory,
    *,
    checkpoint_evidence: Mapping[str, Any],
    heartbeat_evidence: Mapping[str, Any],
) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(
        _MUTABLE_CHECKPOINT_DIRECTORY,
        flags,
        dir_fd=branch.descriptor,
    )
    checkpoint_fd = heartbeat_fd = -1
    try:
        checkpoint_fd, observed_checkpoint = _HeldPlannedCheckpoint._open_leaf(
            descriptor, "training_checkpoint.pt"
        )
        heartbeat_fd, observed_heartbeat = _HeldPlannedCheckpoint._open_leaf(
            descriptor, "training_heartbeat.json"
        )
        if (
            observed_checkpoint != dict(checkpoint_evidence)
            or observed_heartbeat != dict(heartbeat_evidence)
        ):
            raise TasteGlobalGCESmokeError(
                "T8 independent sealed-checkpoint verifier disagreed"
            )
    finally:
        for value in (heartbeat_fd, checkpoint_fd, descriptor):
            if value >= 0:
                os.close(value)


@dataclass(slots=True)
class _PostCallbackCheckpointSeal:
    sealed: _HeldSealedCheckpointTree
    planned_holder: _HeldPlannedCheckpoint
    evidence: Mapping[str, Any]
    resume_checkpoint_evidence: Mapping[str, Any]
    resume_heartbeat_evidence: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        branch: _HeldBranchDirectory,
        *,
        callback_checkpoint: Mapping[str, Any],
        callback_heartbeat: Mapping[str, Any],
    ) -> "_PostCallbackCheckpointSeal":
        """Atomically seal epoch zero after the generator stack has unwound."""

        branch.revalidate()
        for forbidden in (_SEALED_CHECKPOINT_DIRECTORY,):
            try:
                os.stat(
                    forbidden,
                    dir_fd=branch.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise TasteGlobalGCESmokeError(
                    "T8 sealed planned-checkpoint root must be fresh"
                )
        token = secrets.token_hex(16)
        seal_temporary = f".{_MUTABLE_CHECKPOINT_DIRECTORY}.seal-{token}.tmp"
        resume_temporary = f".{_MUTABLE_CHECKPOINT_DIRECTORY}.resume-{token}.tmp"
        sealed: _HeldSealedCheckpointTree | None = None
        planned_holder: _HeldPlannedCheckpoint | None = None
        working_tree: RetainedOutputTree | None = None
        working_fd = -1
        try:
            os.rename(
                _MUTABLE_CHECKPOINT_DIRECTORY,
                seal_temporary,
                src_dir_fd=branch.descriptor,
                dst_dir_fd=branch.descriptor,
            )
            os.fsync(branch.descriptor)
            flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            sealed_fd = os.open(
                seal_temporary,
                flags,
                dir_fd=branch.descriptor,
            )
            try:
                sealed_identity = _BranchDirectoryIdentity.from_stat(
                    os.fstat(sealed_fd)
                )
                callback_checkpoint_ctime = int(callback_checkpoint["ctime_ns"])
                callback_heartbeat_ctime = int(callback_heartbeat["ctime_ns"])
                settled_checkpoint, checkpoint_ctime_settled = (
                    _post_callback_settled_leaf(
                        sealed_fd,
                        "training_checkpoint.pt",
                        callback_evidence=callback_checkpoint,
                    )
                )
                settled_heartbeat, heartbeat_ctime_settled = (
                    _post_callback_settled_leaf(
                        sealed_fd,
                        "training_heartbeat.json",
                        callback_evidence=callback_heartbeat,
                    )
                )
                sealed_tree = RetainedOutputTree.capture(sealed_fd)
                sealed_tree.durably_flush()
                if (
                    sealed_tree.inventory["files"]["training_checkpoint.pt"][
                        "sha256"
                    ]
                    != settled_checkpoint["sha256"]
                    or sealed_tree.inventory["files"]["training_heartbeat.json"][
                        "sha256"
                    ]
                    != settled_heartbeat["sha256"]
                ):
                    sealed_tree.close()
                    raise TasteGlobalGCESmokeError(
                        "T8 sealed tree differs from post-callback leaves"
                    )
                os.rename(
                    seal_temporary,
                    _SEALED_CHECKPOINT_DIRECTORY,
                    src_dir_fd=branch.descriptor,
                    dst_dir_fd=branch.descriptor,
                )
                os.fsync(branch.descriptor)
                sealed = _HeldSealedCheckpointTree(
                    branch,
                    _SEALED_CHECKPOINT_DIRECTORY,
                    sealed_fd,
                    sealed_identity,
                    sealed_tree,
                )
                sealed.revalidate()
                sealed_fd = -1
            finally:
                if sealed_fd >= 0:
                    os.close(sealed_fd)

            sealed_content = _checkpoint_content_manifest(sealed.tree.inventory)
            sealed_content_sha256 = _canonical_sha256(sealed_content)
            working_fd = _copy_retained_checkpoint_tree(
                sealed.tree,
                parent_fd=branch.descriptor,
                name=resume_temporary,
            )
            working_tree = RetainedOutputTree.capture(working_fd)
            working_tree.durably_flush()
            resume_content = _checkpoint_content_manifest(working_tree.inventory)
            resume_content_sha256 = _canonical_sha256(resume_content)
            if resume_content != sealed_content:
                raise TasteGlobalGCESmokeError(
                    "T8 resume copy differs from sealed checkpoint tree"
                )
            try:
                os.stat(
                    _MUTABLE_CHECKPOINT_DIRECTORY,
                    dir_fd=branch.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise TasteGlobalGCESmokeError(
                    "T8 mutable checkpoint directory reappeared before seal commit"
                )
            working_identity = _BranchDirectoryIdentity.from_stat(
                os.fstat(working_fd)
            )
            os.rename(
                resume_temporary,
                _MUTABLE_CHECKPOINT_DIRECTORY,
                src_dir_fd=branch.descriptor,
                dst_dir_fd=branch.descriptor,
            )
            os.fsync(branch.descriptor)
            if (
                _BranchDirectoryIdentity.from_stat(
                    os.stat(
                        _MUTABLE_CHECKPOINT_DIRECTORY,
                        dir_fd=branch.descriptor,
                        follow_symlinks=False,
                    )
                )
                != working_identity
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 atomic resume checkpoint directory identity changed"
                )
            resume_checkpoint, _ = _post_callback_settled_leaf(
                working_fd,
                "training_checkpoint.pt",
                callback_evidence=None,
            )
            resume_heartbeat, _ = _post_callback_settled_leaf(
                working_fd,
                "training_heartbeat.json",
                callback_evidence=None,
            )
            if (
                resume_checkpoint["sha256"] != settled_checkpoint["sha256"]
                or resume_heartbeat["sha256"] != settled_heartbeat["sha256"]
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 atomic resume checkpoint hashes differ from seal"
                )
            working_tree.revalidate()
            working_tree.close()
            working_tree = None
            os.close(working_fd)
            working_fd = -1
            _independently_verify_resume_checkpoint(
                branch,
                checkpoint_evidence=resume_checkpoint,
                heartbeat_evidence=resume_heartbeat,
            )
            planned_holder = _HeldPlannedCheckpoint.capture(
                branch,
                checkpoint_evidence=resume_checkpoint,
                heartbeat_evidence=resume_heartbeat,
            )
            evidence = {
                "checkpoint_seal_schema_version": CHECKPOINT_SEAL_SCHEMA,
                "checkpoint_seal_pass": True,
                "checkpoint_writer_unwound": True,
                "checkpoint_durable_flush": True,
                "checkpoint_sealed_directory": _SEALED_CHECKPOINT_DIRECTORY,
                "checkpoint_sealed_inventory_sha256": sealed_content_sha256,
                "checkpoint_resume_copy_inventory_sha256": resume_content_sha256,
                "checkpoint_no_follow_identity_verified": True,
                "checkpoint_independent_reopen_verified": True,
                "checkpoint_callback_ctime_settled": (
                    checkpoint_ctime_settled or heartbeat_ctime_settled
                ),
                "checkpoint_callback_ctime_ns": callback_checkpoint_ctime,
                "checkpoint_sealed_ctime_ns": settled_checkpoint["ctime_ns"],
                "heartbeat_callback_ctime_ns": callback_heartbeat_ctime,
                "heartbeat_sealed_ctime_ns": settled_heartbeat["ctime_ns"],
            }
            result = cls(
                sealed,
                planned_holder,
                evidence,
                resume_checkpoint,
                resume_heartbeat,
            )
            result.revalidate()
            return result
        except BaseException:
            if planned_holder is not None:
                planned_holder.close()
            if working_tree is not None:
                working_tree.close()
            if working_fd >= 0:
                os.close(working_fd)
            if sealed is not None:
                sealed.close()
            raise

    def revalidate(self) -> None:
        self.sealed.revalidate()
        self.planned_holder.require_named_for_resume()

    def revalidate_after_resume(self) -> None:
        self.sealed.revalidate()
        self.planned_holder.revalidate_held()

    def close(self) -> None:
        self.planned_holder.close()
        self.sealed.close()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_git_oid(value: Any) -> bool:
    return type(value) is str and len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


def _require_sha256(value: Any, *, field: str) -> str:
    if not _is_sha256(value):
        raise TasteGlobalGCESmokeError(f"T8 {field} is not one SHA-256 digest")
    return str(value)


def _canonical_smiles(value: Any, *, field: str) -> str:
    if Chem is None:
        raise TasteGlobalGCESmokeError("T8 requires RDKit")
    text = str(value or "").strip()
    molecule = Chem.MolFromSmiles(text)
    if molecule is None or molecule.GetNumAtoms() <= 0:
        raise TasteGlobalGCESmokeError(f"T8 {field} is not a valid molecule")
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise TasteGlobalGCESmokeError(
            f"T8 {field} failed sanitization"
        ) from exc
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise TasteGlobalGCESmokeError(f"T8 {field} is disconnected")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    if not canonical or "." in canonical:
        raise TasteGlobalGCESmokeError(f"T8 {field} canonicalization failed")
    return canonical


def _exact_label(value: Any, *, row_number: int) -> int:
    if (
        type(value) is not str
        or value not in {"0", "1", "2"}
    ):
        raise TasteGlobalGCESmokeError(
            f"T8 train label at row {row_number} is not canonical 0/1/2 text"
        )
    return int(value)


@dataclass(frozen=True, slots=True)
class _TrainCohort:
    rows: tuple[TrainParent, ...]
    label_counts: Mapping[str, int]
    train_row_count: int


def load_taste_train_cohort(
    payload: bytes,
    *,
    expected_row_count: int,
    expected_label_counts: Mapping[str, int],
) -> _TrainCohort:
    """Parse only the held train CSV bytes and enforce its three-class schema."""

    if type(payload) is not bytes or not payload:
        raise TasteGlobalGCESmokeError("T8 held train payload is empty")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TasteGlobalGCESmokeError("T8 train CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fields = set(reader.fieldnames or ())
    # Match the frozen molecular-GNN dataset loader's schema precedence.  The
    # prepared Taste split intentionally carries raw, canonical, and model
    # columns; GINE was fitted from ``model_smiles`` and T8 must not silently
    # switch to another representation merely because ``smiles`` is absent.
    smiles_field = next(
        (
            name
            for name in (
                "model_smiles",
                "canonical_smiles",
                "smiles",
                "parent_smiles",
            )
            if name in fields
        ),
        "model_smiles",
    )
    required = {"molecule_id", "label", "split", smiles_field}
    if not required.issubset(fields):
        raise TasteGlobalGCESmokeError(
            f"T8 train CSV is missing columns: {sorted(required - fields)}"
        )
    rows: list[TrainParent] = []
    seen_ids: set[str] = set()
    counts = {"0": 0, "1": 0, "2": 0}
    for row_number, raw in enumerate(reader, start=2):
        parent_id = str(raw.get("molecule_id") or "").strip()
        if not parent_id or parent_id in seen_ids:
            raise TasteGlobalGCESmokeError(
                f"T8 train molecule identity is missing/duplicated at row {row_number}"
            )
        split = raw.get("split")
        if type(split) is not str or split != "train":
            raise TasteGlobalGCESmokeError(
                "T8 may consume only rows with explicit split=train"
            )
        label = _exact_label(raw.get("label"), row_number=row_number)
        canonical = _canonical_smiles(
            raw.get(smiles_field), field=f"train row {row_number}"
        )
        rows.append(TrainParent(parent_id, canonical, label, "train"))
        seen_ids.add(parent_id)
        counts[str(label)] += 1
    if (
        type(expected_row_count) is not int
        or expected_row_count <= 0
        or len(rows) != expected_row_count
    ):
        raise TasteGlobalGCESmokeError("T8 train row count differs from GINE authority")
    normalized_expected = {
        str(label): expected_label_counts.get(str(label)) for label in range(NUM_CLASSES)
    }
    if (
        set(expected_label_counts) != {"0", "1", "2"}
        or any(type(value) is not int or value <= 0 for value in normalized_expected.values())
        or counts != normalized_expected
    ):
        raise TasteGlobalGCESmokeError(
            "T8 train label counts differ from the frozen split manifest"
        )
    return _TrainCohort(tuple(rows), counts, len(rows))


def _validate_prediction(
    row: Mapping[str, Any],
    *,
    checkpoint_id: str,
    field: str,
) -> dict[str, Any]:
    probabilities = row.get("probabilities")
    logits = row.get("logits")
    predicted = row.get("predicted_label")
    if (
        type(predicted) is not int
        or predicted not in range(NUM_CLASSES)
        or type(probabilities) is not list
        or type(logits) is not list
        or len(probabilities) != NUM_CLASSES
        or len(logits) != NUM_CLASSES
        or any(
            type(value) not in (int, float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in [*probabilities, *logits]
        )
        or any(float(value) < 0.0 or float(value) > 1.0 for value in probabilities)
        or not math.isclose(sum(float(value) for value in probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6)
        or max(range(NUM_CLASSES), key=lambda index: float(probabilities[index]))
        != predicted
        or row.get("checkpoint_id") != checkpoint_id
        or row.get("num_classes") != NUM_CLASSES
        or row.get("source_label") != SOURCE_LABEL
        or str(row.get("backbone") or "").lower() != "gine"
    ):
        raise TasteGlobalGCESmokeError(
            f"T8 {field} is not an exact calibrated three-class GINE prediction"
        )
    return {
        "predicted_label": predicted,
        "probabilities": [float(value) for value in probabilities],
        "logits": [float(value) for value in logits],
    }


def select_bounded_sweet_parents(
    cohort: _TrainCohort,
    *,
    scorer: TastePredictionScorer,
    config: TasteGlobalGCESmokeConfig,
) -> tuple[list[TrainParent], dict[str, Any]]:
    """Select a deterministic bounded cohort predicted Sweet by the frozen GINE."""

    config.validate()
    if (
        scorer.num_classes != NUM_CLASSES
        or scorer.source_label != SOURCE_LABEL
        or not _is_sha256(scorer.checkpoint_id)
        or not math.isfinite(float(scorer.temperature))
        or float(scorer.temperature) <= 0.0
    ):
        raise TasteGlobalGCESmokeError("T8 scorer is not the frozen three-class GINE")
    sweet = [row for row in cohort.rows if row.label == SOURCE_LABEL]
    sweet.sort(
        key=lambda row: hashlib.sha256(
            f"{config.seed}\0{row.parent_id}\0{row.smiles}".encode("utf-8")
        ).hexdigest()
    )
    scanned = sweet[: config.source_scan_limit]
    if len(scanned) < config.source_parent_count:
        raise TasteGlobalGCESmokeError("T8 has too few bounded Sweet train parents")
    predictions = scorer.score_smiles([row.smiles for row in scanned])
    if len(predictions) != len(scanned):
        raise TasteGlobalGCESmokeError("T8 Sweet preselection prediction count changed")
    selected: list[TrainParent] = []
    for index, (parent, raw_prediction) in enumerate(
        zip(scanned, predictions, strict=True)
    ):
        prediction = _validate_prediction(
            raw_prediction,
            checkpoint_id=scorer.checkpoint_id,
            field=f"Sweet preselection row {index}",
        )
        if prediction["predicted_label"] == SOURCE_LABEL:
            selected.append(parent)
        if len(selected) == config.source_parent_count:
            break
    if len(selected) != config.source_parent_count:
        raise TasteGlobalGCESmokeError(
            "T8 bounded train scan did not yield enough true/predicted Sweet parents"
        )
    cohort_payload = [
        {
            "position": position,
            "parent_id": row.parent_id,
            "canonical_smiles": row.smiles,
            "label": row.label,
            "split": row.split,
        }
        for position, row in enumerate(selected)
    ]
    return selected, {
        "selection": "seeded_sha256_order_then_frozen_gine_predicted_sweet",
        "source_scan_limit": config.source_scan_limit,
        "source_scanned": len(scanned),
        "source_selected": len(selected),
        "all_true_source": True,
        "all_predicted_source": True,
        "selected_cohort_sha256": _canonical_sha256(cohort_payload),
    }


class FrozenTasteGINEScorer:
    """One loaded-once original-order GINE scorer for selection and final audit."""

    def __init__(
        self,
        payloads: Mapping[str, bytes],
        *,
        device: str = VISIBLE_DEVICE,
        batch_size: int = 256,
    ) -> None:
        from src.data.molecular_graph_featurizer import (
            MolecularFeatureSchema,
            MolecularGraphFeaturizer,
        )
        from src.oracles.gnn_oracle import GNNOracle

        if (
            type(payloads) is not dict
            or set(payloads) != FROZEN_GINE_IN_MEMORY_FILES
            or any(
                type(name) is not str or type(payload) is not bytes or not payload
                for name, payload in payloads.items()
            )
        ):
            raise TasteGlobalGCESmokeError(
                "T8 GINE payloads differ from the seven-file native-bytes contract"
            )
        copied = dict(payloads)
        self._oracle = GNNOracle.from_payloads(
            copied,
            device=device,
            batch_size=batch_size,
        )
        feature_payload = json.loads(copied["feature_schema.json"].decode("utf-8"))
        self._featurizer = MolecularGraphFeaturizer(
            MolecularFeatureSchema.from_dict(feature_payload)
        )
        self.checkpoint_id = str(self._oracle.checkpoint_id)
        self.num_classes = int(self._oracle.num_classes)
        self.source_label = int(self._oracle.source_label)
        self.temperature = float(self._oracle.temperature)
        self._batch_size = int(batch_size)
        if (
            self.num_classes != NUM_CLASSES
            or self.source_label != SOURCE_LABEL
            or str(self._oracle.backbone).lower() != "gine"
        ):
            raise TasteGlobalGCESmokeError(
                "T8 loaded classifier differs from the three-class Taste GINE"
            )

    def score_smiles(self, values: Sequence[str]) -> list[dict[str, Any]]:
        from src.data.molecular_graph_dataset import MolecularGraphData

        if not values:
            raise TasteGlobalGCESmokeError("T8 scorer cannot score an empty batch")
        graphs = []
        for position, value in enumerate(values):
            features = self._featurizer.featurize(str(value))
            graphs.append(
                MolecularGraphData(
                    x=features.node_features,
                    edge_index=features.edge_index,
                    edge_attr=features.edge_features,
                    y=SOURCE_LABEL,
                    molecule_id=f"private-t8-{position}",
                    smiles=features.canonical_smiles,
                    split="train",
                    graph_sha256=features.graph_sha256,
                )
            )
        return self._oracle.predict_records(graphs, batch_size=self._batch_size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 cannot read {label}") from exc
    if type(value) is not dict:
        raise TasteGlobalGCESmokeError(f"T8 {label} is not one JSON object")
    return value


def _read_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteGlobalGCESmokeError(f"T8 cannot decode {label}") from exc
    if type(value) is not dict:
        raise TasteGlobalGCESmokeError(f"T8 {label} is not one JSON object")
    return value


def _validate_epoch_event(
    event: Mapping[str, Any],
    *,
    expected_next_epoch: int,
    require_epoch: bool,
) -> dict[str, Any]:
    expected_keys = {
        "checkpoint_schema_version",
        "checkpoint_sha256",
        "checkpoint_file",
        "next_epoch",
        "resume_identity_sha256",
    }
    if require_epoch:
        expected_keys.update(
            {"epoch", "heartbeat_file", "checkpoint_and_heartbeat_durable"}
        )
    else:
        expected_keys.update(
            {
                "rng_state_restored",
                "model_state_restored",
                "optimizer_state_restored",
                "scheduler_state_restored",
            }
        )
    if set(event) != expected_keys:
        raise TasteGlobalGCESmokeError("T8 epoch checkpoint callback schema changed")
    if (
        event.get("checkpoint_schema_version")
        != GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
        or type(event.get("next_epoch")) is not int
        or event["next_epoch"] != expected_next_epoch
        or not _is_sha256(event.get("checkpoint_sha256"))
        or not _is_sha256(event.get("resume_identity_sha256"))
    ):
        raise TasteGlobalGCESmokeError("T8 epoch checkpoint identity changed")
    checkpoint_file = _normalize_retained_file_evidence(
        event.get("checkpoint_file"), field="epoch checkpoint"
    )
    if checkpoint_file["sha256"] != event["checkpoint_sha256"]:
        raise TasteGlobalGCESmokeError("T8 checkpoint callback hash changed")
    if require_epoch:
        _normalize_retained_file_evidence(
            event.get("heartbeat_file"), field="epoch heartbeat"
        )
        if (
            type(event.get("epoch")) is not int
            or event["epoch"] != expected_next_epoch - 1
            or event.get("checkpoint_and_heartbeat_durable") is not True
        ):
            raise TasteGlobalGCESmokeError(
                "T8 planned stop was not after the exact durable epoch checkpoint"
            )
    elif any(
        event.get(field) is not True
        for field in (
            "rng_state_restored",
            "model_state_restored",
            "optimizer_state_restored",
            "scheduler_state_restored",
        )
    ):
        raise TasteGlobalGCESmokeError(
            "T8 resume did not restore model/optimizer/scheduler/RNG state"
        )
    return dict(event)


def _generator_kwargs(
    *,
    branch_dir: Path,
    config: TasteGlobalGCESmokeConfig,
    on_resume_checkpoint: Callable[[dict[str, Any]], None] | None,
    after_epoch_checkpoint: Callable[[dict[str, Any]], None] | None,
    expected_resume_checkpoint: Mapping[str, Any] | None,
    on_generation_complete: Callable[[], None] | None,
) -> dict[str, Any]:
    return {
        "output_dir": branch_dir,
        "seed": config.seed,
        "epochs": config.epochs,
        "top_k_native": config.top_k_native,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "device": VISIBLE_DEVICE,
        "resume": True,
        "generation_chunk_size": config.generation_chunk_size,
        "generation_num_workers": 0,
        "memory_log_every_chunks": 1,
        "gspan_flush_every": config.gspan_flush_every,
        "gspan_max_in_memory_candidates": (
            config.gspan_max_in_memory_candidates
        ),
        "gspan_exact_top_k_pruning": True,
        "gspan_adoption_proof": None,
        "start_parent_offset": 0,
        "on_training_ready": None,
        "on_chunk": None,
        "rules_only": False,
        "expected_resume_checkpoint": expected_resume_checkpoint,
        "on_resume_checkpoint": on_resume_checkpoint,
        "after_epoch_checkpoint": after_epoch_checkpoint,
        "on_generation_complete": on_generation_complete,
    }


def _validate_official_startup_documents(
    *,
    api: Mapping[str, Any],
    provenance: Mapping[str, Any],
    api_sha256: str,
    provenance_sha256: str,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    expected_api = {
        "schema_version": OFFICIAL_GLOBALGCE_API_SIGNATURE_SCHEMA,
        "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
        "signatures": dict(PINNED_OFFICIAL_GLOBALGCE_API_SIGNATURES),
    }
    if api != expected_api:
        raise TasteGlobalGCESmokeError(
            "T8 official GlobalGCE API signature drifted"
        )

    if (
        set(provenance)
        != {
            "schema_version",
            "official_globalgce_commit",
            "isolated_python",
            "no_user_site",
            "entries",
        }
        or provenance.get("schema_version")
        != OFFICIAL_GLOBALGCE_MODULE_PROVENANCE_SCHEMA
        or provenance.get("official_globalgce_commit")
        != OFFICIAL_GLOBALGCE_COMMIT
        or provenance.get("isolated_python") is not True
        or provenance.get("no_user_site") is not True
        or type(provenance.get("entries")) is not list
        or not provenance["entries"]
    ):
        raise TasteGlobalGCESmokeError(
            "T8 isolated Python module provenance changed"
        )
    entry_keys = {
        "module",
        "module_file",
        "realpath",
        "sha256",
        "device",
        "inode",
        "bytes",
        "package_version",
        "expected_roots",
    }
    observed_modules: set[str] = set()
    for row in provenance["entries"]:
        if type(row) is not dict or set(row) != entry_keys:
            raise TasteGlobalGCESmokeError(
                "T8 Python module provenance entry changed"
            )
        module = row.get("module")
        module_file = row.get("module_file")
        realpath = row.get("realpath")
        roots = row.get("expected_roots")
        if (
            type(module) is not str
            or not module
            or module in observed_modules
            or type(module_file) is not str
            or not module_file.startswith("/")
            or module_file.endswith((".pyc", ".pyo"))
            or "__pycache__" in Path(module_file).parts
            or type(realpath) is not str
            or not realpath.startswith("/")
            or realpath.endswith((".pyc", ".pyo"))
            or not _is_sha256(row.get("sha256"))
            or any(
                type(row.get(field)) is not int or row[field] < minimum
                for field, minimum in (
                    ("device", 0),
                    ("inode", 1),
                    ("bytes", 1),
                )
            )
            or type(row.get("package_version")) is not str
            or not row["package_version"]
            or type(roots) is not list
            or not roots
            or any(type(root) is not str or not root.startswith("/") for root in roots)
            or not any(
                realpath == root or realpath.startswith(root.rstrip("/") + "/")
                for root in roots
            )
        ):
            raise TasteGlobalGCESmokeError(
                "T8 Python module provenance entry changed"
            )
        if module.startswith(("models.", "data.")) or module == "utils":
            if row["package_version"] != OFFICIAL_GLOBALGCE_COMMIT:
                raise TasteGlobalGCESmokeError(
                    "T8 official module commit provenance changed"
                )
        elif module.startswith("src.") and row["package_version"] != "project-source":
            raise TasteGlobalGCESmokeError(
                "T8 project module provenance changed"
            )
        observed_modules.add(module)
    required_modules = {
        "models.GTGNN",
        "models.GlobalGCE",
        "models.models_utils",
        "models.fsg",
        "models.gSpan.gSpan",
        "torch",
        "torch_geometric",
        "src.baselines.globalgce_mutagenicity_adapter",
        "src.baselines.globalgce_frozen_gine_bridge",
        "src.oracles.gnn_oracle",
    }
    if not required_modules.issubset(observed_modules):
        raise TasteGlobalGCESmokeError(
            "T8 required Python module provenance is incomplete"
        )
    if not _is_sha256(api_sha256) or not _is_sha256(provenance_sha256):
        raise TasteGlobalGCESmokeError(
            "T8 startup identity hashes are malformed"
        )
    if (
        training_summary.get("official_globalgce_commit")
        != OFFICIAL_GLOBALGCE_COMMIT
        or training_summary.get("official_api_signature_sha256")
        != api_sha256
        or training_summary.get("python_module_provenance_sha256")
        != provenance_sha256
        or training_summary.get("isolated_python") is not True
        or training_summary.get("no_user_site") is not True
    ):
        raise TasteGlobalGCESmokeError(
            "T8 training summary does not bind its startup identity"
        )
    return {
        "official_globalgce_commit": OFFICIAL_GLOBALGCE_COMMIT,
        "official_api_signature_sha256": api_sha256,
        "python_module_provenance_sha256": provenance_sha256,
        "isolated_python": True,
        "no_user_site": True,
    }


def _validate_official_startup_identity(
    *,
    completed_tree: RetainedOutputTree,
    api_relative: str,
    provenance_relative: str,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return _validate_official_startup_documents(
        api=_read_json_bytes(
            completed_tree.read_bytes(api_relative),
            label="official API signature",
        ),
        provenance=_read_json_bytes(
            completed_tree.read_bytes(provenance_relative),
            label="Python module provenance",
        ),
        api_sha256=completed_tree.inventory["files"][api_relative]["sha256"],
        provenance_sha256=completed_tree.inventory["files"][
            provenance_relative
        ]["sha256"],
        training_summary=training_summary,
    )


def run_resumed_target_branch(
    *,
    target_label: int,
    generator: TasteBranchGenerator,
    parents: Sequence[TrainParent],
    branch: _HeldBranchDirectory,
    config: TasteGlobalGCESmokeConfig,
) -> tuple[NativeGenerationResult, dict[str, Any], RetainedOutputTree]:
    """Run one real branch twice around its first durable epoch checkpoint."""

    config.validate()
    _require_exact_branch_generator_signature(generator)
    if (
        type(target_label) is not int
        or target_label not in TARGET_BRANCHES
        or type(getattr(generator, "target_label", None)) is not int
        or generator.target_label != target_label
    ):
        raise TasteGlobalGCESmokeError("T8 branch target identity changed")
    if not parents or any(
        type(parent.label) is not int
        or parent.label != SOURCE_LABEL
        or parent.split != "train"
        for parent in parents
    ):
        raise TasteGlobalGCESmokeError(
            "T8 branch requires a nonempty exact Sweet train cohort"
        )
    branch.revalidate()
    if os.listdir(branch.descriptor):
        raise TasteGlobalGCESmokeError("T8 retained branch root is not fresh")
    branch_dir = branch.runtime_path

    planned: list[dict[str, Any]] = []
    checkpoint_seal: _PostCallbackCheckpointSeal | None = None

    def _planned_stop(event: dict[str, Any]) -> None:
        if planned:
            raise TasteGlobalGCESmokeError(
                "T8 branch attempted more than one planned checkpoint stop"
            )
        normalized = _validate_epoch_event(
            event,
            expected_next_epoch=1,
            require_epoch=True,
        )
        planned.append(normalized)
        raise PlannedGlobalGCECheckpointStop(
            f"planned T8 target-{target_label} checkpoint boundary"
        )

    try:
        generator.generate(
            parents,
            **_generator_kwargs(
                branch_dir=branch_dir,
                config=config,
                on_resume_checkpoint=None,
                after_epoch_checkpoint=_planned_stop,
                expected_resume_checkpoint=None,
                on_generation_complete=None,
            ),
        )
    except PlannedGlobalGCECheckpointStop:
        pass
    else:
        raise TasteGlobalGCESmokeError(
            "T8 branch completed without its mandatory planned interruption"
        )
    if len(planned) != 1:
        raise TasteGlobalGCESmokeError("T8 branch lacks one planned checkpoint")
    checkpoint_seal = _PostCallbackCheckpointSeal.create(
        branch,
        callback_checkpoint=planned[0]["checkpoint_file"],
        callback_heartbeat=planned[0]["heartbeat_file"],
    )
    checkpoint_seal.revalidate()

    resumed: list[dict[str, Any]] = []

    def _record_resume(event: dict[str, Any]) -> None:
        if resumed:
            raise TasteGlobalGCESmokeError(
                "T8 branch adopted more than one epoch checkpoint"
            )
        normalized = _validate_epoch_event(
            event,
            expected_next_epoch=1,
            require_epoch=False,
        )
        if (
            normalized["checkpoint_file"]
            != checkpoint_seal.resume_checkpoint_evidence
        ):
            raise TasteGlobalGCESmokeError(
                "T8 resumed checkpoint physical leaf differs from the planned leaf"
            )
        resumed.append(normalized)

    completed_tree: RetainedOutputTree | None = None

    def _capture_completed_tree() -> None:
        nonlocal completed_tree
        if completed_tree is not None:
            raise TasteGlobalGCESmokeError(
                "T8 branch attempted more than one terminal tree capture"
            )
        branch.revalidate()
        completed_tree = RetainedOutputTree.capture(branch.descriptor)
        completed_tree.durably_flush()
        branch.revalidate()

    try:
        result = generator.generate(
            parents,
            **_generator_kwargs(
                branch_dir=branch_dir,
                config=config,
                on_resume_checkpoint=_record_resume,
                after_epoch_checkpoint=None,
                expected_resume_checkpoint=(
                    checkpoint_seal.resume_checkpoint_evidence
                ),
                on_generation_complete=_capture_completed_tree,
            ),
        )
    except BaseException:
        if completed_tree is not None:
            completed_tree.close()
        checkpoint_seal.close()
        raise
    if len(resumed) != 1:
        raise TasteGlobalGCESmokeError(
            "T8 branch did not adopt exactly one durable epoch checkpoint"
        )
    if completed_tree is None:
        checkpoint_seal.close()
        raise TasteGlobalGCESmokeError(
            "T8 generator returned without retaining its completed branch tree"
        )
    if (
        planned[0]["checkpoint_sha256"] != resumed[0]["checkpoint_sha256"]
        or planned[0]["resume_identity_sha256"]
        != resumed[0]["resume_identity_sha256"]
    ):
        raise TasteGlobalGCESmokeError(
            "T8 branch resumed a checkpoint other than the planned durable state"
        )
    checkpoint_seal.revalidate_after_resume()
    completed_tree.revalidate()
    branch.revalidate()

    paths = {
        "training heartbeat": (
            "globalgce_training_checkpoints/training_heartbeat.json"
        ),
        "training checkpoint": (
            "globalgce_training_checkpoints/training_checkpoint.pt"
        ),
        "training core": "training_core_summary.json",
        "native rule catalog": "native_rule_catalog.jsonl",
        "GlobalGCE model checkpoint": "globalgce_model.pt",
        "GlobalGCE rule checkpoint": "globalgce_rules.pt",
        "official API signature": OFFICIAL_API_SIGNATURE_FILE,
        "Python module provenance": PYTHON_MODULE_PROVENANCE_FILE,
    }
    for label, relative in paths.items():
        entry = completed_tree.inventory["files"].get(relative)
        if type(entry) is not dict or entry.get("bytes", 0) <= 0:
            completed_tree.close()
            checkpoint_seal.close()
            raise TasteGlobalGCESmokeError(f"T8 {label} is absent after resume")
    heartbeat = _read_json_bytes(
        completed_tree.read_bytes(paths["training heartbeat"]),
        label="terminal training heartbeat",
    )
    if (
        heartbeat.get("schema_version") != GLOBALGCE_EPOCH_CHECKPOINT_SCHEMA_V2
        or heartbeat.get("stage") != "complete"
        or type(heartbeat.get("next_epoch")) is not int
        or heartbeat["next_epoch"] != config.epochs + 1
        or heartbeat.get("resume_identity_sha256")
        != resumed[0]["resume_identity_sha256"]
    ):
        completed_tree.close()
        checkpoint_seal.close()
        raise TasteGlobalGCESmokeError(
            "T8 branch terminal heartbeat does not close the resumed identity"
        )
    core = _read_json_bytes(
        completed_tree.read_bytes(paths["training core"]),
        label="training core summary",
    )
    identity = core.get("training_resume_identity")
    if (
        core.get("trained_once") is not True
        or core.get("rule_selection_performed_once") is not True
        or core.get("num_classes") != NUM_CLASSES
        or core.get("source_label") != SOURCE_LABEL
        or core.get("target_label") != target_label
        or core.get("prediction_backend") != "frozen_gine_differentiable_bridge"
        or core.get("rf_oracle_used") is not False
        or core.get("training_resume_identity_sha256")
        != resumed[0]["resume_identity_sha256"]
        or core.get("globalgce_model_checkpoint_sha256")
        != completed_tree.inventory["files"][paths["GlobalGCE model checkpoint"]][
            "sha256"
        ]
        or core.get("rules_checkpoint_sha256")
        != completed_tree.inventory["files"][paths["GlobalGCE rule checkpoint"]][
            "sha256"
        ]
        or type(identity) is not dict
        or identity.get("schema_version")
        != GLOBALGCE_TRAINING_RESUME_IDENTITY_SCHEMA_VERSION
        or identity.get("dataset") not in {"TasteMolNet", "tastemolnet"}
        or identity.get("num_classes") != NUM_CLASSES
        or identity.get("source_label") != SOURCE_LABEL
        or identity.get("target_label") != target_label
    ):
        completed_tree.close()
        checkpoint_seal.close()
        raise TasteGlobalGCESmokeError(
            "T8 branch terminal training identity changed"
        )
    summary = result.training_summary
    if (
        type(summary) is not dict
        or summary.get("prediction_backend")
        != "frozen_gine_differentiable_bridge"
        or summary.get("classifier_family") != "gine"
        or summary.get("oracle_backend") != "gnn"
        or summary.get("rf_oracle_used") is not False
        or summary.get("num_classes") != NUM_CLASSES
        or summary.get("frozen_source_label") != SOURCE_LABEL
        or summary.get("frozen_target_label") != target_label
        or summary.get("calibration_loaded") is not False
        or summary.get("test_loaded") is not False
        or summary.get("generation_input_split") != "train"
        or summary.get("training_resume_identity_sha256")
        != resumed[0]["resume_identity_sha256"]
        or summary.get("native_rule_catalog_sha256")
        != completed_tree.inventory["files"][paths["native rule catalog"]]["sha256"]
        or type(summary.get("valid_native_rule_count")) is not int
        or summary["valid_native_rule_count"] < config.top_k_native
    ):
        completed_tree.close()
        checkpoint_seal.close()
        raise TasteGlobalGCESmokeError(
            "T8 branch result does not preserve the frozen multiclass contract"
        )
    startup_identity = _validate_official_startup_identity(
        completed_tree=completed_tree,
        api_relative=paths["official API signature"],
        provenance_relative=paths["Python module provenance"],
        training_summary=summary,
    )
    evidence = {
        "schema_version": BRANCH_SCHEMA,
        **startup_identity,
        "target_label": target_label,
        "source_label": SOURCE_LABEL,
        "num_classes": NUM_CLASSES,
        "planned_checkpoint_stop_observed": True,
        "planned_checkpoint_next_epoch": planned[0]["next_epoch"],
        "planned_checkpoint_sha256": planned[0]["checkpoint_sha256"],
        "resume_checkpoint_adopted": True,
        "resume_checkpoint_sha256": resumed[0]["checkpoint_sha256"],
        "resume_identity_sha256": resumed[0]["resume_identity_sha256"],
        "model_state_restored": True,
        "optimizer_state_restored": True,
        "scheduler_state_restored": True,
        "rng_state_restored": True,
        "terminal_next_epoch": heartbeat["next_epoch"],
        "terminal_training_checkpoint_sha256": completed_tree.inventory["files"][
            paths["training checkpoint"]
        ]["sha256"],
        "terminal_training_core_sha256": completed_tree.inventory["files"][
            paths["training core"]
        ]["sha256"],
        "terminal_model_checkpoint_sha256": completed_tree.inventory["files"][
            paths["GlobalGCE model checkpoint"]
        ]["sha256"],
        "terminal_rule_checkpoint_sha256": completed_tree.inventory["files"][
            paths["GlobalGCE rule checkpoint"]
        ]["sha256"],
        "native_rule_catalog_sha256": completed_tree.inventory["files"][
            paths["native rule catalog"]
        ]["sha256"],
        "valid_native_rule_count": summary["valid_native_rule_count"],
        "raw_generated_count": len(result.records),
        "train_only": True,
        "external_validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        **checkpoint_seal.evidence,
    }
    checkpoint_seal.close()
    completed_tree.revalidate()
    branch.revalidate()
    return result, evidence, completed_tree


def _iter_jsonl_payload(payload: bytes) -> Iterable[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TasteGlobalGCESmokeError(
            "T8 native rule catalog is not UTF-8"
        ) from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TasteGlobalGCESmokeError(
                f"T8 native rule catalog is invalid at row {line_number}"
            ) from exc
        if type(value) is not dict:
            raise TasteGlobalGCESmokeError(
                "T8 native rule catalog row is not an object"
            )
        yield value


def _canonical_rule_action(row: Mapping[str, Any]) -> dict[str, str]:
    source = row.get("rule")
    if type(source) is not dict:
        raise TasteGlobalGCESmokeError("T8 native rule row lacks its tensor payload")
    exact = {
        "lhs_feature": source.get("lhs_feature"),
        "lhs_adjacency": source.get("lhs_adjacency"),
        "lhs_edge_attr": source.get("lhs_edge_attr"),
        "rhs_feature": source.get("rhs_feature"),
        "rhs_adjacency": source.get("rhs_adjacency"),
        "rhs_edge_attr": source.get("rhs_edge_attr"),
        "atom_symbols": source.get("atom_symbols"),
        "bond_names": source.get("bond_names"),
    }
    if any(value is None for value in exact.values()):
        raise TasteGlobalGCESmokeError("T8 native rule tensor payload is incomplete")
    lhs = {
        "feature": exact["lhs_feature"],
        "adjacency": exact["lhs_adjacency"],
        "edge_attr": exact["lhs_edge_attr"],
        "atom_symbols": exact["atom_symbols"],
        "bond_names": exact["bond_names"],
    }
    rhs = {
        "feature": exact["rhs_feature"],
        "adjacency": exact["rhs_adjacency"],
        "edge_attr": exact["rhs_edge_attr"],
        "atom_symbols": exact["atom_symbols"],
        "bond_names": exact["bond_names"],
    }
    attachment = {
        "engine": ACTION_ENGINE_VERSION,
        "semantics": "native_lhs_to_rhs_attachment_aware_v1",
        "mask_order": "official_mapping_keys_then_appended_nodes",
        "external_boundary_edges": "preserved",
        "lhs_width": len(exact["lhs_feature"]),
        "rhs_width": len(exact["rhs_feature"]),
        "atom_symbols": exact["atom_symbols"],
        "bond_names": exact["bond_names"],
    }
    lhs_hash = _canonical_sha256(lhs)
    rhs_hash = _canonical_sha256(rhs)
    attachment_hash = _canonical_sha256(attachment)
    return {
        "lhs_hash": lhs_hash,
        "rhs_hash": rhs_hash,
        "attachment_map_hash": attachment_hash,
        "rule_hash": _canonical_sha256(
            {
                "lhs_hash": lhs_hash,
                "rhs_hash": rhs_hash,
                "attachment_map_hash": attachment_hash,
                "action_kind": "lhs_rhs_graph_transformation_rule",
            }
        ),
    }


def merge_branch_rule_catalogs(
    *,
    branch_trees: Mapping[int, RetainedOutputTree],
    checkpoint_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(branch_trees) != set(TARGET_BRANCHES):
        raise TasteGlobalGCESmokeError("T8 rule merge requires branches 0 and 2")
    branches: dict[int, list[dict[str, Any]]] = {}
    branch_hashes: dict[int, set[str]] = {}
    for target in TARGET_BRANCHES:
        rows: list[dict[str, Any]] = []
        hashes: set[str] = set()
        branch_trees[target].revalidate()
        for raw in _iter_jsonl_payload(
            branch_trees[target].read_bytes("native_rule_catalog.jsonl")
        ):
            action = _canonical_rule_action(raw)
            if action["rule_hash"] in hashes:
                raise TasteGlobalGCESmokeError(
                    "T8 one target branch emitted a duplicate canonical native rule"
                )
            hashes.add(action["rule_hash"])
            rows.append(
                {
                    **action,
                    "action_kind": "lhs_rhs_graph_transformation_rule",
                    "target_label": target,
                    "source_label": SOURCE_LABEL,
                    "data_split_used": "train",
                    "calibration_loaded": False,
                    "test_loaded": False,
                    "rf_oracle_used": False,
                    "oracle_backend": "gnn",
                    "oracle_checkpoint_hash": checkpoint_id,
                }
            )
        if not rows:
            raise TasteGlobalGCESmokeError(
                f"T8 target-{target} branch produced no valid native rules"
            )
        branches[target] = rows
        branch_hashes[target] = hashes
    merged = merge_globalgce_target_branches(
        branches,
        oracle_checkpoint_hash=checkpoint_id,
    )
    if not merged or len({row["rule_hash"] for row in merged}) != len(merged):
        raise TasteGlobalGCESmokeError("T8 canonical rule merge is empty or duplicated")
    overlap = branch_hashes[0] & branch_hashes[2]
    return merged, {
        "merge_stage": "after_two_train_only_branches_before_calibration",
        "dedup_identity": (
            "sha256(canonical_lhs,rhs,official_attachment_map,action_kind)"
        ),
        "target_0_rule_count": len(branches[0]),
        "target_2_rule_count": len(branches[2]),
        "premerge_rule_count": len(branches[0]) + len(branches[2]),
        "cross_branch_duplicate_count": len(overlap),
        "merged_unique_rule_count": len(merged),
        "hash_collision_or_action_mismatch": False,
        "canonical_dedup_complete": True,
        "merged_rule_set_sha256": _canonical_sha256(
            [
                {
                    "rule_hash": row["rule_hash"],
                    "target_branches": row["target_branches"],
                }
                for row in merged
            ]
        ),
    }


def _deduplicate_generated_candidates(
    records_by_branch: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    selected_parents: Sequence[TrainParent],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(records_by_branch) != set(TARGET_BRANCHES):
        raise TasteGlobalGCESmokeError(
            "T8 candidate merge requires both destination branches"
        )
    selected_by_id = {row.parent_id: row for row in selected_parents}
    if len(selected_by_id) != len(selected_parents):
        raise TasteGlobalGCESmokeError("T8 selected parent IDs are duplicated")
    merged: dict[str, dict[str, Any]] = {}
    branch_valid_counts = {0: 0, 2: 0}
    raw_count = 0
    invalid_count = 0
    for target in TARGET_BRANCHES:
        for raw in records_by_branch[target]:
            raw_count += 1
            if (
                raw.get("source_split") != "train"
                or raw.get("generator_method") != METHOD
                or raw.get("native_conversion_ok") is not True
                or raw.get("native_codec_decoded") is not True
            ):
                invalid_count += 1
                continue
            parent_id = str(raw.get("source_parent_id") or "")
            parent = selected_by_id.get(parent_id)
            if parent is None:
                raise TasteGlobalGCESmokeError(
                    "T8 native candidate escaped its selected train cohort"
                )
            parent_smiles = _canonical_smiles(
                raw.get("source_parent_smiles"), field="generated candidate parent"
            )
            if parent_smiles != parent.smiles:
                raise TasteGlobalGCESmokeError(
                    "T8 generated candidate changed its parent graph"
                )
            try:
                candidate_smiles = _canonical_smiles(
                    raw.get("raw_smiles"), field="generated candidate"
                )
            except TasteGlobalGCESmokeError:
                invalid_count += 1
                continue
            identity = _canonical_sha256(
                {
                    "parent": parent_smiles,
                    "candidate": candidate_smiles,
                    "action_kind": "lhs_rhs_graph_transformation_rule",
                }
            )
            branch_valid_counts[target] += 1
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = {
                    "parent": parent_smiles,
                    "candidate": candidate_smiles,
                    "target_branches": {target},
                }
            else:
                if (
                    existing["parent"] != parent_smiles
                    or existing["candidate"] != candidate_smiles
                ):
                    raise TasteGlobalGCESmokeError(
                        "T8 canonical candidate hash collision or corruption"
                    )
                existing["target_branches"].add(target)
    candidates = [
        {
            "identity": identity,
            "parent": row["parent"],
            "candidate": row["candidate"],
            "target_branches": sorted(row["target_branches"]),
        }
        for identity, row in sorted(merged.items())
    ]
    if not candidates or any(value <= 0 for value in branch_valid_counts.values()):
        raise TasteGlobalGCESmokeError(
            "T8 native branches did not produce valid connected candidate graphs"
        )
    return candidates, {
        "raw_generated_count": raw_count,
        "invalid_or_noncanonical_count": invalid_count,
        "target_0_valid_count": branch_valid_counts[0],
        "target_2_valid_count": branch_valid_counts[2],
        "canonical_unique_parent_candidate_count": len(candidates),
        "canonical_candidate_dedup_complete": True,
        "candidate_hash_collision_or_corruption": False,
    }


def _mean_columns(values: Sequence[Sequence[float]]) -> list[float]:
    if not values or any(len(row) != NUM_CLASSES for row in values):
        raise TasteGlobalGCESmokeError("T8 cannot aggregate malformed class vectors")
    return [
        sum(float(row[column]) for row in values) / len(values)
        for column in range(NUM_CLASSES)
    ]


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise TasteGlobalGCESmokeError("T8 metric aggregation is empty/non-finite")
    normalized = [float(value) for value in values]
    return {
        "mean": sum(normalized) / len(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def validate_candidates_with_original_gine(
    candidates: Sequence[Mapping[str, Any]],
    *,
    scorer: TastePredictionScorer,
    checkpoint_id: str,
    minimum_strict_flips_per_branch: int,
) -> dict[str, Any]:
    """Score final graphs in original class order and emit aggregates only."""

    if not candidates:
        raise TasteGlobalGCESmokeError("T8 has no canonical candidates to validate")
    if (
        scorer.checkpoint_id != checkpoint_id
        or scorer.num_classes != NUM_CLASSES
        or scorer.source_label != SOURCE_LABEL
        or type(minimum_strict_flips_per_branch) is not int
        or minimum_strict_flips_per_branch <= 0
    ):
        raise TasteGlobalGCESmokeError("T8 final scorer/class contract changed")
    score_values: list[str] = []
    for row in candidates:
        score_values.extend((str(row["parent"]), str(row["candidate"])))
    raw_predictions = scorer.score_smiles(score_values)
    if len(raw_predictions) != len(score_values):
        raise TasteGlobalGCESmokeError("T8 final prediction count changed")

    before_logits: list[list[float]] = []
    after_logits: list[list[float]] = []
    before_probabilities: list[list[float]] = []
    after_probabilities: list[list[float]] = []
    source_drops: list[float] = []
    margin_drops: list[float] = []
    destination_counts = {0: 0, 2: 0}
    strict_by_branch = {0: 0, 2: 0}
    strict_count = 0
    source_parent_failures = 0
    for index, row in enumerate(candidates):
        before = _validate_prediction(
            raw_predictions[index * 2],
            checkpoint_id=checkpoint_id,
            field=f"final parent prediction {index}",
        )
        after = _validate_prediction(
            raw_predictions[index * 2 + 1],
            checkpoint_id=checkpoint_id,
            field=f"final candidate prediction {index}",
        )
        before_logits.append(before["logits"])
        after_logits.append(after["logits"])
        before_probabilities.append(before["probabilities"])
        after_probabilities.append(after["probabilities"])
        if before["predicted_label"] != SOURCE_LABEL:
            source_parent_failures += 1
            continue
        strict = after["predicted_label"] != SOURCE_LABEL
        if not strict:
            continue
        destination = after["predicted_label"]
        if destination not in DESTINATION_LABELS:
            raise TasteGlobalGCESmokeError(
                "T8 strict-flip destination escaped Bitter/Tasteless"
            )
        strict_count += 1
        destination_counts[destination] += 1
        branches = row.get("target_branches")
        if (
            type(branches) is not list
            or not branches
            or any(type(value) is not int or value not in TARGET_BRANCHES for value in branches)
        ):
            raise TasteGlobalGCESmokeError(
                "T8 candidate branch provenance is malformed"
            )
        for branch in branches:
            strict_by_branch[branch] += 1
        before_source = before["probabilities"][SOURCE_LABEL]
        after_source = after["probabilities"][SOURCE_LABEL]
        before_margin = before_source - max(
            value
            for label, value in enumerate(before["probabilities"])
            if label != SOURCE_LABEL
        )
        after_margin = after_source - max(
            value
            for label, value in enumerate(after["probabilities"])
            if label != SOURCE_LABEL
        )
        source_drops.append(before_source - after_source)
        margin_drops.append(before_margin - after_margin)
    if source_parent_failures:
        raise TasteGlobalGCESmokeError(
            "T8 final audit found a selected parent no longer predicted Sweet"
        )
    if strict_count <= 0 or any(
        strict_by_branch[branch] < minimum_strict_flips_per_branch
        for branch in TARGET_BRANCHES
    ):
        raise TasteGlobalGCESmokeError(
            "T8 each target branch must contribute an untargeted strict flip"
        )
    return {
        "scored_parent_candidate_pairs": len(candidates),
        "all_parents_predicted_source": True,
        "strict_flip_definition": "pred_before == 1 and pred_after != 1",
        "strict_flip_count": strict_count,
        "strict_flip_rate": strict_count / len(candidates),
        "strict_flip_by_training_branch": {
            "0": strict_by_branch[0],
            "2": strict_by_branch[2],
        },
        "destination_distribution": {
            "0": destination_counts[0],
            "2": destination_counts[2],
        },
        "all_destinations_in_0_or_2": True,
        "class_order": ["Bitter", "Sweet", "Tasteless"],
        "all_class_logits_validated": True,
        "all_class_probabilities_validated": True,
        "mean_logits_before": _mean_columns(before_logits),
        "mean_logits_after": _mean_columns(after_logits),
        "mean_probabilities_before": _mean_columns(before_probabilities),
        "mean_probabilities_after": _mean_columns(after_probabilities),
        "source_probability_drop": _metric_summary(source_drops),
        "source_vs_best_alternative_margin_drop": _metric_summary(margin_drops),
        "binary_projection_used": False,
        "rf_oracle_used": False,
        "individual_outputs_serialized": False,
    }


def _inventory_digest(inventory: Mapping[str, Any]) -> str:
    return _canonical_sha256(dict(inventory))


def run_t8_science(
    *,
    train_payload: bytes,
    expected_train_row_count: int,
    expected_train_label_counts: Mapping[str, int],
    scorer: TastePredictionScorer,
    generator_factory: Callable[[int], TasteBranchGenerator],
    state_root: FreshOutputDirectory,
    config: TasteGlobalGCESmokeConfig,
) -> tuple[dict[str, Any], RetainedOutputTree]:
    """Execute bounded two-branch science and retain its private state tree."""

    config.validate()
    state_root.revalidate()
    cohort = load_taste_train_cohort(
        train_payload,
        expected_row_count=expected_train_row_count,
        expected_label_counts=expected_train_label_counts,
    )
    parents, selection = select_bounded_sweet_parents(
        cohort,
        scorer=scorer,
        config=config,
    )
    branch_results: dict[int, NativeGenerationResult] = {}
    branch_evidence: dict[int, dict[str, Any]] = {}
    branch_handles: dict[int, _HeldBranchDirectory] = {}
    branch_trees: dict[int, RetainedOutputTree] = {}
    state_tree: RetainedOutputTree | None = None
    try:
        for target in TARGET_BRANCHES:
            generator = generator_factory(target)
            branch = _HeldBranchDirectory.create(
                state_root, target_label=target
            )
            branch_handles[target] = branch
            result, evidence, branch_tree = run_resumed_target_branch(
                target_label=target,
                generator=generator,
                parents=parents,
                branch=branch,
                config=config,
            )
            branch_results[target] = result
            branch_evidence[target] = evidence
            branch_trees[target] = branch_tree
        _merged_rules, merge_summary = merge_branch_rule_catalogs(
            branch_trees=branch_trees,
            checkpoint_id=scorer.checkpoint_id,
        )
        candidates, candidate_summary = _deduplicate_generated_candidates(
            {target: branch_results[target].records for target in TARGET_BRANCHES},
            selected_parents=parents,
        )
        strict_summary = validate_candidates_with_original_gine(
            candidates,
            scorer=scorer,
            checkpoint_id=scorer.checkpoint_id,
            minimum_strict_flips_per_branch=(
                config.minimum_strict_flips_per_branch
            ),
        )
        state_tree = RetainedOutputTree.capture(state_root.descriptor)
        state_tree.durably_flush()
        for target in TARGET_BRANCHES:
            branch_handles[target].revalidate()
            branch_inventory = branch_trees[target].revalidate()
            if (
                state_tree.inventory["directories"].get(f"target-{target}")
                != branch_inventory["root"]
            ):
                raise TasteGlobalGCESmokeError(
                    "T8 retained branch root differs from the captured private state"
                )
            for relative, evidence in branch_inventory["directories"].items():
                state_relative = f"target-{target}/{relative}"
                if state_tree.inventory["directories"].get(state_relative) != evidence:
                    raise TasteGlobalGCESmokeError(
                        "T8 retained branch directory differs from the captured private state"
                    )
            for relative, evidence in branch_inventory["files"].items():
                state_relative = f"target-{target}/{relative}"
                if state_tree.inventory["files"].get(state_relative) != evidence:
                    raise TasteGlobalGCESmokeError(
                        "T8 retained branch differs from the captured private state"
                    )
        state_root.revalidate()
        science = {
        "schema_version": SCIENCE_SCHEMA,
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "status": "SCIENCE_PASS_PENDING_TERMINAL_COMMIT",
        "num_classes": NUM_CLASSES,
        "source_label": SOURCE_LABEL,
        "target_branches": list(TARGET_BRANCHES),
        "oracle_backend": "gnn",
        "classifier_family": "gine",
        "oracle_checkpoint_hash": scorer.checkpoint_id,
        "temperature_hex": float(scorer.temperature).hex(),
        "config": config.to_dict(),
        "train_boundary": {
            "train_loaded": True,
            "train_row_count": cohort.train_row_count,
            "train_label_counts": dict(cohort.label_counts),
            "external_validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "data_reprepared": False,
            "data_redistributed": False,
        },
        "selection": selection,
        "branches": {
            "0": branch_evidence[0],
            "2": branch_evidence[2],
        },
        "rule_merge": merge_summary,
        "candidate_merge": candidate_summary,
        "strict_flip_validation": strict_summary,
        "private_state": {
            "inventory_sha256": _inventory_digest(state_tree.inventory),
            "file_count": len(state_tree.leaf_paths),
            "aggregate_only_terminal": True,
            "private_rows_serialized_to_terminal": False,
        },
        "native_action_preserved": True,
        "binary_classifier_trained": False,
        "rf_oracle_used": False,
        "gnn_retrained": False,
        "gnn_ablation_started": False,
        "dataset_redistributed": False,
        "per_example_terminal_payload": False,
        }
        validate_science_summary(science)
        return science, state_tree
    except BaseException:
        if state_tree is not None:
            state_tree.close()
        raise
    finally:
        for target in reversed(TARGET_BRANCHES):
            tree = branch_trees.get(target)
            if tree is not None:
                tree.close()
            branch = branch_handles.get(target)
            if branch is not None:
                branch.close()


def validate_science_summary(science: Mapping[str, Any]) -> None:
    top_keys = {
        "schema_version",
        "stage",
        "dataset",
        "method",
        "status",
        "num_classes",
        "source_label",
        "target_branches",
        "oracle_backend",
        "classifier_family",
        "oracle_checkpoint_hash",
        "temperature_hex",
        "config",
        "train_boundary",
        "selection",
        "branches",
        "rule_merge",
        "candidate_merge",
        "strict_flip_validation",
        "private_state",
        "native_action_preserved",
        "binary_classifier_trained",
        "rf_oracle_used",
        "gnn_retrained",
        "gnn_ablation_started",
        "dataset_redistributed",
        "per_example_terminal_payload",
    }
    if (
        type(science) is not dict
        or set(science) != top_keys
        or science.get("schema_version") != SCIENCE_SCHEMA
        or science.get("stage") != STAGE
        or science.get("dataset") != DATASET
        or science.get("method") != METHOD
        or science.get("status") != "SCIENCE_PASS_PENDING_TERMINAL_COMMIT"
        or science.get("num_classes") != NUM_CLASSES
        or science.get("source_label") != SOURCE_LABEL
        or science.get("target_branches") != [0, 2]
        or science.get("oracle_backend") != "gnn"
        or science.get("classifier_family") != "gine"
        or not _is_sha256(science.get("oracle_checkpoint_hash"))
        or science.get("native_action_preserved") is not True
        or science.get("binary_classifier_trained") is not False
        or science.get("rf_oracle_used") is not False
        or science.get("gnn_retrained") is not False
        or science.get("gnn_ablation_started") is not False
        or science.get("dataset_redistributed") is not False
        or science.get("per_example_terminal_payload") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 science summary top-level contract changed")
    try:
        temperature = float.fromhex(science["temperature_hex"])
    except (TypeError, ValueError) as exc:
        raise TasteGlobalGCESmokeError("T8 science temperature identity changed") from exc
    if (
        type(science["temperature_hex"]) is not str
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or temperature.hex() != science["temperature_hex"]
        or science.get("config") != TasteGlobalGCESmokeConfig().to_dict()
    ):
        raise TasteGlobalGCESmokeError("T8 science config/temperature changed")
    train = science.get("train_boundary")
    if (
        type(train) is not dict
        or set(train)
        != {
            "train_loaded",
            "train_row_count",
            "train_label_counts",
            "external_validation_loaded",
            "calibration_loaded",
            "test_loaded",
            "data_reprepared",
            "data_redistributed",
        }
        or train.get("train_loaded") is not True
        or type(train.get("train_row_count")) is not int
        or train["train_row_count"] <= 0
        or type(train.get("train_label_counts")) is not dict
        or set(train["train_label_counts"]) != {"0", "1", "2"}
        or any(
            type(value) is not int or value <= 0
            for value in train["train_label_counts"].values()
        )
        or sum(train["train_label_counts"].values()) != train["train_row_count"]
        or train.get("external_validation_loaded") is not False
        or train.get("calibration_loaded") is not False
        or train.get("test_loaded") is not False
        or train.get("data_reprepared") is not False
        or train.get("data_redistributed") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 split/data boundary changed")
    selection = science.get("selection")
    if (
        type(selection) is not dict
        or set(selection)
        != {
            "selection",
            "source_scan_limit",
            "source_scanned",
            "source_selected",
            "all_true_source",
            "all_predicted_source",
            "selected_cohort_sha256",
        }
        or selection.get("selection")
        != "seeded_sha256_order_then_frozen_gine_predicted_sweet"
        or selection.get("source_scan_limit") != 64
        or type(selection.get("source_scanned")) is not int
        or not 16 <= selection["source_scanned"] <= 64
        or selection.get("source_selected") != 16
        or selection.get("all_true_source") is not True
        or selection.get("all_predicted_source") is not True
        or not _is_sha256(selection.get("selected_cohort_sha256"))
    ):
        raise TasteGlobalGCESmokeError("T8 bounded source selection changed")
    branches = science.get("branches")
    if type(branches) is not dict or set(branches) != {"0", "2"}:
        raise TasteGlobalGCESmokeError("T8 branch evidence is incomplete")
    for key, target in (("0", 0), ("2", 2)):
        branch = branches[key]
        branch_keys = {
            "schema_version",
            "official_globalgce_commit",
            "official_api_signature_sha256",
            "python_module_provenance_sha256",
            "isolated_python",
            "no_user_site",
            "target_label",
            "source_label",
            "num_classes",
            "planned_checkpoint_stop_observed",
            "planned_checkpoint_next_epoch",
            "planned_checkpoint_sha256",
            "resume_checkpoint_adopted",
            "resume_checkpoint_sha256",
            "resume_identity_sha256",
            "model_state_restored",
            "optimizer_state_restored",
            "scheduler_state_restored",
            "rng_state_restored",
            "terminal_next_epoch",
            "terminal_training_checkpoint_sha256",
            "terminal_training_core_sha256",
            "terminal_model_checkpoint_sha256",
            "terminal_rule_checkpoint_sha256",
            "native_rule_catalog_sha256",
            "valid_native_rule_count",
            "raw_generated_count",
            "train_only",
            "external_validation_loaded",
            "calibration_loaded",
            "test_loaded",
            "rf_oracle_used",
            "checkpoint_seal_schema_version",
            "checkpoint_seal_pass",
            "checkpoint_writer_unwound",
            "checkpoint_durable_flush",
            "checkpoint_sealed_directory",
            "checkpoint_sealed_inventory_sha256",
            "checkpoint_resume_copy_inventory_sha256",
            "checkpoint_no_follow_identity_verified",
            "checkpoint_independent_reopen_verified",
            "checkpoint_callback_ctime_settled",
            "checkpoint_callback_ctime_ns",
            "checkpoint_sealed_ctime_ns",
            "heartbeat_callback_ctime_ns",
            "heartbeat_sealed_ctime_ns",
        }
        if (
            type(branch) is not dict
            or set(branch) != branch_keys
            or branch.get("schema_version") != BRANCH_SCHEMA
            or branch.get("official_globalgce_commit")
            != OFFICIAL_GLOBALGCE_COMMIT
            or not _is_sha256(branch.get("official_api_signature_sha256"))
            or not _is_sha256(branch.get("python_module_provenance_sha256"))
            or branch.get("isolated_python") is not True
            or branch.get("no_user_site") is not True
            or branch.get("target_label") != target
            or branch.get("source_label") != SOURCE_LABEL
            or branch.get("num_classes") != NUM_CLASSES
            or branch.get("planned_checkpoint_stop_observed") is not True
            or branch.get("planned_checkpoint_next_epoch") != 1
            or branch.get("resume_checkpoint_adopted") is not True
            or branch.get("planned_checkpoint_sha256")
            != branch.get("resume_checkpoint_sha256")
            or not _is_sha256(branch.get("planned_checkpoint_sha256"))
            or not _is_sha256(branch.get("resume_identity_sha256"))
            or branch.get("model_state_restored") is not True
            or branch.get("optimizer_state_restored") is not True
            or branch.get("scheduler_state_restored") is not True
            or branch.get("rng_state_restored") is not True
            or branch.get("terminal_next_epoch") != 6
            or any(
                not _is_sha256(branch.get(field))
                for field in (
                    "terminal_training_checkpoint_sha256",
                    "terminal_training_core_sha256",
                    "terminal_model_checkpoint_sha256",
                    "terminal_rule_checkpoint_sha256",
                    "native_rule_catalog_sha256",
                )
            )
            or type(branch.get("valid_native_rule_count")) is not int
            or branch["valid_native_rule_count"] < 20
            or type(branch.get("raw_generated_count")) is not int
            or branch["raw_generated_count"] <= 0
            or branch.get("train_only") is not True
            or branch.get("external_validation_loaded") is not False
            or branch.get("calibration_loaded") is not False
            or branch.get("rf_oracle_used") is not False
            or branch.get("test_loaded") is not False
            or branch.get("checkpoint_seal_schema_version")
            != CHECKPOINT_SEAL_SCHEMA
            or branch.get("checkpoint_seal_pass") is not True
            or branch.get("checkpoint_writer_unwound") is not True
            or branch.get("checkpoint_durable_flush") is not True
            or branch.get("checkpoint_sealed_directory")
            != _SEALED_CHECKPOINT_DIRECTORY
            or not _is_sha256(
                branch.get("checkpoint_sealed_inventory_sha256")
            )
            or branch.get("checkpoint_sealed_inventory_sha256")
            != branch.get("checkpoint_resume_copy_inventory_sha256")
            or branch.get("checkpoint_no_follow_identity_verified") is not True
            or branch.get("checkpoint_independent_reopen_verified") is not True
            or type(branch.get("checkpoint_callback_ctime_settled")) is not bool
            or any(
                type(branch.get(field)) is not int
                or branch[field] < 0
                for field in (
                    "checkpoint_callback_ctime_ns",
                    "checkpoint_sealed_ctime_ns",
                    "heartbeat_callback_ctime_ns",
                    "heartbeat_sealed_ctime_ns",
                )
            )
            or branch["checkpoint_sealed_ctime_ns"]
            < branch["checkpoint_callback_ctime_ns"]
            or branch["heartbeat_sealed_ctime_ns"]
            < branch["heartbeat_callback_ctime_ns"]
        ):
            raise TasteGlobalGCESmokeError(f"T8 target-{target} resume proof changed")
    merge = science.get("rule_merge")
    if (
        type(merge) is not dict
        or set(merge)
        != {
            "merge_stage",
            "dedup_identity",
            "target_0_rule_count",
            "target_2_rule_count",
            "premerge_rule_count",
            "cross_branch_duplicate_count",
            "merged_unique_rule_count",
            "hash_collision_or_action_mismatch",
            "canonical_dedup_complete",
            "merged_rule_set_sha256",
        }
        or merge.get("merge_stage")
        != "after_two_train_only_branches_before_calibration"
        or merge.get("dedup_identity")
        != "sha256(canonical_lhs,rhs,official_attachment_map,action_kind)"
        or any(
            type(merge.get(field)) is not int or merge[field] < minimum
            for field, minimum in (
                ("target_0_rule_count", 20),
                ("target_2_rule_count", 20),
                ("premerge_rule_count", 40),
                ("cross_branch_duplicate_count", 0),
                ("merged_unique_rule_count", 1),
            )
        )
        or merge["premerge_rule_count"]
        != merge["target_0_rule_count"] + merge["target_2_rule_count"]
        or merge["merged_unique_rule_count"]
        != merge["premerge_rule_count"] - merge["cross_branch_duplicate_count"]
        or merge.get("hash_collision_or_action_mismatch") is not False
        or merge.get("canonical_dedup_complete") is not True
        or not _is_sha256(merge.get("merged_rule_set_sha256"))
    ):
        raise TasteGlobalGCESmokeError("T8 canonical native-rule merge changed")
    candidate = science.get("candidate_merge")
    if (
        type(candidate) is not dict
        or set(candidate)
        != {
            "raw_generated_count",
            "invalid_or_noncanonical_count",
            "target_0_valid_count",
            "target_2_valid_count",
            "canonical_unique_parent_candidate_count",
            "canonical_candidate_dedup_complete",
            "candidate_hash_collision_or_corruption",
        }
        or any(
            type(candidate.get(field)) is not int or candidate[field] < minimum
            for field, minimum in (
                ("raw_generated_count", 2),
                ("invalid_or_noncanonical_count", 0),
                ("target_0_valid_count", 1),
                ("target_2_valid_count", 1),
                ("canonical_unique_parent_candidate_count", 1),
            )
        )
        or candidate["target_0_valid_count"] + candidate["target_2_valid_count"]
        > candidate["raw_generated_count"]
        or candidate["canonical_unique_parent_candidate_count"]
        > candidate["target_0_valid_count"] + candidate["target_2_valid_count"]
        or candidate.get("canonical_candidate_dedup_complete") is not True
        or candidate.get("candidate_hash_collision_or_corruption") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 canonical candidate merge changed")
    strict = science.get("strict_flip_validation")
    strict_keys = {
        "scored_parent_candidate_pairs",
        "all_parents_predicted_source",
        "strict_flip_definition",
        "strict_flip_count",
        "strict_flip_rate",
        "strict_flip_by_training_branch",
        "destination_distribution",
        "all_destinations_in_0_or_2",
        "class_order",
        "all_class_logits_validated",
        "all_class_probabilities_validated",
        "mean_logits_before",
        "mean_logits_after",
        "mean_probabilities_before",
        "mean_probabilities_after",
        "source_probability_drop",
        "source_vs_best_alternative_margin_drop",
        "binary_projection_used",
        "rf_oracle_used",
        "individual_outputs_serialized",
    }
    if (
        type(strict) is not dict
        or set(strict) != strict_keys
        or type(strict.get("scored_parent_candidate_pairs")) is not int
        or strict["scored_parent_candidate_pairs"] <= 0
        or strict.get("strict_flip_definition")
        != "pred_before == 1 and pred_after != 1"
        or type(strict.get("strict_flip_count")) is not int
        or strict["strict_flip_count"] <= 0
        or strict["strict_flip_count"] > strict["scored_parent_candidate_pairs"]
        or type(strict.get("strict_flip_rate")) not in (int, float)
        or isinstance(strict.get("strict_flip_rate"), bool)
        or not math.isfinite(float(strict["strict_flip_rate"]))
        or not math.isclose(
            float(strict["strict_flip_rate"]),
            strict["strict_flip_count"] / strict["scored_parent_candidate_pairs"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or type(strict.get("strict_flip_by_training_branch")) is not dict
        or set(strict["strict_flip_by_training_branch"]) != {"0", "2"}
        or any(
            type(value) is not int or value < 1
            for value in strict["strict_flip_by_training_branch"].values()
        )
        or type(strict.get("destination_distribution")) is not dict
        or set(strict["destination_distribution"]) != {"0", "2"}
        or any(
            type(value) is not int or value < 0
            for value in strict["destination_distribution"].values()
        )
        or sum(strict["destination_distribution"].values())
        != strict["strict_flip_count"]
        or strict.get("class_order") != ["Bitter", "Sweet", "Tasteless"]
        or strict.get("all_parents_predicted_source") is not True
        or strict.get("all_destinations_in_0_or_2") is not True
        or strict.get("all_class_logits_validated") is not True
        or strict.get("all_class_probabilities_validated") is not True
        or strict.get("binary_projection_used") is not False
        or strict.get("rf_oracle_used") is not False
        or strict.get("individual_outputs_serialized") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 original-order strict-flip audit changed")
    for field in (
        "mean_logits_before",
        "mean_logits_after",
        "mean_probabilities_before",
        "mean_probabilities_after",
    ):
        values = strict.get(field)
        if (
            type(values) is not list
            or len(values) != NUM_CLASSES
            or any(
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise TasteGlobalGCESmokeError("T8 aggregate class vector changed")
    for field in (
        "source_probability_drop",
        "source_vs_best_alternative_margin_drop",
    ):
        metric = strict.get(field)
        if (
            type(metric) is not dict
            or set(metric) != {"mean", "minimum", "maximum"}
            or any(
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in metric.values()
            )
            or not metric["minimum"] <= metric["mean"] <= metric["maximum"]
        ):
            raise TasteGlobalGCESmokeError("T8 aggregate metric summary changed")
    private = science.get("private_state")
    if (
        type(private) is not dict
        or set(private)
        != {
            "inventory_sha256",
            "file_count",
            "aggregate_only_terminal",
            "private_rows_serialized_to_terminal",
        }
        or not _is_sha256(private.get("inventory_sha256"))
        or type(private.get("file_count")) is not int
        or private["file_count"] <= 0
        or private.get("aggregate_only_terminal") is not True
        or private.get("private_rows_serialized_to_terminal") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 private-state aggregate identity changed")


_INPUT_AUTHORITY_KEYS = frozenset(
    {
        "execution",
        "managed_execution",
        "predecessors",
        "frozen_gine",
        "train_split",
        "official_globalgce",
        "policy",
    }
)


def _validate_terminal_input_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _INPUT_AUTHORITY_KEYS:
        raise TasteGlobalGCESmokeError("T8 terminal input-authority schema changed")
    execution = value["execution"]
    if (
        type(execution) is not dict
        or set(execution)
        != {
            "commit",
            "tree",
            "release_config_sha256",
            "python_entrypoint_sha256",
            "autodl_wrapper_sha256",
            "slurm_wrapper_sha256",
        }
        or not _is_git_oid(execution.get("commit"))
        or not _is_git_oid(execution.get("tree"))
        or any(
            not _is_sha256(execution.get(field))
            for field in (
                "release_config_sha256",
                "python_entrypoint_sha256",
                "autodl_wrapper_sha256",
                "slurm_wrapper_sha256",
            )
        )
    ):
        raise TasteGlobalGCESmokeError("T8 execution identity is incomplete")
    managed = value["managed_execution"]
    if (
        type(managed) is not dict
        or set(managed)
        != {
            "external_authority_schema",
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
            "same_child_revalidated_at_terminal",
        }
        or managed.get("external_authority_schema")
        != MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA
        or managed.get("protocol") != MANAGED_V2_PROTOCOL
        or managed.get("protocol_source_commit") != MANAGED_V2_SOURCE_COMMIT
        or type(managed.get("task_id")) is not str
        or managed.get("task_id") != MANAGED_TASK_ID
        or type(managed.get("run_id")) is not str
        or _SAFE_MANAGED_ID.fullmatch(managed["run_id"]) is None
        or type(managed.get("stage")) is not str
        or managed.get("stage") != STAGE
        or any(
            not _is_sha256(managed.get(field))
            for field in (
                "authority_record_sha256",
                "active_generation_sha256",
                "child_identity_sha256",
                "process_lineage_sha256",
                "expected_closure_sha256",
            )
        )
        or type(managed.get("gpu_index")) is not int
        or managed.get("gpu_index") != PHYSICAL_GPU_INDEX
        or type(managed.get("gpu_uuid")) is not str
        or _GPU_UUID.fullmatch(managed["gpu_uuid"]) is None
        or managed.get("gpu_lock_mode") != "exclusive"
        or managed.get("auto_terminate_uncontrolled_children") is not False
        or managed.get("same_child_revalidated_at_terminal") is not True
    ):
        raise TasteGlobalGCESmokeError("T8 managed GPU2 child authority changed")
    predecessors = value["predecessors"]
    if (
        type(predecessors) is not dict
        or set(predecessors)
        != {
            "t2_gate_sha256",
            "t2_receipt_sha256",
            "t2_source_evidence_sha256",
            "t2_binding_sha256",
            "t3_gate_sha256",
            "t3_root_inventory_sha256",
            "t4_gate_sha256",
            "t4_root_inventory_sha256",
            "t3_t4_same_t2_binding",
            "t3_t4_same_checkpoint",
        }
        or any(
            not _is_sha256(predecessors.get(field))
            for field in (
                "t2_gate_sha256",
                "t2_receipt_sha256",
                "t2_source_evidence_sha256",
                "t2_binding_sha256",
                "t3_gate_sha256",
                "t3_root_inventory_sha256",
                "t4_gate_sha256",
                "t4_root_inventory_sha256",
            )
        )
        or predecessors.get("t3_t4_same_t2_binding") is not True
        or predecessors.get("t3_t4_same_checkpoint") is not True
    ):
        raise TasteGlobalGCESmokeError("T8 T2/T3/T4 predecessor binding changed")
    frozen = value["frozen_gine"]
    if (
        type(frozen) is not dict
        or set(frozen)
        != {
            "checkpoint_id",
            "checkpoint_inventory_sha256",
            "checkpoint_stat_inventory_sha256",
            "checkpoint_sha256s_sha256",
            "feature_schema_sha256",
            "temperature_scaling_sha256",
            "num_classes",
            "source_label",
            "oracle_backend",
            "classifier_family",
            "rf_oracle_used",
        }
        or any(
            not _is_sha256(frozen.get(field))
            for field in (
                "checkpoint_id",
                "checkpoint_inventory_sha256",
                "checkpoint_stat_inventory_sha256",
                "checkpoint_sha256s_sha256",
                "feature_schema_sha256",
                "temperature_scaling_sha256",
            )
        )
        or frozen.get("num_classes") != NUM_CLASSES
        or frozen.get("source_label") != SOURCE_LABEL
        or frozen.get("oracle_backend") != "gnn"
        or frozen.get("classifier_family") != "gine"
        or frozen.get("rf_oracle_used") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 frozen GINE authority changed")
    train = value["train_split"]
    if (
        type(train) is not dict
        or set(train)
        != {"sha256", "bytes", "row_count", "label_counts", "split"}
        or not _is_sha256(train.get("sha256"))
        or type(train.get("bytes")) is not int
        or train["bytes"] <= 0
        or type(train.get("row_count")) is not int
        or train["row_count"] <= 0
        or train.get("split") != "train"
        or type(train.get("label_counts")) is not dict
        or set(train["label_counts"]) != {"0", "1", "2"}
        or any(
            type(count) is not int or count <= 0
            for count in train["label_counts"].values()
        )
        or sum(train["label_counts"].values()) != train["row_count"]
    ):
        raise TasteGlobalGCESmokeError("T8 train-only input authority changed")
    official = value["official_globalgce"]
    if (
        type(official) is not dict
        or set(official)
        != {"commit", "tracked_tree_sha256", "source_inventory_sha256", "clean"}
        or not _is_git_oid(official.get("commit"))
        or not _is_sha256(official.get("tracked_tree_sha256"))
        or not _is_sha256(official.get("source_inventory_sha256"))
        or official.get("clean") is not True
    ):
        raise TasteGlobalGCESmokeError("T8 official GlobalGCE identity changed")
    policy = value["policy"]
    if (
        type(policy) is not dict
        or set(policy)
        != {
            "base_policy_sha256",
            "downstream_policy_sha256",
            "research_compute_allowed",
            "aggregate_reporting_allowed",
            "data_redistribution_allowed",
            "hpc_execution_allowed",
            "train_loaded",
            "external_validation_loaded",
            "calibration_loaded",
            "test_loaded",
        }
        or not _is_sha256(policy.get("base_policy_sha256"))
        or not _is_sha256(policy.get("downstream_policy_sha256"))
        or policy.get("research_compute_allowed") is not True
        or policy.get("aggregate_reporting_allowed") is not True
        or policy.get("data_redistribution_allowed") is not False
        or policy.get("hpc_execution_allowed") is not False
        or policy.get("train_loaded") is not True
        or policy.get("external_validation_loaded") is not False
        or policy.get("calibration_loaded") is not False
        or policy.get("test_loaded") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 downstream data policy changed")
    return json.loads(json.dumps(value))


def build_terminal_documents(
    *,
    science: Mapping[str, Any],
    input_authority: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    validate_science_summary(science)
    authority = _validate_terminal_input_authority(input_authority)
    if science.get("oracle_checkpoint_hash") != authority["frozen_gine"]["checkpoint_id"]:
        raise TasteGlobalGCESmokeError(
            "T8 science used a GINE other than its terminal input authority"
        )
    if science["train_boundary"]["train_row_count"] != authority["train_split"]["row_count"]:
        raise TasteGlobalGCESmokeError("T8 science/train authority row counts differ")
    input_hashes = {
        "schema_version": INPUT_SCHEMA,
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "authorities": authority,
        "authority_sha256": _canonical_sha256(authority),
    }
    state = {
        "schema_version": STATE_SCHEMA,
        "stage": STAGE,
        "status": "PASS",
        "science": json.loads(json.dumps(science)),
        "science_sha256": _canonical_sha256(science),
        "aggregate_only": True,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "status": "PASS",
        "marker": PASS_MARKER,
        "input_hashes_sha256": hashlib.sha256(
            _json_document_bytes(input_hashes)
        ).hexdigest(),
        "state_sha256": hashlib.sha256(_json_document_bytes(state)).hexdigest(),
        "science_sha256": state["science_sha256"],
        "config_sha256": _canonical_sha256(science["config"]),
        "terminal_payload_files": list(OUTPUT_PAYLOAD_FILES),
        "output_hashes_file": "output_hashes.json",
        "pass_file": "PASS",
        "aggregate_only": True,
        "per_example_payload_written": False,
        "data_redistributed": False,
        "rf_oracle_used": False,
        "gnn_ablation_started": False,
    }
    gate = {
        "schema_version": GATE_SCHEMA,
        "stage": STAGE,
        "dataset": DATASET,
        "method": METHOD,
        "status": "PASS",
        "marker": PASS_MARKER,
        "input_hashes_sha256": manifest["input_hashes_sha256"],
        "state_sha256": manifest["state_sha256"],
        "manifest_sha256": hashlib.sha256(
            _json_document_bytes(manifest)
        ).hexdigest(),
        "science_sha256": manifest["science_sha256"],
        "oracle_checkpoint_hash": science["oracle_checkpoint_hash"],
        "strict_flip_count": science["strict_flip_validation"]["strict_flip_count"],
        "destination_distribution": science["strict_flip_validation"][
            "destination_distribution"
        ],
        "both_target_branches_checkpoint_resumed": True,
        "canonical_rule_merge_complete": True,
        "untargeted_original_order_validation_complete": True,
        "aggregate_only": True,
        "data_redistributed": False,
        "rf_oracle_used": False,
    }
    documents = {
        "input_hashes.json": input_hashes,
        "state.json": state,
        "manifest.json": manifest,
        "gate.json": gate,
    }
    _validate_terminal_documents(documents)
    return documents


def _walk_keys(value: Any) -> Iterable[str]:
    if type(value) is dict:
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif type(value) is list:
        for child in value:
            yield from _walk_keys(child)


def _validate_terminal_documents(
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if type(documents) is not dict or set(documents) != set(OUTPUT_PAYLOAD_FILES):
        raise TasteGlobalGCESmokeError("T8 terminal document set changed")
    input_hashes = documents["input_hashes.json"]
    state = documents["state.json"]
    manifest = documents["manifest.json"]
    gate = documents["gate.json"]
    if (
        set(input_hashes)
        != {
            "schema_version",
            "stage",
            "dataset",
            "method",
            "authorities",
            "authority_sha256",
        }
        or set(state)
        != {
            "schema_version",
            "stage",
            "status",
            "science",
            "science_sha256",
            "aggregate_only",
        }
        or set(manifest)
        != {
            "schema_version",
            "stage",
            "dataset",
            "method",
            "status",
            "marker",
            "input_hashes_sha256",
            "state_sha256",
            "science_sha256",
            "config_sha256",
            "terminal_payload_files",
            "output_hashes_file",
            "pass_file",
            "aggregate_only",
            "per_example_payload_written",
            "data_redistributed",
            "rf_oracle_used",
            "gnn_ablation_started",
        }
        or set(gate)
        != {
            "schema_version",
            "stage",
            "dataset",
            "method",
            "status",
            "marker",
            "input_hashes_sha256",
            "state_sha256",
            "manifest_sha256",
            "science_sha256",
            "oracle_checkpoint_hash",
            "strict_flip_count",
            "destination_distribution",
            "both_target_branches_checkpoint_resumed",
            "canonical_rule_merge_complete",
            "untargeted_original_order_validation_complete",
            "aggregate_only",
            "data_redistributed",
            "rf_oracle_used",
        }
        or
        input_hashes.get("schema_version") != INPUT_SCHEMA
        or input_hashes.get("stage") != STAGE
        or input_hashes.get("dataset") != DATASET
        or input_hashes.get("method") != METHOD
        or input_hashes.get("authority_sha256")
        != _canonical_sha256(input_hashes.get("authorities"))
    ):
        raise TasteGlobalGCESmokeError("T8 input_hashes document changed")
    _validate_terminal_input_authority(input_hashes["authorities"])
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("stage") != STAGE
        or state.get("status") != "PASS"
        or state.get("aggregate_only") is not True
        or type(state.get("science")) is not dict
        or state.get("science_sha256") != _canonical_sha256(state["science"])
    ):
        raise TasteGlobalGCESmokeError("T8 state document changed")
    validate_science_summary(state["science"])
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("stage") != STAGE
        or manifest.get("dataset") != DATASET
        or manifest.get("method") != METHOD
        or manifest.get("status") != "PASS"
        or manifest.get("marker") != PASS_MARKER
        or manifest.get("input_hashes_sha256")
        != hashlib.sha256(_json_document_bytes(input_hashes)).hexdigest()
        or manifest.get("state_sha256")
        != hashlib.sha256(_json_document_bytes(state)).hexdigest()
        or manifest.get("science_sha256") != state["science_sha256"]
        or manifest.get("config_sha256")
        != _canonical_sha256(state["science"]["config"])
        or manifest.get("terminal_payload_files") != list(OUTPUT_PAYLOAD_FILES)
        or manifest.get("output_hashes_file") != "output_hashes.json"
        or manifest.get("pass_file") != "PASS"
        or manifest.get("aggregate_only") is not True
        or manifest.get("per_example_payload_written") is not False
        or manifest.get("data_redistributed") is not False
        or manifest.get("rf_oracle_used") is not False
        or manifest.get("gnn_ablation_started") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 manifest document changed")
    strict = state["science"]["strict_flip_validation"]
    if (
        gate.get("schema_version") != GATE_SCHEMA
        or gate.get("stage") != STAGE
        or gate.get("dataset") != DATASET
        or gate.get("method") != METHOD
        or gate.get("status") != "PASS"
        or gate.get("marker") != PASS_MARKER
        or gate.get("input_hashes_sha256") != manifest["input_hashes_sha256"]
        or gate.get("state_sha256") != manifest["state_sha256"]
        or gate.get("manifest_sha256")
        != hashlib.sha256(_json_document_bytes(manifest)).hexdigest()
        or gate.get("science_sha256") != manifest["science_sha256"]
        or gate.get("oracle_checkpoint_hash")
        != state["science"]["oracle_checkpoint_hash"]
        or gate.get("strict_flip_count") != strict["strict_flip_count"]
        or gate.get("destination_distribution") != strict["destination_distribution"]
        or gate.get("both_target_branches_checkpoint_resumed") is not True
        or gate.get("canonical_rule_merge_complete") is not True
        or gate.get("untargeted_original_order_validation_complete") is not True
        or gate.get("aggregate_only") is not True
        or gate.get("data_redistributed") is not False
        or gate.get("rf_oracle_used") is not False
    ):
        raise TasteGlobalGCESmokeError("T8 gate document changed")
    forbidden_key_fragments = (
        "smiles",
        "molecule",
        "parent_id",
        "candidate_id",
        "per_example_prediction",
        "raw_generated_candidate",
    )
    for key in _walk_keys(documents):
        normalized = key.lower()
        if any(fragment in normalized for fragment in forbidden_key_fragments):
            raise TasteGlobalGCESmokeError(
                f"T8 aggregate-only terminal contains forbidden key: {key}"
            )
    encoded = _canonical_bytes(documents).lower()
    for token in (
        b"randomforestclassifier",
        b"rf_model.pkl",
        b"license_pass",
        b"/proc/self/fd/",
    ):
        if token in encoded:
            raise TasteGlobalGCESmokeError(
                "T8 terminal contains forbidden backend/license/path bytes"
            )
    return {
        "gate_sha256": hashlib.sha256(_json_document_bytes(gate)).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            _json_document_bytes(manifest)
        ).hexdigest(),
        "state_sha256": hashlib.sha256(_json_document_bytes(state)).hexdigest(),
        "input_hashes_sha256": hashlib.sha256(
            _json_document_bytes(input_hashes)
        ).hexdigest(),
        "oracle_checkpoint_hash": gate["oracle_checkpoint_hash"],
        "strict_flip_count": gate["strict_flip_count"],
        "destination_distribution": dict(gate["destination_distribution"]),
    }


def _public_terminal_evidence(
    documents: Mapping[str, Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _validate_terminal_documents(documents)
    input_hashes = documents["input_hashes.json"]
    authorities = input_hashes["authorities"]
    managed = authorities["managed_execution"]
    return {
        **evidence,
        "stage": STAGE,
        "status": "PASS",
        "marker": PASS_MARKER,
        "task_id": managed["task_id"],
        "run_id": managed["run_id"],
        "gpu_index": managed["gpu_index"],
        "gpu_uuid": managed["gpu_uuid"],
        "managed_active_generation_sha256": managed[
            "active_generation_sha256"
        ],
        "managed_child_identity_sha256": managed["child_identity_sha256"],
        "input_authority_sha256": input_hashes["authority_sha256"],
        "predecessor_binding_sha256": _canonical_sha256(
            authorities["predecessors"]
        ),
        "frozen_gine_checkpoint_id": authorities["frozen_gine"][
            "checkpoint_id"
        ],
        "train_split_sha256": authorities["train_split"]["sha256"],
        "official_source_inventory_sha256": authorities[
            "official_globalgce"
        ]["source_inventory_sha256"],
        "downstream_policy_sha256": authorities["policy"][
            "downstream_policy_sha256"
        ],
        "root_inventory_sha256": _inventory_digest(inventory),
    }


def _held_external_terminal_authority(
    authority: TasteGlobalGCETerminalAuthority,
) -> dict[str, Any]:
    if isinstance(authority, Mapping):
        raise TasteGlobalGCESmokeError(
            "T8 public consumption requires a held authority, not raw terminal claims"
        )
    revalidate = getattr(authority, "revalidate_t8_terminal_authority", None)
    if not callable(revalidate):
        raise TasteGlobalGCESmokeError(
            "T8 public consumption lacks independent held terminal authority"
        )
    observed = revalidate()
    return _validate_terminal_input_authority(observed)


def publish_terminal_output(
    *,
    output: FreshOutputDirectory,
    documents: Mapping[str, Mapping[str, Any]],
    retained_input_closure: Callable[[], None],
) -> PreparedTerminalOutput:
    """Publish PASS by final rename after every other fallible operation."""

    _validate_terminal_documents(documents)
    output.revalidate()
    for name in OUTPUT_PAYLOAD_FILES:
        held = output.write_new(name, _json_document_bytes(documents[name]))
        held.close()
    prepared = prepare_terminal_output(
        output,
        marker_name="PASS",
        marker_payload=(PASS_MARKER + "\n").encode("utf-8"),
    )
    if set(prepared.tree.leaf_paths) != set(OUTPUT_PAYLOAD_FILES):
        raise TasteGlobalGCESmokeError("T8 prepared terminal layout changed")
    prepared.tree.reject_byte_sequence(
        b"/proc/self/fd/",
        suffixes=(".json",),
    )
    commit = getattr(prepared, "commit_final_rename", None)
    if not callable(commit):
        raise TasteGlobalGCESmokeError(
            "T8 release requires the reviewed marker-last final-rename primitive"
        )
    # The log marker is not success authority.  It is emitted before PASS so a
    # broken stdout cannot become a post-commit failure path.
    print(PASS_MARKER, flush=True)
    commit(retained_input_closure=retained_input_closure)
    # No validation, fsync, pathname reopen, cleanup, or logging follows the
    # atomic PASS rename.  The caller intentionally retains all descriptors to
    # process exit.
    return prepared


@dataclass(slots=True)
class HeldTasteGlobalGCESmokeOutput:
    terminal: HeldPublishedTerminalOutput
    authority: TasteGlobalGCETerminalAuthority
    expected_authority: Mapping[str, Any]
    documents: Mapping[str, Mapping[str, Any]]
    evidence: Mapping[str, Any]

    @property
    def root(self) -> Path:
        return self.terminal.path

    def revalidate(self) -> dict[str, Any]:
        external = _held_external_terminal_authority(self.authority)
        if external != dict(self.expected_authority):
            raise TasteGlobalGCESmokeError(
                "T8 independent held terminal authority changed"
            )
        inventory = self.terminal.revalidate()
        documents: dict[str, dict[str, Any]] = {}
        for name in OUTPUT_PAYLOAD_FILES:
            try:
                value = json.loads(self.terminal.read_bytes(name).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TasteGlobalGCESmokeError(
                    f"T8 held terminal JSON changed: {name}"
                ) from exc
            if type(value) is not dict:
                raise TasteGlobalGCESmokeError(
                    f"T8 held terminal document is not an object: {name}"
                )
            documents[name] = value
        if documents["input_hashes.json"].get("authorities") != external:
            raise TasteGlobalGCESmokeError(
                "T8 terminal differs from independent held authority"
            )
        current = _public_terminal_evidence(documents, inventory)
        if documents != dict(self.documents) or current != dict(self.evidence):
            raise TasteGlobalGCESmokeError("T8 held terminal authority changed")
        return json.loads(json.dumps(current))

    def close(self) -> None:
        self.terminal.close()

    def __enter__(self) -> "HeldTasteGlobalGCESmokeOutput":
        self.revalidate()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def hold_taste_globalgce_smoke_output(
    root: str | Path,
    *,
    authority: TasteGlobalGCETerminalAuthority,
) -> HeldTasteGlobalGCESmokeOutput:
    expected_authority = _held_external_terminal_authority(authority)
    terminal = HeldPublishedTerminalOutput.open(
        root,
        marker_name="PASS",
        marker_payload=(PASS_MARKER + "\n").encode("utf-8"),
    )
    try:
        inventory = terminal.revalidate()
        if (
            type(inventory.get("files")) is not dict
            or set(inventory["files"]) != set(OUTPUT_PAYLOAD_FILES)
            or inventory.get("directories") != {}
        ):
            raise TasteGlobalGCESmokeError("T8 terminal payload inventory changed")
        documents: dict[str, dict[str, Any]] = {}
        for name in OUTPUT_PAYLOAD_FILES:
            value = json.loads(terminal.read_bytes(name).decode("utf-8"))
            if type(value) is not dict:
                raise TasteGlobalGCESmokeError(
                    f"T8 terminal document is not an object: {name}"
                )
            documents[name] = value
        if documents["input_hashes.json"].get("authorities") != expected_authority:
            raise TasteGlobalGCESmokeError(
                "T8 terminal differs from independent held authority"
            )
        evidence = _public_terminal_evidence(documents, inventory)
        held = HeldTasteGlobalGCESmokeOutput(
            terminal=terminal,
            authority=authority,
            expected_authority=json.loads(json.dumps(expected_authority)),
            documents=json.loads(json.dumps(documents)),
            evidence=evidence,
        )
        held.revalidate()
        return held
    except BaseException:
        terminal.close()
        raise


def validate_taste_globalgce_smoke_output(
    root: str | Path,
    *,
    authority: TasteGlobalGCETerminalAuthority,
) -> dict[str, Any]:
    with hold_taste_globalgce_smoke_output(root, authority=authority) as held:
        return held.revalidate()


__all__ = [
    "BRANCH_SCHEMA",
    "DATASET",
    "FrozenTasteGINEScorer",
    "GINE_PAYLOAD_FILES",
    "MANAGED_TASK_ID",
    "MANAGED_V2_EXTERNAL_AUTHORITY_SCHEMA",
    "MANAGED_V2_PROTOCOL",
    "MANAGED_V2_SOURCE_COMMIT",
    "METHOD",
    "NUM_CLASSES",
    "OUTPUT_PAYLOAD_FILES",
    "PASS_MARKER",
    "PHYSICAL_GPU_INDEX",
    "PlannedGlobalGCECheckpointStop",
    "SCIENCE_SCHEMA",
    "SEED",
    "SOURCE_LABEL",
    "STAGE",
    "TARGET_BRANCHES",
    "TasteGlobalGCESmokeConfig",
    "TasteGlobalGCESmokeError",
    "TasteGlobalGCETerminalAuthority",
    "build_terminal_documents",
    "hold_taste_globalgce_smoke_output",
    "load_taste_train_cohort",
    "merge_branch_rule_catalogs",
    "publish_terminal_output",
    "run_resumed_target_branch",
    "run_t8_science",
    "select_bounded_sweet_parents",
    "validate_candidates_with_original_gine",
    "validate_science_summary",
    "validate_taste_globalgce_smoke_output",
]
