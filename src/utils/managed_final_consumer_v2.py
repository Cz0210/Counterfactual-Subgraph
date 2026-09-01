"""Descriptor-retained consumer for one published managed-v2 final root.

Unlike the SEALED staging opener, this consumer admits the verifier-only
``verification.json``, ``gate.json``, and ``PASS`` trio.  It validates those
files separately while closing the worker inventory against both ``SEALED``
and the verifier's published-inventory document.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

from src.utils.managed_execution_v2 import (
    GATE_SCHEMA,
    GENERATION_TOKEN_SCHEMA,
    PASS_MARKER,
    VERIFICATION_SCHEMA,
    WORKER_EXIT_SCHEMA,
    WORKER_RAW_EVIDENCE_SCHEMA,
    WORKER_SEALED_SCHEMA,
)
from src.utils.process_identity_v2 import canonical_json_bytes
from src.utils.retained_readonly_file import hold_readonly_file
from src.utils.terminal_publisher_v2 import (
    DirectoryEvidenceV2,
    FileEvidenceV2,
)


class ManagedFinalConsumerV2Error(RuntimeError):
    """The published managed-v2 final root is incomplete or changed."""


def _json(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManagedFinalConsumerV2Error(f"{label} is not JSON") from exc
    if type(value) is not dict:
        raise ManagedFinalConsumerV2Error(f"{label} is not one JSON object")
    return value


def _inventory(
    raw: Any, *, label: str
) -> tuple[dict[str, FileEvidenceV2], dict[str, DirectoryEvidenceV2]]:
    if type(raw) is not dict or set(raw) != {"files", "directories"}:
        raise ManagedFinalConsumerV2Error(f"{label} inventory shape changed")
    try:
        files = [FileEvidenceV2.from_mapping(item) for item in raw["files"]]
        directories = [
            DirectoryEvidenceV2.from_mapping(item) for item in raw["directories"]
        ]
    except Exception as exc:
        raise ManagedFinalConsumerV2Error(f"{label} inventory is invalid") from exc
    file_map = {item.relative_path: item for item in files}
    directory_map = {item.relative_path: item for item in directories}
    if len(file_map) != len(files) or len(directory_map) != len(directories):
        raise ManagedFinalConsumerV2Error(f"{label} inventory paths repeat")
    return file_map, directory_map


def _directory_digest(relative: str, names: list[str], generation: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "relative_path": relative,
                "names": sorted(names),
                "generation_token": generation,
            }
        )
    ).hexdigest()


def _sha256_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not block:
            raise ManagedFinalConsumerV2Error("published managed file ended early")
        digest.update(block)
        offset += len(block)
    if os.pread(descriptor, 1, size):
        raise ManagedFinalConsumerV2Error("published managed file grew")
    return digest.hexdigest()


@dataclass(slots=True)
class HeldVerifiedFinalV2:
    stack: ExitStack
    root: Path
    root_fd: int
    root_identity: tuple[int, int]
    files: Mapping[str, Any]
    published_files: Mapping[str, FileEvidenceV2]
    published_directories: Mapping[str, DirectoryEvidenceV2]
    mount_session_file_devices: Mapping[str, int]
    mount_session_directory_devices: Mapping[str, int]
    attempt_id: str
    generation_token: str
    seal_sha256: str
    source_inventory_sha256: str
    published_inventory_sha256: str
    verification_payload: Mapping[str, Any]
    gate_payload: Mapping[str, Any]

    @property
    def sealed(self) -> Any:
        return SimpleNamespace(
            attempt_id=self.attempt_id,
            generation_token=self.generation_token,
            seal_sha256=self.seal_sha256,
            inventory_sha256=self.source_inventory_sha256,
        )

    @property
    def inventory(self) -> Any:
        payload = self.verification_payload["published_inventory"]
        return SimpleNamespace(
            payload=lambda: payload,
            sha256=self.published_inventory_sha256,
        )

    def file(self, relative_path: str) -> Any:
        try:
            return self.files[relative_path]
        except KeyError as exc:
            raise ManagedFinalConsumerV2Error(
                f"published managed file is absent: {relative_path}"
            ) from exc

    @property
    def mount_session_device_drift(self) -> tuple[dict[str, int | str], ...]:
        """Describe publication-device changes observed in this mount session.

        ``st_dev`` identifies the mounted filesystem in one running kernel.  A
        persistent managed final can therefore retain the same inode, bytes,
        size, and timestamp while receiving a different device number after a
        host reboot/remount.  The sealed value remains useful provenance, but
        the first descriptor-safe reopen establishes the device identity for
        this consumer session.
        """

        observations: list[dict[str, int | str]] = []
        for relative, evidence in self.published_files.items():
            observed = self.mount_session_file_devices[relative]
            if observed != evidence.st_dev:
                observations.append(
                    {
                        "kind": "file",
                        "relative_path": relative,
                        "sealed_st_dev": evidence.st_dev,
                        "mount_session_st_dev": observed,
                    }
                )
        for relative, evidence in self.published_directories.items():
            observed = self.mount_session_directory_devices[relative]
            if observed != evidence.st_dev:
                observations.append(
                    {
                        "kind": "directory",
                        "relative_path": relative,
                        "sealed_st_dev": evidence.st_dev,
                        "mount_session_st_dev": observed,
                    }
                )
        return tuple(
            sorted(
                observations,
                key=lambda item: (str(item["relative_path"]), str(item["kind"])),
            )
        )

    @property
    def mount_session_remount_detected(self) -> bool:
        return bool(self.mount_session_device_drift)

    def _scan_exact_inventory(self) -> None:
        expected_files = set(self.published_files) | {
            "SEALED.json",
            "verification.json",
            "gate.json",
            PASS_MARKER,
        }
        expected_directories = set(self.published_directories)
        observed_files: set[str] = set()
        observed_directories: set[str] = set()
        for current, directory_names, file_names in os.walk(
            self.root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            for name in directory_names:
                child = current_path / name
                info = os.lstat(child)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ManagedFinalConsumerV2Error(
                        "published managed root contains an unsafe directory"
                    )
                observed_directories.add(child.relative_to(self.root).as_posix())
            for name in file_names:
                child = current_path / name
                info = os.lstat(child)
                if (
                    stat.S_ISLNK(info.st_mode)
                    or not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                ):
                    raise ManagedFinalConsumerV2Error(
                        "published managed root contains an unsafe file"
                    )
                observed_files.add(child.relative_to(self.root).as_posix())
        if observed_files != expected_files or observed_directories != expected_directories:
            raise ManagedFinalConsumerV2Error(
                "published managed final exact inventory changed"
            )
        for relative, evidence in self.published_files.items():
            path = self.root / relative
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                info = os.fstat(descriptor)
                named = os.stat(path, follow_symlinks=False)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)
                    or int(info.st_dev)
                    != self.mount_session_file_devices[relative]
                    or int(info.st_ino) != evidence.st_ino
                    or int(info.st_size) != evidence.size
                    or int(info.st_mtime_ns) != evidence.mtime_ns
                    or _sha256_fd(descriptor, evidence.size) != evidence.sha256
                ):
                    raise ManagedFinalConsumerV2Error(
                        f"published managed file changed: {relative}"
                    )
            finally:
                os.close(descriptor)
        for relative, evidence in self.published_directories.items():
            path = self.root / relative
            info = os.stat(path, follow_symlinks=False)
            names = os.listdir(path)
            if (
                int(info.st_dev)
                != self.mount_session_directory_devices[relative]
                or int(info.st_ino) != evidence.st_ino
                or int(info.st_size) != evidence.size
                or int(info.st_mtime_ns) != evidence.mtime_ns
                or _directory_digest(relative, names, self.generation_token)
                != evidence.sha256
            ):
                raise ManagedFinalConsumerV2Error(
                    f"published managed directory changed: {relative}"
                )

    def revalidate(self) -> Mapping[str, Any]:
        root = os.fstat(self.root_fd)
        named = os.stat(self.root, follow_symlinks=False)
        if (
            (int(root.st_dev), int(root.st_ino)) != self.root_identity
            or (root.st_dev, root.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ManagedFinalConsumerV2Error("published managed root suffered ABA")
        for item in self.files.values():
            item.revalidate()
        self._scan_exact_inventory()
        return {
            "attempt_id": self.attempt_id,
            "generation_token": self.generation_token,
            "seal_sha256": self.seal_sha256,
            "source_inventory_sha256": self.source_inventory_sha256,
            "published_inventory_sha256": self.published_inventory_sha256,
            "verification_sha256": self.files["verification.json"].sha256,
            "gate_sha256": self.files["gate.json"].sha256,
            "pass_sha256": self.files[PASS_MARKER].sha256,
        }

    def close(self) -> None:
        self.stack.close()
        if self.root_fd >= 0:
            os.close(self.root_fd)
            self.root_fd = -1

    def __enter__(self) -> "HeldVerifiedFinalV2":
        self.revalidate()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def hold_verified_managed_final(
    final_root: str | Path,
    *,
    expected_attempt_id: str | None = None,
    expected_generation_token: str | None = None,
    required_relative_paths: tuple[str, ...] = (),
) -> HeldVerifiedFinalV2:
    """Hold a published final root, including its verifier-only terminal trio."""

    root = Path(final_root)
    if not root.is_absolute() or root.resolve(strict=True) != root:
        raise ManagedFinalConsumerV2Error(
            "published managed root must be one physical absolute directory"
        )
    root_fd = os.open(
        root,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    root_info = os.fstat(root_fd)
    stack = ExitStack()
    try:
        terminal_names = (
            "SEALED.json",
            "verification.json",
            "gate.json",
            PASS_MARKER,
        )
        terminal = {
            name: stack.enter_context(hold_readonly_file(root / name))
            for name in terminal_names
        }
        seal = _json(terminal["SEALED.json"].read_bytes(), label="SEALED")
        verification = _json(
            terminal["verification.json"].read_bytes(), label="verification"
        )
        gate = _json(terminal["gate.json"].read_bytes(), label="gate")
        attempt_id = str(gate.get("attempt_id"))
        generation = str(gate.get("generation_token"))
        if (
            seal.get("schema_version") != WORKER_SEALED_SCHEMA
            or seal.get("status") != "SEALED"
            or verification.get("schema_version") != VERIFICATION_SCHEMA
            or verification.get("status") != "PASS"
            or verification.get("independent_verifier") is not True
            or gate.get("schema_version") != GATE_SCHEMA
            or gate.get("status") != "PASS"
            or gate.get("independent_verifier") is not True
            or gate.get("science_adopted") is not True
            or gate.get("downstream_released") is not True
            or terminal[PASS_MARKER].read_bytes()
            != b"[MANAGED_EXECUTION_V2_PASS]\n"
            or terminal["verification.json"].sha256
            != gate.get("verification_sha256")
            or terminal["SEALED.json"].sha256 != gate.get("sealed_sha256")
            or verification.get("attempt_id") != attempt_id
            or verification.get("generation_token") != generation
            or seal.get("attempt_id") != attempt_id
            or seal.get("generation_token") != generation
            or expected_attempt_id is not None
            and attempt_id != expected_attempt_id
            or expected_generation_token is not None
            and generation != expected_generation_token
        ):
            raise ManagedFinalConsumerV2Error(
                "managed gate/verification/PASS/SEALED binding changed"
            )
        published_raw = verification.get("published_inventory")
        published_files, published_directories = _inventory(
            published_raw, label="published"
        )
        if (
            hashlib.sha256(canonical_json_bytes(published_raw)).hexdigest()
            != verification.get("published_inventory_sha256")
            or verification.get("published_inventory_sha256")
            != gate.get("published_inventory_sha256")
        ):
            raise ManagedFinalConsumerV2Error(
                "managed published inventory hash changed"
            )
        source_raw = {
            "files": seal.get("files"),
            "directories": seal.get("directories"),
        }
        source_files, source_directories = _inventory(source_raw, label="SEALED")
        if (
            hashlib.sha256(canonical_json_bytes(source_raw)).hexdigest()
            != seal.get("inventory_sha256")
            or seal.get("inventory_sha256")
            != verification.get("source_inventory_sha256")
            or {name: item.sha256 for name, item in source_files.items()}
            != {name: item.sha256 for name, item in published_files.items()}
            or {name: item.sha256 for name, item in source_directories.items()}
            != {name: item.sha256 for name, item in published_directories.items()}
        ):
            raise ManagedFinalConsumerV2Error("SEALED/published inventory changed")
        retained_worker_paths = {
            ".generation_token.json",
            "raw_evidence.json",
            "worker_exit.json",
            *required_relative_paths,
        }
        if not retained_worker_paths <= set(published_files):
            raise ManagedFinalConsumerV2Error(
                "required managed final file is outside the published inventory"
            )
        held_files: dict[str, Any] = dict(terminal)
        mount_session_file_devices: dict[str, int] = {}
        for relative, evidence in published_files.items():
            descriptor = os.open(
                root / relative,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                info = os.fstat(descriptor)
                digest = _sha256_fd(descriptor, evidence.size)
            finally:
                os.close(descriptor)
            mount_session_file_devices[relative] = int(info.st_dev)
            if (
                int(info.st_ino) != evidence.st_ino
                or int(info.st_size) != evidence.size
                or int(info.st_mtime_ns) != evidence.mtime_ns
                or digest != evidence.sha256
            ):
                raise ManagedFinalConsumerV2Error(
                    f"published file identity changed: {relative}"
                )
            if relative in retained_worker_paths:
                held_files[relative] = stack.enter_context(
                    hold_readonly_file(
                        root / relative, expected_sha256=evidence.sha256
                    )
                )
        mount_session_directory_devices: dict[str, int] = {}
        for relative, evidence in published_directories.items():
            info = os.stat(root / relative, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ManagedFinalConsumerV2Error(
                    f"published managed directory changed: {relative}"
                )
            mount_session_directory_devices[relative] = int(info.st_dev)
        generation_payload = _json(
            held_files[".generation_token.json"].read_bytes(),
            label="generation token",
        )
        raw_payload = _json(
            held_files["raw_evidence.json"].read_bytes(), label="worker raw evidence"
        )
        exit_payload = _json(
            held_files["worker_exit.json"].read_bytes(), label="worker exit evidence"
        )
        if (
            generation_payload.get("schema_version") != GENERATION_TOKEN_SCHEMA
            or generation_payload.get("attempt_id") != attempt_id
            or generation_payload.get("generation_token") != generation
            or raw_payload.get("schema_version") != WORKER_RAW_EVIDENCE_SCHEMA
            or raw_payload.get("attempt_id") != attempt_id
            or raw_payload.get("generation_token") != generation
            or exit_payload.get("schema_version") != WORKER_EXIT_SCHEMA
            or exit_payload.get("attempt_id") != attempt_id
            or exit_payload.get("generation_token") != generation
            or seal.get("worker_raw_evidence_sha256")
            != held_files["raw_evidence.json"].sha256
            or seal.get("worker_exit_sha256")
            != held_files["worker_exit.json"].sha256
        ):
            raise ManagedFinalConsumerV2Error(
                "managed worker/generation evidence binding changed"
            )
        result = HeldVerifiedFinalV2(
            stack=stack,
            root=root,
            root_fd=root_fd,
            root_identity=(int(root_info.st_dev), int(root_info.st_ino)),
            files=held_files,
            published_files=published_files,
            published_directories=published_directories,
            mount_session_file_devices=MappingProxyType(
                mount_session_file_devices
            ),
            mount_session_directory_devices=MappingProxyType(
                mount_session_directory_devices
            ),
            attempt_id=attempt_id,
            generation_token=generation,
            seal_sha256=terminal["SEALED.json"].sha256,
            source_inventory_sha256=str(seal["inventory_sha256"]),
            published_inventory_sha256=str(
                verification["published_inventory_sha256"]
            ),
            verification_payload=verification,
            gate_payload=gate,
        )
        result.revalidate()
        return result
    except BaseException:
        stack.close()
        os.close(root_fd)
        raise


__all__ = [
    "HeldVerifiedFinalV2",
    "ManagedFinalConsumerV2Error",
    "hold_verified_managed_final",
]
