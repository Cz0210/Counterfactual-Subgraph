"""Retained fresh-directory authority and terminal tree snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping


class RetainedOutputDirectoryError(RuntimeError):
    """A fresh output or one of its retained terminal leaves drifted."""


@dataclass(frozen=True, slots=True)
class _Identity:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    link_count: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def directory(cls, value: os.stat_result) -> "_Identity":
        if not stat.S_ISDIR(value.st_mode):
            raise RetainedOutputDirectoryError("retained output node is not a directory")
        # Directory size, nlink, and timestamps legitimately change as the
        # fresh tree grows.  Its name authority is the stable physical tuple
        # below; every terminal child is retained and checked separately.
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            uid=int(value.st_uid),
            gid=int(value.st_gid),
            link_count=0,
            size=0,
            mtime_ns=0,
            ctime_ns=0,
        )

    @classmethod
    def file(cls, value: os.stat_result) -> "_Identity":
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RetainedOutputDirectoryError(
                "retained output leaf is not a single-link regular file"
            )
        return cls._from_stat(value)

    @classmethod
    def _from_stat(cls, value: os.stat_result) -> "_Identity":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            uid=int(value.st_uid),
            gid=int(value.st_gid),
            link_count=int(value.st_nlink),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def evidence(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "uid": self.uid,
            "gid": self.gid,
            "link_count": self.link_count,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


def _hash_fd(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise RetainedOutputDirectoryError("retained output leaf ended early")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise RetainedOutputDirectoryError("retained output leaf grew while hashing")
    return digest.hexdigest()


def _read_fd(descriptor: int, size: int) -> bytes:
    payload = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(8 * 1024 * 1024, size - offset), offset)
        if not chunk:
            raise RetainedOutputDirectoryError("retained output leaf ended early")
        payload.extend(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise RetainedOutputDirectoryError("retained output leaf grew while reading")
    return bytes(payload)


def _fd_contains(descriptor: int, size: int, needle: bytes) -> bool:
    if not needle:
        raise ValueError("retained output search needle must be nonempty")
    offset = 0
    overlap = b""
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RetainedOutputDirectoryError("retained output leaf ended early")
        if needle in overlap + chunk:
            return True
        overlap = (overlap + chunk)[-(len(needle) - 1) :] if len(needle) > 1 else b""
        offset += len(chunk)
    if os.pread(descriptor, 1, size):
        raise RetainedOutputDirectoryError("retained output leaf grew while scanning")
    return False


@dataclass(slots=True)
class _HeldLeaf:
    parent_fd: int
    name: str
    descriptor: int
    identity: _Identity
    sha256: str

    @classmethod
    def open(cls, parent_fd: int, name: str) -> "_HeldLeaf":
        named = _Identity.file(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            held = _Identity.file(os.fstat(descriptor))
            if held != named:
                raise RetainedOutputDirectoryError("output leaf changed while opening")
            return cls(
                parent_fd=parent_fd,
                name=name,
                descriptor=descriptor,
                identity=held,
                sha256=_hash_fd(descriptor, held.size),
            )
        except Exception:
            os.close(descriptor)
            raise

    def revalidate(self) -> None:
        if _Identity.file(os.fstat(self.descriptor)) != self.identity:
            raise RetainedOutputDirectoryError("held output leaf identity changed")
        if _Identity.file(
            os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        ) != self.identity:
            raise RetainedOutputDirectoryError("named output leaf identity changed")
        if _hash_fd(self.descriptor, self.identity.size) != self.sha256:
            raise RetainedOutputDirectoryError("held output leaf bytes changed")

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor, self.descriptor = self.descriptor, -1
            os.close(descriptor)


@dataclass(slots=True)
class _HeldDirectory:
    parent_fd: int
    name: str
    descriptor: int
    identity: _Identity

    def revalidate(self) -> None:
        if _Identity.directory(os.fstat(self.descriptor)) != self.identity:
            raise RetainedOutputDirectoryError("held output directory changed")
        if _Identity.directory(
            os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        ) != self.identity:
            raise RetainedOutputDirectoryError("named output directory changed")


@dataclass(slots=True)
class RetainedOutputTree:
    root_fd: int
    root_identity: _Identity
    directories: list[_HeldDirectory]
    leaves: list[_HeldLeaf]
    leaf_paths: list[str]
    excluded: frozenset[str]
    inventory: Mapping[str, Any]

    @classmethod
    def capture(
        cls,
        root_fd: int,
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> "RetainedOutputTree":
        directories: list[_HeldDirectory] = []
        leaves: list[_HeldLeaf] = []
        leaf_paths: list[str] = []
        file_evidence: dict[str, dict[str, Any]] = {}
        directory_evidence: dict[str, dict[str, int]] = {}

        def visit(directory_fd: int, prefix: str) -> None:
            names = sorted(os.listdir(directory_fd))
            for name in names:
                relative = f"{prefix}/{name}" if prefix else name
                if relative in excluded:
                    continue
                named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(named.st_mode):
                    child_fd = os.open(
                        name,
                        os.O_RDONLY
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_fd,
                    )
                    held = _Identity.directory(os.fstat(child_fd))
                    if held != _Identity.directory(named):
                        os.close(child_fd)
                        raise RetainedOutputDirectoryError(
                            "output directory changed while capturing"
                        )
                    node = _HeldDirectory(directory_fd, name, child_fd, held)
                    directories.append(node)
                    directory_evidence[relative] = held.evidence()
                    visit(child_fd, relative)
                    continue
                leaf = _HeldLeaf.open(directory_fd, name)
                leaves.append(leaf)
                leaf_paths.append(relative)
                file_evidence[relative] = {
                    "sha256": leaf.sha256,
                    "bytes": leaf.identity.size,
                    **leaf.identity.evidence(),
                }

        try:
            root_identity = _Identity.directory(os.fstat(root_fd))
            visit(root_fd, "")
            payload = {
                "schema_version": "tastemolnet_retained_output_inventory_v1",
                "root": root_identity.evidence(),
                "directories": {
                    name: directory_evidence[name] for name in sorted(directory_evidence)
                },
                "files": {name: file_evidence[name] for name in sorted(file_evidence)},
            }
            inventory = {
                **payload,
                "inventory_sha256": hashlib.sha256(
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ).hexdigest(),
            }
            result = cls(
                root_fd,
                root_identity,
                directories,
                leaves,
                leaf_paths,
                excluded,
                inventory,
            )
            result.revalidate()
            return result
        except Exception:
            for leaf in reversed(leaves):
                leaf.close()
            for node in reversed(directories):
                os.close(node.descriptor)
            raise

    def revalidate(self) -> Mapping[str, Any]:
        if _Identity.directory(os.fstat(self.root_fd)) != self.root_identity:
            raise RetainedOutputDirectoryError("retained output root identity changed")
        observed: set[str] = set()
        directory_by_name = {
            (item.parent_fd, item.name): item for item in self.directories
        }
        leaf_by_name = {(item.parent_fd, item.name): item for item in self.leaves}

        def scan(directory_fd: int, prefix: str) -> None:
            for name in os.listdir(directory_fd):
                relative = f"{prefix}/{name}" if prefix else name
                if relative in self.excluded:
                    continue
                observed.add(relative)
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    node = directory_by_name.get((directory_fd, name))
                    if node is None:
                        raise RetainedOutputDirectoryError(
                            "output gained an unretained directory"
                        )
                    named = _Identity.directory(info)
                    held = _Identity.directory(os.fstat(node.descriptor))
                    if named != node.identity or held != node.identity:
                        raise RetainedOutputDirectoryError(
                            "retained output directory identity changed"
                        )
                    scan(node.descriptor, relative)
                    continue
                leaf = leaf_by_name.get((directory_fd, name))
                if leaf is None:
                    raise RetainedOutputDirectoryError(
                        "output gained an unretained leaf"
                    )
                named = _Identity.file(info)
                held = _Identity.file(os.fstat(leaf.descriptor))
                if named != leaf.identity or held != leaf.identity:
                    raise RetainedOutputDirectoryError(
                        "retained output leaf identity changed"
                    )
                if _hash_fd(leaf.descriptor, leaf.identity.size) != leaf.sha256:
                    raise RetainedOutputDirectoryError(
                        "retained output leaf bytes changed"
                    )

        scan(self.root_fd, "")
        expected = {
            *self.inventory["directories"].keys(),
            *self.inventory["files"].keys(),
        }
        if observed != expected:
            raise RetainedOutputDirectoryError("retained output tree inventory changed")
        return self.inventory

    def reject_byte_sequence(
        self,
        needle: bytes,
        *,
        suffixes: tuple[str, ...],
        excluded: frozenset[str] = frozenset(),
    ) -> None:
        """Reject a forbidden byte sequence using the already-held leaf FDs."""

        self.revalidate()
        for relative, leaf in zip(self.leaf_paths, self.leaves, strict=True):
            if relative in excluded or not relative.endswith(suffixes):
                continue
            if _fd_contains(leaf.descriptor, leaf.identity.size, needle):
                raise RetainedOutputDirectoryError(
                    f"retained output contains forbidden bytes: {relative}"
                )
        self.revalidate()

    def read_bytes(self, relative: str) -> bytes:
        if relative not in self.leaf_paths:
            raise RetainedOutputDirectoryError(
                f"retained output lacks required leaf: {relative}"
            )
        index = self.leaf_paths.index(relative)
        leaf = self.leaves[index]
        leaf.revalidate()
        payload = _read_fd(leaf.descriptor, leaf.identity.size)
        leaf.revalidate()
        return payload

    def durably_flush(self) -> None:
        """Fsync every retained leaf, then each directory bottom-up."""

        self.revalidate()
        for leaf in self.leaves:
            os.fsync(leaf.descriptor)
        for node in reversed(self.directories):
            os.fsync(node.descriptor)
        os.fsync(self.root_fd)
        self.revalidate()

    def close(self) -> None:
        for leaf in reversed(self.leaves):
            leaf.close()
        for node in reversed(self.directories):
            if node.descriptor >= 0:
                descriptor, node.descriptor = node.descriptor, -1
                os.close(descriptor)


@dataclass(slots=True)
class FreshOutputDirectory:
    path: Path
    ancestor_fds: list[int]
    ancestor_names: list[str]
    ancestor_identities: list[_Identity]
    descriptor: int
    identity: _Identity
    committed: bool = False

    @classmethod
    def create(cls, path: str | Path) -> "FreshOutputDirectory":
        requested = Path(path).expanduser()
        if not requested.is_absolute() or Path(os.path.abspath(requested)) != requested:
            raise RetainedOutputDirectoryError(
                "fresh output path must be normalized and absolute"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fds: list[int] = []
        names: list[str] = []
        identities: list[_Identity] = []
        output_fd = -1
        try:
            root_fd = os.open(os.sep, flags)
            fds.append(root_fd)
            identities.append(_Identity.directory(os.fstat(root_fd)))
            for name in requested.parts[1:-1]:
                named = _Identity.directory(
                    os.stat(name, dir_fd=fds[-1], follow_symlinks=False)
                )
                child_fd = os.open(name, flags, dir_fd=fds[-1])
                held = _Identity.directory(os.fstat(child_fd))
                if held != named:
                    raise RetainedOutputDirectoryError(
                        "fresh output ancestor changed while opening"
                    )
                names.append(name)
                fds.append(child_fd)
                identities.append(held)
            leaf = requested.name
            try:
                os.stat(leaf, dir_fd=fds[-1], follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(f"fresh output already exists: {requested}")
            os.mkdir(leaf, 0o700, dir_fd=fds[-1])
            output_fd = os.open(leaf, flags, dir_fd=fds[-1])
            identity = _Identity.directory(os.fstat(output_fd))
            if identity != _Identity.directory(
                os.stat(leaf, dir_fd=fds[-1], follow_symlinks=False)
            ):
                raise RetainedOutputDirectoryError(
                    "fresh output changed while opening"
                )
            os.fsync(fds[-1])
            result = cls(requested, fds, names, identities, output_fd, identity)
            result.revalidate()
            return result
        except Exception:
            if output_fd >= 0:
                os.close(output_fd)
            for descriptor in reversed(fds):
                os.close(descriptor)
            raise

    @property
    def stable_path(self) -> Path:
        if not sys.platform.startswith("linux"):
            raise RetainedOutputDirectoryError(
                "descriptor-backed output paths require Linux"
            )
        return Path(f"/proc/self/fd/{self.descriptor}")

    def revalidate(self) -> None:
        for index, (descriptor, expected) in enumerate(
            zip(self.ancestor_fds, self.ancestor_identities, strict=True)
        ):
            if _Identity.directory(os.fstat(descriptor)) != expected:
                raise RetainedOutputDirectoryError("held output ancestor changed")
            if index and _Identity.directory(
                os.stat(
                    self.ancestor_names[index - 1],
                    dir_fd=self.ancestor_fds[index - 1],
                    follow_symlinks=False,
                )
            ) != expected:
                raise RetainedOutputDirectoryError("named output ancestor changed")
        if _Identity.directory(os.fstat(self.descriptor)) != self.identity:
            raise RetainedOutputDirectoryError("held fresh output changed")
        if _Identity.directory(
            os.stat(
                self.path.name,
                dir_fd=self.ancestor_fds[-1],
                follow_symlinks=False,
            )
        ) != self.identity:
            raise RetainedOutputDirectoryError("named fresh output changed")
        if stat.S_IMODE(self.identity.mode) != 0o700:
            raise RetainedOutputDirectoryError("fresh output mode is not private")

    def write_new(self, name: str, data: bytes) -> _HeldLeaf:
        if "/" in name or name in {"", ".", ".."}:
            raise RetainedOutputDirectoryError("fresh output leaf name is invalid")
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=self.descriptor,
        )
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RetainedOutputDirectoryError("fresh output write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            held = _Identity.file(os.fstat(descriptor))
            named = _Identity.file(
                os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            )
            if named != held:
                raise RetainedOutputDirectoryError(
                    "fresh output leaf changed while writing"
                )
            digest = _hash_fd(descriptor, held.size)
            if digest != hashlib.sha256(data).hexdigest():
                raise RetainedOutputDirectoryError(
                    "fresh output leaf bytes changed while writing"
                )
            return _HeldLeaf(
                parent_fd=self.descriptor,
                name=name,
                descriptor=descriptor,
                identity=held,
                sha256=digest,
            )
        except Exception:
            os.close(descriptor)
            raise

    def close(self) -> None:
        descriptors = [self.descriptor, *reversed(self.ancestor_fds)]
        self.descriptor = -1
        self.ancestor_fds = []
        for descriptor in descriptors:
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except Exception:
                if not self.committed:
                    raise


def _hold_existing_output(path: str | Path) -> FreshOutputDirectory:
    requested = Path(path).expanduser()
    if not requested.is_absolute() or Path(os.path.abspath(requested)) != requested:
        raise RetainedOutputDirectoryError(
            "published output path must be normalized and absolute"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fds: list[int] = []
    names: list[str] = []
    identities: list[_Identity] = []
    output_fd = -1
    try:
        root_fd = os.open(os.sep, flags)
        fds.append(root_fd)
        identities.append(_Identity.directory(os.fstat(root_fd)))
        for name in requested.parts[1:-1]:
            named = _Identity.directory(
                os.stat(name, dir_fd=fds[-1], follow_symlinks=False)
            )
            child_fd = os.open(name, flags, dir_fd=fds[-1])
            held = _Identity.directory(os.fstat(child_fd))
            if held != named:
                raise RetainedOutputDirectoryError(
                    "published output ancestor changed while opening"
                )
            names.append(name)
            fds.append(child_fd)
            identities.append(held)
        leaf = requested.name
        named_output = _Identity.directory(
            os.stat(leaf, dir_fd=fds[-1], follow_symlinks=False)
        )
        output_fd = os.open(leaf, flags, dir_fd=fds[-1])
        held_output = _Identity.directory(os.fstat(output_fd))
        if held_output != named_output:
            raise RetainedOutputDirectoryError(
                "published output changed while opening"
            )
        result = FreshOutputDirectory(
            requested,
            fds,
            names,
            identities,
            output_fd,
            held_output,
            committed=True,
        )
        result.revalidate()
        return result
    except Exception:
        if output_fd >= 0:
            os.close(output_fd)
        for descriptor in reversed(fds):
            os.close(descriptor)
        raise


@dataclass(slots=True)
class HeldPublishedTerminalOutput:
    output: FreshOutputDirectory
    tree: RetainedOutputTree
    output_hashes: _HeldLeaf
    marker: _HeldLeaf
    marker_name: str
    marker_payload: bytes

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        marker_name: str,
        marker_payload: bytes,
    ) -> "HeldPublishedTerminalOutput":
        output = _hold_existing_output(path)
        tree: RetainedOutputTree | None = None
        output_hashes: _HeldLeaf | None = None
        marker: _HeldLeaf | None = None
        try:
            tree = RetainedOutputTree.capture(
                output.descriptor,
                excluded=frozenset({"output_hashes.json", marker_name}),
            )
            output_hashes = _HeldLeaf.open(output.descriptor, "output_hashes.json")
            marker = _HeldLeaf.open(output.descriptor, marker_name)
            result = cls(
                output,
                tree,
                output_hashes,
                marker,
                marker_name,
                marker_payload,
            )
            result.revalidate()
            return result
        except Exception:
            if marker is not None:
                marker.close()
            if output_hashes is not None:
                output_hashes.close()
            if tree is not None:
                tree.close()
            output.close()
            raise

    @property
    def stable_path(self) -> Path:
        return self.output.stable_path

    @property
    def path(self) -> Path:
        return self.output.path

    def read_bytes(self, relative: str) -> bytes:
        return self.tree.read_bytes(relative)

    def revalidate(self) -> Mapping[str, Any]:
        self.output.revalidate()
        inventory = self.tree.revalidate()
        self.output_hashes.revalidate()
        self.marker.revalidate()
        expected_inventory = (
            json.dumps(
                dict(inventory),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if _read_fd(
            self.output_hashes.descriptor,
            self.output_hashes.identity.size,
        ) != expected_inventory:
            raise RetainedOutputDirectoryError(
                "published output inventory document drifted"
            )
        if _read_fd(self.marker.descriptor, self.marker.identity.size) != self.marker_payload:
            raise RetainedOutputDirectoryError("published output marker drifted")
        self.output.revalidate()
        return inventory

    def close(self) -> None:
        for held in (self.marker, self.output_hashes):
            held.close()
        self.tree.close()
        self.output.close()

    def __enter__(self) -> "HeldPublishedTerminalOutput":
        self.revalidate()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()


def _link_held_noreplace(directory_fd: int, source_fd: int, target: str) -> None:
    if not sys.platform.startswith("linux"):
        raise RetainedOutputDirectoryError("terminal output commit requires Linux")
    library = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(library, "linkat", None)
    if linkat is None:
        raise RetainedOutputDirectoryError("linkat is unavailable")
    linkat.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    linkat.restype = ctypes.c_int
    # Following /proc/self/fd/<N> links the inode already retained by source_fd;
    # a concurrently substituted prepared pathname cannot change the source.
    result = int(
        linkat(
            -100,
            os.fsencode(f"/proc/self/fd/{source_fd}"),
            directory_fd,
            os.fsencode(target),
            0x400,
        )
    )
    if result:
        observed = ctypes.get_errno()
        if observed in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(target)
        raise OSError(observed, os.strerror(observed), target)


@dataclass(slots=True)
class PreparedTerminalOutput:
    output: FreshOutputDirectory
    tree: RetainedOutputTree
    output_hashes: _HeldLeaf
    marker: _HeldLeaf
    marker_name: str

    def revalidate(self) -> None:
        self.output.revalidate()
        self.tree.revalidate()
        self.output_hashes.revalidate()
        self.marker.revalidate()

    def commit(self, *, retained_input_closure: Callable[[], None]) -> None:
        self.revalidate()
        retained_input_closure()
        self.revalidate()
        _link_held_noreplace(
            self.output.descriptor,
            self.marker.descriptor,
            self.marker_name,
        )
        # Until the prepared name is removed, strict consumers see an extra
        # terminal entry and must reject the stage.  A crash/failure here is
        # therefore fail-closed even though PASS has been linked.
        os.unlink(self.marker.name, dir_fd=self.output.descriptor)
        self.output.committed = True

    def close(self) -> None:
        for held in (self.marker, self.output_hashes):
            try:
                held.close()
            except Exception:
                if not self.output.committed:
                    raise
        try:
            self.tree.close()
        except Exception:
            if not self.output.committed:
                raise
        self.output.close()


def prepare_terminal_output(
    output: FreshOutputDirectory,
    *,
    marker_name: str,
    marker_payload: bytes,
) -> PreparedTerminalOutput:
    if "/" in marker_name or marker_name in {"", ".", ".."}:
        raise RetainedOutputDirectoryError("terminal marker name is invalid")
    prepared_name = f".{marker_name}.prepared"
    excluded = frozenset({"output_hashes.json", prepared_name})
    tree: RetainedOutputTree | None = None
    output_hashes: _HeldLeaf | None = None
    marker: _HeldLeaf | None = None
    try:
        tree = RetainedOutputTree.capture(output.descriptor, excluded=excluded)
        tree.durably_flush()
        output_hashes = output.write_new(
            "output_hashes.json",
            (
                json.dumps(
                    dict(tree.inventory),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        marker = output.write_new(prepared_name, marker_payload)
        os.fsync(output.descriptor)
        prepared = PreparedTerminalOutput(
            output=output,
            tree=tree,
            output_hashes=output_hashes,
            marker=marker,
            marker_name=marker_name,
        )
        prepared.revalidate()
        return prepared
    except Exception:
        if marker is not None:
            marker.close()
        if output_hashes is not None:
            output_hashes.close()
        if tree is not None:
            tree.close()
        raise


__all__ = [
    "FreshOutputDirectory",
    "HeldPublishedTerminalOutput",
    "PreparedTerminalOutput",
    "RetainedOutputDirectoryError",
    "RetainedOutputTree",
    "prepare_terminal_output",
]
