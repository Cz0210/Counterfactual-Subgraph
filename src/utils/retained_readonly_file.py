"""Descriptor-retained authority for one immutable read-only input file."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any


class RetainedReadonlyFileError(RuntimeError):
    """Raised when a retained file or one of its named ancestors drifts."""


def _is_sha256(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _DirectoryIdentity:
        if not stat.S_ISDIR(value.st_mode):
            raise RetainedReadonlyFileError("retained path ancestor is not a directory")
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            uid=int(value.st_uid),
            gid=int(value.st_gid),
        )


@dataclass(frozen=True, slots=True)
class _FileIdentity:
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
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RetainedReadonlyFileError(
                "retained input must be one single-link regular file"
            )
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


@dataclass(slots=True)
class RetainedReadonlyFile:
    """An absolute file opened from ``/`` with every ancestor retained.

    Consumers read through the held leaf descriptor.  ``revalidate`` checks
    both the held objects and every parent-to-child named relation, so a
    lexical rename/replacement cannot silently redirect a later read.
    """

    path: Path
    directory_fds: list[int]
    directory_names: list[str]
    directory_identities: list[_DirectoryIdentity]
    file_fd: int
    file_identity: _FileIdentity
    sha256: str

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        require_nonempty: bool = True,
    ) -> RetainedReadonlyFile:
        requested = Path(path).expanduser()
        if not requested.is_absolute() or Path(os.path.abspath(requested)) != requested:
            raise RetainedReadonlyFileError(
                "retained input path must be one normalized absolute path"
            )
        if expected_sha256 is not None and not _is_sha256(expected_sha256):
            raise RetainedReadonlyFileError("expected retained input SHA-256 is invalid")
        parts = requested.parts
        if len(parts) < 2 or parts[0] != os.sep or parts[-1] in {"", ".", ".."}:
            raise RetainedReadonlyFileError("retained input path has no file leaf")
        open_directory_flags = os.O_RDONLY | os.O_DIRECTORY
        open_file_flags = os.O_RDONLY
        for flag_name in ("O_CLOEXEC", "O_NOFOLLOW"):
            flag = getattr(os, flag_name, 0)
            open_directory_flags |= flag
            open_file_flags |= flag
        directory_fds: list[int] = []
        directory_names: list[str] = []
        directory_identities: list[_DirectoryIdentity] = []
        file_fd = -1
        try:
            root_fd = os.open(os.sep, open_directory_flags)
            directory_fds.append(root_fd)
            directory_identities.append(_DirectoryIdentity.from_stat(os.fstat(root_fd)))
            for name in parts[1:-1]:
                if name in {"", ".", ".."} or os.sep in name:
                    raise RetainedReadonlyFileError(
                        "retained input has an unsafe ancestor name"
                    )
                parent_fd = directory_fds[-1]
                named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                child_fd = os.open(name, open_directory_flags, dir_fd=parent_fd)
                child_identity = _DirectoryIdentity.from_stat(os.fstat(child_fd))
                if _DirectoryIdentity.from_stat(named) != child_identity:
                    os.close(child_fd)
                    raise RetainedReadonlyFileError(
                        "retained input ancestor changed while opening"
                    )
                directory_names.append(name)
                directory_fds.append(child_fd)
                directory_identities.append(child_identity)
            leaf = parts[-1]
            if leaf in {"", ".", ".."} or os.sep in leaf:
                raise RetainedReadonlyFileError("retained input has an unsafe leaf name")
            named_file = os.stat(
                leaf, dir_fd=directory_fds[-1], follow_symlinks=False
            )
            file_fd = os.open(leaf, open_file_flags, dir_fd=directory_fds[-1])
            file_identity = _FileIdentity.from_stat(os.fstat(file_fd))
            if _FileIdentity.from_stat(named_file) != file_identity:
                raise RetainedReadonlyFileError(
                    "retained input leaf changed while opening"
                )
            if require_nonempty and file_identity.size <= 0:
                raise RetainedReadonlyFileError("retained input is empty")
            digest = cls._hash_fd(file_fd, file_identity.size)
            if expected_sha256 is not None and digest != expected_sha256:
                raise RetainedReadonlyFileError("retained input SHA-256 changed")
            result = cls(
                path=requested,
                directory_fds=directory_fds,
                directory_names=directory_names,
                directory_identities=directory_identities,
                file_fd=file_fd,
                file_identity=file_identity,
                sha256=digest,
            )
            result.revalidate()
            return result
        except Exception:
            if file_fd >= 0:
                os.close(file_fd)
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)
            raise

    @staticmethod
    def _hash_fd(file_fd: int, size: int) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            data = os.pread(file_fd, min(8 * 1024 * 1024, size - offset), offset)
            if not data:
                raise RetainedReadonlyFileError(
                    "retained input ended before its frozen size"
                )
            digest.update(data)
            offset += len(data)
        if os.pread(file_fd, 1, size):
            raise RetainedReadonlyFileError("retained input grew while hashing")
        return digest.hexdigest()

    def revalidate(self) -> dict[str, Any]:
        if self.file_fd < 0 or not self.directory_fds:
            raise RetainedReadonlyFileError("retained input authority is closed")
        for index, (directory_fd, expected) in enumerate(
            zip(self.directory_fds, self.directory_identities, strict=True)
        ):
            if _DirectoryIdentity.from_stat(os.fstat(directory_fd)) != expected:
                raise RetainedReadonlyFileError("held input ancestor identity changed")
            if index:
                named = os.stat(
                    self.directory_names[index - 1],
                    dir_fd=self.directory_fds[index - 1],
                    follow_symlinks=False,
                )
                if _DirectoryIdentity.from_stat(named) != expected:
                    raise RetainedReadonlyFileError(
                        "named input ancestor no longer identifies the held directory"
                    )
        current_file = _FileIdentity.from_stat(os.fstat(self.file_fd))
        if current_file != self.file_identity:
            raise RetainedReadonlyFileError("held input file identity changed")
        named_file = os.stat(
            self.path.name,
            dir_fd=self.directory_fds[-1],
            follow_symlinks=False,
        )
        if _FileIdentity.from_stat(named_file) != self.file_identity:
            raise RetainedReadonlyFileError(
                "named input leaf no longer identifies the held file"
            )
        if self._hash_fd(self.file_fd, self.file_identity.size) != self.sha256:
            raise RetainedReadonlyFileError("held input bytes changed")
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.file_identity.size,
            "device": self.file_identity.device,
            "inode": self.file_identity.inode,
        }

    def read_bytes(self) -> bytes:
        self.revalidate()
        chunks: list[bytes] = []
        offset = 0
        while offset < self.file_identity.size:
            data = os.pread(
                self.file_fd,
                min(8 * 1024 * 1024, self.file_identity.size - offset),
                offset,
            )
            if not data:
                raise RetainedReadonlyFileError(
                    "retained input ended during descriptor-relative read"
                )
            chunks.append(data)
            offset += len(data)
        self.revalidate()
        return b"".join(chunks)

    def close(self) -> None:
        if self.file_fd >= 0:
            os.close(self.file_fd)
            self.file_fd = -1
        for directory_fd in reversed(self.directory_fds):
            os.close(directory_fd)
        self.directory_fds.clear()

    def __enter__(self) -> RetainedReadonlyFile:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - process-exit cleanup.
        self.close()


def hold_readonly_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_nonempty: bool = True,
) -> RetainedReadonlyFile:
    return RetainedReadonlyFile.open(
        path,
        expected_sha256=expected_sha256,
        require_nonempty=require_nonempty,
    )


__all__ = [
    "RetainedReadonlyFile",
    "RetainedReadonlyFileError",
    "hold_readonly_file",
]
