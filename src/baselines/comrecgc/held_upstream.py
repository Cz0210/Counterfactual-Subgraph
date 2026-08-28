"""Descriptor-held import boundary for the pinned COMRECGC sources.

The legacy :mod:`src.baselines.comrecgc.upstream` helper is retained for the
already-reviewed binary routes.  Taste T9 has a stronger boundary: every
upstream Python file that can execute is opened once, its exact SHA-256 is
checked, and Python loads that held inode through ``/proc/self/fd``.  The
checkout directory is never added to ``sys.path``.  Consequently a
rename/load/restore race cannot substitute same-named source bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import ModuleType
from typing import Any, Mapping

from src.baselines.comrecgc.contracts import UPSTREAM_COMMIT
from src.utils.retained_readonly_file import (
    RetainedReadonlyFile,
    hold_readonly_file,
)


OFFICIAL_SOURCE_FILES = (
    "comrecgc.py",
    "common_recourse.py",
    "data.py",
    "distance.py",
    "gnn.py",
    "neurosed/models.py",
    "util.py",
)
OFFICIAL_SOURCE_SHA256: Mapping[str, str] = {
    "comrecgc.py": "921b9bfc1cc0e3efff90bf24bf9c7b754ea99563a62bba6d7197ede37785f90d",
    "common_recourse.py": "c5009ef5d73059dbea2d77e983a36a8140f1c2cca3b89664fec08f1ad7b4d6c5",
    "data.py": "9674555b455a0d306e3272bb26c5b756ccd10f188ce2d3c97d7357752fd3e37f",
    "distance.py": "d81182ccb31ef0fc5aef6a95a7debc6c17e3b495596e4ee3ff1642adf29745c3",
    "gnn.py": "7f5cdd6eeb0d97566854f8078194527e09614b8e8d255ad189a0f7f777325fd8",
    "neurosed/models.py": "8025f0cdc187625fb9d469a9ec0791694f3e923ee94e3d9084cb74a066397a60",
    "util.py": "6489a02e7a0d6498a5f9e7b1a9a4ebc137e3d26541bd2a605bff9f54b1cf74ce",
}

_MODULE_FILES = (
    ("util", "util.py"),
    ("data", "data.py"),
    ("neurosed.models", "neurosed/models.py"),
    ("distance", "distance.py"),
    ("gnn", "gnn.py"),
    ("comrecgc", "comrecgc.py"),
    ("common_recourse", "common_recourse.py"),
)
_DISPLACED_MODULES = (
    "util",
    "data",
    "neurosed",
    "neurosed.models",
    "distance",
    "gnn",
    "comrecgc",
    "common_recourse",
)


class HeldCOMRECGCUpstreamError(RuntimeError):
    """The pinned official source authority is malformed or drifted."""


def _sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HeldCOMRECGCUpstreamError(f"{field} must be one lowercase SHA-256")
    return value


def _absolute_root(value: str | Path) -> Path:
    requested = Path(value).expanduser()
    normalized = Path(os.path.abspath(requested))
    if not requested.is_absolute() or requested != normalized:
        raise HeldCOMRECGCUpstreamError(
            "COMRECGC official root must be one normalized absolute path"
        )
    if requested.name in {"", ".", ".."}:
        raise HeldCOMRECGCUpstreamError("COMRECGC official root has no leaf")
    return requested


def _directory_identity(value: os.stat_result) -> dict[str, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise HeldCOMRECGCUpstreamError(
            "COMRECGC official root is not one physical directory"
        )
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
    }


def _held_root_identity(
    held: RetainedReadonlyFile, root: Path
) -> dict[str, int]:
    # directory_fds starts at '/' and then follows every absolute path part.
    root_index = len(root.parts) - 1
    if root_index < 0 or root_index >= len(held.directory_fds):
        raise HeldCOMRECGCUpstreamError(
            "COMRECGC source is not below its declared official root"
        )
    return _directory_identity(os.fstat(held.directory_fds[root_index]))


def _load_source_module(name: str, held: RetainedReadonlyFile) -> ModuleType:
    source_path = _descriptor_path(held.file_fd)
    loader = importlib.machinery.SourceFileLoader(name, source_path)
    specification = importlib.util.spec_from_loader(name, loader)
    if specification is None:
        raise HeldCOMRECGCUpstreamError(
            f"cannot construct the held upstream module {name}"
        )
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    loader.exec_module(module)
    if (
        module.__spec__ is None
        or module.__spec__.origin != source_path
        or module.__file__ != source_path
    ):
        raise HeldCOMRECGCUpstreamError(
            f"held upstream module {name} escaped its descriptor"
        )
    held.revalidate()
    return module


def _descriptor_path(descriptor: int) -> str:
    prefix = "/proc/self/fd" if sys.platform.startswith("linux") else "/dev/fd"
    return f"{prefix}/{descriptor}"


@dataclass(slots=True)
class HeldImportedCOMRECGC:
    """Exact official source files retained for the full T9 runtime."""

    root: Path
    root_identity: Mapping[str, int]
    sources: Mapping[str, RetainedReadonlyFile]
    expected_sha256: Mapping[str, str]
    modules: Mapping[str, ModuleType]
    displaced: Mapping[str, ModuleType | None]
    old_dont_write_bytecode: bool
    closed: bool = False

    def revalidate(self) -> dict[str, Any]:
        if self.closed:
            raise HeldCOMRECGCUpstreamError(
                "held COMRECGC upstream authority is closed"
            )
        observed_root: dict[str, int] | None = None
        for relative in OFFICIAL_SOURCE_FILES:
            held = self.sources[relative]
            evidence = held.revalidate()
            if evidence["sha256"] != self.expected_sha256[relative]:
                raise HeldCOMRECGCUpstreamError(
                    f"COMRECGC official source drifted: {relative}"
                )
            current_root = _held_root_identity(held, self.root)
            if observed_root is None:
                observed_root = current_root
            elif current_root != observed_root:
                raise HeldCOMRECGCUpstreamError(
                    "COMRECGC official sources do not share one physical root"
                )
        if observed_root != dict(self.root_identity):
            raise HeldCOMRECGCUpstreamError(
                "COMRECGC official physical root changed"
            )
        for name, relative in _MODULE_FILES:
            module = self.modules.get(name)
            expected_origin = _descriptor_path(self.sources[relative].file_fd)
            if (
                not isinstance(module, ModuleType)
                or sys.modules.get(name) is not module
                or module.__spec__ is None
                or module.__spec__.origin != expected_origin
                or module.__file__ != expected_origin
            ):
                raise HeldCOMRECGCUpstreamError(
                    f"COMRECGC imported module authority changed: {name}"
                )
        package = sys.modules.get("neurosed")
        if (
            package is not self.modules.get("neurosed")
            or getattr(package, "models", None)
            is not self.modules.get("neurosed.models")
        ):
            raise HeldCOMRECGCUpstreamError(
                "COMRECGC NeuroSED namespace authority changed"
            )
        return {
            "schema_version": "comrecgc_held_official_sources_v1",
            "commit": UPSTREAM_COMMIT,
            "root": str(self.root),
            "root_identity": dict(self.root_identity),
            "file_sha256": dict(self.expected_sha256),
            "module_names": [name for name, _relative in _MODULE_FILES],
            "descriptor_loaded": True,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for name in reversed(_DISPLACED_MODULES):
            current = self.modules.get(name)
            if current is None or sys.modules.get(name) is current:
                sys.modules.pop(name, None)
            previous = self.displaced.get(name)
            if previous is not None:
                sys.modules[name] = previous
        sys.dont_write_bytecode = self.old_dont_write_bytecode
        for relative in reversed(OFFICIAL_SOURCE_FILES):
            try:
                self.sources[relative].close()
            except BaseException:
                pass

    def __enter__(self) -> "HeldImportedCOMRECGC":
        self.revalidate()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def hold_imported_comrecgc(
    root: str | Path,
    *,
    expected_file_sha256: Mapping[str, str],
) -> HeldImportedCOMRECGC:
    """Hold and import only the exact reviewed official source closure."""

    official_root = _absolute_root(root)
    if type(expected_file_sha256) is not dict or set(expected_file_sha256) != set(
        OFFICIAL_SOURCE_FILES
    ):
        raise HeldCOMRECGCUpstreamError(
            "COMRECGC expected source inventory changed"
        )
    expected = {
        relative: _sha256(
            expected_file_sha256[relative],
            field=f"official_file_sha256.{relative}",
        )
        for relative in OFFICIAL_SOURCE_FILES
    }
    if expected != dict(OFFICIAL_SOURCE_SHA256):
        raise HeldCOMRECGCUpstreamError(
            "COMRECGC source inventory is not the reviewed 122f9341 closure"
        )
    sources: dict[str, RetainedReadonlyFile] = {}
    displaced = {name: sys.modules.get(name) for name in _DISPLACED_MODULES}
    modules: dict[str, ModuleType] = {}
    old_dont_write_bytecode = sys.dont_write_bytecode
    try:
        for relative in OFFICIAL_SOURCE_FILES:
            sources[relative] = hold_readonly_file(
                official_root / relative,
                expected_sha256=expected[relative],
            )
        root_identity = _held_root_identity(
            sources[OFFICIAL_SOURCE_FILES[0]], official_root
        )
        for held in sources.values():
            if _held_root_identity(held, official_root) != root_identity:
                raise HeldCOMRECGCUpstreamError(
                    "COMRECGC official files span multiple physical roots"
                )
        for name in _DISPLACED_MODULES:
            sys.modules.pop(name, None)
        sys.dont_write_bytecode = True

        modules["util"] = _load_source_module("util", sources["util.py"])
        modules["data"] = _load_source_module("data", sources["data.py"])
        package = ModuleType("neurosed")
        package.__package__ = "neurosed"
        package.__path__ = []  # type: ignore[attr-defined]
        package.__spec__ = importlib.machinery.ModuleSpec(
            "neurosed", loader=None, is_package=True
        )
        sys.modules["neurosed"] = package
        modules["neurosed"] = package
        models = _load_source_module(
            "neurosed.models", sources["neurosed/models.py"]
        )
        package.models = models  # type: ignore[attr-defined]
        modules["neurosed.models"] = models
        for name, relative in _MODULE_FILES:
            if name in modules:
                continue
            modules[name] = _load_source_module(name, sources[relative])

        held = HeldImportedCOMRECGC(
            root=official_root,
            root_identity=root_identity,
            sources=sources,
            expected_sha256=expected,
            modules=modules,
            displaced=displaced,
            old_dont_write_bytecode=old_dont_write_bytecode,
        )
        held.revalidate()
        return held
    except BaseException:
        for name in reversed(_DISPLACED_MODULES):
            sys.modules.pop(name, None)
            previous = displaced.get(name)
            if previous is not None:
                sys.modules[name] = previous
        sys.dont_write_bytecode = old_dont_write_bytecode
        for held_source in sources.values():
            try:
                held_source.close()
            except BaseException:
                pass
        raise


__all__ = [
    "HeldCOMRECGCUpstreamError",
    "HeldImportedCOMRECGC",
    "OFFICIAL_SOURCE_FILES",
    "OFFICIAL_SOURCE_SHA256",
    "hold_imported_comrecgc",
]
