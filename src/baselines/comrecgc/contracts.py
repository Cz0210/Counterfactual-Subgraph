"""Shared, dependency-light contracts for COMRECGC adaptation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

UPSTREAM_URL = "https://github.com/ssggreg/COMRECGC.git"
UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
UPSTREAM_COMMIT_DATE = "2025-05-23T03:01:37+02:00"

METHOD = "COMRECGC"
CF_MODE = "strict_flip"
DISTANCE_LINE = "MolCLR-Node-Wasserstein"
ADAPTATION_MODE = "common_recourse_cluster_medoid_fullgraph"


class ContractError(ValueError):
    """Raised when frozen protocol or provenance does not match."""


@dataclass(frozen=True)
class GenerationParameters:
    theta: float
    teleport: float
    steps: int
    heads: int
    candidate_capacity: int
    sample_size: int
    seed: int

    @classmethod
    def for_mode(cls, mode: str) -> "GenerationParameters":
        if mode == "smoke":
            return cls(
                theta=0.1,
                teleport=0.1,
                steps=50,
                heads=2,
                candidate_capacity=200,
                sample_size=64,
                seed=0,
            )
        if mode == "full":
            return cls(
                theta=0.1,
                teleport=0.1,
                steps=50_000,
                heads=5,
                candidate_capacity=100_000,
                sample_size=10_000,
                seed=0,
            )
        raise ContractError(f"Unsupported mode: {mode!r}; expected smoke or full.")

    def validate(self, mode: str) -> None:
        expected = type(self).for_mode(mode)
        if self != expected:
            raise ContractError(
                f"{mode} generation parameters differ from the frozen contract: "
                f"actual={asdict(self)}, expected={asdict(expected)}"
            )


@dataclass(frozen=True)
class RecourseParameters:
    theta: float
    delta: float
    recourse_size: int
    cf_size: int
    cluster_size: int
    seed: int

    @classmethod
    def for_mode(cls, mode: str) -> "RecourseParameters":
        if mode == "smoke":
            return cls(0.1, 0.02, 5, 200, 2, 0)
        if mode == "full":
            return cls(0.1, 0.02, 100, 100_000, 3, 0)
        raise ContractError(f"Unsupported mode: {mode!r}; expected smoke or full.")

    def validate(self, mode: str) -> None:
        expected = type(self).for_mode(mode)
        if self != expected:
            raise ContractError(
                f"{mode} common-recourse parameters differ from the frozen "
                f"contract: actual={asdict(self)}, expected={asdict(expected)}"
            )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_ids_sha256(values: Iterable[str]) -> str:
    return stable_json_sha256([str(value) for value in values])


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n").encode(
            "utf-8"
        ),
    )


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_finite(value: Any, field: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ContractError(f"{field} must be finite; received {value!r}.")
    return resolved


def require_empty_output(path: str | Path, *, resume: bool = False) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "_FINALIZED.json").exists() or (root / "_RUN_COMPLETE.json").exists():
        raise FileExistsError(f"Completed/finalized output cannot be overwritten: {root}")
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"Output directory is non-empty and resume=false: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def assert_project_relative(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ContractError(f"Expected project-relative path without '..': {path!s}")
    return candidate
