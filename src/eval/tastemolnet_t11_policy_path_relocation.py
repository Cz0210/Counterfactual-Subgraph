"""Content-addressed TasteMolNet T11 policy-path relocation.

The scoped TasteMolNet policy receipt records the absolute path of the policy
file used by the original immutable execution checkout.  Publication may run
from a later immutable checkout, where the same tracked policy bytes have a
different absolute path.  This module permits exactly that deployment-only
path change for the T11 publication route.

The original policy receipt remains immutable and is still passed through the
ordinary strict validator.  A synthetic policy identity supplies its recorded
path while retaining the securely reopened current policy bytes, raw SHA-256,
canonical SHA-256, and validated payload.  Consequently every receipt field
except ``policy_path`` remains exact; no generic path tolerance is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from typing import Any, Mapping

from src.train.molecular_gnn_resume import atomic_rename_directory_noreplace
from src.utils.tastemolnet_research_policy import (
    TasteLocalDataAuthority,
    TastePolicyReceipt,
    TasteResearchPolicy,
    TasteResearchPolicyError,
    load_tastemolnet_research_policy,
    validate_tastemolnet_local_authority,
    validate_tastemolnet_policy_receipt,
)


RELOCATION_SCHEMA = "tastemolnet_policy_path_relocation_receipt_v1"
RELOCATION_FILENAME = "policy_path_relocation_receipt.json"
_HEX = frozenset("0123456789abcdef")
_RELOCATION_KEYS = {
    "schema_version",
    "status",
    "science_rerun",
    "source_policy_path",
    "current_policy_path",
    "source_policy_sha256",
    "current_policy_sha256",
    "byte_identical",
    "source_regular_nonsymlink",
    "current_regular_nonsymlink",
    "policy_id",
    "policy_version",
    "policy_schema_version",
    "policy_receipt_path",
    "policy_receipt_sha256",
    "paper_reporting_authorized",
    "dataset_redistribution_authorized",
    "license_conclusion",
    "path_difference_is_deployment_metadata_only",
}


class T11PolicyPathRelocationError(RuntimeError):
    """The T11 publication-only policy-path relocation failed closed."""


@dataclass(frozen=True, slots=True)
class T11PolicyPathRelocation:
    """Reopened relocation evidence and its underlying strict authorities."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]
    policy: TasteResearchPolicy
    authority: TasteLocalDataAuthority
    policy_receipt: TastePolicyReceipt

    def publication_evidence(self) -> dict[str, Any]:
        return {
            "schema_version": RELOCATION_SCHEMA,
            "relocation_receipt_path": str(self.path),
            "relocation_receipt_sha256": self.sha256,
            "source_policy_path": str(self.payload["source_policy_path"]),
            "current_policy_path": str(self.payload["current_policy_path"]),
            "policy_file_sha256": self.policy.file_sha256,
            "policy_canonical_sha256": self.policy.canonical_sha256,
            "policy_receipt_path": str(self.policy_receipt.path),
            "policy_receipt_sha256": self.policy_receipt.sha256,
            "only_policy_path_relocated": True,
            "science_rerun": False,
        }

    def matrix_policy_binding(self) -> dict[str, Any]:
        """Return the strict policy identity consumed by the matrix publisher."""

        return {
            "policy_receipt_path": str(self.policy_receipt.path),
            "policy_receipt_sha256": self.policy_receipt.sha256,
            "policy_id": self.policy.policy_id,
            "policy_version": self.policy.version,
            "paper_reporting_authorized": True,
            "dataset_redistribution_authorized": False,
            "license_conclusion": "NOT_GRANTED_OR_INFERRED",
            "legacy_license_pass_claimed": False,
            "policy_path_relocation": self.publication_evidence(),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise T11PolicyPathRelocationError(f"{field} must be one lowercase SHA-256")
    return value


def _read_regular_file(path_like: str | Path, *, field: str) -> tuple[Path, bytes]:
    logical = Path(path_like).expanduser()
    if not logical.is_absolute():
        raise T11PolicyPathRelocationError(f"{field} must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(logical, flags)
    except OSError as exc:
        raise T11PolicyPathRelocationError(f"cannot open {field}: {logical}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise T11PolicyPathRelocationError(
                f"{field} must be one physical regular non-symlink file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(logical, follow_symlinks=False)
        identity = lambda item: (  # noqa: E731 - immutable stat projection.
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(after) != identity(named):
            raise T11PolicyPathRelocationError(f"{field} changed while it was read")
        return logical.resolve(strict=True), b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path_like: str | Path, *, field: str) -> tuple[Path, bytes, dict[str, Any]]:
    path, data = _read_regular_file(path_like, field=field)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T11PolicyPathRelocationError(f"{field} is not valid JSON") from exc
    if type(payload) is not dict:
        raise T11PolicyPathRelocationError(f"{field} must contain one JSON object")
    return path, data, payload


def _exact_equal(observed: Any, expected: Any, *, field: str) -> None:
    """Compare JSON values without Python's bool/int equality coercion."""

    if type(observed) is not type(expected):
        raise T11PolicyPathRelocationError(f"{field} changed native JSON type")
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise T11PolicyPathRelocationError(f"{field} keys changed")
        for key in expected:
            _exact_equal(observed[key], expected[key], field=f"{field}.{key}")
        return
    if type(expected) is list:
        if len(observed) != len(expected):
            raise T11PolicyPathRelocationError(f"{field} length changed")
        for index, (actual, wanted) in enumerate(zip(observed, expected, strict=True)):
            _exact_equal(actual, wanted, field=f"{field}[{index}]")
        return
    if observed != expected:
        raise T11PolicyPathRelocationError(f"{field} changed value")


def _recorded_policy_path(policy_receipt: str | Path) -> Path:
    _, _, raw = _read_json(policy_receipt, field="source policy receipt")
    policy_evidence = raw.get("policy")
    if type(policy_evidence) is not dict:
        raise T11PolicyPathRelocationError("source receipt policy evidence is untyped")
    value = policy_evidence.get("policy_path")
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise T11PolicyPathRelocationError(
            "source receipt policy_path must be one absolute path"
        )
    return Path(value)


def _strict_source_binding(
    *,
    current_policy_path: str | Path,
    policy_receipt: str | Path,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
) -> tuple[TasteResearchPolicy, TasteLocalDataAuthority, TastePolicyReceipt, Path]:
    current_policy = load_tastemolnet_research_policy(current_policy_path)
    current_policy.require_main_route()
    authority = validate_tastemolnet_local_authority(
        current_policy,
        prepared_root=prepared_root,
        graph_cache_root=graph_cache_root,
    )
    recorded_path = _recorded_policy_path(policy_receipt)
    if str(recorded_path) == str(current_policy.path):
        raise T11PolicyPathRelocationError(
            "policy path did not relocate; use the ordinary strict receipt route"
        )

    # The ordinary validator stays strict.  Only its path identity is restored
    # to the immutable source receipt's recorded value; all content and typed
    # semantics come from the securely reopened current file.
    recorded_identity = TasteResearchPolicy(
        path=recorded_path,
        file_sha256=current_policy.file_sha256,
        canonical_sha256=current_policy.canonical_sha256,
        payload=current_policy.payload,
    )
    source_receipt = validate_tastemolnet_policy_receipt(
        policy_receipt,
        policy=recorded_identity,
        authority=authority,
        require_active=True,
        require_policy_version=2,
    )
    source_evidence = source_receipt.payload.get("policy")
    if type(source_evidence) is not dict:
        raise T11PolicyPathRelocationError("source policy evidence is untyped")
    expected_evidence = dict(current_policy.evidence())
    expected_evidence["policy_path"] = str(recorded_path)
    _exact_equal(source_evidence, expected_evidence, field="source receipt policy")
    return current_policy, authority, source_receipt, recorded_path


def _expected_payload(
    *,
    current_policy: TasteResearchPolicy,
    source_receipt: TastePolicyReceipt,
    recorded_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": RELOCATION_SCHEMA,
        "status": "PASS",
        "science_rerun": False,
        "source_policy_path": str(recorded_path),
        "current_policy_path": str(current_policy.path),
        "source_policy_sha256": current_policy.file_sha256,
        "current_policy_sha256": current_policy.file_sha256,
        "byte_identical": True,
        "source_regular_nonsymlink": True,
        "current_regular_nonsymlink": True,
        "policy_id": str(current_policy.payload["policy_id"]),
        "policy_version": current_policy.version,
        "policy_schema_version": str(current_policy.payload["schema_version"]),
        "policy_receipt_path": str(source_receipt.path),
        "policy_receipt_sha256": source_receipt.sha256,
        "paper_reporting_authorized": True,
        "dataset_redistribution_authorized": False,
        "license_conclusion": "NOT_GRANTED_OR_INFERRED",
        "path_difference_is_deployment_metadata_only": True,
    }


def validate_t11_policy_path_relocation(
    relocation_receipt: str | Path,
    *,
    current_policy_path: str | Path,
    policy_receipt: str | Path,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
) -> T11PolicyPathRelocation:
    """Validate one existing T11 relocation receipt without writing anything."""

    try:
        current, authority, source_receipt, recorded = _strict_source_binding(
            current_policy_path=current_policy_path,
            policy_receipt=policy_receipt,
            prepared_root=prepared_root,
            graph_cache_root=graph_cache_root,
        )
    except TasteResearchPolicyError as exc:
        raise T11PolicyPathRelocationError(str(exc)) from exc
    path, data, payload = _read_json(
        relocation_receipt, field="T11 policy-path relocation receipt"
    )
    if set(payload) != _RELOCATION_KEYS:
        raise T11PolicyPathRelocationError(
            "T11 policy-path relocation receipt keys changed"
        )
    expected = _expected_payload(
        current_policy=current,
        source_receipt=source_receipt,
        recorded_path=recorded,
    )
    _exact_equal(payload, expected, field="T11 policy-path relocation receipt")
    return T11PolicyPathRelocation(
        path=path,
        sha256=_sha256_bytes(data),
        payload=payload,
        policy=current,
        authority=authority,
        policy_receipt=source_receipt,
    )


def _write_fsync(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_t11_policy_path_relocation(
    *,
    current_policy_path: str | Path,
    policy_receipt: str | Path,
    prepared_root: str | Path,
    graph_cache_root: str | Path,
    output_root: str | Path,
) -> T11PolicyPathRelocation:
    """Create and independently reopen one fresh publication-only overlay."""

    destination = Path(output_root).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise T11PolicyPathRelocationError(
            "T11 policy-path relocation output must be an absolute non-symlink path"
        )
    destination = destination.resolve(strict=False)
    if destination.exists():
        raise T11PolicyPathRelocationError(
            f"T11 policy-path relocation output must be fresh: {destination}"
        )
    try:
        current, authority, source_receipt, recorded = _strict_source_binding(
            current_policy_path=current_policy_path,
            policy_receipt=policy_receipt,
            prepared_root=prepared_root,
            graph_cache_root=graph_cache_root,
        )
    except TasteResearchPolicyError as exc:
        raise T11PolicyPathRelocationError(str(exc)) from exc

    # At construction time reopen the recorded source file as well.  A later
    # publisher does not require the old checkout to survive: the immutable
    # source receipt plus both policy hashes remain sufficient authority.
    try:
        source_policy = load_tastemolnet_research_policy(recorded)
    except (OSError, TasteResearchPolicyError) as exc:
        raise T11PolicyPathRelocationError(
            "recorded source policy is unavailable for relocation construction"
        ) from exc
    if (
        source_policy.file_sha256 != current.file_sha256
        or source_policy.canonical_sha256 != current.canonical_sha256
        or source_policy.payload != current.payload
    ):
        raise T11PolicyPathRelocationError(
            "source and current policy content are not byte/semantic identical"
        )

    payload = _expected_payload(
        current_policy=current,
        source_receipt=source_receipt,
        recorded_path=recorded,
    )
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.staging-{secrets.token_hex(16)}"
    )
    try:
        staging.mkdir(mode=0o755)
        _write_fsync(staging / RELOCATION_FILENAME, encoded)
        _fsync_directory(staging)
        atomic_rename_directory_noreplace(staging, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return validate_t11_policy_path_relocation(
        destination / RELOCATION_FILENAME,
        current_policy_path=current_policy_path,
        policy_receipt=policy_receipt,
        prepared_root=prepared_root,
        graph_cache_root=graph_cache_root,
    )


__all__ = [
    "RELOCATION_FILENAME",
    "RELOCATION_SCHEMA",
    "T11PolicyPathRelocation",
    "T11PolicyPathRelocationError",
    "build_t11_policy_path_relocation",
    "validate_t11_policy_path_relocation",
]
