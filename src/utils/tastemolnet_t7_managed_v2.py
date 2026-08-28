"""Narrow worker-only adapter from Taste T7 to managed execution v2.

The worker side deliberately imports only the five managed-v2 operations it
is authorized to use: create an attempt, create staging, write raw evidence,
write worker-exit evidence, and seal staging.  Publishing, verification,
terminal gates, and PASS markers are absent from this module by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping
import uuid


NEUROSED_PREDECESSOR_KIND = "TASTE_GCF_NEUROSED_PASS"
T7_RAW_EVIDENCE_SCHEMA = "tastemolnet_t7_gcf_worker_raw_evidence_v2"
T7_WORKER_RESULT_SCHEMA = "tastemolnet_t7_gcf_worker_result_v2"


class TasteT7ManagedV2Error(RuntimeError):
    """The exact managed-v2 worker contract is unavailable or drifted."""


def _absolute(path: str | Path, *, label: str) -> Path:
    value = Path(path).expanduser()
    normalized = Path(os.path.abspath(value))
    if not value.is_absolute() or value != normalized:
        raise TasteT7ManagedV2Error(f"{label} must be normalized and absolute")
    return normalized


def _sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TasteT7ManagedV2Error(f"{label} must be lowercase SHA-256")
    return value


def _uuid4(value: Any, *, label: str) -> str:
    if type(value) is not str:
        raise TasteT7ManagedV2Error(f"{label} must be a UUIDv4 string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise TasteT7ManagedV2Error(f"{label} must be a UUIDv4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise TasteT7ManagedV2Error(f"{label} must be canonical UUIDv4")
    return value


def _managed_worker_api() -> tuple[Any, Any, Any, Any, Any]:
    """Load only the exact frozen worker-authorized API surface."""

    try:
        from src.utils.managed_execution_v2 import (
            create_managed_attempt,
            create_worker_staging,
            write_worker_exit,
            write_worker_raw_evidence,
        )
        from src.utils.terminal_publisher_v2 import seal_worker_staging
    except (ImportError, AttributeError) as exc:
        raise TasteT7ManagedV2Error(
            "MANAGED_EXECUTION_V2_FROZEN_API_UNAVAILABLE"
        ) from exc
    return (
        create_managed_attempt,
        create_worker_staging,
        write_worker_raw_evidence,
        write_worker_exit,
        seal_worker_staging,
    )


@dataclass(slots=True)
class HeldTasteT7ManagedWorkerV2:
    """Held unique attempt/staging authority owned by the T7 worker."""

    attempt: Any
    staging: Any
    stage_root: Path
    expected_final_path: Path
    predecessor_path: Path
    predecessor_sha256: str
    managed_input_hashes: Mapping[str, str]
    _write_raw: Any
    _write_exit: Any
    _seal: Any
    _sealed: bool = False
    _closed: bool = False

    @property
    def attempt_id(self) -> str:
        return _uuid4(self.attempt.attempt_id, label="managed attempt_id")

    @property
    def generation_token(self) -> str:
        return _uuid4(
            self.staging.generation_token,
            label="managed staging generation_token",
        )

    @property
    def artifact_root(self) -> Path:
        return Path(self.staging.artifact_root)

    def predecessor_evidence(self) -> list[dict[str, str]]:
        return [
            {
                "kind": NEUROSED_PREDECESSOR_KIND,
                "path": str(self.predecessor_path),
                "sha256": self.predecessor_sha256,
            }
        ]

    def attempt_input_hashes(self) -> dict[str, str]:
        """Return the exact hashes retained in the managed attempt manifest."""

        manifest = self.attempt.revalidate()
        observed = manifest.get("input_hashes")
        expected = dict(self.managed_input_hashes)
        if observed != expected:
            raise TasteT7ManagedV2Error(
                "managed T7 attempt input-hash binding drifted"
            )
        return expected

    def revalidate(self) -> None:
        if self._closed:
            raise TasteT7ManagedV2Error("managed T7 worker authority is closed")
        self.attempt_input_hashes()
        self.staging.revalidate()
        attempt_id = self.attempt_id
        generation_token = self.generation_token
        attempt_path = self.stage_root / "attempts" / attempt_id
        staging_path = (
            attempt_path / "worker_staging" / self.staging.staging_id
        )
        if (
            Path(self.attempt.attempt_path) != attempt_path
            or Path(self.staging.path) != staging_path
            or self.artifact_root != staging_path / "artifacts"
            or self.staging.attempt is not self.attempt
            or not generation_token
        ):
            raise TasteT7ManagedV2Error(
                "managed T7 attempt/staging namespace drifted"
            )

    def seal_raw_evidence(self, payload: Mapping[str, Any]) -> Any:
        """Write raw/exit evidence and SEALED; never verify or publish."""

        self.revalidate()
        if self._sealed:
            raise TasteT7ManagedV2Error("managed T7 staging is already sealed")
        if (
            type(payload) is not dict
            or payload.get("schema_version") != T7_RAW_EVIDENCE_SCHEMA
            or payload.get("attempt_id") != self.attempt_id
            or payload.get("generation_token") != self.generation_token
            or payload.get("expected_final_path")
            != str(self.expected_final_path)
            or payload.get("predecessors") != self.predecessor_evidence()
            or "PASS" in payload
            or "gate" in payload
            or "verification" in payload
        ):
            raise TasteT7ManagedV2Error(
                "Taste T7 raw evidence is not exact worker-only evidence"
            )
        raw = self._write_raw(self.staging, payload)
        try:
            raw.revalidate()
        finally:
            raw.close()
        worker_exit = self._write_exit(
            self.staging,
            {
                "schema_version": T7_WORKER_RESULT_SCHEMA,
                "status": "COMPLETED_PENDING_INDEPENDENT_VERIFICATION",
                "exit_code": 0,
                "attempt_id": self.attempt_id,
                "generation_token": self.generation_token,
            },
        )
        try:
            worker_exit.revalidate()
        finally:
            worker_exit.close()
        self.revalidate()
        sealed = self._seal(self.staging)
        if (
            sealed.attempt_id != self.attempt_id
            or sealed.generation_token != self.generation_token
            or Path(sealed.staging_path) != Path(self.staging.path)
            or Path(sealed.artifact_root) != self.artifact_root
        ):
            raise TasteT7ManagedV2Error("managed-v2 SEALED binding drifted")
        self._sealed = True
        return sealed

    def record_failure(self, exc: BaseException) -> None:
        """Best-effort worker-exit evidence; it never authorizes adoption."""

        if self._closed or self._sealed:
            return
        try:
            self.revalidate()
            held = self._write_exit(
                self.staging,
                {
                    "schema_version": T7_WORKER_RESULT_SCHEMA,
                    "status": "FAILED",
                    "exit_code": 1,
                    "attempt_id": self.attempt_id,
                    "generation_token": self.generation_token,
                    "error_type": type(exc).__name__,
                },
            )
            held.close()
        except BaseException:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.staging.close()
        finally:
            self.attempt.close()

    def __enter__(self) -> "HeldTasteT7ManagedWorkerV2":
        self.revalidate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def create_t7_managed_worker(
    *,
    stage_root: str | Path,
    expected_final_path: str | Path,
    controller_id: str,
    task_id: str,
    git_commit: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    neurosed_pass_path: str | Path,
    neurosed_pass_sha256: str,
) -> HeldTasteT7ManagedWorkerV2:
    """Create the exact unique attempt and worker staging for one T7 run."""

    root = _absolute(stage_root, label="managed stage_root")
    final_path = _absolute(expected_final_path, label="expected final path")
    predecessor_path = _absolute(
        neurosed_pass_path, label="Taste NeuroSED PASS path"
    )
    predecessor_sha = _sha256(
        neurosed_pass_sha256, label="Taste NeuroSED PASS SHA-256"
    )
    normalized_inputs = {
        str(key): _sha256(value, label=f"input_hashes[{key!r}]")
        for key, value in sorted(input_hashes.items())
    }
    if normalized_inputs.get("taste_gcf_neurosed_pass") != predecessor_sha:
        raise TasteT7ManagedV2Error(
            "managed attempt does not bind the Taste NeuroSED PASS"
        )
    if "managed_execution_v2_pass" not in normalized_inputs:
        raise TasteT7ManagedV2Error(
            "managed execution v2 PASS input hash is absent"
        )
    (
        create_attempt,
        create_staging,
        write_raw,
        write_exit,
        seal_staging,
    ) = _managed_worker_api()
    attempt = create_attempt(
        stage_root=root,
        controller_id=controller_id,
        task_id=task_id,
        git_commit=git_commit,
        config_hash=_sha256(config_hash, label="config_hash"),
        input_hashes=normalized_inputs,
    )
    try:
        staging = create_staging(attempt)
    except BaseException:
        attempt.close()
        raise
    held = HeldTasteT7ManagedWorkerV2(
        attempt=attempt,
        staging=staging,
        stage_root=root,
        expected_final_path=final_path,
        predecessor_path=predecessor_path,
        predecessor_sha256=predecessor_sha,
        managed_input_hashes=normalized_inputs,
        _write_raw=write_raw,
        _write_exit=write_exit,
        _seal=seal_staging,
    )
    try:
        held.revalidate()
        return held
    except BaseException:
        held.close()
        raise


__all__ = [
    "HeldTasteT7ManagedWorkerV2",
    "NEUROSED_PREDECESSOR_KIND",
    "T7_RAW_EVIDENCE_SCHEMA",
    "T7_WORKER_RESULT_SCHEMA",
    "TasteT7ManagedV2Error",
    "create_t7_managed_worker",
]
