#!/usr/bin/env python3
"""Read-only verification helpers for adopting legacy experiment artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ops.gate_runner import nested_get, values_equal
from scripts.ops.ssh_ops import SSHConfig, build_ssh_argv


EVIDENCE_MARKER = "[ADOPT_EXISTING_EVIDENCE_B64]"
SUPPORTED_MODE = "legacy_manifest_sha256"
READ_ONLY_OPERATION_CAPABILITIES = {
    "remote_write": False,
    "slurm_submit": False,
    "execute_stage": False,
    "advance_downstream": False,
    "artifact_overwrite": False,
}


class AdoptExistingVerificationError(ValueError):
    """Legacy artifacts or their manifest violate the adoption contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_project_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def _allowed_external_artifact_paths(
    project_root: Path,
    output_root: Path,
    values: Sequence[str],
) -> dict[Path, str]:
    resolved_paths: dict[Path, str] = {}
    for raw_value in values:
        value = str(raw_value)
        relative = PurePosixPath(value)
        if (
            not value
            or value == "."
            or relative.is_absolute()
            or ".." in relative.parts
            or any(token in value for token in ("*", "?", "[", "]"))
            or relative.as_posix() != value
        ):
            raise AdoptExistingVerificationError(
                "External manifest artifact allowlist entries must be exact, "
                f"normalized repository-relative file paths: {value!r}"
            )
        resolved = (project_root / Path(value)).resolve(strict=False)
        if not _path_under(resolved, project_root):
            raise AdoptExistingVerificationError(
                "External manifest artifact escapes project root through its "
                f"path or a symlink: {value!r}"
            )
        if _path_under(resolved, output_root):
            raise AdoptExistingVerificationError(
                "External manifest artifact allowlist entry is already under "
                f"output_root: {value!r}"
            )
        if resolved in resolved_paths:
            raise AdoptExistingVerificationError(
                "External manifest artifact allowlist entries resolve to the "
                f"same file: {resolved_paths[resolved]!r}, {value!r}"
            )
        resolved_paths[resolved] = value
    return resolved_paths


def _resolve_artifact_path(
    project_root: Path,
    output_root: Path,
    value: str,
    allowed_external_paths: Mapping[Path, str],
) -> tuple[Path, str]:
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        project_candidate = (project_root / candidate).resolve(strict=False)
        if project_candidate in allowed_external_paths:
            resolved = project_candidate
        elif value == output_root.name or value.startswith("outputs/"):
            resolved = project_candidate
        else:
            resolved = (output_root / candidate).resolve(strict=False)
    if _path_under(resolved, output_root):
        return resolved, "output_root"
    if resolved in allowed_external_paths:
        return resolved, "allowed_external_manifest_artifact"
    raise AdoptExistingVerificationError(
        "Manifest artifact escapes output_root and is not an exact allowed "
        f"external artifact: {value!r}"
    )


def _extract_commit(payload: Mapping[str, Any]) -> str | None:
    for key in ("git_commit", "generation_commit", "source_git_commit"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for container_key in ("provenance", "metadata"):
        nested = payload.get(container_key)
        if isinstance(nested, Mapping):
            value = _extract_commit(nested)
            if value:
                return value
    return None


def _manifest_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    container: Any = payload.get("artifacts")
    if container is None:
        container = payload.get("files")
    rows: list[dict[str, Any]] = []
    if isinstance(container, Mapping):
        for path, metadata in container.items():
            if isinstance(metadata, Mapping):
                rows.append({"path": str(path), **dict(metadata)})
    elif isinstance(container, list):
        rows = [dict(item) for item in container if isinstance(item, Mapping)]
    if not rows:
        raise AdoptExistingVerificationError(
            "Legacy manifest contains no supported artifact entries."
        )
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        path = row.get("path") or row.get("file") or row.get("relative_path")
        size = row.get("size")
        if size is None:
            size = row.get("size_bytes")
        if size is None:
            size = row.get("bytes")
        digest = row.get("sha256") or row.get("file_sha256")
        if not isinstance(path, str) or not path:
            raise AdoptExistingVerificationError(
                f"Manifest artifact {index} has no path."
            )
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise AdoptExistingVerificationError(
                f"Manifest artifact {path!r} has invalid size."
            )
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise AdoptExistingVerificationError(
                f"Manifest artifact {path!r} has invalid SHA256."
            )
        normalized.append(
            {
                "path": path,
                "expected_size": size,
                "expected_sha256": digest.lower(),
            }
        )
    return normalized


def _resolved_alias(
    project_root: Path, aliases: Mapping[str, str], current_path: str
) -> tuple[Path, bool]:
    selected = aliases.get(current_path, current_path)
    return _resolve_project_path(project_root, selected), selected != current_path


def verify_existing_artifacts(
    project_root: str | Path,
    config: Mapping[str, Any],
    *,
    current_remote_commit: str,
) -> dict[str, Any]:
    """Verify legacy artifacts without modifying any file."""

    root = Path(project_root).expanduser().resolve()
    configured_capabilities = config.get("operation_capabilities")
    if configured_capabilities is None:
        capabilities = dict(READ_ONLY_OPERATION_CAPABILITIES)
    elif isinstance(configured_capabilities, Mapping):
        capabilities = dict(configured_capabilities)
    else:
        raise AdoptExistingVerificationError(
            "adopt-existing operation_capabilities must be a mapping."
        )
    if capabilities != READ_ONLY_OPERATION_CAPABILITIES:
        raise AdoptExistingVerificationError(
            "adopt-existing operation capabilities must remain read-only: "
            f"actual={capabilities!r}"
        )
    if config.get("mode") != SUPPORTED_MODE:
        raise AdoptExistingVerificationError(
            f"Unsupported adoption mode: {config.get('mode')!r}"
        )
    output_root = _resolve_project_path(root, str(config["output_root"]))
    if not _path_under(output_root, root):
        raise AdoptExistingVerificationError(
            f"Adoption output_root escapes project root: {output_root}"
        )
    allowed_external_values = [
        str(value)
        for value in (
            config.get("allowed_external_manifest_artifacts") or []
        )
    ]
    allowed_external_paths = _allowed_external_artifact_paths(
        root, output_root, allowed_external_values
    )
    completion_path = _resolve_project_path(
        root, str(config["completion_marker"])
    )
    manifest_path = _resolve_project_path(root, str(config["manifest_path"]))
    finalized_path = _resolve_project_path(
        root, str(config["finalized_marker"])
    )
    controlled_paths = {
        "completion_marker": completion_path,
        "manifest_path": manifest_path,
        "finalized_marker": finalized_path,
    }
    for field, path in controlled_paths.items():
        if not _path_under(path, output_root):
            raise AdoptExistingVerificationError(
                f"{field} escapes output_root: {path}"
            )
    aliases = {
        str(key): str(value)
        for key, value in (config.get("artifact_aliases") or {}).items()
    }
    for current, legacy in aliases.items():
        for role, value in (("source", current), ("destination", legacy)):
            resolved = _resolve_project_path(root, value)
            if not _path_under(resolved, output_root):
                raise AdoptExistingVerificationError(
                    f"Alias {role} escapes output_root: {value!r}"
                )
    failures: list[str] = []
    artifact_checks: list[dict[str, Any]] = []
    scientific_checks: list[dict[str, Any]] = []
    stage_gate_checks: list[dict[str, Any]] = []
    jsonl_checks: list[dict[str, Any]] = []

    if not output_root.is_dir():
        failures.append("output_root_missing")
    if not completion_path.is_file():
        failures.append("completion_marker_missing")
    if not manifest_path.is_file():
        failures.append("manifest_missing")
    finalized_exists = finalized_path.exists()
    if finalized_exists:
        failures.append("finalized_marker_exists")

    completion: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if completion_path.is_file():
        loaded = json.loads(completion_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            failures.append("completion_marker_not_object")
        else:
            completion = loaded
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            failures.append("manifest_not_object")
        else:
            manifest = loaded

    if completion and completion.get("phase_a_complete") is not True:
        failures.append("phase_a_complete_not_true")
    completion_manifest = completion.get("manifest")
    if completion and not isinstance(completion_manifest, str):
        failures.append("completion_manifest_missing")
    elif isinstance(completion_manifest, str):
        if "/" in completion_manifest:
            declared_manifest = _resolve_project_path(root, completion_manifest)
        else:
            declared_manifest = (output_root / completion_manifest).resolve(
                strict=False
            )
        if declared_manifest != manifest_path:
            failures.append("completion_manifest_mismatch")

    completion_commit = _extract_commit(completion)
    manifest_commit = _extract_commit(manifest)
    expected_generation_commit = str(config["expected_generation_commit"])
    if completion_commit != manifest_commit:
        failures.append("completion_manifest_commit_mismatch")
    if manifest_commit != expected_generation_commit:
        failures.append("generation_commit_mismatch")
    expected_current_commit = str(config["current_local_commit"])
    if current_remote_commit != expected_current_commit:
        failures.append("current_local_remote_commit_mismatch")

    verified_paths: set[Path] = set()
    manifested_external_paths: set[Path] = set()
    verified_external_paths: set[Path] = set()
    if manifest:
        try:
            entries = _manifest_entries(manifest)
        except AdoptExistingVerificationError as exc:
            failures.append(f"manifest_schema_error:{exc}")
            entries = []
        for entry in entries:
            try:
                path, scope = _resolve_artifact_path(
                    root,
                    output_root,
                    str(entry["path"]),
                    allowed_external_paths,
                )
            except AdoptExistingVerificationError as exc:
                failures.append(f"manifest_path_error:{exc}")
                artifact_checks.append(
                    {
                        "manifest_path": str(entry["path"]),
                        "path": None,
                        "scope": "rejected",
                        "expected_size": entry["expected_size"],
                        "actual_size": None,
                        "expected_sha256": entry["expected_sha256"],
                        "actual_sha256": None,
                        "exists": False,
                        "regular_file": False,
                        "size_matches": False,
                        "sha256_matches": False,
                    }
                )
                continue
            if scope == "allowed_external_manifest_artifact":
                manifested_external_paths.add(path)
            exists = path.exists()
            regular_file = path.is_file()
            actual_size = path.stat().st_size if regular_file else None
            actual_sha256 = _sha256_file(path) if regular_file else None
            size_matches = actual_size == entry["expected_size"]
            sha_matches = actual_sha256 == entry["expected_sha256"]
            artifact_checks.append(
                {
                    "manifest_path": str(entry["path"]),
                    "path": str(path),
                    "scope": scope,
                    "expected_size": entry["expected_size"],
                    "actual_size": actual_size,
                    "expected_sha256": entry["expected_sha256"],
                    "actual_sha256": actual_sha256,
                    "exists": exists,
                    "regular_file": regular_file,
                    "size_matches": size_matches,
                    "sha256_matches": sha_matches,
                }
            )
            if not exists:
                failures.append(f"artifact_missing:{entry['path']}")
            elif not regular_file:
                failures.append(f"artifact_not_regular_file:{entry['path']}")
            elif not size_matches:
                failures.append(f"artifact_size_mismatch:{entry['path']}")
            elif not sha_matches:
                failures.append(f"artifact_sha256_mismatch:{entry['path']}")
            else:
                verified_paths.add(path)
                if scope == "allowed_external_manifest_artifact":
                    verified_external_paths.add(path)

    for path, declared in allowed_external_paths.items():
        if path not in manifested_external_paths:
            failures.append(
                f"allowed_external_artifact_not_manifested:{declared}"
            )

    alias_checks: list[dict[str, Any]] = []
    for current, legacy in aliases.items():
        current_path = _resolve_project_path(root, current)
        legacy_path = _resolve_project_path(root, legacy)
        manifested = legacy_path in verified_paths
        alias_checks.append(
            {
                "current_path": str(current_path),
                "legacy_path": str(legacy_path),
                "current_exists": current_path.exists(),
                "legacy_exists": legacy_path.is_file(),
                "legacy_manifest_verified": manifested,
            }
        )
        if not legacy_path.is_file():
            failures.append(f"alias_destination_missing:{legacy}")
        if not manifested:
            failures.append(f"alias_destination_not_manifested:{legacy}")

    missing_current_markers: list[str] = []
    tolerance = 1e-12
    for stage in config.get("stage_gates") or []:
        current_json = stage.get("json_path")
        if not current_json:
            continue
        resolved_json, used_alias = _resolved_alias(
            root, aliases, str(current_json)
        )
        current_present = _resolve_project_path(root, str(current_json)).is_file()
        if not current_present:
            missing_current_markers.append(str(current_json))
            if not bool(config.get("allow_missing_current_markers", False)):
                failures.append(
                    f"current_stage_gate_missing:{stage['stage_id']}"
                )
        if not resolved_json.is_file():
            if bool(config.get("allow_missing_current_markers", False)):
                stage_gate_checks.append(
                    {
                        "stage_id": stage["stage_id"],
                        "current_json_path": str(current_json),
                        "resolved_json_path": str(resolved_json),
                        "current_present": current_present,
                        "used_alias": used_alias,
                        "accepted_missing_current_marker": True,
                    }
                )
                continue
            failures.append(f"stage_gate_json_missing:{stage['stage_id']}")
            continue
        payload = json.loads(resolved_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            failures.append(f"stage_gate_json_not_object:{stage['stage_id']}")
            continue
        stage_failures: list[str] = []
        for field, expected in (stage.get("required_fields") or {}).items():
            try:
                actual = nested_get(payload, str(field))
            except KeyError:
                actual = None
                matched = False
            else:
                matched = values_equal(actual, expected, tolerance)
            scientific_checks.append(
                {
                    "stage_id": stage["stage_id"],
                    "field": str(field),
                    "expected": expected,
                    "actual": actual,
                    "matched": matched,
                }
            )
            if not matched:
                stage_failures.append(str(field))
                failures.append(
                    f"scientific_field_mismatch:{stage['stage_id']}:{field}"
                )
        for field, forbidden in (stage.get("forbidden_fields") or {}).items():
            try:
                actual = nested_get(payload, str(field))
            except KeyError:
                continue
            matched = not values_equal(actual, forbidden, tolerance)
            scientific_checks.append(
                {
                    "stage_id": stage["stage_id"],
                    "field": str(field),
                    "forbidden": forbidden,
                    "actual": actual,
                    "matched": matched,
                }
            )
            if not matched:
                stage_failures.append(str(field))
                failures.append(
                    f"scientific_forbidden_field:{stage['stage_id']}:{field}"
                )
        stage_gate_checks.append(
            {
                "stage_id": stage["stage_id"],
                "current_json_path": str(current_json),
                "resolved_json_path": str(resolved_json),
                "current_present": current_present,
                "used_alias": used_alias,
                "failed_fields": stage_failures,
            }
        )

    for current, expected_rows in (
        config.get("jsonl_row_counts") or {}
    ).items():
        resolved, used_alias = _resolved_alias(root, aliases, str(current))
        exists = resolved.is_file()
        actual_rows: int | None = None
        valid_jsonl = False
        if exists:
            lines = resolved.read_text(encoding="utf-8").splitlines()
            actual_rows = len(lines)
            try:
                valid_jsonl = all(
                    bool(line.strip()) and json.loads(line) is not None
                    for line in lines
                )
            except json.JSONDecodeError:
                valid_jsonl = False
        matched = actual_rows == int(expected_rows) and valid_jsonl
        jsonl_checks.append(
            {
                "current_path": str(current),
                "resolved_path": str(resolved),
                "used_alias": used_alias,
                "expected_rows": int(expected_rows),
                "actual_rows": actual_rows,
                "valid_jsonl": valid_jsonl,
                "matched": matched,
            }
        )
        if not exists:
            failures.append(f"jsonl_missing:{current}")
        elif actual_rows != int(expected_rows):
            failures.append(f"jsonl_row_count_mismatch:{current}")
        elif not valid_jsonl:
            failures.append(f"jsonl_invalid:{current}")

    accepted_legacy = (
        bool(missing_current_markers)
        and bool(config.get("allow_missing_current_markers", False))
        and not failures
    )
    external_artifact_checks = [
        check
        for check in artifact_checks
        if check["scope"] == "allowed_external_manifest_artifact"
    ]
    evidence = {
        "schema_version": 1,
        "mode": SUPPORTED_MODE,
        "verification_passed": not failures,
        "failed_hard_checks": failures,
        "output_root": str(output_root),
        "completion_marker": str(completion_path),
        "manifest_path": str(manifest_path),
        "current_local_commit": expected_current_commit,
        "current_remote_commit": current_remote_commit,
        "legacy_generation_commit": manifest_commit,
        "completion_generation_commit": completion_commit,
        "artifact_count": len(artifact_checks),
        "artifacts": artifact_checks,
        "allowed_external_manifest_artifacts": allowed_external_values,
        "external_artifact_count": len(external_artifact_checks),
        "external_artifacts_verified": (
            len(external_artifact_checks) == len(allowed_external_paths)
            and set(allowed_external_paths) == verified_external_paths
            and all(
                check["regular_file"]
                and check["size_matches"]
                and check["sha256_matches"]
                for check in external_artifact_checks
            )
        ),
        "external_artifact_verified_paths": [
            value
            for value in allowed_external_values
            if (root / Path(value)).resolve(strict=False)
            in verified_external_paths
        ],
        "scientific_field_checks": scientific_checks,
        "stage_gate_checks": stage_gate_checks,
        "jsonl_row_checks": jsonl_checks,
        "alias_mapping": alias_checks,
        "finalized_marker": str(finalized_path),
        "finalized_marker_exists": finalized_exists,
        "missing_current_markers": missing_current_markers,
        "current_required_marker_present": not bool(missing_current_markers),
        "accepted_via_legacy_manifest_integrity": accepted_legacy,
        "remote_write_performed": False,
        "operation_capabilities": dict(READ_ONLY_OPERATION_CAPABILITIES),
        "slurm_jobs": [],
        "adopted_stages": list(config.get("adopted_stages") or []),
        "next_stage": config.get("next_stage"),
    }
    return evidence


def encode_evidence(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return base64.b64encode(encoded).decode("ascii")


def parse_evidence(stdout: str) -> dict[str, Any]:
    matches = [
        line.removeprefix(EVIDENCE_MARKER).strip()
        for line in stdout.splitlines()
        if line.startswith(EVIDENCE_MARKER)
    ]
    if len(matches) != 1:
        raise AdoptExistingVerificationError(
            f"Expected exactly one evidence marker, found {len(matches)}."
        )
    payload = json.loads(base64.b64decode(matches[0]).decode("utf-8"))
    if not isinstance(payload, dict):
        raise AdoptExistingVerificationError("Evidence payload is not an object.")
    return payload


def build_remote_script(
    ssh_config: SSHConfig, verification_config: Mapping[str, Any]
) -> str:
    config_b64 = base64.b64encode(
        json.dumps(verification_config, sort_keys=True).encode("utf-8")
    ).decode("ascii")
    root = shlex.quote(str(PurePosixPath(ssh_config.remote_root)))
    env_name = shlex.quote(ssh_config.conda_env)
    script_path = shlex.quote("scripts/ops/adopt_existing.py")
    return "\n".join(
        [
            "set -eo pipefail",
            "set +u",
            "source ~/.bashrc",
            f"conda activate {env_name}",
            f"cd {root}",
            "export PYTHONDONTWRITEBYTECODE=1",
            "REMOTE_COMMIT=$(git rev-parse HEAD)",
            f"python {script_path} verify-remote "
            f"--config-b64 {shlex.quote(config_b64)} "
            '"--remote-commit=$REMOTE_COMMIT"',
        ]
    )


def build_verification_argv(
    ssh_config: SSHConfig, verification_config: Mapping[str, Any]
) -> list[str]:
    return build_ssh_argv(
        ssh_config, build_remote_script(ssh_config, verification_config)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-remote")
    verify.add_argument("--config-b64", required=True)
    verify.add_argument("--remote-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-remote":
        config = json.loads(base64.b64decode(args.config_b64).decode("utf-8"))
        evidence = verify_existing_artifacts(
            Path.cwd(), config, current_remote_commit=args.remote_commit
        )
        print(f"{EVIDENCE_MARKER} {encode_evidence(evidence)}")
        return 0 if evidence["verification_passed"] else 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
