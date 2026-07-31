"""Machine-readable scientific and engineering gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    passed: bool
    failed_hard_checks: tuple[str, ...]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed_hard_checks": list(self.failed_hard_checks),
            "checks": self.checks,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_get(payload: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def values_equal(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        return math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    return actual == expected


def evaluate_gate(
    *,
    task_id: str,
    run_id: str,
    stage_id: str,
    gate_spec: Mapping[str, Any],
    expected_artifacts: list[str],
    root: Path,
    stdout: str = "",
    slurm_exit_code: str | None = None,
) -> GateEvaluation:
    failures: list[str] = []
    checks: dict[str, Any] = {}
    tolerance = float(gate_spec.get("float_tolerance", 1e-12))
    payload: dict[str, Any] = {}
    json_path_value = gate_spec.get("json_path")
    if json_path_value:
        json_path = Path(str(json_path_value)).expanduser()
        if not json_path.is_absolute():
            json_path = root / json_path
        if not json_path.is_file():
            failures.append(f"gate_json_missing:{json_path}")
        elif json_path.stat().st_size == 0:
            failures.append(f"gate_json_empty:{json_path}")
        else:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                failures.append(f"gate_json_not_object:{json_path}")
            else:
                payload = loaded
    for key, expected in (gate_spec.get("required_fields") or {}).items():
        try:
            actual = nested_get(payload, str(key))
        except KeyError:
            failures.append(f"required_field_missing:{key}")
            continue
        matched = values_equal(actual, expected, tolerance)
        checks[f"required:{key}"] = {"actual": actual, "expected": expected}
        if not matched:
            failures.append(f"required_field_mismatch:{key}")
    for key, forbidden in (gate_spec.get("forbidden_fields") or {}).items():
        try:
            actual = nested_get(payload, str(key))
        except KeyError:
            continue
        checks[f"forbidden:{key}"] = {"actual": actual, "forbidden": forbidden}
        if values_equal(actual, forbidden, tolerance):
            failures.append(f"forbidden_field_value:{key}")
    marker = gate_spec.get("required_marker")
    if marker:
        checks["marker"] = marker in stdout
        if marker not in stdout:
            failures.append(f"marker_missing:{marker}")
    for artifact_value in expected_artifacts:
        artifact = Path(artifact_value).expanduser()
        if not artifact.is_absolute():
            artifact = root / artifact
        exists = artifact.exists()
        nonempty = exists and (
            artifact.is_dir() or artifact.stat().st_size > 0
        )
        checks[f"artifact:{artifact_value}"] = {
            "exists": exists,
            "nonempty": nonempty,
        }
        if not exists:
            failures.append(f"artifact_missing:{artifact}")
        elif not nonempty:
            failures.append(f"artifact_empty:{artifact}")
    for artifact_value, expected_hash in (gate_spec.get("sha256") or {}).items():
        artifact = Path(str(artifact_value)).expanduser()
        if not artifact.is_absolute():
            artifact = root / artifact
        if not artifact.is_file():
            failures.append(f"sha256_file_missing:{artifact}")
            continue
        actual_hash = sha256_file(artifact)
        checks[f"sha256:{artifact_value}"] = actual_hash
        if actual_hash != expected_hash:
            failures.append(f"sha256_mismatch:{artifact}")
    if slurm_exit_code is not None:
        checks["slurm_exit_code"] = slurm_exit_code
        if slurm_exit_code != "0:0":
            failures.append(f"slurm_exit_code:{slurm_exit_code}")
    if "audit_passed" in payload and payload.get("audit_passed") is not True:
            failures.append("audit_passed_not_true")
    if "run_complete" in payload and payload.get("run_complete") is not True:
            failures.append("run_complete_not_true")
    if (
        "failed_hard_checks" in payload
        and payload.get("failed_hard_checks") not in ([], ())
    ):
        failures.append("failed_hard_checks_not_empty")
    return GateEvaluation(
        passed=not failures,
        failed_hard_checks=tuple(failures),
        checks=checks,
    )


def build_gate_json(
    *,
    task_id: str,
    run_id: str,
    stage_id: str,
    evaluation: GateEvaluation,
    artifacts: Mapping[str, Any],
    provenance: Mapping[str, Any],
    next_stage: str | None,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "run_id": run_id,
        "stage_id": stage_id,
        "audit_passed": evaluation.passed,
        "run_complete": evaluation.passed,
        "failed_hard_checks": list(evaluation.failed_hard_checks),
        "checks": evaluation.checks,
        "artifacts": dict(artifacts),
        "provenance": dict(provenance),
        "next_stage": next_stage,
        "message": message,
    }
