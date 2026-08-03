"""Strict YAML task specification loading and semantic validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency gate
    raise RuntimeError("PyYAML is required; install requirements_hpc.txt") from exc

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - dependency gate
    raise RuntimeError("jsonschema is required; install requirements_hpc.txt") from exc


TASK_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DANGEROUS_REMOTE_ROOTS = {
    "/",
    "$HOME",
    "${HOME}",
    "~",
    "/share/home/u20526",
}
PROTECTED_AUTOMATION_PATHS = (
    "scripts/ops",
    "tests/ops",
    "ops/specs",
)


class SpecValidationError(ValueError):
    """Actionable task specification error."""


@dataclass(frozen=True, slots=True)
class TaskSpec:
    path: Path
    data: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.data["task_id"])

    @property
    def local_root(self) -> Path:
        return Path(self.data["project"]["local_root"]).expanduser().resolve()

    @property
    def stage_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(stage["id"]): stage for stage in self.data["stages"]}

    def topological_stage_ids(self) -> list[str]:
        return topological_order(self.data["stages"])


def _schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "ops/schemas/task_spec.schema.json"


def _load_schema() -> dict[str, Any]:
    return json.loads(_schema_path().read_text(encoding="utf-8"))


def _stage_text(stage: dict[str, Any]) -> str:
    values: list[str] = [
        str(stage.get("id") or ""),
        str((stage.get("resources") or {}).get("tags") or ""),
        str(stage.get("script") or ""),
    ]
    values.extend(str(value) for value in (stage.get("command") or []))
    values.extend(str(value) for value in stage.get("expected_artifacts") or [])
    return " ".join(values).lower()


def _contains_test_split(text: str) -> bool:
    normalized = text.replace("\\", "/").lower()
    forbidden = (
        "test_source",
        "test_target",
        "split=test",
        "split test",
        "cohort-name test",
        "cohort_name=test",
        "/test.csv",
        "/test.json",
        "/test.jsonl",
    )
    return any(token in normalized for token in forbidden)


def _declares_full(stage: dict[str, Any]) -> bool:
    values = [
        str(stage.get("id") or ""),
        str((stage.get("resources") or {}).get("tags") or ""),
    ]
    for value in values:
        tokens = re.split(r"[^a-z0-9]+", value.lower())
        if "full" in tokens:
            return True
    return False


def _contains_calibration_input(stage: dict[str, Any]) -> bool:
    values = [
        str(value).replace("\\", "/").lower()
        for value in stage.get("command") or []
        if not str(value).startswith("--forbid-")
    ]
    values.extend(
        str(value).replace("\\", "/").lower()
        for value in stage.get("expected_artifacts") or []
    )
    values.append(str(stage.get("script") or "").replace("\\", "/").lower())
    forbidden = (
        "calibration_source",
        "calibration_target",
        "split=calibration",
        "split calibration",
        "cohort-name calibration",
        "cohort_name=calibration",
        "/calibration.csv",
        "/calibration.json",
        "/calibration.jsonl",
    )
    return any(token in value for value in values for token in forbidden)


def topological_order(stages: Iterable[dict[str, Any]]) -> list[str]:
    stage_list = list(stages)
    ids = [str(stage["id"]) for stage in stage_list]
    if len(ids) != len(set(ids)):
        raise SpecValidationError("Stage IDs must be unique.")
    dependencies = {
        str(stage["id"]): [str(item) for item in stage["dependencies"]]
        for stage in stage_list
    }
    unknown = sorted(
        {
            dependency
            for values in dependencies.values()
            for dependency in values
            if dependency not in dependencies
        }
    )
    if unknown:
        raise SpecValidationError(f"Unknown stage dependencies: {unknown}")
    incoming = {stage_id: len(values) for stage_id, values in dependencies.items()}
    children: dict[str, list[str]] = {stage_id: [] for stage_id in ids}
    for stage_id, values in dependencies.items():
        for dependency in values:
            children[dependency].append(stage_id)
    ready = [stage_id for stage_id in ids if incoming[stage_id] == 0]
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for child in children[current]:
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if len(ordered) != len(ids):
        raise SpecValidationError("Stage dependencies contain a cycle.")
    return ordered


def _validate_stage_contract(stage: dict[str, Any]) -> None:
    kind = stage["kind"]
    command = stage.get("command")
    script = stage.get("script")
    if kind in {"local_command", "remote_command"} and not command:
        raise SpecValidationError(
            f"Stage {stage['id']} requires a command argv array."
        )
    if kind == "slurm_job":
        if not script:
            raise SpecValidationError(
                f"Slurm stage {stage['id']} requires a controlled script."
            )
        path = Path(str(script))
        if path.is_absolute() or ".." in path.parts:
            raise SpecValidationError(
                f"Slurm script must be repository-relative: {script}"
            )
        if not str(path).startswith("scripts/slurm/"):
            raise SpecValidationError(
                f"Slurm script is outside scripts/slurm: {script}"
            )
    if isinstance(command, str):
        raise SpecValidationError(
            f"Stage {stage['id']} command must be an argv array, not shell text."
        )


def _validate_relative_policy_path(value: str, *, field: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise SpecValidationError(
            f"{field} must be a safe repository-relative path: {value!r}"
        )


def _validate_exact_remote_tracked_dirty_path(value: str) -> None:
    path = PurePosixPath(value)
    field = "remote_dirty_policy.allowed_tracked_paths"
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or any(token in value for token in ("*", "?", "[", "]"))
        or path.as_posix() != value
    ):
        raise SpecValidationError(
            f"{field} entries must be exact, normalized repository-relative "
            f"POSIX paths without glob syntax: {value!r}"
        )
    for protected in PROTECTED_AUTOMATION_PATHS:
        if value == protected or value.startswith(protected + "/"):
            raise SpecValidationError(
                f"{field} cannot allowlist protected automation path: "
                f"{value!r}"
            )


def _validate_adopt_path_under(
    value: str, output_root: PurePosixPath, *, field: str
) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise SpecValidationError(
            f"{field} must be a safe repository-relative path: {value!r}"
        )
    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise SpecValidationError(
            f"{field} must remain under adopt_existing.output_root: {value!r}"
        ) from exc


def _validate_external_manifest_artifact(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or any(token in value for token in ("*", "?", "[", "]"))
        or path.as_posix() != value
    ):
        raise SpecValidationError(
            "allowed_external_manifest_artifacts entries must be exact, "
            f"normalized repository-relative file paths: {value!r}"
        )


def semantic_validate(data: dict[str, Any]) -> None:
    task_id = str(data["task_id"])
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise SpecValidationError(f"Invalid task_id: {task_id!r}")
    remote_root = str(data["project"]["remote_root"]).rstrip("/") or "/"
    if remote_root in DANGEROUS_REMOTE_ROOTS:
        raise SpecValidationError(
            f"Dangerous remote_root is forbidden: {remote_root}"
        )
    if not Path(remote_root).is_absolute():
        raise SpecValidationError("remote_root must be an absolute path.")
    allowed_paths = data["git"]["allowed_paths"]
    if not allowed_paths:
        raise SpecValidationError("git.allowed_paths must not be empty.")
    for value in allowed_paths:
        path = Path(str(value))
        if path.is_absolute() or ".." in path.parts:
            raise SpecValidationError(
                f"allowed_paths must be safe repository-relative paths: {value}"
            )
    remote_dirty_policy = data["remote_dirty_policy"]
    for value in remote_dirty_policy["allowed_tracked_paths"]:
        _validate_exact_remote_tracked_dirty_path(str(value))
    submodule_paths: list[str] = []
    for submodule in remote_dirty_policy["allowed_patched_submodules"]:
        submodule_path = str(submodule["path"])
        _validate_relative_policy_path(
            submodule_path,
            field="remote_dirty_policy.allowed_patched_submodules.path",
        )
        submodule_paths.append(submodule_path)
        for value in submodule["allowed_modified_paths"]:
            _validate_relative_policy_path(
                str(value), field="allowed_modified_paths"
            )
        for value in submodule.get("allowed_untracked_paths") or []:
            _validate_relative_policy_path(
                str(value), field="allowed_untracked_paths"
            )
        for marker in submodule["required_markers"]:
            _validate_relative_policy_path(
                str(marker["file"]), field="required_markers.file"
            )
    if len(submodule_paths) != len(set(submodule_paths)):
        raise SpecValidationError(
            "remote_dirty_policy patched submodule paths must be unique."
        )
    proxy_policy = data["proxy_policy"]
    if proxy_policy["preserve_existing"] is not True:
        raise SpecValidationError("proxy_policy.preserve_existing must be true.")
    stage_ids = {str(stage["id"]) for stage in data["stages"]}
    adopt = data.get("adopt_existing")
    if adopt is not None:
        output_root = PurePosixPath(str(adopt["output_root"]))
        if (
            output_root.is_absolute()
            or ".." in output_root.parts
            or str(output_root) in {"", "."}
        ):
            raise SpecValidationError(
                "adopt_existing.output_root must be repository-relative."
            )
        marker_paths = []
        for field in (
            "completion_marker",
            "manifest_path",
            "finalized_marker",
        ):
            value = str(adopt[field])
            _validate_adopt_path_under(value, output_root, field=field)
            marker_paths.append(value)
        if len(marker_paths) != len(set(marker_paths)):
            raise SpecValidationError(
                "Adoption completion, manifest, and finalized paths must differ."
            )
        unknown_stages = sorted(set(adopt["stages"]) - stage_ids)
        if unknown_stages:
            raise SpecValidationError(
                f"adopt_existing references unknown stages: {unknown_stages}"
            )
        if "phase_b_gpu_smoke" in adopt["stages"]:
            raise SpecValidationError(
                "adopt_existing must stop before phase_b_gpu_smoke."
            )
        for current, legacy in adopt["artifact_aliases"].items():
            _validate_adopt_path_under(
                str(current), output_root, field="artifact_aliases source"
            )
            _validate_adopt_path_under(
                str(legacy), output_root, field="artifact_aliases destination"
            )
        for value in adopt["jsonl_row_counts"]:
            _validate_adopt_path_under(
                str(value), output_root, field="jsonl_row_counts"
            )
        for value in adopt["allowed_external_manifest_artifacts"]:
            _validate_external_manifest_artifact(str(value))
    for key in ("auto_until", "stop_before"):
        value = data["execution"].get(key)
        if value is not None and value not in stage_ids:
            raise SpecValidationError(
                f"execution.{key} references unknown stage {value!r}."
            )
    topological_order(data["stages"])
    permissions = data["permissions"]
    for stage in data["stages"]:
        _validate_stage_contract(stage)
        text = _stage_text(stage)
        if not permissions["allow_test"] and _contains_test_split(text):
            raise SpecValidationError(
                f"Stage {stage['id']} reads or declares a test split."
            )
        if (
            not permissions["allow_calibration"]
            and _contains_calibration_input(stage)
        ):
            raise SpecValidationError(
                f"Stage {stage['id']} reads calibration while forbidden."
            )
        if not permissions["allow_full"] and _declares_full(stage):
            raise SpecValidationError(
                f"Stage {stage['id']} declares a full run while forbidden."
            )
        if stage["kind"] == "slurm_job" and not permissions["allow_sbatch"]:
            raise SpecValidationError(
                f"Stage {stage['id']} is Slurm but allow_sbatch=false."
            )
        output_root = (stage.get("resources") or {}).get(
            "expected_output_root"
        )
        if output_root and not permissions["allow_overwrite"]:
            candidate = Path(str(output_root)).expanduser()
            if not candidate.is_absolute():
                candidate = (
                    Path(data["project"]["local_root"]).expanduser() / candidate
                )
            if candidate.exists() and (candidate / "_FINALIZED.json").exists():
                raise SpecValidationError(
                    f"Finalized output cannot be overwritten: {candidate}"
                )
    if data["permissions"].get("preserve_proxy_environment") is not True:
        raise SpecValidationError(
            "preserve_proxy_environment must remain true by default."
        )


def load_task_spec(path_like: str | Path) -> TaskSpec:
    path = Path(path_like).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Task specification does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpecValidationError("Task YAML root must be a mapping.")
    permissions = payload.get("permissions")
    if isinstance(permissions, dict):
        permissions.setdefault("preserve_proxy_environment", True)
    remote_dirty_policy = payload.setdefault(
        "remote_dirty_policy",
        {
            "allowed_tracked_paths": [],
            "allowed_patched_submodules": [],
        },
    )
    if isinstance(remote_dirty_policy, dict):
        remote_dirty_policy.setdefault("allowed_tracked_paths", [])
        remote_dirty_policy.setdefault("allowed_patched_submodules", [])
    payload.setdefault(
        "proxy_policy",
        {
            "preserve_existing": True,
            "require_any_present_for_git_network": False,
            "required_for_stages": [],
        },
    )
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(payload)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(value) for value in exc.absolute_path)
        raise SpecValidationError(
            f"JSON Schema validation failed at {location or '<root>'}: "
            f"{exc.message}"
        ) from exc
    semantic_validate(payload)
    return TaskSpec(path=path, data=payload)


def dump_spec_snapshot(spec: TaskSpec, path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(spec.data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
