"""Runtime environment and lightweight YAML config helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


class ExecutionEnvironment(str, Enum):
    """Supported execution environments for local and HPC runs."""

    LOCAL = "local"
    HPC = "hpc"


def detect_execution_environment(
    environ: Mapping[str, str] | None = None,
) -> ExecutionEnvironment:
    """Infer whether the current process is running locally or under Slurm."""

    env = dict(os.environ if environ is None else environ)
    override = env.get("RUN_ENV")
    if override in (ExecutionEnvironment.LOCAL.value, ExecutionEnvironment.HPC.value):
        return ExecutionEnvironment(override)
    if any(key in env for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_CLUSTER_NAME")):
        return ExecutionEnvironment.HPC
    return ExecutionEnvironment.LOCAL


def parse_scalar(raw_value: str) -> Any:
    """Parse a scalar value from a simple YAML subset."""

    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {'"', "'"} and value[-1] == value[0]:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a lightweight mapping-only YAML file without external dependencies."""

    file_path = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped_line = raw_line.rstrip()
            if not stripped_line.strip():
                continue
            content = stripped_line.lstrip(" ")
            if content.startswith("#"):
                continue

            indent = len(stripped_line) - len(content)
            if indent % 2 != 0:
                raise ValueError(
                    f"{file_path}:{line_number} uses unsupported odd indentation."
                )
            if ":" not in content:
                raise ValueError(
                    f"{file_path}:{line_number} is not a supported key/value YAML line."
                )

            key, _, raw_value = content.partition(":")
            key = key.strip()
            if not key:
                raise ValueError(f"{file_path}:{line_number} has an empty key.")

            while stack and indent <= stack[-1][0]:
                stack.pop()
            current = stack[-1][1]

            value = raw_value.strip()
            if not value:
                nested: dict[str, Any] = {}
                current[key] = nested
                stack.append((indent, nested))
                continue

            current[key] = parse_scalar(value)

    return root


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_and_merge_config_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Load and merge config files in the provided order."""

    merged: dict[str, Any] = {}
    for path in paths:
        config = load_yaml_config(path)
        merged = deep_merge_dicts(merged, config)
    return merged


def apply_dotted_override(config: Mapping[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Return a copy of a config with one dotted-key override applied."""

    updated = deepcopy(dict(config))
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError("Override key must not be empty.")

    cursor: dict[str, Any] = updated
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value
    return updated


def apply_dotlist_overrides(
    config: Mapping[str, Any],
    overrides: Sequence[str],
) -> dict[str, Any]:
    """Apply CLI overrides of the form ``section.key=value``."""

    updated = deepcopy(dict(config))
    for item in overrides:
        key, separator, raw_value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE format.")
        updated = apply_dotted_override(updated, key, parse_scalar(raw_value))
    return updated
