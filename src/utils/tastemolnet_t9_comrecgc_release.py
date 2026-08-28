"""Static release boundary for the managed TasteMolNet T9 smoke.

This tracked document never acts as a dynamic controller or GPU lease.  The
future one-parent release child must bind an immutable implementation and one
external route authority; the live process/lock proof comes exclusively from
``autodl_managed_execution``.  The checked-in candidate stays disabled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from src.baselines.comrecgc.held_upstream import (
    OFFICIAL_SOURCE_FILES,
    OFFICIAL_SOURCE_SHA256,
)
from src.baselines.tastemolnet_comrecgc_smoke import (
    OFFICIAL_COMRECGC_COMMIT,
    TASK_ID,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_CONFIG_PATH = (
    REPO_ROOT
    / "configs/autodl/tastemolnet_t9_comrecgc_smoke_release_v1.json"
)
RELEASE_SCHEMA = "tastemolnet_t9_comrecgc_smoke_release_v1"
RECEIPT_KIND = "taste_t9_gpu2_v1"
VALIDATOR = "taste_t9_v1"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TasteComRecGCReleaseError(RuntimeError):
    """The static T9 release authority is disabled or malformed."""


class TasteComRecGCReleaseDisabled(TasteComRecGCReleaseError):
    """The checked-in T9 candidate is intentionally non-executable."""


def _absolute(value: Any, *, field: str) -> Path:
    if type(value) is not str:
        raise TasteComRecGCReleaseError(f"{field} must be one absolute path")
    path = Path(value)
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise TasteComRecGCReleaseError(f"{field} must be one normalized absolute path")
    return path


def load_t9_release_config(
    path: str | Path = RELEASE_CONFIG_PATH,
) -> dict[str, Any]:
    requested = Path(path)
    if requested != RELEASE_CONFIG_PATH:
        raise TasteComRecGCReleaseError("Taste T9 release config path changed")
    try:
        raw = requested.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TasteComRecGCReleaseError("Taste T9 release config is unreadable") from exc
    expected_keys = {
        "schema_version",
        "release_enabled",
        "release_state",
        "implementation_commit",
        "implementation_tree",
        "route_authority_path",
        "route_authority_sha256",
        "output_parent",
        "managed_receipt_kind",
        "managed_task_id",
        "managed_validator",
        "gpu_index",
        "official_commit",
        "official_file_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise TasteComRecGCReleaseError("Taste T9 release config keys changed")
    if (
        value.get("schema_version") != RELEASE_SCHEMA
        or type(value.get("release_enabled")) is not bool
        or type(value.get("release_state")) is not str
        or not value["release_state"]
        or value.get("managed_receipt_kind") != RECEIPT_KIND
        or value.get("managed_task_id") != TASK_ID
        or value.get("managed_validator") != VALIDATOR
        or type(value.get("gpu_index")) is not int
        or value["gpu_index"] != 2
        or value.get("official_commit") != OFFICIAL_COMRECGC_COMMIT
    ):
        raise TasteComRecGCReleaseError("Taste T9 fixed release contract changed")
    official = value.get("official_file_sha256")
    if type(official) is not dict or set(official) != set(OFFICIAL_SOURCE_FILES):
        raise TasteComRecGCReleaseError("Taste T9 official source inventory changed")
    if (
        any(
            type(digest) is not str or not _SHA256_RE.fullmatch(digest)
            for digest in official.values()
        )
        or official != dict(OFFICIAL_SOURCE_SHA256)
    ):
        raise TasteComRecGCReleaseError("Taste T9 official source SHA-256 changed")
    optional = (
        "implementation_commit",
        "implementation_tree",
        "route_authority_path",
        "route_authority_sha256",
        "output_parent",
    )
    if value["release_enabled"] is False:
        if any(value[key] is not None for key in optional):
            raise TasteComRecGCReleaseError(
                "disabled Taste T9 config must not carry partial release pins"
            )
        return value
    if value["release_state"] != "RELEASED_BY_REVIEWED_ONE_PARENT_SUCCESSOR":
        raise TasteComRecGCReleaseError("Taste T9 enabled release state changed")
    if (
        type(value["implementation_commit"]) is not str
        or not _SHA1_RE.fullmatch(value["implementation_commit"])
        or type(value["implementation_tree"]) is not str
        or not _SHA1_RE.fullmatch(value["implementation_tree"])
        or type(value["route_authority_sha256"]) is not str
        or not _SHA256_RE.fullmatch(value["route_authority_sha256"])
    ):
        raise TasteComRecGCReleaseError("Taste T9 immutable release pins are malformed")
    _absolute(value["route_authority_path"], field="route_authority_path")
    _absolute(value["output_parent"], field="output_parent")
    return value


def assert_t9_execution_released() -> dict[str, Any]:
    value = load_t9_release_config()
    if value["release_enabled"] is not True:
        raise TasteComRecGCReleaseDisabled(value["release_state"])
    return value


__all__ = [
    "RECEIPT_KIND",
    "RELEASE_CONFIG_PATH",
    "TASK_ID",
    "VALIDATOR",
    "TasteComRecGCReleaseDisabled",
    "TasteComRecGCReleaseError",
    "assert_t9_execution_released",
    "load_t9_release_config",
]
