"""Policy evaluation for read-only remote preflight evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from scripts.ops.git_ops import path_allowed
from scripts.ops.ssh_ops import RemotePreflight, RemoteSubmoduleStatus


@dataclass(frozen=True, slots=True)
class RemoteDirtyPolicyResult:
    dynamic_tracked: tuple[str, ...]
    verified_patched_submodules: tuple[str, ...]
    blocked: tuple[str, ...]
    submodule_audits: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        return not self.blocked

    def to_dict(self) -> dict[str, object]:
        return {
            "dynamic_tracked": list(self.dynamic_tracked),
            "verified_patched_submodules": list(
                self.verified_patched_submodules
            ),
            "blocked": list(self.blocked),
        }


def _status_path(line: str) -> str:
    return line[3:].strip() if len(line) >= 4 else line.strip()


def _untracked_paths(status: RemoteSubmoduleStatus) -> list[str]:
    return sorted(
        {
            _status_path(line)
            for line in status.status_lines
            if line.startswith("??")
        }
    )


def evaluate_remote_dirty_policy(
    preflight: RemotePreflight,
    policy: Mapping[str, Any],
) -> RemoteDirtyPolicyResult:
    """Classify root dirt and verify declared patched nested repositories."""

    allowed_tracked = list(policy.get("allowed_tracked_paths") or [])
    submodule_policies = {
        str(item["path"]): item
        for item in policy.get("allowed_patched_submodules") or []
    }
    evidence = {item.path: item for item in preflight.submodules}
    dynamic: list[str] = []
    blocked: list[str] = []
    root_submodule_lines: dict[str, list[str]] = {
        path: [] for path in submodule_policies
    }

    for line in preflight.dirty_lines:
        path = _status_path(line)
        if path in submodule_policies:
            root_submodule_lines[path].append(line)
            if len(line) < 2 or line[0] != " " or line[1] != "m":
                blocked.append(path)
            continue
        if path_allowed(path, allowed_tracked):
            dynamic.append(path)
        else:
            blocked.append(path)

    verified: list[str] = []
    audits: list[dict[str, Any]] = []
    for path, item in submodule_policies.items():
        nested = evidence.get(path)
        if nested is None:
            audits.append(
                {
                    "path": path,
                    "nested_modified": [],
                    "nested_staged": [],
                    "nested_untracked": [],
                    "unexpected_nested_paths": [],
                    "missing_markers": [
                        str(marker["file"])
                        for marker in item.get("required_markers") or []
                    ],
                    "root_status_lines": root_submodule_lines[path],
                    "verified": False,
                    "patched_submodule_verified": False,
                    "error": "missing_submodule_preflight_evidence",
                }
            )
            blocked.append(path)
            continue

        modified = sorted(set(nested.modified_paths))
        staged = sorted(set(nested.staged_paths))
        untracked = _untracked_paths(nested)
        allowed_modified = list(item.get("allowed_modified_paths") or [])
        allowed_untracked_patterns = list(
            item.get("allowed_untracked_paths") or []
        )
        unexpected = sorted(
            value
            for value in modified
            if not path_allowed(value, allowed_modified)
        )
        unexpected_untracked = sorted(
            value
            for value in untracked
            if not path_allowed(value, allowed_untracked_patterns)
        )
        allowed_untracked = sorted(set(untracked) - set(unexpected_untracked))
        missing_markers = sorted(
            str(marker["file"])
            for marker in item.get("required_markers") or []
            if not nested.marker_results.get(str(marker["file"]), False)
        )
        violations: list[str] = []
        if modified and not bool(item.get("allow_modified", False)):
            violations.append("modified_not_allowed")
        if staged and not bool(item.get("allow_staged", False)):
            violations.append("staged_not_allowed")
        if (
            unexpected_untracked
            and not bool(item.get("allow_untracked", False))
        ):
            violations.append("untracked_not_allowed")
        if unexpected:
            violations.append("unexpected_modified_paths")
        if missing_markers:
            violations.append("missing_required_markers")
        if any(
            len(line) < 2 or line[0] != " " or line[1] != "m"
            for line in root_submodule_lines[path]
        ):
            violations.append("top_level_submodule_index_dirty")

        is_verified = not violations
        if is_verified:
            verified.append(path)
        else:
            blocked.append(path)
        audits.append(
            {
                "path": path,
                "nested_modified": modified,
                "nested_staged": staged,
                "nested_untracked": untracked,
                "allowed_nested_untracked": allowed_untracked,
                "unexpected_nested_untracked": unexpected_untracked,
                "unexpected_nested_paths": unexpected,
                "missing_markers": missing_markers,
                "root_status_lines": root_submodule_lines[path],
                "violations": violations,
                "verified": is_verified,
                "patched_submodule_verified": is_verified,
            }
        )

    return RemoteDirtyPolicyResult(
        dynamic_tracked=tuple(sorted(set(dynamic))),
        verified_patched_submodules=tuple(sorted(set(verified))),
        blocked=tuple(sorted(set(blocked))),
        submodule_audits=tuple(audits),
    )


def proxy_is_ready(
    proxy_present: Mapping[str, bool], policy: Mapping[str, Any]
) -> bool:
    """Return availability only; proxy values are intentionally unavailable."""

    if not bool(policy.get("require_any_present_for_git_network", False)):
        return True
    return any(bool(value) for value in proxy_present.values())
