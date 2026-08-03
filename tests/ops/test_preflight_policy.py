from __future__ import annotations

from scripts.ops.preflight_policy import evaluate_remote_dirty_policy
from scripts.ops.ssh_ops import RemotePreflight, RemoteSubmoduleStatus


POLICY = {
    "allowed_tracked_paths": [
        "baselines/clear_official",
        "docs/EXPERIMENT_LOG.md",
    ],
    "allowed_patched_submodules": [
        {
            "path": "baselines/clear_official",
            "allow_modified": True,
            "allow_untracked": False,
            "allow_staged": False,
            "required_markers": [
                {
                    "file": "src/main.py",
                    "contains": "CLEAR_WRAPPER_SUPPORT_MUTAGENICITY_DATASET",
                }
            ],
            "allowed_modified_paths": [
                "src/data_preprocessing.py",
                "src/main.py",
                "src/models.py",
                "src/train_pred.py",
            ],
            "allowed_untracked_paths": ["dataset", "src/__pycache__"],
        }
    ],
}


def preflight(
    *,
    modified: tuple[str, ...] = ("src/main.py",),
    staged: tuple[str, ...] = (),
    status: tuple[str, ...] = (" M src/main.py",),
    marker: bool = True,
    dirty_lines: tuple[str, ...] | None = None,
) -> RemotePreflight:
    return RemotePreflight(
        hostname="logini02",
        pwd="/remote/project",
        branch="main",
        commit="abc",
        python_version="Python 3.10",
        dirty_lines=dirty_lines
        if dirty_lines is not None
        else (
            " m baselines/clear_official",
            " M docs/EXPERIMENT_LOG.md",
        ),
        conda_ready=True,
        sbatch_ready=True,
        sacct_ready=True,
        finalized_output_blocked=False,
        finalized_paths=(),
        proxy_present={},
        submodules=(
            RemoteSubmoduleStatus(
                path="baselines/clear_official",
                status_lines=status,
                modified_paths=modified,
                staged_paths=staged,
                marker_results={"src/main.py": marker},
            ),
        ),
    )


def test_dynamic_log_and_verified_patched_submodule_are_allowed() -> None:
    result = evaluate_remote_dirty_policy(preflight(), POLICY)
    assert result.passed
    assert result.dynamic_tracked == ("docs/EXPERIMENT_LOG.md",)
    assert result.verified_patched_submodules == (
        "baselines/clear_official",
    )
    assert result.blocked == ()
    assert result.remote_tracked_dirty_paths == (
        "baselines/clear_official",
        "docs/EXPERIMENT_LOG.md",
    )
    assert result.allowed_remote_tracked_dirty_paths == (
        "baselines/clear_official",
        "docs/EXPERIMENT_LOG.md",
    )
    assert result.disallowed_remote_tracked_dirty_paths == ()


def test_missing_tracked_allowlist_preserves_default_blocking() -> None:
    policy = {**POLICY, "allowed_tracked_paths": []}
    result = evaluate_remote_dirty_policy(preflight(), policy)
    assert not result.passed
    assert result.allowed_remote_tracked_dirty_paths == ()
    assert result.disallowed_remote_tracked_dirty_paths == (
        "baselines/clear_official",
        "docs/EXPERIMENT_LOG.md",
    )


def test_clean_remote_passes_with_empty_allowlist() -> None:
    policy = {**POLICY, "allowed_tracked_paths": []}
    result = evaluate_remote_dirty_policy(
        preflight(dirty_lines=()), policy
    )
    assert result.passed
    assert result.remote_tracked_dirty_paths == ()
    assert result.allowed_remote_tracked_dirty_paths == ()
    assert result.disallowed_remote_tracked_dirty_paths == ()


def test_unknown_tracked_path_blocks_even_when_known_paths_are_allowed() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(
            dirty_lines=(
                " m baselines/clear_official",
                " M docs/EXPERIMENT_LOG.md",
                " M notes/unexpected.txt",
            )
        ),
        POLICY,
    )
    assert result.blocked == ("notes/unexpected.txt",)
    assert result.disallowed_remote_tracked_dirty_paths == (
        "notes/unexpected.txt",
    )


def test_allowlist_is_exact_and_never_matches_a_prefix() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(
            dirty_lines=(
                " M docs/EXPERIMENT_LOG.md.backup",
                " M docs/EXPERIMENT_LOG.md/child",
            )
        ),
        POLICY,
    )
    assert result.allowed_remote_tracked_dirty_paths == ()
    assert result.disallowed_remote_tracked_dirty_paths == (
        "docs/EXPERIMENT_LOG.md.backup",
        "docs/EXPERIMENT_LOG.md/child",
    )


def test_root_untracked_policy_is_not_relaxed() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(dirty_lines=("?? scripts/paper/local.py",)), POLICY
    )
    assert result.remote_tracked_dirty_paths == ()
    assert result.remote_untracked_paths == ("scripts/paper/local.py",)
    assert result.blocked == ("scripts/paper/local.py",)


def test_lowercase_m_identifies_exact_submodule_parent_path() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(dirty_lines=(" m baselines/clear_official",)), POLICY
    )
    assert result.passed
    assert result.remote_tracked_dirty_paths == (
        "baselines/clear_official",
    )
    assert result.allowed_remote_tracked_dirty_paths == (
        "baselines/clear_official",
    )


def test_unexpected_modified_path_blocks_submodule() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(modified=("src/main.py", "surprise.py")), POLICY
    )
    assert result.blocked == ("baselines/clear_official",)
    assert result.submodule_audits[0]["unexpected_nested_paths"] == [
        "surprise.py"
    ]


def test_untracked_path_blocks_submodule() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(status=(" M src/main.py", "?? scratch.txt")), POLICY
    )
    assert result.blocked == ("baselines/clear_official",)
    assert result.submodule_audits[0]["nested_untracked"] == ["scratch.txt"]


def test_only_declared_generated_untracked_paths_are_allowed() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(
            status=(
                " M src/main.py",
                "?? dataset/",
                "?? src/__pycache__/",
            )
        ),
        POLICY,
    )
    assert result.passed
    audit = result.submodule_audits[0]
    assert audit["allowed_nested_untracked"] == [
        "dataset/",
        "src/__pycache__/",
    ]
    assert audit["unexpected_nested_untracked"] == []


def test_staged_path_blocks_submodule() -> None:
    result = evaluate_remote_dirty_policy(
        preflight(staged=("src/main.py",), status=("M  src/main.py",)), POLICY
    )
    assert result.blocked == ("baselines/clear_official",)
    assert result.submodule_audits[0]["nested_staged"] == ["src/main.py"]


def test_missing_marker_blocks_submodule() -> None:
    result = evaluate_remote_dirty_policy(preflight(marker=False), POLICY)
    assert result.blocked == ("baselines/clear_official",)
    assert result.submodule_audits[0]["missing_markers"] == ["src/main.py"]
