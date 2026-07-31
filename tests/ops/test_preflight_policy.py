from __future__ import annotations

from scripts.ops.preflight_policy import evaluate_remote_dirty_policy
from scripts.ops.ssh_ops import RemotePreflight, RemoteSubmoduleStatus


POLICY = {
    "allowed_tracked_paths": ["docs/EXPERIMENT_LOG.md"],
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
) -> RemotePreflight:
    return RemotePreflight(
        hostname="logini02",
        pwd="/remote/project",
        branch="main",
        commit="abc",
        python_version="Python 3.10",
        dirty_lines=(
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
