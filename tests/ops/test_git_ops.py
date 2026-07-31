from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ops.git_ops import (
    GitSafetyError,
    path_allowed,
    stage_allowed_changes,
)
from scripts.ops.subprocess_utils import CommandResult


class FakeGitRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def run(self, argv, *, cwd, dry_run=False, **kwargs):
        self.calls.append(list(argv))
        response = self.responses.pop(0)
        return CommandResult(
            argv=list(argv),
            cwd=str(cwd),
            returncode=response[0],
            stdout=response[1],
            stderr=response[2],
            dry_run=dry_run,
        )


def test_allowlist_matches_exact_files_and_globs() -> None:
    assert path_allowed("scripts/ops/spec.py", ["scripts/ops/**"])
    assert path_allowed("README.md", ["README.md"])
    assert not path_allowed("scripts/train.py", ["scripts/ops/**"])


def test_staged_extra_file_blocks_before_git_add(tmp_path: Path) -> None:
    runner = FakeGitRunner(
        [
            (0, " M scripts/ops/spec.py\0 M README.md\0", ""),
            (0, "README.md\n", ""),
        ]
    )
    with pytest.raises(GitSafetyError, match="Already-staged"):
        stage_allowed_changes(
            runner, tmp_path, ["scripts/ops/**"], dry_run=False
        )
    assert not any(call[1:3] == ["add", "--"] for call in runner.calls)


def test_exact_allowed_dirty_set_is_staged_and_unrelated_is_untouched(
    tmp_path: Path,
) -> None:
    runner = FakeGitRunner(
        [
            (0, " M scripts/ops/spec.py\0 M docs/EXPERIMENT_LOG.md\0", ""),
            (0, "", ""),
            (0, "", ""),
            (0, " M scripts/ops/spec.py\0 M docs/EXPERIMENT_LOG.md\0", ""),
            (0, "scripts/ops/spec.py\n", ""),
        ]
    )
    status = stage_allowed_changes(
        runner, tmp_path, ["scripts/ops/**"], dry_run=False
    )
    add_call = runner.calls[2]
    assert add_call == ["git", "add", "--", "scripts/ops/spec.py"]
    assert status.unrelated_modified_paths == ("docs/EXPERIMENT_LOG.md",)


def test_dry_run_never_invokes_git_add(tmp_path: Path) -> None:
    runner = FakeGitRunner(
        [
            (0, " M scripts/ops/spec.py\0", ""),
            (0, "", ""),
        ]
    )
    stage_allowed_changes(runner, tmp_path, ["scripts/ops/**"], dry_run=True)
    assert all(call[1] != "add" for call in runner.calls)


def test_git_module_contains_no_destructive_or_broad_commands() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts/ops/git_ops.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "reset --hard",
        "git clean",
        "git stash",
        '["add", "-A"]',
        '["add", "."]',
    ):
        assert forbidden not in source
