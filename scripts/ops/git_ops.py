"""Git operations constrained to an explicit task allowlist."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

from scripts.ops.subprocess_utils import CommandResult, CommandRunner


class GitSafetyError(RuntimeError):
    """A Git action would exceed the task's declared scope."""


@dataclass(frozen=True, slots=True)
class GitStatus:
    modified_paths: tuple[str, ...]
    staged_paths: tuple[str, ...]
    allowed_modified_paths: tuple[str, ...]
    unrelated_modified_paths: tuple[str, ...]


def _run_git(
    runner: CommandRunner,
    root: Path,
    args: list[str],
    *,
    dry_run: bool = False,
) -> CommandResult:
    return runner.run(["git", *args], cwd=root, dry_run=dry_run)


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def path_allowed(path: str, allowed_paths: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in allowed_paths:
        candidate = pattern.replace("\\", "/").rstrip("/")
        if any(token in candidate for token in ("*", "?", "[")):
            if fnmatch(normalized, candidate):
                return True
        elif normalized == candidate or normalized.startswith(candidate + "/"):
            return True
    return False


def modified_paths(runner: CommandRunner, root: Path) -> list[str]:
    result = _run_git(
        runner,
        root,
        [
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--no-renames",
        ],
    )
    if result.returncode != 0:
        raise GitSafetyError(result.stderr or "git status failed")
    entries = result.stdout.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4:
            raise GitSafetyError(f"Unexpected porcelain entry: {entry!r}")
        path = entry[3:]
        paths.append(path)
    return sorted(set(paths))


def staged_paths(runner: CommandRunner, root: Path) -> list[str]:
    result = _run_git(
        runner, root, ["diff", "--cached", "--name-only", "--diff-filter=ACMRD"]
    )
    if result.returncode != 0:
        raise GitSafetyError(result.stderr or "git diff --cached failed")
    return sorted(set(_lines(result.stdout)))


def inspect_status(
    runner: CommandRunner, root: Path, allowed_paths: Iterable[str]
) -> GitStatus:
    modified = modified_paths(runner, root)
    staged = staged_paths(runner, root)
    allowed = sorted(
        path for path in modified if path_allowed(path, allowed_paths)
    )
    unrelated = sorted(set(modified) - set(allowed))
    return GitStatus(
        modified_paths=tuple(modified),
        staged_paths=tuple(staged),
        allowed_modified_paths=tuple(allowed),
        unrelated_modified_paths=tuple(unrelated),
    )


def stage_allowed_changes(
    runner: CommandRunner,
    root: Path,
    allowed_paths: list[str],
    *,
    dry_run: bool,
) -> GitStatus:
    before = inspect_status(runner, root, allowed_paths)
    staged_extras = [
        path
        for path in before.staged_paths
        if not path_allowed(path, allowed_paths)
    ]
    if staged_extras:
        raise GitSafetyError(
            f"Already-staged paths exceed allowed_paths: {staged_extras}"
        )
    targets = list(before.allowed_modified_paths)
    if not targets:
        return before
    if not dry_run:
        result = _run_git(runner, root, ["add", "--", *targets])
        if result.returncode != 0:
            raise GitSafetyError(result.stderr or "git add failed")
        after = inspect_status(runner, root, allowed_paths)
        expected = set(targets) | set(before.staged_paths)
        actual = set(after.staged_paths)
        if actual != expected:
            raise GitSafetyError(
                "Staged set differs from the exact allowed dirty set: "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        return after
    return before


def cached_diff_check(runner: CommandRunner, root: Path) -> None:
    result = _run_git(runner, root, ["diff", "--cached", "--check"])
    if result.returncode != 0:
        raise GitSafetyError(result.stdout + result.stderr)


def current_branch(runner: CommandRunner, root: Path) -> str:
    result = _run_git(runner, root, ["branch", "--show-current"])
    if result.returncode != 0:
        raise GitSafetyError(result.stderr or "Could not resolve Git branch")
    return result.stdout.strip()


def head_commit(runner: CommandRunner, root: Path) -> str:
    result = _run_git(runner, root, ["rev-parse", "HEAD"])
    if result.returncode != 0:
        raise GitSafetyError(result.stderr or "Could not resolve Git HEAD")
    return result.stdout.strip()


def commits_changed_paths(
    runner: CommandRunner, root: Path, branch: str
) -> list[str]:
    result = _run_git(
        runner,
        root,
        ["diff", "--name-only", f"origin/{branch}..HEAD"],
    )
    if result.returncode != 0:
        raise GitSafetyError(
            result.stderr or f"Could not diff against origin/{branch}"
        )
    return sorted(set(_lines(result.stdout)))


def assert_not_behind_origin(
    runner: CommandRunner, root: Path, branch: str
) -> None:
    result = _run_git(
        runner,
        root,
        ["merge-base", "--is-ancestor", f"origin/{branch}", "HEAD"],
    )
    if result.returncode != 0:
        raise GitSafetyError(
            f"HEAD is behind or unrelated to origin/{branch}; update manually."
        )


def commit_allowed(
    runner: CommandRunner,
    root: Path,
    *,
    branch: str,
    message: str,
    dry_run: bool,
) -> str:
    if current_branch(runner, root) != branch:
        raise GitSafetyError(f"Expected branch {branch!r}.")
    cached_diff_check(runner, root)
    if dry_run:
        return head_commit(runner, root)
    result = _run_git(runner, root, ["commit", "-m", message])
    if result.returncode != 0:
        raise GitSafetyError(result.stdout + result.stderr)
    return head_commit(runner, root)


def push_head(
    runner: CommandRunner,
    root: Path,
    *,
    branch: str,
    dry_run: bool,
) -> str:
    if current_branch(runner, root) != branch:
        raise GitSafetyError(f"Expected branch {branch!r}.")
    assert_not_behind_origin(runner, root, branch)
    commit = head_commit(runner, root)
    if not dry_run:
        result = _run_git(
            runner, root, ["push", "origin", f"HEAD:{branch}"]
        )
        if result.returncode != 0:
            raise GitSafetyError(result.stdout + result.stderr)
    return commit
