#!/usr/bin/env python3
"""Production stage runner for the four-lane AutoDL recovery campaign.

This module is intentionally a *scientific* stage boundary, not a scheduler.
``run_three_lines.py`` owns processes and dependencies; this file owns exact
CLI wiring, immutable-input checks, crash-safe materialisation, output digests,
and the scientific PASS sentinels consumed by the controller.

Mutagenicity and AIDS are freeze-only recoveries.  They fail closed unless
``DISALLOW_GENERATION=1`` and this runner refuses to launch a generation
entrypoint for either dataset.  BACE is the only lane allowed to run fresh
generation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import re
import runpy
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


RUN_ID = "autodl_three_lines_20260821_v1"
UPSTREAM_COMMIT = "122f9341a360e9f06bb58a2f5823bb596021f6bf"
MUT_SOURCE_COMMIT = "7f7ed51a1176de1c23344cda0fbf0e6c5ba210b4"
AIDS_SOURCE_COMMIT = "a418692b75b888297222d31d87f49148505e10d0"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SECRET = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|authorization|credential|private[_-]?key)"
)
SECRET_VALUE = re.compile(
    r"(?i)(?:BEGIN [A-Z ]*PRIVATE KEY|"
    r"(?:^|[?&;\s])(?:password|passwd|token|secret|api[_-]?key|authorization|credential)\s*=|"
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}|"
    r"\bsk-[A-Za-z0-9_-]{16,}|"
    r"\bAKIA[0-9A-Z]{16}\b)"
)


class StageError(RuntimeError):
    """A fail-closed stage error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
            "utf-8"
        ),
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StageError(f"Invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise StageError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise StageError(f"Unsafe manifest path: {value!r}")
    return path


def verify_sha256_manifest(
    root: Path,
    manifest: Path,
    *,
    exact_inventory: bool = False,
    allowed_unlisted: Iterable[Path] = (),
) -> str:
    """Verify a GNU-style SHA manifest and return its own digest."""

    if not root.is_dir() or not manifest.is_file() or manifest.is_symlink():
        raise StageError(f"Missing physical input root/manifest: {root}, {manifest}")
    seen: set[Path] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        if len(line) < 67 or line[64:66] not in {"  ", " *"}:
            raise StageError(f"Malformed SHA256 line {line_number}: {manifest}")
        expected = line[:64].lower()
        if not SHA256.fullmatch(expected):
            raise StageError(f"Malformed SHA256 at line {line_number}: {manifest}")
        relative = _safe_relative(line[66:])
        if relative in seen:
            raise StageError(f"Duplicate SHA256 path: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise StageError(f"Manifest file missing or non-physical: {path}")
        resolved = path.resolve(strict=True)
        if not _is_within(resolved, root):
            raise StageError(f"Manifest path escapes its root: {path} -> {resolved}")
        if sha256_file(resolved) != expected:
            raise StageError(f"Input SHA256 mismatch: {path}")
    if not seen:
        raise StageError(f"Empty SHA256 manifest: {manifest}")
    if exact_inventory:
        allowed = {
            value.resolve(strict=False)
            for value in [manifest, *allowed_unlisted]
        }
        actual = {
            path.relative_to(root)
            for path in _physical_files(root)
            if path.resolve(strict=False) not in allowed
        }
        extras = sorted(actual - seen, key=lambda value: value.as_posix())
        missing = sorted(seen - actual, key=lambda value: value.as_posix())
        if extras or missing:
            raise StageError(
                "SHA256 manifest inventory mismatch: "
                f"unlisted={extras[:8]}, absent={missing[:8]}"
            )
    return sha256_file(manifest)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _physical_files(root: Path) -> Iterator[Path]:
    if root.is_symlink():
        raise StageError(f"Output root must not be a symlink: {root}")
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise StageError(f"Output item is missing: {root}")
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise StageError(f"Output contains a symlink: {path}")
        for name in sorted(files):
            yield current_path / name


def write_sha256_manifest(
    *, base: Path, items: Iterable[Path], manifest: Path, exclude: Iterable[Path] = ()
) -> str:
    excluded = {value.resolve(strict=False) for value in exclude}
    rows: dict[str, str] = {}
    for item in items:
        for path in _physical_files(item):
            resolved = path.resolve(strict=True)
            if resolved in excluded:
                continue
            if not _is_within(resolved, base):
                raise StageError(f"Manifest output escapes base: {resolved} not in {base}")
            relative = resolved.relative_to(base.resolve()).as_posix()
            rows[relative] = sha256_file(resolved)
    if not rows:
        raise StageError(f"Refusing to publish empty output manifest: {manifest}")
    body = "".join(f"{digest}  {name}\n" for name, digest in sorted(rows.items()))
    atomic_write_bytes(manifest, body.encode("utf-8"))
    verify_sha256_manifest(base, manifest)
    return sha256_file(manifest)


def _assert_nonempty_file(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise StageError(f"Required physical file is missing or empty: {path}")


def _assert_empty_or_absent(path: Path, *, label: str) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise StageError(f"{label} collision; refusing implicit overwrite: {path}")


def _copy_file_atomic(source: Path, destination: Path) -> None:
    _assert_nonempty_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as src, os.fdopen(descriptor, "wb") as dst:
            shutil.copyfileobj(src, dst, length=32 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if sha256_file(source) != sha256_file(temporary):
            raise StageError(f"Atomic copy digest mismatch: {source}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_tree_atomic(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise StageError(f"Tree source must be a physical directory: {source}")
    if destination.exists() or destination.is_symlink():
        raise StageError(f"Tree destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    published = False
    try:
        # copytree into an existing mkdtemp needs dirs_exist_ok.  symlinks=False
        # dereferences only safe physical inputs; reject links explicitly first.
        list(_physical_files(source))
        shutil.copytree(source, temporary, dirs_exist_ok=True, symlinks=False)
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def _assert_tree_contains_identical(source: Path, destination: Path) -> None:
    for source_file in _physical_files(source):
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        _assert_nonempty_file(destination_file)
        if (
            source_file.stat().st_size != destination_file.stat().st_size
            or sha256_file(source_file) != sha256_file(destination_file)
        ):
            raise StageError(f"Preserved input changed after materialisation: {relative}")


def _assert_no_secret(command: Sequence[str]) -> None:
    for index, token in enumerate(command):
        if SECRET.search(token):
            raise StageError(f"Command argument {index} appears to contain a secret")


def _assert_no_secret_environment(environment: Mapping[str, str]) -> None:
    """Reject explicit stage overrides that could expose a credential."""

    for key, value in environment.items():
        if SECRET.search(str(key)):
            raise StageError("Stage environment contains a credential-named key")
        if SECRET_VALUE.search(str(value)):
            raise StageError("Stage environment contains a credential-like value")


def _sanitized_inherited_environment() -> dict[str, str]:
    """Drop credential-bearing shell variables before spawning science code."""

    return {
        str(key): str(value)
        for key, value in os.environ.items()
        if not SECRET.search(str(key)) and not SECRET_VALUE.search(str(value))
    }


def _assert_no_generation(command: Sequence[str]) -> None:
    forbidden = {
        "run_generation.py",
        "comrecgc_project_generate.sh",
        "comrecgc_mut_full.sh",
        "comrecgc_aids_native_full.sh",
    }
    names = {Path(value).name for value in command}
    unsafe = sorted(names & forbidden)
    if unsafe:
        raise StageError(f"Freeze-only lane attempted generation: {unsafe}")


_CHILD_TERM_GRACE_SECONDS = 2.0


def _signal_child_process_group(
    process: subprocess.Popen[Any], signum: int
) -> None:
    """Signal the isolated process group owned by ``process``.

    Production children are always launched with ``start_new_session=True``,
    so their PID is also their process-group ID.  The ``send_signal`` fallback
    exists only for the small Popen test doubles which intentionally have no
    operating-system PID.
    """

    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 1:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            return
        return
    process.send_signal(signum)


def _terminate_child_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = _CHILD_TERM_GRACE_SECONDS,
    initial_signal: int = signal.SIGTERM,
) -> None:
    """Best-effort TERM/KILL/reap without masking an in-flight exception."""

    try:
        running = process.poll() is None
    except BaseException:
        running = True
    if running:
        try:
            _signal_child_process_group(process, initial_signal)
        except BaseException:
            # Continue to wait/kill; cleanup must never replace the scientific
            # or monitoring exception which selected this path.
            pass
    try:
        process.wait(timeout=float(grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    except BaseException:
        # A broken wait implementation is treated as an unconfirmed stop and
        # receives the same fail-safe KILL attempt.
        pass
    try:
        _signal_child_process_group(process, signal.SIGKILL)
    except BaseException:
        pass
    try:
        process.wait()
    except BaseException:
        pass


@contextmanager
def _forward_signals(process: subprocess.Popen[Any]) -> Iterator[None]:
    previous: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        if process.poll() is None:
            # Merely forwarding a signal is insufficient when the scientific
            # process ignores TERM: the outer controller could later kill this
            # stage's distinct PGID while leaving the scientific PGID alive.
            # Complete the child group's bounded TERM/INT -> KILL -> reap
            # sequence before this stage exits with conventional signal status.
            _terminate_child_process_group(process, initial_signal=signum)
        raise SystemExit(128 + int(signum))

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    env_extra: Mapping[str, str] | None = None,
    allow_generation: bool = False,
    monitor: Callable[[], None] | None = None,
) -> None:
    argv = [str(value) for value in command]
    _assert_no_secret(argv)
    if not allow_generation:
        _assert_no_generation(argv)
    environment = _sanitized_inherited_environment()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(cwd) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    explicit_environment = {
        str(key): str(value) for key, value in (env_extra or {}).items()
    }
    _assert_no_secret_environment(explicit_environment)
    environment.update(explicit_environment)
    process = subprocess.Popen(
        argv, cwd=cwd, env=environment, start_new_session=True
    )
    try:
        with _forward_signals(process):
            while process.poll() is None:
                if monitor is not None:
                    monitor()
                time.sleep(1.0)
        if monitor is not None:
            monitor()
    except BaseException:
        _terminate_child_process_group(process)
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(int(process.returncode), argv)


@dataclass(frozen=True, slots=True)
class Context:
    project: Path
    step0: Path
    external: Path
    persistent: Path
    fast: Path
    python: Path
    resume: bool
    stage_start_code_lineage: Mapping[str, str] | None = None

    @property
    def static_input(self) -> Path:
        return self.persistent / "inputs" / "static_project"

    @property
    def bace_input(self) -> Path:
        return self.persistent / "inputs" / "bace_preserved"

    def script(self, relative: str) -> str:
        return str(self.project / relative)


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    primary_root: Path
    primary_manifest: Path
    primary_digest: str
    static_manifest: Path
    static_digest: str
    required_static_manifest: Path
    required_static_digest: str
    required_static_source_root: Path


def _validate_roots(context: Context) -> None:
    for label, path in (
        ("project", context.project),
        ("step0", context.step0),
        ("external", context.external),
        ("persistent", context.persistent),
    ):
        if not path.is_dir() or path.is_symlink():
            raise StageError(f"{label} root is missing or a symlink: {path}")
    _assert_nonempty_file(context.python)
    if not os.access(context.python, os.X_OK):
        raise StageError(f"Python interpreter is not executable: {context.python}")
    if (
        context.persistent == context.fast
        or _is_within(context.fast, context.persistent)
        or _is_within(context.persistent, context.fast)
    ):
        raise StageError("Fast and persistent run roots must be disjoint")


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise StageError(f"git rev-parse failed under {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _external_worktree_lineage(context: Context) -> dict[str, str]:
    """Bind the pinned upstream checkout to its exact, non-code worktree.

    The deployed vendor checkout contains one deliberately retained provenance
    file (``vendor_manifest.json``).  It is not executable upstream code, so it
    is allowed only at that exact root-relative path and is hashed into the
    lineage.  Every tracked modification, staged change, or other untracked
    path fails closed.
    """

    status = subprocess.run(
        [
            "git",
            "-C",
            str(context.external),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise StageError("Unable to inspect the external COMRECGC worktree")
    records = [value for value in status.stdout.split(b"\0") if value]
    allowed = b"?? vendor_manifest.json"
    unexpected = [value for value in records if value != allowed]
    if unexpected:
        raise StageError(
            "External COMRECGC worktree has tracked/staged or unapproved "
            "untracked changes"
        )
    provenance = context.external / "vendor_manifest.json"
    if allowed in records:
        if (
            not provenance.is_file()
            or provenance.is_symlink()
            or provenance.parent.resolve(strict=True) != context.external
        ):
            raise StageError("External vendor_manifest.json is not a physical root file")
        provenance_digest = sha256_file(provenance)
    else:
        provenance_digest = "ABSENT"
    tree = subprocess.run(
        ["git", "-C", str(context.external), "rev-parse", "HEAD^{tree}"],
        text=True,
        capture_output=True,
        check=False,
    )
    tree_id = tree.stdout.strip()
    if tree.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", tree_id):
        raise StageError("Unable to resolve the external COMRECGC tree object")
    return {
        "external_comrecgc_tree": tree_id,
        "external_provenance_sha256": provenance_digest,
    }


def _repair_code_closure_sha256(context: Context) -> str:
    """Bind smoke evidence to exact repair content, independent of Git HEAD."""

    # Recovery, downstream evaluation, common4 auditing and checkpointing cross
    # several packages.  Hash the complete executable Python/shell closure plus
    # the exact production config/spec so a dirty or untracked science change
    # cannot be hidden behind an unchanged Git HEAD.
    code_paths = list((context.project / "src").glob("**/*.py"))
    code_paths.extend((context.project / "scripts").glob("**/*.py"))
    code_paths.extend((context.project / "scripts").glob("**/*.sh"))
    code_paths.extend(
        [
            context.project / "configs/hpc.yaml",
            context.project / "ops/specs/autodl_three_lines_20260821.yaml",
        ]
    )
    relative_paths = sorted(
        {path.relative_to(context.project) for path in code_paths},
        key=lambda path: path.as_posix(),
    )
    if not relative_paths:
        raise StageError("Repair code closure is empty")
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = context.project / relative
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file() or not _is_within(target, context.project):
                raise StageError(
                    f"Repair code closure symlink escapes the project: {path}"
                )
            digest.update(
                (
                    f"L {os.readlink(path)} {sha256_file(target)}  "
                    f"{relative.as_posix()}\n"
                ).encode("utf-8")
            )
            continue
        if not path.is_file():
            raise StageError(f"Repair code closure contains a non-physical file: {path}")
        if not _is_within(path.resolve(strict=True), context.project):
            raise StageError(f"Repair code closure escapes project: {path}")
        digest.update(
            f"{sha256_file(path)}  {relative.as_posix()}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _current_code_lineage(context: Context) -> dict[str, str]:
    external_commit = _git_head(context.external)
    if external_commit != UPSTREAM_COMMIT:
        raise StageError(
            f"External COMRECGC commit changed: {external_commit} != {UPSTREAM_COMMIT}"
        )
    return {
        "repair_code_closure_sha256": _repair_code_closure_sha256(context),
        "project_commit": _git_head(context.project),
        "external_comrecgc_commit": external_commit,
        **_external_worktree_lineage(context),
    }


def _assert_stage_lineage_unchanged(context: Context) -> dict[str, str]:
    current = _current_code_lineage(context)
    started = context.stage_start_code_lineage
    if started is not None and current != dict(started):
        raise StageError("Project/config/vendor lineage changed while the stage was running")
    return current


def _input_gate(context: Context, name: str) -> tuple[Path, Path, str]:
    root = context.persistent / "inputs" / name
    manifest = root / "MANIFEST.sha256"
    return root, manifest, verify_sha256_manifest(
        root, manifest, exact_inventory=True
    )


def _required_step0_items(context: Context) -> list[Path]:
    """Return the complete shared Step0 subset read by all eight stages."""

    step0 = context.step0
    return [
        step0 / "outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset",
        step0 / "outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt",
        step0
        / "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"
        / "test_source_label1_teacher_correct.csv",
        step0 / "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl",
        step0 / "outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1/thresholds.json",
        step0 / "outputs/hpc/current/gcfexplainer/aids/dataset",
        step0 / "outputs/hpc/oracle/aids_rf_model.pkl",
        step0
        / "outputs/hpc/eval/paper/molclr_node_wasserstein_figure4_redline_k10"
        / "wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv",
        step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset",
        step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/gnn/model_best.pth",
        step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/neurosed/best_model.pt",
        step0 / "outputs/hpc/oracle/bace/bace_teacher.pkl",
        step0 / "pretrained_models/MolCLR",
    ]


def _required_static_gate(
    context: Context, *, publish_if_missing: bool = True
) -> tuple[Path, str]:
    """Create once, then verify, the immutable shared Step0 input manifest.

    Four lane processes can enter this function concurrently.  The lock keeps
    first publication single-writer; the resulting manifest and metadata are
    mode 0444 so a later stage cannot silently redefine the input cohort.
    """

    root = context.persistent / "manifests"
    manifest = root / "required_static_inputs.sha256"
    metadata = root / "required_static_inputs.json"
    lock = root / "required_static_inputs.lock"
    if not publish_if_missing:
        if not manifest.is_file() or manifest.is_symlink():
            raise StageError("Shared required-static manifest is unavailable")
        if not metadata.is_file() or metadata.is_symlink():
            raise StageError("Shared required-static metadata is unavailable")
        payload = read_json(metadata)
        if payload.get("schema_version") != "autodl_required_static_inputs_v1":
            raise StageError("Shared required-static metadata schema mismatch")
        if payload.get("source_root") != str(context.step0):
            raise StageError("Shared required-static source root changed")
        digest = _verify_required_static_manifest(manifest, source_root=context.step0)
        if payload.get("manifest_sha256") != digest:
            raise StageError("Shared required-static metadata digest mismatch")
        return manifest, digest
    root.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if manifest.exists() or metadata.exists():
                if not manifest.is_file() or not metadata.is_file():
                    raise StageError("Shared required-static manifest is partially published")
                payload = read_json(metadata)
                if payload.get("schema_version") != "autodl_required_static_inputs_v1":
                    raise StageError("Shared required-static metadata schema mismatch")
                if payload.get("source_root") != str(context.step0):
                    raise StageError("Shared required-static source root changed")
                digest = _verify_required_static_manifest(
                    manifest, source_root=context.step0
                )
                if payload.get("manifest_sha256") != digest:
                    raise StageError("Shared required-static metadata digest mismatch")
                return manifest, digest
            digest = write_sha256_manifest(
                base=Path("/"),
                items=_required_step0_items(context),
                manifest=manifest,
                exclude=[manifest, metadata, lock],
            )
            atomic_write_json(
                metadata,
                {
                    "schema_version": "autodl_required_static_inputs_v1",
                    "manifest": str(manifest),
                    "manifest_root": "/",
                    "manifest_sha256": digest,
                    "source_root": str(context.step0),
                    "published_at": utc_now(),
                },
            )
            os.chmod(manifest, 0o444)
            os.chmod(metadata, 0o444)
            _fsync_directory(root)
            _verify_required_static_manifest(manifest, source_root=context.step0)
            return manifest, digest
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _all_input_gates(
    context: Context,
    primary_name: str,
    *,
    publish_required_static: bool = True,
) -> InputSnapshot:
    primary_root, primary_manifest, primary_digest = _input_gate(
        context, primary_name
    )
    static_root, static_manifest, static_digest = _input_gate(
        context, "static_project"
    )
    del static_root
    required_manifest, required_digest = _required_static_gate(
        context, publish_if_missing=publish_required_static
    )
    return InputSnapshot(
        primary_root=primary_root,
        primary_manifest=primary_manifest,
        primary_digest=primary_digest,
        static_manifest=static_manifest,
        static_digest=static_digest,
        required_static_manifest=required_manifest,
        required_static_digest=required_digest,
        required_static_source_root=context.step0,
    )


def _verify_required_static_manifest(manifest: Path, *, source_root: Path) -> str:
    digest = verify_sha256_manifest(Path("/"), manifest)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        resolved = (Path("/") / _safe_relative(line[66:])).resolve(strict=True)
        if not _is_within(resolved, source_root):
            raise StageError(
                f"Required-static manifest escapes Step0 source: {resolved}"
            )
    return digest


def _verify_all_input_gates(snapshot: InputSnapshot) -> dict[str, str]:
    after = {
        "primary": verify_sha256_manifest(
            snapshot.primary_root,
            snapshot.primary_manifest,
            exact_inventory=True,
        ),
        "static_project": verify_sha256_manifest(
            snapshot.static_manifest.parent,
            snapshot.static_manifest,
            exact_inventory=True,
        ),
        "required_static": _verify_required_static_manifest(
            snapshot.required_static_manifest,
            source_root=snapshot.required_static_source_root,
        ),
    }
    before = {
        "primary": snapshot.primary_digest,
        "static_project": snapshot.static_digest,
        "required_static": snapshot.required_static_digest,
    }
    if after != before:
        raise StageError(f"Input manifest digest changed during stage: {before} -> {after}")
    return after


def _input_manifest_digests(snapshot: InputSnapshot) -> dict[str, str]:
    return {
        "primary": snapshot.primary_digest,
        "static_project": snapshot.static_digest,
        "required_static": snapshot.required_static_digest,
    }


def _verify_sentinel(
    context: Context,
    path: Path,
    manifest: Path,
    fields: Mapping[str, Any],
    *,
    input_manifests: Mapping[str, str],
    allow_project_commit_transition: bool = False,
) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or path.is_symlink():
        raise StageError(f"Existing sentinel is not a physical file: {path}")
    payload = read_json(path)
    expected_inputs = dict(input_manifests)
    if set(expected_inputs) != {"primary", "static_project", "required_static"} or any(
        not SHA256.fullmatch(str(value)) for value in expected_inputs.values()
    ):
        raise StageError("Current input manifest digest mapping is incomplete")
    failures = {
        key: (expected, payload.get(key))
        for key, expected in fields.items()
        if payload.get(key) != expected
    }
    lineage_failures = {}
    if payload.get("schema_version") != "autodl_three_lines_stage_v1":
        lineage_failures["schema_version"] = (
            "autodl_three_lines_stage_v1",
            payload.get("schema_version"),
        )
    if payload.get("run_id") != RUN_ID:
        lineage_failures["run_id"] = (RUN_ID, payload.get("run_id"))
    if payload.get("input_manifest_sha256_before") != expected_inputs["primary"]:
        lineage_failures["input_manifest_sha256_before"] = (
            expected_inputs["primary"],
            payload.get("input_manifest_sha256_before"),
        )
    if payload.get("input_manifest_sha256_after") != expected_inputs["primary"]:
        lineage_failures["input_manifest_sha256_after"] = (
            expected_inputs["primary"],
            payload.get("input_manifest_sha256_after"),
        )
    if payload.get("input_manifests_sha256") != expected_inputs:
        lineage_failures["input_manifests_sha256"] = (
            expected_inputs,
            payload.get("input_manifests_sha256"),
        )
    current_code = _current_code_lineage(context)
    for key, expected in current_code.items():
        if key == "project_commit" and allow_project_commit_transition:
            recorded_commit = str(payload.get(key) or "")
            if not re.fullmatch(r"[0-9a-f]{40}", recorded_commit):
                lineage_failures[key] = ("40_hex_commit", payload.get(key))
            continue
        if payload.get(key) != expected:
            lineage_failures[key] = (expected, payload.get(key))
    failures.update(lineage_failures)
    if failures:
        raise StageError(f"Existing sentinel is invalid: {path}: {failures}")
    recorded_manifest = Path(str(payload.get("output_manifest") or ""))
    manifest_root = Path(str(payload.get("output_manifest_root") or ""))
    if (
        not recorded_manifest.is_absolute()
        or recorded_manifest.resolve(strict=False) != manifest.resolve(strict=False)
        or not manifest_root.is_absolute()
        or not manifest_root.is_dir()
        or manifest_root.is_symlink()
        or not _is_within(manifest, manifest_root)
    ):
        raise StageError(f"Existing sentinel manifest paths are unsafe: {path}")
    digest = verify_sha256_manifest(manifest_root, manifest)
    if payload.get("output_manifest_sha256") != digest:
        raise StageError(f"Existing sentinel manifest digest is stale: {path}")
    return True


def _publish_sentinel(
    *,
    context: Context,
    path: Path,
    manifest: Path,
    manifest_root: Path,
    input_digest_before: str,
    input_digest_after: str,
    payload: Mapping[str, Any],
    input_manifests: Mapping[str, str],
) -> None:
    if (
        not manifest_root.is_dir()
        or manifest_root.is_symlink()
        or not _is_within(manifest, manifest_root)
    ):
        raise StageError(f"Cannot publish sentinel for unsafe manifest root: {manifest}")
    expected_inputs = dict(input_manifests)
    if (
        set(expected_inputs) != {"primary", "static_project", "required_static"}
        or any(not SHA256.fullmatch(str(value)) for value in expected_inputs.values())
        or input_digest_before != expected_inputs["primary"]
        or input_digest_after != expected_inputs["primary"]
    ):
        raise StageError("Cannot publish a sentinel with inconsistent input lineage")
    digest = verify_sha256_manifest(manifest_root, manifest)
    code_lineage = _assert_stage_lineage_unchanged(context)
    scientific_payload = dict(payload)
    scientific_schema = scientific_payload.pop("schema_version", None)
    for key, expected in code_lineage.items():
        if key not in scientific_payload:
            continue
        if scientific_payload.pop(key) != expected:
            raise StageError(f"Scientific payload has stale code lineage field: {key}")
    reserved = {
        "run_id",
        "input_manifest_sha256_before",
        "input_manifest_sha256_after",
        "input_manifests_sha256",
        "output_manifest",
        "output_manifest_root",
        "output_manifest_sha256",
        "repair_code_closure_sha256",
        "project_commit",
        "external_comrecgc_commit",
        "external_comrecgc_tree",
        "external_provenance_sha256",
        "finished_at",
    }
    collisions = sorted(reserved & set(scientific_payload))
    if collisions:
        raise StageError(
            f"Scientific sentinel payload tries to override reserved fields: {collisions}"
        )
    if scientific_schema is not None:
        scientific_payload["scientific_payload_schema_version"] = scientific_schema
    atomic_write_json(
        path,
        {
            **scientific_payload,
            "schema_version": "autodl_three_lines_stage_v1",
            "run_id": RUN_ID,
            "input_manifest_sha256_before": input_digest_before,
            "input_manifest_sha256_after": input_digest_after,
            "input_manifests_sha256": expected_inputs,
            "output_manifest": str(manifest),
            "output_manifest_root": str(manifest_root),
            "output_manifest_sha256": digest,
            **code_lineage,
            "finished_at": utc_now(),
        },
    )


def _require_disallow_generation() -> None:
    if os.environ.get("DISALLOW_GENERATION") != "1":
        raise StageError("Freeze-only stage requires DISALLOW_GENERATION=1")


def _lineage_smoke_paths(context: Context, dataset: str) -> tuple[Path, Path, Path]:
    short = "aids" if dataset == "aids" else "mut"
    output = context.persistent / "outputs" / "smoke" / f"{short}_lineage"
    sentinel = output.parent / f"{short.upper()}_LINEAGE_SMOKE_PASS.json"
    return output, output / "MANIFEST.sha256", sentinel


def _require_lineage_smoke_gate(
    context: Context,
    dataset: str,
    *,
    input_manifests: Mapping[str, str] | None = None,
) -> None:
    output, manifest, sentinel = _lineage_smoke_paths(context, dataset)
    if not sentinel.exists():
        raise StageError(
            f"{dataset} formal freeze requires its preserved-lineage smoke gate: "
            f"{sentinel}"
        )
    if input_manifests is None:
        input_name = "aids_generation" if dataset == "aids" else "mut_generation"
        input_manifests = _input_manifest_digests(
            _all_input_gates(context, input_name)
        )
    if not _verify_sentinel(
        context,
        sentinel,
        manifest,
        {"status": "PASS", "dataset": dataset, "formal_output_written": False},
        input_manifests=input_manifests,
        allow_project_commit_transition=True,
    ):
        raise StageError(
            f"{dataset} formal freeze requires its preserved-lineage smoke gate: "
            f"{sentinel}"
        )
    payload = read_json(sentinel)
    if payload.get("repair_code_closure_sha256") != _repair_code_closure_sha256(
        context
    ):
        raise StageError(f"Lineage smoke belongs to different repair code: {sentinel}")
    if int(payload.get("recorded_action_replay_mismatch_count", -1)) != 0:
        raise StageError(f"Recorded-action smoke mismatch is non-zero: {sentinel}")
    if int(payload.get("legacy_inference_called_count", -1)) != 0:
        raise StageError(f"Lineage smoke invoked legacy inference: {sentinel}")
    if dataset == "mutagenicity" and int(
        payload.get("recorded_action_replay_ok_count", 0)
    ) <= 0:
        raise StageError(f"Mutagenicity smoke replayed no recorded actions: {sentinel}")
    if dataset == "aids":
        original_sample_count = int(
            payload.get("original_trace_roundtrip_sample_count", 0)
        )
        original_ok_count = int(
            payload.get("original_trace_roundtrip_ok_count", 0)
        )
        alias_entry_count = int(payload.get("alias_map_entry_count", -1))
        alias_sample_count = int(payload.get("alias_roundtrip_sample_count", -1))
        alias_ok_count = int(payload.get("alias_roundtrip_ok_count", -1))
        if (
            payload.get("alias_map_persisted") is not True
            or original_sample_count <= 0
            or original_ok_count != original_sample_count
            or int(payload.get("original_trace_roundtrip_mismatch_count", -1)) != 0
            or alias_entry_count < 0
            or alias_sample_count < 0
            or alias_ok_count != alias_sample_count
            or int(payload.get("alias_roundtrip_mismatch_count", -1)) != 0
            or (alias_entry_count > 0 and alias_sample_count <= 0)
        ):
            raise StageError(
                f"AIDS smoke failed direct/alias serialization round trips: {sentinel}"
            )
    del output


def _freeze_validation_parameters(
    context: Context, dataset: str
) -> dict[str, Any]:
    if dataset == "aids":
        return {
            "source_generation_dir": context.persistent / "inputs/aids_generation",
            "dataset": dataset,
            "dataset_dir": context.step0
            / "outputs/hpc/current/gcfexplainer/aids/dataset",
            "source_csv": context.static_input
            / "outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset"
            / "sft_v3_hiv_ppo_prompts_train_label1.csv",
            "expected_steps": 50_000,
            "expected_project_commit": AIDS_SOURCE_COMMIT,
        }
    return {
        "source_generation_dir": context.persistent / "inputs/mut_generation",
        "dataset": dataset,
        "dataset_dir": context.step0
        / "outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset",
        "source_csv": None,
        "expected_steps": 50_000,
        "expected_project_commit": MUT_SOURCE_COMMIT,
    }


def _recorded_action_sample(
    payload: Mapping[str, Any], trace_manifest: Path, *, limit: int = 64
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay a deterministic recorded-action sample through the real iterator."""

    from src.baselines.comrecgc.graph_trace import (
        iter_candidate_lineage_from_selected_trace,
        iter_selected_trace,
    )

    selected: list[dict[str, Any]] = []
    candidate_hashes: list[str] = []
    for raw in iter_selected_trace(trace_manifest):
        if raw.get("event") != "selected_transition" or raw.get("action") is None:
            continue
        row = dict(raw)
        selected.append(row)
        target = str(row["target_official_hash"])
        if target not in candidate_hashes:
            candidate_hashes.append(target)
        if len(selected) >= limit:
            break
    if not selected:
        raise StageError("Preserved selected trace contains no recorded-action event")
    sample_payload = dict(payload)
    sample_payload["counterfactual_candidates"] = [
        {"graph_hash": graph_hash} for graph_hash in candidate_hashes
    ]
    audit: dict[str, Any] = {}
    lineage = list(
        iter_candidate_lineage_from_selected_trace(
            sample_payload,
            iter(selected),
            include_actions=False,
            recovery_audit=audit,
        )
    )
    if int(audit.get("recorded_action_replay_ok_count", 0)) != len(selected):
        raise StageError(f"Recorded-action sample replay was incomplete: {audit}")
    if int(audit.get("recorded_action_replay_mismatch_count", -1)) != 0:
        raise StageError(f"Recorded-action sample replay mismatch: {audit}")
    if int(audit.get("legacy_inference_called_count", -1)) != 0:
        raise StageError(f"Recorded-action sample used legacy inference: {audit}")
    return lineage, audit


def _aids_alias_roundtrip(
    payload: Mapping[str, Any], output: Path, *, limit: int = 64
) -> dict[str, Any]:
    """Verify direct and aliased original hashes survive torch serialization.

    An empty alias map is a valid closure: an original trace hash may already
    be the physical canonical-record key.  Conversely, when aliases are
    present, they remain part of the closure contract and are sampled here as
    well.  Never manufacture an alias merely to make this smoke non-empty.
    """

    from src.baselines.comrecgc.frozen_payload import (
        atomic_torch_save,
        payload_graphs_by_official_hash,
        torch_load_payload,
    )
    from src.baselines.comrecgc.graph_trace import normalized_untyped_graph_payload

    raw_aliases = payload.get("alias_to_canonical")
    if not isinstance(raw_aliases, Mapping):
        raise StageError("AIDS preserved payload has no typed alias mapping")
    raw_original_trace_hashes = payload.get("original_trace_hashes")
    if not isinstance(raw_original_trace_hashes, list):
        raise StageError(
            "AIDS preserved payload has no typed original trace hash list"
        )
    aliases = {
        str(alias): str(canonical)
        for alias, canonical in raw_aliases.items()
    }
    original_trace_hashes = [str(value) for value in raw_original_trace_hashes]
    if not original_trace_hashes:
        raise StageError("AIDS preserved payload exposes no original trace hashes")

    def resolve_alias(graph_hash: str) -> str:
        current = str(graph_hash)
        seen: set[str] = set()
        while current in aliases:
            if current in seen:
                raise StageError(f"AIDS alias cycle reached during smoke: {graph_hash}")
            seen.add(current)
            current = aliases[current]
        return current

    selected_originals = sorted(set(original_trace_hashes))[:limit]
    selected_aliases = sorted(aliases.items())[:limit]
    graphs = payload_graphs_by_official_hash(payload)
    canonical_graphs: dict[str, list[Any]] = {}
    before: dict[str, dict[str, Any]] = {}
    sample_aliases: dict[str, str] = {}
    sample_keys: list[tuple[str, str, str]] = []
    for original in selected_originals:
        canonical = resolve_alias(original)
        original_graph = graphs.get(original)
        canonical_graph = graphs.get(canonical)
        if original_graph is None or canonical_graph is None:
            raise StageError(
                f"AIDS original trace hash is absent: {original} -> {canonical}"
            )
        original_payload = normalized_untyped_graph_payload(original_graph)
        canonical_payload = normalized_untyped_graph_payload(canonical_graph)
        if original_payload != canonical_payload:
            raise StageError(
                f"AIDS original/canonical graph differs: {original} -> {canonical}"
            )
        canonical_graphs[canonical] = [canonical_graph]
        before[original] = original_payload
        if original != canonical:
            sample_aliases[original] = canonical
            sample_keys.append((original, canonical, "alias"))
        else:
            sample_keys.append((original, canonical, "direct"))
    # Also prove that a persisted alias not selected by the original-hash
    # sample remains resolvable with identical normalized graph content after
    # serialization.
    for alias, _declared_canonical in selected_aliases:
        canonical = resolve_alias(alias)
        alias_graph = graphs.get(alias)
        canonical_graph = graphs.get(canonical)
        if alias_graph is None or canonical_graph is None:
            raise StageError(f"AIDS alias target is absent: {alias} -> {canonical}")
        alias_payload = normalized_untyped_graph_payload(alias_graph)
        canonical_payload = normalized_untyped_graph_payload(canonical_graph)
        if alias_payload != canonical_payload:
            raise StageError(
                f"AIDS original/canonical graph differs: {alias} -> {canonical}"
            )
        canonical_graphs[canonical] = [canonical_graph]
        before[alias] = alias_payload
        sample_aliases[alias] = canonical
        if (alias, canonical, "alias") not in sample_keys:
            sample_keys.append((alias, canonical, "alias"))
    sample_payload = {
        "schema_version": "comrecgc_alias_roundtrip_smoke_v1",
        "dataset": "aids",
        "graph_map": canonical_graphs,
        "alias_to_canonical": sample_aliases,
        "original_trace_hashes": selected_originals,
        "counterfactual_candidates": [],
    }
    serialized = output / "alias_roundtrip_sample.pt"
    atomic_torch_save(sample_payload, serialized)
    reloaded = torch_load_payload(serialized)
    reloaded_aliases = reloaded.get("alias_to_canonical")
    if not isinstance(reloaded_aliases, Mapping):
        raise StageError("AIDS serialized sample lost alias_to_canonical mapping")
    if {
        str(alias): str(canonical)
        for alias, canonical in reloaded_aliases.items()
    } != sample_aliases:
        raise StageError("AIDS serialized sample changed alias_to_canonical mapping")
    if [str(value) for value in (reloaded.get("original_trace_hashes") or [])] != (
        selected_originals
    ):
        raise StageError("AIDS serialized sample changed original trace hashes")
    reloaded_graphs = payload_graphs_by_official_hash(reloaded)
    direct_verified = 0
    alias_verified = 0
    for original, canonical, resolution in sample_keys:
        if original not in reloaded_graphs or canonical not in reloaded_graphs:
            raise StageError(
                f"AIDS serialization round trip lost {original} -> {canonical}"
            )
        original_payload = normalized_untyped_graph_payload(
            reloaded_graphs[original]
        )
        canonical_payload = normalized_untyped_graph_payload(
            reloaded_graphs[canonical]
        )
        if (
            original_payload != canonical_payload
            or original_payload != before[original]
        ):
            raise StageError(
                f"AIDS original-hash serialization round trip changed {original}"
            )
        if resolution == "alias":
            alias_verified += 1
        else:
            direct_verified += 1
    return {
        "alias_map_persisted": True,
        "alias_map_entry_count": len(aliases),
        "original_trace_roundtrip_sample_count": len(selected_originals),
        "original_trace_roundtrip_ok_count": len(selected_originals),
        "original_trace_roundtrip_mismatch_count": 0,
        "direct_roundtrip_sample_count": direct_verified,
        "direct_roundtrip_ok_count": direct_verified,
        "alias_roundtrip_sample_count": alias_verified,
        "alias_roundtrip_ok_count": alias_verified,
        "alias_roundtrip_mismatch_count": 0,
        "serialized_sample": serialized.name,
        "serialized_sample_sha256": sha256_file(serialized),
    }


def _run_lineage_smoke(context: Context, dataset: str) -> None:
    """Validate closure, recorded actions, and AIDS aliases outside formal outputs."""

    _require_disallow_generation()
    input_name = "aids_generation" if dataset == "aids" else "mut_generation"
    inputs = _all_input_gates(context, input_name)
    expected_inputs = _input_manifest_digests(inputs)
    output, manifest, sentinel = _lineage_smoke_paths(context, dataset)
    expected = {"status": "PASS", "dataset": dataset, "formal_output_written": False}
    if _verify_sentinel(
        context,
        sentinel,
        manifest,
        expected,
        input_manifests=expected_inputs,
        allow_project_commit_transition=True,
    ):
        _require_lineage_smoke_gate(
            context, dataset, input_manifests=expected_inputs
        )
        _verify_all_input_gates(inputs)
        return
    _assert_empty_or_absent(output, label=f"{dataset} lineage smoke output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    published = False
    try:
        closure_audit = temporary / "closure_validate_only_audit.json"
        validate_command = _recovery_command(
            context,
            dataset=dataset,
            audit=closure_audit,
            output=temporary / "unused_formal_output",
            validate=True,
        )
        _run_checked(
            validate_command,
            cwd=context.project,
            env_extra={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
            if dataset == "aids"
            else None,
        )
        closure = read_json(closure_audit)
        if closure.get("FREEZE_ONLY_RECOVERY_SAFE") is not True:
            raise StageError(f"{dataset} full validate-only closure gate failed")

        # The validate-only CLI intentionally discards its closed payload.  Run
        # the same validator in-process once more so the smoke can exercise the
        # actual recorded-action iterator without writing a formal generation.
        if str(context.project) not in sys.path:
            sys.path.insert(0, str(context.project))
        from src.baselines.comrecgc.freeze_recovery import (
            validate_completed_generation_freeze,
        )

        direct_audit, closed_payload = validate_completed_generation_freeze(
            **_freeze_validation_parameters(context, dataset)
        )
        if (
            direct_audit.get("FREEZE_ONLY_RECOVERY_SAFE") is not True
            or closed_payload is None
        ):
            raise StageError(f"{dataset} closed payload was unavailable after validation")
        source = Path(
            str(
                _freeze_validation_parameters(context, dataset)[
                    "source_generation_dir"
                ]
            )
        )
        lineage, replay = _recorded_action_sample(
            closed_payload,
            source / "trace/selected_action_trace_manifest.json",
        )
        alias = (
            _aids_alias_roundtrip(closed_payload, temporary)
            if dataset == "aids"
            else {
                "alias_roundtrip_sample_count": 0,
                "alias_roundtrip_ok_count": 0,
                "alias_roundtrip_mismatch_count": 0,
            }
        )
        report = {
            "schema_version": "autodl_preserved_lineage_smoke_v1",
            "status": "PASS",
            "dataset": dataset,
            "formal_output_written": False,
            "repair_code_closure_sha256": _repair_code_closure_sha256(context),
            "closure_validate_only_passed": True,
            "closure_audit_sha256": sha256_file(closure_audit),
            "sample_policy": "first_64_recorded_actions_in_selected_trace_order",
            "sample_candidate_count": len(lineage),
            **{key: int(value) for key, value in replay.items() if key.endswith("_count")},
            **alias,
            "completed_at": utc_now(),
        }
        atomic_write_json(temporary / "lineage_smoke_report.json", report)
        temporary_manifest = temporary / manifest.name
        write_sha256_manifest(
            base=temporary,
            items=[temporary],
            manifest=temporary_manifest,
            exclude=[temporary_manifest],
        )
        os.rename(temporary, output)
        published = True
        _fsync_directory(output.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    input_manifests = _verify_all_input_gates(inputs)
    report = read_json(output / "lineage_smoke_report.json")
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=output,
        input_digest_before=inputs.primary_digest,
        input_digest_after=input_manifests["primary"],
        input_manifests=input_manifests,
        payload=report,
    )
    _require_lineage_smoke_gate(
        context, dataset, input_manifests=input_manifests
    )


def _complete_marker(path: Path, *, field: str = "run_complete") -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    return read_json(path).get(field) is True


def _marker_has_fields(path: Path, fields: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or path.is_symlink():
        raise StageError(f"Scientific completion marker is not physical: {path}")
    payload = read_json(path)
    failures = {
        key: (expected, payload.get(key))
        for key, expected in fields.items()
        if payload.get(key) != expected
    }
    if failures:
        raise StageError(f"Scientific completion marker failed: {path}: {failures}")
    return True


def _canonical_scientific_command(command: Sequence[str]) -> list[str]:
    # ``--resume`` changes only the launch path.  All scientific options and
    # input/output roots remain byte-for-byte bound by this digest.
    return [str(value) for value in command if str(value) != "--resume"]


def _scientific_reuse_lineage(
    *,
    context: Context,
    command: Sequence[str],
    input_manifests: Mapping[str, str],
    scientific_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = {
        str(key): str(value)
        for key, value in (scientific_environment or {}).items()
    }
    _assert_no_secret_environment(environment)
    return {
        "run_id": RUN_ID,
        "command_sha256": _stable_digest(
            _canonical_scientific_command(command)
        ),
        "input_manifests_sha256": dict(input_manifests),
        "scientific_environment_sha256": _stable_digest(environment),
        **_current_code_lineage(context),
    }


def _reuse_proof_paths(output: Path) -> tuple[Path, Path]:
    return (
        output / "_AUTODL_REUSE_MANIFEST.sha256",
        output / "_AUTODL_REUSE_PROOF.json",
    )


def _verify_reuse_proof(
    *,
    context: Context,
    output: Path,
    marker: Path,
    marker_fields: Mapping[str, Any],
    command: Sequence[str],
    input_manifests: Mapping[str, str],
    scientific_environment: Mapping[str, str] | None = None,
) -> None:
    if output.is_symlink() or not output.is_dir() or not _is_within(marker, output):
        raise StageError(f"Scientific reuse output/marker paths are unsafe: {output}")
    if not _marker_has_fields(marker, marker_fields):
        raise StageError(f"Scientific output lacks its completion marker: {marker}")
    manifest, proof_path = _reuse_proof_paths(output)
    _assert_nonempty_file(proof_path)
    digest = verify_sha256_manifest(
        output,
        manifest,
        exact_inventory=True,
        allowed_unlisted=[proof_path],
    )
    proof = read_json(proof_path)
    expected = {
        "schema_version": "autodl_scientific_reuse_proof_v1",
        "output_root": str(output.resolve(strict=True)),
        "marker": str(marker.relative_to(output)),
        "marker_sha256": sha256_file(marker),
        "marker_fields": dict(marker_fields),
        "output_manifest": str(manifest),
        "output_manifest_sha256": digest,
        **_scientific_reuse_lineage(
            context=context,
            command=command,
            input_manifests=input_manifests,
            scientific_environment=scientific_environment,
        ),
    }
    failures = {
        key: (value, proof.get(key))
        for key, value in expected.items()
        if proof.get(key) != value
    }
    if failures:
        raise StageError(f"Scientific reuse proof is stale: {proof_path}: {failures}")


def _publish_reuse_proof(
    *,
    context: Context,
    output: Path,
    marker: Path,
    marker_fields: Mapping[str, Any],
    command: Sequence[str],
    input_manifests: Mapping[str, str],
    scientific_environment: Mapping[str, str] | None = None,
) -> None:
    if output.is_symlink() or not output.is_dir() or not _is_within(marker, output):
        raise StageError(f"Scientific reuse output/marker paths are unsafe: {output}")
    if not _marker_has_fields(marker, marker_fields):
        raise StageError(f"Scientific command omitted completion marker: {marker}")
    manifest, proof_path = _reuse_proof_paths(output)
    write_sha256_manifest(
        base=output,
        items=[output],
        manifest=manifest,
        exclude=[manifest, proof_path],
    )
    _assert_stage_lineage_unchanged(context)
    lineage = _scientific_reuse_lineage(
        context=context,
        command=command,
        input_manifests=input_manifests,
        scientific_environment=scientific_environment,
    )
    atomic_write_json(
        proof_path,
        {
            "schema_version": "autodl_scientific_reuse_proof_v1",
            "output_root": str(output.resolve(strict=True)),
            "marker": str(marker.relative_to(output)),
            "marker_sha256": sha256_file(marker),
            "marker_fields": dict(marker_fields),
            "output_manifest": str(manifest),
            "output_manifest_sha256": sha256_file(manifest),
            **lineage,
            "published_at": utc_now(),
        },
    )
    _verify_reuse_proof(
        context=context,
        output=output,
        marker=marker,
        marker_fields=marker_fields,
        command=command,
        input_manifests=input_manifests,
        scientific_environment=scientific_environment,
    )


def _run_or_reuse(
    *,
    context: Context,
    output: Path,
    marker: Path,
    command: list[str],
    resumable: bool,
    input_manifests: Mapping[str, str],
    marker_fields: Mapping[str, Any] | None = None,
    env_extra: Mapping[str, str] | None = None,
) -> None:
    expected_marker = dict(marker_fields or {"run_complete": True})
    if _marker_has_fields(marker, expected_marker):
        _verify_reuse_proof(
            context=context,
            output=output,
            marker=marker,
            marker_fields=expected_marker,
            command=command,
            input_manifests=input_manifests,
            scientific_environment=env_extra,
        )
        return
    nonempty = output.exists() and (not output.is_dir() or any(output.iterdir()))
    if nonempty and not (context.resume and resumable):
        raise StageError(f"Partial non-resumable output exists: {output}")
    if nonempty and context.resume and resumable and "--resume" not in command:
        command = [*command, "--resume"]
    _run_checked(command, cwd=context.project, env_extra=env_extra)
    _publish_reuse_proof(
        context=context,
        output=output,
        marker=marker,
        marker_fields=expected_marker,
        command=command,
        input_manifests=input_manifests,
        scientific_environment=env_extra,
    )


def _recovery_command(
    context: Context, *, dataset: str, audit: Path, output: Path, validate: bool
) -> list[str]:
    source = (
        context.persistent
        / "inputs"
        / f"{dataset if dataset == 'aids' else 'mut'}_generation"
    )
    if dataset == "aids":
        dataset_dir = context.step0 / "outputs/hpc/current/gcfexplainer/aids/dataset"
        source_csv = (
            context.static_input
            / "outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset"
            / "sft_v3_hiv_ppo_prompts_train_label1.csv"
        )
        expected_commit = AIDS_SOURCE_COMMIT
    else:
        dataset_dir = (
            context.step0
            / "outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset"
        )
        source_csv = None
        expected_commit = MUT_SOURCE_COMMIT
    command = [
        str(context.python),
        context.script("scripts/baselines/comrecgc/recover_completed_generation_freeze.py"),
        "--source-generation-dir",
        str(source),
        "--dataset",
        dataset,
        "--dataset-dir",
        str(dataset_dir),
        "--audit-output",
        str(audit),
        "--expected-steps",
        "50000",
        "--expected-project-commit",
        expected_commit,
    ]
    if source_csv is not None:
        command += ["--source-csv", str(source_csv)]
    command += ["--validate-only"] if validate else ["--output-dir", str(output)]
    return command


def _validate_recovered_generation_exact_gate(
    *,
    dataset: str,
    generation: Path,
    source_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume the complete MUT/AID recovery evidence required by the run."""

    trace_manifest_path = generation / "trace/selected_action_trace_manifest.json"
    trace_manifest = read_json(trace_manifest_path)
    chunks = trace_manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise StageError("Recovered generation has no selected-trace chunks")
    row_count = 0
    chunk_paths: set[Path] = set()
    for expected_index, row in enumerate(chunks):
        if not isinstance(row, Mapping) or int(row.get("index", -1)) != expected_index:
            raise StageError("Recovered selected-trace chunk order changed")
        relative = _safe_relative(str(row.get("path") or ""))
        if relative in chunk_paths:
            raise StageError(f"Recovered selected-trace chunk path repeats: {relative}")
        chunk_paths.add(relative)
        chunk = trace_manifest_path.parent / relative
        _assert_nonempty_file(chunk)
        if (
            int(row.get("bytes", -1)) != chunk.stat().st_size
            or row.get("sha256") != sha256_file(chunk)
        ):
            raise StageError(f"Recovered selected-trace chunk digest changed: {chunk}")
        row_count += int(row.get("row_count", -1))
    if row_count != int(trace_manifest.get("row_count", -1)):
        raise StageError("Recovered selected-trace row total is inconsistent")

    database = generation / "graph_state/authoritative_graph_store.sqlite3"
    _assert_nonempty_file(database)
    for suffix in ("-wal", "-shm"):
        if Path(f"{database}{suffix}").exists():
            raise StageError(f"Recovered immutable SQLite has a sidecar: {database}{suffix}")
    connection = sqlite3.connect(
        f"{database.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    try:
        sqlite_integrity = str(
            connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        sqlite_entries = int(
            connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0]
        )
    finally:
        connection.close()
    if sqlite_integrity != "ok":
        raise StageError("Recovered immutable SQLite integrity_check is not ok")

    run_manifest = read_json(generation / "run_manifest.json")
    closure = read_json(generation / "frozen_payload_closure_audit.json")
    trace_summary = read_json(generation / "trace/trace_summary.json")
    counters = {
        name: int(trace_summary.get(name, -1))
        for name in (
            "selected_transition_count",
            "recorded_action_present_count",
            "recorded_action_replay_ok_count",
            "recorded_action_replay_mismatch_count",
            "legacy_missing_action_count",
            "legacy_inference_called_count",
            "legacy_inference_ambiguous_count",
        )
    }
    common_failures = {
        "completed_steps": int(source_validation.get("completed_steps", -1)) != 50_000,
        "closure_complete": closure.get("closure_complete") is not True,
        "post_write_reload_verified": closure.get("post_write_reload_verified") is not True,
        "roundtrip_verified": closure.get("original_trace_hash_roundtrip_verified") is not True,
        "unresolved_hash_count": int(closure.get("unresolved_hash_count", -1)) != 0,
        "graph_replacement_count": int(closure.get("graph_replacement_count", -1)) != 0,
        "trace_summary_algorithm_rerun": trace_summary.get("algorithm_rerun") is not False,
        "run_manifest_algorithm_rerun": run_manifest.get("algorithm_rerun") is not False,
    }
    common_failures = {key: value for key, value in common_failures.items() if value}
    if common_failures:
        raise StageError(f"Recovered generation common exact gate failed: {common_failures}")

    evidence: dict[str, Any] = {
        "sqlite_integrity_check": sqlite_integrity,
        "sqlite_entry_count": sqlite_entries,
        "trace_chunk_count": len(chunks),
        "trace_row_count": row_count,
        "closure_complete": True,
        "unresolved_hash_count": 0,
        "graph_replacement_count": 0,
        **counters,
    }
    source_backing = source_validation.get("backing_store_audit") or {}
    source_trace = source_validation.get("selected_trace_audit") or {}
    source_counts = {
        "sqlite_entry_count": int(source_backing.get("entry_count", -1)),
        "trace_chunk_count": int(source_trace.get("chunk_count", -1)),
        "trace_row_count": int(source_trace.get("row_count", -1)),
    }
    if source_counts != {
        "sqlite_entry_count": sqlite_entries,
        "trace_chunk_count": len(chunks),
        "trace_row_count": row_count,
    }:
        raise StageError(
            "Recovered generation changed source SQLite/trace counts: "
            f"source={source_counts}, recovered="
            f"{dict(sqlite_entry_count=sqlite_entries, trace_chunk_count=len(chunks), trace_row_count=row_count)}"
        )
    evidence["source_count_alignment"] = source_counts
    if dataset == "mutagenicity":
        # ``counterfactual_candidate_count`` is the frozen payload's unique
        # candidate population.  A candidate can be selected by more than one
        # recorded walk transition, so it must not be conflated with the
        # selected-transition/replay multiplicity below.
        expected = {
            "sqlite_entry_count": 124_206,
            "trace_chunk_count": 449,
            "trace_row_count": 229_752,
            "candidate_count": 100_235,
            "selected_transition_count": 224_690,
            "recorded_action_present_count": 224_690,
            "recorded_action_replay_ok_count": 224_690,
            "recorded_action_replay_mismatch_count": 0,
            "legacy_missing_action_count": 0,
            "legacy_inference_called_count": 0,
            "legacy_inference_ambiguous_count": 0,
        }
        evidence["candidate_count"] = int(run_manifest.get("counterfactual_candidate_count", -1))
        failures = {
            key: (value, evidence.get(key))
            for key, value in expected.items()
            if evidence.get(key) != value
        }
        if failures:
            raise StageError(f"Mutagenicity formal recovery counts changed: {failures}")
    elif dataset == "aids":
        alias_count = closure.get("alias_count")
        canonical_graph_record_count = closure.get("canonical_graph_record_count")
        original_trace_hash_count = closure.get("original_trace_hash_count")
        original_trace_hash_roundtrip_count = closure.get(
            "original_trace_hash_roundtrip_count"
        )
        aids_failures = {
            "canonical_graph_records": (
                not isinstance(canonical_graph_record_count, int)
                or isinstance(canonical_graph_record_count, bool)
                or canonical_graph_record_count <= 0
            ),
            "canonical_graph_records_persisted": closure.get(
                "canonical_graph_records_persisted"
            )
            is not True,
            "alias_to_canonical_persisted": closure.get(
                "alias_to_canonical_persisted"
            )
            is not True,
            "alias_count": (
                not isinstance(alias_count, int)
                or isinstance(alias_count, bool)
                or alias_count < 0
            ),
            "original_trace_hashes": (
                not isinstance(original_trace_hash_count, int)
                or isinstance(original_trace_hash_count, bool)
                or original_trace_hash_count <= 0
            ),
            "original_trace_hashes_persisted": closure.get(
                "original_trace_hashes_persisted"
            )
            is not True,
            "original_trace_hash_roundtrip_count": (
                not isinstance(original_trace_hash_roundtrip_count, int)
                or isinstance(original_trace_hash_roundtrip_count, bool)
                or original_trace_hash_roundtrip_count != original_trace_hash_count
            ),
            "alias_cycle_count": int(closure.get("alias_cycle_count", -1)) != 0,
            "dangling_alias_count": int(closure.get("dangling_alias_count", -1)) != 0,
            "trace_row_preservation": int(
                closure.get("selected_trace_row_count", -1)
            )
            != row_count,
        }
        aids_failures = {key: value for key, value in aids_failures.items() if value}
        if aids_failures:
            raise StageError(f"AIDS formal frozen-closure gate failed: {aids_failures}")
        evidence.update(
            {
                "canonical_graph_record_count": int(
                    canonical_graph_record_count
                ),
                "alias_count": alias_count,
                "alias_to_canonical_persisted": True,
                "original_trace_hash_count": int(original_trace_hash_count),
                "original_trace_hash_roundtrip_count": int(
                    original_trace_hash_roundtrip_count
                ),
                "alias_cycle_count": 0,
                "dangling_alias_count": 0,
            }
        )
    else:
        raise StageError(f"Unsupported recovered generation dataset: {dataset}")
    return evidence


def _validate_bace_generation_exact_gate(
    *,
    context: Context,
    generation: Path,
    mirror: Path,
    metadata: Path,
) -> dict[str, Any]:
    """Validate the complete 50k BACE generation and its durable resume chain."""

    progress = read_json(generation / "progress.json")
    run_manifest = read_json(generation / "run_manifest.json")
    resolved_config = read_json(generation / "resolved_config.json")
    closure = read_json(generation / "frozen_payload_closure_audit.json")
    trace_manifest_path = generation / "trace/selected_action_trace_manifest.json"
    trace_manifest = read_json(trace_manifest_path)
    trace_complete = read_json(generation / "trace/_TRACE_COMPLETE.json")
    chunks = trace_manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise StageError("BACE final selected-trace manifest has no chunks")
    trace_rows = 0
    seen_chunk_paths: set[Path] = set()
    for expected_index, row in enumerate(chunks):
        if not isinstance(row, Mapping) or int(row.get("index", -1)) != expected_index:
            raise StageError("BACE final selected-trace chunk order changed")
        relative = _safe_relative(str(row.get("path") or ""))
        if relative in seen_chunk_paths:
            raise StageError(f"BACE final selected-trace chunk repeats: {relative}")
        seen_chunk_paths.add(relative)
        chunk = trace_manifest_path.parent / relative
        _assert_nonempty_file(chunk)
        if (
            chunk.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(chunk) != str(row.get("sha256") or "")
        ):
            raise StageError(f"BACE final selected-trace chunk changed: {chunk}")
        rows = int(row.get("row_count", -1))
        if rows < 0:
            raise StageError(f"BACE final selected-trace row count is invalid: {chunk}")
        trace_rows += rows
    if trace_rows != int(trace_manifest.get("row_count", -1)):
        raise StageError("BACE final selected-trace row total is inconsistent")
    trace_manifest_sha256 = sha256_file(trace_manifest_path)
    if (
        trace_complete.get("trace_complete") is not True
        or trace_complete.get("selected_trace_manifest_sha256")
        != trace_manifest_sha256
    ):
        raise StageError("BACE final trace completion proof does not match its manifest")

    checkpoint_interval = int(
        resolved_config.get("generation_checkpoint_interval_steps", -1)
    )
    progress_failures = {
        "progress_run_complete": progress.get("run_complete") is not True,
        "progress_current_step": int(progress.get("current_step", -1)) != 50_000,
        "progress_completed_step": int(progress.get("completed_step", -1)) != 50_000,
        "progress_next_step": int(progress.get("next_step", -1)) != 50_001,
        "progress_total_steps": int(progress.get("total_steps", -1)) != 50_000,
        "progress_last_checkpoint": int(progress.get("last_checkpoint_step", -1))
        != 50_000,
        "run_complete": run_manifest.get("run_complete") is not True,
        "algorithm_rerun": run_manifest.get("algorithm_rerun") is not True,
        "traversed_step_count": int(run_manifest.get("traversed_step_count", -1))
        != 50_000,
        "checkpoint_interval": checkpoint_interval != 500,
        "manifest_checkpoint_interval": int(
            run_manifest.get("generation_checkpoint_interval_steps", -1)
        )
        != 500,
        "closure_complete": closure.get("closure_complete") is not True,
        "closure_reload": closure.get("post_write_reload_verified") is not True,
        "closure_unresolved": int(closure.get("unresolved_hash_count", -1)) != 0,
    }
    progress_failures = {
        key: value for key, value in progress_failures.items() if value
    }
    if progress_failures:
        raise StageError(f"BACE generation exact final gate failed: {progress_failures}")

    module = _checkpoint_module(context)
    live_checkpoints = module.list_generation_checkpoints(mirror)
    live_validations = [
        _validate_mirrored_checkpoint(module, checkpoint)
        for checkpoint in live_checkpoints
    ]
    if [int(value.completed_step) for value in live_validations] != [49_500, 50_000]:
        raise StageError(
            "BACE persistent checkpoint mirror does not retain exact latest-2"
        )
    for validation in live_validations:
        # This repeats the complete manifest/state/SQLite validation with the
        # final target bound explicitly, rather than trusting marker fields.
        rebound = module.validate_generation_checkpoint(
            validation.checkpoint_dir,
            expected_total_steps=50_000,
            expected_completed_step=int(validation.completed_step),
        )
        if rebound.checkpoint_digest != validation.checkpoint_digest:
            raise StageError("BACE mirrored checkpoint digest changed during final Gate")
    latest = module.validate_generation_checkpoint(
        mirror,
        expected_total_steps=50_000,
        expected_completed_step=50_000,
    )
    if latest.checkpoint_digest != live_validations[-1].checkpoint_digest:
        raise StageError("BACE persistent checkpoint LATEST digest changed")

    history_root = mirror / str(module.RETENTION_HISTORY_DIRNAME)
    if history_root.is_symlink() or not history_root.is_dir():
        raise StageError("BACE checkpoint retention history is missing")
    historical: list[dict[str, Any]] = []
    for path in sorted(history_root.iterdir(), key=lambda value: value.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise StageError(f"BACE checkpoint retention entry is unsafe: {path}")
        row = read_json(path)
        step = int(row.get("completed_step", -1))
        expected_name = f"step-{step:012d}.json"
        if (
            row.get("schema_version")
            != "comrecgc_generation_checkpoint_retention_v1"
            or row.get("checkpoint_mirrored") is not True
            or path.name != expected_name
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("checkpoint_digest") or ""))
            is None
        ):
            raise StageError(f"BACE checkpoint retention proof is invalid: {path}")
        historical.append(row)
    all_steps = [int(row["completed_step"]) for row in historical] + [
        int(value.completed_step) for value in live_validations
    ]
    expected_steps = list(range(500, 50_001, 500))
    duplicate_step_count = len(all_steps) - len(set(all_steps))
    skipped_steps = sorted(set(expected_steps) - set(all_steps))
    unexpected_steps = sorted(set(all_steps) - set(expected_steps))
    if (
        all_steps != expected_steps
        or duplicate_step_count != 0
        or skipped_steps
        or unexpected_steps
    ):
        raise StageError(
            "BACE checkpoint sequence is not contiguous: "
            f"duplicates={duplicate_step_count}, skipped={skipped_steps[:8]}, "
            f"unexpected={unexpected_steps[:8]}"
        )
    _assert_nonempty_file(metadata / "resolved_config.json")
    if sha256_file(metadata / "resolved_config.json") != sha256_file(
        generation / "resolved_config.json"
    ):
        raise StageError("BACE persistent resume config differs from final generation")

    return {
        "fresh_start_step": 0,
        "imported_old_partial_state": False,
        "completed_step": 50_000,
        "checkpoint_interval_steps": checkpoint_interval,
        "published_checkpoint_count": len(all_steps),
        "retained_checkpoint_steps": [49_500, 50_000],
        "retention_history_count": len(historical),
        "latest_checkpoint_digest": latest.checkpoint_digest,
        "latest_checkpoint_sqlite_integrity": "ok",
        "all_published_checkpoint_manifests_validated_before_retention": True,
        "duplicate_step_count": duplicate_step_count,
        "skipped_step_count": len(skipped_steps),
        "trace_chunk_count": len(chunks),
        "trace_row_count": trace_rows,
        "trace_manifest_sha256": trace_manifest_sha256,
        "trace_manifest_verified": True,
        "closure_complete": True,
        "unresolved_hash_count": 0,
        "traversed_step_count": 50_000,
    }


def _run_freeze(context: Context, dataset: str) -> None:
    _require_disallow_generation()
    _require_lineage_smoke_gate(context, dataset)
    input_name = "aids_generation" if dataset == "aids" else "mut_generation"
    inputs = _all_input_gates(context, input_name)
    expected_inputs = _input_manifest_digests(inputs)
    input_before = inputs.primary_digest
    output_root = (
        context.persistent
        / "outputs"
        / f"{dataset if dataset == 'aids' else 'mut'}_comrecgc"
    )
    generation = output_root / "generation"
    audit = output_root / "freeze_recovery_audit.json"
    sentinel = output_root / (
        "AIDS_FREEZE_RECOVERY_PASS.json"
        if dataset == "aids"
        else "MUT_FREEZE_RECOVERY_PASS.json"
    )
    manifest = output_root / "manifests" / f"{dataset}_freeze_recovery.sha256"
    fields = {"status": "PASS", "generation_rerun_performed": False}
    if _verify_sentinel(
        context, sentinel, manifest, fields, input_manifests=expected_inputs
    ):
        return
    output_root.mkdir(parents=True, exist_ok=True)
    env = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"} if dataset == "aids" else None
    validate_command = _recovery_command(
        context, dataset=dataset, audit=audit, output=generation, validate=True
    )
    _assert_no_generation(validate_command)
    _run_checked(validate_command, cwd=context.project, env_extra=env)
    validation = read_json(audit)
    if validation.get("FREEZE_ONLY_RECOVERY_SAFE") is not True:
        raise StageError(f"Freeze-only source audit did not pass: {audit}")
    recovery_command = _recovery_command(
        context, dataset=dataset, audit=audit, output=generation, validate=False
    )
    _assert_no_generation(recovery_command)
    _run_or_reuse(
        context=context,
        output=generation,
        marker=generation / "_RUN_COMPLETE.json",
        marker_fields={"run_complete": True, "freeze_only_recovery": True},
        command=recovery_command,
        resumable=False,
        input_manifests=expected_inputs,
        env_extra=env,
    )
    manifest_payload = read_json(generation / "run_manifest.json")
    if manifest_payload.get("algorithm_rerun") is not False:
        raise StageError("Recovered generation claims an algorithm rerun")
    complete = read_json(generation / "_RUN_COMPLETE.json")
    closure = read_json(generation / "frozen_payload_closure_audit.json")
    recovery = read_json(generation / "freeze_only_recovery.json")
    if (
        complete.get("freeze_only_recovery") is not True
        or recovery.get("algorithm_rerun") is not False
        or recovery.get("recovery_completed") is not True
        or int(recovery.get("completed_steps", -1)) != 50_000
        or closure.get("closure_complete") is not True
        or closure.get("post_write_reload_verified") is not True
    ):
        raise StageError(f"Recovered {dataset} generation failed its exact freeze gate")
    exact_gate = _validate_recovered_generation_exact_gate(
        dataset=dataset,
        generation=generation,
        source_validation=validation,
    )
    write_sha256_manifest(
        base=output_root,
        items=[generation, audit],
        manifest=manifest,
        exclude=[manifest, sentinel],
    )
    input_manifests = _verify_all_input_gates(inputs)
    input_after = input_manifests["primary"]
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=output_root,
        input_digest_before=input_before,
        input_digest_after=input_after,
        input_manifests=input_manifests,
        payload={
            **fields,
            "dataset": dataset,
            "completed_step": 50_000,
            "trace_persistence": "preserved_freeze_only",
            "exact_recovery_gate": exact_gate,
        },
    )


def _common_recourse_command(context: Context, dataset: str, base: Path) -> list[str]:
    if dataset == "aids":
        dataset_dir = context.step0 / "outputs/hpc/current/gcfexplainer/aids/dataset"
        distance = context.static_input / "outputs/hpc/greed_hiv/checkpoints/best_greed_hiv_ged.pt"
        source_args = [
            "--source-csv",
            str(
                context.static_input
                / "outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset"
                / "sft_v3_hiv_ppo_prompts_train_label1.csv"
            ),
        ]
        parent_limit = "1283"
    else:
        dataset_dir = (
            context.step0
            / "outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset"
        )
        distance = (
            context.step0
            / "outputs/hpc/pretrained/gcfexplainer/mutagenicity/neurosed/best_model.pt"
        )
        source_args = []
        parent_limit = "1448"
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/run_common_recourse.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        dataset,
        "--mode",
        "full",
        "--upstream-root",
        str(context.external),
        "--dataset-dir",
        str(dataset_dir),
        *source_args,
        "--generation-dir",
        str(base / "generation"),
        "--distance-checkpoint",
        str(distance),
        "--output-dir",
        str(base / "common_recourse"),
        "--parent-limit",
        parent_limit,
        "--device",
        "cuda:0",
    ]


def _chemistry_command(context: Context, dataset: str, base: Path) -> list[str]:
    if dataset == "aids":
        dataset_dir = context.step0 / "outputs/hpc/current/gcfexplainer/aids/dataset"
        source_args = [
            "--source-csv",
            str(
                context.static_input
                / "outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset"
                / "sft_v3_hiv_ppo_prompts_train_label1.csv"
            ),
        ]
        parent_limit = "1283"
    else:
        dataset_dir = (
            context.step0
            / "outputs/hpc/mutagenicity/baselines/gcfexplainer/smoke_v1/dataset"
        )
        source_args = []
        parent_limit = "1448"
    trace = base / "generation" / "trace"
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(context.project),
        "--dataset",
        dataset,
        "--dataset-dir",
        str(dataset_dir),
        *source_args,
        "--generation-dir",
        str(base / "generation"),
        "--trace-lineage-path",
        str(trace / "candidate_action_lineage.json"),
        "--trace-evidence-path",
        str(trace / "trace_summary.json"),
        "--common-recourse-dir",
        str(base / "common_recourse"),
        "--output-dir",
        str(base / "chemistry"),
        "--preregistration-path",
        str(base / "preregistration/deterministic_chem_repair.json"),
        "--parent-limit",
        parent_limit,
    ]


def _slot_eval_command(context: Context, dataset: str, base: Path) -> list[str]:
    molclr_root = context.step0 / "pretrained_models/MolCLR"
    molclr_checkpoint = molclr_root / "ckpt/pretrained_gin/checkpoints/model.pth"
    if dataset == "aids":
        dataset_csv = (
            context.static_input
            / "outputs/hpc/sft_v3_hiv_runs/sft_v3_hiv_20260508_resplit/dataset"
            / "sft_v3_hiv_ppo_prompts_train_label1.csv"
        )
        teacher = context.step0 / "outputs/hpc/oracle/aids_rf_model.pkl"
        thresholds = (
            context.step0
            / "outputs/hpc/eval/paper/molclr_node_wasserstein_figure4_redline_k10"
            / "wnode_figure4_redline_k10_figure4_wnode_coverage_vs_threshold.csv"
        )
        extra = ["--theta-star", "0.05", "--cost-cap", "0.0535"]
        expected = "1283"
    else:
        dataset_csv = (
            context.step0
            / "outputs/hpc/datasets/mutagenicity_v1_teacher_consistent"
            / "test_source_label1_teacher_correct.csv"
        )
        teacher = (
            context.step0
            / "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl"
        )
        thresholds = (
            context.step0
            / "outputs/hpc/mutagenicity/final/ours_wnode_a2_test_v1/thresholds.json"
        )
        extra = []
        expected = "217"
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/run_slot_unified_eval.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        dataset,
        "--mode",
        "full",
        "--chemistry-dir",
        str(base / "chemistry"),
        "--dataset-csv",
        str(dataset_csv),
        "--teacher-path",
        str(teacher),
        "--molclr-root",
        str(molclr_root),
        "--molclr-checkpoint",
        str(molclr_checkpoint),
        "--thresholds-json",
        str(thresholds),
        *extra,
        "--output-dir",
        str(base / "unified_eval"),
        "--expected-parent-count",
        expected,
        "--max-k",
        "20",
        "--device",
        "cuda",
    ]


def _validate_standardized_freeze(
    output: Path, *, source: Path, gate: Path
) -> None:
    finalized = read_json(output / "_FINALIZED.json")
    freeze_manifest_path = output / "freeze_manifest.json"
    _assert_nonempty_file(freeze_manifest_path)
    if finalized.get("freeze_manifest_sha256") != sha256_file(
        freeze_manifest_path
    ):
        raise StageError("Standardized freeze manifest digest chain failed")
    freeze_manifest = read_json(freeze_manifest_path)
    if freeze_manifest.get("source_run_manifest_sha256") != sha256_file(
        source / "run_manifest.json"
    ):
        raise StageError("Standardized freeze source run manifest changed")
    if freeze_manifest.get("source_gate_result_sha256") != sha256_file(
        gate / "gate_result.json"
    ):
        raise StageError("Standardized freeze source gate changed")
    files = freeze_manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise StageError("Standardized freeze inventory is empty")
    for relative_name, expected in files.items():
        relative = _safe_relative(str(relative_name))
        if relative.parent != Path(".") or not isinstance(expected, Mapping):
            raise StageError("Standardized freeze inventory path is invalid")
        artifact = output / relative
        if not artifact.is_file() or artifact.is_symlink():
            raise StageError(f"Standardized freeze inventory file is missing: {relative}")
        if (
            int(expected.get("bytes", -1)) != artifact.stat().st_size
            or expected.get("sha256") != sha256_file(artifact)
        ):
            raise StageError(f"Standardized freeze inventory mismatch: {relative}")


def _run_downstream(context: Context, dataset: str) -> None:
    _require_disallow_generation()
    input_name = "aids_generation" if dataset == "aids" else "mut_generation"
    inputs = _all_input_gates(context, input_name)
    expected_inputs = _input_manifest_digests(inputs)
    input_before = inputs.primary_digest
    base = context.persistent / "outputs" / f"{dataset if dataset == 'aids' else 'mut'}_comrecgc"
    freeze_sentinel = base / (
        "AIDS_FREEZE_RECOVERY_PASS.json" if dataset == "aids" else "MUT_FREEZE_RECOVERY_PASS.json"
    )
    freeze_manifest = (
        base / "manifests" / f"{dataset}_freeze_recovery.sha256"
    )
    if not _verify_sentinel(
        context,
        freeze_sentinel,
        freeze_manifest,
        {"status": "PASS", "generation_rerun_performed": False},
        input_manifests=expected_inputs,
    ):
        raise StageError(f"Missing verified freeze dependency: {freeze_sentinel}")
    sentinel = base / (
        "AIDS_COMRECGC_COMPLETE.json" if dataset == "aids" else "MUT_COMRECGC_COMPLETE.json"
    )
    manifest = base / "MANIFEST.sha256"
    fields = {"status": "PASS", "generation_rerun_performed": False}
    if _verify_sentinel(
        context, sentinel, manifest, fields, input_manifests=expected_inputs
    ):
        return
    _run_or_reuse(
        context=context,
        output=base / "common_recourse",
        marker=base / "common_recourse/_RUN_COMPLETE.json",
        command=_common_recourse_command(context, dataset, base),
        resumable=True,
        input_manifests=expected_inputs,
    )
    _run_or_reuse(
        context=context,
        output=base / "chemistry",
        marker=base / "chemistry/_RUN_COMPLETE.json",
        command=_chemistry_command(context, dataset, base),
        resumable=False,
        input_manifests=expected_inputs,
    )
    _run_or_reuse(
        context=context,
        output=base / "unified_eval",
        marker=base / "unified_eval/_RUN_COMPLETE.json",
        command=_slot_eval_command(context, dataset, base),
        resumable=True,
        input_manifests=expected_inputs,
    )
    teacher = (
        context.step0 / "outputs/hpc/oracle/aids_rf_model.pkl"
        if dataset == "aids"
        else context.step0
        / "outputs/hpc/oracle/mutagenicity_rf_v1/mutagenicity_rf_model.pkl"
    )
    expected_parents = "1283" if dataset == "aids" else "217"
    gate_command = [
        str(context.python),
        context.script("scripts/baselines/comrecgc/gate_recovery.py"),
        "--stage",
        "project-full",
        "--dataset",
        dataset,
        "--expected-parent-count",
        expected_parents,
        "--expected-teacher-sha256",
        sha256_file(teacher),
        "--expected-project-commit",
        _git_head(context.project),
        "--input-dir",
        str(base / "unified_eval"),
        "--output-dir",
        str(base / "full_gate"),
    ]
    _run_or_reuse(
        context=context,
        output=base / "full_gate",
        marker=base / "full_gate/_RUN_COMPLETE.json",
        command=gate_command,
        resumable=False,
        input_manifests=expected_inputs,
    )
    freeze_command = [
        str(context.python),
        context.script("scripts/baselines/comrecgc/freeze_recovery_result.py"),
        "--dataset",
        dataset,
        "--source-dir",
        str(base / "unified_eval"),
        "--gate-dir",
        str(base / "full_gate"),
        "--output-dir",
        str(base / "standardized"),
    ]
    finalized = base / "standardized/_FINALIZED.json"
    _run_or_reuse(
        context=context,
        output=base / "standardized",
        marker=finalized,
        marker_fields={"finalized": True, "gate_passed": True},
        command=freeze_command,
        resumable=False,
        input_manifests=expected_inputs,
    )
    _validate_standardized_freeze(
        base / "standardized",
        source=base / "unified_eval",
        gate=base / "full_gate",
    )
    write_sha256_manifest(
        base=base,
        items=[
            base / "generation",
            base / "common_recourse",
            base / "chemistry",
            base / "unified_eval",
            base / "full_gate",
            base / "standardized",
            base / "preregistration",
        ],
        manifest=manifest,
        exclude=[manifest, sentinel],
    )
    input_manifests = _verify_all_input_gates(inputs)
    input_after = input_manifests["primary"]
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=base,
        input_digest_before=input_before,
        input_digest_after=input_after,
        input_manifests=input_manifests,
        payload={**fields, "dataset": dataset, "static_runtime_ready": True},
    )


def bace_generation_command(context: Context, *, resume: bool) -> list[str]:
    output = context.fast / "active/bace_comrecgc/generation_fresh"
    checkpoint = context.fast / "active/bace_comrecgc/generation_checkpoints"
    mirror = context.persistent / "outputs/bace_comrecgc/generation_checkpoint_mirror"
    trace = context.persistent / "outputs/bace_comrecgc/generation_resume_metadata/trace"
    command = [
        str(context.python),
        context.script("scripts/baselines/comrecgc/run_generation.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--route",
        "project",
        "--dataset",
        "bace",
        "--mode",
        "full",
        "--project-root",
        str(context.project),
        "--upstream-root",
        str(context.external),
        "--dataset-dir",
        str(context.step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset"),
        "--gnn-checkpoint",
        str(context.step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/gnn/model_best.pth"),
        "--distance-checkpoint",
        str(
            context.step0
            / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/neurosed/best_model.pt"
        ),
        "--output-dir",
        str(output),
        "--parent-limit",
        "360",
        "--device",
        "cuda:0",
        "--batch-size",
        "128",
        "--trace-output-dir",
        str(trace),
        "--graph-state-dir",
        str(context.fast / "active/bace_comrecgc/graph_state"),
        "--checkpoint-root",
        str(checkpoint),
        "--checkpoint-mirror-root",
        str(mirror),
        "--checkpoint-interval-steps",
        "500",
        "--checkpoint-keep-last",
        "2",
        "--progress-interval-steps",
        "25",
        "--storage-guard-root",
        str(context.fast / "active/bace_comrecgc"),
        "--storage-check-every-steps",
        "500",
        "--storage-min-free-gib",
        "8",
        "--storage-min-free-ratio",
        ".10",
        "--storage-min-free-inodes",
        "100000",
    ]
    if resume:
        command.append("--resume")
    return command


def _checkpoint_module(context: Context) -> Any:
    if str(context.project) not in sys.path:
        sys.path.insert(0, str(context.project))
    from src.baselines.comrecgc import generation_checkpoint

    return generation_checkpoint


def _validate_mirrored_checkpoint(module: Any, checkpoint: Path) -> Any:
    validation = module.validate_generation_checkpoint(checkpoint)
    marker_path = validation.checkpoint_dir / module.MIRRORED_FILENAME
    marker = read_json(marker_path)
    if (
        marker.get("checkpoint_mirrored") is not True
        or marker.get("checkpoint_digest") != validation.checkpoint_digest
        or int(marker.get("completed_step", -1)) != int(validation.completed_step)
    ):
        raise StageError(f"Persistent checkpoint mirror proof failed: {checkpoint}")
    return validation


def _select_fully_mirrored_checkpoints(
    module: Any, mirror_root: Path, *, keep_last: int = 2
) -> tuple[list[Any], list[dict[str, Any]]]:
    candidates = module.list_generation_checkpoints(mirror_root)
    selected: list[Any] = []
    ignored: list[dict[str, Any]] = []
    for checkpoint in reversed(candidates):
        marker = checkpoint / module.MIRRORED_FILENAME
        if marker.is_symlink() or not marker.is_file():
            ignored.append(
                {
                    "checkpoint": str(checkpoint),
                    "reason": "valid_checkpoint_without_fully_mirrored_marker",
                }
            )
            continue
        # A present but invalid proof is evidence of corruption, not a benign
        # crash-left directory, and therefore remains fail-closed.
        selected.append(_validate_mirrored_checkpoint(module, checkpoint))
        if len(selected) >= int(keep_last):
            break
    if not selected:
        raise StageError(
            f"No fully committed persistent checkpoint mirror: {mirror_root}"
        )
    selected.reverse()
    return selected, ignored


def _restore_checkpoint_mirror(context: Context) -> int:
    module = _checkpoint_module(context)
    fast_root = context.fast / "active/bace_comrecgc/generation_checkpoints"
    mirror_root = context.persistent / "outputs/bace_comrecgc/generation_checkpoint_mirror"
    mirror_validations, ignored = _select_fully_mirrored_checkpoints(
        module, mirror_root, keep_last=2
    )
    audit_path = (
        context.persistent
        / "outputs/bace_comrecgc/generation_resume_metadata/mirror_selection_audit.json"
    )
    atomic_write_json(
        audit_path,
        {
            "schema_version": "bace_checkpoint_mirror_selection_v1",
            "selected": [
                {
                    "checkpoint": str(value.checkpoint_dir),
                    "completed_step": int(value.completed_step),
                    "checkpoint_digest": str(value.checkpoint_digest),
                }
                for value in mirror_validations
            ],
            "ignored_uncommitted": ignored,
            "selected_at": utc_now(),
        },
    )
    persistent_latest = mirror_validations[-1]
    if fast_root.is_symlink():
        raise StageError(f"Fast checkpoint root must be physical: {fast_root}")
    fast_root.mkdir(parents=True, exist_ok=True)

    # Persistent checkpoints carrying a valid fully-mirrored proof are the
    # only recovery authority.  Inspect every already-published fast
    # checkpoint before writing anything: a same-step digest conflict or a
    # checkpoint newer than the persistent authority is not a benign crash
    # window and must never be selected implicitly by LATEST recovery.
    selected_by_step = {
        int(validation.completed_step): validation
        for validation in mirror_validations
    }
    checkpoint_name = re.compile(r"^step-(?P<step>[0-9]{12})$")
    for entry in sorted(fast_root.iterdir(), key=lambda value: value.name):
        match = checkpoint_name.fullmatch(entry.name)
        if match is None:
            continue
        if entry.is_symlink() or not entry.is_dir():
            raise StageError(f"Unsafe fast checkpoint entry: {entry}")
        try:
            fast_validation = module.validate_generation_checkpoint(entry)
        except Exception as exc:
            raise StageError(f"Untrusted fast checkpoint exists: {entry}") from exc
        fast_step = int(fast_validation.completed_step)
        if fast_step > int(persistent_latest.completed_step):
            raise StageError(
                "Fast checkpoint is newer than the latest verified persistent "
                f"mirror: {entry}"
            )
        authoritative = selected_by_step.get(fast_step)
        if (
            authoritative is not None
            and fast_validation.checkpoint_digest
            != authoritative.checkpoint_digest
        ):
            raise StageError(f"Conflicting fast checkpoint exists: {entry}")

    # Materialise both authoritative persistent checkpoints whenever two are
    # available.  This deliberately runs even when the fast root already has
    # a valid latest checkpoint: the prior checkpoint may have been lost in a
    # crash window and is required for the latest-2 retention invariant.
    for validation in mirror_validations:
        source = validation.checkpoint_dir
        destination = fast_root / source.name
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise StageError(f"Unsafe fast checkpoint collision: {destination}")
            try:
                existing = module.validate_generation_checkpoint(destination)
            except Exception as exc:
                raise StageError(
                    f"Untrusted fast checkpoint exists: {destination}"
                ) from exc
            if (
                int(existing.completed_step) != int(validation.completed_step)
                or existing.checkpoint_digest != validation.checkpoint_digest
            ):
                raise StageError(f"Conflicting fast checkpoint exists: {destination}")
            continue
        _copy_tree_atomic(source, destination)
        restored = module.validate_generation_checkpoint(destination)
        if (
            int(restored.completed_step) != int(validation.completed_step)
            or restored.checkpoint_digest != validation.checkpoint_digest
        ):
            raise StageError(f"Restored checkpoint digest mismatch: {destination}")

    # Publish LATEST only after the complete authoritative latest-2 set is
    # present.  Do not rely on validate_generation_checkpoint(root)'s repair
    # side effect: explicitly materialising this pointer makes the recovery
    # crash boundary and the final authority check unambiguous.
    latest_payload = {
        "schema_version": str(module.LATEST_SCHEMA_VERSION),
        "checkpoint_dir": persistent_latest.checkpoint_dir.name,
        "completed_step": int(persistent_latest.completed_step),
        "checkpoint_digest": str(persistent_latest.checkpoint_digest),
    }
    atomic_write_json(fast_root / str(module.LATEST_FILENAME), latest_payload)
    latest = module.validate_generation_checkpoint(fast_root)
    if (
        int(latest.completed_step) != int(persistent_latest.completed_step)
        or latest.checkpoint_digest != persistent_latest.checkpoint_digest
        or read_json(fast_root / str(module.LATEST_FILENAME)) != latest_payload
    ):
        raise StageError("Fast checkpoint LATEST is not the persistent mirror LATEST")
    _reconcile_trace_to_checkpoint(
        context,
        checkpoint_root=fast_root,
        trace_root=(
            context.persistent
            / "outputs/bace_comrecgc/generation_resume_metadata/trace"
        ),
        quarantine_root=(
            context.persistent
            / "outputs/bace_comrecgc/generation_resume_metadata/trace_recovery_quarantine"
        ),
    )
    return int(latest.completed_step)


def _reconcile_trace_to_checkpoint(
    context: Context,
    *,
    checkpoint_root: Path,
    trace_root: Path,
    quarantine_root: Path,
) -> dict[str, Any]:
    """Retain the checkpoint trace prefix and quarantine crash-window extras."""

    module = _checkpoint_module(context)
    loaded = module.load_generation_checkpoint(checkpoint_root)
    chunks = loaded.trace_state.get("chunks") or []
    expected: dict[Path, tuple[int, str]] = {}
    for row in chunks:
        relative = _safe_relative(str(row.get("path") or ""))
        if relative.parent != Path("selected_action_trace_chunks"):
            raise StageError(f"Checkpoint trace chunk path is invalid: {relative}")
        expected[relative] = (int(row.get("bytes", -1)), str(row.get("sha256") or ""))
    for relative, (size, digest) in expected.items():
        path = trace_root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise StageError(f"Persistent trace prefix differs from checkpoint: {path}")
    extras: list[Path] = []
    if trace_root.is_dir():
        for path in _physical_files(trace_root):
            relative = path.relative_to(trace_root)
            if relative not in expected:
                extras.append(relative)
    if extras:
        run_quarantine = quarantine_root / (
            f"checkpoint-{loaded.completed_step:012d}-{int(time.time_ns())}"
        )
        for relative in extras:
            source = trace_root / relative
            destination = run_quarantine / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, destination)
        _fsync_directory(trace_root)
        _fsync_directory(run_quarantine)
    audit = {
        "schema_version": "comrecgc_trace_checkpoint_reconciliation_v1",
        "checkpoint_step": loaded.completed_step,
        "checkpoint_digest": loaded.validation.checkpoint_digest,
        "expected_chunk_count": len(expected),
        "quarantined_extra_files": [value.as_posix() for value in extras],
        "audited_at": utc_now(),
    }
    atomic_write_json(quarantine_root.parent / "trace_checkpoint_reconciliation.json", audit)
    return audit


def _monitor_bace_resume_metadata(context: Context) -> None:
    source = context.fast / "active/bace_comrecgc/generation_fresh/resolved_config.json"
    destination = (
        context.persistent
        / "outputs/bace_comrecgc/generation_resume_metadata/resolved_config.json"
    )
    if not source.is_file() or source.is_symlink():
        return
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise StageError("BACE resolved_config changed after persistent publication")
        return
    _copy_file_atomic(source, destination)


def _restore_bace_resume_metadata(context: Context) -> None:
    output = context.fast / "active/bace_comrecgc/generation_fresh"
    source = (
        context.persistent
        / "outputs/bace_comrecgc/generation_resume_metadata/resolved_config.json"
    )
    destination = output / "resolved_config.json"
    _assert_nonempty_file(source)
    output.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(source) != sha256_file(destination):
            raise StageError("Fast and persistent resolved_config disagree")
    else:
        _copy_file_atomic(source, destination)


def _persist_bace_generation(
    context: Context, *, input_manifests: Mapping[str, str]
) -> Path:
    fast_generation = context.fast / "active/bace_comrecgc/generation_fresh"
    persistent_generation = context.persistent / "outputs/bace_comrecgc/generation"
    persistent_trace = (
        context.persistent / "outputs/bace_comrecgc/generation_resume_metadata/trace"
    )
    if persistent_generation.is_dir():
        _verify_reuse_proof(
            context=context,
            output=persistent_generation,
            marker=persistent_generation / "_RUN_COMPLETE.json",
            marker_fields={"run_complete": True},
            command=bace_generation_command(context, resume=False),
            input_manifests=input_manifests,
        )
        return persistent_generation
    _assert_nonempty_file(fast_generation / "_RUN_COMPLETE.json")
    _assert_nonempty_file(persistent_trace / "_TRACE_COMPLETE.json")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".generation.",
            suffix=".tmp",
            dir=(context.persistent / "outputs/bace_comrecgc"),
        )
    )
    published = False
    try:
        list(_physical_files(fast_generation))
        shutil.copytree(fast_generation, temporary, dirs_exist_ok=True, symlinks=False)
        trace_destination = temporary / "trace"
        if trace_destination.exists():
            raise StageError("Fast BACE output unexpectedly contains a second trace tree")
        shutil.copytree(persistent_trace, trace_destination, symlinks=False)
        _fsync_directory(temporary)
        os.rename(temporary, persistent_generation)
        published = True
        _fsync_directory(persistent_generation.parent)
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)
    _publish_reuse_proof(
        context=context,
        output=persistent_generation,
        marker=persistent_generation / "_RUN_COMPLETE.json",
        marker_fields={"run_complete": True},
        command=bace_generation_command(context, resume=False),
        input_manifests=input_manifests,
    )
    return persistent_generation


def _run_bace_generate(context: Context) -> None:
    inputs = _all_input_gates(context, "bace_preserved")
    expected_inputs = _input_manifest_digests(inputs)
    profile_gate = _require_bace_profile_smoke_gate(
        context, input_manifests=expected_inputs
    )
    input_before = inputs.primary_digest
    output_root = context.persistent / "outputs/bace_comrecgc"
    output_root.mkdir(parents=True, exist_ok=True)
    generation = output_root / "generation"
    manifest = generation / "MANIFEST.sha256"
    sentinel = output_root / "BACE_GENERATION_50000_PASS.json"
    fields = {
        "status": "PASS",
        "fresh_start_step": 0,
        "imported_old_partial_state": False,
        "completed_step": 50_000,
    }
    if _verify_sentinel(
        context, sentinel, manifest, fields, input_manifests=expected_inputs
    ):
        return
    fast_generation = context.fast / "active/bace_comrecgc/generation_fresh"
    fast_checkpoints = context.fast / "active/bace_comrecgc/generation_checkpoints"
    fast_graph_state = context.fast / "active/bace_comrecgc/graph_state"
    mirror = output_root / "generation_checkpoint_mirror"
    metadata = output_root / "generation_resume_metadata"
    launched_generation = False
    if context.resume:
        if not _complete_marker(fast_generation / "_RUN_COMPLETE.json"):
            _restore_bace_resume_metadata(context)
            restored_step = _restore_checkpoint_mirror(context)
            if restored_step <= 0 or restored_step >= 50_000:
                # Step 50000 may need only finalisation, and is still a valid
                # exact resume boundary.  Greater values are impossible.
                if restored_step != 50_000:
                    raise StageError(f"Invalid mirrored BACE checkpoint step: {restored_step}")
            _run_checked(
                bace_generation_command(context, resume=True),
                cwd=context.project,
                allow_generation=True,
                monitor=lambda: _monitor_bace_resume_metadata(context),
            )
            launched_generation = True
        else:
            _verify_reuse_proof(
                context=context,
                output=fast_generation,
                marker=fast_generation / "_RUN_COMPLETE.json",
                marker_fields={"run_complete": True},
                command=bace_generation_command(context, resume=False),
                input_manifests=expected_inputs,
            )
    else:
        for path, label in (
            (generation, "persistent generation"),
            (fast_generation, "fresh generation"),
            (fast_checkpoints, "fast checkpoint"),
            (fast_graph_state, "fast graph state"),
            (mirror, "persistent checkpoint mirror"),
            (metadata, "persistent resume metadata"),
        ):
            _assert_empty_or_absent(path, label=label)
        _run_checked(
            bace_generation_command(context, resume=False),
            cwd=context.project,
            allow_generation=True,
            monitor=lambda: _monitor_bace_resume_metadata(context),
        )
        launched_generation = True
    progress = read_json(fast_generation / "progress.json")
    complete = read_json(fast_generation / "_RUN_COMPLETE.json")
    run_manifest = read_json(fast_generation / "run_manifest.json")
    if (
        progress.get("run_complete") is not True
        or int(progress.get("current_step", -1)) != 50_000
        or complete.get("run_complete") is not True
        or run_manifest.get("run_complete") is not True
        or run_manifest.get("algorithm_rerun") is not True
    ):
        raise StageError("BACE generation did not reach the exact 50000-step boundary")
    if launched_generation:
        _publish_reuse_proof(
            context=context,
            output=fast_generation,
            marker=fast_generation / "_RUN_COMPLETE.json",
            marker_fields={"run_complete": True},
            command=bace_generation_command(context, resume=context.resume),
            input_manifests=expected_inputs,
        )
    _monitor_bace_resume_metadata(context)
    generation = _persist_bace_generation(
        context, input_manifests=expected_inputs
    )
    exact_generation_gate = _validate_bace_generation_exact_gate(
        context=context,
        generation=generation,
        mirror=mirror,
        metadata=metadata,
    )
    exact_generation_gate.update(
        {
            "uninterrupted_resume_equivalence_test_passed": bool(
                profile_gate["checkpoint_parity"]
            ),
            "abrupt_kill_recovery_test_passed": bool(
                profile_gate["abrupt_kill_test_passed"]
            ),
        }
    )
    write_sha256_manifest(
        base=output_root,
        items=[generation, mirror, metadata],
        manifest=manifest,
        exclude=[manifest, sentinel],
    )
    input_manifests = _verify_all_input_gates(inputs)
    input_after = input_manifests["primary"]
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=output_root,
        input_digest_before=input_before,
        input_digest_after=input_after,
        input_manifests=input_manifests,
        payload={
            **fields,
            "generation_rerun_performed": True,
            "trace_persistence": "persistent_atomic_chunks",
            "active_sqlite_location": "fast_ephemeral",
            "checkpoint_location": "fast_with_persistent_latest2_mirror",
            "exact_generation_gate": exact_generation_gate,
            "checkpoint_parity": profile_gate["checkpoint_parity"],
            "abrupt_kill_test_passed": profile_gate["abrupt_kill_test_passed"],
            "profile_report_json_sha256": profile_gate["report_json_sha256"],
            "profile_report_text_sha256": profile_gate["report_text_sha256"],
        },
    )


def _bace_common_command(context: Context, base: Path) -> list[str]:
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/run_common_recourse.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--dataset",
        "bace",
        "--mode",
        "full",
        "--upstream-root",
        str(context.external),
        "--dataset-dir",
        str(context.step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset"),
        "--generation-dir",
        str(base / "generation"),
        "--distance-checkpoint",
        str(
            context.step0
            / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/neurosed/best_model.pt"
        ),
        "--output-dir",
        str(base / "common_recourse"),
        "--parent-limit",
        "360",
        "--device",
        "cuda:0",
    ]


def _bace_chemistry_command(context: Context, base: Path) -> list[str]:
    trace = base / "generation/trace"
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/audit_mutagenicity_chemistry.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--project-root",
        str(context.project),
        "--dataset",
        "bace",
        "--dataset-dir",
        str(context.step0 / "outputs/hpc/bace/baselines/gcfexplainer/full_v2/dataset"),
        "--generation-dir",
        str(base / "generation"),
        "--trace-lineage-path",
        str(trace / "candidate_action_lineage.json"),
        "--trace-evidence-path",
        str(trace / "trace_summary.json"),
        "--common-recourse-dir",
        str(base / "common_recourse"),
        "--output-dir",
        str(base / "chemistry"),
        "--preregistration-path",
        str(base / "preregistration/deterministic_chem_repair.json"),
        "--parent-limit",
        "360",
    ]


def _bace_eval_command(context: Context, base: Path) -> list[str]:
    molclr = context.step0 / "pretrained_models/MolCLR"
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/run_slot_unified_eval.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--mode",
        "full",
        "--dataset",
        "bace",
        "--chemistry-dir",
        str(base / "chemistry"),
        "--dataset-csv",
        str(
            context.static_input
            / "outputs/hpc/oracle/bace/teacher_consistent"
            / "test_source_label1_teacher_correct.csv"
        ),
        "--teacher-path",
        str(context.step0 / "outputs/hpc/oracle/bace/bace_teacher.pkl"),
        "--molclr-root",
        str(molclr),
        "--molclr-checkpoint",
        str(molclr / "ckpt/pretrained_gin/checkpoints/model.pth"),
        "--thresholds-json",
        str(context.persistent / "inputs/bace_preserved/common4/thresholds.json"),
        "--output-dir",
        str(base / "paper/comrecgc"),
        "--expected-parent-count",
        "116",
        "--max-k",
        "20",
        "--device",
        "cuda",
    ]


def _bace_artifact_gate_command(context: Context, paper: Path) -> list[str]:
    return [
        str(context.python),
        context.script("scripts/baselines/comrecgc/audit_bace_artifacts.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--root",
        str(paper),
        "--thresholds-json",
        str(context.bace_input / "common4/thresholds.json"),
        "--expected-parent-count",
        "116",
    ]


def _run_bace_final(context: Context) -> None:
    inputs = _all_input_gates(context, "bace_preserved")
    expected_inputs = _input_manifest_digests(inputs)
    input_before = inputs.primary_digest
    base = context.persistent / "outputs/bace_comrecgc"
    if not _verify_sentinel(
        context,
        base / "BACE_GENERATION_50000_PASS.json",
        base / "generation/MANIFEST.sha256",
        {
            "status": "PASS",
            "fresh_start_step": 0,
            "imported_old_partial_state": False,
            "completed_step": 50_000,
        },
        input_manifests=expected_inputs,
    ):
        raise StageError("Missing verified BACE generation dependency")
    sentinel = base / "BACE_COMRECGC_COMPLETE.json"
    manifest = base / "MANIFEST.sha256"
    fields = {"status": "PASS"}
    if _verify_sentinel(
        context, sentinel, manifest, fields, input_manifests=expected_inputs
    ):
        return
    _run_or_reuse(
        context=context,
        output=base / "common_recourse",
        marker=base / "common_recourse/_RUN_COMPLETE.json",
        command=_bace_common_command(context, base),
        resumable=True,
        input_manifests=expected_inputs,
    )
    _run_or_reuse(
        context=context,
        output=base / "chemistry",
        marker=base / "chemistry/_RUN_COMPLETE.json",
        command=_bace_chemistry_command(context, base),
        resumable=False,
        input_manifests=expected_inputs,
    )
    paper = base / "paper/comrecgc"
    _run_or_reuse(
        context=context,
        output=paper,
        marker=paper / "final_artifact_audit.json",
        marker_fields={"audit_passed": True},
        command=_bace_eval_command(context, base),
        resumable=True,
        input_manifests=expected_inputs,
    )
    audit = read_json(paper / "final_artifact_audit.json")
    if audit.get("audit_passed") is not True:
        raise StageError("BACE ComRecGC final artifact audit did not pass")
    artifact_gate = paper / "bace_comrecgc_artifact_gate.json"
    _run_checked(
        _bace_artifact_gate_command(context, paper),
        cwd=context.project,
    )
    gate_payload = read_json(artifact_gate)
    if gate_payload.get("passed") is not True:
        raise StageError("BACE ComRecGC artifact gate did not pass")
    write_sha256_manifest(
        base=base,
        items=[
            base / "generation",
            base / "generation_checkpoint_mirror",
            base / "generation_resume_metadata",
            base / "common_recourse",
            base / "chemistry",
            base / "preregistration",
            paper,
        ],
        manifest=manifest,
        exclude=[manifest, sentinel],
    )
    input_manifests = _verify_all_input_gates(inputs)
    input_after = input_manifests["primary"]
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=base,
        input_digest_before=input_before,
        input_digest_after=input_after,
        input_manifests=input_manifests,
        payload={
            **fields,
            "dataset": "bace",
            "final_artifact_audit_passed": True,
            "bace_comrecgc_artifact_gate_passed": True,
        },
    )


def bace_globalgce_command(context: Context, *, resume: bool = False) -> list[str]:
    common = context.persistent / "outputs/bace_globalgce_common4/common4"
    molclr = context.step0 / "pretrained_models/MolCLR"
    command = [
        str(context.python),
        context.script("scripts/evaluate_bace_method.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--method",
        "globalgce",
        "--candidate-path",
        str(context.bace_input / "globalgce_selector/selected_top20_for_eval.csv"),
        "--selection-manifest",
        str(context.bace_input / "globalgce_selector/frozen_selection.json"),
        "--teacher-path",
        str(context.step0 / "outputs/hpc/oracle/bace/bace_teacher.pkl"),
        "--molclr-root",
        str(molclr),
        "--molclr-checkpoint",
        str(molclr / "ckpt/pretrained_gin/checkpoints/model.pth"),
        "--test-csv",
        str(
            context.static_input
            / "outputs/hpc/oracle/bace/teacher_consistent"
            / "test_source_label1_teacher_correct.csv"
        ),
        "--thresholds-json",
        str(common / "thresholds.json"),
        "--work-dir",
        str(context.fast / "cache/bace_globalgce_common4/globalgce"),
        "--output-dir",
        str(common / "globalgce"),
        "--expected-test-parents",
        "116",
        "--device",
        "cuda",
        "--test-evaluation-count",
        "1",
        "--reference-artifact-root",
        str(common / "ours"),
        "--action-semantics-version",
        "connected_sanitized_residual_v1",
        "--match-selection-policy",
        "existential_min_wnode_among_valid_connected_strict_flips_v1",
        "--wnode-cache-db",
        str(context.fast / "cache/bace_globalgce_common4/test_wnode.sqlite3"),
    ]
    if resume:
        command.append("--resume")
    return command


def _materialize_preserved_common4(context: Context) -> Path:
    source = context.bace_input / "common4"
    destination = context.persistent / "outputs/bace_globalgce_common4/common4"
    if destination.exists():
        if not destination.is_dir() or destination.is_symlink():
            raise StageError(f"Invalid common4 materialisation: {destination}")
        _assert_tree_contains_identical(source, destination)
    else:
        _copy_tree_atomic(source, destination)
    return destination


def _run_bace_globalgce(context: Context) -> None:
    inputs = _all_input_gates(context, "bace_preserved")
    expected_inputs = _input_manifest_digests(inputs)
    input_before = inputs.primary_digest
    base = context.persistent / "outputs/bace_globalgce_common4"
    base.mkdir(parents=True, exist_ok=True)
    common = _materialize_preserved_common4(context)
    output = common / "globalgce"
    sentinel = base / "BACE_GLOBALGCE_WNODE_COMPLETE.json"
    manifest = output / "MANIFEST.sha256"
    fields = {
        "status": "PASS",
        "ours_generation_rerun": False,
        "gcf_generation_rerun": False,
        "globalgce_selection_rerun": False,
    }
    if _verify_sentinel(
        context, sentinel, manifest, fields, input_manifests=expected_inputs
    ):
        return
    _run_or_reuse(
        context=context,
        output=output,
        marker=output / "final_artifact_audit.json",
        marker_fields={"passed": True},
        command=bace_globalgce_command(context, resume=False),
        resumable=True,
        input_manifests=expected_inputs,
    )
    audit = read_json(output / "final_artifact_audit.json")
    if audit.get("passed") is not True and audit.get("audit_passed") is not True:
        raise StageError("BACE GlobalGCE final artifact audit did not pass")
    write_sha256_manifest(
        base=output,
        items=[output],
        manifest=manifest,
        exclude=[manifest],
    )
    input_manifests = _verify_all_input_gates(inputs)
    input_after = input_manifests["primary"]
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=output,
        input_digest_before=input_before,
        input_digest_after=input_after,
        input_manifests=input_manifests,
        payload={**fields, "dataset": "bace", "evaluation_resume_supported": True},
    )


def _run_bace_common4(context: Context) -> None:
    inputs = _all_input_gates(context, "bace_preserved")
    expected_inputs = _input_manifest_digests(inputs)
    input_before = inputs.primary_digest
    base = context.persistent / "outputs/bace_globalgce_common4"
    common = _materialize_preserved_common4(context)
    comrecgc_base = context.persistent / "outputs/bace_comrecgc"
    if not _verify_sentinel(
        context,
        comrecgc_base / "BACE_COMRECGC_COMPLETE.json",
        comrecgc_base / "MANIFEST.sha256",
        {"status": "PASS"},
        input_manifests=expected_inputs,
    ):
        raise StageError("Missing verified BACE ComRecGC dependency")
    if not _verify_sentinel(
        context,
        base / "BACE_GLOBALGCE_WNODE_COMPLETE.json",
        common / "globalgce/MANIFEST.sha256",
        {
            "status": "PASS",
            "ours_generation_rerun": False,
            "gcf_generation_rerun": False,
            "globalgce_selection_rerun": False,
        },
        input_manifests=expected_inputs,
    ):
        raise StageError("Missing verified BACE GlobalGCE dependency")
    source = context.persistent / "outputs/bace_comrecgc/paper/comrecgc"
    destination = common / "comrecgc"
    if destination.exists():
        if not (destination / "final_artifact_audit.json").is_file():
            raise StageError("Partial common4 ComRecGC import exists")
        _assert_tree_contains_identical(source, destination)
    else:
        _copy_tree_atomic(source, destination)
    for method in ("ours", "gcfexplainer", "globalgce", "comrecgc"):
        _assert_nonempty_file(common / method / "final_artifact_audit.json")
    command = [
        str(context.python),
        context.script("scripts/audit_bace_common4_connected.py"),
        "--config",
        "configs/hpc.yaml",
        "--set",
        "inference.fallback_to_heuristic=false",
        "--root",
        str(common),
    ]
    sentinel = base / "BACE_COMMON4_COMPLETE.json"
    manifest = common / "MANIFEST.sha256"
    fields = {"status": "PASS", "canonical_method_count": 4}
    if _verify_sentinel(
        context, sentinel, manifest, fields, input_manifests=expected_inputs
    ):
        return
    audit_outputs = (
        "common_protocol_audit.json",
        "bace_paper_artifact_audit.json",
        "cohort_parity_audit.json",
        "threshold_parity_audit.json",
        "figure3_coverage_vs_k.csv",
        "figure4_coverage_vs_threshold.csv",
        "table2_bace_k10.csv",
    )
    audit_proof = base / "common4_audit_proof"
    proof_marker = audit_proof / "common_protocol_audit.json"
    proof_manifest, proof_metadata = _reuse_proof_paths(audit_proof)
    if proof_manifest.is_file() or proof_metadata.is_file():
        _verify_reuse_proof(
            context=context,
            output=audit_proof,
            marker=proof_marker,
            marker_fields={"passed": True},
            command=command,
            input_manifests=expected_inputs,
        )
    else:
        _run_checked(command, cwd=context.project)
        if audit_proof.exists() or audit_proof.is_symlink():
            raise StageError("Partial BACE common4 audit proof exists")
        temporary = Path(
            tempfile.mkdtemp(
                prefix=".common4_audit_proof.", suffix=".tmp", dir=base
            )
        )
        published = False
        try:
            for name in audit_outputs:
                _copy_file_atomic(common / name, temporary / name)
            os.rename(temporary, audit_proof)
            published = True
            _fsync_directory(base)
        finally:
            if not published and temporary.exists():
                shutil.rmtree(temporary)
        _publish_reuse_proof(
            context=context,
            output=audit_proof,
            marker=proof_marker,
            marker_fields={"passed": True},
            command=command,
            input_manifests=expected_inputs,
        )
    for name in audit_outputs:
        source_output = common / name
        proof_output = audit_proof / name
        _assert_nonempty_file(source_output)
        _assert_nonempty_file(proof_output)
        if sha256_file(source_output) != sha256_file(proof_output):
            raise StageError(f"BACE common4 audit output changed after proof: {name}")
    audit = read_json(common / "common_protocol_audit.json")
    if audit.get("passed") is not True:
        raise StageError("BACE common4 protocol audit did not pass")
    for path in (
        common / "figure3_coverage_vs_k.csv",
        common / "figure4_coverage_vs_threshold.csv",
        common / "table2_bace_k10.csv",
    ):
        _assert_nonempty_file(path)
    write_sha256_manifest(
        base=base,
        items=[common, audit_proof],
        manifest=manifest,
        exclude=[manifest],
    )
    input_manifests = _verify_all_input_gates(inputs)
    input_after = input_manifests["primary"]
    _publish_sentinel(
        context=context,
        path=sentinel,
        manifest=manifest,
        manifest_root=base,
        input_digest_before=input_before,
        input_digest_after=input_after,
        input_manifests=input_manifests,
        payload={
            **fields,
            "dataset": "bace",
            "cf_mode": "strict_flip",
            "dependency_sentinels_verified": True,
        },
    )


class _ProfileStop(BaseException):
    """Internal non-scientific stop used after an atomic profile checkpoint."""


PROFILE_OBSERVATION_SCHEMA = "bace_profile_runtime_observations_v1"
PROFILE_PERFORMANCE_SCHEMA = "bace_structured_performance_v1"
PROFILE_RUN_IDS = (
    "uninterrupted_0_to_1000",
    "resume_path_0_to_post_checkpoint_kill",
    "resume_path_500_to_1000",
)
PROFILE_REQUIRED_RUNTIME_MEASUREMENTS = (
    "gpu",
    "process_cpu",
    "system_iowait",
    "process_io",
)
PROFILE_REQUIRED_FUNCTION_CATEGORIES = (
    "transition_reconstruction",
    "sqlite",
    "trace_serialization",
    "model_inference",
)
PROFILE_OPTIONAL_FUNCTION_CATEGORIES = (
    "graph_diff",
    "canonical_hash",
    "rdkit",
)
PROFILE_OPTIONAL_ABSENCE_REASONS = {
    "graph_diff": "optimized_generation_path_did_not_call_graph_diff",
    "canonical_hash": "optimized_generation_path_did_not_call_named_canonical_hash",
    "rdkit": "bace_generation_consumes_frozen_pyg_graphs_without_rdkit_conversion",
}
PROFILE_FUNCTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "graph_diff": (
        r"graph[_ ]?(?:diff|difference)",
        r"(?:diff|difference)[_ ]?graph",
    ),
    "transition_reconstruction": (
        r"transition.*(?:reconstruct|replay|lineage|resolve|restore)",
        r"(?:reconstruct|replay|lineage|resolve|restore).*transition",
        r"iter_candidate_lineage",
    ),
    "canonical_hash": (
        r"canonical",
        r"(?:stable|graph|official).*hash",
        r"sha256",
    ),
    "rdkit": (
        r"rdkit",
        r"(?:smiles|sanitize|molfrom|molto|chem\.)",
    ),
    "sqlite": (
        r"sqlite",
        r"transition_cache",
        r"(?:^|[.:])(?:execute|executemany|cursor)(?:$|[.:])",
    ),
    "trace_serialization": (
        r"trace.*(?:serial|write|dump|flush|chunk|manifest)",
        r"(?:serial|write|dump|flush).*trace",
        r"selected_action_trace",
    ),
    "model_inference": (
        r"(?:model|classifier).*(?:infer|predict|forward)",
        r"(?:infer|predict|forward).*(?:model|classifier)",
        r"(?:^|[.:])forward(?:$|[.:])",
    ),
}


def _not_observed(reason: str) -> dict[str, str]:
    return {"status": "NOT_OBSERVED", "reason": reason}


def _read_process_cpu(pid: int | None) -> tuple[dict[str, Any] | None, str | None]:
    if pid is None:
        return None, "child_pid_unavailable"
    path = Path(f"/proc/{pid}/stat")
    if not path.is_file():
        return None, "proc_process_stat_unavailable"
    try:
        value = path.read_text(encoding="utf-8")
        fields = value[value.rfind(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        if ticks_per_second <= 0:
            raise ValueError("SC_CLK_TCK must be positive")
        return {"cpu_seconds": ticks / ticks_per_second}, None
    except (IndexError, OSError, ValueError) as exc:
        return None, f"proc_process_stat_parse_failed:{type(exc).__name__}"


def _read_process_io(pid: int | None) -> tuple[dict[str, Any] | None, str | None]:
    if pid is None:
        return None, "child_pid_unavailable"
    path = Path(f"/proc/{pid}/io")
    if not path.is_file():
        return None, "proc_process_io_unavailable"
    try:
        values: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition(":")
            if separator:
                values[key.strip()] = int(raw.strip())
        return {
            "read_bytes": int(values["read_bytes"]),
            "write_bytes": int(values["write_bytes"]),
        }, None
    except (KeyError, OSError, ValueError) as exc:
        return None, f"proc_process_io_parse_failed:{type(exc).__name__}"


def _read_system_iowait() -> tuple[dict[str, Any] | None, str | None]:
    path = Path("/proc/stat")
    if not path.is_file():
        return None, "proc_system_stat_unavailable"
    try:
        cpu = next(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("cpu ")
        )
        counters = [int(value) for value in cpu.split()[1:]]
        if len(counters) < 5:
            raise ValueError("cpu aggregate lacks iowait")
        # guest/guest_nice are already included in user/nice on Linux.
        return {
            "total_jiffies": sum(counters[:8]),
            "iowait_jiffies": counters[4],
        }, None
    except (OSError, StopIteration, ValueError) as exc:
        return None, f"proc_system_stat_parse_failed:{type(exc).__name__}"


def _read_gpu_observation() -> tuple[list[dict[str, Any]] | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "nvidia_smi_unavailable"
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if result.returncode != 0:
        return None, f"nvidia_smi_failed_rc_{result.returncode}"
    rows: list[dict[str, Any]] = []
    visible_tokens = {
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    }
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            values = [value.strip() for value in line.split(",")]
            if len(values) != 5:
                raise ValueError("unexpected nvidia-smi column count")
            gpu_index = int(values[0])
            gpu_uuid = values[1]
            if visible_tokens and (
                str(gpu_index) not in visible_tokens and gpu_uuid not in visible_tokens
            ):
                continue
            rows.append(
                {
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "measurement_scope": (
                        "CUDA_VISIBLE_DEVICES"
                        if visible_tokens
                        else "all_nvidia_smi_visible_devices"
                    ),
                    "utilization_percent": float(values[2]),
                    "memory_used_mib": float(values[3]),
                    "memory_total_mib": float(values[4]),
                }
            )
    except ValueError as exc:
        return None, f"nvidia_smi_parse_failed:{type(exc).__name__}"
    if not rows:
        return None, (
            "nvidia_smi_returned_no_assigned_gpu_rows"
            if visible_tokens
            else "nvidia_smi_returned_no_gpu_rows"
        )
    return rows, None


class _ProfileObservationCollector:
    """Best-effort Linux/NVIDIA sampling with explicit missing-data states."""

    def __init__(
        self,
        *,
        pid: int | None,
        started_monotonic: float,
        metric_interval_seconds: float = 1.0,
    ) -> None:
        self.pid = pid
        self.started = started_monotonic
        self.metric_interval = float(metric_interval_seconds)
        self.next_metric_at = started_monotonic
        self.progress_samples: list[dict[str, Any]] = []
        self.cpu_samples: list[dict[str, Any]] = []
        self.io_samples: list[dict[str, Any]] = []
        self.iowait_samples: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, Any]] = []
        self.reasons: dict[str, str] = {}

    def observe(
        self, *, progress_step: int | None = None, force_metrics: bool = False
    ) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self.started)
        if progress_step is not None and progress_step >= 0:
            if (
                not self.progress_samples
                or int(self.progress_samples[-1]["completed_step"]) != progress_step
            ):
                self.progress_samples.append(
                    {
                        "elapsed_seconds": round(elapsed, 6),
                        "completed_step": int(progress_step),
                    }
                )
        if not force_metrics and now < self.next_metric_at:
            return
        self.next_metric_at = now + self.metric_interval
        for name, reader, destination in (
            ("process_cpu", lambda: _read_process_cpu(self.pid), self.cpu_samples),
            ("process_io", lambda: _read_process_io(self.pid), self.io_samples),
            ("system_iowait", _read_system_iowait, self.iowait_samples),
        ):
            value, reason = reader()
            if value is None:
                self.reasons.setdefault(name, str(reason or "observation_failed"))
            else:
                destination.append({"elapsed_seconds": round(elapsed, 6), **value})
        try:
            gpu, reason = _read_gpu_observation()
        except (OSError, subprocess.SubprocessError) as exc:
            gpu, reason = None, f"nvidia_smi_observation_failed:{type(exc).__name__}"
        if gpu is None:
            self.reasons.setdefault("gpu", str(reason or "observation_failed"))
        else:
            self.gpu_samples.append(
                {"elapsed_seconds": round(elapsed, 6), "devices": gpu}
            )

    @staticmethod
    def _delta_summary(
        samples: Sequence[Mapping[str, Any]], *, value_key: str, scale: float = 1.0
    ) -> dict[str, Any]:
        if len(samples) < 2:
            return _not_observed("at_least_two_samples_required")
        intervals: list[float] = []
        for before, after in zip(samples, samples[1:]):
            elapsed = float(after["elapsed_seconds"]) - float(before["elapsed_seconds"])
            delta = float(after[value_key]) - float(before[value_key])
            if elapsed > 0 and delta >= 0:
                intervals.append(delta / elapsed * scale)
        if not intervals:
            return _not_observed("no_valid_monotonic_sample_interval")
        return {
            "status": "OBSERVED",
            "interval_count": len(intervals),
            "mean": sum(intervals) / len(intervals),
            "peak": max(intervals),
        }

    def finish(self, *, elapsed_seconds: float) -> dict[str, Any]:
        progress: dict[str, Any]
        if self.progress_samples:
            first = self.progress_samples[0]
            last = self.progress_samples[-1]
            step_delta = int(last["completed_step"]) - int(first["completed_step"])
            time_delta = float(last["elapsed_seconds"]) - float(first["elapsed_seconds"])
            step_intervals: list[dict[str, Any]] = []
            for before, after in zip(
                self.progress_samples, self.progress_samples[1:]
            ):
                interval_steps = int(after["completed_step"]) - int(
                    before["completed_step"]
                )
                interval_seconds = float(after["elapsed_seconds"]) - float(
                    before["elapsed_seconds"]
                )
                if interval_steps > 0 and interval_seconds > 0:
                    step_intervals.append(
                        {
                            "from_step": int(before["completed_step"]),
                            "to_step": int(after["completed_step"]),
                            "observed_step_delta": interval_steps,
                            "observed_elapsed_seconds": interval_seconds,
                            "seconds_per_step": interval_seconds / interval_steps,
                            "steps_per_second": interval_steps / interval_seconds,
                        }
                    )
            per_step = (
                {
                    "status": "OBSERVED",
                    "observed_step_delta": step_delta,
                    "observed_elapsed_seconds": time_delta,
                    "seconds_per_step": time_delta / step_delta,
                    "steps_per_second": step_delta / time_delta,
                    "interval_count": len(step_intervals),
                    "intervals": step_intervals,
                }
                if step_delta > 0 and time_delta > 0 and step_intervals
                else _not_observed("progress_did_not_span_two_distinct_timed_steps")
            )
            progress = {
                "status": "OBSERVED",
                "sample_count": len(self.progress_samples),
                "samples": self.progress_samples,
                "per_step": per_step,
            }
        else:
            progress = {
                **_not_observed("progress_json_was_never_observed"),
                "sample_count": 0,
                "samples": [],
                "per_step": _not_observed("progress_json_was_never_observed"),
            }

        cpu = (
            {
                "status": "OBSERVED",
                "sample_count": len(self.cpu_samples),
                "samples": self.cpu_samples,
                "utilization_percent": self._delta_summary(
                    self.cpu_samples, value_key="cpu_seconds", scale=100.0
                ),
            }
            if self.cpu_samples
            else {
                **_not_observed(self.reasons.get("process_cpu", "no_samples")),
                "sample_count": 0,
                "samples": [],
                "utilization_percent": _not_observed("no_samples"),
            }
        )
        if self.io_samples:
            first_io = self.io_samples[0]
            last_io = self.io_samples[-1]
            io_delta: dict[str, Any] = (
                {
                    "status": "OBSERVED",
                    "read_bytes": int(last_io["read_bytes"])
                    - int(first_io["read_bytes"]),
                    "write_bytes": int(last_io["write_bytes"])
                    - int(first_io["write_bytes"]),
                }
                if len(self.io_samples) >= 2
                else _not_observed("at_least_two_samples_required")
            )
            process_io = {
                "status": "OBSERVED",
                "sample_count": len(self.io_samples),
                "samples": self.io_samples,
                "byte_delta": io_delta,
            }
        else:
            process_io = {
                **_not_observed(self.reasons.get("process_io", "no_samples")),
                "sample_count": 0,
                "samples": [],
                "byte_delta": _not_observed("no_samples"),
            }

        iowait = (
            {
                "status": "OBSERVED",
                "sample_count": len(self.iowait_samples),
                "samples": self.iowait_samples,
                "percent": _not_observed("at_least_two_samples_required"),
            }
            if self.iowait_samples
            else {
                **_not_observed(self.reasons.get("system_iowait", "no_samples")),
                "sample_count": 0,
                "samples": [],
                "percent": _not_observed("no_samples"),
            }
        )
        # I/O wait is a share of total CPU jiffies, not a per-second counter.
        if len(self.iowait_samples) >= 2:
            percentages: list[float] = []
            for before, after in zip(self.iowait_samples, self.iowait_samples[1:]):
                total_delta = int(after["total_jiffies"]) - int(before["total_jiffies"])
                wait_delta = int(after["iowait_jiffies"]) - int(before["iowait_jiffies"])
                if total_delta > 0 and wait_delta >= 0:
                    percentages.append(wait_delta / total_delta * 100.0)
            iowait["percent"] = (
                {
                    "status": "OBSERVED",
                    "interval_count": len(percentages),
                    "mean": sum(percentages) / len(percentages),
                    "peak": max(percentages),
                }
                if percentages
                else _not_observed("no_valid_system_jiffy_interval")
            )

        if self.gpu_samples:
            devices: dict[int, list[Mapping[str, Any]]] = {}
            for sample in self.gpu_samples:
                for device in sample["devices"]:
                    devices.setdefault(int(device["gpu_index"]), []).append(device)
            summary = {
                str(index): {
                    "sample_count": len(rows),
                    "mean_utilization_percent": sum(
                        float(row["utilization_percent"]) for row in rows
                    )
                    / len(rows),
                    "peak_utilization_percent": max(
                        float(row["utilization_percent"]) for row in rows
                    ),
                    "peak_memory_used_mib": max(
                        float(row["memory_used_mib"]) for row in rows
                    ),
                    "memory_total_mib": max(
                        float(row["memory_total_mib"]) for row in rows
                    ),
                }
                for index, rows in sorted(devices.items())
            }
            gpu = {
                "status": "OBSERVED",
                "sample_count": len(self.gpu_samples),
                "samples": self.gpu_samples,
                "devices": summary,
            }
        else:
            gpu = {
                **_not_observed(self.reasons.get("gpu", "no_samples")),
                "sample_count": 0,
                "samples": [],
                "devices": {},
            }
        return {
            "schema_version": PROFILE_OBSERVATION_SCHEMA,
            "sampling_interval_seconds": self.metric_interval,
            "elapsed_seconds": float(elapsed_seconds),
            "progress": progress,
            "gpu": gpu,
            "process_cpu": cpu,
            "system_iowait": iowait,
            "process_io": process_io,
        }


def _profile_exec(argv: Sequence[str]) -> int:
    """Run a Python script under cProfile and durably dump on SIGTERM.

    ``BaseException`` deliberately bypasses the generation runtime's
    ``except Exception`` failure writer: the last *published* checkpoint is the
    only trusted profile product, and the interrupted active state is never
    promoted.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise StageError("_profile-exec requires a Python script command")
    script = Path(command[0]).expanduser().resolve()
    _assert_nonempty_file(script)
    profile_output = Path(args.profile_output).expanduser().resolve()
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profiler = __import__("cProfile").Profile()
    previous = signal.getsignal(signal.SIGTERM)

    def stop(_signum: int, _frame: Any) -> None:
        raise _ProfileStop()

    signal.signal(signal.SIGTERM, stop)
    status = 0
    try:
        sys.argv = [str(script), *command[1:]]
        profiler.enable()
        try:
            runpy.run_path(str(script), run_name="__main__")
        except _ProfileStop:
            status = 143
        except SystemExit as exc:
            status = int(exc.code or 0)
    finally:
        profiler.disable()
        profiler.dump_stats(str(profile_output))
        with profile_output.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(profile_output.parent)
        signal.signal(signal.SIGTERM, previous)
    return status


def _replace_option(command: list[str], option: str, value: Path | str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise StageError(f"Scientific command omitted expected option {option}") from exc
    if index + 1 >= len(command):
        raise StageError(f"Scientific command has no value for {option}")
    command[index + 1] = str(value)


_PROFILE_OPERATIONAL_ROOT_OPTIONS = (
    "--output-dir",
    "--trace-output-dir",
    "--graph-state-dir",
    "--checkpoint-root",
    "--checkpoint-mirror-root",
    "--storage-guard-root",
)


def _profile_parity_normalized_argv(command: Sequence[str]) -> tuple[str, ...]:
    """Normalize only profile transport roots and the ``--resume`` flag.

    This accepts both the concrete command emitted by
    :func:`_profile_generation_command` and the canonical ``--name=<json>``
    argv persisted in generation checkpoints.  All scientific parameters and
    every non-operational path remain byte-for-byte significant.
    """

    values = [str(value) for value in command]
    if "_profile-exec" in values:
        profile_index = values.index("_profile-exec")
        try:
            delimiter = values.index("--", profile_index + 1)
        except ValueError as exc:
            raise StageError("Profile wrapper command lacks -- delimiter") from exc
        if not values or delimiter + 1 >= len(values):
            raise StageError("Profile wrapper command lacks a scientific command")
        # The wrapper uses the same interpreter as the scientific command and
        # intentionally omits only scientific[0] after the delimiter.
        values = [values[0], *values[delimiter + 1 :]]

    normalized: list[str] = []
    index = 0
    while index < len(values):
        token = values[index]
        if token == "--resume" or token.startswith("--resume="):
            index += 1
            continue
        matched = False
        for option in _PROFILE_OPERATIONAL_ROOT_OPTIONS:
            placeholder = f"<PROFILE_OPERATIONAL_ROOT:{option[2:]}>"
            if token == option:
                if index + 1 >= len(values):
                    raise StageError(f"Profile command has no value for {option}")
                normalized.extend([option, placeholder])
                index += 2
                matched = True
                break
            if token.startswith(option + "="):
                # Canonical scientific argv JSON-encodes every parsed value.
                normalized.append(
                    f"{option}={json.dumps(placeholder, separators=(',', ':'))}"
                )
                index += 1
                matched = True
                break
        if not matched:
            normalized.append(token)
            index += 1
    if not normalized:
        raise StageError("Cannot normalize an empty profile scientific command")
    return tuple(normalized)


def _profile_command_evidence(command: Sequence[str]) -> dict[str, Any]:
    raw = tuple(str(value) for value in command)
    normalized = _profile_parity_normalized_argv(raw)

    def digest(schema: str, argv: tuple[str, ...]) -> str:
        encoded = json.dumps(
            {"schema_version": schema, "argv": list(argv)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    return {
        "raw_scientific_argv": list(raw),
        "raw_scientific_command_sha256": digest(
            "bace_profile_raw_scientific_command_v1", raw
        ),
        "parity_normalized_scientific_argv": list(normalized),
        "parity_normalized_scientific_command_sha256": digest(
            "bace_profile_parity_scientific_command_v1", normalized
        ),
        "parity_normalization": {
            "ignored_flag": "--resume",
            "operational_root_options": list(_PROFILE_OPERATIONAL_ROOT_OPTIONS),
            "all_other_arguments_significant": True,
        },
    }


def _profile_generation_command(
    context: Context,
    *,
    label: str,
    resume: bool,
    profile_output: Path | None,
) -> tuple[list[str], dict[str, Path]]:
    fast = context.fast / "profile/bace_comrecgc" / label
    persistent = context.persistent / "outputs/profile/bace_comrecgc" / label
    paths = {
        "output": fast / "generation",
        "trace": persistent / "trace",
        "graph_state": fast / "graph_state",
        "checkpoint": fast / "generation_checkpoints",
        "mirror": persistent / "generation_checkpoint_mirror",
        "guard": fast,
    }
    scientific = bace_generation_command(context, resume=resume)
    for option, key in (
        ("--output-dir", "output"),
        ("--trace-output-dir", "trace"),
        ("--graph-state-dir", "graph_state"),
        ("--checkpoint-root", "checkpoint"),
        ("--checkpoint-mirror-root", "mirror"),
        ("--storage-guard-root", "guard"),
    ):
        _replace_option(scientific, option, paths[key])
    if profile_output is None:
        return scientific, paths
    profiled = [
        str(context.python),
        str(Path(__file__).resolve()),
        "_profile-exec",
        "--profile-output",
        str(profile_output),
        "--",
        *scientific[1:],
    ]
    return profiled, paths


def _run_profile_until_checkpoint(
    context: Context,
    *,
    command: Sequence[str],
    mirror_root: Path,
    target_step: int,
    termination_signal: int = signal.SIGTERM,
    progress_path: Path | None = None,
    stop_after_progress_step: int | None = None,
    resolved_config_source: Path | None = None,
    resolved_config_destination: Path | None = None,
) -> dict[str, Any]:
    environment = _sanitized_inherited_environment()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(context.project) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        cwd=context.project,
        env=environment,
        start_new_session=True,
    )
    observations = _ProfileObservationCollector(
        pid=getattr(process, "pid", None), started_monotonic=started
    )
    stop_sent = False
    selected_digest: str | None = None
    checkpoint_validated = False
    observed_process_step = -1
    try:
        with _forward_signals(process):
            while process.poll() is None:
                if (
                    resolved_config_source is not None
                    and resolved_config_destination is not None
                    and resolved_config_source.is_file()
                ):
                    if resolved_config_destination.is_file():
                        if sha256_file(resolved_config_source) != sha256_file(
                            resolved_config_destination
                        ):
                            raise StageError(
                                "Profile resolved_config changed after publication"
                            )
                    else:
                        _copy_file_atomic(
                            resolved_config_source, resolved_config_destination
                        )
                latest_path = mirror_root / "LATEST"
                if latest_path.is_file() and not latest_path.is_symlink():
                    try:
                        latest = read_json(latest_path)
                        completed = int(latest.get("completed_step", -1))
                    except (StageError, TypeError, ValueError):
                        completed = -1
                    if not stop_sent and completed >= int(target_step):
                        if completed != int(target_step):
                            raise StageError(
                                "Profile skipped the requested checkpoint before stop: "
                                f"target={target_step}, latest={completed}"
                            )
                        module = _checkpoint_module(context)
                        validation = module.validate_generation_checkpoint(
                            mirror_root, expected_completed_step=target_step
                        )
                        _validate_mirrored_checkpoint(
                            module, validation.checkpoint_dir
                        )
                        selected_digest = str(validation.checkpoint_digest)
                        checkpoint_validated = True
                if progress_path is not None and progress_path.is_file():
                    try:
                        progress = read_json(progress_path)
                        observed_process_step = max(
                            observed_process_step,
                            int(
                                progress.get(
                                    "completed_step", progress.get("current_step", -1)
                                )
                            ),
                        )
                    except (StageError, TypeError, ValueError):
                        pass
                observations.observe(
                    progress_step=(
                        observed_process_step if observed_process_step >= 0 else None
                    )
                )
                ready_to_stop = checkpoint_validated and (
                    stop_after_progress_step is None
                    or observed_process_step >= int(stop_after_progress_step)
                )
                if not stop_sent and ready_to_stop:
                    _signal_child_process_group(process, termination_signal)
                    stop_sent = True
                time.sleep(0.1)
    except BaseException:
        _terminate_child_process_group(process)
        raise
    elapsed = time.monotonic() - started
    observations.observe(
        progress_step=(observed_process_step if observed_process_step >= 0 else None),
        force_metrics=True,
    )
    performance_observations = observations.finish(elapsed_seconds=elapsed)
    expected_returncode = 143 if termination_signal == signal.SIGTERM else -termination_signal
    if (
        not stop_sent
        or process.returncode != expected_returncode
        or selected_digest is None
    ):
        raise StageError(
            "Profile generation did not stop at the requested atomic checkpoint: "
            f"target={target_step}, rc={process.returncode}, stop_sent={stop_sent}"
        )
    if (
        resolved_config_source is not None
        and resolved_config_destination is not None
        and not resolved_config_destination.is_file()
    ):
        raise StageError("Profile run never published persistent resolved_config")
    resolved_config = (
        {
            "source": str(resolved_config_source),
            "persistent_copy": str(resolved_config_destination),
            "sha256": sha256_file(resolved_config_destination),
        }
        if resolved_config_source is not None
        and resolved_config_destination is not None
        else None
    )
    return {
        "target_checkpoint_step": int(target_step),
        "checkpoint_digest": selected_digest,
        "checkpoint_validated_before_signal": checkpoint_validated,
        "observed_process_step_at_signal": observed_process_step,
        "required_post_checkpoint_progress_step": stop_after_progress_step,
        "elapsed_seconds": elapsed,
        "termination_signal": (
            "SIGTERM_after_persistent_checkpoint_validation"
            if termination_signal == signal.SIGTERM
            else "SIGKILL_after_persistent_checkpoint_validation"
        ),
        "trusted_state": "published_checkpoint_only",
        "resolved_config": resolved_config,
        "performance_observations": performance_observations,
    }


def _restore_checkpoint_roots(
    context: Context, *, fast_root: Path, mirror_root: Path
) -> int:
    module = _checkpoint_module(context)
    mirror_validations, _ignored = _select_fully_mirrored_checkpoints(
        module, mirror_root, keep_last=2
    )
    fast_root.mkdir(parents=True, exist_ok=True)
    for source_validation in mirror_validations:
        source = source_validation.checkpoint_dir
        destination = fast_root / source.name
        if destination.exists():
            raise StageError(f"Profile restore collision: {destination}")
        _copy_tree_atomic(source, destination)
        destination_validation = module.validate_generation_checkpoint(destination)
        if destination_validation.checkpoint_digest != source_validation.checkpoint_digest:
            raise StageError(f"Profile checkpoint restore mismatch: {destination}")
    return int(module.validate_generation_checkpoint(fast_root).completed_step)


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        tensor = value.detach().cpu().contiguous()
        payload = tensor.numpy().tobytes()
        return {
            "tensor_dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if hasattr(value, "edge_index") and hasattr(value, "num_nodes"):
        from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256

        return {"graph_sha256": stable_untyped_graph_sha256(value)}
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    raise StageError(f"Unsupported checkpoint value in logical audit: {type(value)!r}")


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sqlite_logical_digest(path: Path) -> str:
    digest = hashlib.sha256()
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise StageError(f"Profile checkpoint SQLite integrity failed: {path}")
        for line in connection.iterdump():
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
    finally:
        connection.close()
    return digest.hexdigest()


def _trace_checkpoint_logical_audit(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return recovery-stable trace rows and counters for parity comparison."""

    trace_rows = [
        {
            "index": int(row.get("index", -1)),
            "path": str(row.get("path") or ""),
            "row_count": int(row.get("row_count", -1)),
            "bytes": int(row.get("bytes", -1)),
            "sha256": str(row.get("sha256") or ""),
        }
        for row in trace.get("chunks") or ()
    ]
    trace_counters = {
        key: int(trace.get(key, -1))
        for key in (
            "move_index",
            "enumerated_transition_count",
            "selected_transition_count",
            "teleport_count",
            "transition_cache_hit_count",
            "transition_cache_miss_count",
        )
    }
    pending_events = list(trace.get("pending_events") or ())
    return {
        # ``materialization`` may legitimately change from ``atomic_write`` to
        # ``adopt_existing_identical`` after a crash.  Exact trace state is the
        # completed chunk prefix *plus* the pending rows, so both participate in
        # parity while only the operational materialization label is ignored.
        "trace_rows_logical_sha256": _stable_digest(trace_rows),
        "trace_chunk_count": len(trace_rows),
        "pending_event_count": len(pending_events),
        "pending_events_logical_sha256": _stable_digest(pending_events),
        "trace_counters": trace_counters,
        "trace_counters_sha256": _stable_digest(trace_counters),
    }


def _logical_checkpoint_audit(context: Context, checkpoint: Path) -> dict[str, Any]:
    module = _checkpoint_module(context)
    loaded = module.load_generation_checkpoint(checkpoint, expected_completed_step=1000)
    algorithm = loaded.algorithm_state
    official = algorithm.get("official_state") or {}
    command_evidence = _profile_command_evidence(
        loaded.validation.scientific_argv
    )
    command_evidence["raw_scientific_argv"] = list(
        loaded.validation.scientific_argv
    )
    command_evidence["raw_scientific_command_sha256"] = str(
        loaded.validation.command_sha256
    )
    parity_provenance = {
        key: value
        for key, value in loaded.validation.provenance_fingerprints.items()
        if key != "scientific_command_sha256"
    }
    sequence = {
        "loop_state": algorithm.get("loop_state"),
        "traversed_hashes": official.get("traversed_hashes"),
        "counterfactual_candidates": official.get("counterfactual_candidates"),
    }
    return {
        "completed_step": loaded.completed_step,
        # Preserve the exact per-run command/provenance for audit.  Profile
        # parity intentionally compares the separately normalized evidence
        # below, because the two runs use different isolated output roots.
        "provenance_fingerprints": loaded.validation.provenance_fingerprints,
        **command_evidence,
        "parity_provenance_fingerprints": parity_provenance,
        "algorithm_logical_sha256": _stable_digest(algorithm),
        **_trace_checkpoint_logical_audit(loaded.trace_state),
        "rng_logical_sha256": _stable_digest(loaded.rng_state),
        "sqlite_logical_sha256": _sqlite_logical_digest(loaded.sqlite_snapshot_path),
        "sequence_sha256": _stable_digest(sequence),
    }


def _unobserved_function_profile(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "bace_pstats_function_categories_v1",
        "status": "NOT_OBSERVED",
        "reason": reason,
        "source_profiles": [],
        "aggregation_policy": (
            "independent_regex_matches_over_filename_and_function_name;"
            "categories_may_overlap"
        ),
        "categories": {
            category: {
                **_not_observed(reason),
                "patterns": list(patterns),
                "matched_function_count": 0,
                "calls": "NOT_OBSERVED",
                "primitive_calls": "NOT_OBSERVED",
                "total_seconds": "NOT_OBSERVED",
                "cumulative_seconds": "NOT_OBSERVED",
                "functions": [],
            }
            for category, patterns in PROFILE_FUNCTION_PATTERNS.items()
        },
    }


def _aggregate_pstats(profile_paths: Sequence[Path]) -> dict[str, Any]:
    paths = [Path(path).expanduser().resolve() for path in profile_paths]
    if not paths:
        return _unobserved_function_profile("no_cprofile_was_requested")
    for path in paths:
        _assert_nonempty_file(path)
    try:
        stats = pstats.Stats(str(paths[0]))
        for path in paths[1:]:
            stats.add(str(path))
    except Exception as exc:
        raise StageError(
            f"Unable to parse cProfile evidence: {type(exc).__name__}"
        ) from exc

    categories: dict[str, dict[str, Any]] = {}
    for category, patterns in PROFILE_FUNCTION_PATTERNS.items():
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        functions: list[dict[str, Any]] = []
        for key, values in stats.stats.items():
            if not isinstance(key, tuple) or len(key) != 3 or len(values) < 4:
                continue
            filename, line_number, function_name = key
            identity = f"{filename}:{line_number}:{function_name}"
            if not any(pattern.search(identity) for pattern in compiled):
                continue
            primitive_calls, calls, total_seconds, cumulative_seconds = values[:4]
            functions.append(
                {
                    "function": identity,
                    "calls": int(calls),
                    "primitive_calls": int(primitive_calls),
                    "total_seconds": float(total_seconds),
                    "cumulative_seconds": float(cumulative_seconds),
                }
            )
        functions.sort(
            key=lambda row: (-float(row["cumulative_seconds"]), str(row["function"]))
        )
        if functions:
            categories[category] = {
                "status": "OBSERVED",
                "patterns": list(patterns),
                "matched_function_count": len(functions),
                "calls": sum(int(row["calls"]) for row in functions),
                "primitive_calls": sum(
                    int(row["primitive_calls"]) for row in functions
                ),
                "total_seconds": sum(
                    float(row["total_seconds"]) for row in functions
                ),
                "cumulative_seconds": sum(
                    float(row["cumulative_seconds"]) for row in functions
                ),
                "functions": functions,
            }
        else:
            absence_reason = PROFILE_OPTIONAL_ABSENCE_REASONS.get(
                category,
                "no_profiled_function_matched_required_category_patterns",
            )
            categories[category] = {
                **_not_observed(absence_reason),
                "patterns": list(patterns),
                "matched_function_count": 0,
                "calls": 0,
                "primitive_calls": 0,
                "total_seconds": 0.0,
                "cumulative_seconds": 0.0,
                "functions": [],
            }
    return {
        "schema_version": "bace_pstats_function_categories_v1",
        "status": "OBSERVED",
        "source_profiles": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "aggregation_policy": (
            "independent_regex_matches_over_filename_and_function_name;"
            "categories_may_overlap"
        ),
        "categories": categories,
    }


def _validate_structured_profile_evidence(
    report: Mapping[str, Any], *, evidence_root: Path | None = None
) -> None:
    performance = report.get("structured_performance")
    if not isinstance(performance, Mapping):
        raise StageError("BACE profile report lacks structured_performance")
    if performance.get("schema_version") != PROFILE_PERFORMANCE_SCHEMA:
        raise StageError("BACE profile structured performance schema mismatch")
    expected_patterns = {
        key: list(value) for key, value in PROFILE_FUNCTION_PATTERNS.items()
    }
    if performance.get("function_category_patterns") != expected_patterns:
        raise StageError("BACE profile function category patterns changed")
    if performance.get("required_runtime_measurements") != list(
        PROFILE_REQUIRED_RUNTIME_MEASUREMENTS
    ):
        raise StageError("BACE profile required runtime measurements changed")
    if performance.get("required_combined_function_categories") != list(
        PROFILE_REQUIRED_FUNCTION_CATEGORIES
    ):
        raise StageError("BACE profile required function categories changed")
    if performance.get("optional_function_categories") != list(
        PROFILE_OPTIONAL_FUNCTION_CATEGORIES
    ):
        raise StageError("BACE profile optional function categories changed")
    if performance.get("optional_function_absence_reasons") != (
        PROFILE_OPTIONAL_ABSENCE_REASONS
    ):
        raise StageError("BACE profile optional absence reasons changed")
    runs = performance.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(PROFILE_RUN_IDS):
        raise StageError("BACE profile must contain exactly three structured runs")
    for run_id in PROFILE_RUN_IDS:
        run = runs.get(run_id)
        if not isinstance(run, Mapping):
            raise StageError(f"BACE profile run is not an object: {run_id}")
        observations = run.get("runtime_observations")
        if not isinstance(observations, Mapping):
            raise StageError(f"BACE profile run lacks observations: {run_id}")
        if observations.get("schema_version") != PROFILE_OBSERVATION_SCHEMA:
            raise StageError(f"BACE profile observation schema mismatch: {run_id}")
        for field in (
            "progress",
            "gpu",
            "process_cpu",
            "system_iowait",
            "process_io",
        ):
            measurement = observations.get(field)
            if not isinstance(measurement, Mapping) or measurement.get("status") not in {
                "OBSERVED",
                "NOT_OBSERVED",
            }:
                raise StageError(
                    f"BACE profile observation lacks explicit status: {run_id}.{field}"
                )
        progress = observations["progress"]
        if progress.get("status") != "OBSERVED" or not progress.get("samples"):
            raise StageError(f"BACE profile did not observe progress: {run_id}")
        per_step = progress.get("per_step")
        if not isinstance(per_step, Mapping) or per_step.get("status") != "OBSERVED":
            raise StageError(f"BACE profile did not observe per-step timing: {run_id}")
        for field in PROFILE_REQUIRED_RUNTIME_MEASUREMENTS:
            measurement = observations[field]
            if measurement.get("status") != "OBSERVED" or not measurement.get(
                "samples"
            ):
                raise StageError(
                    f"BACE profile required resource was not observed: "
                    f"{run_id}.{field}"
                )
        resource_summaries = {
            "process_cpu": "utilization_percent",
            "system_iowait": "percent",
            "process_io": "byte_delta",
        }
        for field, summary in resource_summaries.items():
            aggregation = observations[field].get(summary)
            if not isinstance(aggregation, Mapping) or aggregation.get(
                "status"
            ) != "OBSERVED":
                raise StageError(
                    f"BACE profile required resource summary was not observed: "
                    f"{run_id}.{field}.{summary}"
                )
        if not observations["gpu"].get("devices"):
            raise StageError(f"BACE profile observed no assigned GPU: {run_id}")

        function_profile = run.get("function_profile")
        if not isinstance(function_profile, Mapping):
            raise StageError(f"BACE profile lacks pstats evidence: {run_id}")
        if function_profile.get("schema_version") != (
            "bace_pstats_function_categories_v1"
        ):
            raise StageError(f"BACE profile pstats schema mismatch: {run_id}")
        if function_profile.get("status") not in {"OBSERVED", "NOT_OBSERVED"}:
            raise StageError(f"BACE profile pstats status is invalid: {run_id}")
        source_profiles = function_profile.get("source_profiles")
        if not isinstance(source_profiles, list):
            raise StageError(f"BACE profile pstats sources are invalid: {run_id}")
        if function_profile.get("status") == "OBSERVED" and not source_profiles:
            raise StageError(f"BACE profile pstats sources are empty: {run_id}")
        if evidence_root is not None:
            for source in source_profiles:
                if not isinstance(source, Mapping):
                    raise StageError(f"BACE profile pstats source is invalid: {run_id}")
                path = Path(str(source.get("path") or "")).expanduser()
                if not path.is_absolute() or not _is_within(path, evidence_root):
                    raise StageError(f"BACE profile pstats source escapes root: {path}")
                _assert_nonempty_file(path)
                if source.get("sha256") != sha256_file(path):
                    raise StageError(f"BACE profile pstats source digest mismatch: {path}")
        categories = function_profile.get("categories")
        if not isinstance(categories, Mapping) or set(categories) != set(
            PROFILE_FUNCTION_PATTERNS
        ):
            raise StageError(f"BACE profile pstats categories are incomplete: {run_id}")
        for category, aggregation in categories.items():
            if not isinstance(aggregation, Mapping) or aggregation.get("status") not in {
                "OBSERVED",
                "NOT_OBSERVED",
            }:
                raise StageError(
                    f"BACE profile pstats category lacks status: {run_id}.{category}"
                )
            for field in (
                "calls",
                "total_seconds",
                "cumulative_seconds",
            ):
                if field not in aggregation:
                    raise StageError(
                        f"BACE profile pstats category lacks {field}: "
                        f"{run_id}.{category}"
                    )
            if aggregation.get("status") == "OBSERVED":
                if not isinstance(aggregation.get("calls"), int) or not isinstance(
                    aggregation.get("total_seconds"), (int, float)
                ) or not isinstance(
                    aggregation.get("cumulative_seconds"), (int, float)
                ):
                    raise StageError(
                        f"BACE profile observed pstats values are not numeric: "
                        f"{run_id}.{category}"
                    )
    crash_run = runs["resume_path_0_to_post_checkpoint_kill"]
    try:
        trusted_checkpoint_step = int(crash_run.get("trusted_checkpoint_step", -1))
        process_step_at_kill = int(crash_run.get("process_step_at_kill", -1))
        recorded_stop_step = int(crash_run.get("stop_step", -1))
    except (TypeError, ValueError) as exc:
        raise StageError("SIGKILL profile run has invalid step evidence") from exc
    if (
        trusted_checkpoint_step != 500
        or process_step_at_kill < 525
        or recorded_stop_step != process_step_at_kill
    ):
        raise StageError(
            "SIGKILL profile must distinguish checkpoint 500 from the later kill step"
        )
    if crash_run["function_profile"].get("status") != "NOT_OBSERVED":
        raise StageError("SIGKILL profile run must explicitly report unobserved pstats")
    for run_id in ("uninterrupted_0_to_1000", "resume_path_500_to_1000"):
        if runs[run_id]["function_profile"].get("status") != "OBSERVED":
            raise StageError(f"Expected durable cProfile evidence for {run_id}")
    combined = performance.get("combined_observed_function_profile")
    if not isinstance(combined, Mapping) or combined.get("status") != "OBSERVED":
        raise StageError("BACE profile lacks combined observed pstats evidence")
    if combined.get("schema_version") != "bace_pstats_function_categories_v1":
        raise StageError("BACE combined pstats schema mismatch")
    combined_categories = combined.get("categories")
    if not isinstance(combined_categories, Mapping) or set(combined_categories) != set(
        PROFILE_FUNCTION_PATTERNS
    ):
        raise StageError("BACE combined pstats categories are incomplete")
    for category in PROFILE_REQUIRED_FUNCTION_CATEGORIES:
        aggregation = combined_categories[category]
        try:
            matched_function_count = int(aggregation.get("matched_function_count", 0))
            calls = int(aggregation.get("calls", 0))
        except (AttributeError, TypeError, ValueError) as exc:
            raise StageError(
                "BACE combined pstats required category has invalid counters: "
                f"{category}"
            ) from exc
        if (
            not isinstance(aggregation, Mapping)
            or aggregation.get("status") != "OBSERVED"
            or matched_function_count <= 0
            or calls <= 0
        ):
            raise StageError(
                "BACE combined pstats required category was not observed: "
                f"{category}"
            )
    for category in PROFILE_OPTIONAL_FUNCTION_CATEGORIES:
        aggregation = combined_categories[category]
        if aggregation.get("status") == "NOT_OBSERVED" and not str(
            aggregation.get("reason") or ""
        ).strip():
            raise StageError(
                "BACE combined pstats optional absence lacks an audit reason: "
                f"{category}"
            )
    combined_sources = combined.get("source_profiles")
    if not isinstance(combined_sources, list) or len(combined_sources) != 2:
        raise StageError("BACE combined pstats must reference two durable profiles")
    if evidence_root is not None:
        for source in combined_sources:
            if not isinstance(source, Mapping):
                raise StageError("BACE combined pstats source is invalid")
            path = Path(str(source.get("path") or "")).expanduser()
            if not path.is_absolute() or not _is_within(path, evidence_root):
                raise StageError(f"BACE combined pstats source escapes root: {path}")
            _assert_nonempty_file(path)
            if source.get("sha256") != sha256_file(path):
                raise StageError(f"BACE combined pstats source digest mismatch: {path}")


def _require_bace_profile_smoke_gate(
    context: Context,
    *,
    input_manifests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report_root = context.persistent / "outputs/profile"
    sentinel = report_root / "BACE_PROFILE_SMOKE_PASS.json"
    if not sentinel.is_file() or sentinel.is_symlink():
        raise StageError(f"Missing BACE_PROFILE_SMOKE_PASS gate: {sentinel}")
    if input_manifests is None:
        input_manifests = _input_manifest_digests(
            _all_input_gates(context, "bace_preserved")
        )
    expected_inputs = dict(input_manifests)
    payload = read_json(sentinel)
    expected = {
        "status": "PASS",
        "checkpoint_parity": True,
        "abrupt_kill_test_passed": True,
        "formal_generation_launched": False,
        "structured_performance_schema_version": PROFILE_PERFORMANCE_SCHEMA,
        "structured_performance_run_count": 3,
    }
    failures = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    lineage_expected = {
        "run_id": RUN_ID,
        "input_manifest_sha256_before": expected_inputs["primary"],
        "input_manifest_sha256_after": expected_inputs["primary"],
        "input_manifests_sha256": expected_inputs,
    }
    failures.update(
        {
            key: (value, payload.get(key))
            for key, value in lineage_expected.items()
            if payload.get(key) != value
        }
    )
    if failures:
        raise StageError(f"BACE profile smoke sentinel failed: {failures}")
    code_lineage = _current_code_lineage(context)
    for key, value in code_lineage.items():
        if key == "project_commit":
            if not re.fullmatch(r"[0-9a-f]{40}", str(payload.get(key) or "")):
                raise StageError("BACE profile smoke has invalid project commit")
            continue
        if payload.get(key) != value:
            raise StageError(f"BACE profile smoke has stale code lineage: {key}")
    for field, expected_path in (
        ("report_json", report_root / "bace_comrecgc_profile.json"),
        ("report_text", report_root / "bace_comrecgc_profile.txt"),
    ):
        recorded = Path(str(payload.get(field) or "")).expanduser()
        if recorded != expected_path or not _is_within(recorded, report_root):
            raise StageError(f"BACE profile sentinel has unsafe {field}: {recorded}")
        _assert_nonempty_file(recorded)
        if payload.get(f"{field}_sha256") != sha256_file(recorded):
            raise StageError(f"BACE profile {field} digest mismatch")
    report = read_json(report_root / "bace_comrecgc_profile.json")
    for key, value in expected.items():
        if report.get(key) != value:
            raise StageError(f"BACE profile report fails {key}: {report.get(key)!r}")
    for key, value in code_lineage.items():
        if key == "project_commit":
            if not re.fullmatch(r"[0-9a-f]{40}", str(report.get(key) or "")):
                raise StageError("BACE profile report has invalid project commit")
            continue
        if report.get(key) != value:
            raise StageError(f"BACE profile report has stale code lineage: {key}")
    if report.get("input_manifests_sha256") != expected_inputs:
        raise StageError("BACE profile report belongs to different input manifests")
    _validate_structured_profile_evidence(report, evidence_root=report_root)
    return payload


def _profile_text(profile_paths: Sequence[Path]) -> str:
    output = io.StringIO()
    for path in profile_paths:
        _assert_nonempty_file(path)
        output.write(f"\n===== {path.name} =====\n")
        (
            pstats.Stats(str(path), stream=output)
            .strip_dirs()
            .sort_stats("cumulative")
            .print_stats(80)
        )
    return output.getvalue()


def _run_bace_profile_smoke(context: Context) -> None:
    inputs = _all_input_gates(context, "bace_preserved")
    expected_inputs = _input_manifest_digests(inputs)
    report_root = context.persistent / "outputs/profile"
    root = report_root / "bace_comrecgc"
    sentinel = report_root / "BACE_PROFILE_SMOKE_PASS.json"
    report_json = report_root / "bace_comrecgc_profile.json"
    report_text = report_root / "bace_comrecgc_profile.txt"
    if sentinel.is_file():
        _require_bace_profile_smoke_gate(
            context, input_manifests=expected_inputs
        )
        _verify_all_input_gates(inputs)
        return
    _assert_empty_or_absent(root, label="BACE profile output")
    root.mkdir(parents=True, exist_ok=True)
    profile_paths = [
        root / "uninterrupted_1000.cprofile",
        root / "resume_500_to_1000.cprofile",
    ]

    command_a, paths_a = _profile_generation_command(
        context,
        label="uninterrupted",
        resume=False,
        profile_output=profile_paths[0],
    )
    timing_a = _run_profile_until_checkpoint(
        context,
        command=command_a,
        mirror_root=paths_a["mirror"],
        target_step=1000,
        progress_path=paths_a["output"] / "progress.json",
        resolved_config_source=paths_a["output"] / "resolved_config.json",
        resolved_config_destination=(paths_a["mirror"].parent / "resolved_config.json"),
    )

    command_b_fresh, paths_b = _profile_generation_command(
        context,
        label="resume_path",
        resume=False,
        profile_output=None,
    )
    timing_b_fresh = _run_profile_until_checkpoint(
        context,
        command=command_b_fresh,
        mirror_root=paths_b["mirror"],
        target_step=500,
        termination_signal=signal.SIGKILL,
        progress_path=paths_b["output"] / "progress.json",
        stop_after_progress_step=525,
        resolved_config_source=paths_b["output"] / "resolved_config.json",
        resolved_config_destination=(paths_b["mirror"].parent / "resolved_config.json"),
    )
    # Simulate complete loss of /root/autodl-tmp without deleting evidence.
    # The whole fast profile root (generation, graph state, and checkpoints)
    # is quarantined and the active root is rebuilt from persistent state only.
    active_profile_root = paths_b["guard"]
    quarantined = active_profile_root.with_name("resume_path.pre_restore")
    if quarantined.exists() or not active_profile_root.is_dir():
        raise StageError("Profile fast-loss simulation has an unsafe collision")
    os.rename(active_profile_root, quarantined)
    _fsync_directory(quarantined.parent)
    persistent_resolved_config = paths_b["mirror"].parent / "resolved_config.json"
    paths_b["output"].mkdir(parents=True, exist_ok=True)
    _copy_file_atomic(
        persistent_resolved_config, paths_b["output"] / "resolved_config.json"
    )
    restored_step = _restore_checkpoint_roots(
        context, fast_root=paths_b["checkpoint"], mirror_root=paths_b["mirror"]
    )
    if restored_step != 500:
        raise StageError(f"Expected restored profile step 500, got {restored_step}")
    trace_reconciliation = _reconcile_trace_to_checkpoint(
        context,
        checkpoint_root=paths_b["checkpoint"],
        trace_root=paths_b["trace"],
        quarantine_root=(
            context.persistent
            / "outputs/profile/bace_comrecgc/resume_path/trace_recovery_quarantine"
        ),
    )
    command_b_resume, _paths_b_resume = _profile_generation_command(
        context,
        label="resume_path",
        resume=True,
        profile_output=profile_paths[1],
    )
    timing_b_resume = _run_profile_until_checkpoint(
        context,
        command=command_b_resume,
        mirror_root=paths_b["mirror"],
        target_step=1000,
        progress_path=paths_b["output"] / "progress.json",
        resolved_config_source=paths_b["output"] / "resolved_config.json",
        resolved_config_destination=persistent_resolved_config,
    )
    module = _checkpoint_module(context)
    checkpoint_a = module.validate_generation_checkpoint(
        paths_a["mirror"], expected_completed_step=1000
    ).checkpoint_dir
    checkpoint_b = module.validate_generation_checkpoint(
        paths_b["mirror"], expected_completed_step=1000
    ).checkpoint_dir
    audit_a = _logical_checkpoint_audit(context, checkpoint_a)
    audit_b = _logical_checkpoint_audit(context, checkpoint_b)
    compared = (
        "completed_step",
        "parity_normalized_scientific_command_sha256",
        "parity_provenance_fingerprints",
        "algorithm_logical_sha256",
        "trace_rows_logical_sha256",
        "trace_chunk_count",
        "pending_event_count",
        "pending_events_logical_sha256",
        "trace_counters",
        "trace_counters_sha256",
        "rng_logical_sha256",
        "sqlite_logical_sha256",
        "sequence_sha256",
    )
    mismatches = [name for name in compared if audit_a[name] != audit_b[name]]
    if mismatches:
        raise StageError(f"BACE uninterrupted/resume checkpoint parity failed: {mismatches}")
    input_manifests = _verify_all_input_gates(inputs)
    observations_a = timing_a.pop("performance_observations")
    observations_b_fresh = timing_b_fresh.pop("performance_observations")
    observations_b_resume = timing_b_resume.pop("performance_observations")
    structured_performance = {
        "schema_version": PROFILE_PERFORMANCE_SCHEMA,
        "status_values": ["OBSERVED", "NOT_OBSERVED"],
        "required_runtime_measurements": list(
            PROFILE_REQUIRED_RUNTIME_MEASUREMENTS
        ),
        "required_combined_function_categories": list(
            PROFILE_REQUIRED_FUNCTION_CATEGORIES
        ),
        "optional_function_categories": list(
            PROFILE_OPTIONAL_FUNCTION_CATEGORIES
        ),
        "optional_function_absence_reasons": dict(
            PROFILE_OPTIONAL_ABSENCE_REASONS
        ),
        "function_category_patterns": {
            key: list(value) for key, value in PROFILE_FUNCTION_PATTERNS.items()
        },
        "runs": {
            "uninterrupted_0_to_1000": {
                "start_step": 0,
                "stop_step": 1000,
                "runtime_observations": observations_a,
                "function_profile": _aggregate_pstats([profile_paths[0]]),
            },
            "resume_path_0_to_post_checkpoint_kill": {
                "start_step": 0,
                "trusted_checkpoint_step": 500,
                "process_step_at_kill": int(
                    timing_b_fresh["observed_process_step_at_signal"]
                ),
                "stop_step": int(timing_b_fresh["observed_process_step_at_signal"]),
                "runtime_observations": observations_b_fresh,
                "function_profile": _unobserved_function_profile(
                    "intentional_SIGKILL_prevents_durable_cprofile_dump"
                ),
            },
            "resume_path_500_to_1000": {
                "start_step": 500,
                "stop_step": 1000,
                "runtime_observations": observations_b_resume,
                "function_profile": _aggregate_pstats([profile_paths[1]]),
            },
        },
        "combined_observed_function_profile": _aggregate_pstats(profile_paths),
    }
    code_lineage = _assert_stage_lineage_unchanged(context)
    report = {
        "schema_version": "bace_comrecgc_profile_v1",
        "status": "PASS",
        "formal_generation_launched": False,
        "structured_performance_schema_version": PROFILE_PERFORMANCE_SCHEMA,
        "structured_performance_run_count": 3,
        **code_lineage,
        "scientific_configuration": "full_50000_unchanged",
        "profile_observation_stop_step": 1000,
        "checkpoint_interval_steps": 500,
        "uninterrupted": timing_a,
        "resume_path_fresh": timing_b_fresh,
        "abrupt_kill_test_passed": (
            timing_b_fresh["termination_signal"]
            == "SIGKILL_after_persistent_checkpoint_validation"
            and int(timing_b_fresh["observed_process_step_at_signal"]) >= 525
            and int(timing_b_fresh["target_checkpoint_step"]) == 500
        ),
        "resume_path_restored_checkpoint_step": restored_step,
        "resume_path_trace_reconciliation": trace_reconciliation,
        "resume_restore_sources": {
            "resolved_config": {
                "source": str(persistent_resolved_config),
                "destination": str(paths_b["output"] / "resolved_config.json"),
                "sha256": sha256_file(persistent_resolved_config),
            },
            "checkpoint_mirror": str(paths_b["mirror"]),
            "checkpoint_mirror_latest2": True,
            "trace": str(paths_b["trace"]),
            "trace_persistence": "persistent_atomic_chunks",
            "quarantined_fast_active_state": str(quarantined),
        },
        "resume_path_resumed": timing_b_resume,
        "uninterrupted_checkpoint": audit_a,
        "resumed_checkpoint": audit_b,
        "compared_fields": list(compared),
        "checkpoint_parity": True,
        "input_manifests_sha256": input_manifests,
        "cprofile_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in profile_paths
        ],
        "structured_performance": structured_performance,
        "completed_at": utc_now(),
    }
    _validate_structured_profile_evidence(report, evidence_root=report_root)
    atomic_write_json(report_json, report)
    atomic_write_bytes(report_text, _profile_text(profile_paths).encode("utf-8"))
    atomic_write_json(
        sentinel,
        {
            "schema_version": "bace_comrecgc_profile_gate_v1",
            "run_id": RUN_ID,
            "status": "PASS",
            "checkpoint_parity": True,
            "abrupt_kill_test_passed": True,
            "formal_generation_launched": False,
            "structured_performance_schema_version": PROFILE_PERFORMANCE_SCHEMA,
            "structured_performance_run_count": 3,
            **code_lineage,
            "input_manifest_sha256_before": expected_inputs["primary"],
            "input_manifest_sha256_after": input_manifests["primary"],
            "input_manifests_sha256": input_manifests,
            "report_json": str(report_json),
            "report_json_sha256": sha256_file(report_json),
            "report_text": str(report_text),
            "report_text_sha256": sha256_file(report_text),
            "finished_at": utc_now(),
        },
    )
    _require_bace_profile_smoke_gate(
        context, input_manifests=input_manifests
    )


def _verify_formal_stage_completion(context: Context, stage: str) -> None:
    """Read-only revalidation used by controller status/resume/dependencies.

    The controller's orchestration sentinel is intentionally not sufficient:
    adoption must re-hash all three current input cohorts, the full code/config
    closure, the pinned vendor tree, and the scientific output manifest.
    """

    contracts: dict[str, tuple[str, Path, Path, Path, dict[str, Any]]] = {
        "mut-freeze": (
            "mut_generation",
            context.persistent / "outputs/mut_comrecgc/MUT_FREEZE_RECOVERY_PASS.json",
            context.persistent
            / "outputs/mut_comrecgc/manifests/mutagenicity_freeze_recovery.sha256",
            context.persistent / "outputs/mut_comrecgc",
            {"status": "PASS", "generation_rerun_performed": False},
        ),
        "mut-downstream": (
            "mut_generation",
            context.persistent / "outputs/mut_comrecgc/MUT_COMRECGC_COMPLETE.json",
            context.persistent / "outputs/mut_comrecgc/MANIFEST.sha256",
            context.persistent / "outputs/mut_comrecgc",
            {"status": "PASS", "generation_rerun_performed": False},
        ),
        "aids-freeze": (
            "aids_generation",
            context.persistent / "outputs/aids_comrecgc/AIDS_FREEZE_RECOVERY_PASS.json",
            context.persistent
            / "outputs/aids_comrecgc/manifests/aids_freeze_recovery.sha256",
            context.persistent / "outputs/aids_comrecgc",
            {"status": "PASS", "generation_rerun_performed": False},
        ),
        "aids-downstream": (
            "aids_generation",
            context.persistent / "outputs/aids_comrecgc/AIDS_COMRECGC_COMPLETE.json",
            context.persistent / "outputs/aids_comrecgc/MANIFEST.sha256",
            context.persistent / "outputs/aids_comrecgc",
            {"status": "PASS", "generation_rerun_performed": False},
        ),
        "bace-generate": (
            "bace_preserved",
            context.persistent
            / "outputs/bace_comrecgc/BACE_GENERATION_50000_PASS.json",
            context.persistent / "outputs/bace_comrecgc/generation/MANIFEST.sha256",
            context.persistent / "outputs/bace_comrecgc",
            {
                "status": "PASS",
                "fresh_start_step": 0,
                "imported_old_partial_state": False,
                "completed_step": 50_000,
            },
        ),
        "bace-final": (
            "bace_preserved",
            context.persistent / "outputs/bace_comrecgc/BACE_COMRECGC_COMPLETE.json",
            context.persistent / "outputs/bace_comrecgc/MANIFEST.sha256",
            context.persistent / "outputs/bace_comrecgc",
            {"status": "PASS"},
        ),
        "bace-globalgce": (
            "bace_preserved",
            context.persistent
            / "outputs/bace_globalgce_common4/BACE_GLOBALGCE_WNODE_COMPLETE.json",
            context.persistent
            / "outputs/bace_globalgce_common4/common4/globalgce/MANIFEST.sha256",
            context.persistent / "outputs/bace_globalgce_common4/common4/globalgce",
            {
                "status": "PASS",
                "ours_generation_rerun": False,
                "gcf_generation_rerun": False,
                "globalgce_selection_rerun": False,
            },
        ),
        "bace-common4": (
            "bace_preserved",
            context.persistent
            / "outputs/bace_globalgce_common4/BACE_COMMON4_COMPLETE.json",
            context.persistent
            / "outputs/bace_globalgce_common4/common4/MANIFEST.sha256",
            context.persistent / "outputs/bace_globalgce_common4",
            {"status": "PASS", "canonical_method_count": 4},
        ),
    }
    if stage not in contracts:
        raise StageError(f"--verify-only is unsupported for non-formal stage: {stage}")
    primary, sentinel, manifest, manifest_root, fields = contracts[stage]
    snapshot = _all_input_gates(
        context, primary, publish_required_static=False
    )
    expected_inputs = _input_manifest_digests(snapshot)
    if stage in {"mut-freeze", "mut-downstream"}:
        _require_lineage_smoke_gate(
            context, "mutagenicity", input_manifests=expected_inputs
        )
    elif stage in {"aids-freeze", "aids-downstream"}:
        _require_lineage_smoke_gate(
            context, "aids", input_manifests=expected_inputs
        )
    elif stage in {"bace-generate", "bace-final"}:
        _require_bace_profile_smoke_gate(
            context, input_manifests=expected_inputs
        )
    if not _verify_sentinel(
        context,
        sentinel,
        manifest,
        fields,
        input_manifests=expected_inputs,
    ):
        raise StageError(f"Formal completion sentinel is absent: {sentinel}")
    if Path(read_json(sentinel).get("output_manifest_root", "")) != manifest_root:
        raise StageError(f"Formal completion manifest root changed: {sentinel}")
    _verify_all_input_gates(snapshot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in (
        "mut-lineage-smoke",
        "mut-freeze",
        "mut-downstream",
        "aids-lineage-smoke",
        "aids-freeze",
        "aids-downstream",
        "bace-generate",
        "bace-final",
        "bace-globalgce",
        "bace-common4",
        "bace-profile-smoke",
    ):
        stage = subparsers.add_parser(name)
        stage.add_argument("--project-root", required=True)
        stage.add_argument("--step0-project-root", required=True)
        stage.add_argument("--external-root", required=True)
        stage.add_argument("--persistent-root", required=True)
        stage.add_argument("--fast-root", required=True)
        stage.add_argument("--python", required=True)
        stage.add_argument("--config", default=None, help=argparse.SUPPRESS)
        stage.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
        stage.add_argument("--resume", action="store_true")
        stage.add_argument("--verify-only", action="store_true")
    return parser


def _context(args: argparse.Namespace) -> Context:
    raw_paths = {
        "project": Path(args.project_root).expanduser(),
        "step0": Path(args.step0_project_root).expanduser(),
        "external": Path(args.external_root).expanduser(),
        "persistent": Path(args.persistent_root).expanduser(),
        "fast": Path(args.fast_root).expanduser(),
        "python": Path(args.python).expanduser(),
    }
    # The five managed roots must be physical paths.  A Python interpreter is
    # conventionally a symlink (for example ``python -> python3.10``), so it is
    # resolved to its physical target and validated as an executable below.
    for label in ("project", "step0", "external", "persistent", "fast"):
        path = raw_paths[label]
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                raise StageError(
                    f"{label} path contains a symlink component: {current}"
                )
            if not current.exists():
                break
    return Context(
        project=raw_paths["project"].resolve(),
        step0=raw_paths["step0"].resolve(),
        external=raw_paths["external"].resolve(),
        persistent=raw_paths["persistent"].resolve(),
        fast=raw_paths["fast"].resolve(),
        python=raw_paths["python"].resolve(),
        resume=bool(args.resume),
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "_profile-exec":
        return _profile_exec(raw[1:])
    args = build_parser().parse_args(raw)
    context = _context(args)
    _validate_roots(context)
    context = replace(
        context, stage_start_code_lineage=_current_code_lineage(context)
    )
    if args.verify_only:
        if args.resume:
            raise StageError("--verify-only and --resume are mutually exclusive")
        _verify_formal_stage_completion(context, args.stage)
        print(
            f"[AUTODL_THREE_LINES_STAGE_VERIFY_PASS] stage={args.stage}",
            flush=True,
        )
        return 0
    actions: dict[str, Callable[[Context], None]] = {
        "mut-lineage-smoke": lambda value: _run_lineage_smoke(
            value, "mutagenicity"
        ),
        "mut-freeze": lambda value: _run_freeze(value, "mutagenicity"),
        "mut-downstream": lambda value: _run_downstream(value, "mutagenicity"),
        "aids-lineage-smoke": lambda value: _run_lineage_smoke(value, "aids"),
        "aids-freeze": lambda value: _run_freeze(value, "aids"),
        "aids-downstream": lambda value: _run_downstream(value, "aids"),
        "bace-generate": _run_bace_generate,
        "bace-final": _run_bace_final,
        "bace-globalgce": _run_bace_globalgce,
        "bace-common4": _run_bace_common4,
        "bace-profile-smoke": _run_bace_profile_smoke,
    }
    actions[args.stage](context)
    _assert_stage_lineage_unchanged(context)
    print(f"[AUTODL_THREE_LINES_STAGE_PASS] stage={args.stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
