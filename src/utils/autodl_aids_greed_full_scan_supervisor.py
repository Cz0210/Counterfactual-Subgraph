"""Fail-closed supervisor for the production AIDS GREED close-pair scan.

The generic AutoDL launcher deliberately requires every ``expected_output`` to
be fresh.  The production GREED scan, however, has an exact, chunk-authenticated
checkpoint and must not restart 91,916,686 rows after a worker or host loss.
This module separates those two lifetimes:

* every controller attempt publishes a fresh, small receipt root;
* one campaign-owned science root contains the distance arrays/checkpoint;
* only this supervisor may add ``--resume`` to the reviewed child command;
* resume is bounded and requires a dead prior PID generation, an exact
  checkpoint/prefix, unchanged root/lock identity, no scientific PASS, and no
  semantic/provenance failure marker.

No output is copied between roots.  A receipt hash-binds the terminal science
manifest and is published before its exact ``PASS`` sentinel.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from src.baselines.comrecgc.aids_pair_semantics import (
    AIDS_PAIR_SEMANTICS_SCHEMA,
    _resolve_pair_chunk_authority,
)
from src.baselines.comrecgc.close_pair_scan import (
    CLOSE_BITMAP_DTYPE,
    CLOSE_PAIR_SCAN_SCHEMA,
    DISTANCE_DTYPE,
    _validate_resume_prefix,
)
from src.baselines.comrecgc.contracts import (
    atomic_write_bytes,
    sha256_file,
    stable_json_sha256,
)
from src.baselines.comrecgc.external_memory_recourse import (
    _find_writable_process_references,
)


SUPERVISOR_SCHEMA = "aids_greed_full_scan_supervisor_v1"
RECEIPT_SCHEMA = "aids_greed_full_scan_supervisor_receipt_v1"
RECEIPT_NAME = "pair_semantics_supervisor_receipt.json"
CONTROL_DIRECTORY = ".aids_greed_full_scan_supervisor"
CONTRACT_NAME = "contract.json"
STATE_NAME = "state.json"
EVENTS_NAME = "events.jsonl"
TERMINAL_NAME = "terminal_supervisor_manifest.json"
TERMINAL_PASS_NAME = "TERMINAL_PASS"
LOCK_NAME = ".aids_greed_full_scan.lock"

EXPECTED_PARENT_COUNT = 1_283
EXPECTED_CANDIDATE_COUNT = 71_642
EXPECTED_PAIR_COUNT = EXPECTED_PARENT_COUNT * EXPECTED_CANDIDATE_COUNT
EXPECTED_THETA = 0.1
EXPECTED_CHUNK_COUNT = 560

TRANSIENT_SIGNALS = frozenset(
    {signal.SIGHUP, signal.SIGINT, signal.SIGKILL, signal.SIGTERM}
)
TRANSIENT_EXIT_CODES = frozenset(
    {75, *(128 + int(value) for value in TRANSIENT_SIGNALS)}
)
TRANSIENT_TEXT_MARKERS = (
    "input/output error",
    "errno 5",
    "stale file handle",
    "transport endpoint is not connected",
    "connection reset by peer",
    "connection timed out",
    "resource temporarily unavailable",
)
DEFAULT_SEMANTIC_MARKERS = (
    "aidspairsemanticserror",
    "closepairscanerror",
    "provenance contract failed",
    "source binding failed",
    "dataset provenance failed",
    "dataset fingerprint differs",
    "identity differs",
    "sha256 differs",
    "physical pair axes differ",
    "read-only source stat identity changed",
    "non-finite values",
    "normalization differ",
    "formula sample audit failed",
    "resume prefix differs",
    "resume checkpoint identity differs",
    "resume partial array schema differs",
    "incomplete scan is missing a partial array",
    "direct pair-store sha256 differs",
    "insufficient free space",
    "another pair-semantics audit owns",
    "fresh close-pair scan artifacts exist",
)


class AIDSGreedFullScanSupervisorError(RuntimeError):
    """Raised when a fresh launch or same-root resume cannot be certified."""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _read_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise AIDSGreedFullScanSupervisorError(f"{label} may not be a symlink")
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size <= 0:
        raise AIDSGreedFullScanSupervisorError(
            f"{label} must be a physical nonempty file"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AIDSGreedFullScanSupervisorError(
            f"{label} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise AIDSGreedFullScanSupervisorError(f"{label} must be one JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(
        path,
        (
            json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        ).encode("utf-8"),
    )


def _append_event(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "mode": int(value.st_mode),
    }


def _directory_identity(path: Path) -> dict[str, int]:
    value = _file_identity(path)
    if not stat.S_ISDIR(value["mode"]):
        raise AIDSGreedFullScanSupervisorError(f"not a directory: {path}")
    return {"device": value["device"], "inode": value["inode"]}


def _absolute(path: str | Path, *, label: str, must_exist: bool) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        raise AIDSGreedFullScanSupervisorError(f"{label} must be absolute")
    if value.is_symlink():
        raise AIDSGreedFullScanSupervisorError(f"{label} may not be a symlink")
    resolved = value.resolve(strict=must_exist)
    if must_exist and not resolved.exists():
        raise AIDSGreedFullScanSupervisorError(f"{label} does not exist")
    return resolved


def _under(path: Path, parent: Path, *, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise AIDSGreedFullScanSupervisorError(
            f"{label} must stay under campaign root {parent}"
        ) from exc


def _git_head(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AIDSGreedFullScanSupervisorError(
            "cannot resolve execution worktree HEAD"
        ) from exc


def read_process_identity(pid: int, *, proc_root: str | Path = "/proc") -> dict[str, Any] | None:
    """Read one Linux PID generation without ever signalling it."""

    if isinstance(pid, bool) or int(pid) <= 0:
        return None
    root = Path(proc_root)
    try:
        raw_stat = (root / str(int(pid)) / "stat").read_text(encoding="utf-8")
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2 :].split()
        start_ticks = int(fields[19])
        raw_cmdline = (root / str(int(pid)) / "cmdline").read_bytes()
    except (OSError, ValueError, IndexError):
        return None
    return {
        "pid": int(pid),
        "start_ticks": start_ticks,
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
    }


def process_identity_matches(
    expected: Mapping[str, Any] | None,
    *,
    proc_root: str | Path = "/proc",
) -> bool:
    if not isinstance(expected, Mapping):
        return False
    pid = expected.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return False
    current = read_process_identity(pid, proc_root=proc_root)
    return bool(
        current
        and int(expected.get("start_ticks", -1)) == current["start_ticks"]
        and str(expected.get("cmdline_sha256", "")) == current["cmdline_sha256"]
    )


def find_live_science_writers(
    *,
    science_root: Path,
    proc_root: str | Path = "/proc",
    exclude_pids: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Find direct supervisor/science commands that name the fixed work root."""

    excluded = {int(value) for value in exclude_pids}
    writers: list[dict[str, Any]] = []
    root = Path(proc_root)
    for entry in root.iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [
            value.decode("utf-8", errors="surrogateescape")
            for value in raw.split(b"\0")
            if value
        ]
        if not argv:
            continue
        basenames = {Path(value).name for value in argv[:4]}
        flag = None
        if "run_aids_comrecgc_pair_semantics.py" in basenames:
            flag = "--output-dir"
        elif "run_aids_greed_full_scan_supervisor.py" in basenames:
            flag = "--science-root"
        if flag is None:
            continue
        try:
            value = Path(_option(argv, flag)).expanduser().resolve(strict=False)
        except (AIDSGreedFullScanSupervisorError, OSError):
            continue
        if value != science_root:
            continue
        identity = read_process_identity(int(entry.name), proc_root=root)
        if identity is not None:
            writers.append({**identity, "argv": argv, "matched_flag": flag})
    return writers


def _option(argv: Sequence[str], flag: str) -> str:
    locations = [index for index, value in enumerate(argv) if value == flag]
    if len(locations) != 1 or locations[0] + 1 >= len(argv):
        raise AIDSGreedFullScanSupervisorError(
            f"reviewed child command must contain one {flag} value"
        )
    value = argv[locations[0] + 1]
    if value.startswith("--"):
        raise AIDSGreedFullScanSupervisorError(f"reviewed child {flag} value is absent")
    return value


def validate_child_command(
    child_argv: Sequence[str],
    *,
    project_root: Path,
    science_root: Path,
) -> dict[str, str]:
    command = [str(value) for value in child_argv]
    if not command:
        raise AIDSGreedFullScanSupervisorError("reviewed child command is empty")
    if "--resume" in command:
        raise AIDSGreedFullScanSupervisorError(
            "reviewed base child command must not contain --resume"
        )
    for forbidden in ("--max-chunks", "--skip-source-array-hash-verification"):
        if forbidden in command:
            raise AIDSGreedFullScanSupervisorError(
                f"production full scan forbids {forbidden}"
            )
    if len(command) < 2:
        raise AIDSGreedFullScanSupervisorError("reviewed child entrypoint is absent")
    expected_script = (
        project_root / "scripts/autodl/run_aids_comrecgc_pair_semantics.py"
    ).resolve(strict=True)
    script = Path(command[1]).expanduser().resolve(strict=True)
    if script != expected_script or script.is_symlink():
        raise AIDSGreedFullScanSupervisorError(
            "reviewed child pair-semantics entrypoint changed"
        )
    output = _absolute(
        _option(command, "--output-dir"), label="child science output", must_exist=False
    )
    if output != science_root:
        raise AIDSGreedFullScanSupervisorError(
            "reviewed child output does not equal the campaign science root"
        )
    if _option(command, "--device") != "cpu":
        raise AIDSGreedFullScanSupervisorError("GREED full scan must remain CPU-only")
    if int(_option(command, "--parent-limit")) != EXPECTED_PARENT_COUNT:
        raise AIDSGreedFullScanSupervisorError("AIDS parent count changed")
    if float(_option(command, "--theta")) != EXPECTED_THETA:
        raise AIDSGreedFullScanSupervisorError("AIDS theta changed")
    child_project = _absolute(
        _option(command, "--project-root"), label="child project root", must_exist=True
    )
    if child_project != project_root:
        raise AIDSGreedFullScanSupervisorError("child project root changed")
    return {
        "pair_store_manifest": str(
            _absolute(
                _option(command, "--pair-store-manifest"),
                label="pair-store manifest",
                must_exist=True,
            )
        ),
        "expected_pair_store_manifest_sha256": _option(
            command, "--expected-pair-store-manifest-sha256"
        ),
        "generation_dir": str(
            _absolute(
                _option(command, "--generation-dir"),
                label="generation directory",
                must_exist=True,
            )
        ),
        "distance_checkpoint": str(
            _absolute(
                _option(command, "--distance-checkpoint"),
                label="distance checkpoint",
                must_exist=True,
            )
        ),
    }


def build_contract(
    *,
    project_root: Path,
    execution_commit: str,
    campaign_root: Path,
    science_root: Path,
    lock_path: Path,
    lock_identity: Mapping[str, int],
    child_argv: Sequence[str],
    child_sources: Mapping[str, str],
    semantic_markers: Sequence[str],
    max_same_root_resumes: int,
    proc_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SUPERVISOR_SCHEMA,
        "status": "FROZEN",
        "dataset": "aids",
        "stage": "GREED_THETA_CLOSE_FULL_SCAN",
        "project_root": str(project_root),
        "execution_commit": execution_commit,
        "campaign_root": str(campaign_root),
        "science_root": str(science_root),
        "science_root_identity_at_contract_creation": None,
        "lock_path": str(lock_path),
        "lock_identity": dict(lock_identity),
        "child_argv_without_resume": list(child_argv),
        "child_argv_without_resume_sha256": stable_json_sha256(list(child_argv)),
        "child_sources": dict(child_sources),
        "semantic_failure_markers": sorted(set(semantic_markers)),
        "max_same_root_resumes": int(max_same_root_resumes),
        "proc_root": str(proc_root),
        "first_launch_requires_fresh_science_root": True,
        "first_launch_resume_flag": False,
        "resume_requires_dead_prior_generation": True,
        "resume_requires_exact_checkpoint_prefix": True,
        "resume_requires_pass_absence": True,
        "terminal_files": [
            "PASS",
            "pair_semantics_audit.json",
            "close_pair_contract.json",
            "distance_scan/run_manifest.json",
            "distance_scan/normalized_distances.greed.float32.npy",
            "distance_scan/close_pair_bitmap.greed.uint8.npy",
        ],
        "created_at": _utc_now(),
    }


def _contract_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "schema_version",
            "status",
            "dataset",
            "stage",
            "project_root",
            "execution_commit",
            "campaign_root",
            "science_root",
            "science_root_identity_at_contract_creation",
            "lock_path",
            "lock_identity",
            "child_argv_without_resume",
            "child_argv_without_resume_sha256",
            "child_sources",
            "semantic_failure_markers",
            "max_same_root_resumes",
            "proc_root",
            "first_launch_requires_fresh_science_root",
            "first_launch_resume_flag",
            "resume_requires_dead_prior_generation",
            "resume_requires_exact_checkpoint_prefix",
            "resume_requires_pass_absence",
            "terminal_files",
        )
    }


def _reject_symlinks(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        base = Path(current)
        for name in [*directories, *files]:
            path = base / name
            if path.is_symlink():
                raise AIDSGreedFullScanSupervisorError(
                    f"science root contains a symlink: {path}"
                )


def _assert_no_writable_science_references(
    science_root: Path, *, proc_root: Path
) -> None:
    files = [path for path in science_root.rglob("*") if path.is_file()]
    writers = _find_writable_process_references(files, proc_root=proc_root)
    if writers:
        summary = ",".join(
            f"pid={value['pid']}:{value['kind']}:{value['path']}"
            for value in writers[:8]
        )
        raise AIDSGreedFullScanSupervisorError(
            "science tree has writable process references: " + summary
        )


def _semantic_log_match(control_root: Path, markers: Sequence[str]) -> str | None:
    lowered = tuple(marker.lower() for marker in markers)
    for log_path in sorted(control_root.glob("child-attempt-*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in lowered:
            if marker and marker in text:
                return f"{marker} in {log_path.name}"
    return None


def validate_resume_checkpoint(
    *, science_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen every committed distance block before authorizing ``--resume``."""

    if (science_root / "PASS").exists() or (science_root / "PASS").is_symlink():
        raise AIDSGreedFullScanSupervisorError(
            "scientific PASS exists; nonterminal resume is forbidden"
        )
    _reject_symlinks(science_root)
    _assert_no_writable_science_references(
        science_root, proc_root=Path(str(contract.get("proc_root") or "/proc"))
    )
    checkpoint_path = science_root / "distance_scan/checkpoint.json"
    checkpoint = _read_object(checkpoint_path, label="distance-scan checkpoint")
    identity = checkpoint.get("identity")
    if not isinstance(identity, Mapping):
        raise AIDSGreedFullScanSupervisorError("distance checkpoint identity is absent")
    if (
        checkpoint.get("schema_version") != CLOSE_PAIR_SCAN_SCHEMA
        or checkpoint.get("identity_sha256") != stable_json_sha256(identity)
        or identity.get("schema_version") != CLOSE_PAIR_SCAN_SCHEMA
        or int(identity.get("parent_count", -1)) != EXPECTED_PARENT_COUNT
        or int(identity.get("candidate_count", -1)) != EXPECTED_CANDIDATE_COUNT
        or int(identity.get("physical_pair_count", -1)) != EXPECTED_PAIR_COUNT
        or float(identity.get("theta", -1)) != EXPECTED_THETA
        or identity.get("filter_operator") != "<="
        or identity.get("pair_orientation")
        != ["parent_index", "candidate_index"]
        or identity.get("pair_order") != "candidate_major_parent_minor"
        or identity.get("distance_dtype") != str(DISTANCE_DTYPE)
        or identity.get("close_bitmap_dtype") != str(CLOSE_BITMAP_DTYPE)
        or int(identity.get("chunk_count", -1)) != EXPECTED_CHUNK_COUNT
    ):
        raise AIDSGreedFullScanSupervisorError(
            "distance checkpoint production identity differs"
        )
    scientific = identity.get("scientific_identity")
    sources = contract.get("child_sources")
    if not isinstance(scientific, Mapping) or not isinstance(sources, Mapping):
        raise AIDSGreedFullScanSupervisorError(
            "distance checkpoint scientific/source identity is absent"
        )
    pair_manifest = Path(str(sources["pair_store_manifest"])).resolve(strict=True)
    generation_manifest = (
        Path(str(sources["generation_dir"])) / "run_manifest.json"
    ).resolve(strict=True)
    distance_checkpoint = Path(str(sources["distance_checkpoint"])).resolve(
        strict=True
    )
    expected_source_fields = {
        "project_commit": contract["execution_commit"],
        "pair_store_manifest": str(pair_manifest),
        "pair_store_manifest_sha256": str(
            sources["expected_pair_store_manifest_sha256"]
        ),
        "generation_manifest": str(generation_manifest),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "distance_checkpoint": str(distance_checkpoint),
        "distance_checkpoint_sha256": sha256_file(distance_checkpoint),
    }
    failures = [
        key
        for key, expected in expected_source_fields.items()
        if scientific.get(key) != expected
    ]
    if failures or sha256_file(pair_manifest) != str(
        sources["expected_pair_store_manifest_sha256"]
    ):
        raise AIDSGreedFullScanSupervisorError(
            "distance checkpoint source identity differs: " + ", ".join(failures)
        )
    pair_payload = _read_object(pair_manifest, label="pair-store manifest")
    chunks, _authority = _resolve_pair_chunk_authority(pair_manifest, pair_payload)
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise AIDSGreedFullScanSupervisorError("distance checkpoint chunk count differs")
    pair_raw = Path(str(pair_payload["pairs_path"])).expanduser()
    vector_raw = Path(str(pair_payload["vectors_path"])).expanduser()
    if pair_raw.is_symlink() or vector_raw.is_symlink():
        raise AIDSGreedFullScanSupervisorError(
            "pair-store arrays may not be symlinks"
        )
    pair_path = pair_raw.resolve(strict=True)
    partial_distance = science_root / (
        "distance_scan/normalized_distances.greed.float32.partial.npy"
    )
    partial_bitmap = science_root / (
        "distance_scan/close_pair_bitmap.greed.uint8.partial.npy"
    )
    final_distance = science_root / (
        "distance_scan/normalized_distances.greed.float32.npy"
    )
    final_bitmap = science_root / "distance_scan/close_pair_bitmap.greed.uint8.npy"
    if partial_distance.exists() and final_distance.exists():
        raise AIDSGreedFullScanSupervisorError(
            "partial and final distance arrays coexist"
        )
    if partial_bitmap.exists() and final_bitmap.exists():
        raise AIDSGreedFullScanSupervisorError(
            "partial and final bitmap arrays coexist"
        )
    distance_path = partial_distance if partial_distance.exists() else final_distance
    bitmap_path = partial_bitmap if partial_bitmap.exists() else final_bitmap
    if distance_path.is_symlink() or bitmap_path.is_symlink():
        raise AIDSGreedFullScanSupervisorError("resume arrays may not be symlinks")
    distances = np.load(distance_path.resolve(strict=True), mmap_mode="r", allow_pickle=False)
    bitmap = np.load(bitmap_path.resolve(strict=True), mmap_mode="r", allow_pickle=False)
    if (
        distances.shape != (EXPECTED_PAIR_COUNT,)
        or distances.dtype != DISTANCE_DTYPE
        or bitmap.shape != (EXPECTED_PAIR_COUNT,)
        or bitmap.dtype != CLOSE_BITMAP_DTYPE
    ):
        raise AIDSGreedFullScanSupervisorError("resume array production schema differs")
    _validate_resume_prefix(
        checkpoint=checkpoint,
        chunks=chunks,
        distances=distances,
        bitmap=bitmap,
        parent_count=EXPECTED_PARENT_COUNT,
    )
    records = checkpoint.get("chunks")
    if not isinstance(records, list):
        raise AIDSGreedFullScanSupervisorError("distance checkpoint records are absent")
    rows_processed = sum(int(record.get("row_count", -1)) for record in records)
    close_count = sum(int(record.get("close_pair_count", -1)) for record in records)
    finite_count = sum(int(record.get("finite_count", -1)) for record in records)
    if (
        int(checkpoint.get("next_chunk_index", -1)) != len(records)
        or int(checkpoint.get("rows_processed", -1)) != rows_processed
        or int(checkpoint.get("logical_close_pair_count", -1)) != close_count
        or int(checkpoint.get("finite_count", -1)) != finite_count
        or rows_processed < 0
        or rows_processed > EXPECTED_PAIR_COUNT
    ):
        raise AIDSGreedFullScanSupervisorError(
            "distance checkpoint cumulative counters differ"
        )
    normalization = _read_object(
        science_root / "normalization_audit.json", label="normalization audit"
    )
    normalization_records = normalization.get("records")
    if not isinstance(normalization_records, Mapping):
        raise AIDSGreedFullScanSupervisorError(
            "normalization checkpoint records are absent"
        )
    committed_keys = {
        f"{chunk.candidate_start}:{chunk.candidate_stop}"
        for chunk in chunks[: len(records)]
    }
    if not committed_keys.issubset(normalization_records) or any(
        normalization_records[key].get("exact_equal") is not True
        for key in committed_keys
    ):
        raise AIDSGreedFullScanSupervisorError(
            "normalization checkpoint does not cover the committed prefix"
        )
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint_path.resolve(strict=True)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "next_chunk_index": len(records),
        "rows_processed": rows_processed,
        "distance_array": str(distance_path.resolve(strict=True)),
        "bitmap_array": str(bitmap_path.resolve(strict=True)),
        "committed_prefix_rehashed": True,
        "pair_array": str(pair_path),
        "pass_absent": True,
    }


def validate_terminal_science(
    *, science_root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    _reject_symlinks(science_root)
    _assert_no_writable_science_references(
        science_root, proc_root=Path(str(contract.get("proc_root") or "/proc"))
    )
    pass_path = science_root / "PASS"
    if not pass_path.is_file() or pass_path.read_bytes() != b"PASS\n":
        raise AIDSGreedFullScanSupervisorError("terminal science PASS is absent")
    close_path = science_root / "close_pair_contract.json"
    audit_path = science_root / "pair_semantics_audit.json"
    scan_path = science_root / "distance_scan/run_manifest.json"
    close = _read_object(close_path, label="close-pair contract")
    audit = _read_object(audit_path, label="pair-semantics audit")
    scan = _read_object(scan_path, label="distance-scan manifest")
    distance_path = science_root / "distance_scan/normalized_distances.greed.float32.npy"
    bitmap_path = science_root / "distance_scan/close_pair_bitmap.greed.uint8.npy"
    distance_sha256 = sha256_file(distance_path.resolve(strict=True))
    bitmap_sha256 = sha256_file(bitmap_path.resolve(strict=True))
    if (
        close.get("schema_version") != AIDS_PAIR_SEMANTICS_SCHEMA
        or close.get("status") != "PASS"
        or int(close.get("physical_store_rows", -1)) != EXPECTED_PAIR_COUNT
        or int(close.get("dbscan_input_count_must_equal", -1))
        != int(close.get("logical_close_rows", -2))
        or close.get("project_commit") != contract.get("execution_commit")
        or close.get("distance_scan_manifest") != str(scan_path.resolve(strict=True))
        or close.get("distance_scan_manifest_sha256") != sha256_file(scan_path)
        or audit.get("close_pair_contract") != str(close_path.resolve(strict=True))
        or audit.get("close_pair_contract_sha256") != sha256_file(close_path)
        or scan.get("identity", {}).get("scientific_identity", {}).get(
            "project_commit"
        )
        != contract.get("execution_commit")
        or scan.get("status") != "PASS"
        or scan.get("run_complete") is not True
        or int(scan.get("physical_pair_count", -1)) != EXPECTED_PAIR_COUNT
        or int(scan.get("logical_close_pair_count", -1))
        != int(close.get("logical_close_rows", -2))
        or scan.get("normalized_distances_path")
        != str(distance_path.resolve(strict=True))
        or scan.get("normalized_distances_sha256") != distance_sha256
        or close.get("normalized_distances")
        != str(distance_path.resolve(strict=True))
        or close.get("normalized_distances_sha256") != distance_sha256
        or scan.get("close_bitmap_path") != str(bitmap_path.resolve(strict=True))
        or scan.get("close_bitmap_sha256") != bitmap_sha256
        or close.get("close_bitmap") != str(bitmap_path.resolve(strict=True))
        or close.get("close_bitmap_hash") != bitmap_sha256
    ):
        raise AIDSGreedFullScanSupervisorError(
            "terminal science manifest closure differs"
        )
    terminal_files: dict[str, dict[str, Any]] = {}
    for relative in contract.get("terminal_files", []):
        path = (science_root / str(relative)).resolve(strict=True)
        if not path.is_file() or path.stat().st_size <= 0:
            raise AIDSGreedFullScanSupervisorError(
                f"terminal science file is absent: {relative}"
            )
        terminal_files[str(relative)] = {
            "path": str(path),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return {
        "status": "PASS",
        "science_root": str(science_root),
        "science_root_identity": _directory_identity(science_root),
        "close_pair_contract": str(close_path.resolve(strict=True)),
        "close_pair_contract_sha256": sha256_file(close_path),
        "pair_semantics_audit_sha256": sha256_file(audit_path),
        "distance_scan_manifest_sha256": sha256_file(scan_path),
        "terminal_files": terminal_files,
    }


def _freeze_terminal_supervisor_manifest(
    *,
    control_root: Path,
    contract_path: Path,
    terminal: Mapping[str, Any],
    resume_count: int,
    resume_reasons: Sequence[Mapping[str, Any]],
    owner_identity: Mapping[str, Any],
    lock_path: Path,
    lock_identity: Mapping[str, int],
) -> tuple[Path, dict[str, Any]]:
    path = control_root / TERMINAL_NAME
    pass_path = control_root / TERMINAL_PASS_NAME
    if pass_path.exists() and not path.exists():
        raise AIDSGreedFullScanSupervisorError(
            "terminal supervisor PASS exists without its manifest"
        )
    if path.exists():
        payload = _read_object(path, label="terminal supervisor manifest")
        if (
            payload.get("status") != "PASS"
            or payload.get("terminal_science") != dict(terminal)
            or payload.get("science_contract")
            != str(contract_path.resolve(strict=True))
            or payload.get("science_contract_sha256") != sha256_file(contract_path)
            or payload.get("lock_path") != str(lock_path.resolve(strict=True))
            or payload.get("lock_identity") != dict(lock_identity)
        ):
            raise AIDSGreedFullScanSupervisorError(
                "frozen terminal supervisor manifest differs"
            )
        if not pass_path.exists():
            descriptor = os.open(
                pass_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
            try:
                os.write(descriptor, b"PASS\n")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif pass_path.is_symlink() or pass_path.read_bytes() != b"PASS\n":
            raise AIDSGreedFullScanSupervisorError(
                "terminal supervisor PASS differs"
            )
        return path, payload
    payload = {
        "schema_version": SUPERVISOR_SCHEMA,
        "status": "PASS",
        "terminal_science": dict(terminal),
        "science_contract": str(contract_path.resolve(strict=True)),
        "science_contract_sha256": sha256_file(contract_path),
        "same_root_resume_count": int(resume_count),
        "same_root_resume_reasons": list(resume_reasons),
        "terminal_owner_identity": dict(owner_identity),
        "lock_path": str(lock_path.resolve(strict=True)),
        "lock_identity": dict(lock_identity),
        "published_at": _utc_now(),
    }
    _atomic_json(path, payload)
    descriptor = os.open(pass_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path, payload


def validate_receipt(
    *,
    receipt_path: str | Path,
    expected_science_root: str | Path,
    expected_execution_commit: str | None = None,
) -> dict[str, Any]:
    """Reopen a receipt and every fixed-root artifact it authorizes."""

    path = _absolute(receipt_path, label="pair-semantics receipt", must_exist=True)
    receipt = _read_object(path, label="pair-semantics receipt")
    pass_path = path.parent / "PASS"
    if pass_path.is_symlink() or not pass_path.is_file() or pass_path.read_bytes() != b"PASS\n":
        raise AIDSGreedFullScanSupervisorError("pair-semantics receipt PASS is absent")
    science = _absolute(
        expected_science_root, label="expected science root", must_exist=True
    )
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("science_root") != str(science)
        or receipt.get("science_root_identity") != _directory_identity(science)
        or receipt.get("controller_expected_output_is_receipt_only") is not True
        or receipt.get("large_science_artifacts_copied") is not False
    ):
        raise AIDSGreedFullScanSupervisorError(
            "pair-semantics receipt fixed science-root identity differs"
        )
    contract_path = _absolute(
        str(receipt.get("science_contract") or ""),
        label="receipt science contract",
        must_exist=True,
    )
    if sha256_file(contract_path) != receipt.get("science_contract_sha256"):
        raise AIDSGreedFullScanSupervisorError("receipt science-contract hash differs")
    contract = _read_object(contract_path, label="receipt science contract")
    if (
        contract.get("schema_version") != SUPERVISOR_SCHEMA
        or contract.get("status") != "FROZEN"
        or contract.get("science_root") != str(science)
        or contract.get("science_root_identity_at_contract_creation") is not None
    ):
        raise AIDSGreedFullScanSupervisorError("receipt science contract differs")
    if (
        expected_execution_commit is not None
        and contract.get("execution_commit") != expected_execution_commit
    ):
        raise AIDSGreedFullScanSupervisorError("receipt execution commit differs")
    terminal = validate_terminal_science(science_root=science, contract=contract)
    terminal_manifest_path = _absolute(
        str(receipt.get("terminal_supervisor_manifest") or ""),
        label="terminal supervisor manifest",
        must_exist=True,
    )
    if (
        sha256_file(terminal_manifest_path)
        != receipt.get("terminal_supervisor_manifest_sha256")
        or terminal_manifest_path.parent.joinpath(TERMINAL_PASS_NAME).is_symlink()
        or terminal_manifest_path.parent.joinpath(TERMINAL_PASS_NAME).read_bytes()
        != b"PASS\n"
    ):
        raise AIDSGreedFullScanSupervisorError(
            "terminal supervisor manifest/PASS differs"
        )
    terminal_manifest = _read_object(
        terminal_manifest_path, label="terminal supervisor manifest"
    )
    if (
        terminal_manifest.get("status") != "PASS"
        or terminal_manifest.get("terminal_science") != terminal
        or terminal_manifest.get("science_contract") != str(contract_path)
        or terminal_manifest.get("science_contract_sha256")
        != sha256_file(contract_path)
        or receipt.get("terminal_owner_identity")
        != terminal_manifest.get("terminal_owner_identity")
        or receipt.get("lock_path") != terminal_manifest.get("lock_path")
        or receipt.get("lock_identity") != terminal_manifest.get("lock_identity")
    ):
        raise AIDSGreedFullScanSupervisorError(
            "receipt terminal-supervisor closure differs"
        )
    lock_path = _absolute(
        str(receipt.get("lock_path") or ""), label="receipt lock", must_exist=True
    )
    lock_stat = os.lstat(lock_path)
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or receipt.get("lock_identity")
        != {"device": int(lock_stat.st_dev), "inode": int(lock_stat.st_ino)}
    ):
        raise AIDSGreedFullScanSupervisorError("receipt lock inode differs")
    for key in (
        "close_pair_contract",
        "close_pair_contract_sha256",
        "pair_semantics_audit_sha256",
        "distance_scan_manifest_sha256",
        "terminal_files",
    ):
        if receipt.get(key) != terminal.get(key):
            raise AIDSGreedFullScanSupervisorError(
                f"receipt terminal science closure differs: {key}"
            )
    return {
        "status": "PASS",
        "receipt": str(path),
        "receipt_sha256": sha256_file(path),
        "science_root": str(science),
        "science_contract": str(contract_path),
        "execution_commit": contract.get("execution_commit"),
        **terminal,
    }


def classify_child_failure(
    *, returncode: int, log_text: str, semantic_markers: Sequence[str]
) -> tuple[str, str]:
    normalized = log_text.lower()
    for marker in semantic_markers:
        if marker.lower() in normalized:
            return "SEMANTIC_OR_PROVENANCE", marker
    if returncode < 0 and -returncode in TRANSIENT_SIGNALS:
        return "TRANSIENT_PROCESS_LOSS", f"signal={-returncode}"
    if returncode in TRANSIENT_EXIT_CODES:
        return "TRANSIENT_PROCESS_LOSS", f"exit_code={returncode}"
    for marker in TRANSIENT_TEXT_MARKERS:
        if marker in normalized:
            return "TRANSIENT_IO", marker
    return "NONRESUMABLE_EXECUTION", f"exit_code={returncode}"


def _publish_receipt(
    *,
    receipt_output: Path,
    contract_path: Path,
    terminal_manifest_path: Path,
    terminal_manifest: Mapping[str, Any],
    terminal: Mapping[str, Any],
    resume_count: int,
    resume_reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if receipt_output.exists() or receipt_output.is_symlink():
        raise AIDSGreedFullScanSupervisorError(
            "controller receipt output must be fresh"
        )
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.mkdir(mode=0o755)
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "science_root": terminal["science_root"],
        "science_root_identity": terminal["science_root_identity"],
        "science_contract": str(contract_path.resolve(strict=True)),
        "science_contract_sha256": sha256_file(contract_path),
        "terminal_supervisor_manifest": str(
            terminal_manifest_path.resolve(strict=True)
        ),
        "terminal_supervisor_manifest_sha256": sha256_file(
            terminal_manifest_path
        ),
        "terminal_owner_identity": terminal_manifest["terminal_owner_identity"],
        "lock_path": terminal_manifest["lock_path"],
        "lock_identity": terminal_manifest["lock_identity"],
        "close_pair_contract": terminal["close_pair_contract"],
        "close_pair_contract_sha256": terminal["close_pair_contract_sha256"],
        "pair_semantics_audit_sha256": terminal[
            "pair_semantics_audit_sha256"
        ],
        "distance_scan_manifest_sha256": terminal[
            "distance_scan_manifest_sha256"
        ],
        "terminal_files": terminal["terminal_files"],
        "same_root_resume_count": int(resume_count),
        "same_root_resume_reasons": list(resume_reasons),
        "controller_expected_output_is_receipt_only": True,
        "large_science_artifacts_copied": False,
        "published_at": _utc_now(),
    }
    receipt_path = receipt_output / RECEIPT_NAME
    _atomic_json(receipt_path, payload)
    descriptor = os.open(
        receipt_output / "PASS", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        os.write(descriptor, b"PASS\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


class _HeldFlock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "_HeldFlock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode):
            os.close(descriptor)
            raise AIDSGreedFullScanSupervisorError(
                "GREED full-scan lock is not a regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise AIDSGreedFullScanSupervisorError(
                "another GREED full-scan supervisor owns the campaign flock"
            ) from exc
        self.descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _pdeathsig(expected_parent_pid: int) -> None:
    """Make the scientific child die with its supervisor on Linux."""

    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        os._exit(126)
    if os.getppid() != int(expected_parent_pid):
        os.kill(os.getpid(), signal.SIGTERM)


def _run_child(
    *,
    argv: Sequence[str],
    log_path: Path,
    proc_root: Path,
    state_path: Path,
    state: dict[str, Any],
    events_path: Path,
    resume: bool,
) -> tuple[int, str, bool]:
    command = [*argv, *(["--resume"] if resume else [])]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False
    child: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, Any] = {}

    def forward(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[int(signum)] = signal.signal(signum, forward)
    chunks: list[bytes] = []
    expected_parent_pid = os.getpid()
    try:
        with log_path.open("ab") as output:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=(
                    functools.partial(_pdeathsig, expected_parent_pid)
                    if sys.platform.startswith("linux")
                    else None
                ),
            )
            child_identity = read_process_identity(child.pid, proc_root=proc_root)
            authorization = state.get("resume_authorization")
            if resume and isinstance(authorization, dict):
                authorization["state"] = "RESUME_LAUNCHED"
                authorization["launched_at"] = _utc_now()
            state.update(
                {
                    "status": "RUNNING",
                    "child_pid": child.pid,
                    "child_identity": child_identity,
                    "child_command": command,
                    "child_command_sha256": stable_json_sha256(command),
                    "child_resume": bool(resume),
                    "child_log": str(log_path),
                    "updated_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
            _append_event(
                events_path,
                {
                    "event": "CHILD_STARTED",
                    "resume": bool(resume),
                    "child_identity": child_identity,
                    "log": str(log_path),
                    "timestamp": _utc_now(),
                },
            )
            assert child.stdout is not None
            while True:
                block = child.stdout.read(64 * 1024)
                if not block:
                    break
                output.write(block)
                output.flush()
                chunks.append(block)
                try:
                    sys.stdout.buffer.write(block)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    pass
            returncode = int(child.wait())
            output.flush()
            os.fsync(output.fileno())
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return returncode, b"".join(chunks).decode("utf-8", errors="replace"), interrupted


def run_supervisor(
    *,
    project_root: str | Path,
    execution_commit: str,
    campaign_root: str | Path,
    science_root: str | Path,
    receipt_output: str | Path,
    proc_root: str | Path,
    max_same_root_resumes: int,
    semantic_failure_markers: Sequence[str],
    child_argv: Sequence[str],
) -> dict[str, Any]:
    project = _absolute(project_root, label="project root", must_exist=True)
    campaign = _absolute(campaign_root, label="campaign root", must_exist=True)
    science = _absolute(science_root, label="science root", must_exist=False)
    receipt = _absolute(receipt_output, label="receipt output", must_exist=False)
    proc = _absolute(proc_root, label="proc root", must_exist=True)
    _under(science, campaign, label="science root")
    _under(receipt, campaign, label="receipt output")
    if receipt.exists() or receipt.is_symlink():
        raise AIDSGreedFullScanSupervisorError(
            "controller receipt output must be fresh"
        )
    if not isinstance(max_same_root_resumes, int) or isinstance(
        max_same_root_resumes, bool
    ) or max_same_root_resumes != 1:
        raise AIDSGreedFullScanSupervisorError(
            "production GREED full scan permits exactly one same-root resume"
        )
    if len(execution_commit) != 40 or _git_head(project) != execution_commit:
        raise AIDSGreedFullScanSupervisorError("execution commit/HEAD differs")
    markers = tuple(
        sorted(
            set(
                marker.lower()
                for marker in [*DEFAULT_SEMANTIC_MARKERS, *semantic_failure_markers]
                if marker
            )
        )
    )
    child_sources = validate_child_command(
        child_argv, project_root=project, science_root=science
    )
    lock_path = science.parent / f".{science.name}{LOCK_NAME}"
    with _HeldFlock(lock_path):
        lock_identity = _file_identity(lock_path)
        lock_identity = {
            "device": lock_identity["device"],
            "inode": lock_identity["inode"],
        }
        science_exists = science.exists()
        if science_exists and (science.is_symlink() or not science.is_dir()):
            raise AIDSGreedFullScanSupervisorError(
                "existing science root is not a physical directory"
            )
        writers = find_live_science_writers(
            science_root=science,
            proc_root=proc,
            exclude_pids=(os.getpid(),),
        )
        if writers:
            raise AIDSGreedFullScanSupervisorError(
                "fixed science root has a live writer: "
                + ",".join(str(item["pid"]) for item in writers)
            )
        control = campaign / CONTROL_DIRECTORY
        fresh_control = not control.exists()
        if fresh_control:
            if science_exists:
                raise AIDSGreedFullScanSupervisorError(
                    "preexisting science root lacks authenticated campaign control"
                )
            control.mkdir(mode=0o700)
        elif control.is_symlink() or not control.is_dir():
            raise AIDSGreedFullScanSupervisorError(
                "campaign supervisor control path is not a physical directory"
            )
        contract_path = control / CONTRACT_NAME
        state_path = control / STATE_NAME
        events_path = control / EVENTS_NAME
        candidate_contract = build_contract(
            project_root=project,
            execution_commit=execution_commit,
            campaign_root=campaign,
            science_root=science,
            lock_path=lock_path,
            lock_identity=lock_identity,
            child_argv=child_argv,
            child_sources=child_sources,
            semantic_markers=markers,
            max_same_root_resumes=max_same_root_resumes,
            proc_root=proc,
        )
        replay_pending_attempt: int | None = None
        if fresh_control:
            _atomic_json(contract_path, candidate_contract)
            state: dict[str, Any] = {
                "schema_version": SUPERVISOR_SCHEMA,
                "status": "INITIALIZED",
                "science_root": str(science),
                "science_root_identity": None,
                "lock_identity": lock_identity,
                "resume_count": 0,
                "resume_reasons": [],
                "child_attempt_count": 0,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
            }
            _atomic_json(state_path, state)
            _append_event(
                events_path,
                {
                    "event": "FRESH_SCIENCE_ROOT_RESERVED",
                    "science_root": str(science),
                    "science_root_absent": True,
                    "timestamp": _utc_now(),
                },
            )
            resume = False
        else:
            contract = _read_object(contract_path, label="science supervisor contract")
            if _contract_identity(contract) != _contract_identity(candidate_contract):
                raise AIDSGreedFullScanSupervisorError(
                    "existing science-root supervisor contract differs"
                )
            state = _read_object(state_path, label="science supervisor state")
            resume = False
            recorded_science_identity = state.get("science_root_identity")
            current_science_identity = (
                _directory_identity(science) if science_exists else None
            )
            if (
                state.get("schema_version") != SUPERVISOR_SCHEMA
                or state.get("science_root") != str(science)
                or (
                    recorded_science_identity is not None
                    and recorded_science_identity != current_science_identity
                )
                or state.get("lock_identity") != lock_identity
            ):
                raise AIDSGreedFullScanSupervisorError(
                    "existing science-root/lock identity differs"
                )
            if (science / "PASS").is_file():
                terminal = validate_terminal_science(
                    science_root=science, contract=contract
                )
                terminal_manifest_path, terminal_manifest = (
                    _freeze_terminal_supervisor_manifest(
                        control_root=control,
                        contract_path=contract_path,
                        terminal=terminal,
                        resume_count=int(state.get("resume_count", 0)),
                        resume_reasons=list(state.get("resume_reasons") or []),
                        owner_identity=(
                            state.get("owner_identity")
                            if isinstance(state.get("owner_identity"), Mapping)
                            else {"pid": None, "restart_adoption": True}
                        ),
                        lock_path=lock_path,
                        lock_identity=lock_identity,
                    )
                )
                receipt_payload = _publish_receipt(
                    receipt_output=receipt,
                    contract_path=contract_path,
                    terminal_manifest_path=terminal_manifest_path,
                    terminal_manifest=terminal_manifest,
                    terminal=terminal,
                    resume_count=int(state.get("resume_count", 0)),
                    resume_reasons=list(state.get("resume_reasons") or []),
                )
                print(
                    "[AIDS_GREED_FULL_SCAN_SUPERVISOR_PASS] "
                    f"resumes={state.get('resume_count', 0)} adopted_terminal=1",
                    flush=True,
                )
                return receipt_payload
            fresh_pre_spawn = (
                not science_exists
                and recorded_science_identity is None
                and not isinstance(state.get("child_identity"), Mapping)
                and state.get("status")
                in {"INITIALIZED", "STARTING_CHILD", "CHILD_SPAWN_PENDING"}
                and all(path.stat().st_size == 0 for path in control.glob("child-attempt-*.log"))
            )
            if fresh_pre_spawn:
                resume = False
                if state.get("status") == "CHILD_SPAWN_PENDING":
                    replay_pending_attempt = int(state["pending_child_attempt"])
            else:
                if not science_exists:
                    raise AIDSGreedFullScanSupervisorError(
                        "started science root disappeared"
                    )
                if recorded_science_identity is None:
                    state["science_root_identity"] = current_science_identity
                    _atomic_json(state_path, state)
            if state.get("status") in {
                "FAILED_SEMANTIC_OR_PROVENANCE",
                "FAILED_NONRESUMABLE_EXECUTION",
                "FAILED_RESUME_EXHAUSTED",
            }:
                raise AIDSGreedFullScanSupervisorError(
                    "existing science root has a terminal nonresumable supervisor state"
                )
            marker = _semantic_log_match(control, markers)
            if marker is not None:
                raise AIDSGreedFullScanSupervisorError(
                    "semantic/provenance failure forbids resume: " + marker
                )
            if process_identity_matches(state.get("owner_identity"), proc_root=proc):
                raise AIDSGreedFullScanSupervisorError(
                    "prior supervisor PID generation is still alive"
                )
            if process_identity_matches(state.get("child_identity"), proc_root=proc):
                raise AIDSGreedFullScanSupervisorError(
                    "prior GREED child PID generation is still alive"
                )
            pending = state.get("resume_authorization")
            pending_resume = (
                isinstance(pending, Mapping)
                and pending.get("state") == "PENDING_RESUME"
            )
            if resume is False and not science_exists:
                pass
            elif pending_resume:
                validate_resume_checkpoint(science_root=science, contract=contract)
                resume = True
            elif int(state.get("resume_count", -1)) >= max_same_root_resumes:
                raise AIDSGreedFullScanSupervisorError(
                    "bounded same-root resume count is exhausted"
                )
            else:
                checkpoint = validate_resume_checkpoint(
                    science_root=science, contract=contract
                )
                reason = {
                    "classification": "TRANSIENT_PROCESS_LOSS",
                    "reason": "prior_supervisor_and_child_generations_absent",
                    "prior_owner_identity": state.get("owner_identity"),
                    "prior_child_identity": state.get("child_identity"),
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "checkpoint_identity_sha256": checkpoint[
                        "checkpoint_identity_sha256"
                    ],
                    "rows_processed": checkpoint["rows_processed"],
                    "verified_at": _utc_now(),
                }
                state["resume_count"] = int(state.get("resume_count", 0)) + 1
                state.setdefault("resume_reasons", []).append(reason)
                state["resume_authorization"] = {
                    "state": "PENDING_RESUME",
                    **reason,
                }
                _atomic_json(state_path, state)
                _append_event(
                    events_path,
                    {"event": "RESTART_RESUME_AUTHORIZED", **reason},
                )
                resume = True

        contract = _read_object(contract_path, label="science supervisor contract")
        owner_identity = read_process_identity(os.getpid(), proc_root=proc)
        if owner_identity is None:
            raise AIDSGreedFullScanSupervisorError(
                "cannot bind current supervisor Linux PID generation"
            )
        state.update(
            {
                "status": "STARTING_CHILD",
                "owner_identity": owner_identity,
                "lock_identity": lock_identity,
                "updated_at": _utc_now(),
            }
        )
        _atomic_json(state_path, state)
        while True:
            child_attempt = (
                replay_pending_attempt
                if replay_pending_attempt is not None
                else int(state.get("child_attempt_count", 0))
            )
            if replay_pending_attempt is None:
                state["child_attempt_count"] = child_attempt + 1
            log_path = control / f"child-attempt-{child_attempt:03d}.log"
            state.update(
                {
                    "status": "CHILD_SPAWN_PENDING",
                    "pending_child_attempt": child_attempt,
                    "pending_child_log": str(log_path),
                    "pending_child_resume": bool(resume),
                    "updated_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
            if log_path.exists():
                if log_path.is_symlink() or log_path.stat().st_size != 0:
                    raise AIDSGreedFullScanSupervisorError(
                        "pre-spawn child log is not an empty physical file"
                    )
            else:
                descriptor = os.open(
                    log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                os.fsync(descriptor)
                os.close(descriptor)
            replay_pending_attempt = None
            returncode, log_text, interrupted = _run_child(
                argv=child_argv,
                log_path=log_path,
                proc_root=proc,
                state_path=state_path,
                state=state,
                events_path=events_path,
                resume=resume,
            )
            if not science.is_dir() or science.is_symlink():
                raise AIDSGreedFullScanSupervisorError(
                    "scientific child did not create a physical science root"
                )
            current_identity = _directory_identity(science)
            if state.get("science_root_identity") is None:
                state["science_root_identity"] = current_identity
            elif state.get("science_root_identity") != current_identity:
                raise AIDSGreedFullScanSupervisorError(
                    "science-root inode changed during child execution"
                )
            if returncode == 0:
                terminal = validate_terminal_science(
                    science_root=science, contract=contract
                )
                state.update(
                    {
                        "status": "PASS",
                        "child_returncode": 0,
                        "terminal": terminal,
                        "updated_at": _utc_now(),
                    }
                )
                _atomic_json(state_path, state)
                _append_event(
                    events_path,
                    {
                        "event": "SCIENCE_TERMINAL_PASS",
                        "resume_count": int(state["resume_count"]),
                        "timestamp": _utc_now(),
                    },
                )
                terminal_manifest_path, terminal_manifest = (
                    _freeze_terminal_supervisor_manifest(
                        control_root=control,
                        contract_path=contract_path,
                        terminal=terminal,
                        resume_count=int(state["resume_count"]),
                        resume_reasons=list(state.get("resume_reasons") or []),
                        owner_identity=owner_identity,
                        lock_path=lock_path,
                        lock_identity=lock_identity,
                    )
                )
                receipt_payload = _publish_receipt(
                    receipt_output=receipt,
                    contract_path=contract_path,
                    terminal_manifest_path=terminal_manifest_path,
                    terminal_manifest=terminal_manifest,
                    terminal=terminal,
                    resume_count=int(state["resume_count"]),
                    resume_reasons=list(state.get("resume_reasons") or []),
                )
                print(
                    "[AIDS_GREED_FULL_SCAN_SUPERVISOR_PASS] "
                    f"resumes={state['resume_count']} adopted_terminal=0",
                    flush=True,
                )
                return receipt_payload
            failure_class, failure_reason = classify_child_failure(
                returncode=returncode,
                log_text=log_text,
                semantic_markers=markers,
            )
            state.update(
                {
                    "status": "CHILD_FAILED",
                    "child_returncode": returncode,
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                    "updated_at": _utc_now(),
                }
            )
            _atomic_json(state_path, state)
            _append_event(
                events_path,
                {
                    "event": "CHILD_FAILED",
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                    "returncode": returncode,
                    "timestamp": _utc_now(),
                },
            )
            if interrupted:
                state["status"] = "INTERRUPTED_TRANSIENT_PROCESS_LOSS"
                _atomic_json(state_path, state)
                print(
                    "[AIDS_GREED_FULL_SCAN_PROCESS_LOSS] resource temporarily "
                    "unavailable; controller restart may consume the exact checkpoint",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(143)
            if failure_class == "SEMANTIC_OR_PROVENANCE":
                state["status"] = "FAILED_SEMANTIC_OR_PROVENANCE"
                _atomic_json(state_path, state)
                print(
                    "[AIDS_GREED_FULL_SCAN_SEMANTIC_FAILURE] " + failure_reason,
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(returncode or 70)
            if failure_class not in {
                "TRANSIENT_PROCESS_LOSS",
                "TRANSIENT_IO",
            }:
                state["status"] = "FAILED_NONRESUMABLE_EXECUTION"
                _atomic_json(state_path, state)
                print(
                    "[AIDS_GREED_FULL_SCAN_NONRESUMABLE_FAILURE] " + failure_reason,
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(returncode or 70)
            if int(state["resume_count"]) >= max_same_root_resumes:
                state["status"] = "FAILED_RESUME_EXHAUSTED"
                _atomic_json(state_path, state)
                print(
                    "[AIDS_GREED_FULL_SCAN_RESUME_EXHAUSTED] " + failure_reason,
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(returncode or 75)
            checkpoint = validate_resume_checkpoint(
                science_root=science, contract=contract
            )
            reason = {
                "classification": failure_class,
                "reason": failure_reason,
                "child_returncode": returncode,
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_identity_sha256": checkpoint[
                    "checkpoint_identity_sha256"
                ],
                "rows_processed": checkpoint["rows_processed"],
                "verified_at": _utc_now(),
            }
            state["resume_count"] = int(state["resume_count"]) + 1
            state.setdefault("resume_reasons", []).append(reason)
            state["status"] = "RESUME_AUTHORIZED"
            state["resume_authorization"] = {
                "state": "PENDING_RESUME",
                **reason,
            }
            _atomic_json(state_path, state)
            _append_event(events_path, {"event": "RESUME_AUTHORIZED", **reason})
            print(
                "[AIDS_GREED_FULL_SCAN_SAME_ROOT_RESUME] "
                f"count={state['resume_count']} rows={checkpoint['rows_processed']} "
                f"reason={failure_class}",
                flush=True,
            )
            resume = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hpc.yaml", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--science-root", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument("--max-same-root-resumes", type=int, default=1)
    parser.add_argument(
        "--semantic-failure-marker", action="append", default=[]
    )
    parser.add_argument("child_argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    child_argv = list(args.child_argv)
    if child_argv and child_argv[0] == "--":
        child_argv = child_argv[1:]
    result = run_supervisor(
        project_root=args.project_root,
        execution_commit=args.execution_commit,
        campaign_root=args.campaign_root,
        science_root=args.science_root,
        receipt_output=args.receipt_output,
        proc_root=args.proc_root,
        max_same_root_resumes=args.max_same_root_resumes,
        semantic_failure_markers=args.semantic_failure_marker,
        child_argv=child_argv,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


__all__ = [
    "AIDSGreedFullScanSupervisorError",
    "DEFAULT_SEMANTIC_MARKERS",
    "LOCK_NAME",
    "RECEIPT_NAME",
    "RECEIPT_SCHEMA",
    "SUPERVISOR_SCHEMA",
    "build_contract",
    "classify_child_failure",
    "find_live_science_writers",
    "main",
    "process_identity_matches",
    "read_process_identity",
    "run_supervisor",
    "validate_child_command",
    "validate_resume_checkpoint",
    "validate_receipt",
    "validate_terminal_science",
]
