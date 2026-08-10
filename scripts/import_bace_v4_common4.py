#!/usr/bin/env python3
"""Atomically import frozen BACE v4 Ours/GCF artifacts into common4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(payload)


def _copy_tree_exact(source: Path, target: Path) -> dict[str, str]:
    source_tree = _tree(source)
    if target.exists():
        if _tree(target) != source_tree:
            raise FileExistsError(f"Existing imported BACE method differs: {target}")
        return source_tree
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True, copy_function=shutil.copy2)
        if _tree(temporary) != source_tree:
            raise IOError(f"BACE artifact copy checksum mismatch: {source}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return source_tree


def _copy_file_exact(source: Path, target: Path) -> str:
    digest = _sha(source)
    if target.exists():
        if _sha(target) != digest:
            raise FileExistsError(f"Existing imported BACE file differs: {target}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle, source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    if _sha(target) != digest:
        raise IOError(f"BACE imported file checksum mismatch: {target}")
    return digest


def _write_exact_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Existing BACE import manifest differs: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def import_v4(*, source_root: str | Path, target_root: str | Path) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    target = Path(target_root).expanduser().resolve()
    protocol_audit_path = source / "bace_connected_protocol_audit.json"
    threshold_path = source / "threshold_protocol/thresholds.json"
    threshold_audit_path = source / "threshold_protocol/threshold_protocol_audit.json"
    gcf_connectivity_path = source / "gcf_candidate_connectivity_audit.json"
    for path in (
        source / "ours/final_artifact_audit.json",
        source / "gcfexplainer/final_artifact_audit.json",
        protocol_audit_path,
        threshold_path,
        threshold_audit_path,
        gcf_connectivity_path,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    protocol = _json(protocol_audit_path)
    threshold_audit = _json(threshold_audit_path)
    gcf_connectivity = _json(gcf_connectivity_path)
    source_method_audits = {
        method: _json(source / method / "final_artifact_audit.json")
        for method in ("ours", "gcfexplainer")
    }
    if any(payload.get("passed") is not True for payload in source_method_audits.values()):
        raise ValueError("Source BACE v4 Ours/GCF artifact audit did not pass.")
    if protocol.get("passed") is not True:
        raise ValueError("Source BACE v4 connected protocol audit did not pass.")
    if not (
        threshold_audit.get("THRESHOLD_METHOD_INDEPENDENT") is True
        and threshold_audit.get("THRESHOLD_TEST_INDEPENDENT") is True
    ):
        raise ValueError("Source BACE v4 threshold protocol is not frozen fairly.")
    if gcf_connectivity.get("all_candidates_connected") is not True:
        raise ValueError("Source BACE v4 GCF candidates are not all connected.")
    if protocol.get("same_cf_mode") is not True:
        raise ValueError("Source BACE v4 protocol is not strict-flip.")
    if protocol.get("action_semantics_version") != "connected_sanitized_residual_v1":
        raise ValueError("Source BACE v4 protocol lacks connected action semantics.")
    if protocol.get("test_used_for_selection") is not False:
        raise ValueError("Source BACE v4 protocol does not prove test exclusion.")
    target.mkdir(parents=True, exist_ok=True)
    methods = {
        method: _copy_tree_exact(source / method, target / method)
        for method in ("ours", "gcfexplainer")
    }
    provenance_files = {
        "thresholds.json": _copy_file_exact(threshold_path, target / "thresholds.json"),
        "source_v4_connected_protocol_audit.json": _copy_file_exact(
            protocol_audit_path, target / "source_v4_connected_protocol_audit.json"
        ),
        "source_v4_threshold_protocol_audit.json": _copy_file_exact(
            threshold_audit_path, target / "source_v4_threshold_protocol_audit.json"
        ),
        "source_v4_gcf_connectivity_audit.json": _copy_file_exact(
            gcf_connectivity_path, target / "source_v4_gcf_connectivity_audit.json"
        ),
    }
    result = {
        "schema_version": "bace_common4_import_v1",
        "passed": True,
        "source_root": str(source),
        "target_root": str(target),
        "methods": methods,
        "provenance_files": provenance_files,
        "artifacts_recomputed": False,
        "test_reexecuted": False,
        "source_artifacts_unchanged": True,
    }
    _write_exact_json(target / "v4_import_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--set", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    args = parser.parse_args(argv)
    result = import_v4(source_root=args.source_root, target_root=args.target_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("[BACE_V4_COMMON4_IMPORT_PASS]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
