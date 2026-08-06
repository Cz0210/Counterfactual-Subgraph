#!/usr/bin/env python3
"""Atomically freeze audited COMRECGC Mutagenicity paper artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.comrecgc.contracts import sha256_file, write_json  # noqa: E402


REQUIRED = (
    "pair_matrix.jsonl",
    "figure3_coverage_vs_k.csv",
    "figure4_coverage_vs_threshold.csv",
    "table2_comrecgc_k10.csv",
    "table2_comrecgc_k20.csv",
    "prefix_metrics.csv",
    "prefix_metrics.json",
    "parent_best_distances.csv",
    "selected_common_recourses.json",
    "representative_counterfactuals.jsonl",
    "selected_sequence.jsonl",
    "summary.json",
    "run_manifest.json",
    "final_artifact_audit.json",
)
EMPTY_ALLOWED = frozenset(
    {
        "pair_matrix.jsonl",
        "selected_sequence.jsonl",
        "representative_counterfactuals.jsonl",
    }
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _materialize(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return "atomic_copy"


def freeze(
    *,
    source_dir: str | Path,
    gate_dir: str | Path,
    output_dir: str | Path,
    dataset: str = "mutagenicity",
    automation_state: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    gate = Path(gate_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Standardized root already exists: {destination}")
    run_manifest = _load(source / "run_manifest.json")
    gate_result = _load(gate / "gate_result.json")
    if run_manifest.get("run_complete") is not True or gate_result.get("audit_passed") is not True:
        raise ValueError("Only a completed full run with a passing full gate may be frozen.")
    if run_manifest.get("method") != "COMRECGC-Adapted-DeterministicChemRepair":
        raise ValueError("Unexpected method in source run manifest.")
    for name in REQUIRED:
        path = source / name
        if not path.is_file() or (
            path.stat().st_size <= 0 and name not in EMPTY_ALLOWED
        ):
            raise FileNotFoundError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        inventory: dict[str, dict[str, Any]] = {}
        for name in REQUIRED:
            source_path = source / name
            target = temporary / name
            mode = _materialize(source_path, target)
            actual_sha = sha256_file(target)
            expected_sha = sha256_file(source_path)
            if actual_sha != expected_sha:
                raise ValueError(f"Frozen artifact SHA256 mismatch: {name}")
            inventory[name] = {
                "bytes": target.stat().st_size,
                "sha256": actual_sha,
                "materialization_mode": mode,
            }
        state_payload: dict[str, Any] | None = None
        state_path: str | None = None
        if automation_state is not None:
            state = Path(automation_state).expanduser().resolve()
            state_payload = _load(state)
            state_path = str(state)
        if dataset not in {"aids", "mutagenicity"}:
            raise ValueError(f"Unsupported COMRECGC freeze dataset: {dataset}")
        freeze_manifest = {
            "schema_version": 1,
            "dataset": "AIDS" if dataset == "aids" else "Mutagenicity",
            "dataset_key": dataset,
            "method": "COMRECGC-Adapted-DeterministicChemRepair",
            "source_output_root": str(source),
            "standardized_output_root": str(destination),
            "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
            "source_gate_result_path": str(gate / "gate_result.json"),
            "source_gate_result_sha256": sha256_file(gate / "gate_result.json"),
            "project_commit": run_manifest.get("project_commit"),
            "upstream_commit": run_manifest.get("upstream_commit"),
            "repair_policy_sha256": run_manifest.get("repair_policy_sha256"),
            "dataset_fingerprint": run_manifest.get("dataset_fingerprint"),
            "teacher_sha256": run_manifest.get("teacher_sha256"),
            "molclr_checkpoint_sha256": run_manifest.get("molclr_checkpoint_sha256"),
            "gate_return_code": 0,
            "automation_state_path": state_path,
            "automation_jobs": (state_payload or {}).get("jobs", []),
            "files": inventory,
            "freeze_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_json(temporary / "freeze_manifest.json", freeze_manifest)
        write_json(
            temporary / "_FINALIZED.json",
            {
                "finalized": True,
                "gate_passed": True,
                "freeze_manifest_sha256": sha256_file(temporary / "freeze_manifest.json"),
            },
        )
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load(destination / "freeze_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("aids", "mutagenicity"), default="mutagenicity")
    parser.add_argument("--automation-state")
    args = parser.parse_args()
    result = freeze(
        source_dir=args.source_dir,
        gate_dir=args.gate_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        automation_state=args.automation_state,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
