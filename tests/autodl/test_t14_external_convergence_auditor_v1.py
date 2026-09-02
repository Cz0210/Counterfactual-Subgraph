from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.baselines.comrecgc.contracts import stable_json_sha256
from src.baselines.tastemolnet_comrecgc_smoke import _identity_graph_sha256
from src.eval import tastemolnet_t14_external_convergence as audit


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _checkpoint(root: Path, step: int) -> tuple[audit.CommittedCheckpoint, dict[str, Any]]:
    checkpoint = root / f"step-{step:012d}"
    checkpoint.mkdir(parents=True)
    state_path = checkpoint / "generation_state.pt"
    sqlite_path = checkpoint / "authoritative_graph_store.sqlite3"
    state_path.write_bytes(b"state")
    sqlite_path.write_bytes(b"sqlite-never-open")
    provenance = {
        "dataset": "tastemolnet",
        "method": "comrecgc",
        "stage": "T14_COMRECGC_FULL",
        "runtime_state_schema": audit.RUNTIME_SCHEMA,
        "total_steps": "25000",
        "train_csv_sha256": "a" * 64,
        "scientific_command_sha256": "b" * 64,
    }
    manifest: dict[str, Any] = {
        "schema_version": audit.CHECKPOINT_SCHEMA,
        "state_schema_version": audit.STATE_SCHEMA,
        "file_digest_algorithm": "sha256",
        "checkpoint_digest_scheme": "stable_json_sha256_v1",
        "boundary": audit.BOUNDARY,
        "atomic_complete": True,
        "checkpoint_dir": checkpoint.name,
        "completed_step": step,
        "next_step": step + 1,
        "total_steps": 25_000,
        "provenance_fingerprints": provenance,
        "provenance_sha256": stable_json_sha256(provenance),
        "scientific_argv": ["run-t14", "--train-csv", "/train.csv"],
        "command_sha256": "b" * 64,
        "files": {
            "generation_state.pt": {
                "bytes": state_path.stat().st_size,
                "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            },
            "authoritative_graph_store.sqlite3": {
                "bytes": sqlite_path.stat().st_size,
                "sha256": hashlib.sha256(sqlite_path.read_bytes()).hexdigest(),
            },
        },
    }
    manifest["checkpoint_digest"] = stable_json_sha256(manifest)
    manifest_sha = _write_json(checkpoint / "checkpoint_manifest.json", manifest)
    _write_json(
        checkpoint / "_CHECKPOINT_COMPLETE.json",
        {
            "checkpoint_digest": manifest["checkpoint_digest"],
            "manifest_sha256": manifest_sha,
            "schema_version": audit.CHECKPOINT_SCHEMA,
        },
    )
    return audit.validate_committed_checkpoint(checkpoint), provenance


def _graph(index: int) -> tuple[str, dict[str, Any]]:
    collision = {
        "canonical_graph": "C" * (index + 1),
        "num_nodes": index + 1,
        "num_edges": max(0, index),
    }
    return _identity_graph_sha256(collision), collision


def _state(
    step: int,
    provenance: dict[str, str],
    *,
    coverage: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    records: dict[str, Any] = {}
    collisions: dict[str, Any] = {}
    lineages: dict[str, Any] = {}
    parent = "d" * 64
    for index in range(25):
        graph_hash, collision = _graph(index)
        candidates.append(
            {"graph_hash": graph_hash, "frequency": 100 - index}
        )
        collisions[graph_hash] = collision
        records[graph_hash] = {
            "graph_identity_sha256": graph_hash,
            "canonical_graph": collision["canonical_graph"],
            "prediction": 0 if index % 2 == 0 else 2,
            "candidate": True,
            "valid_fullgraph": True,
        }
        lineages[graph_hash] = {parent: 1}
    return {
        "schema_version": audit.STATE_SCHEMA,
        "boundary": audit.BOUNDARY,
        "fully_completed_step": True,
        "completed_step": step,
        "next_step": step + 1,
        "provenance_sha256": stable_json_sha256(provenance),
        "algorithm_state": {
            "schema_version": audit.RUNTIME_SCHEMA,
            "official_state": {
                "counterfactual_candidates": candidates,
                "input_graphs_covered": [1.0] * int(coverage * 1_000)
                + [0.0] * (1_000 - int(coverage * 1_000)),
            },
            "bridge_state": {
                "schema_version": "tastemolnet_comrecgc_bridge_checkpoint_v3",
                "records": records,
                "graph_collision_payloads": collisions,
                "lineage_occurrences": lineages,
            },
        },
    }


def test_t14_external_auditor_waits_for_12500_without_loading_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    _checkpoint(root, 5_000)
    _checkpoint(root, 10_000)

    def forbidden(_path: Path) -> dict[str, Any]:
        raise AssertionError("state loader must not run before 12.5k")

    result = audit.audit_t14_external_convergence(root, state_loader=forbidden)
    assert result["status"] == "WAITING_FOR_12500"
    assert result["checkpoint_state_loaded"] is False
    assert result["sqlite_accessed"] is False


def test_t14_checkpoint_validation_only_stats_sqlite(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoint, _ = _checkpoint(root, 5_000)
    # If validation tried to open SQLite, mode 000 would fail; lstat remains valid.
    checkpoint.sqlite_path.chmod(0)
    validated = audit.validate_committed_checkpoint(checkpoint.root)
    assert validated.sqlite_bytes == len(b"sqlite-never-open")
    assert validated.sqlite_sha256 == checkpoint.sqlite_sha256


def test_t14_external_convergence_auditor_two_windows_pass(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    states: dict[Path, dict[str, Any]] = {}
    for step, coverage in ((5_000, 0.500), (10_000, 0.503), (12_500, 0.505)):
        checkpoint, provenance = _checkpoint(root, step)
        states[checkpoint.state_path] = _state(
            step, provenance, coverage=coverage
        )
    result = audit.audit_t14_external_convergence(
        root, state_loader=lambda path: states[path]
    )
    assert result["status"] == "CONVERGED_EARLY_STOP"
    assert result["converged"] is True
    assert result["consecutive_passing_windows"] == 2
    assert all(window["pass"] for window in result["windows"])
    assert result["sqlite_accessed"] is False
    assert result["test_loaded"] is False


def test_t14_external_auditor_fails_lineage_gate(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    states: dict[Path, dict[str, Any]] = {}
    for step in (5_000, 10_000, 12_500):
        checkpoint, provenance = _checkpoint(root, step)
        state = _state(step, provenance, coverage=0.5)
        if step == 12_500:
            state["algorithm_state"]["bridge_state"]["lineage_occurrences"].clear()
        states[checkpoint.state_path] = state
    result = audit.audit_t14_external_convergence(
        root, state_loader=lambda path: states[path]
    )
    assert result["status"] == "CONTINUE_T14"
    assert result["windows"][-1]["lineage_error_count"] > 0
    assert result["safe_stop_authorized"] is False


def test_t14_auditor_has_no_signal_or_sqlite_read_path() -> None:
    project = Path(__file__).resolve().parents[2]
    core = (
        project / "src/eval/tastemolnet_t14_external_convergence.py"
    ).read_text(encoding="utf-8")
    cli = (
        project / "scripts/autodl/run_t14_external_convergence_auditor_v1.py"
    ).read_text(encoding="utf-8")
    assert "sqlite3.connect" not in core
    assert "os.kill" not in core
    assert "os.kill" not in cli
    assert "SIGTERM" not in core
    assert "signal_sent" in cli


def test_t14_auditor_rejects_calibration_or_test_generation_argv(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoint, _ = _checkpoint(root, 5_000)
    manifest_path = checkpoint.root / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scientific_argv"].extend(["--test-csv", "/test.csv"])
    manifest["checkpoint_digest"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "checkpoint_digest"}
    )
    manifest_sha = _write_json(manifest_path, manifest)
    _write_json(
        checkpoint.root / "_CHECKPOINT_COMPLETE.json",
        {
            "checkpoint_digest": manifest["checkpoint_digest"],
            "manifest_sha256": manifest_sha,
            "schema_version": audit.CHECKPOINT_SCHEMA,
        },
    )
    with pytest.raises(audit.T14ExternalConvergenceError, match="Calibration/test"):
        audit.validate_committed_checkpoint(checkpoint.root)

