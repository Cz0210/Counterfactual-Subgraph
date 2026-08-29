from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.eval.bace_comrecgc_convergence as convergence
from src.baselines.comrecgc.contracts import stable_json_sha256


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(root: Path, policy: convergence.ConvergencePolicy) -> tuple[Path, str, str]:
    parent_ids = [f"BACE_{index:03d}" for index in range(policy.parent_count)]
    parent_sha = stable_json_sha256(parent_ids)
    value = {
        "dataset": "bace",
        "mode": "full",
        "parent_limit": policy.parent_count,
        "total_steps": policy.m_max,
        "parameters": {"steps": policy.m_max, "heads": policy.heads},
        "generation_parent_ids": parent_ids,
        "generation_parent_ids_sha256": parent_sha,
        "generation_checkpoint_interval_steps": 2,
        "scientific_argv": ["run_generation.py", "--dataset=bace"],
        "command_sha256": "c" * 64,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
    }
    value["config_sha256"] = stable_json_sha256(value)
    path = root / "resolved_config.json"
    _json(path, value)
    return path, value["config_sha256"], parent_sha


def _checkpoint_manifest(
    directory: Path,
    *,
    step: int,
    config: dict[str, object],
) -> dict[str, object]:
    directory.mkdir(parents=True)
    state = directory / convergence.CHECKPOINT_STATE_FILENAME
    sqlite = directory / convergence.CHECKPOINT_SQLITE_FILENAME
    state.write_bytes(b"opaque torch payload")
    sqlite.write_bytes(b"opaque sqlite payload")
    provenance = {
        "dataset": "bace",
        "generation_parent_ids_sha256": config["generation_parent_ids_sha256"],
        "mode": "full",
        "parameters_sha256": stable_json_sha256(config["parameters"]),
        "scientific_command_sha256": config["command_sha256"],
        "total_steps": str(config["total_steps"]),
    }
    manifest: dict[str, object] = {
        "schema_version": convergence.CHECKPOINT_SCHEMA_VERSION,
        "state_schema_version": convergence.CHECKPOINT_STATE_SCHEMA_VERSION,
        "file_digest_algorithm": "sha256",
        "checkpoint_digest_scheme": "stable_json_sha256_v1",
        "boundary": convergence.CHECKPOINT_BOUNDARY,
        "atomic_complete": True,
        "checkpoint_dir": directory.name,
        "completed_step": step,
        "next_step": step + 1,
        "provenance_fingerprints": provenance,
        "provenance_sha256": stable_json_sha256(provenance),
        "scientific_argv": config["scientific_argv"],
        "command_sha256": config["command_sha256"],
        "total_steps": config["total_steps"],
        "files": {
            state.name: {"bytes": state.stat().st_size, "sha256": _sha(state)},
            sqlite.name: {"bytes": sqlite.stat().st_size, "sha256": _sha(sqlite)},
        },
    }
    manifest["checkpoint_digest"] = stable_json_sha256(manifest)
    _json(directory / convergence.CHECKPOINT_MANIFEST_FILENAME, manifest)
    _json(
        directory / convergence.CHECKPOINT_COMPLETE_FILENAME,
        {
            "checkpoint_digest": manifest["checkpoint_digest"],
            "manifest_sha256": _sha(
                directory / convergence.CHECKPOINT_MANIFEST_FILENAME
            ),
            "schema_version": convergence.CHECKPOINT_SCHEMA_VERSION,
        },
    )
    return manifest


def _live_checkpoint_pair(
    local_root: Path,
    mirror_root: Path,
    *,
    step: int,
    config: dict[str, object],
) -> str:
    name = f"step-{step:012d}"
    local = _checkpoint_manifest(local_root / name, step=step, config=config)
    mirror = _checkpoint_manifest(mirror_root / name, step=step, config=config)
    assert local == mirror
    marker = {
        "schema_version": "comrecgc_generation_checkpoint_mirror_v1",
        "checkpoint_mirrored": True,
        "completed_step": step,
        "checkpoint_digest": local["checkpoint_digest"],
        "source_checkpoint": str(local_root / name),
        "mirror_checkpoint": str(mirror_root / name),
        "mirrored_at": "2026-08-30T00:00:00+00:00",
    }
    _json(local_root / name / convergence.CHECKPOINT_MIRRORED_FILENAME, marker)
    _json(mirror_root / name / convergence.CHECKPOINT_MIRRORED_FILENAME, marker)
    return str(local["checkpoint_digest"])


def _retention_pair(
    local_root: Path, mirror_root: Path, *, step: int, digest: str
) -> None:
    name = f"step-{step:012d}"
    value = {
        "schema_version": "comrecgc_generation_checkpoint_retention_v1",
        "checkpoint_mirrored": True,
        "completed_step": step,
        "checkpoint_digest": digest,
        "local_checkpoint": str(local_root / name),
        "mirror_checkpoint": str(mirror_root / name),
        "mirror_marker_sha256": "d" * 64,
        "retention_keep_last": 2,
        "pruned_at": "2026-08-30T00:00:00+00:00",
    }
    _json(local_root / convergence.RETENTION_HISTORY_DIRNAME / f"{name}.json", value)
    _json(mirror_root / convergence.RETENTION_HISTORY_DIRNAME / f"{name}.json", value)


def _trace(chunks: Path, policy: convergence.ConvergencePolicy) -> None:
    chunks.mkdir(parents=True)
    for move in range(policy.m_max + 1):
        rows = [
            {
                "event": "selected_transition",
                "action": ["NLC", head, head + 1],
                "action_resolution": "exact",
                "head_index": head,
                "move_index": move,
                "parent_id": f"BACE_{head % policy.parent_count:03d}",
                "source_graph_sha256": f"{(head + 10):064x}",
                "target_graph_sha256": f"{(head + 100):064x}",
            }
            for head in range(policy.heads)
        ]
        assert len(rows) == policy.trace_chunk_rows
        (chunks / f"part-{move:06d}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_public_audit_uses_retention_and_never_opens_large_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = convergence.ConvergencePolicy(
        m_max=10,
        m_min=6,
        check_interval=2,
        patience_checks=2,
        minimum_valid_unique_count=2,
        parent_count=2,
        heads=5,
        trace_chunk_rows=5,
        top100_size=100,
        top20_size=20,
        missing_rank=101,
    )
    monkeypatch.setattr(convergence, "POLICY", policy)
    source = tmp_path / "source"
    source.mkdir()
    config_path, config_sha, parent_sha = _config(source, policy)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    local = source / "local"
    mirror = source / "mirror"
    local.mkdir()
    mirror.mkdir()
    digest6 = _live_checkpoint_pair(local, mirror, step=6, config=config)
    _retention_pair(local, mirror, step=6, digest=digest6)
    for directory in (local / "step-000000000006", mirror / "step-000000000006"):
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
    _live_checkpoint_pair(local, mirror, step=8, config=config)
    _live_checkpoint_pair(local, mirror, step=10, config=config)
    chunks = source / "trace" / "selected_action_trace_chunks"
    _trace(chunks, policy)
    state_before = {
        path: path.read_bytes()
        for path in source.rglob("*")
        if path.name in {
            convergence.CHECKPOINT_STATE_FILENAME,
            convergence.CHECKPOINT_SQLITE_FILENAME,
        }
    }

    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name in {
            convergence.CHECKPOINT_STATE_FILENAME,
            convergence.CHECKPOINT_SQLITE_FILENAME,
        }:
            raise AssertionError(f"large payload was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    audit_root = tmp_path / "audits" / "step10"
    audit_root.parent.mkdir()
    result = convergence.audit_bace_comrecgc_convergence(
        resolved_config_path=config_path,
        trace_chunks_dir=chunks,
        local_checkpoint_root=local,
        mirror_checkpoint_root=mirror,
        audit_root=audit_root,
        evaluation_step=10,
        expected_config_sha256=config_sha,
        expected_parent_ids_sha256=parent_sha,
    )

    assert result["status"] == "CONVERGED_EARLY_STOP"
    assert result["m_effective"] == 10
    assert [row["kind"] for row in result["checkpoint_evidence"]] == [
        "paired_retention_history",
        "live_local_and_mirror",
        "live_local_and_mirror",
    ]
    assert all(row["pass"] for row in result["windows"])
    assert (audit_root / "convergence.json").is_file()
    assert (audit_root / "CONVERGED_EARLY_STOP.json").is_file()
    for path, payload in state_before.items():
        with original_open(path, "rb") as handle:
            assert handle.read() == payload


def test_last_trace_part_is_not_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    policy = convergence.ConvergencePolicy(
        m_max=10,
        m_min=6,
        check_interval=2,
        parent_count=1,
        heads=1,
        trace_chunk_rows=1,
    )
    monkeypatch.setattr(convergence, "POLICY", policy)
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for move in range(2):
        row = {
            "event": "teleport",
            "move_index": move,
            "source_official_hashes": ["source"],
        }
        (chunks / f"part-{move:06d}.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
    scan = convergence._read_closed_trace(chunks)
    assert scan.closed_row_count == 1
    assert scan.closed_through_move == 0
    assert set(scan.move_rows) == {0}


def test_rank_metric_is_direct_pearson_with_missing_rank_101() -> None:
    left = [f"{value:064x}" for value in range(100)]
    right = [*left[:90], *[f"{value:064x}" for value in range(200, 210)]]
    actual = convergence._rank_spearman(left, right)
    union = sorted(set(left) | set(right))
    left_rank = {value: index + 1 for index, value in enumerate(left)}
    right_rank = {value: index + 1 for index, value in enumerate(right)}
    expected = convergence._pearson(
        [left_rank.get(value, 101) for value in union],
        [right_rank.get(value, 101) for value in union],
    )
    assert actual == pytest.approx(expected, abs=1e-15)


def test_selected_transition_lineage_errors_block_window() -> None:
    previous = {
        "step": 1,
        "top100": ["a" * 64, "b" * 64],
        "top20": ["a" * 64, "b" * 64],
        "train_coverage": 0.5,
        "lineage_error_count": 0,
        "valid_unique_count": 20,
    }
    current = {**previous, "step": 2, "lineage_error_count": 1}
    window = convergence._window_metrics(previous, current)
    assert window["lineage_error_count"] == 1
    assert window["gates"]["lineage_error_count"] is False
    assert window["pass"] is False
