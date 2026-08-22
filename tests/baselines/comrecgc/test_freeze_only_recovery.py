from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.comrecgc.contracts import GenerationParameters, sha256_file, write_json
from src.baselines.comrecgc import freeze_recovery
from src.baselines.comrecgc.freeze_recovery import (
    UnsafeCompletedGenerationFreezeError,
    recover_completed_generation_freeze,
    validate_completed_generation_freeze,
)
from src.baselines.comrecgc.graph_trace import stable_untyped_graph_sha256
from src.baselines.comrecgc.live_graph_state import AuthoritativeGraphStore


torch = pytest.importorskip("torch")


def _graph(values: list[int], edges: list[tuple[int, int]]) -> SimpleNamespace:
    return SimpleNamespace(
        x=np.asarray([[value] for value in values], dtype=np.int64),
        edge_index=np.asarray(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64),
        num_nodes=len(values),
        comrecgc_parent_id="parent",
    )


def _source_root(
    tmp_path,
    *,
    invalid_destinations: int = 0,
    malformed_serialized_transition: bool = False,
):
    root = tmp_path / "generation"
    trace = root / "trace"
    chunks = trace / "selected_action_trace_chunks"
    graph_state = root / "graph_state"
    chunks.mkdir(parents=True)
    graph_state.mkdir(parents=True)
    source = _graph([1, 2], [(0, 1), (1, 0)])
    target = _graph([2], [])
    store = AuthoritativeGraphStore(
        graph_state / "authoritative_graph_store.sqlite3"
    )
    store.put("source", [source, np.asarray([1.0]), np.asarray([2.0])])
    store_audit = store.integrity_audit()
    store.close()
    payload = {
        "graph_map": {
            "target": [target, np.asarray([1.0]), np.asarray([2.0])]
        },
        "counterfactual_candidates": [{"graph_hash": "target", "frequency": 1}],
        "traversed_hashes": ["source", "target"],
    }
    if malformed_serialized_transition:
        payload["transitions"] = {
            "source": (["target", "missing"], [target])
        }
    torch.save(payload, root / "counterfactuals.pt")
    event = {
        "move_index": 49_999,
        "head_index": 0,
        "event": "selected_transition",
        "parent_id": "parent",
        "source_official_hash": "source",
        "target_official_hash": "target",
        "source_graph_sha256": stable_untyped_graph_sha256(source),
        "target_graph_sha256": stable_untyped_graph_sha256(target),
        "action": ["NR", 0, 0],
    }
    chunk = chunks / "part-000000.jsonl"
    chunk.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    write_json(
        trace / "selected_action_trace_manifest.json",
        {
            "schema_version": 1,
            "format": "chunked_jsonl",
            "row_count": 1,
            "chunks": [
                {
                    "index": 0,
                    "path": "selected_action_trace_chunks/part-000000.jsonl",
                    "row_count": 1,
                    "bytes": chunk.stat().st_size,
                    "sha256": sha256_file(chunk),
                }
            ],
        },
    )
    write_json(
        root / "resolved_config.json",
        {
            "dataset": "aids",
            "mode": "full",
            "project_commit": "base",
            "parent_limit": 1,
            "parameters": GenerationParameters.for_mode("full").__dict__,
        },
    )
    write_json(
        root / "_RUN_FAILED.json",
        {
            "stage": "project_generation",
            "message": "Selected trace references a graph absent from the frozen payload.",
        },
    )
    write_json(
        root / "graph_state_audit.json",
        {
            "move_count": 50_000,
            "unresolved_lookups": 0,
            "unresolved_transition_source_count": 0,
            "invalid_transition_destination_count": invalid_destinations,
            "backing_store": store_audit,
        },
    )
    return root


def test_completed_walk_is_safe_for_freeze_without_rng_resume_state(tmp_path) -> None:
    root = _source_root(tmp_path)

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["random_walk_complete"] is True
    assert audit["RNG_state_present"] is False
    assert audit["rng_state_required_for_freeze_only"] is False
    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is True
    assert payload is not None
    assert "source" in payload["graph_map"]


def test_completed_mut_walk_accepts_exact_recorded_action_failure_signature(
    tmp_path,
) -> None:
    root = _source_root(tmp_path)
    config_path = root / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"] = "mutagenicity"
    write_json(config_path, config)
    write_json(
        root / "_RUN_FAILED.json",
        {
            "stage": "project_generation",
            "message": (
                "Selected COMRECGC transition is not one unique "
                "pinned-upstream single edit."
            ),
        },
    )

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="mutagenicity",
        dataset_dir=tmp_path / "unused",
        source_csv=None,
        expected_project_commit="base",
    )

    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is True
    assert audit["matched_post_generation_failure_signatures"] == [
        "recorded_action_lineage_ambiguity"
    ]
    assert payload is not None


def test_project_commit_gate_reports_wrong_expected_and_accepts_actual(
    tmp_path,
) -> None:
    root = _source_root(tmp_path)
    fixture = json.loads(
        (
            Path(__file__).parents[2]
            / "fixtures/comrecgc_lineage/mutagenicity_recovery_counts.json"
        ).read_text(encoding="utf-8")
    )
    regression = fixture["failed_v2_project_commit_gate"]
    actual_commit = regression["actual_project_commit"]
    incorrect_expected = regression["incorrect_expected_project_commit"]
    config_path = root / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset"] = "mutagenicity"
    config["project_commit"] = actual_commit
    write_json(config_path, config)

    wrong, wrong_payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="mutagenicity",
        dataset_dir=tmp_path / "unused",
        source_csv=None,
        expected_project_commit=incorrect_expected,
    )
    actual, actual_payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="mutagenicity",
        dataset_dir=tmp_path / "unused",
        source_csv=None,
        expected_project_commit=actual_commit,
    )

    assert wrong["actual_project_commit"] == actual_commit
    assert wrong["expected_project_commit"] == incorrect_expected
    assert wrong["project_commit_identity"] == {
        "actual": actual_commit,
        "expected": incorrect_expected,
        "actual_type": "str",
        "expected_type": "str",
        "actual_repr": repr(actual_commit),
        "expected_repr": repr(incorrect_expected),
        "actual_length": regression["actual_length"],
        "expected_length": regression["incorrect_expected_length"],
        "matches": False,
    }
    assert wrong["checks"]["project_commit_matches"] is False
    assert wrong["FREEZE_ONLY_RECOVERY_SAFE"] is False
    assert wrong_payload is None
    assert actual["actual_project_commit"] == actual_commit
    assert actual["expected_project_commit"] == actual_commit
    assert actual["project_commit_identity"]["matches"] is True
    assert actual["project_commit_identity"]["actual_repr"] == repr(actual_commit)
    assert actual["project_commit_identity"]["expected_repr"] == repr(actual_commit)
    assert actual["checks"]["project_commit_matches"] is True
    assert actual["FREEZE_ONLY_RECOVERY_SAFE"] is True
    assert actual_payload is not None


def test_known_failure_signature_with_extra_text_remains_fail_closed(
    tmp_path,
) -> None:
    root = _source_root(tmp_path)
    write_json(
        root / "_RUN_FAILED.json",
        {
            "stage": "project_generation",
            "message": (
                "Selected trace references a graph absent from the frozen payload. "
                "Additional unrelated failure"
            ),
        },
    )

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["matched_post_generation_failure_signatures"] == []
    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is False
    assert payload is None


def test_unknown_post_generation_failure_signature_remains_fail_closed(
    tmp_path,
) -> None:
    root = _source_root(tmp_path)
    write_json(
        root / "_RUN_FAILED.json",
        {"stage": "project_generation", "message": "unrelated runtime failure"},
    )

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["checks"]["failure_is_post_generation_freeze"] is False
    assert audit["matched_post_generation_failure_signatures"] == []
    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is False
    assert payload is None


def test_historical_transition_cache_mismatch_is_diagnostic_after_completed_walk(
    tmp_path,
) -> None:
    root = _source_root(tmp_path, invalid_destinations=1)

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is True
    assert audit["serialized_transition_state_present"] is False
    assert audit["historical_transition_state_required_for_freeze_only"] is False
    assert audit["historical_transition_audit"]["invalid_destination_count"] == 1
    assert audit["historical_transition_audit"]["passed"] is False
    assert payload is not None


def test_malformed_serialized_transition_blocks_freeze_only(tmp_path) -> None:
    root = _source_root(tmp_path, malformed_serialized_transition=True)

    audit, payload = validate_completed_generation_freeze(
        source_generation_dir=root,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is False
    assert audit["serialized_transition_state_present"] is True
    assert audit["checks"]["serialized_transition_closure_complete"] is False
    assert payload is None


def test_unsafe_recovery_persists_one_complete_validation_audit_before_failure(
    tmp_path, monkeypatch
) -> None:
    root = _source_root(tmp_path, malformed_serialized_transition=True)
    output = tmp_path / "failed-fresh-root"
    audit_output = output / "fresh_recovery_audit.json"
    real_validate = freeze_recovery.validate_completed_generation_freeze
    validation_calls = 0

    def counted_validate(**kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return real_validate(**kwargs)

    monkeypatch.setattr(
        freeze_recovery,
        "validate_completed_generation_freeze",
        counted_validate,
    )

    with pytest.raises(UnsafeCompletedGenerationFreezeError) as raised:
        recover_completed_generation_freeze(
            source_generation_dir=root,
            output_dir=output,
            dataset="aids",
            dataset_dir=tmp_path / "unused",
            source_csv=tmp_path / "unused.csv",
            expected_project_commit="base",
            audit_output=audit_output,
        )

    persisted = json.loads(audit_output.read_text(encoding="utf-8"))
    assert validation_calls == 1
    assert persisted == raised.value.audit
    assert raised.value.audit_output == audit_output.resolve()
    assert persisted["FREEZE_ONLY_RECOVERY_SAFE"] is False
    assert persisted["checks"]["serialized_transition_closure_complete"] is False
    assert "closure_error" in persisted
    assert not (output / "_RUN_COMPLETE.json").exists()
    assert not list(output.glob(".*.tmp"))


def test_unsafe_recovery_defaults_failure_audit_inside_fresh_root(tmp_path) -> None:
    root = _source_root(tmp_path, malformed_serialized_transition=True)
    output = tmp_path / "failed-default-audit-root"

    with pytest.raises(UnsafeCompletedGenerationFreezeError) as raised:
        recover_completed_generation_freeze(
            source_generation_dir=root,
            output_dir=output,
            dataset="aids",
            dataset_dir=tmp_path / "unused",
            source_csv=tmp_path / "unused.csv",
            expected_project_commit="base",
        )

    expected = (output / "fresh_recovery_audit.json").resolve()
    assert raised.value.audit_output == expected
    assert json.loads(expected.read_text(encoding="utf-8")) == raised.value.audit


def _immutable_file_state(path):
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def test_freeze_validation_never_mutates_readonly_sqlite_snapshot(tmp_path) -> None:
    root = _source_root(tmp_path)
    graph_state = root / "graph_state"
    database = graph_state / "authoritative_graph_store.sqlite3"
    observed_paths = (database, Path(f"{database}-wal"), Path(f"{database}-shm"))
    before = {path.name: _immutable_file_state(path) for path in observed_paths}
    os.chmod(database, 0o444)
    os.chmod(graph_state, 0o555)
    try:
        audit, payload = validate_completed_generation_freeze(
            source_generation_dir=root,
            dataset="aids",
            dataset_dir=tmp_path / "unused",
            source_csv=tmp_path / "unused.csv",
            expected_project_commit="base",
        )
        after = {path.name: _immutable_file_state(path) for path in observed_paths}
    finally:
        os.chmod(graph_state, 0o755)
        os.chmod(database, 0o644)

    assert audit["FREEZE_ONLY_RECOVERY_SAFE"] is True
    assert payload is not None
    assert after == before


def test_freeze_recovery_exposes_recorded_action_first_counters(
    tmp_path, monkeypatch
) -> None:
    root = _source_root(tmp_path)
    source_database = root / "graph_state/authoritative_graph_store.sqlite3"
    source_database_sha256 = sha256_file(source_database)
    source = _graph([1, 2], [(0, 1), (1, 0)])
    monkeypatch.setattr(
        freeze_recovery,
        "_load_sources",
        lambda **_kwargs: ({"parent": source}, "dataset-fingerprint"),
    )

    output = tmp_path / "recovered"
    recovery = recover_completed_generation_freeze(
        source_generation_dir=root,
        output_dir=output,
        dataset="aids",
        dataset_dir=tmp_path / "unused",
        source_csv=tmp_path / "unused.csv",
        expected_project_commit="base",
    )

    recovered_database = output / "graph_state/authoritative_graph_store.sqlite3"
    assert recovery["materialization"]["backing_store"] == "atomic_copy"
    assert source_database.stat().st_ino != recovered_database.stat().st_ino
    assert sha256_file(source_database) == source_database_sha256
    assert sha256_file(recovered_database) == source_database_sha256

    closure_audit = json.loads(
        (output / "frozen_payload_closure_audit.json").read_text(encoding="utf-8")
    )
    recovered_payload = freeze_recovery.torch_load_payload(
        output / "counterfactuals.pt"
    )
    assert closure_audit["canonical_graph_records_persisted"] is True
    assert closure_audit["alias_to_canonical_persisted"] is True
    assert closure_audit["original_trace_hashes_persisted"] is True
    assert closure_audit["canonical_graph_records_roundtrip_verified"] is True
    assert closure_audit["alias_to_canonical_roundtrip_verified"] is True
    assert closure_audit["original_trace_hashes_roundtrip_verified"] is True
    assert closure_audit["alias_count"] == 0
    assert closure_audit["original_trace_hash_roundtrip_count"] > 0
    assert recovered_payload["alias_to_canonical"] == {}
    assert closure_audit["original_trace_hash_roundtrip_count"] == len(
        recovered_payload["original_trace_hashes"]
    )
    adoption = json.loads(
        (output / "adoption_manifest.json").read_text(encoding="utf-8")
    )
    assert adoption["generation_mode"] == "adopted_read_only_cache"
    assert adoption["adopted_from"] == str(root.resolve())
    assert adoption["serialization_rerun"] is True
    assert adoption["lineage_resolution_rerun"] is True
    assert adoption["freeze_rerun"] is True
    assert adoption["bare_symlink_used"] is False
    assert adoption["source_checksums"][
        "graph_state/authoritative_graph_store.sqlite3"
    ] == source_database_sha256

    documents = [
        json.loads(
            (output / "trace" / "candidate_action_lineage.json").read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (output / "trace" / "trace_summary.json").read_text(encoding="utf-8")
        ),
        json.loads((output / "run_manifest.json").read_text(encoding="utf-8")),
        recovery,
    ]
    for document in documents:
        assert document["recorded_action_present_count"] == 1
        assert document["recorded_action_replay_ok_count"] == 1
        assert document["recorded_action_replay_mismatch_count"] == 0
        assert document["legacy_missing_action_count"] == 0
        assert document["legacy_inference_called_count"] == 0
        assert document["lineage_recovery_audit"][
            "recorded_action_replay_ok_count"
        ] == 1


def test_freeze_recovery_rejects_missing_serialized_original_trace_hashes(
    tmp_path, monkeypatch
) -> None:
    root = _source_root(tmp_path)
    source = _graph([1, 2], [(0, 1), (1, 0)])
    monkeypatch.setattr(
        freeze_recovery,
        "_load_sources",
        lambda **_kwargs: ({"parent": source}, "dataset-fingerprint"),
    )
    real_save = freeze_recovery.atomic_torch_save

    def save_without_original_hashes(payload, path) -> None:
        real_save(payload, path)
        corrupted = copy.deepcopy(freeze_recovery.torch_load_payload(path))
        corrupted["original_trace_hashes"] = corrupted["original_trace_hashes"][1:]
        real_save(corrupted, path)

    monkeypatch.setattr(
        freeze_recovery, "atomic_torch_save", save_without_original_hashes
    )
    output = tmp_path / "missing-original-hashes"

    with pytest.raises(
        RuntimeError, match="frozen-closure fields changed across serialization"
    ):
        recover_completed_generation_freeze(
            source_generation_dir=root,
            output_dir=output,
            dataset="aids",
            dataset_dir=tmp_path / "unused",
            source_csv=tmp_path / "unused.csv",
            expected_project_commit="base",
        )

    assert not (output / "_RUN_COMPLETE.json").exists()


def test_freeze_recovery_rejects_canonical_graph_serialization_drift(
    tmp_path, monkeypatch
) -> None:
    root = _source_root(tmp_path)
    source = _graph([1, 2], [(0, 1), (1, 0)])
    monkeypatch.setattr(
        freeze_recovery,
        "_load_sources",
        lambda **_kwargs: ({"parent": source}, "dataset-fingerprint"),
    )
    real_save = freeze_recovery.atomic_torch_save

    def save_with_graph_drift(payload, path) -> None:
        real_save(payload, path)
        corrupted = copy.deepcopy(freeze_recovery.torch_load_payload(path))
        graph = next(iter(corrupted["canonical_graph_records"].values()))
        drifted = np.asarray(graph.x).copy()
        drifted.flat[0] += 100
        graph.x = drifted
        real_save(corrupted, path)

    monkeypatch.setattr(
        freeze_recovery, "atomic_torch_save", save_with_graph_drift
    )
    output = tmp_path / "canonical-drift"

    with pytest.raises(
        RuntimeError, match="frozen-closure fields changed across serialization"
    ):
        recover_completed_generation_freeze(
            source_generation_dir=root,
            output_dir=output,
            dataset="aids",
            dataset_dir=tmp_path / "unused",
            source_csv=tmp_path / "unused.csv",
            expected_project_commit="base",
        )

    assert not (output / "_RUN_COMPLETE.json").exists()
