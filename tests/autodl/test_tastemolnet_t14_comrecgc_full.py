from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines import tastemolnet_comrecgc_full as t14

from src.baselines.tastemolnet_comrecgc_full import (
    M_FALLBACK_MAX,
    M_MAX,
    CHECKPOINT_PROVENANCE_SCHEMA,
    GENERATION_PASS_MARKER,
    RUNTIME_STATE_SCHEMA,
    TRANSITION_EXPANDED_CAPACITY,
    TasteComRecGCFullBridge,
    TasteComRecGCFullError,
    build_full_train_correct_source_cohort,
    _bounded_t14_runtime,
    fallback_checkpoint_targets,
    resource_cap_decision,
    validate_t14_full_output,
)
from src.baselines.comrecgc.generation_checkpoint import (
    save_generation_checkpoint,
    scientific_command_sha256,
)
from src.baselines.comrecgc import generation_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Row:
    molecule_id: str
    label: int = 1


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_full_cohort_is_all_and_only_train_correct_sweet_with_stable_order() -> None:
    graph_a = "a" * 64
    graph_b = "b" * 64
    rows = [_Row("z"), _Row("a"), _Row("m"), _Row("b")]
    selected, manifest, payload = build_full_train_correct_source_cohort(
        true_sweet_rows=rows,
        predictions=[1, 0, 1, 1],
        source_probabilities=[0.9, 0.2, 0.8, 0.7],
        canonical_graph_hashes=[graph_a, graph_b, graph_a, graph_b],
        train_csv_sha256="c" * 64,
        checkpoint_id="d" * 64,
    )

    assert [row["parent_id"] for row in selected] == ["b", "m", "z"]
    assert [row["canonical_graph_hash"] for row in selected].count(graph_a) == 2
    assert manifest["selection"] == "true_label == 1 and frozen_T3_GINE_prediction == 1"
    assert manifest["cohort_count"] == 3
    assert manifest["cohort_jsonl_sha256"] == _sha(payload)
    assert manifest["validation_loaded"] is False
    assert manifest["calibration_loaded"] is False
    assert manifest["test_loaded"] is False


def test_full_cohort_rejects_non_sweet_input_even_if_prediction_is_sweet() -> None:
    with pytest.raises(TasteComRecGCFullError, match="not true Sweet"):
        build_full_train_correct_source_cohort(
            true_sweet_rows=[_Row("x", label=0)],
            predictions=[1],
            source_probabilities=[0.9],
            canonical_graph_hashes=["a" * 64],
            train_csv_sha256="b" * 64,
            checkpoint_id="c" * 64,
        )


def _reconciliation_fixture() -> tuple[
    list[dict[str, object]], dict[str, object], bytes, bytes
]:
    rows, manifest, cohort_bytes = build_full_train_correct_source_cohort(
        true_sweet_rows=[_Row("a"), _Row("b")],
        predictions=[1, 1],
        source_probabilities=[0.75, 0.625],
        canonical_graph_hashes=["a" * 64, "b" * 64],
        train_csv_sha256="c" * 64,
        checkpoint_id="d" * 64,
    )
    manifest_bytes = t14._canonical_bytes(manifest) + b"\n"
    return rows, manifest, cohort_bytes, manifest_bytes


def test_t14_reconciles_only_low_bit_probability_and_preserves_frozen_bytes(
    tmp_path: Path,
) -> None:
    frozen_rows, frozen_manifest, frozen_bytes, manifest_bytes = (
        _reconciliation_fixture()
    )
    replayed = deepcopy(frozen_rows)
    replayed[0]["source_probability"] = 0.75000002
    replayed[1]["source_probability"] = 0.62499998
    replayed_manifest = {
        **frozen_manifest,
        "cohort_jsonl_sha256": _sha(t14._cohort_lines(replayed)),
    }

    adopted, adopted_manifest, primary, observation = (
        t14.reconcile_t14_resume_cohort(
            frozen_cohort_bytes=frozen_bytes,
            frozen_manifest_bytes=manifest_bytes,
            replayed_rows=replayed,
            replayed_manifest=replayed_manifest,
        )
    )
    assert adopted == frozen_rows
    assert adopted_manifest == frozen_manifest
    assert t14._cohort_lines(adopted) == frozen_bytes
    assert observation["source_probability_mismatch_count"] == 2
    assert observation["source_probability_max_abs_delta"] == pytest.approx(2e-8)
    assert primary["frozen_cohort_sha256"] == _sha(frozen_bytes)
    assert observation["current_replayed_cohort_sha256"] == _sha(
        t14._cohort_lines(replayed)
    )
    assert observation["identity_and_discrete_fields_exact"] is True
    assert observation["frozen_cohort_rewritten"] is False

    frozen_path = tmp_path / "cohort.jsonl"
    manifest_path = tmp_path / "cohort_manifest.json"
    frozen_path.write_bytes(frozen_bytes)
    manifest_path.write_bytes(manifest_bytes)
    t14._persist_cohort_reconciliation_evidence(
        tmp_path,
        primary_receipt=primary,
        observation=observation,
    )
    primary_path = tmp_path / t14.COHORT_RECONCILIATION_RECEIPT
    first_primary = primary_path.read_bytes()
    first_observation_path = (
        tmp_path
        / t14.COHORT_RECONCILIATION_OBSERVATIONS
        / f'{observation["current_replayed_cohort_sha256"]}.json'
    )
    first_observation = first_observation_path.read_bytes()
    t14._persist_cohort_reconciliation_evidence(
        tmp_path,
        primary_receipt=primary,
        observation=observation,
    )
    assert primary_path.read_bytes() == first_primary
    assert first_observation_path.read_bytes() == first_observation

    replayed_again = deepcopy(frozen_rows)
    replayed_again[0]["source_probability"] = 0.75000003
    replayed_again_manifest = {
        **frozen_manifest,
        "cohort_jsonl_sha256": _sha(t14._cohort_lines(replayed_again)),
    }
    _, _, second_primary, second_observation = t14.reconcile_t14_resume_cohort(
        frozen_cohort_bytes=frozen_bytes,
        frozen_manifest_bytes=manifest_bytes,
        replayed_rows=replayed_again,
        replayed_manifest=replayed_again_manifest,
    )
    assert second_primary == primary
    assert (
        second_observation["current_replayed_cohort_sha256"]
        != observation["current_replayed_cohort_sha256"]
    )
    t14._persist_cohort_reconciliation_evidence(
        tmp_path,
        primary_receipt=second_primary,
        observation=second_observation,
    )
    binding = t14._validate_cohort_reconciliation_evidence(
        tmp_path,
        frozen_cohort_bytes=frozen_bytes,
        frozen_manifest_bytes=manifest_bytes,
    )
    assert binding is not None
    assert binding["observation_count"] == 2
    assert frozen_path.read_bytes() == frozen_bytes
    assert manifest_path.read_bytes() == manifest_bytes

    changed_observation = {
        **observation,
        "source_probability_mismatch_count": 1,
    }
    with pytest.raises(
        TasteComRecGCFullError, match="observation conflicts"
    ):
        t14._persist_cohort_reconciliation_evidence(
            tmp_path,
            primary_receipt=primary,
            observation=changed_observation,
        )


def test_t14_reconciliation_uses_frozen_reference_allclose_boundary() -> None:
    rows, manifest, cohort_bytes, manifest_bytes = _reconciliation_fixture()
    frozen = float(rows[0]["source_probability"])
    limit = t14.GINE_CANONICAL_REUSE_ATOL + (
        t14.GINE_CANONICAL_REUSE_RTOL * abs(frozen)
    )
    inside = math.nextafter(frozen + limit, frozen)
    outside = math.nextafter(frozen + limit, math.inf)
    assert np.allclose(
        np.asarray([inside]),
        np.asarray([frozen]),
        rtol=t14.GINE_CANONICAL_REUSE_RTOL,
        atol=t14.GINE_CANONICAL_REUSE_ATOL,
    )
    assert not np.allclose(
        np.asarray([outside]),
        np.asarray([frozen]),
        rtol=t14.GINE_CANONICAL_REUSE_RTOL,
        atol=t14.GINE_CANONICAL_REUSE_ATOL,
    )

    accepted = deepcopy(rows)
    accepted[0]["source_probability"] = inside
    accepted_manifest = {
        **manifest,
        "cohort_jsonl_sha256": _sha(t14._cohort_lines(accepted)),
    }
    t14.reconcile_t14_resume_cohort(
        frozen_cohort_bytes=cohort_bytes,
        frozen_manifest_bytes=manifest_bytes,
        replayed_rows=accepted,
        replayed_manifest=accepted_manifest,
    )

    rejected = deepcopy(rows)
    rejected[0]["source_probability"] = outside
    rejected_manifest = {
        **manifest,
        "cohort_jsonl_sha256": _sha(t14._cohort_lines(rejected)),
    }
    with pytest.raises(TasteComRecGCFullError, match="beyond low-bit replay"):
        t14.reconcile_t14_resume_cohort(
            frozen_cohort_bytes=cohort_bytes,
            frozen_manifest_bytes=manifest_bytes,
            replayed_rows=rejected,
            replayed_manifest=rejected_manifest,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("parent_id", "changed"),
        ("canonical_graph_hash", "f" * 64),
        ("true_label", 0),
        ("predicted_label", 2),
        ("split", "validation"),
    ),
)
def test_t14_reconciliation_rejects_identity_or_discrete_drift(
    field: str, value: object
) -> None:
    rows, manifest, cohort_bytes, manifest_bytes = _reconciliation_fixture()
    replayed = deepcopy(rows)
    replayed[0][field] = value
    replayed_manifest = {
        **manifest,
        "cohort_jsonl_sha256": _sha(t14._cohort_lines(replayed)),
    }
    with pytest.raises(TasteComRecGCFullError, match="cohort changed on resume"):
        t14.reconcile_t14_resume_cohort(
            frozen_cohort_bytes=cohort_bytes,
            frozen_manifest_bytes=manifest_bytes,
            replayed_rows=replayed,
            replayed_manifest=replayed_manifest,
        )


def test_t14_reconciliation_rejects_order_manifest_and_non_low_bit_drift() -> None:
    rows, manifest, cohort_bytes, manifest_bytes = _reconciliation_fixture()
    cases = []
    reordered = list(reversed(deepcopy(rows)))
    cases.append(
        (
            reordered,
            {**manifest, "cohort_jsonl_sha256": _sha(t14._cohort_lines(reordered))},
        )
    )
    large_delta = deepcopy(rows)
    large_delta[0]["source_probability"] = 0.5
    cases.append(
        (
            large_delta,
            {
                **manifest,
                "cohort_jsonl_sha256": _sha(t14._cohort_lines(large_delta)),
            },
        )
    )
    changed_manifest = {**manifest, "selection": "changed"}
    cases.append((deepcopy(rows), changed_manifest))
    for replayed, replayed_manifest in cases:
        with pytest.raises(TasteComRecGCFullError):
            t14.reconcile_t14_resume_cohort(
                frozen_cohort_bytes=cohort_bytes,
                frozen_manifest_bytes=manifest_bytes,
                replayed_rows=replayed,
                replayed_manifest=replayed_manifest,
            )

    nonfinite = deepcopy(rows)
    nonfinite[0]["source_probability"] = float("nan")
    nonfinite_manifest = {**manifest, "cohort_jsonl_sha256": "0" * 64}
    with pytest.raises(TasteComRecGCFullError):
        t14.reconcile_t14_resume_cohort(
            frozen_cohort_bytes=cohort_bytes,
            frozen_manifest_bytes=manifest_bytes,
            replayed_rows=nonfinite,
            replayed_manifest=nonfinite_manifest,
        )


def test_t14_old_checkpoint_cohort_replay_stays_uncached_until_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_modes: list[bool] = []

    class _Clone:
        def clone(self):
            return self

    class _Adapter:
        def __init__(self, *_args: object, canonical_replay_cache: bool = False, **_kwargs: object):
            self.canonical_replay_cache_enabled = canonical_replay_cache
            self._canonical_replay_cache = {}

        def score(self, graphs: object) -> SimpleNamespace:
            cache_modes.append(self.canonical_replay_cache_enabled)
            count = len(graphs)  # type: ignore[arg-type]
            return SimpleNamespace(
                valid_fullgraphs=(True,) * count,
                predictions=(1,) * count,
            )

        def enable_canonical_replay_cache(self) -> None:
            assert self.canonical_replay_cache_enabled is False
            assert self._canonical_replay_cache == {}
            self.canonical_replay_cache_enabled = True

    monkeypatch.setattr(
        t14,
        "encode_taste_source_graph",
        lambda row, _schema: {"source_label": row.label},
    )
    monkeypatch.setattr(
        t14,
        "taste_record_to_pyg",
        lambda _record, origin_index: SimpleNamespace(
            gcf_node_origin=_Clone(), num_nodes=1, origin_index=origin_index
        ),
    )
    monkeypatch.setattr(t14, "TasteFrozenGINENativeAdapter", _Adapter)
    monkeypatch.setattr(
        t14,
        "canonical_attributed_graph",
        lambda graph, **_kwargs: SimpleNamespace(
            graph_identity_sha256=hashlib.sha256(
                str(graph.origin_index).encode("ascii")
            ).hexdigest()
        ),
    )
    schema = SimpleNamespace(feature_atomic_numbers=(6,))
    graphs, _records, adapter, evidence = t14._initialize_full_source_graphs(
        checkpoint_payloads={},
        source_rows=[_Row("old-a"), _Row("old-b")],
        graph_schema=schema,
        device="cpu",
    )
    assert len(graphs) == 2
    assert cache_modes == [False]
    assert adapter.canonical_replay_cache_enabled is False
    assert evidence["source_count"] == 2

    adapter.enable_canonical_replay_cache()
    assert adapter.canonical_replay_cache_enabled is True

    # Keep this ordering explicit: the existing 5k cohort is reproduced with
    # old uncached semantics, then bridge restore primes generation records.
    source = inspect.getsource(t14.run_t14_full)
    initialize_at = source.index("_initialize_full_source_graphs(")
    enable_at = source.index("adapter.enable_canonical_replay_cache()")
    restore_at = source.index("_restore_checkpoint_state(")
    assert initialize_at < enable_at < restore_at


def test_resource_cap_uses_20k_then_one_25k_fallback() -> None:
    assert resource_cap_decision(
        completed_step=M_MAX, valid_unique_rule_count=10
    )["stop_reason"] == "RESOURCE_CAP_20K_VALID_UNIQUE_PASS"
    assert resource_cap_decision(
        completed_step=M_MAX, valid_unique_rule_count=9
    )["state"] == "EXTEND_ONCE_TO_25K"
    assert resource_cap_decision(
        completed_step=M_FALLBACK_MAX, valid_unique_rule_count=10
    )["stop_reason"] == "FALLBACK_CAP_25K_VALID_UNIQUE_PASS"
    assert resource_cap_decision(
        completed_step=M_FALLBACK_MAX, valid_unique_rule_count=9
    )["state"] == "SCIENTIFIC_FAILED_INSUFFICIENT_VALID_RULES"
    with pytest.raises(TasteComRecGCFullError, match="off cadence"):
        resource_cap_decision(completed_step=22_500, valid_unique_rule_count=99)
    assert fallback_checkpoint_targets(20_000) == (22_500, 25_000)
    assert fallback_checkpoint_targets(22_500) == (25_000,)
    with pytest.raises(TasteComRecGCFullError, match="cursor"):
        fallback_checkpoint_targets(17_500)


@pytest.mark.parametrize("route_c_updater", [None, object()], ids=["reference", "lowmemory"])
def test_route_c_step50_checkpoint_does_not_masquerade_as_parameter_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route_c_updater: object | None,
) -> None:
    captured: dict[str, object] = {}

    def _save_generation_checkpoint(
        checkpoint_root: Path, **kwargs: object
    ) -> SimpleNamespace:
        captured.update(kwargs)
        checkpoint_dir = checkpoint_root / "step-000000000050"
        checkpoint_dir.mkdir(parents=True)
        return SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            checkpoint_digest="a" * 64,
        )

    monkeypatch.setattr(
        generation_checkpoint,
        "save_generation_checkpoint",
        _save_generation_checkpoint,
    )
    monkeypatch.setattr(
        t14,
        "_checkpoint_algorithm_state",
        lambda **_kwargs: {"schema_version": t14.ROUTE_C_RUNTIME_STATE_SCHEMA},
    )
    connection = object()
    handles = SimpleNamespace(
        route_c_updater=route_c_updater,
        live_graph_state=SimpleNamespace(
            store=SimpleNamespace(checkpoint_connection=connection)
        ),
    )
    parameters = t14.TasteComRecGCFullParameters(
        source_pool=1,
        source_count=1,
    )

    evidence = t14._write_checkpoint(
        module=object(),
        bridge=object(),
        loop_state=SimpleNamespace(completed_step=50),
        parameters=parameters,
        checkpoint_root=tmp_path / "checkpoints",
        handles=handles,
        provenance={"identity": "frozen"},
        scientific_argv=("tastemolnet_t14_comrecgc_full_v1",),
        command_sha256="b" * 64,
        route_c=True,
    )

    assert parameters.checkpoint_step == t14.CHECK_INTERVAL
    assert captured["completed_step"] == 50
    assert captured["sqlite_source"] is connection
    assert evidence["checkpoint_step"] == 50
    assert evidence["next_step"] == 51


def test_legacy_step50_checkpoint_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        t14,
        "_checkpoint_algorithm_state",
        lambda **_kwargs: {"schema_version": t14.RUNTIME_STATE_SCHEMA},
    )
    handles = SimpleNamespace(route_c_updater=None)
    parameters = t14.TasteComRecGCFullParameters(
        source_pool=1,
        source_count=1,
    )

    with pytest.raises(TasteComRecGCFullError, match="off cadence: 50"):
        t14._write_checkpoint(
            module=object(),
            bridge=object(),
            loop_state=SimpleNamespace(completed_step=50),
            parameters=parameters,
            checkpoint_root=tmp_path / "checkpoints",
            handles=handles,
            provenance={"identity": "frozen"},
            scientific_argv=("tastemolnet_t14_comrecgc_full_v1",),
            command_sha256="b" * 64,
            route_c=False,
        )


def test_full_bridge_rejects_candidate_lineage_outside_frozen_train_cohort() -> None:
    bridge = TasteComRecGCFullBridge(
        cohort_count=3,
        adapter=object(),
        feature_atomic_numbers=(1, 6, 7),
    )
    graph = SimpleNamespace(comrecgc_source_index=3)
    with pytest.raises(TasteComRecGCFullError, match="escapes the train cohort"):
        bridge.call([graph], {})


def test_t14_installs_existing_bounded_full_runtime_without_parameter_change(
    tmp_path: Path,
) -> None:
    class _Bridge:
        @contextmanager
        def installed(self, module: object, *, neighbor_wrapper: object):
            original = module.neighbor_graph_access
            module.neighbor_graph_access = neighbor_wrapper(original)
            try:
                yield
            finally:
                module.neighbor_graph_access = original

    def _move(*_args: object, **_kwargs: object) -> tuple[None, bool, None, None, None]:
        return None, False, None, None, None

    module = SimpleNamespace(
        graph_map={},
        graph_index_map={},
        counterfactual_candidates=[],
        covering_graphs=set(),
        transitions={},
        move_to_next_graph=_move,
        neighbor_graph_access=lambda graph, _action: graph,
    )
    with _bounded_t14_runtime(
        module=module,
        bridge=_Bridge(),
        graph_store_path=tmp_path / "graph-state.sqlite3",
        seed=7,
        expanded_capacity=TRANSITION_EXPANDED_CAPACITY,
    ) as handles:
        assert type(handles.transition_map).__name__ == "CompactMoveScopedTransitionMap"
        assert type(handles.live_graph_state).__name__ == "LiveGraphState"
        assert handles.transition_map.audit()["scientific_parameters_changed"] is False
        assert handles.transition_map.audit()["expanded_capacity"] == 5
        module.move_to_next_graph(graphs_hash=[], start_graphs_hash=[])
        assert handles.transition_map.move_count == 1
        assert handles.live_graph_state.move_count == 1
    assert module.transitions == {}


def test_independent_terminal_verifier_reopens_bounded_train_only_closure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "t14"
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    cohort_rows = [
        {
            "canonical_graph_hash": "f" * 64,
            "parent_id": "x",
            "predicted_label": 1,
            "source_probability": 0.75,
            "split": "train",
            "true_label": 1,
        }
    ]
    cohort = t14._cohort_lines(cohort_rows)
    (root / "cohort.jsonl").write_bytes(cohort)
    cohort_manifest = {
        "status": "PASS",
        "policy": "FULL_TRAIN_CORRECT_SOURCE",
        "cohort_jsonl_sha256": _sha(cohort),
    }
    cohort_manifest_bytes = t14._canonical_bytes(cohort_manifest) + b"\n"
    (root / "cohort_manifest.json").write_bytes(cohort_manifest_bytes)
    valid = {"valid_unique_rule_count": 10}
    resource = {"state": "STOP_AND_POSTPROCESS", "m_effective": M_MAX}
    (root / "valid_unique.json").write_text(json.dumps(valid))
    (root / "resource_cap_receipt.json").write_text(json.dumps(resource))
    provenance = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA,
        "dataset": "tastemolnet",
        "method": "comrecgc",
        "stage": "T14_COMRECGC_FULL",
        "train_csv_sha256": "a" * 64,
        "checkpoint_id": "b" * 64,
        "cohort_jsonl_sha256": _sha(cohort),
        "parameters_sha256": "c" * 64,
        "official_authority_sha256": "d" * 64,
        "execution_commit": "e" * 40,
        "runtime_state_schema": RUNTIME_STATE_SCHEMA,
        "transition_cache_policy": "compact_transition_action_replay_lru_v1",
        "graph_state_policy": "authoritative_backing_live_graph_resolution_v2",
        "scientific_command_sha256": "",
        "total_steps": str(M_FALLBACK_MAX),
    }
    argv = ("tastemolnet_t14_comrecgc_full_v1", "fixture=true")
    command_sha = scientific_command_sha256(argv)
    provenance["scientific_command_sha256"] = command_sha
    database = tmp_path / "source.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE graphs (id INTEGER PRIMARY KEY)")
    connection.commit()
    validation = save_generation_checkpoint(
        checkpoint_root,
        completed_step=M_MAX,
        step_complete=True,
        algorithm_state={"schema_version": RUNTIME_STATE_SCHEMA},
        trace_state={"enabled": False},
        sqlite_source=connection,
        provenance_fingerprints=provenance,
        scientific_argv=argv,
        command_sha256=command_sha,
        total_steps=M_FALLBACK_MAX,
    )
    connection.close()
    (checkpoint_root / f"checkpoint-{M_MAX:06d}.json").write_text(
        json.dumps(
            {
                "schema_version": "tastemolnet_t14_checkpoint_v2",
                "checkpoint_dir": str(validation.checkpoint_dir),
                "checkpoint_digest": validation.checkpoint_digest,
                "checkpoint_step": M_MAX,
                "next_step": M_MAX + 1,
                "checkpoint_persisted_in_output": True,
                "bounded_transition_state": True,
                "authoritative_graph_store_snapshot": True,
                "written_at": "fixture",
            }
        )
    )
    checkpoint_identity = {
        "schema_version": CHECKPOINT_PROVENANCE_SCHEMA,
        "status": "FROZEN",
        "provenance": provenance,
        "scientific_argv": list(argv),
        "command_sha256": command_sha,
        "total_steps": M_FALLBACK_MAX,
        "checkpoint_interval": 2500,
        "transition_expanded_capacity": TRANSITION_EXPANDED_CAPACITY,
        "raw_neighbor_graphs_retained_unbounded": False,
    }
    (root / "checkpoint_identity.json").write_text(json.dumps(checkpoint_identity))
    (root / "progress.json").write_text(
        json.dumps({"status": "PASS", "completed_step": M_MAX})
    )
    manifest = {
        "schema_version": "tastemolnet_t14_comrecgc_full_v1",
        "status": "PASS",
        "stage": "T14_COMRECGC_FULL",
        "train_loaded": True,
        "validation_loaded": False,
        "calibration_loaded": False,
        "test_loaded": False,
        "rf_oracle_used": False,
        "calibration_status": "NOT_EVALUATED",
        "held_out_test_status": "NOT_EVALUATED",
        "export_status": "NOT_EVALUATED",
        "paper_result_eligible": False,
        "method_cell_pass": False,
        "cohort_manifest_sha256": _sha((root / "cohort_manifest.json").read_bytes()),
        "cohort_jsonl_sha256": _sha(cohort),
        "resource_cap": resource,
        "valid_unique": valid,
        "bounded_runtime": {
            "transition_cache": {
                "patch": "compact_transition_action_replay_lru_v1",
                "scientific_parameters_changed": False,
            },
            "live_graph_state": {"unresolved_lookups": 0},
            "checkpoint_schema": RUNTIME_STATE_SCHEMA,
            "checkpoint_identity_sha256": _sha(
                (root / "checkpoint_identity.json").read_bytes()
            ),
            "raw_neighbor_graphs_retained_unbounded": False,
        },
    }
    (root / "generation_manifest.json").write_text(json.dumps(manifest))
    (root / "GENERATION_PASS").write_text(f"{GENERATION_PASS_MARKER}\n")

    receipt = validate_t14_full_output(root)
    assert receipt["status"] == "PASS"
    assert receipt["m_effective"] == M_MAX
    assert receipt["test_loaded"] is False
    assert receipt["method_cell_pass"] is False

    replayed_rows = deepcopy(cohort_rows)
    replayed_rows[0]["source_probability"] = 0.75000002
    replayed_manifest = {
        **cohort_manifest,
        "cohort_jsonl_sha256": _sha(t14._cohort_lines(replayed_rows)),
    }
    _, _, primary, observation = t14.reconcile_t14_resume_cohort(
        frozen_cohort_bytes=cohort,
        frozen_manifest_bytes=cohort_manifest_bytes,
        replayed_rows=replayed_rows,
        replayed_manifest=replayed_manifest,
    )
    t14._persist_cohort_reconciliation_evidence(
        root,
        primary_receipt=primary,
        observation=observation,
    )
    binding = t14._validate_cohort_reconciliation_evidence(
        root,
        frozen_cohort_bytes=cohort,
        frozen_manifest_bytes=cohort_manifest_bytes,
    )
    manifest["cohort_reconciliation"] = binding
    (root / "generation_manifest.json").write_text(json.dumps(manifest))
    assert validate_t14_full_output(root)["status"] == "PASS"

    observation_path = (
        root
        / t14.COHORT_RECONCILIATION_OBSERVATIONS
        / f'{observation["current_replayed_cohort_sha256"]}.json'
    )
    observation_bytes = observation_path.read_bytes()
    observation_path.unlink()
    with pytest.raises(TasteComRecGCFullError, match="has no observations"):
        validate_t14_full_output(root)
    observation_path.write_bytes(observation_bytes)
    tampered = {**observation, "status": "FAIL"}
    observation_path.write_bytes(t14._canonical_bytes(tampered) + b"\n")
    with pytest.raises(TasteComRecGCFullError, match="observation changed"):
        validate_t14_full_output(root)
    observation_path.write_bytes(observation_bytes)
    primary_path = root / t14.COHORT_RECONCILIATION_RECEIPT
    primary_bytes = primary_path.read_bytes()
    primary_path.write_bytes(
        t14._canonical_bytes({**primary, "status": "FAIL"}) + b"\n"
    )
    with pytest.raises(TasteComRecGCFullError, match="receipt changed"):
        validate_t14_full_output(root)
    primary_path.write_bytes(primary_bytes)

    manifest["test_loaded"] = True
    (root / "generation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(TasteComRecGCFullError, match="closure changed"):
        validate_t14_full_output(root)


def test_t14_launchers_keep_explicit_gpu_budget_and_slurm_contract() -> None:
    autodl = (PROJECT_ROOT / "scripts/autodl/run_tastemolnet_t14_comrecgc_full.sh").read_text()
    slurm = (PROJECT_ROOT / "scripts/slurm/run_tastemolnet_comrecgc_full.sh").read_text()
    for token in (
        'TASTEMOLNET_T14_GPU_INDEX="${TASTEMOLNET_T14_GPU_INDEX:-1}"',
        '--gpu-index "$TASTEMOLNET_T14_GPU_INDEX"',
        '--physical-gpu-index "$TASTEMOLNET_T14_GPU_INDEX"',
        "TASTEMOLNET_T14_OUTPUT",
        "TASTEMOLNET_T14_GPU_INDEX",
        "RUN_GNN_ABLATION",
        "inference.fallback_to_heuristic=false",
        "TASTEMOLNET_T14_RESUME",
        "--resume",
    ):
        assert token in autodl
    for token in (
        "#SBATCH --partition=A800",
        "#SBATCH --gres=gpu:a800:1",
        "#SBATCH --output=logs/%j.out",
        "#SBATCH --error=logs/%j.err",
        "source ~/.bashrc",
        "conda activate smiles_pip118",
        "cd /share/home/u20526/czx/counterfactual-subgraph",
        "export PYTHONPATH=$PWD",
        "--config configs/hpc.yaml",
        "--set inference.fallback_to_heuristic=false",
    ):
        assert token in slurm
