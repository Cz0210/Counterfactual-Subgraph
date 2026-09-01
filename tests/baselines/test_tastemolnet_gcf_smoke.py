from __future__ import annotations

import copy
import json
import importlib.util
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_tastemolnet_gcf_smoke as cli
from src.baselines import tastemolnet_gcf_smoke as taste_gcf
from src.baselines.tastemolnet_gcf_smoke import (
    DISABLED_RELEASE_STATE,
    LABEL_MAP,
    NativeScoreBatch,
    RELEASE_CONFIG_PATH,
    SMOKE_GPU_INDEX,
    TasteGCFGraphSchema,
    TasteGCFImportanceBridge,
    TasteFrozenGINENativeAdapter,
    TasteGCFSmokeError,
    TasteGCFSmokeReleaseDisabled,
    _installed_official_importance_args,
    build_worker_raw_evidence,
    encode_taste_source_graph,
    parse_candidate_trace,
    score_and_candidate,
    taste_record_to_pyg,
    verify_t7_worker_raw_evidence,
)
from src.utils import tastemolnet_t7_gcf_release as t7_release
from src.utils import tastemolnet_t7_managed_v2 as t7_managed
from src.utils.tastemolnet_t7_gcf_release import assert_execution_released


def test_t7_structured_pass_marker_matches_final_contract() -> None:
    assert taste_gcf.PASS_MARKER == "[TASTE_T7_GCF_SMOKE_PASS]"


ROOT = Path(__file__).resolve().parents[2]


def _trace_row(
    probabilities: list[float],
    *,
    rank: int = 0,
    candidate_condition: bool | None = None,
    score: float | None = None,
) -> dict[str, object]:
    expected_score, prediction, expected_candidate = score_and_candidate(
        probabilities
    )
    return {
        "schema_version": "tastemolnet_gcf_candidate_trace_v1",
        "rank": rank,
        "graph_identity_sha256": "a" * 64,
        "frequency": 2,
        "probabilities": probabilities,
        "pred_before": 1,
        "pred_candidate": prediction,
        "source_label": 1,
        "score": expected_score if score is None else score,
        "covered_parent_count": 4,
        "coverage_ratio": 0.5,
        "score_definition": "1.0 - probabilities[source_label]",
        "candidate_condition": (
            expected_candidate
            if candidate_condition is None
            else candidate_condition
        ),
        "candidate_condition_definition": "pred_candidate != source_label",
        "valid_fullgraph": True,
        "failure_reason": "",
        "native_action_kind": "full_counterfactual_graph",
    }


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


_CHECKPOINT_UUID = "123e4567-e89b-42d3-a456-426614174000"


def _checkpoint_path(runtime_root: Path) -> Path:
    return runtime_root / "checkpoints" / _CHECKPOINT_UUID


def _checkpoint_fixture(torch: object) -> dict[str, object]:
    return {
        "schema_version": "fixture",
        "checkpoint_uuid": _CHECKPOINT_UUID,
        "generation_token": "f" * 64,
        "tensor": torch.arange(4),
    }


def _dummy_inputs() -> SimpleNamespace:
    return SimpleNamespace(
        execution={
            "implementation_commit": "1" * 40,
            "implementation_tree": "2" * 40,
            "execution_commit": "3" * 40,
            "execution_tree": "4" * 40,
        },
        release={
            "external_authority_sha256": "5" * 64,
            "controller_receipt_sha256": "6" * 64,
            "gpu_lease_receipt_sha256": "7" * 64,
            "managed_execution_v2_pass_sha256": "8" * 64,
        },
        t2={
            "schema_version": "tastemolnet_t2_gine_downstream_binding_v1",
            "stage": "T2_GINE_FULL",
            "status": "PASS",
            "state": "T2_GINE_FULL_PASS_ADOPTED",
            "source_cid": "source",
            "source_run_id": "run",
            "adoption_root": "/private/t2",
            "adoption_root_inventory_sha256": "7" * 64,
            "gate_path": "/private/t2/gate.json",
            "gate_sha256": "9" * 64,
            "receipt_path": "/private/t2/manifest.json",
            "receipt_sha256": "8" * 64,
            "source_evidence_sha256": "6" * 64,
            "formal_bundle_root": "/private/checkpoint",
            "formal_bundle_inventory": [],
            "formal_bundle_inventory_sha256": "5" * 64,
            "formal_bundle_model_sha256": "c" * 64,
            "formal_bundle_sha256s_sha256": "4" * 64,
        },
        stage_evidence={
            "t3": {
                "gate_sha256": "a" * 64,
                "root_inventory_sha256": "b" * 64,
                "checkpoint_id": "c" * 64,
            },
            "t4": {
                "gate_sha256": "d" * 64,
                "root_inventory_sha256": "e" * 64,
            },
        },
        checkpoint_payloads={
            "feature_schema.json": b"feature",
            "temperature_scaling.json": b"temperature",
        },
        train_contract={"sha256": "f" * 64},
        neurosed_evidence={
            "pass_sha256": "9" * 64,
            "gate_sha256": "1" * 64,
            "verification_sha256": "2" * 64,
            "checkpoint_sha256": "0" * 64,
            "feature_schema_sha256": "3" * 64,
            "sha256s_sha256": "4" * 64,
        },
        controller={"run_id": "run-t7"},
        gpu={"gpu_index": SMOKE_GPU_INDEX, "gpu_uuid": "GPU-test"},
    )


def _dummy_managed_worker() -> SimpleNamespace:
    predecessor = {
        "kind": "TASTE_GCF_NEUROSED_PASS",
        "path": "/private/neurosed/PASS.json",
        "sha256": "9" * 64,
    }
    managed_input_hashes = {
        "managed_execution_v2_pass": "8" * 64,
        "taste_gcf_neurosed_pass": "9" * 64,
        "taste_gcf_neurosed_gate": "1" * 64,
        "taste_gcf_neurosed_verification": "2" * 64,
        "taste_gcf_neurosed_checkpoint": "0" * 64,
        "taste_gcf_neurosed_feature_schema": "3" * 64,
        "taste_gcf_neurosed_sha256s": "4" * 64,
        "taste_gine_t2_gate": "9" * 64,
        "taste_gine_t3_gate": "a" * 64,
        "taste_oracle_t4_gate": "d" * 64,
        "taste_train_csv": "f" * 64,
    }
    return SimpleNamespace(
        attempt_id="123e4567-e89b-42d3-a456-426614174000",
        generation_token="123e4567-e89b-42d3-a456-426614174001",
        expected_final_path=Path("/private/final/t7"),
        predecessor_evidence=lambda: [predecessor],
        attempt_input_hashes=lambda: dict(managed_input_hashes),
    )


def _science_summary() -> dict[str, object]:
    checkpoint_identity = {
        "st_dev": 1,
        "st_ino": 2,
        "st_mode": stat.S_IFREG | 0o600,
        "st_nlink": 1,
        "st_uid": 3,
        "st_gid": 4,
        "st_size": 4096,
        "st_mtime_ns": 5,
        "st_ctime_ns": 6,
    }
    progress_checkpoint: dict[str, object] = {
        "schema_version": "tastemolnet_t7_gcf_vrrw_progress_resume_v1",
        "checkpoint_written": True,
        "checkpoint_durable": True,
        "planned_interruption_observed": True,
        "checkpoint_reloaded": True,
        "resumed": True,
        "resume_entry_used_saved_graph": True,
        "deterministic_restart_from_seed_used": False,
        "checkpoint_held_through_resume_evidence": True,
        "checkpoint_path_cleanup_delegated_to_temporary_runtime": True,
        "checkpoint_unlinked_by_t7_security_boundary": False,
        "checkpoint_payload_persisted_to_terminal_output": False,
        "checkpoint_uuid": "123e4567-e89b-42d3-a456-426614174000",
        "generation_token": "9" * 64,
        "interruption_after_step": 8,
        "resume_start_step": 9,
        "pre_resume_step_count": 8,
        "post_resume_step_count": 8,
        "total_step_count": 16,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_size_bytes": 4096,
        "checkpoint_physical_identity": checkpoint_identity,
        "checkpoint_physical_identity_sha256": taste_gcf._sha256_bytes(
            taste_gcf._canonical_bytes(checkpoint_identity)
        ),
        "checkpoint_binding_sha256": "",
        "saved_progress_state_sha256": "b" * 64,
        "reset_progress_state_sha256": "c" * 64,
        "restored_progress_state_sha256": "b" * 64,
        "saved_rng_state_sha256": "d" * 64,
        "reset_rng_state_sha256": "e" * 64,
        "restored_rng_state_sha256": "d" * 64,
        "checkpoint_trace_prefix_sha256": "f" * 64,
        "final_trace_prefix_sha256": "f" * 64,
        "post_resume_trace_sha256": "0" * 64,
        "full_trace_sha256": "1" * 64,
        "resume_graph_identity_sha256": "2" * 64,
        "first_post_resume_graph_identity_sha256": "2" * 64,
        "trace_continuity_proven": True,
        "trace_continuity_sha256": "",
    }
    progress_checkpoint["checkpoint_binding_sha256"] = (
        taste_gcf._checkpoint_binding_sha256(progress_checkpoint)
    )
    progress_checkpoint["trace_continuity_sha256"] = (
        taste_gcf._trace_continuity_sha256(progress_checkpoint)
    )
    return {
        "schema_version": "tastemolnet_gcf_native_vrrw_smoke_v2",
        "stage": "T7_GCF_SMOKE",
        "dataset": "tastemolnet",
        "method": "GCFExplainer",
        "parent_evidence": {
            "source_pool_count": 64,
            "source_pool_gine_correct_sweet": 8,
            "selected_parent_count": 8,
            "selected_parent_graph_hashes_sha256": "1" * 64,
            "pred_before": 1,
        },
        "official_random_walk_steps": 16,
        "progress_checkpoint": progress_checkpoint,
        "official_candidate_count": 1,
        "strict_counterfactual_candidate_count": 1,
        "destination_prediction_counts": {"0": 1, "2": 0},
        "native_action_invocation_counts": {"NLC": 1},
        "importance_bridge_calls": 1,
        "importance_bridge_evaluated_graphs": 1,
        "neurosed_distance_calls": 1,
        "neurosed_distance_evaluated_graphs": 1,
        "adapter": {
            "schema_version": "tastemolnet_gcf_native_gine_adapter_v1",
            "checkpoint_id": "c" * 64,
            "num_classes": 3,
            "source_label": 1,
            "call_count": 2,
            "decode_success_count": 9,
            "empty_valid_batch_count": 0,
            "decode_failures": {},
            "batch_scorer": {
                "schema_version": 1,
                "calls": 2,
                "cache_capacity": 0,
                "cache_entries": 0,
                "cache_hits": 0,
                "cache_misses": 9,
                "scored_rows": 9,
                "cache_scope": "exact_complete_ordered_batch_v1",
                "partial_row_reuse": False,
                "deduplication": False,
                "chunking": False,
            },
            "rf_oracle_used": False,
        },
        "alpha": 1.0,
        "coverage_mode": "official_taste_neurosed_threshold_coverage",
        "neurosed_distance_threshold": 0.25,
        "neurosed_predecessor": {
            "schema_version": "tastemolnet_gcf_neurosed_pass_v1",
            "status": "PASS",
            "marker": "TASTE_GCF_NEUROSED_PASS",
            "pass_path": "/private/neurosed/PASS.json",
            "pass_sha256": "9" * 64,
            "checkpoint_sha256": "0" * 64,
            "feature_schema_sha256": "1" * 64,
            "neurosed_train_graph_ids_hash": "2" * 64,
            "neurosed_validation_graph_ids_hash": "3" * 64,
            "calibration_loaded": False,
            "test_loaded": False,
            "role": "GCF_AUXILIARY_DISTANCE_MODEL",
            "classifier": False,
            "source_label_independent": True,
            "train_only_fit": True,
            "validation_only_selection": True,
            "health_gate_status": "PASS",
        },
        "candidate_condition": "pred_candidate != source_label",
        "score_definition": "1.0 - p_source",
        "native_full_graph_semantics": True,
        "deletion_only_semantics": False,
        "neurosed_status": "PASS_INPUT_REVALIDATED",
        "distance_status": "EVALUATED",
        "selector_status": "NOT_EVALUATED",
        "full_route_status": "NOT_EVALUATED",
        "bace_artifacts_used": False,
        "rf_oracle_used": False,
        "train_loaded": True,
        "validation_payload_loaded": False,
        "calibration_payload_loaded": False,
        "test_payload_loaded": False,
        "native_graph_payload_persisted": False,
        "molecule_payload_persisted": False,
        "paper_result_eligible": False,
    }


def test_multiclass_candidate_is_not_a_binary_half_threshold() -> None:
    score, prediction, candidate = score_and_candidate([0.3, 0.4, 0.3])
    assert score == pytest.approx(0.6)
    assert prediction == 1
    assert candidate is False

    score, prediction, candidate = score_and_candidate([0.6, 0.1, 0.3])
    assert score == pytest.approx(0.9)
    assert prediction == 0
    assert candidate is True


@pytest.mark.parametrize(
    ("label", "smiles", "destinations"),
    (
        (0, "C", [1, 2]),
        (1, "N", [0, 2]),
        (2, "O", [0, 1]),
    ),
)
def test_stable_taste_graph_exports_preserve_native_three_class_labels(
    label: int,
    smiles: str,
    destinations: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometric = types.ModuleType("torch_geometric")
    geometric.__path__ = []  # type: ignore[attr-defined]
    geometric_data = types.ModuleType("torch_geometric.data")

    class Data:
        def __init__(self, **values: object) -> None:
            vars(self).update(values)

    geometric.data = geometric_data  # type: ignore[attr-defined]
    geometric_data.Data = Data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_geometric", geometric)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", geometric_data)
    schema = TasteGCFGraphSchema(
        atom_vocabulary=(1, 6, 7, 8),
        feature_atomic_numbers=(1, 6, 7, 8),
        formal_charge_vocabulary=(0,),
        aromaticity_vocabulary=(False,),
        bond_type_vocabulary=("SINGLE",),
        max_num_nodes=5,
    )
    row = SimpleNamespace(
        molecule_id=f"native-{label}",
        smiles=smiles,
        canonical_smiles=smiles,
        label=label,
        split="train",
        semantic_label=("Bitter", "Sweet", "Tasteless")[label],
        source_row_index=label,
        source_path="/private/train.csv",
    )
    record = encode_taste_source_graph(row, schema)
    schema_payload = schema.to_dict()
    assert schema_payload["native_num_classes"] == 3
    assert schema_payload["native_label_projection"] == "identity_0_1_2"
    schema_payload["native_label_map"]["0"] = "MUTATED"
    assert LABEL_MAP["0"] == "Bitter"
    assert schema.to_dict()["native_label_map"]["0"] == "Bitter"
    assert record["label"] == label
    assert record["gnn_label"] == label
    assert record["source_label"] == label
    assert record["destination_labels"] == destinations
    assert "target_label" not in record

    graph = taste_record_to_pyg(record, origin_index=label)
    assert int(graph.y.item()) == label
    assert graph.gcf_source_label.tolist() == [label]
    assert graph.gcf_destination_labels.tolist() == destinations
    assert graph.gcf_origin_index.tolist() == [label]


def test_taste_record_to_pyg_rejects_binary_or_targeted_label_projection() -> None:
    binary = {
        "label": 2,
        "gnn_label": 1,
        "source_label": 2,
        "destination_labels": [0, 1],
    }
    with pytest.raises(TasteGCFSmokeError, match="native untargeted"):
        taste_record_to_pyg(binary, origin_index=0)

    targeted = {**binary, "gnn_label": 2, "target_label": 0}
    with pytest.raises(TasteGCFSmokeError, match="native untargeted"):
        taste_record_to_pyg(targeted, origin_index=0)

    boolean_destination = {
        **binary,
        "label": 1,
        "gnn_label": 1,
        "source_label": 1,
        "destination_labels": [False, 2],
    }
    with pytest.raises(TasteGCFSmokeError, match="native untargeted"):
        taste_record_to_pyg(boolean_destination, origin_index=0)


@pytest.mark.parametrize("origin", (True, 0.9, "0"))
def test_native_adapter_rejects_non_integer_lineage_aliases(origin: object) -> None:
    adapter = object.__new__(TasteFrozenGINENativeAdapter)
    adapter.source_records = ({"source_label": 1},)
    graph = SimpleNamespace(gcf_origin_index=origin)
    with pytest.raises(TasteGCFSmokeError, match="source index"):
        adapter._portable(graph, 0)


def test_native_adapter_canonical_replay_uses_exact_model_input_bytes() -> None:
    adapter = object.__new__(TasteFrozenGINENativeAdapter)
    adapter._torch = torch
    adapter.parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    adapter.hidden_dim = 2
    adapter.canonical_replay_cache_enabled = True
    adapter._canonical_replay_cache = {}
    adapter.canonical_replay_cache_hits = 0
    adapter.canonical_replay_cache_misses = 0
    scorer_calls = []

    class _ReplayScorer:
        @staticmethod
        def score(graphs):
            scorer_calls.append(len(graphs))
            return SimpleNamespace(
                project_logits=torch.log(
                    torch.tensor([[0.2, 0.7, 0.1]], dtype=torch.float32)
                ),
                graph_hidden=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            )

    adapter.scorer = _ReplayScorer()
    payload = {
        "schema_version": "test_gine_model_graph_v1",
        "canonical_smiles": "CO",
        "node_features": [[1], [2]],
    }
    observed_probabilities, observed_embeddings = (
        adapter._score_canonical_model_inputs(
            portable=[object()],
            valid_positions=[0],
            model_graph_payloads=[payload],
        )
    )
    first_probabilities = observed_probabilities.clone()
    first_embeddings = observed_embeddings.clone()

    canonical_probabilities, canonical_embeddings = (
        adapter._score_canonical_model_inputs(
            portable=[object()],
            valid_positions=[0],
            model_graph_payloads=[payload],
        )
    )
    assert torch.equal(canonical_probabilities, first_probabilities)
    assert torch.equal(canonical_embeddings, first_embeddings)
    assert scorer_calls == [1]
    assert adapter.canonical_replay_cache_hits == 1
    assert adapter.canonical_replay_cache_misses == 1


def test_native_adapter_primes_canonical_replay_from_validated_checkpoint() -> None:
    adapter = object.__new__(TasteFrozenGINENativeAdapter)
    adapter._torch = torch
    adapter.parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    adapter.hidden_dim = 2
    adapter.canonical_replay_cache_enabled = True
    adapter._canonical_replay_cache = {}
    adapter.canonical_replay_cache_hits = 0
    adapter.canonical_replay_cache_misses = 0
    payload = {
        "schema_version": "test_gine_model_graph_v1",
        "canonical_smiles": "CO",
    }
    model_sha, _encoded = adapter._model_input_cache_key(payload)
    record = SimpleNamespace(
        valid_fullgraph=True,
        model_graph_payload=payload,
        model_graph_sha256=model_sha,
        probabilities=(0.2, 0.7, 0.1),
        embedding_values=(1.0, 2.0),
        embedding_dtype="<f4",
    )
    assert adapter.prime_canonical_replay_cache({"identity": record}) == 1
    adapter.scorer = SimpleNamespace(
        score=lambda _graphs: pytest.fail("checkpointed model input was rescored")
    )
    probabilities, embeddings = adapter._score_canonical_model_inputs(
        portable=[object()],
        valid_positions=[0],
        model_graph_payloads=[payload],
    )
    assert torch.equal(
        probabilities, torch.tensor([[0.2, 0.7, 0.1]], dtype=torch.float32)
    )
    assert torch.equal(
        embeddings, torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    )


@pytest.mark.parametrize("source_label", (0, 2))
def test_sweet_scoring_adapter_rejects_non_sweet_source_semantics(
    source_label: int,
) -> None:
    with pytest.raises(TasteGCFSmokeError, match="only native Sweet"):
        TasteFrozenGINENativeAdapter(
            {},
            source_records=(
                {
                    "label": source_label,
                    "gnn_label": source_label,
                    "source_label": source_label,
                    "destination_labels": [
                        label for label in range(3) if label != source_label
                    ],
                },
            ),
            graph_schema=object(),
            device="cpu",
        )


@pytest.mark.parametrize(
    "probabilities",
    ([0.2, 0.8], [0.2, float("nan"), 0.8], [0.2, -0.1, 0.9]),
)
def test_multiclass_score_rejects_malformed_probabilities(
    probabilities: list[float],
) -> None:
    with pytest.raises(TasteGCFSmokeError):
        score_and_candidate(probabilities)


def test_candidate_trace_enforces_score_and_argmax_separately() -> None:
    valid = _trace_row([0.6, 0.1, 0.3])
    assert parse_candidate_trace(_jsonl([valid])) == [valid]

    wrong_predicate = _trace_row(
        [0.3, 0.4, 0.3], candidate_condition=True
    )
    with pytest.raises(TasteGCFSmokeError, match="multiclass semantics"):
        parse_candidate_trace(_jsonl([wrong_predicate]))

    wrong_score = _trace_row([0.6, 0.1, 0.3], score=0.5)
    with pytest.raises(TasteGCFSmokeError, match="multiclass semantics"):
        parse_candidate_trace(_jsonl([wrong_score]))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("rank", False),
        ("pred_before", True),
        ("pred_candidate", False),
        ("source_label", True),
        ("score", "0.9"),
    ),
)
def test_candidate_trace_rejects_bool_and_numeric_string_aliases(
    field: str, value: object
) -> None:
    row = _trace_row([0.6, 0.1, 0.3])
    row[field] = value
    with pytest.raises(TasteGCFSmokeError):
        parse_candidate_trace(_jsonl([row]))


class _FakeVRRW:
    torch = torch

    def __init__(self) -> None:
        self.is_graph_counterfactual = lambda _value: False

    @staticmethod
    def calculate_hash(embedding: np.ndarray) -> bytes:
        return np.ascontiguousarray(embedding).tobytes()


class _FakeAdapter:
    @staticmethod
    def score(_graphs: object) -> NativeScoreBatch:
        probabilities = np.asarray(
            [[0.3, 0.4, 0.3], [0.6, 0.1, 0.3]], dtype=float
        )
        return NativeScoreBatch(
            probabilities=probabilities,
            predictions=(1, 0),
            scores=(0.6, 0.9),
            candidate_flags=(False, True),
            graph_embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            valid_fullgraphs=(True, True),
            failure_reasons=("", ""),
        )


class _FakeImportance:
    @staticmethod
    def neurosed_threshold_coverage_estimation(
        _model: object,
        _graphs: object,
        _counts: object,
        _threshold: float,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )


def test_importance_bridge_uses_official_neurosed_and_argmax_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = [SimpleNamespace(tag="a"), SimpleNamespace(tag="b")]

    def canonical(graph: SimpleNamespace, **_kwargs: object) -> SimpleNamespace:
        digest = ("a" if graph.tag == "a" else "b") * 64
        payload = {
            "canonical_graph": f"[{graph.tag}]",
            "num_nodes": 1,
            "num_edges": 0,
        }
        return SimpleNamespace(
            graph_identity_sha256=digest,
            collision_payload=lambda: payload,
        )

    monkeypatch.setattr(taste_gcf, "canonical_attributed_graph", canonical)
    vrrw = _FakeVRRW()
    bridge = TasteGCFImportanceBridge(
        adapter=_FakeAdapter(),
        vrrw=vrrw,
        importance=_FakeImportance(),
        neurosed_model=object(),
        original_graph_element_counts=torch.ones(8),
        distance_threshold=0.25,
        parent_count=8,
        feature_atomic_numbers=(6, 7),
    )
    importance = SimpleNamespace(call=lambda *_args: None)
    original_call = importance.call
    original_hash = vrrw.calculate_hash
    original_predicate = vrrw.is_graph_counterfactual
    with bridge.installed(importance):
        assert importance.call == bridge.call
        assert vrrw.calculate_hash == bridge.calculate_hash
        assert vrrw.is_graph_counterfactual == bridge.is_graph_counterfactual
        parts, embeddings, coverage = importance.call(graphs, {})
        hashes = [vrrw.calculate_hash(row) for row in embeddings]
        assert hashes == ["a" * 64, "b" * 64]
        assert parts.tolist() == [[0.6, 0.5], [0.9, 1.0]]
        assert embeddings.shape == (2, 2)
        assert coverage.shape == (2, 8)
        assert coverage[0].sum().item() == 4
        assert coverage[1].sum().item() == 8
        assert bridge.is_graph_counterfactual(hashes[0]) is False
        assert bridge.is_graph_counterfactual(hashes[1]) is True
    assert bridge.distance_call_count == 1
    assert bridge.calculate_hash_count == 2
    assert importance.call is original_call
    assert vrrw.calculate_hash is original_hash
    assert vrrw.is_graph_counterfactual is original_predicate


def test_importance_bridge_separates_same_embedding_different_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = [SimpleNamespace(tag="a"), SimpleNamespace(tag="b")]

    def canonical(graph: SimpleNamespace, **_kwargs: object) -> SimpleNamespace:
        digest = ("a" if graph.tag == "a" else "b") * 64
        payload = {
            "canonical_graph": f"[{graph.tag}]",
            "num_nodes": 1,
            "num_edges": 0,
        }
        return SimpleNamespace(
            graph_identity_sha256=digest,
            collision_payload=lambda: payload,
        )

    monkeypatch.setattr(taste_gcf, "canonical_attributed_graph", canonical)

    class CollisionAdapter:
        @staticmethod
        def score(_graphs: object) -> NativeScoreBatch:
            return NativeScoreBatch(
                probabilities=np.asarray(
                    [[0.0, 1.0, 0.0], [0.6, 0.1, 0.3]], dtype=float
                ),
                predictions=(1, 0),
                scores=(0.0, 0.9),
                candidate_flags=(False, True),
                graph_embeddings=np.zeros((2, 2), dtype=np.float32),
                valid_fullgraphs=(False, True),
                failure_reasons=("invalid_fullgraph", ""),
            )

    vrrw = _FakeVRRW()
    bridge = TasteGCFImportanceBridge(
        adapter=CollisionAdapter(),
        vrrw=vrrw,
        importance=_FakeImportance(),
        neurosed_model=object(),
        original_graph_element_counts=torch.ones(8),
        distance_threshold=0.25,
        parent_count=8,
        feature_atomic_numbers=(6, 7),
    )
    importance = SimpleNamespace(call=lambda *_args: None)
    with bridge.installed(importance):
        _parts, embeddings, _coverage = importance.call(graphs, {})
        hashes = [vrrw.calculate_hash(row) for row in embeddings]
        assert hashes == ["a" * 64, "b" * 64]
        assert bridge.is_graph_counterfactual(hashes[0]) is False
        assert bridge.is_graph_counterfactual(hashes[1]) is True
    assert len(bridge.records) == 2
    assert all(type(value) is str and len(value) == 64 for value in bridge.records)


def test_importance_bridge_rejects_graph_embedding_call_order_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = SimpleNamespace(tag="a")
    monkeypatch.setattr(
        taste_gcf,
        "canonical_attributed_graph",
        lambda *_args, **_kwargs: SimpleNamespace(
            graph_identity_sha256="a" * 64,
            collision_payload=lambda: {
                "canonical_graph": "[a]",
                "num_nodes": 1,
                "num_edges": 0,
            },
        ),
    )
    vrrw = _FakeVRRW()
    bridge = TasteGCFImportanceBridge(
        adapter=_FakeAdapter(),
        vrrw=vrrw,
        importance=_FakeImportance(),
        neurosed_model=object(),
        original_graph_element_counts=torch.ones(8),
        distance_threshold=0.25,
        parent_count=8,
        feature_atomic_numbers=(6, 7),
    )
    # Use a one-row adapter so the pending queue has one unambiguous row.
    bridge.adapter = SimpleNamespace(
        score=lambda _graphs: NativeScoreBatch(
            probabilities=np.asarray([[0.3, 0.4, 0.3]], dtype=float),
            predictions=(1,),
            scores=(0.6,),
            candidate_flags=(False,),
            graph_embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
            valid_fullgraphs=(True,),
            failure_reasons=("",),
        )
    )
    bridge.importance = SimpleNamespace(
        neurosed_threshold_coverage_estimation=lambda *_args: torch.ones((1, 8))
    )
    _parts, _embeddings, _coverage = bridge.call([graph], {})
    with pytest.raises(TasteGCFSmokeError, match="call order changed"):
        bridge.calculate_hash(np.asarray([2.0, 1.0], dtype=np.float32))


def test_real_official_restart_reads_scoped_global_and_restores_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh-load the vendored module and execute its real restart function."""

    import torch

    geometric = types.ModuleType("torch_geometric")
    geometric.__path__ = []  # type: ignore[attr-defined]
    geometric_utils = types.ModuleType("torch_geometric.utils")
    geometric.utils = geometric_utils  # type: ignore[attr-defined]
    importance = types.ModuleType("importance")
    observed: list[object] = []

    def call(graphs: object, args: object) -> tuple[object, object, object]:
        del graphs
        observed.append(args)
        return (
            np.asarray([[0.4, 1.0]], dtype=float),
            np.asarray([[1.0, 2.0]], dtype=float),
            torch.ones((1, 1), dtype=torch.float),
        )

    importance.call = call  # type: ignore[attr-defined]
    data = types.ModuleType("data")
    data.load_dataset = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    gnn = types.ModuleType("gnn")
    gnn.load_trained_gnn = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    gnn.load_trained_prediction = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    for name, module in (
        ("torch_geometric", geometric),
        ("torch_geometric.utils", geometric_utils),
        ("importance", importance),
        ("data", data),
        ("gnn", gnn),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_taste_t7_fresh_official_vrrw"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "baselines/gcfexplainer_official/vrrw.py"
    )
    assert spec is not None and spec.loader is not None
    vrrw = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, vrrw)
    spec.loader.exec_module(vrrw)
    assert not hasattr(vrrw, "importance_args")
    vrrw.input_graphs_covered = torch.zeros(1, dtype=torch.float)
    vrrw.MAX_COUNTERFACTUAL_SIZE = 2
    args = {"alpha": 1.0, "sentinel": "fresh"}
    with _installed_official_importance_args(vrrw, args):
        assert vrrw.restart_randomwalk([object()]) == vrrw.calculate_hash(
            np.asarray([1.0, 2.0], dtype=float)
        )
    assert observed == [args]
    assert not hasattr(vrrw, "importance_args")

    old = object()
    vrrw.importance_args = old
    replacement = {"alpha": 1.0, "sentinel": "replacement"}
    with _installed_official_importance_args(vrrw, replacement):
        vrrw.restart_randomwalk([object()])
    assert observed[-1] is replacement
    assert vrrw.importance_args is old


@pytest.mark.parametrize("failure", ("missing", "content_drift", "replacement"))
def test_private_progress_checkpoint_rejects_missing_or_drifted_inode(
    tmp_path: Path, failure: str
) -> None:
    import torch

    checkpoint = taste_gcf._HeldVRRWProgressCheckpoint.write(
        _checkpoint_path(tmp_path),
        _checkpoint_fixture(torch),
        torch=torch,
    )
    try:
        if failure == "missing":
            os.unlink(checkpoint.path)
        elif failure == "content_drift":
            descriptor = os.open(checkpoint.path, os.O_WRONLY)
            try:
                os.pwrite(descriptor, b"X", 0)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            original = checkpoint.path.read_bytes()
            os.unlink(checkpoint.path)
            descriptor = os.open(
                checkpoint.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                assert os.write(descriptor, original) == len(original)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        with pytest.raises(
            TasteGCFSmokeError,
            match="missing|replaced|identity drifted|SHA-256 drifted",
        ):
            checkpoint.load(torch=torch)
    finally:
        checkpoint.close()


def test_progress_checkpoint_close_never_unlinks_a_replacement(
    tmp_path: Path,
) -> None:
    import torch

    checkpoint = taste_gcf._HeldVRRWProgressCheckpoint.write(
        _checkpoint_path(tmp_path),
        _checkpoint_fixture(torch),
        torch=torch,
    )
    os.unlink(checkpoint.path)
    checkpoint.path.write_bytes(b"FOREIGN-REPLACEMENT\n")
    os.chmod(checkpoint.path, 0o600)
    checkpoint.close()
    assert checkpoint.path.read_bytes() == b"FOREIGN-REPLACEMENT\n"


def test_progress_checkpoint_rejects_runtime_rename_equal_copy_restore_without_escape(
    tmp_path: Path,
) -> None:
    import torch

    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary_parent:
        envelope = Path(temporary_parent)
        runtime = envelope / "runtime"
        runtime.mkdir(mode=0o700)
        checkpoint = taste_gcf._HeldVRRWProgressCheckpoint.write(
            _checkpoint_path(runtime),
            _checkpoint_fixture(torch),
            torch=torch,
        )
        parked = envelope / "runtime.parked"
        runtime.rename(parked)
        shutil.copytree(parked, runtime)
        shutil.rmtree(runtime)
        parked.rename(runtime)
        try:
            with pytest.raises(
                TasteGCFSmokeError,
                match="temporary parent namespace drifted",
            ):
                checkpoint.load(torch=torch)
            assert list(envelope.rglob("vrrw_progress.pt")) == [
                _checkpoint_path(runtime) / "vrrw_progress.pt"
            ]
        finally:
            checkpoint.close()
    assert not list(tmp_path.rglob("vrrw_progress.pt"))


def test_progress_checkpoint_rejects_directory_rename_equal_copy_restore_without_escape(
    tmp_path: Path,
) -> None:
    import torch

    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary_parent:
        envelope = Path(temporary_parent)
        runtime = envelope / "runtime"
        runtime.mkdir(mode=0o700)
        checkpoint_dir = _checkpoint_path(runtime)
        checkpoint = taste_gcf._HeldVRRWProgressCheckpoint.write(
            checkpoint_dir,
            _checkpoint_fixture(torch),
            torch=torch,
        )
        parked = checkpoint_dir.parent / f"{_CHECKPOINT_UUID}.parked"
        checkpoint_dir.rename(parked)
        shutil.copytree(parked, checkpoint_dir)
        shutil.rmtree(checkpoint_dir)
        parked.rename(checkpoint_dir)
        try:
            with pytest.raises(
                TasteGCFSmokeError,
                match="container identity drifted|directory identity drifted|missing/replaced",
            ):
                checkpoint.load(torch=torch)
            assert list(envelope.rglob("vrrw_progress.pt")) == [
                checkpoint_dir / "vrrw_progress.pt"
            ]
        finally:
            checkpoint.close()
    assert not list(tmp_path.rglob("vrrw_progress.pt"))


def test_progress_checkpoint_rejects_container_rename_equal_copy_restore_without_escape(
    tmp_path: Path,
) -> None:
    import torch

    with tempfile.TemporaryDirectory(dir=tmp_path) as temporary_parent:
        envelope = Path(temporary_parent)
        runtime = envelope / "runtime"
        runtime.mkdir(mode=0o700)
        checkpoint = taste_gcf._HeldVRRWProgressCheckpoint.write(
            _checkpoint_path(runtime),
            _checkpoint_fixture(torch),
            torch=torch,
        )
        container = runtime / "checkpoints"
        parked = runtime / "checkpoints.parked"
        container.rename(parked)
        shutil.copytree(parked, container)
        shutil.rmtree(container)
        parked.rename(container)
        try:
            with pytest.raises(
                TasteGCFSmokeError,
                match="runtime root|container identity|missing or replaced",
            ):
                checkpoint.load(torch=torch)
            assert list(envelope.rglob("vrrw_progress.pt")) == [
                _checkpoint_path(runtime) / "vrrw_progress.pt"
            ]
        finally:
            checkpoint.close()
    assert not list(tmp_path.rglob("vrrw_progress.pt"))


def test_progress_checkpoint_requires_uuid_and_never_reuses_path(
    tmp_path: Path,
) -> None:
    import torch

    with pytest.raises(TasteGCFSmokeError, match="checkpoints/<UUIDv4>"):
        taste_gcf._HeldVRRWProgressCheckpoint.write(
            tmp_path / "progress-checkpoint",
            _checkpoint_fixture(torch),
            torch=torch,
        )
    first = taste_gcf._HeldVRRWProgressCheckpoint.write(
        _checkpoint_path(tmp_path),
        _checkpoint_fixture(torch),
        torch=torch,
    )
    try:
        with pytest.raises(FileExistsError):
            taste_gcf._HeldVRRWProgressCheckpoint.write(
                _checkpoint_path(tmp_path),
                _checkpoint_fixture(torch),
                torch=torch,
            )
    finally:
        first.close()


def _fake_progress_runtime() -> tuple[object, object, object, object]:
    import torch

    identity = "a" * 64
    vrrw = SimpleNamespace(
        graph_map={11: {"x": torch.tensor([[1.0]])}},
        graph_index_map={11: 0},
        counterfactual_candidates=[
            {
                "frequency": 2,
                "graph_hash": 11,
                "importance_parts": [0.5, 1.0],
                "input_graphs_covering_list": torch.zeros(1).to_sparse(),
            }
        ],
        input_graphs_covered=torch.zeros(1),
        covering_graphs=set(),
        transitions={},
        traversed_hashes=[11] * 8,
        MAX_COUNTERFACTUAL_SIZE=512,
        starting_step=1,
        dataset_name="tastemolnet",
        alpha=1.0,
        sample_size=128,
        is_sample=True,
        importance_args={"alpha": 1.0},
    )
    bridge = SimpleNamespace(
        records={
            11: {
                "graph_identity_sha256": identity,
                "probabilities": [0.2, 0.7, 0.1],
                "pred_candidate": 1,
                "score": 0.3,
                "candidate_condition": False,
                "valid_fullgraph": True,
                "failure_reason": "",
            }
        },
        call_count=2,
        evaluated_graph_count=3,
        calculate_hash_count=3,
        distance_call_count=2,
        distance_evaluated_graph_count=3,
        canonical_row_reuse_count=0,
        _pending_hashes=taste_gcf.deque(),
    )
    bridge._assert_idle = lambda: None
    scorer = SimpleNamespace(
        cache_capacity=0,
        _cache={},
        calls=2,
        cache_hits=0,
        cache_misses=3,
        scored_rows=3,
        last_trace=None,
    )
    adapter = SimpleNamespace(
        scorer=scorer,
        decode_failures=taste_gcf.Counter(),
        decode_success_count=3,
        empty_valid_batch_count=0,
        call_count=2,
    )
    return vrrw, bridge, adapter, taste_gcf.Counter({"NLC": 1})


def test_progress_restore_detects_live_restore_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    vrrw, bridge, adapter, action_counts = _fake_progress_runtime()
    progress = taste_gcf._capture_progress_state(
        vrrw=vrrw,
        bridge=bridge,
        adapter=adapter,
        action_counts=action_counts,
        current_graph_hash=11,
    )
    rng = taste_gcf._capture_rng_state(np=np, torch=torch)
    payload = {
        "schema_version": "tastemolnet_t7_gcf_vrrw_progress_checkpoint_v2",
        "stage": "T7_GCF_SMOKE",
        "checkpoint_uuid": _CHECKPOINT_UUID,
        "generation_token": "f" * 64,
        "completed_steps": 8,
        "total_steps": 16,
        "resume_start_step": 9,
        "progress": copy.deepcopy(progress),
        "rng": copy.deepcopy(rng),
        "progress_state_sha256": taste_gcf._semantic_sha256(progress),
        "rng_state_sha256": taste_gcf._semantic_sha256(rng),
        "trace_prefix_sha256": "b" * 64,
        "resume_graph_identity_sha256": "a" * 64,
    }
    original_capture = taste_gcf._capture_progress_state

    def mismatched_capture(**kwargs: object) -> dict[str, object]:
        captured = original_capture(**kwargs)
        captured = dict(captured)
        captured["current_graph_hash"] = 12
        return captured

    monkeypatch.setattr(taste_gcf, "_capture_progress_state", mismatched_capture)
    with pytest.raises(TasteGCFSmokeError, match="restore state mismatch"):
        taste_gcf._apply_progress_state(
            payload,
            vrrw=vrrw,
            bridge=bridge,
            adapter=adapter,
            action_counts=action_counts,
            np=np,
            torch=torch,
        )


def test_real_split_runner_restores_prefix_rng_and_resume_cursor(
    tmp_path: Path,
) -> None:
    import torch

    bridge = SimpleNamespace(
        records={},
        call_count=0,
        evaluated_graph_count=0,
        calculate_hash_count=0,
        distance_call_count=0,
        distance_evaluated_graph_count=0,
        canonical_row_reuse_count=0,
        _pending_hashes=taste_gcf.deque(),
    )
    bridge._assert_idle = lambda: None
    scorer = SimpleNamespace(
        cache_capacity=0,
        _cache={},
        calls=0,
        cache_hits=0,
        cache_misses=0,
        scored_rows=0,
        last_trace=None,
    )
    adapter = SimpleNamespace(
        scorer=scorer,
        decode_failures=taste_gcf.Counter(),
        decode_success_count=0,
        empty_valid_batch_count=0,
        call_count=0,
    )
    action_counts = taste_gcf.Counter()

    class FakeOfficialVRRW:
        def __init__(self) -> None:
            self.segment_traces: list[list[int]] = []
            self.segment_cursors: list[int] = []
            self.reset()

        def reset(self) -> None:
            self.graph_map: dict[int, object] = {}
            self.graph_index_map: dict[int, int] = {}
            self.counterfactual_candidates: list[dict[str, object]] = []
            self.input_graphs_covered = torch.zeros(1)
            self.covering_graphs: set[int] = set()
            self.transitions: dict[int, object] = {}
            self.traversed_hashes: list[int] = []
            self.MAX_COUNTERFACTUAL_SIZE = 512
            self.starting_step = 1
            self.dataset_name = "tastemolnet"
            self.alpha = 1.0
            self.sample_size = 128
            self.is_sample = True
            self.importance_args = {"alpha": 1.0}

        def ensure_graph(self, graph_hash: int) -> None:
            if graph_hash in self.graph_map:
                candidate = self.counterfactual_candidates[
                    self.graph_index_map[graph_hash]
                ]
                candidate["frequency"] = int(candidate["frequency"]) + 1
                self.counterfactual_candidates.sort(
                    key=lambda row: int(row["frequency"]), reverse=True
                )
                self.graph_index_map = {
                    int(row["graph_hash"]): index
                    for index, row in enumerate(self.counterfactual_candidates)
                }
                return
            self.graph_map[graph_hash] = {
                "x": torch.tensor([[float(graph_hash)]])
            }
            self.graph_index_map[graph_hash] = len(
                self.counterfactual_candidates
            )
            self.counterfactual_candidates.append(
                {
                    "frequency": 2,
                    "graph_hash": graph_hash,
                    "importance_parts": [0.5, 1.0],
                    "input_graphs_covering_list": torch.zeros(1).to_sparse(),
                }
            )
            bridge.records[graph_hash] = {
                "graph_identity_sha256": taste_gcf._sha256_bytes(
                    str(graph_hash).encode("ascii")
                )
            }

        def restart_randomwalk(self, input_graphs: object) -> int:
            graph_hash = int(random.choice(input_graphs))
            self.ensure_graph(graph_hash)
            return graph_hash

        def move_to_next_graph(
            self, graph_hash: int, **_kwargs: object
        ) -> tuple[int | None, bool]:
            if random.random() < 0.25:
                return None, True
            next_hash = graph_hash + random.choice((1, 2, 3))
            self.ensure_graph(next_hash)
            action_counts["NLC"] += 1
            bridge.call_count += 1
            bridge.evaluated_graph_count += 1
            bridge.distance_call_count += 1
            bridge.distance_evaluated_graph_count += 1
            adapter.call_count += 1
            adapter.decode_success_count += 1
            scorer.calls += 1
            scorer.cache_misses += 1
            scorer.scored_rows += 1
            return next_hash, False

        def counterfactual_summary_with_randomwalk(
            self,
            input_graphs: object,
            importance_args: object,
            teleport_probability: float,
            max_steps: int,
        ) -> None:
            del importance_args, teleport_probability
            current = self.restart_randomwalk(input_graphs)
            for _step in range(self.starting_step, max_steps + 1):
                self.traversed_hashes.append(current)
                next_hash, teleported = self.move_to_next_graph(
                    current,
                    importance_args={},
                    teleport_probability=0.1,
                )
                current = (
                    self.restart_randomwalk(input_graphs)
                    if teleported
                    else int(next_hash)
                )
                assert len(self.graph_map) == len(self.graph_index_map) == len(
                    self.counterfactual_candidates
                )
            self.segment_traces.append(list(self.traversed_hashes))
            self.segment_cursors.append(current)

    vrrw = FakeOfficialVRRW()

    def reset_official(target: FakeOfficialVRRW) -> None:
        target.reset()

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    evidence = taste_gcf._execute_checkpointed_official_walk(
        vrrw=vrrw,
        bridge=bridge,
        adapter=adapter,
        action_counts=action_counts,
        input_graphs=[100, 200],
        importance_args={"alpha": 1.0},
        teleport_probability=0.1,
        runtime_root=tmp_path,
        reset_official_vrrw=reset_official,
        np=np,
        torch=torch,
    )
    taste_gcf._validate_progress_evidence(evidence)
    assert evidence["checkpoint_written"] is True
    assert evidence["planned_interruption_observed"] is True
    assert evidence["checkpoint_reloaded"] is True
    assert evidence["resumed"] is True
    assert evidence["pre_resume_step_count"] == 8
    assert evidence["post_resume_step_count"] == 8
    assert evidence["total_step_count"] == 16
    assert len(vrrw.segment_traces) == 2
    assert len(vrrw.segment_traces[0]) == 8
    assert len(vrrw.segment_traces[1]) == 16
    assert vrrw.segment_traces[1][:8] == vrrw.segment_traces[0]
    assert vrrw.segment_traces[1][8] == vrrw.segment_cursors[0]
    split_identities = taste_gcf._trace_identities(
        bridge, list(vrrw.traversed_hashes)
    )
    split_candidates = [
        (row["graph_hash"], row["frequency"])
        for row in vrrw.counterfactual_candidates
    ]
    split_actions = dict(action_counts)
    split_rng_sha256 = taste_gcf._semantic_sha256(
        taste_gcf._capture_rng_state(np=np, torch=torch)
    )

    # Run the same fake official state machine uninterrupted from the same
    # seed.  Resume must be a state restoration, not a replay variant with a
    # second initial restart/reinsertion or an off-by-one move.
    reset_official(vrrw)
    bridge.records = {}
    bridge.call_count = 0
    bridge.evaluated_graph_count = 0
    bridge.distance_call_count = 0
    bridge.distance_evaluated_graph_count = 0
    adapter.decode_failures = taste_gcf.Counter()
    adapter.decode_success_count = 0
    adapter.empty_valid_batch_count = 0
    adapter.call_count = 0
    scorer._cache.clear()
    scorer.calls = 0
    scorer.cache_hits = 0
    scorer.cache_misses = 0
    scorer.scored_rows = 0
    scorer.last_trace = None
    action_counts.clear()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    vrrw.counterfactual_summary_with_randomwalk(
        input_graphs=[100, 200],
        importance_args={"alpha": 1.0},
        teleport_probability=0.1,
        max_steps=16,
    )
    uninterrupted_identities = taste_gcf._trace_identities(
        bridge, list(vrrw.traversed_hashes)
    )
    uninterrupted_candidates = [
        (row["graph_hash"], row["frequency"])
        for row in vrrw.counterfactual_candidates
    ]
    uninterrupted_rng_sha256 = taste_gcf._semantic_sha256(
        taste_gcf._capture_rng_state(np=np, torch=torch)
    )
    assert uninterrupted_identities == split_identities
    assert uninterrupted_candidates == split_candidates
    assert dict(action_counts) == split_actions
    assert uninterrupted_rng_sha256 == split_rng_sha256
    progress_files = list(tmp_path.glob("checkpoints/*/vrrw_progress.pt"))
    assert len(progress_files) == 1
    assert progress_files[0].parent.name != "progress-checkpoint"


def test_release_is_disabled_and_all_pins_are_null(tmp_path: Path) -> None:
    release = json.loads(RELEASE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert release["release_enabled"] is False
    assert release["release_state"] == DISABLED_RELEASE_STATE
    assert release["gpu_index"] == SMOKE_GPU_INDEX
    assert all(
        value is None
        for key, value in release.items()
        if key
        not in {"schema_version", "release_enabled", "release_state", "gpu_index"}
    )
    with pytest.raises(TasteGCFSmokeReleaseDisabled, match="NOT_RELEASED"):
        assert_execution_released()
    output = tmp_path / "must-not-exist"
    result = cli.main(
        [
            "--config",
            str(ROOT / "configs/hpc.yaml"),
            "--output-dir",
            str(output),
            "--set",
            "inference.fallback_to_heuristic=false",
        ]
    )
    assert result == 78
    assert not output.exists()


def test_runtime_cli_emits_only_worker_sealed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[object, object]] = []

    def finished_runner(*, output_dir: object, config_path: object) -> dict[str, str]:
        calls.append((output_dir, config_path))
        return {"status": "SEALED_PENDING_INDEPENDENT_VERIFICATION"}

    monkeypatch.setattr(cli, "run_tastemolnet_gcf_smoke", finished_runner)
    monkeypatch.setattr(
        cli,
        "load_tastemolnet_gcf_verified_gate",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("worker CLI attempted verifier consumption")
        ),
    )
    result = cli.main(
        [
            "--config",
            str(ROOT / "configs/hpc.yaml"),
            "--output-dir",
            str(tmp_path / "output"),
            "--set",
            "inference.fallback_to_heuristic=false",
        ]
    )
    assert result == 0
    assert len(calls) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "SEALED_PENDING_INDEPENDENT_VERIFICATION"
    }
    assert captured.err == ""


def test_worker_raw_evidence_is_private_summary_only() -> None:
    trace = [_trace_row([0.6, 0.1, 0.3])]
    raw = build_worker_raw_evidence(
        inputs=_dummy_inputs(),
        managed_worker=_dummy_managed_worker(),
        train_evidence={"graph_schema_sha256": "0" * 64},
        science={"trace": trace, "summary": _science_summary()},
    )
    serialized = taste_gcf._canonical_bytes(raw).lower()
    assert raw["status"] == "SEALED_PENDING_INDEPENDENT_VERIFICATION"
    assert raw["worker_terminal_authority"] is False
    assert raw["independent_verification_required"] is True
    assert b'"smiles"' not in serialized
    assert b'"molecule_id"' not in serialized
    assert b"counterfactuals.pt" not in serialized
    assert b"vrrw_progress.pt" not in serialized
    assert b'"graph_map"' not in serialized
    assert b"randomforestclassifier" not in serialized


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_written", 1),
        ("planned_interruption_observed", False),
        ("checkpoint_reloaded", "true"),
        ("resumed", 1),
        ("pre_resume_step_count", True),
        ("post_resume_step_count", 7),
        ("total_step_count", "16"),
        ("restored_progress_state_sha256", "9" * 64),
        ("first_post_resume_graph_identity_sha256", "8" * 64),
    ),
)
def test_independent_verifier_rejects_fake_checkpoint_resume_evidence(
    field: str, value: object
) -> None:
    summary = copy.deepcopy(_science_summary())
    summary["progress_checkpoint"][field] = value  # type: ignore[index]
    managed = _dummy_managed_worker()
    raw = build_worker_raw_evidence(
        inputs=_dummy_inputs(),
        managed_worker=managed,
        train_evidence={"graph_schema_sha256": "0" * 64},
        science={
            "trace": [_trace_row([0.6, 0.1, 0.3])],
            "summary": summary,
        },
    )
    with pytest.raises(TasteGCFSmokeError):
        verify_t7_worker_raw_evidence(
            raw,
            expected_attempt_id=managed.attempt_id,
            expected_generation_token=managed.generation_token,
            expected_final_path=managed.expected_final_path,
            expected_predecessor=managed.predecessor_evidence()[0],
            expected_input_hashes=raw["input_hashes"],
        )


def _valid_worker_raw() -> tuple[SimpleNamespace, dict[str, object]]:
    managed = _dummy_managed_worker()
    raw = build_worker_raw_evidence(
        inputs=_dummy_inputs(),
        managed_worker=managed,
        train_evidence={"graph_schema_sha256": "0" * 64},
        science={
            "trace": [_trace_row([0.6, 0.1, 0.3])],
            "summary": _science_summary(),
        },
    )
    return managed, raw


def test_independent_verifier_accepts_exact_managed_raw_evidence() -> None:
    managed, raw = _valid_worker_raw()
    result = verify_t7_worker_raw_evidence(
        raw,
        expected_attempt_id=managed.attempt_id,
        expected_generation_token=managed.generation_token,
        expected_final_path=managed.expected_final_path,
        expected_predecessor=managed.predecessor_evidence()[0],
        expected_input_hashes=raw["input_hashes"],
    )
    assert result["status"] == "PASS"
    assert result["independent_verifier"] is True
    assert result["same_calibrated_three_class_gine"] is True
    assert result["taste_neurosed_revalidated"] is True


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("expected_final_path",), "/private/other"),
        (("predecessors", 0, "sha256"), "8" * 64),
        (
            ("managed_attempt_input_hashes", "taste_gcf_neurosed_pass"),
            "8" * 64,
        ),
        (("scientific_summary", "deletion_only_semantics"), True),
        (("scientific_summary", "distance_status"), "NOT_EVALUATED"),
        (
            ("scientific_summary", "neurosed_predecessor", "test_loaded"),
            True,
        ),
    ),
)
def test_independent_verifier_rejects_managed_cross_binding_drift(
    path: tuple[object, ...], value: object
) -> None:
    managed, raw = _valid_worker_raw()
    target: object = raw
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(TasteGCFSmokeError):
        verify_t7_worker_raw_evidence(
            raw,
            expected_attempt_id=managed.attempt_id,
            expected_generation_token=managed.generation_token,
            expected_final_path=managed.expected_final_path,
            expected_predecessor=managed.predecessor_evidence()[0],
            expected_input_hashes=raw["input_hashes"],
        )


class _FakeManagedDocument:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name

    def revalidate(self) -> None:
        self.events.append(f"{self.name}.revalidate")

    def close(self) -> None:
        self.events.append(f"{self.name}.close")


def test_t7_managed_adapter_is_worker_only_and_seals_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_id = "123e4567-e89b-42d3-a456-426614174000"
    staging_id = "123e4567-e89b-42d3-a456-426614174001"
    generation = "123e4567-e89b-42d3-a456-426614174002"
    stage_root = tmp_path / "stage"
    stage_root.mkdir()
    events: list[str] = []

    class Attempt:
        attempt_path = stage_root / "attempts" / attempt_id

        def __init__(self) -> None:
            self.attempt_id = attempt_id

        def revalidate(self) -> dict[str, object]:
            events.append("attempt.revalidate")
            return {
                "input_hashes": {
                    "managed_execution_v2_pass": "8" * 64,
                    "taste_gcf_neurosed_pass": "9" * 64,
                    "taste_gcf_neurosed_gate": "1" * 64,
                    "taste_gcf_neurosed_verification": "2" * 64,
                    "taste_gcf_neurosed_feature_schema": "3" * 64,
                    "taste_gcf_neurosed_sha256s": "4" * 64,
                }
            }

        def close(self) -> None:
            events.append("attempt.close")

    attempt = Attempt()

    class Staging:
        def __init__(self) -> None:
            self.attempt = attempt
            self.staging_id = staging_id
            self.generation_token = generation
            self.path = (
                attempt.attempt_path / "worker_staging" / staging_id
            )
            self.artifact_root = self.path / "artifacts"

        def revalidate(self) -> None:
            events.append("staging.revalidate")

        def close(self) -> None:
            events.append("staging.close")

    staging = Staging()

    def create_attempt(**kwargs: object) -> Attempt:
        assert kwargs["input_hashes"]["taste_gcf_neurosed_pass"] == "9" * 64  # type: ignore[index]
        events.append("create_attempt")
        return attempt

    def create_staging(observed: object) -> Staging:
        assert observed is attempt
        events.append("create_staging")
        return staging

    def write_raw(observed: object, payload: object) -> _FakeManagedDocument:
        assert observed is staging
        assert payload["expected_final_path"] == str(tmp_path / "final")  # type: ignore[index]
        events.append("write_raw")
        return _FakeManagedDocument(events, "raw")

    def write_exit(observed: object, payload: object) -> _FakeManagedDocument:
        assert observed is staging
        assert payload["status"] == "COMPLETED_PENDING_INDEPENDENT_VERIFICATION"  # type: ignore[index]
        events.append("write_exit")
        return _FakeManagedDocument(events, "exit")

    def seal(observed: object) -> SimpleNamespace:
        assert observed is staging
        events.append("seal")
        return SimpleNamespace(
            attempt_id=attempt_id,
            generation_token=generation,
            staging_path=staging.path,
            artifact_root=staging.artifact_root,
        )

    monkeypatch.setattr(
        t7_managed,
        "_managed_worker_api",
        lambda: (create_attempt, create_staging, write_raw, write_exit, seal),
    )
    worker = t7_managed.create_t7_managed_worker(
        stage_root=stage_root,
        expected_final_path=tmp_path / "final",
        controller_id="controller",
        task_id="task",
        git_commit="a" * 40,
        config_hash="1" * 64,
        input_hashes={
            "managed_execution_v2_pass": "8" * 64,
            "taste_gcf_neurosed_pass": "9" * 64,
            "taste_gcf_neurosed_gate": "1" * 64,
            "taste_gcf_neurosed_verification": "2" * 64,
            "taste_gcf_neurosed_feature_schema": "3" * 64,
            "taste_gcf_neurosed_sha256s": "4" * 64,
        },
        neurosed_pass_path=tmp_path / "neurosed-pass.json",
        neurosed_pass_sha256="9" * 64,
    )
    payload = {
        "schema_version": t7_managed.T7_RAW_EVIDENCE_SCHEMA,
        "attempt_id": attempt_id,
        "generation_token": generation,
        "expected_final_path": str(tmp_path / "final"),
        "predecessors": worker.predecessor_evidence(),
    }
    worker.seal_raw_evidence(payload)
    assert events.index("write_raw") < events.index("write_exit") < events.index("seal")
    worker.close()


def test_t7_managed_adapter_uses_exact_frozen_v2_api_unmocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    stage_root = tmp_path / "managed-stage"
    stage_root.mkdir(mode=0o700)
    final_path = tmp_path / "published" / "t7"
    input_hashes = {
        "managed_execution_v2_pass": "8" * 64,
        "taste_gcf_neurosed_pass": "9" * 64,
        "taste_gcf_neurosed_gate": "1" * 64,
        "taste_gcf_neurosed_verification": "2" * 64,
        "taste_gcf_neurosed_feature_schema": "3" * 64,
        "taste_gcf_neurosed_sha256s": "4" * 64,
    }
    worker = t7_managed.create_t7_managed_worker(
        stage_root=stage_root,
        expected_final_path=final_path,
        controller_id="controller",
        task_id="T7_GCF_SMOKE:test",
        git_commit="a" * 40,
        config_hash="1" * 64,
        input_hashes=input_hashes,
        neurosed_pass_path=tmp_path / "neurosed" / "PASS.json",
        neurosed_pass_sha256="9" * 64,
    )
    payload = {
        "schema_version": t7_managed.T7_RAW_EVIDENCE_SCHEMA,
        "attempt_id": worker.attempt_id,
        "generation_token": worker.generation_token,
        "expected_final_path": str(final_path),
        "predecessors": worker.predecessor_evidence(),
    }
    try:
        assert worker.attempt_input_hashes() == input_hashes
        sealed = worker.seal_raw_evidence(payload)
        assert len(sealed.seal_sha256) == 64
        assert len(sealed.inventory_sha256) == 64
        outer = json.loads(
            (sealed.staging_path / "raw_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        assert outer["schema_version"] == "managed_worker_raw_evidence_v2"
        assert outer["evidence"] == payload
        assert not (sealed.staging_path / "PASS").exists()
        assert not (sealed.staging_path / "gate.json").exists()
        assert not (sealed.staging_path / "verification.json").exists()
    finally:
        worker.close()


def test_t7_independent_verification_publishes_only_through_managed_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.utils.terminal_publisher_v2 import (
        open_sealed_worker_artifact,
        verify_and_publish_sealed_attempt,
    )

    monkeypatch.setenv("AUTO_TERMINATE_UNCONTROLLED_CHILDREN", "0")
    stage_root = tmp_path / "managed-stage"
    stage_root.mkdir(mode=0o700)
    final_path = tmp_path / "published" / "t7"
    final_path.parent.mkdir(mode=0o700)
    inputs = _dummy_inputs()
    managed_input_hashes = {
        "managed_execution_v2_pass": "8" * 64,
        "taste_gcf_neurosed_pass": "9" * 64,
        "taste_gcf_neurosed_gate": "1" * 64,
        "taste_gcf_neurosed_verification": "2" * 64,
        "taste_gcf_neurosed_checkpoint": "0" * 64,
        "taste_gcf_neurosed_feature_schema": "3" * 64,
        "taste_gcf_neurosed_sha256s": "4" * 64,
        "taste_gine_t2_gate": "9" * 64,
        "taste_gine_t3_gate": "a" * 64,
        "taste_oracle_t4_gate": "d" * 64,
        "taste_train_csv": "f" * 64,
    }
    worker = t7_managed.create_t7_managed_worker(
        stage_root=stage_root,
        expected_final_path=final_path,
        controller_id="controller",
        task_id="T7_GCF_SMOKE:test",
        git_commit="a" * 40,
        config_hash="1" * 64,
        input_hashes=managed_input_hashes,
        neurosed_pass_path=tmp_path / "neurosed" / "PASS.json",
        neurosed_pass_sha256="9" * 64,
    )
    try:
        raw = build_worker_raw_evidence(
            inputs=inputs,
            managed_worker=worker,
            train_evidence={"graph_schema_sha256": "0" * 64},
            science={
                "trace": [_trace_row([0.6, 0.1, 0.3])],
                "summary": {
                    **_science_summary(),
                    "neurosed_predecessor": {
                        **_science_summary()["neurosed_predecessor"],
                        "pass_path": str(tmp_path / "neurosed" / "PASS.json"),
                    },
                },
            },
        )
        sealed = worker.seal_raw_evidence(raw)
        verification = verify_t7_worker_raw_evidence(
            raw,
            expected_attempt_id=worker.attempt_id,
            expected_generation_token=worker.generation_token,
            expected_final_path=final_path,
            expected_predecessor=worker.predecessor_evidence()[0],
            expected_input_hashes=raw["input_hashes"],
        )
        with open_sealed_worker_artifact(
            sealed.seal_path,
            expected_attempt_id=worker.attempt_id,
            expected_generation_token=worker.generation_token,
        ) as held:
            publication = verify_and_publish_sealed_attempt(
                held,
                final_path=final_path,
                verification=verification,
            )
        assert publication.final_path == final_path
        assert (final_path / "PASS").read_bytes() == (
            b"[MANAGED_EXECUTION_V2_PASS]\n"
        )
        assert taste_gcf.load_tastemolnet_gcf_verified_gate(final_path)[
            "status"
        ] == "PASS"
    finally:
        worker.close()


def test_t7_worker_source_has_no_verifier_or_legacy_publisher() -> None:
    adapter_source = (
        ROOT / "src/utils/tastemolnet_t7_managed_v2.py"
    ).read_text(encoding="utf-8")
    core_source = (
        ROOT / "src/baselines/tastemolnet_gcf_smoke.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "verify_and_publish_sealed_attempt",
        "open_sealed_worker_artifact",
        "commit_final_rename",
        "prepare_terminal_output",
        "probe_terminal_link_capability",
    ):
        assert forbidden not in adapter_source
        assert forbidden not in core_source


@pytest.mark.parametrize("tag", ("S", "h"))
def test_checkout_rejects_hidden_index_flags(
    monkeypatch: pytest.MonkeyPatch, tag: str
) -> None:
    monkeypatch.setattr(
        t7_release, "_git_output", lambda *_args: f"{tag} tracked.py\0"
    )
    with pytest.raises(TasteGCFSmokeError, match="skip-worktree"):
        t7_release._reject_hidden_index_flags()


def test_wrappers_are_autodl_only_and_static_refusal() -> None:
    wrapper = (
        ROOT / "scripts/autodl/run_tastemolnet_gcf_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "TASTE_T7_GCF_WRAPPER_RELEASED=0" in wrapper
    assert "TASTE_T7_GCF_WRAPPER_NOT_RELEASED" in wrapper
    assert wrapper.index("TASTE_T7_GCF_WRAPPER_RELEASED=0") < wrapper.index(
        'source "$SCRIPT_DIR/common.sh"'
    )
    assert wrapper.index("TASTE_T7_GCF_WRAPPER_NOT_RELEASED") < wrapper.index(
        'source "$SCRIPT_DIR/common.sh"'
    )
    assert "--stage T7_GCF_SMOKE" in wrapper
    assert "--gpu-index 0" in wrapper
    assert "--required-log-marker" not in wrapper

    slurm = (
        ROOT / "scripts/slurm/run_tastemolnet_gcf_smoke.sh"
    ).read_text(encoding="utf-8")
    for required in (
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
        assert required in slurm
    assert slurm.index("exit 64") < slurm.index(
        "python scripts/run_tastemolnet_gcf_smoke.py"
    )


def test_runtime_source_binds_predecessors_and_never_uses_bace_adapter() -> None:
    release_source = (
        ROOT / "src/utils/tastemolnet_t7_gcf_release.py"
    ).read_text(encoding="utf-8")
    core_source = (
        ROOT / "src/baselines/tastemolnet_gcf_smoke.py"
    ).read_text(encoding="utf-8")
    for required in (
        "hold_t2_gine_pass_adoption",
        "hold_taste_stage_output",
        "hold_taste_checkpoint_bundle",
        "controller_receipt_sha256",
        "gpu_lease_receipt_sha256",
        "verify_execution_checkout(self.release)",
        "_verify_critical_blobs(self.authority)",
        "_verify_controller_process(self.controller)",
        "t2_adoption_binding_sha256",
    ):
        assert required in release_source
    t2_source = release_source.split("def hold_t2_adoption", 1)[1].split(
        "def hold_stages_and_checkpoint", 1
    )[0]
    for forbidden in (
        "T2PassAdoptionSources",
        '"control_root"',
        '"controller_root"',
        '"training_state_root"',
        '"execution_project_root"',
        '"identity_fix_project_root"',
    ):
        assert forbidden not in t2_source
    assert "bace_gine_native_adapter" not in core_source
    assert '"TasteFrozenGINENativeAdapter"' in core_source
    assert "_installed_official_importance_args" in core_source
    assert "importance.prepare_and_get" not in core_source
    assert "distance.load_neurosed" in core_source
    assert "neurosed_threshold_coverage_estimation" in core_source
    assert '"neurosed_status": "PASS_INPUT_REVALIDATED"' in core_source
    assert '"distance_status": "EVALUATED"' in core_source
    assert "create_t7_managed_worker" in core_source
    assert "commit_final_rename" not in core_source
    assert "prepare_terminal_output" not in core_source
    assert '"selector_status": "NOT_EVALUATED"' in core_source
    assert '"bace_artifacts_used": False' in core_source
