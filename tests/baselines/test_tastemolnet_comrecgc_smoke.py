from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.baselines.tastemolnet_comrecgc_smoke as smoke_module
import src.baselines.comrecgc.held_upstream as held_upstream_module

from src.baselines.tastemolnet_comrecgc_smoke import (
    OFFICIAL_COMRECGC_COMMIT,
    SMOKE_STEPS,
    PASS_BYTES,
    PASS_FILE,
    TASK_ID,
    TasteComRecGCMulticlassBridge,
    TasteComRecGCSmokeError,
    TasteComRecGCSmokeParameters,
    _common_recourse_summary,
    _checkpoint_state_sha256,
    _restore_reload_checkpoint,
    _write_reload_checkpoint,
    build_terminal_documents,
    canonical_attributed_graph,
    score_and_candidate,
    hold_tastemolnet_comrecgc_output,
    validate_tastemolnet_comrecgc_output,
    validate_native_comrecgc_smoke_result,
    validate_terminal_input_authority,
)
from src.baselines.comrecgc.held_upstream import (
    OFFICIAL_SOURCE_FILES,
    OFFICIAL_SOURCE_SHA256,
    hold_imported_comrecgc,
)


ATOMS = (1, 6, 8)


def test_t9_stage_marker_contract_is_exact() -> None:
    assert smoke_module.PASS_MARKER == "[TASTE_T9_COMRECGC_SMOKE_PASS]"
    assert PASS_BYTES == b"[TASTE_T9_COMRECGC_SMOKE_PASS]\n"


def _fake_official_checkout(root: Path) -> dict[str, str]:
    payloads = {
        "util.py": b"TOKEN = 'reviewed-util'\n",
        "data.py": b"import util\nTOKEN = 'reviewed-data:' + util.TOKEN\n",
        "neurosed/models.py": b"TOKEN = 'reviewed-neurosed'\n",
        "distance.py": (
            b"from neurosed import models\n"
            b"TOKEN = 'reviewed-distance:' + models.TOKEN\n"
        ),
        "gnn.py": b"import data\nTOKEN = 'reviewed-gnn:' + data.TOKEN\n",
        "comrecgc.py": (
            b"import util, distance\n"
            b"TOKEN = 'reviewed-comrecgc:' + util.TOKEN + ':' + distance.TOKEN\n"
        ),
        "common_recourse.py": (
            b"import util, data, distance, gnn\n"
            b"TOKEN = 'reviewed-common:' + gnn.TOKEN\n"
        ),
    }
    assert set(payloads) == set(OFFICIAL_SOURCE_FILES)
    for relative, data in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)
    return {
        relative: hashlib.sha256(data).hexdigest()
        for relative, data in payloads.items()
    }


def test_official_sources_load_only_from_held_descriptors_and_restore_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "official"
    expected = _fake_official_checkout(root)
    monkeypatch.setattr(
        held_upstream_module, "OFFICIAL_SOURCE_SHA256", expected
    )
    sentinel = SimpleNamespace(name="preexisting-util")
    previous = __import__("sys").modules.get("util")
    __import__("sys").modules["util"] = sentinel
    try:
        with hold_imported_comrecgc(
            root, expected_file_sha256=expected
        ) as held:
            evidence = held.revalidate()
            assert evidence["commit"] == OFFICIAL_COMRECGC_COMMIT
            assert evidence["file_sha256"] == expected
            assert evidence["descriptor_loaded"] is True
            assert held.modules["comrecgc"].TOKEN.startswith(
                "reviewed-comrecgc:"
            )
            assert held.modules["common_recourse"].TOKEN.startswith(
                "reviewed-common:"
            )
            for name, relative in held_upstream_module._MODULE_FILES:
                assert held.modules[name].__file__ == (
                    held_upstream_module._descriptor_path(
                        held.sources[relative].file_fd
                    )
                )
        assert __import__("sys").modules["util"] is sentinel
    finally:
        if previous is None:
            __import__("sys").modules.pop("util", None)
        else:
            __import__("sys").modules["util"] = previous


def test_official_source_equal_byte_named_replacement_fails_while_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "official"
    expected = _fake_official_checkout(root)
    monkeypatch.setattr(
        held_upstream_module, "OFFICIAL_SOURCE_SHA256", expected
    )
    with hold_imported_comrecgc(root, expected_file_sha256=expected) as held:
        target = root / "comrecgc.py"
        replacement = root / "comrecgc.replacement"
        replacement.write_bytes(target.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, target)
        with pytest.raises(Exception, match="single-link|identity|named input"):
            held.revalidate()


def test_official_source_swap_load_restore_cannot_change_executed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "official"
    expected = _fake_official_checkout(root)
    monkeypatch.setattr(
        held_upstream_module, "OFFICIAL_SOURCE_SHA256", expected
    )
    real_loader = held_upstream_module._load_source_module
    observed = {"injected": False}

    def swapping_loader(name, held):
        if name != "comrecgc":
            return real_loader(name, held)
        target = root / "comrecgc.py"
        original = root / "comrecgc.original"
        target.rename(original)
        target.write_text("TOKEN = 'malicious'\n", encoding="utf-8")
        try:
            module = real_loader(name, held)
            observed["injected"] = module.TOKEN == "malicious"
            return module
        finally:
            target.unlink()
            original.rename(target)

    monkeypatch.setattr(
        held_upstream_module, "_load_source_module", swapping_loader
    )
    with pytest.raises(Exception, match="identity|named input"):
        hold_imported_comrecgc(root, expected_file_sha256=expected)
    assert observed["injected"] is False


def test_official_sources_reject_a_self_signed_nonreviewed_checkout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "official"
    self_signed = _fake_official_checkout(root)
    with pytest.raises(Exception, match="not the reviewed 122f9341 closure"):
        hold_imported_comrecgc(root, expected_file_sha256=self_signed)


def _terminal_input_authority(tmp_path: Path) -> dict[str, object]:
    from src.utils.tastemolnet_gine_pass_adoption_v1 import (
        ADOPTION_MARKER,
        CHECKPOINT_FILES,
        DOWNSTREAM_BINDING_SCHEMA,
        SOURCE_CID,
        SOURCE_RUN_ID,
    )

    inventory = []
    for index, name in enumerate(sorted(CHECKPOINT_FILES), start=1):
        inventory.append(
            {
                "path": name,
                "kind": "file",
                "identity": {
                    "device": 1,
                    "inode": index,
                    "mode": 0o100600,
                    "uid": 501,
                    "nlink": 1,
                    "size": index,
                    "mtime_ns": index * 10,
                    "ctime_ns": index * 11,
                },
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            }
        )
    inventory_sha = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    adoption_root = tmp_path / "t2-adoption"
    formal_root = tmp_path / "formal-bundle"
    binding = {
        "schema_version": DOWNSTREAM_BINDING_SCHEMA,
        "stage": "T2_GINE_FULL",
        "status": "PASS",
        "state": ADOPTION_MARKER,
        "source_cid": SOURCE_CID,
        "source_run_id": SOURCE_RUN_ID,
        "adoption_root": str(adoption_root),
        "adoption_root_inventory_sha256": "4" * 64,
        "gate_path": str(adoption_root / "gate.json"),
        "gate_sha256": "5" * 64,
        "receipt_path": str(adoption_root / "manifest.json"),
        "receipt_sha256": "6" * 64,
        "source_evidence_sha256": "7" * 64,
        "formal_bundle_root": str(formal_root),
        "formal_bundle_inventory": inventory,
        "formal_bundle_inventory_sha256": inventory_sha,
        "formal_bundle_model_sha256": next(
            row["sha256"] for row in inventory if row["path"] == "model.pt"
        ),
        "formal_bundle_sha256s_sha256": next(
            row["sha256"]
            for row in inventory
            if row["path"] == "sha256sums.txt"
        ),
    }
    binding_sha = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    def stage(stage_name: str, gate: str) -> dict[str, object]:
        return {
            "stage": stage_name,
            "gate_sha256": gate * 64,
            "root_inventory_sha256": "9" * 64,
            "checkpoint_dir": str(formal_root),
            "checkpoint_id": binding["formal_bundle_model_sha256"],
            "checkpoint_inventory_sha256": inventory_sha,
            "checkpoint_stat_inventory_sha256": "a" * 64,
            "checkpoint_sha256s_sha256": binding[
                "formal_bundle_sha256s_sha256"
            ],
            "t2_adoption_gate_sha256": binding["gate_sha256"],
            "t2_adoption_receipt_sha256": binding["receipt_sha256"],
            "t2_adoption_binding_sha256": binding_sha,
        }

    return {
        "schema_version": "tastemolnet_t9_input_authority_v1",
        "managed_active_receipt_sha256": "b" * 64,
        "t2_adoption_binding": binding,
        "t2_adoption_binding_sha256": binding_sha,
        "t3_stage_evidence": stage("T3_GINE_CALIBRATED", "c"),
        "t4_stage_evidence": stage("T4_ORACLE_SMOKE", "d"),
        "train_csv_sha256": "e" * 64,
        "feature_schema_sha256": next(
            row["sha256"]
            for row in inventory
            if row["path"] == "feature_schema.json"
        ),
        "temperature_scaling_sha256": next(
            row["sha256"]
            for row in inventory
            if row["path"] == "temperature_scaling.json"
        ),
        "official_commit": OFFICIAL_COMRECGC_COMMIT,
        "official_file_sha256": dict(OFFICIAL_SOURCE_SHA256),
    }


def _publish_terminal_fixture(root: Path, documents: dict[str, bytes]) -> None:
    from src.utils.retained_output_directory import (
        FreshOutputDirectory,
        prepare_terminal_output,
    )

    output = FreshOutputDirectory.create(root)
    prepared = None
    try:
        for name in sorted(documents):
            output.write_new(name, documents[name]).close()
        prepared = prepare_terminal_output(
            output,
            marker_name=PASS_FILE,
            marker_payload=PASS_BYTES,
        )
        os.rename(
            prepared.marker.name,
            PASS_FILE,
            src_dir_fd=output.descriptor,
            dst_dir_fd=output.descriptor,
        )
        output.committed = True
    finally:
        if prepared is not None:
            prepared.close()
        else:
            output.close()


def _valid_result() -> dict[str, object]:
    parameters = TasteComRecGCSmokeParameters()
    identity = "a" * 64
    result: dict[str, object] = {
        "schema_version": "tastemolnet_comrecgc_native_smoke_v1",
        "stage": "T9_COMRECGC_SMOKE",
        "dataset": "tastemolnet",
        "method": "ComRecGC",
        "parameters": asdict(parameters),
        "source_cohort": {
            "schema_version": "tastemolnet_comrecgc_source_cohort_v1",
            "source_split": "train",
            "source_label": 1,
            "source_pool_count": 64,
            "source_count": 8,
            "source_cohort_sha256": "b" * 64,
            "source_graph_identities_unique": True,
            "validation_loaded": False,
            "calibration_loaded": False,
            "test_loaded": False,
            "molecule_identifiers_persisted": False,
        },
        "checkpoint_reload": {
            "schema_version": "tastemolnet_comrecgc_checkpoint_reload_v1",
            "checkpoint_step": 250,
            "next_step": 251,
            "total_steps": 500,
            "checkpoint_sha256": "c" * 64,
            "checkpoint_bytes": 4096,
            "loop_state_sha256": "0" * 64,
            "official_state_sha256": "1" * 64,
            "transition_state_sha256": "2" * 64,
            "bridge_state_sha256": "d" * 64,
            "rng_state_sha256": "3" * 64,
            "complete_state_sha256": "",
            "checkpoint_reloaded": True,
            "checkpoint_persisted_in_output": False,
        },
        "bridge": {
            "schema_version": "tastemolnet_comrecgc_multiclass_bridge_v1",
            "num_classes": 3,
            "source_label": 1,
            "candidate_condition": "predicted_label != source_label",
            "importance": "1.0 - probabilities[:, source_label]",
            "graph_identity": "canonical_parent_free_attributed_graph_sha256",
            "embedding_identity_used": False,
            "python_builtin_hash_used": False,
            "parent_metadata_in_graph_identity": False,
            "distance_embedding": "frozen_gine_graph_hidden",
            "canonical_row_policy": "first_allclose_frozen_gine_row_reused",
            "canonical_reuse_rtol": 1e-5,
            "canonical_reuse_atol": 1e-7,
            "canonical_row_reuse_count": 0,
            "canonical_row_cache_checkpointed": True,
            "call_count": 4,
            "evaluated_graph_count": 9,
            "calculate_hash_count": 9,
            "unique_graph_count": 9,
            "evaluated_strict_graph_count": 1,
            "destination_prediction_counts": {"0": 1, "2": 0},
            "unique_lineage_count": 9,
            "lineage_occurrence_count": 9,
        },
        "common_recourse": {
            "schema_version": "tastemolnet_comrecgc_common_recourse_smoke_v1",
            "distance_embedding": "frozen_gine_graph_hidden",
            "theta": 0.1,
            "delta": 0.02,
            "cluster_size": 3,
            "recourse_size": 5,
            "retained_strict_candidate_count": 1,
            "theta_eligible_pair_count": 8,
            "dbscan_invoked": True,
            "dbscan_cluster_count": 1,
            "dbscan_noise_count": 0,
            "official_coverage_summary_invoked": True,
            "official_coverage_summary_sha256": "e" * 64,
            "official_greedy_summary_invoked": True,
            "selected_common_recourse_count": 1,
            "selected_common_recourses": [
                {
                    "rank": 1,
                    "cluster_id": 0,
                    "representative_graph_identity_sha256": identity,
                    "destination_label": 0,
                    "score": 0.8,
                    "frequency": 4,
                    "covered_parent_count": 8,
                    "cluster_size": 8,
                    "lineage_count": 1,
                }
            ],
            "graph_payload_persisted": False,
            "molecule_payload_persisted": False,
        },
        "official_native_random_walk": True,
        "official_stateful_heads_preserved": True,
        "official_rng_and_collector_serial": True,
        "random_walk_steps": 500,
        "smoke_budget": True,
        "full_budget": False,
        "full_required_steps": 50_000,
        "same_frozen_three_class_gine": True,
        "second_classifier_used": False,
        "rf_oracle_used": False,
        "validation_loaded": False,
        "calibration_payload_loaded": False,
        "test_loaded": False,
        "dataset_redistributed": False,
        "paper_result_eligible": False,
    }
    component_hashes = {
        key: result["checkpoint_reload"][key]
        for key in (
            "loop_state_sha256",
            "official_state_sha256",
            "transition_state_sha256",
            "bridge_state_sha256",
            "rng_state_sha256",
        )
    }
    result["checkpoint_reload"]["complete_state_sha256"] = hashlib.sha256(
        json.dumps(
            component_hashes,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def _graph(
    *,
    permutation: tuple[int, ...] = (0, 1, 2),
    source_index: int = 0,
    atomic: tuple[int, int, int] = (6, 8, 1),
) -> SimpleNamespace:
    # Native graph before permutation: C-O-H with untyped symmetric edges.
    edges = ((0, 1), (1, 0), (1, 2), (2, 1))
    inverse = {old: new for new, old in enumerate(permutation)}
    x = []
    origins = []
    for old in permutation:
        row = [0.0] * len(ATOMS)
        row[ATOMS.index(atomic[old])] = 1.0
        x.append(row)
        origins.append(old)
    remapped = [(inverse[source], inverse[target]) for source, target in edges]
    return SimpleNamespace(
        x=x,
        edge_index=[
            [source for source, _ in remapped],
            [target for _, target in remapped],
        ],
        num_nodes=3,
        comrecgc_node_origin=origins,
        comrecgc_source_index=source_index,
        # These fields must never participate in structural identity.
        comrecgc_parent_id=f"parent-{source_index}",
        comrecgc_source_smiles="forbidden-parent-metadata",
    )


def _empty_graph(*, source_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        x=[],
        edge_index=[[], []],
        num_nodes=0,
        comrecgc_node_origin=[],
        comrecgc_source_index=source_index,
        comrecgc_parent_id=f"parent-{source_index}",
        comrecgc_source_smiles="forbidden-parent-metadata",
    )


class _Adapter:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, graphs: list[SimpleNamespace]) -> SimpleNamespace:
        self.calls += 1
        probabilities = np.asarray(
            [[0.2, 0.7, 0.1] for _graph_value in graphs],
            dtype=np.float64,
        )
        embeddings = np.asarray(
            [[1.0, 2.0, 3.0] for _graph_value in graphs],
            dtype=np.float32,
        )
        return SimpleNamespace(
            probabilities=probabilities,
            graph_embeddings=embeddings,
            valid_fullgraphs=tuple(True for _ in graphs),
        )


class _IndexVaryingAdapter(_Adapter):
    def score(self, graphs: list[SimpleNamespace]) -> SimpleNamespace:
        return SimpleNamespace(
            probabilities=np.asarray(
                [[0.2, 0.7, 0.1] if index == 0 else [0.6, 0.3, 0.1]
                 for index, _graph_value in enumerate(graphs)],
                dtype=np.float64,
            ),
            graph_embeddings=np.asarray(
                [[float(index + 1), 2.0, 3.0]
                 for index, _graph_value in enumerate(graphs)],
                dtype=np.float32,
            ),
            valid_fullgraphs=tuple(True for _ in graphs),
        )


class _CandidateAdapter(_Adapter):
    def score(self, graphs: list[object]) -> SimpleNamespace:
        return SimpleNamespace(
            probabilities=np.asarray(
                [[0.7, 0.2, 0.1] for _graph_value in graphs],
                dtype=np.float64,
            ),
            predictions=tuple(0 for _graph_value in graphs),
            graph_embeddings=np.asarray(
                [[1.0, 2.0, 3.0] for _graph_value in graphs],
                dtype=np.float32,
            ),
            valid_fullgraphs=tuple(True for _graph_value in graphs),
        )


class _CheckpointNativeGraph:
    """Small picklable native graph used by the real checkpoint path."""

    def __init__(self, source_index: int) -> None:
        import torch

        self.x = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.edge_index = torch.tensor(
            [[0, 1], [1, 0]], dtype=torch.long
        )
        self.num_nodes = 2
        self.comrecgc_node_origin = torch.tensor([0, 1], dtype=torch.long)
        self.comrecgc_source_index = source_index

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "edge_index": self.edge_index,
            "num_nodes": self.num_nodes,
            "comrecgc_node_origin": self.comrecgc_node_origin,
            "comrecgc_source_index": self.comrecgc_source_index,
        }


def test_smoke_budget_is_exactly_500() -> None:
    assert SMOKE_STEPS == 500
    assert TasteComRecGCSmokeParameters().validate().steps == 500


def test_smoke_parameters_reject_a_smaller_or_easier_self_declared_route() -> None:
    hostile = TasteComRecGCSmokeParameters(
        steps=500,
        checkpoint_step=250,
        source_pool=1,
        source_count=1,
        heads=1,
        candidate_capacity=1,
        sample_size=1,
        teleport_probability=1.0,
        theta=999.0,
        delta=999.0,
        cluster_size=1,
        recourse_size=1,
        seed=999,
    )
    with pytest.raises(
        TasteComRecGCSmokeError,
        match="bounded-smoke parameters changed",
    ):
        hostile.validate()


def test_checkpoint_state_digest_is_typed_and_tensor_byte_exact() -> None:
    torch = pytest.importorskip("torch")
    original = {
        "heads": ("a" * 64, "b" * 64),
        "collector": [{"frequency": 2, "candidate": True}],
        "tensor": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    same = deepcopy(original)
    same["tensor"] = original["tensor"].clone()
    digest = _checkpoint_state_sha256(original, field="fixture")
    assert _checkpoint_state_sha256(same, field="fixture") == digest

    integer_drift = deepcopy(original)
    integer_drift["collector"][0]["candidate"] = 1
    assert _checkpoint_state_sha256(
        integer_drift, field="fixture"
    ) != digest

    tensor_drift = deepcopy(original)
    tensor_drift["tensor"][0, 1] = 3.0
    assert _checkpoint_state_sha256(tensor_drift, field="fixture") != digest

    nonfinite = deepcopy(original)
    nonfinite["tensor"][0, 0] = float("nan")
    with pytest.raises(TasteComRecGCSmokeError, match="non-finite"):
        _checkpoint_state_sha256(nonfinite, field="fixture")


def test_midpoint_checkpoint_reopens_all_serial_state_and_rejects_tamper(
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    from src.baselines.comrecgc.generation_loop import GenerationLoopState

    module = SimpleNamespace(
        graph_map={},
        graph_index_map={},
        counterfactual_candidates=[],
        input_graphs_covered=torch.zeros(5, dtype=torch.float32),
        covering_graphs=set(),
        transitions={"a" * 64: ("b" * 64, 0.25)},
        start={"a" * 64: 3},
        is_sample=True,
        starting_step=251,
        traversed_hashes=[["a" * 64] for _ in range(250)],
        sample_size=10_000,
        MAX_COUNTERFACTUAL_SIZE=2048,
    )
    state = GenerationLoopState(
        completed_step=250,
        start_graph_hashes=("a" * 64,) * 5,
        current_graph_hashes=("a" * 64,) * 5,
        restart_indices=(0, 1, 2, 3, 4),
    )
    bridge = TasteComRecGCMulticlassBridge(
        adapter=_Adapter(), feature_atomic_numbers=ATOMS
    )
    loaded = _write_reload_checkpoint(
        module=module,
        bridge=bridge,
        loop_state=state,
        parameters=TasteComRecGCSmokeParameters(),
        path=tmp_path / "checkpoint.pt",
    )
    restored = _restore_reload_checkpoint(
        module=module, bridge=bridge, loaded=loaded
    )
    assert restored.to_checkpoint_state() == state.to_checkpoint_state()
    assert loaded["evidence"]["complete_state_sha256"] == hashlib.sha256(
        json.dumps(
            {
                key: loaded["evidence"][key]
                for key in (
                    "loop_state_sha256",
                    "official_state_sha256",
                    "transition_state_sha256",
                    "bridge_state_sha256",
                    "rng_state_sha256",
                )
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    hostile = deepcopy(loaded)
    hostile["payload"]["official_state"]["starting_step"] = 252
    with pytest.raises(TasteComRecGCSmokeError, match="differs from serialized"):
        _restore_reload_checkpoint(
            module=module, bridge=bridge, loaded=hostile
        )


def test_midpoint_checkpoint_load_uses_held_inode_and_rejects_named_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from src.baselines.comrecgc.generation_loop import GenerationLoopState

    module = SimpleNamespace(
        graph_map={},
        graph_index_map={},
        counterfactual_candidates=[],
        input_graphs_covered=torch.zeros(1, dtype=torch.float32),
        covering_graphs=set(),
        transitions={},
        start={},
        is_sample=True,
        starting_step=251,
        traversed_hashes=[[] for _ in range(250)],
        sample_size=10_000,
        MAX_COUNTERFACTUAL_SIZE=2048,
    )
    state = GenerationLoopState(
        completed_step=250,
        start_graph_hashes=("a" * 64,),
        current_graph_hashes=("a" * 64,),
        restart_indices=(0,),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    original_load = smoke_module._torch_load_handle

    def swap_named_checkpoint(handle):
        raw = checkpoint.read_bytes()
        replacement = tmp_path / "checkpoint.replacement"
        replacement.write_bytes(raw)
        replacement.chmod(0o600)
        os.replace(replacement, checkpoint)
        return original_load(handle)

    monkeypatch.setattr(
        smoke_module, "_torch_load_handle", swap_named_checkpoint
    )
    with pytest.raises(
        TasteComRecGCSmokeError, match="reload closure changed"
    ):
        _write_reload_checkpoint(
            module=module,
            bridge=TasteComRecGCMulticlassBridge(
                adapter=_Adapter(), feature_atomic_numbers=ATOMS
            ),
            loop_state=state,
            parameters=TasteComRecGCSmokeParameters(),
            path=checkpoint,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"steps": 50_000},
        {"checkpoint_step": 499},
        {"heads": True},
        {"teleport_probability": 1},
        {"theta": float("nan")},
    ),
)
def test_smoke_parameters_reject_full_or_coerced_authority(
    changes: dict[str, object],
) -> None:
    values = {
        field: getattr(TasteComRecGCSmokeParameters(), field)
        for field in TasteComRecGCSmokeParameters.__dataclass_fields__
    }
    values.update(changes)
    with pytest.raises(TasteComRecGCSmokeError):
        TasteComRecGCSmokeParameters(**values).validate()


def test_score_is_not_the_multiclass_candidate_predicate() -> None:
    score, prediction, candidate = score_and_candidate([0.35, 0.40, 0.25])
    assert score == pytest.approx(0.60)
    assert prediction == 1
    assert candidate is False

    score, prediction, candidate = score_and_candidate([0.60, 0.30, 0.10])
    assert score == pytest.approx(0.70)
    assert prediction == 0
    assert candidate is True


@pytest.mark.parametrize(
    "probabilities",
    (
        [True, 0.0, 0.0],
        [0.2, 0.3],
        [0.2, float("nan"), 0.8],
        [0.2, 0.2, 0.2],
    ),
)
def test_score_rejects_untyped_or_nonprobability_rows(probabilities: list[object]) -> None:
    with pytest.raises(TasteComRecGCSmokeError):
        score_and_candidate(probabilities)


def test_canonical_identity_is_permutation_and_parent_invariant() -> None:
    first = canonical_attributed_graph(_graph(), feature_atomic_numbers=ATOMS)
    permuted = canonical_attributed_graph(
        _graph(permutation=(2, 0, 1), source_index=7),
        feature_atomic_numbers=ATOMS,
    )
    assert first.graph_identity_sha256 == permuted.graph_identity_sha256
    assert first.canonical_graph == permuted.canonical_graph
    assert first.collision_payload() == permuted.collision_payload()


def test_empty_native_walk_state_has_parent_free_structural_identity() -> None:
    first = canonical_attributed_graph(
        _empty_graph(source_index=0), feature_atomic_numbers=ATOMS
    )
    second = canonical_attributed_graph(
        _empty_graph(source_index=19), feature_atomic_numbers=ATOMS
    )
    assert first == second
    assert first.canonical_graph == "<EMPTY_ATTRIBUTED_GRAPH>"
    assert first.num_nodes == 0
    assert first.num_edges == 0


def test_canonical_identity_rejects_asymmetric_or_non_one_hot_graph() -> None:
    asymmetric = _graph()
    asymmetric.edge_index = [[0], [1]]
    with pytest.raises(TasteComRecGCSmokeError, match="exactly symmetric"):
        canonical_attributed_graph(asymmetric, feature_atomic_numbers=ATOMS)

    non_one_hot = _graph()
    non_one_hot.x[0] = [0.5, 0.5, 0.0]
    with pytest.raises(TasteComRecGCSmokeError, match="exact one-hot"):
        canonical_attributed_graph(non_one_hot, feature_atomic_numbers=ATOMS)


def test_source_cohort_replay_uses_the_same_decoded_identity_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.baselines import tastemolnet_gcf_smoke as gcf_smoke

    class _CloneableOrigins(list):
        def clone(self):
            return _CloneableOrigins(self)

    def fake_encode(row, _schema):
        return dict(row)

    def fake_to_pyg(record, *, origin_index):
        graph = _graph(source_index=origin_index)
        graph.gcf_node_origin = _CloneableOrigins([0, 1, 2])
        graph.record_key = record["record_key"]
        return graph

    class _DecodedIdentityAdapter:
        def __init__(
            self,
            _checkpoint_payloads,
            *,
            source_records,
            graph_schema,
            device,
        ) -> None:
            assert graph_schema.feature_atomic_numbers == ATOMS
            assert device == "cpu"
            self.source_records = tuple(source_records)

        @staticmethod
        def score(graphs):
            return SimpleNamespace(
                valid_fullgraphs=tuple(True for _ in graphs),
                predictions=tuple(1 for _ in graphs),
                # The decoded GINE graph can intentionally differ from the
                # expanded, untyped native graph (for example implicit versus
                # explicit hydrogens).  This payload is the identity authority
                # used by the COMRECGC bridge and must also be used on replay.
                identity_graph_payloads=tuple(
                    {
                        "canonical_graph": f"decoded-{graph.record_key}",
                        "num_nodes": 2,
                        "num_edges": 1,
                    }
                    for graph in graphs
                ),
            )

    monkeypatch.setattr(gcf_smoke, "encode_taste_source_graph", fake_encode)
    monkeypatch.setattr(gcf_smoke, "taste_record_to_pyg", fake_to_pyg)
    monkeypatch.setattr(
        gcf_smoke,
        "TasteFrozenGINENativeAdapter",
        _DecodedIdentityAdapter,
    )
    source_rows = tuple(
        {"record_key": f"source-{index:03d}"} for index in range(64)
    )
    graphs, records, _adapter, evidence = smoke_module._initialize_source_graphs(
        checkpoint_payloads={"model.pt": b"held"},
        source_rows=source_rows,
        graph_schema=SimpleNamespace(feature_atomic_numbers=ATOMS),
        device="cpu",
        parameters=TasteComRecGCSmokeParameters(),
    )
    assert [record["record_key"] for record in records] == [
        f"source-{index:03d}" for index in range(8)
    ]
    assert [graph.record_key for graph in graphs] == [
        f"source-{index:03d}" for index in range(8)
    ]
    assert evidence["source_count"] == 8
    assert evidence["source_graph_identities_unique"] is True


def test_bridge_uses_structure_for_identity_and_gine_hidden_only_for_alignment() -> None:
    adapter = _Adapter()
    bridge = TasteComRecGCMulticlassBridge(
        adapter=adapter,
        feature_atomic_numbers=ATOMS,
    )
    graphs = [_graph(source_index=0), _graph(source_index=1)]
    importance, embeddings = bridge.call(graphs, {})
    assert np.allclose(importance, np.asarray([[0.3, 1.0], [0.3, 1.0]]))
    first_hash = bridge.calculate_hash(embeddings[0])
    second_hash = bridge.calculate_hash(embeddings[1])
    # Same attributed graph across parents intentionally shares identity.
    assert first_hash == second_hash
    assert bridge.is_graph_counterfactual(first_hash) is False
    report = bridge.report()
    assert report["embedding_identity_used"] is False
    assert report["python_builtin_hash_used"] is False
    assert report["parent_metadata_in_graph_identity"] is False
    assert report["unique_graph_count"] == 1
    assert report["unique_lineage_count"] == 2


def _model_graph_payload(channel: int) -> dict[str, object]:
    return {
        "schema_version": "tastemolnet_gine_model_graph_v1",
        "canonical_smiles": "C" if channel == 0 else "C=O",
        "graph_sha256": hashlib.sha256(str(channel).encode()).hexdigest(),
        "feature_schema_sha256": "f" * 64,
        "node_features": {
            "dtype": "int64",
            "shape": [1, 2],
            "values": [[channel, 1]],
        },
        "edge_index": {
            "dtype": "int64",
            "shape": [2, 0],
            "values": [[], []],
        },
        "edge_attr": {
            "dtype": "int64",
            "shape": [0, 4],
            "values": [],
        },
    }


def test_bridge_separates_canonical_identity_from_lossless_gine_model_graph() -> None:
    class _SeparatedAdapter:
        @staticmethod
        def score(graphs):
            channels = [int(graph.comrecgc_source_index) for graph in graphs]
            return SimpleNamespace(
                probabilities=np.asarray(
                    [[0.2, 0.7, 0.1] for _ in graphs], dtype=np.float64
                ),
                predictions=tuple(1 for _ in graphs),
                graph_embeddings=np.asarray(
                    [[float(channel), 2.0] for channel in channels],
                    dtype=np.float32,
                ),
                valid_fullgraphs=tuple(True for _ in graphs),
                identity_graph_payloads=tuple(
                    {
                        "canonical_graph": "C" if channel == 0 else "C=O",
                        "num_nodes": 1 if channel == 0 else 2,
                        "num_edges": 0 if channel == 0 else 1,
                    }
                    for channel in channels
                ),
                model_graph_payloads=tuple(
                    _model_graph_payload(channel) for channel in channels
                ),
            )

    bridge = TasteComRecGCMulticlassBridge(
        adapter=_SeparatedAdapter(), feature_atomic_numbers=ATOMS
    )
    _importance, embeddings = bridge.call(
        [_graph(source_index=0), _graph(source_index=1)], {}
    )
    identities = [bridge.calculate_hash(row) for row in embeddings]
    assert identities[0] != identities[1]
    assert bridge.records[identities[0]].model_graph_sha256 != (
        bridge.records[identities[1]].model_graph_sha256
    )
    assert bridge.records[identities[1]].model_graph_payload[
        "node_features"
    ]["values"] == [[1, 1]]

    state = bridge.checkpoint_state()
    assert state["schema_version"] == "tastemolnet_comrecgc_bridge_checkpoint_v3"
    restored = TasteComRecGCMulticlassBridge(
        adapter=_SeparatedAdapter(), feature_atomic_numbers=ATOMS
    )
    restored.restore_checkpoint_state(state)
    assert restored.checkpoint_state() == state


def test_bridge_does_not_relax_semantics_for_same_identity_model_graph_drift() -> None:
    class _ModelGraphDriftAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, graphs):
            self.calls += 1
            return SimpleNamespace(
                probabilities=np.asarray([[0.2, 0.7, 0.1]], dtype=np.float64),
                predictions=(1,),
                graph_embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
                valid_fullgraphs=(True,),
                identity_graph_payloads=(
                    {"canonical_graph": "C", "num_nodes": 1, "num_edges": 0},
                ),
                model_graph_payloads=(_model_graph_payload(self.calls - 1),),
            )

    bridge = TasteComRecGCMulticlassBridge(
        adapter=_ModelGraphDriftAdapter(), feature_atomic_numbers=ATOMS
    )
    _importance, embeddings = bridge.call([_graph()], {})
    bridge.calculate_hash(embeddings[0])
    with pytest.raises(TasteComRecGCSmokeError, match="changed GINE semantics"):
        bridge.call([_graph()], {})


def test_bridge_reuses_invalid_unscored_identity_across_node_permutations() -> None:
    class _InvalidUnscoredAdapter:
        @staticmethod
        def score(graphs):
            return SimpleNamespace(
                probabilities=np.asarray(
                    [[0.0, 1.0, 0.0] for _ in graphs], dtype=np.float64
                ),
                graph_embeddings=np.zeros((len(graphs), 3), dtype=np.float32),
                valid_fullgraphs=tuple(False for _ in graphs),
                identity_graph_payloads=tuple(None for _ in graphs),
                model_graph_payloads=tuple(None for _ in graphs),
            )

    bridge = TasteComRecGCMulticlassBridge(
        adapter=_InvalidUnscoredAdapter(),
        feature_atomic_numbers=ATOMS,
    )
    importance, embeddings = bridge.call(
        [_graph(), _graph(permutation=(2, 0, 1), source_index=1)],
        {},
    )
    identities = [bridge.calculate_hash(row) for row in embeddings]

    assert identities[0] == identities[1]
    assert np.array_equal(
        importance,
        np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
    )
    assert bridge.is_graph_counterfactual(identities[0]) is False
    record = bridge.records[identities[0]]
    assert record.probabilities == (0.0, 1.0, 0.0)
    assert record.valid_fullgraph is False
    assert record.model_graph_payload == {
        "schema_version": "tastemolnet_gine_invalid_unscored_graph_v1",
        "frozen_gine_scored": False,
        "valid_fullgraph": False,
    }
    state = bridge.checkpoint_state()
    restored = TasteComRecGCMulticlassBridge(
        adapter=_InvalidUnscoredAdapter(),
        feature_atomic_numbers=ATOMS,
    )
    restored.restore_checkpoint_state(state)
    assert restored.checkpoint_state() == state


def test_bridge_rejects_missing_explicit_model_evidence_for_valid_graph() -> None:
    class _MissingValidModelEvidenceAdapter:
        @staticmethod
        def score(graphs):
            return SimpleNamespace(
                probabilities=np.asarray(
                    [[0.2, 0.7, 0.1] for _ in graphs], dtype=np.float64
                ),
                graph_embeddings=np.asarray(
                    [[1.0, 2.0, 3.0] for _ in graphs], dtype=np.float32
                ),
                valid_fullgraphs=tuple(True for _ in graphs),
                identity_graph_payloads=tuple(
                    {
                        "canonical_graph": "CO",
                        "num_nodes": 3,
                        "num_edges": 2,
                    }
                    for _ in graphs
                ),
                model_graph_payloads=tuple(None for _ in graphs),
            )

    bridge = TasteComRecGCMulticlassBridge(
        adapter=_MissingValidModelEvidenceAdapter(),
        feature_atomic_numbers=ATOMS,
    )
    with pytest.raises(TasteComRecGCSmokeError, match="lacks GINE model evidence"):
        bridge.call([_graph()], {})


def test_bridge_reuses_first_canonical_row_across_cuda_low_bit_rescores() -> None:
    class _LowBitAdapter:
        def __init__(self) -> None:
            self.calls = 0
            self.last_raw_embedding = None

        def score(self, graphs):
            self.calls += 1
            if self.calls == 1:
                probabilities = [0.2, 0.7, 0.1]
                embedding = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
            else:
                probabilities = [0.2000001, 0.6999999, 0.1]
                embedding = np.nextafter(
                    np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
                )
            self.last_raw_embedding = embedding.copy()
            return SimpleNamespace(
                probabilities=np.asarray(
                    [probabilities for _ in graphs], dtype=np.float64
                ),
                predictions=tuple(1 for _ in graphs),
                graph_embeddings=np.stack(
                    [embedding.copy() for _ in graphs]
                ),
                valid_fullgraphs=tuple(True for _ in graphs),
            )

    adapter = _LowBitAdapter()
    bridge = TasteComRecGCMulticlassBridge(
        adapter=adapter,
        feature_atomic_numbers=ATOMS,
    )
    first_importance, first_embeddings = bridge.call([_graph()], {})
    graph_identity = bridge.calculate_hash(first_embeddings[0])
    second_importance, second_embeddings = bridge.call(
        [_graph(source_index=1)], {}
    )
    assert not np.array_equal(adapter.last_raw_embedding, first_embeddings[0])
    assert np.array_equal(second_embeddings, first_embeddings)
    assert np.array_equal(second_importance, first_importance)
    assert bridge.calculate_hash(second_embeddings[0]) == graph_identity
    assert bridge.report()["unique_graph_count"] == 1
    assert bridge.report()["evaluated_graph_count"] == 2

    state = bridge.checkpoint_state()
    restored = TasteComRecGCMulticlassBridge(
        adapter=_LowBitAdapter(), feature_atomic_numbers=ATOMS
    )
    restored.restore_checkpoint_state(state)
    assert restored.checkpoint_state() == state


@pytest.mark.parametrize("drift_kind", ("probabilities", "embedding"))
def test_bridge_rejects_gross_same_prediction_rescore_drift(
    drift_kind: str,
) -> None:
    class _GrossDriftAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, graphs):
            self.calls += 1
            probabilities = (
                [0.2, 0.7, 0.1]
                if self.calls == 1 or drift_kind == "embedding"
                else [0.49, 0.50, 0.01]
            )
            embedding = (
                [1.0, 2.0, 3.0]
                if self.calls == 1 or drift_kind == "probabilities"
                else [1.0e9, -1.0e9, 3.0]
            )
            return SimpleNamespace(
                probabilities=np.asarray(
                    [probabilities for _ in graphs], dtype=np.float64
                ),
                predictions=tuple(1 for _ in graphs),
                graph_embeddings=np.asarray(
                    [embedding for _ in graphs], dtype=np.float32
                ),
                valid_fullgraphs=tuple(True for _ in graphs),
            )

    bridge = TasteComRecGCMulticlassBridge(
        adapter=_GrossDriftAdapter(),
        feature_atomic_numbers=ATOMS,
    )
    _importance, embeddings = bridge.call([_graph()], {})
    bridge.calculate_hash(embeddings[0])
    with pytest.raises(TasteComRecGCSmokeError, match="changed GINE semantics"):
        bridge.call([_graph(source_index=1)], {})


def test_bridge_fails_if_parent_metadata_changes_same_graph_semantics() -> None:
    bridge = TasteComRecGCMulticlassBridge(
        adapter=_IndexVaryingAdapter(),
        feature_atomic_numbers=ATOMS,
    )
    _importance, embeddings = bridge.call([_graph()], {})
    bridge.calculate_hash(embeddings[0])
    # The second row in one two-row call would receive different semantics for
    # the same graph identity, which must never be silently parent-keyed.
    with pytest.raises(TasteComRecGCSmokeError, match="changed GINE semantics"):
        bridge.call([_graph(source_index=1), _graph(source_index=2)], {})


def test_bridge_rejects_embedding_call_order_drift() -> None:
    bridge = TasteComRecGCMulticlassBridge(
        adapter=_IndexVaryingAdapter(),
        feature_atomic_numbers=ATOMS,
    )
    _importance, embeddings = bridge.call(
        [_graph(), _graph(source_index=1, atomic=(8, 8, 1))], {}
    )
    with pytest.raises(TasteComRecGCSmokeError, match="call order drifted"):
        bridge.calculate_hash(embeddings[1])


def test_bridge_checkpoint_round_trip_and_hostile_bool_rejection() -> None:
    bridge = TasteComRecGCMulticlassBridge(
        adapter=_Adapter(),
        feature_atomic_numbers=ATOMS,
    )
    _importance, embeddings = bridge.call([_graph()], {})
    bridge.calculate_hash(embeddings[0])
    state = bridge.checkpoint_state()

    restored = TasteComRecGCMulticlassBridge(
        adapter=_Adapter(),
        feature_atomic_numbers=ATOMS,
    )
    restored.restore_checkpoint_state(state)
    assert restored.checkpoint_state() == state

    hostile = dict(state)
    hostile["call_count"] = False
    with pytest.raises(TasteComRecGCSmokeError, match="native integer"):
        TasteComRecGCMulticlassBridge(
            adapter=_Adapter(), feature_atomic_numbers=ATOMS
        ).restore_checkpoint_state(hostile)


def test_bridge_checkpoint_round_trip_preserves_empty_native_state() -> None:
    bridge = TasteComRecGCMulticlassBridge(
        adapter=_Adapter(), feature_atomic_numbers=ATOMS
    )
    _importance, embeddings = bridge.call([_empty_graph()], {})
    graph_identity = bridge.calculate_hash(embeddings[0])
    assert bridge.records[graph_identity].candidate is False
    state = bridge.checkpoint_state()

    restored = TasteComRecGCMulticlassBridge(
        adapter=_Adapter(), feature_atomic_numbers=ATOMS
    )
    restored.restore_checkpoint_state(state)
    assert restored.checkpoint_state() == state


def test_bridge_install_restores_every_official_function() -> None:
    module = SimpleNamespace(
        call=lambda *_args: "old-call",
        calculate_hash=lambda *_args: "old-hash",
        is_graph_counterfactual=lambda *_args: False,
        neighbor_graph_access=lambda graph, _action: graph,
    )
    original = (
        module.call,
        module.calculate_hash,
        module.is_graph_counterfactual,
        module.neighbor_graph_access,
    )
    bridge = TasteComRecGCMulticlassBridge(
        adapter=_Adapter(), feature_atomic_numbers=ATOMS
    )

    def wrapper(function: object) -> object:
        assert function is original[3]
        return lambda graph, action: function(graph, action)

    with bridge.installed(module, neighbor_wrapper=wrapper):
        assert module.call == bridge.call
        assert module.calculate_hash == bridge.calculate_hash
        assert module.is_graph_counterfactual == bridge.is_graph_counterfactual
    assert (
        module.call,
        module.calculate_hash,
        module.is_graph_counterfactual,
        module.neighbor_graph_access,
    ) == original


def test_common_recourse_uses_native_dbscan_coverage_and_greedy() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")

    candidate = _graph(source_index=7)
    sources = [_graph(source_index=index) for index in range(3)]

    class _DistanceAdapter:
        def score(self, graphs):
            probabilities = []
            embeddings = []
            for graph in graphs:
                if graph is candidate:
                    probabilities.append([0.7, 0.2, 0.1])
                    embeddings.append([0.0, 0.0])
                else:
                    probabilities.append([0.1, 0.8, 0.1])
                    embeddings.append([0.005 + graph.comrecgc_source_index * 0.001, 0.0])
            return SimpleNamespace(
                probabilities=np.asarray(probabilities, dtype=np.float64),
                predictions=tuple(
                    int(np.argmax(row)) for row in probabilities
                ),
                graph_embeddings=np.asarray(embeddings, dtype=np.float64),
                valid_fullgraphs=tuple(True for _ in graphs),
            )

    adapter = _DistanceAdapter()
    bridge = TasteComRecGCMulticlassBridge(
        adapter=adapter,
        feature_atomic_numbers=ATOMS,
    )
    _importance, candidate_embeddings = bridge.call([candidate], {})
    candidate_hash = bridge.calculate_hash(candidate_embeddings[0])
    assert bridge.records[candidate_hash].candidate is True

    calls = {"coverage": 0, "greedy": 0}

    def coverage_summary(**kwargs):
        calls["coverage"] += 1
        assert kwargs["recourse_size"] == 5
        assert kwargs["idxs"] == [(0, 0), (1, 0), (2, 0)]
        return [(0, [0, 1, 2])]

    def greedy_counterfactual_summary_from_covering_sets(
        *, counterfactual_covering, graphs_covered_by, k
    ):
        calls["greedy"] += 1
        assert k == 1
        assert counterfactual_covering == {0: {0, 1, 2}}
        assert graphs_covered_by == {0: {0}, 1: {0}, 2: {0}}
        return {1: (0, 3)}

    module = SimpleNamespace(
        counterfactual_candidates=[
            {"graph_hash": candidate_hash, "frequency": 4}
        ],
        graph_map={
            candidate_hash: [candidate, candidate_embeddings[0], np.asarray([1.0])]
        },
    )
    modules = {
        "common_recourse": SimpleNamespace(
            coverage_summary=coverage_summary,
            greedy_counterfactual_summary_from_covering_sets=(
                greedy_counterfactual_summary_from_covering_sets
            ),
        ),
        "util": SimpleNamespace(
            graph_element_counts=lambda graphs: torch.ones(
                (len(graphs),), dtype=torch.float64
            )
        ),
    }
    summary = _common_recourse_summary(
        modules=modules,
        module=module,
        bridge=bridge,
        source_graphs=sources,
        adapter=adapter,
        parameters=TasteComRecGCSmokeParameters(),
    )

    assert calls == {"coverage": 1, "greedy": 1}
    assert summary["dbscan_invoked"] is True
    assert summary["official_coverage_summary_invoked"] is True
    assert summary["official_greedy_summary_invoked"] is True
    assert summary["selected_common_recourse_count"] == 1
    assert summary["selected_common_recourses"] == [
        {
            "rank": 1,
            "cluster_id": 0,
            "representative_graph_identity_sha256": candidate_hash,
            "destination_label": 0,
            "score": pytest.approx(0.8),
            "frequency": 4,
            "covered_parent_count": 3,
            "cluster_size": 3,
            "lineage_count": 1,
        }
    ]


def test_common_recourse_preserves_official_source_major_pair_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    sklearn_cluster = pytest.importorskip("sklearn.cluster")

    sources = [_graph(source_index=index) for index in range(2)]
    candidates = [
        _graph(source_index=10, atomic=(6, 8, 1)),
        _graph(source_index=11, atomic=(6, 6, 1)),
        _graph(source_index=12, atomic=(8, 8, 1)),
    ]
    candidate_positions = {id(graph): index for index, graph in enumerate(candidates)}
    source_positions = {id(graph): index for index, graph in enumerate(sources)}

    class _OrderAdapter:
        def score(self, graphs):
            probabilities = []
            embeddings = []
            predictions = []
            for graph in graphs:
                if id(graph) in candidate_positions:
                    index = candidate_positions[id(graph)]
                    probabilities.append([0.7, 0.2, 0.1])
                    embeddings.append([0.001 * index, 0.0])
                    predictions.append(0)
                else:
                    index = source_positions[id(graph)]
                    probabilities.append([0.1, 0.8, 0.1])
                    embeddings.append([0.003 + 0.001 * index, 0.0])
                    predictions.append(1)
            return SimpleNamespace(
                probabilities=np.asarray(probabilities, dtype=np.float64),
                predictions=tuple(predictions),
                graph_embeddings=np.asarray(embeddings, dtype=np.float64),
                valid_fullgraphs=tuple(True for _ in graphs),
            )

    class _OneClusterDBSCAN:
        def __init__(self, *, eps, min_samples):
            assert eps == pytest.approx(0.02)
            assert min_samples == 3

        def fit(self, rows):
            assert len(rows) == 6
            self.labels_ = np.zeros(6, dtype=int)
            return self

    monkeypatch.setattr(sklearn_cluster, "DBSCAN", _OneClusterDBSCAN)

    adapter = _OrderAdapter()
    bridge = TasteComRecGCMulticlassBridge(
        adapter=adapter,
        feature_atomic_numbers=ATOMS,
    )
    _importance, embeddings = bridge.call(candidates, {})
    hashes = [bridge.calculate_hash(row) for row in embeddings]
    expected_pairs = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    observed_pairs: list[tuple[int, int]] = []

    def coverage_summary(**kwargs):
        observed_pairs.extend(kwargs["idxs"])
        return [(0, [0, 1])]

    def greedy_counterfactual_summary_from_covering_sets(
        *, counterfactual_covering, graphs_covered_by, k
    ):
        assert counterfactual_covering == {0: {0, 1}}
        assert graphs_covered_by == {0: {0}, 1: {0}}
        assert k == 1
        return {1: (0, 2)}

    module = SimpleNamespace(
        counterfactual_candidates=[
            {"graph_hash": graph_hash, "frequency": index + 1}
            for index, graph_hash in enumerate(hashes)
        ],
        graph_map={
            graph_hash: [candidates[index], embeddings[index], np.asarray([1.0])]
            for index, graph_hash in enumerate(hashes)
        },
    )
    modules = {
        "common_recourse": SimpleNamespace(
            coverage_summary=coverage_summary,
            greedy_counterfactual_summary_from_covering_sets=(
                greedy_counterfactual_summary_from_covering_sets
            ),
        ),
        "util": SimpleNamespace(
            graph_element_counts=lambda graphs: torch.ones(
                (len(graphs),), dtype=torch.float64
            )
        ),
    }

    summary = _common_recourse_summary(
        modules=modules,
        module=module,
        bridge=bridge,
        source_graphs=sources,
        adapter=adapter,
        parameters=TasteComRecGCSmokeParameters(),
    )

    assert observed_pairs == expected_pairs
    assert summary["theta_eligible_pair_count"] == 6
    assert summary["official_greedy_summary_invoked"] is True


def test_native_execute_runs_exact_m500_with_real_midpoint_state_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from src.baselines.comrecgc import runtime as comrecgc_runtime

    graphs = [_CheckpointNativeGraph(index) for index in range(8)]
    adapter = _CandidateAdapter()
    source_evidence = deepcopy(_valid_result()["source_cohort"])

    def initialize(**_kwargs: object):
        return graphs, tuple({"source_label": 1} for _ in graphs), adapter, source_evidence

    common = deepcopy(_valid_result()["common_recourse"])
    monkeypatch.setattr(smoke_module, "_initialize_source_graphs", initialize)
    monkeypatch.setattr(
        smoke_module,
        "_common_recourse_summary",
        lambda **_kwargs: deepcopy(common),
    )

    def reset_official_state(
        module: object, *, candidate_capacity: int, sample_size: int
    ) -> None:
        module.MAX_COUNTERFACTUAL_SIZE = candidate_capacity
        module.graph_map = {}
        module.graph_index_map = {}
        module.counterfactual_candidates = []
        module.input_graphs_covered = torch.zeros(0, dtype=torch.float32)
        module.covering_graphs = set()
        module.transitions = {}
        module.start = {}
        module.is_sample = True
        module.starting_step = 1
        module.traversed_hashes = []
        module.sample_size = sample_size

    monkeypatch.setattr(
        comrecgc_runtime, "reset_official_state", reset_official_state
    )

    module = SimpleNamespace(
        MAX_COUNTERFACTUAL_SIZE=2048,
        graph_map={},
        graph_index_map={},
        counterfactual_candidates=[],
        input_graphs_covered=torch.zeros(0, dtype=torch.float32),
        covering_graphs=set(),
        transitions={},
        start={},
        is_sample=True,
        starting_step=1,
        traversed_hashes=[],
        sample_size=10_000,
        call=lambda *_args: None,
        calculate_hash=lambda *_args: "unpatched",
        is_graph_counterfactual=lambda *_args: False,
        neighbor_graph_access=lambda graph, _action: graph,
    )

    def restart_randomwalk(input_graphs, heads, importance_args):
        values = list(input_graphs)
        _importance, embeddings = module.call(values, importance_args)
        graph_hashes = [
            module.calculate_hash(embeddings[index])
            for index in range(len(values))
        ]
        first_hash = graph_hashes[0]
        module.graph_map = {
            first_hash: [values[0], embeddings[0], np.asarray([2.0])]
        }
        module.graph_index_map = {first_hash: 0}
        module.counterfactual_candidates = [
            {"graph_hash": first_hash, "frequency": len(values)}
        ]
        module.input_graphs_covered = torch.zeros(
            len(values), dtype=torch.float32
        )
        return graph_hashes[:heads], torch.arange(heads, dtype=torch.long)

    def move_to_next_graph(**kwargs):
        current = list(kwargs["graphs_hash"])
        return current, False, None, None, None

    module.restart_randomwalk = restart_randomwalk
    module.move_to_next_graph = move_to_next_graph
    modules = {
        "comrecgc": module,
        "common_recourse": SimpleNamespace(),
        "util": SimpleNamespace(),
    }
    result = smoke_module.execute_native_comrecgc_smoke(
        modules=modules,
        checkpoint_payloads={},
        source_rows=[object() for _ in range(64)],
        graph_schema=SimpleNamespace(feature_atomic_numbers=ATOMS),
        device="cpu",
    )
    assert result["random_walk_steps"] == 500
    assert result["checkpoint_reload"]["checkpoint_step"] == 250
    assert result["checkpoint_reload"]["next_step"] == 251
    assert result["checkpoint_reload"]["checkpoint_reloaded"] is True
    assert result["bridge"]["evaluated_strict_graph_count"] == 1
    assert len(module.traversed_hashes) == 500
    assert validate_native_comrecgc_smoke_result(result) == result


def test_strict_smoke_result_reopens_scientific_claims() -> None:
    result = _valid_result()
    assert validate_native_comrecgc_smoke_result(result) == result


def test_strict_smoke_rejects_duplicate_official_greedy_cluster() -> None:
    result = _valid_result()
    duplicate = deepcopy(result["common_recourse"]["selected_common_recourses"][0])
    duplicate["rank"] = 2
    result["common_recourse"]["selected_common_recourses"].append(duplicate)
    result["common_recourse"]["selected_common_recourse_count"] = 2
    result["common_recourse"]["dbscan_cluster_count"] = 2
    with pytest.raises(TasteComRecGCSmokeError, match="selected one cluster twice"):
        validate_native_comrecgc_smoke_result(result)


def test_terminal_documents_bind_t2_t3_t4_managed_and_aggregate_only(
    tmp_path: Path,
) -> None:
    authority = _terminal_input_authority(tmp_path)
    assert validate_terminal_input_authority(authority) == authority
    documents = build_terminal_documents(
        science=_valid_result(),
        input_authority=authority,
        task_id=TASK_ID,
        run_id="managed-t9-run",
        gpu_uuid="GPU-1234abcd",
    )
    assert set(documents) == {
        "input_hashes.json",
        "state.json",
        "manifest.json",
        "comrecgc_smoke.json",
        "gate.json",
    }
    serialized = b"".join(documents.values()).lower()
    assert b'"smiles"' not in serialized
    assert b'"molecule_id"' not in serialized
    assert b"/proc/self/fd/" not in serialized

    with pytest.raises(TasteComRecGCSmokeError, match="task_id changed"):
        build_terminal_documents(
            science=_valid_result(),
            input_authority=authority,
            task_id="wrong-task",
            run_id="managed-t9-run",
            gpu_uuid="GPU-1234abcd",
        )
    with pytest.raises(TasteComRecGCSmokeError, match="GPU UUID is malformed"):
        build_terminal_documents(
            science=_valid_result(),
            input_authority=authority,
            task_id=TASK_ID,
            run_id="managed-t9-run",
            gpu_uuid="GPU-",
        )


def test_terminal_input_rejects_t3_t4_or_physical_inventory_drift(
    tmp_path: Path,
) -> None:
    authority = _terminal_input_authority(tmp_path)
    mismatch = deepcopy(authority)
    mismatch["t4_stage_evidence"]["checkpoint_id"] = "8" * 64
    with pytest.raises(TasteComRecGCSmokeError, match="checkpoint_id changed"):
        validate_terminal_input_authority(mismatch)

    hostile_bool = deepcopy(authority)
    hostile_bool["t2_adoption_binding"]["formal_bundle_inventory"][0][
        "identity"
    ]["inode"] = True
    binding = hostile_bool["t2_adoption_binding"]
    binding["formal_bundle_inventory_sha256"] = hashlib.sha256(
        json.dumps(
            binding["formal_bundle_inventory"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    hostile_bool["t2_adoption_binding_sha256"] = hashlib.sha256(
        json.dumps(
            binding,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    for stage in ("t3_stage_evidence", "t4_stage_evidence"):
        hostile_bool[stage]["t2_adoption_binding_sha256"] = hostile_bool[
            "t2_adoption_binding_sha256"
        ]
    with pytest.raises(TasteComRecGCSmokeError, match="file identity"):
        validate_terminal_input_authority(hostile_bool)


def test_strict_terminal_consumer_reopens_hash_dag_and_retains_inodes(
    tmp_path: Path,
) -> None:
    documents = build_terminal_documents(
        science=_valid_result(),
        input_authority=_terminal_input_authority(tmp_path),
        task_id=TASK_ID,
        run_id="managed-t9-run",
        gpu_uuid="GPU-1234abcd",
    )
    root = tmp_path / "t9-output"
    _publish_terminal_fixture(root, documents)
    evidence = validate_tastemolnet_comrecgc_output(root)
    assert evidence["status"] == "PASS"
    assert evidence["stage"] == "T9_COMRECGC_SMOKE"
    assert evidence["task_id"] == TASK_ID
    assert evidence["run_id"] == "managed-t9-run"
    assert evidence["gpu_index"] == 2
    assert evidence["gpu_uuid"] == "GPU-1234abcd"
    assert evidence["managed_active_receipt_sha256"] == "b" * 64
    assert evidence["strict_counterfactual_count"] == 1
    assert evidence["destination_prediction_counts"] == {"0": 1, "2": 0}
    assert evidence["official_commit"] == OFFICIAL_COMRECGC_COMMIT
    assert evidence["train_csv_sha256"] == "e" * 64
    assert evidence["t3_gate_sha256"] == "c" * 64
    assert evidence["t4_gate_sha256"] == "d" * 64

    with hold_tastemolnet_comrecgc_output(root) as held:
        gate = root / "gate.json"
        replacement = root / "gate.replacement"
        replacement.write_bytes(gate.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, gate)
        with pytest.raises(Exception, match="single-link|changed|identity"):
            held.revalidate()


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("official_stateful_heads_preserved",), 1),
        (("parameters", "steps"), True),
        (("source_cohort", "source_label"), True),
        (("checkpoint_reload", "checkpoint_step"), 250.0),
        (("bridge", "destination_prediction_counts", "0"), 0),
        (
            (
                "common_recourse",
                "selected_common_recourses",
                0,
                "destination_label",
            ),
            1,
        ),
    ),
)
def test_strict_smoke_result_rejects_typed_or_semantic_drift(
    path: tuple[object, ...], replacement: object
) -> None:
    result = deepcopy(_valid_result())
    target: object = result
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(TasteComRecGCSmokeError):
        validate_native_comrecgc_smoke_result(result)
