from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

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
    cohort = b'{"parent_id":"x"}\n'
    (root / "cohort.jsonl").write_bytes(cohort)
    cohort_manifest = {
        "status": "PASS",
        "policy": "FULL_TRAIN_CORRECT_SOURCE",
        "cohort_jsonl_sha256": _sha(cohort),
    }
    (root / "cohort_manifest.json").write_text(json.dumps(cohort_manifest))
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
